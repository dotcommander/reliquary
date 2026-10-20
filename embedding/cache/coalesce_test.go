package cache

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"testing/synctest"

	"github.com/dotcommander/reliquary/embedding"
)

func coalescingCache(t *testing.T, base embedding.Embedder) *Embedder {
	t.Helper()
	e, err := New(base, newMemoryStore(), Config{Identity: "flights", CoalesceRequests: true})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestCoalescedVectorsAndFollowerCancellation(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		base := &fakeEmbedder{hook: func(context.Context, embedding.Request) { <-release }}
		e := coalescingCache(t, base)
		req := embedding.Request{Inputs: []string{"one", "two"}}
		leader := make(chan embedding.Result, 1)
		go func() { leader <- mustEmbed(t, e, req) }()
		synctest.Wait()
		canceled, cancel := context.WithCancel(context.Background())
		followerErr := make(chan error, 1)
		go func() { _, err := e.Embed(canceled, req); followerErr <- err }()
		followers := make(chan embedding.Result, 2)
		for range 2 {
			go func() { followers <- mustEmbed(t, e, req) }()
		}
		synctest.Wait()
		cancel()
		if err := <-followerErr; !errors.Is(err, context.Canceled) {
			t.Fatalf("follower: %v", err)
		}
		if base.callCount() != 1 {
			t.Fatalf("calls = %d", base.callCount())
		}
		close(release)
		a, b, c := <-leader, <-followers, <-followers
		a.Vectors[0][0] = -1
		b.Vectors[0][0] = -2
		if c.Vectors[0][0] != 3 {
			t.Fatal("shared follower vectors")
		}
		if got := mustEmbed(t, e, req); got.Vectors[0][0] != 3 {
			t.Fatal("shared cache vector")
		}
	})
}

func TestCoalescedFailureAndAbnormalExit(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"error", "cancel", "panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			synctest.Test(t, func(t *testing.T) {
				release := make(chan struct{})
				want := errors.New("provider failed")
				base := &fakeEmbedder{hook: func(ctx context.Context, _ embedding.Request) {
					<-release
					switch mode {
					case "panic":
						panic("leader panic")
					case "goexit":
						runtime.Goexit()
					case "cancel":
						<-ctx.Done()
					}
				}}
				if mode == "error" {
					base.err = want
				}
				e := coalescingCache(t, base)
				req := embedding.Request{Inputs: []string{"one"}}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				leaderDone := make(chan struct{})
				leaderErr := make(chan error, 1)
				var recovered any
				go func() {
					defer close(leaderDone)
					defer func() { recovered = recover() }()
					_, err := e.Embed(ctx, req)
					leaderErr <- err
				}()
				synctest.Wait()
				follower := make(chan error, 1)
				go func() { _, err := e.Embed(context.Background(), req); follower <- err }()
				synctest.Wait()
				if mode == "cancel" {
					cancel()
				}
				close(release)
				err := <-follower
				<-leaderDone
				switch mode {
				case "error":
					if !errors.Is(err, want) || !errors.Is(<-leaderErr, want) {
						t.Fatalf("shared failure: %v", err)
					}
				case "cancel":
					if !errors.Is(err, context.Canceled) || !errors.Is(<-leaderErr, context.Canceled) {
						t.Fatalf("leader cancellation: %v", err)
					}
				default:
					if !errors.Is(err, ErrCoalescedCallAborted) {
						t.Fatalf("aborted follower: %v", err)
					}
					if mode == "panic" && recovered != "leader panic" {
						t.Fatalf("panic = %v", recovered)
					}
				}
				base.mu.Lock()
				base.hook = nil
				base.err = nil
				base.mu.Unlock()
				mustEmbed(t, e, req)
				if base.callCount() != 2 {
					t.Fatalf("retry calls = %d", base.callCount())
				}
			})
		})
	}
}

func TestCoalescedDistinctRequestsRemainIndependent(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		release := make(chan struct{})
		base := &fakeEmbedder{hook: func(context.Context, embedding.Request) { <-release }}
		e := coalescingCache(t, base)
		requests := []embedding.Request{
			{Inputs: []string{"a", "b"}},
			{Inputs: []string{"b", "a"}},
			{Inputs: []string{"a", "b"}, Kind: embedding.KindQuery},
			{Inputs: []string{"a", "b"}, Model: embedding.ModelRef{Name: "other"}},
			{Inputs: []string{"a", "c"}},
			{Inputs: []string{"a"}},
			{Inputs: []string{"a", "b", ""}},
			{}, {},
		}
		done := make(chan struct{}, len(requests))
		for _, req := range requests {
			go func() { mustEmbed(t, e, req); done <- struct{}{} }()
		}
		synctest.Wait()
		if base.callCount() != len(requests) {
			t.Fatalf("independent calls = %d", base.callCount())
		}
		close(release)
		for range requests {
			<-done
		}
		if requestKey("ab", embedding.Request{Inputs: []string{"c"}}) == requestKey("a", embedding.Request{Inputs: []string{"bc"}}) {
			t.Fatal("unframed identity")
		}
	})
}
