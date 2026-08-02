package cache

import (
	"context"
	"errors"
	"github.com/dotcommander/reliquary/embedding"
	"testing"
)

func TestEmbedCancellation(t *testing.T) {
	t.Parallel()
	t.Run("entry", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		base, store := &fakeEmbedder{}, newMemoryStore()
		result, err := mustNew(t, base, store).Embed(ctx, embedding.Request{Inputs: []string{"input"}})
		if !errors.Is(err, context.Canceled) || !isZeroResult(result) || base.callCount() != 0 || store.getCount() != 0 {
			t.Fatalf("entry cancellation: result=%#v error=%v base=%d gets=%d", result, err, base.callCount(), store.getCount())
		}
	})

	t.Run("after lookup", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		store := newMemoryStore()
		store.getHook = func(context.Context, string) { cancel() }
		base := &fakeEmbedder{}
		result, err := mustNew(t, base, store).Embed(ctx, embedding.Request{Inputs: []string{"input"}})
		if !errors.Is(err, context.Canceled) || !isZeroResult(result) || base.callCount() != 0 || store.setCount() != 0 {
			t.Fatalf("lookup cancellation: result=%#v error=%v base=%d sets=%d", result, err, base.callCount(), store.setCount())
		}
	})

	t.Run("before later lookup", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		store := newMemoryStore()
		store.getHook = func(context.Context, string) { cancel() }
		base := &fakeEmbedder{}
		result, err := mustNew(t, base, store).Embed(ctx, embedding.Request{Inputs: []string{"first", "second"}})
		if !errors.Is(err, context.Canceled) || !isZeroResult(result) {
			t.Fatalf("later lookup cancellation: result=%#v error=%v", result, err)
		}
		if base.callCount() != 0 || store.getCount() != 1 || store.setCount() != 0 {
			t.Fatalf("later lookup cancellation called base=%d gets=%d sets=%d, want 0/1/0", base.callCount(), store.getCount(), store.setCount())
		}
	})

	t.Run("after base", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		base := &fakeEmbedder{hook: func(context.Context, embedding.Request) { cancel() }}
		store := newMemoryStore()
		result, err := mustNew(t, base, store).Embed(ctx, embedding.Request{Inputs: []string{"input"}})
		if !errors.Is(err, context.Canceled) || !isZeroResult(result) || store.setCount() != 0 {
			t.Fatalf("base cancellation: result=%#v error=%v sets=%d", result, err, store.setCount())
		}
	})

	t.Run("after write", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		store := newMemoryStore()
		store.setHook = func(context.Context, string, Entry) { cancel() }
		result, err := mustNew(t, &fakeEmbedder{}, store).Embed(ctx, embedding.Request{Inputs: []string{"first", "second"}})
		if !errors.Is(err, context.Canceled) || !isZeroResult(result) {
			t.Fatalf("write cancellation: result=%#v error=%v", result, err)
		}
		if got := store.setCount(); got != 1 {
			t.Fatalf("completed writes = %d, want 1", got)
		}
	})

	t.Run("after final write", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(context.Background())
		store := newMemoryStore()
		sets := 0
		store.setHook = func(context.Context, string, Entry) {
			sets++
			if sets == 2 {
				cancel()
			}
		}
		result, err := mustNew(t, &fakeEmbedder{}, store).Embed(ctx, embedding.Request{Inputs: []string{"first", "second"}})
		if !errors.Is(err, context.Canceled) || !isZeroResult(result) {
			t.Fatalf("final write cancellation: result=%#v error=%v", result, err)
		}
		if got := store.setCount(); got != 2 {
			t.Fatalf("completed writes = %d, want 2", got)
		}
	})
}
