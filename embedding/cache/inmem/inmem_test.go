package inmem

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/cache"
)

func TestNewAndZeroValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		store *Store
	}{
		{name: "constructor", store: New()},
		{name: "zero value", store: &Store{}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			if entry, ok, err := test.store.Get(ctx, "missing"); err != nil || ok || !isZeroEntry(entry) {
				t.Fatalf("Get(missing) = (%#v, %t, %v), want zero, false, nil", entry, ok, err)
			}

			want := cache.Entry{
				Model:  embedding.ModelRef{Name: "model", Dim: 2},
				Vector: embedding.Vector{1, 2},
			}
			if err := test.store.Set(ctx, "", want); err != nil {
				t.Fatalf("Set(empty key): %v", err)
			}
			got, ok, err := test.store.Get(ctx, "")
			if err != nil || !ok || got.Model != want.Model || !equalVector(got.Vector, want.Vector) {
				t.Fatalf("Get(empty key) = (%#v, %t, %v), want %#v, true, nil", got, ok, err, want)
			}
		})
	}
}

func TestOverwriteAndCloneOwnership(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := New()
	source := embedding.Vector{1, 2}
	first := cache.Entry{Model: embedding.ModelRef{Name: "first", Dim: 2}, Vector: source}
	if err := store.Set(ctx, "key", first); err != nil {
		t.Fatalf("Set(first): %v", err)
	}

	source[0] = 9
	got, ok, err := store.Get(ctx, "key")
	if err != nil || !ok {
		t.Fatalf("Get(first) = (%#v, %t, %v), want hit", got, ok, err)
	}
	if got.Vector[0] != 1 {
		t.Fatalf("stored vector[0] = %v, want 1 after caller mutation", got.Vector[0])
	}

	got.Vector[0] = 8
	again, ok, err := store.Get(ctx, "key")
	if err != nil || !ok {
		t.Fatalf("Get(first again) = (%#v, %t, %v), want hit", again, ok, err)
	}
	if again.Vector[0] != 1 {
		t.Fatalf("stored vector[0] = %v, want 1 after result mutation", again.Vector[0])
	}

	second := cache.Entry{Model: embedding.ModelRef{Name: "second", Dim: 1}, Vector: embedding.Vector{3}}
	if err := store.Set(ctx, "key", second); err != nil {
		t.Fatalf("Set(second): %v", err)
	}
	overwritten, ok, err := store.Get(ctx, "key")
	if err != nil || !ok || overwritten.Model != second.Model || !equalVector(overwritten.Vector, second.Vector) {
		t.Fatalf("Get(overwritten) = (%#v, %t, %v), want %#v, true, nil", overwritten, ok, err, second)
	}
}

func TestCancellationDoesNotMutate(t *testing.T) {
	t.Parallel()

	store := New()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if entry, ok, err := store.Get(ctx, "key"); !errors.Is(err, context.Canceled) || ok || !isZeroEntry(entry) {
		t.Fatalf("Get(canceled) = (%#v, %t, %v), want zero, false, context.Canceled", entry, ok, err)
	}
	if err := store.Set(ctx, "key", cache.Entry{Vector: embedding.Vector{1}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Set(canceled) error = %v, want context.Canceled", err)
	}
	if entry, ok, err := store.Get(context.Background(), "key"); err != nil || ok || !isZeroEntry(entry) {
		t.Fatalf("Get after canceled Set = (%#v, %t, %v), want zero, false, nil", entry, ok, err)
	}
}

func TestCancellationWhileWaitingForLockDoesNotMutate(t *testing.T) {
	t.Parallel()

	store := New()
	store.mu.Lock()

	ctx, cancel := context.WithCancel(context.Background())
	observed := &observedContext{Context: ctx, checked: make(chan struct{})}
	done := make(chan error, 1)
	go func() {
		done <- store.Set(observed, "blocked", cache.Entry{Vector: embedding.Vector{1}})
	}()

	<-observed.checked
	cancel()
	store.mu.Unlock()

	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Set(canceled while waiting) error = %v, want context.Canceled", err)
	}
	if entry, ok, err := store.Get(context.Background(), "blocked"); err != nil || ok || !isZeroEntry(entry) {
		t.Fatalf("Get after blocked canceled Set = (%#v, %t, %v), want zero, false, nil", entry, ok, err)
	}
}

func TestConcurrentGetSet(t *testing.T) {
	t.Parallel()

	store := New()
	ctx := context.Background()
	const workers = 64

	var group sync.WaitGroup
	group.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			defer group.Done()
			key := fmt.Sprintf("key-%d", worker)
			entry := cache.Entry{
				Model:  embedding.ModelRef{Name: "concurrent", Dim: 1},
				Vector: embedding.Vector{float32(worker)},
			}
			if err := store.Set(ctx, key, entry); err != nil {
				t.Errorf("Set(%s): %v", key, err)
				return
			}
			got, ok, err := store.Get(ctx, key)
			if err != nil || !ok || !equalVector(got.Vector, entry.Vector) {
				t.Errorf("Get(%s) = (%#v, %t, %v), want %#v, true, nil", key, got, ok, err, entry)
			}
		}()
	}
	group.Wait()

	for worker := 0; worker < workers; worker++ {
		key := fmt.Sprintf("key-%d", worker)
		entry, ok, err := store.Get(ctx, key)
		if err != nil || !ok || entry.Vector[0] != float32(worker) {
			t.Fatalf("Get(%s) = (%#v, %t, %v), want worker value", key, entry, ok, err)
		}
	}
}

func equalVector(left, right embedding.Vector) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func isZeroEntry(entry cache.Entry) bool {
	return entry.Model == (embedding.ModelRef{}) && entry.Vector == nil
}

type observedContext struct {
	context.Context
	checked chan struct{}
	once    sync.Once
}

func (c *observedContext) Err() error {
	err := c.Context.Err()
	c.once.Do(func() { close(c.checked) })
	return err
}
