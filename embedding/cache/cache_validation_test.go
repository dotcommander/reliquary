package cache

import (
	"context"
	"errors"
	"github.com/dotcommander/reliquary/embedding"
	"testing"
)

func TestEmbedModelMismatchPreventsWrites(t *testing.T) {
	t.Parallel()
	store := newMemoryStore()
	request := embedding.Request{Inputs: []string{"hit", "miss"}}
	store.entries[cacheKey("test-cache", request.Model, "hit")] = Entry{Model: testModel, Vector: vectorFor("hit")}
	base := &fakeEmbedder{model: embedding.ModelRef{Provider: "other", Name: "vectors", Dim: 2}}
	cached := mustNew(t, base, store)

	result, err := cached.Embed(context.Background(), request)
	if !errors.Is(err, ErrModelMismatch) {
		t.Fatalf("Embed() error = %v, want model mismatch", err)
	}
	if !isZeroResult(result) {
		t.Fatalf("Embed() result = %#v, want zero result", result)
	}
	if got := store.setCount(); got != 0 {
		t.Fatalf("Set calls = %d, want 0", got)
	}
}

func TestEmbedCachedModelMismatchPreventsWrites(t *testing.T) {
	t.Parallel()
	store := newMemoryStore()
	request := embedding.Request{Inputs: []string{"first", "second"}}
	store.entries[cacheKey("test-cache", request.Model, "first")] = Entry{Model: testModel, Vector: vectorFor("first")}
	store.entries[cacheKey("test-cache", request.Model, "second")] = Entry{
		Model:  embedding.ModelRef{Provider: "other", Name: "vectors", Version: "v1", Revision: "r1", Dim: 2},
		Vector: vectorFor("second"),
	}
	base := &fakeEmbedder{}
	result, err := mustNew(t, base, store).Embed(context.Background(), request)
	if !errors.Is(err, ErrModelMismatch) || !isZeroResult(result) {
		t.Fatalf("Embed() result=%#v error=%v, want zero result and model mismatch", result, err)
	}
	if base.callCount() != 0 || store.setCount() != 0 {
		t.Fatalf("mismatched hits called base=%d sets=%d, want 0/0", base.callCount(), store.setCount())
	}
}

func TestEmbedAssembledResultValidationPreventsWrites(t *testing.T) {
	t.Parallel()
	store := newMemoryStore()
	request := embedding.Request{Inputs: []string{"one", "two"}}
	store.entries[cacheKey("test-cache", request.Model, "one")] = Entry{Vector: embedding.Vector{1}}
	store.entries[cacheKey("test-cache", request.Model, "two")] = Entry{Vector: embedding.Vector{1, 2}}
	base := &fakeEmbedder{}
	result, err := mustNew(t, base, store).Embed(context.Background(), request)
	if !errors.Is(err, embedding.ErrInvalidResult) || !isZeroResult(result) {
		t.Fatalf("Embed() result=%#v error=%v, want zero result and invalid result", result, err)
	}
	if base.callCount() != 0 || store.setCount() != 0 {
		t.Fatalf("inconsistent hits called base=%d sets=%d, want 0/0", base.callCount(), store.setCount())
	}
}

func TestEmbedErrorsRemainClassifiable(t *testing.T) {
	t.Parallel()
	lookupErr := errors.New("lookup failed")
	baseErr := errors.New("base failed")
	writeErr := errors.New("write failed")
	tests := []struct {
		name  string
		setup func(*fakeEmbedder, *memoryStore)
		want  error
	}{
		{
			name:  "lookup",
			setup: func(_ *fakeEmbedder, store *memoryStore) { store.getErr = lookupErr },
			want:  lookupErr,
		},
		{
			name:  "base",
			setup: func(base *fakeEmbedder, _ *memoryStore) { base.err = baseErr },
			want:  baseErr,
		},
		{
			name:  "write",
			setup: func(_ *fakeEmbedder, store *memoryStore) { store.setErr = writeErr },
			want:  writeErr,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			base := &fakeEmbedder{}
			store := newMemoryStore()
			test.setup(base, store)
			result, err := mustNew(t, base, store).Embed(context.Background(), embedding.Request{Inputs: []string{"input"}})
			if !errors.Is(err, test.want) {
				t.Fatalf("Embed() error = %v, want errors.Is(_, %v)", err, test.want)
			}
			if !isZeroResult(result) {
				t.Fatalf("Embed() result = %#v, want zero result", result)
			}
		})
	}
}

func TestEmbedMalformedResultsFailClosed(t *testing.T) {
	t.Parallel()
	t.Run("cached entry", func(t *testing.T) {
		t.Parallel()
		store := newMemoryStore()
		request := embedding.Request{Inputs: []string{"bad"}}
		store.entries[cacheKey("test-cache", request.Model, "bad")] = Entry{Model: testModel, Vector: embedding.Vector{1}}
		base := &fakeEmbedder{}
		result, err := mustNew(t, base, store).Embed(context.Background(), request)
		if !errors.Is(err, embedding.ErrInvalidResult) {
			t.Fatalf("Embed() error = %v, want invalid result", err)
		}
		if !isZeroResult(result) || base.callCount() != 0 || store.setCount() != 0 {
			t.Fatalf("malformed hit did not fail closed: result=%#v base=%d sets=%d", result, base.callCount(), store.setCount())
		}
	})

	t.Run("generated result", func(t *testing.T) {
		t.Parallel()
		base := &fakeEmbedder{response: func(embedding.Request) (embedding.Result, error) {
			return embedding.Result{Model: testModel, Vectors: []embedding.Vector{{1}}}, nil
		}}
		store := newMemoryStore()
		result, err := mustNew(t, base, store).Embed(context.Background(), embedding.Request{Inputs: []string{"bad"}})
		if !errors.Is(err, embedding.ErrInvalidResult) {
			t.Fatalf("Embed() error = %v, want invalid result", err)
		}
		if !isZeroResult(result) || store.setCount() != 0 {
			t.Fatalf("malformed generated result did not fail closed: result=%#v sets=%d", result, store.setCount())
		}
	})
}
