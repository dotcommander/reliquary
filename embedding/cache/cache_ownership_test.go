package cache

import (
	"context"
	"github.com/dotcommander/reliquary/embedding"
	"sync"
	"testing"
)

func TestEmbedClonesOwnershipBoundaries(t *testing.T) {
	t.Parallel()
	t.Run("store output and duplicate results", func(t *testing.T) {
		t.Parallel()
		store := newMemoryStore()
		request := embedding.Request{Inputs: []string{"cached", "cached"}}
		source := embedding.Vector{1, 2}
		key := cacheKey("test-cache", request.Model, "cached")
		store.entries[key] = Entry{Model: testModel, Vector: source}
		result := mustEmbed(t, mustNew(t, &fakeEmbedder{}, store), request)

		source[0] = 7
		if result.Vectors[0][0] == 7 {
			t.Fatal("returned vector aliases Store output")
		}
		result.Vectors[0][0] = 8
		if result.Vectors[1][0] == 8 {
			t.Fatal("duplicate returned vectors alias")
		}
	})

	t.Run("base output store input and result", func(t *testing.T) {
		t.Parallel()
		source := embedding.Vector{1, 2}
		base := &fakeEmbedder{response: func(request embedding.Request) (embedding.Result, error) {
			return embedding.Result{Model: testModel, Vectors: []embedding.Vector{source}}, nil
		}}
		store := newMemoryStore()
		cached := mustNew(t, base, store)
		result := mustEmbed(t, cached, embedding.Request{Inputs: []string{"new"}})
		key := cacheKey("test-cache", embedding.ModelRef{}, "new")

		source[0] = 7
		if result.Vectors[0][0] == 7 || store.entries[key].Vector[0] == 7 {
			t.Fatal("base output aliases a retained vector")
		}
		result.Vectors[0][0] = 8
		if store.entries[key].Vector[0] == 8 {
			t.Fatal("Store input aliases returned vector")
		}
		store.entries[key].Vector[0] = 9
		if result.Vectors[0][0] == 9 {
			t.Fatal("returned vector aliases Store input")
		}
	})

	t.Run("unique writes receive independent vectors", func(t *testing.T) {
		t.Parallel()
		store := newMemoryStore()
		mustEmbed(t, mustNew(t, &fakeEmbedder{}, store), embedding.Request{Inputs: []string{"first", "second"}})
		first := store.entries[cacheKey("test-cache", embedding.ModelRef{}, "first")].Vector
		second := store.entries[cacheKey("test-cache", embedding.ModelRef{}, "second")].Vector
		if &first[0] == &second[0] {
			t.Fatal("two Store.Set calls received aliasing vectors")
		}
		first[0] = 99
		if second[0] == 99 {
			t.Fatal("mutating one Store.Set vector changed another")
		}
	})
}

func TestEmbedConcurrent(t *testing.T) {
	t.Parallel()
	base := &fakeEmbedder{}
	store := newMemoryStore()
	cached := mustNew(t, base, store)
	const calls = 32
	var group sync.WaitGroup
	group.Add(calls)
	for index := 0; index < calls; index++ {
		go func() {
			defer group.Done()
			result, err := cached.Embed(context.Background(), embedding.Request{Inputs: []string{"shared", "shared"}})
			if err != nil || len(result.Vectors) != 2 {
				t.Errorf("Embed() result=%#v error=%v", result, err)
			}
		}()
	}
	group.Wait()
}
