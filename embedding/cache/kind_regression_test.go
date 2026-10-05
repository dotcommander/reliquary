package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

type kindSpyEmbedder struct {
	requests []embedding.Request
}

func (s *kindSpyEmbedder) Embed(_ context.Context, request embedding.Request) (embedding.Result, error) {
	s.requests = append(s.requests, embedding.Request{
		Model: request.Model, Kind: request.Kind, Inputs: append([]string(nil), request.Inputs...),
	})
	vectors := make([]embedding.Vector, len(request.Inputs))
	for i := range vectors {
		vectors[i] = embedding.Vector{float32(request.Kind + 1), 1}
	}
	return embedding.Result{Model: request.Model, Vectors: vectors}, nil
}

func TestCachePartitionsKindsAndForwardsMisses(t *testing.T) {
	t.Parallel()
	spy := &kindSpyEmbedder{}
	store := newMemoryStore()
	cached := mustNew(t, spy, store)
	model := embedding.ModelRef{Name: "task-model", Dim: 2}
	for _, kind := range []embedding.Kind{embedding.KindDocument, embedding.KindQuery, embedding.KindDocument, embedding.KindQuery} {
		got := mustEmbed(t, cached, embedding.Request{Model: model, Kind: kind, Inputs: []string{"same", "same"}})
		want := []embedding.Vector{{float32(kind + 1), 1}, {float32(kind + 1), 1}}
		if !reflect.DeepEqual(got.Vectors, want) {
			t.Fatalf("kind %v vectors = %v, want %v", kind, got.Vectors, want)
		}
	}
	wantRequests := []embedding.Request{
		{Model: model, Kind: embedding.KindDocument, Inputs: []string{"same"}},
		{Model: model, Kind: embedding.KindQuery, Inputs: []string{"same"}},
	}
	if !reflect.DeepEqual(spy.requests, wantRequests) {
		t.Fatalf("miss requests = %#v, want %#v", spy.requests, wantRequests)
	}
	if len(store.entries) != 2 || store.setCount() != 2 {
		t.Fatalf("cache entries = %d, writes = %d; want two independent entries", len(store.entries), store.setCount())
	}
	// A partial hit must preserve the query task and send only uncached inputs.
	mustEmbed(t, cached, embedding.Request{Model: model, Kind: embedding.KindQuery, Inputs: []string{"same", "new", "new"}})
	last := spy.requests[len(spy.requests)-1]
	if last.Kind != embedding.KindQuery || last.Model != model || !reflect.DeepEqual(last.Inputs, []string{"new"}) {
		t.Fatalf("partial-hit request = %#v", last)
	}
}

func TestCacheDoesNotReuseOrDeleteLegacyAmbiguousEntry(t *testing.T) {
	t.Parallel()
	spy := &kindSpyEmbedder{}
	store := newMemoryStore()
	model := embedding.ModelRef{Name: "task-model", Dim: 2}
	legacyPreimage := "reliquary:embedding-cache:v1:" + frame("test-cache") + frame(embedding.CacheKey(model, "same"))
	sum := sha256.Sum256([]byte(legacyPreimage))
	legacyKey := hex.EncodeToString(sum[:])
	legacy := Entry{Model: model, Vector: embedding.Vector{99, 99}}
	store.entries[legacyKey] = legacy
	cached := mustNew(t, spy, store)
	for _, kind := range []embedding.Kind{embedding.KindDocument, embedding.KindQuery} {
		got := mustEmbed(t, cached, embedding.Request{Model: model, Kind: kind, Inputs: []string{"same"}})
		if reflect.DeepEqual(got.Vectors[0], legacy.Vector) {
			t.Fatal("reused legacy ambiguous embedding")
		}
	}
	if len(spy.requests) != 2 || len(store.entries) != 3 || !reflect.DeepEqual(store.entries[legacyKey], legacy) {
		t.Fatalf("legacy entry changed or new kinds not independently generated: requests=%d entries=%d", len(spy.requests), len(store.entries))
	}
}
