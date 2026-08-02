package cache

import (
	"context"
	"github.com/dotcommander/reliquary/embedding"
	"sync"
	"testing"
)

func mustNew(t *testing.T, base embedding.Embedder, store Store) *Embedder {
	t.Helper()
	cached, err := New(base, store, Config{Identity: "test-cache"})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return cached
}

func mustEmbed(t *testing.T, embedder embedding.Embedder, request embedding.Request) embedding.Result {
	t.Helper()
	result, err := embedder.Embed(context.Background(), request)
	if err != nil {
		t.Fatalf("Embed(): %v", err)
	}
	return result
}

func isZeroResult(result embedding.Result) bool {
	return result.Model == (embedding.ModelRef{}) && result.Vectors == nil
}

func vectorFor(input string) embedding.Vector {
	first := float32(0)
	if len(input) > 0 {
		first = float32(input[0])
	}
	return embedding.Vector{float32(len(input)), first}
}

type fakeEmbedder struct {
	mu       sync.Mutex
	model    embedding.ModelRef
	err      error
	response func(embedding.Request) (embedding.Result, error)
	hook     func(context.Context, embedding.Request)
	calls    []embedding.Request
}

func (e *fakeEmbedder) Embed(ctx context.Context, request embedding.Request) (embedding.Result, error) {
	e.mu.Lock()
	e.calls = append(e.calls, embedding.Request{Model: request.Model, Inputs: append([]string(nil), request.Inputs...)})
	response, err, hook, model := e.response, e.err, e.hook, e.model
	e.mu.Unlock()
	if hook != nil {
		hook(ctx, request)
	}
	if err != nil {
		return embedding.Result{}, err
	}
	if response != nil {
		return response(request)
	}
	if model == (embedding.ModelRef{}) {
		model = testModel
	}
	vectors := make([]embedding.Vector, len(request.Inputs))
	for index, input := range request.Inputs {
		vectors[index] = vectorFor(input)
	}
	return embedding.Result{Model: model, Vectors: vectors}, nil
}

func (e *fakeEmbedder) callCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.calls)
}

func (e *fakeEmbedder) requests() [][]string {
	e.mu.Lock()
	defer e.mu.Unlock()
	requests := make([][]string, len(e.calls))
	for index, request := range e.calls {
		requests[index] = append([]string(nil), request.Inputs...)
	}
	return requests
}

type memoryStore struct {
	mu      sync.Mutex
	entries map[string]Entry
	gets    []string
	sets    []string
	getErr  error
	setErr  error
	getHook func(context.Context, string)
	setHook func(context.Context, string, Entry)
}

func newMemoryStore() *memoryStore {
	return &memoryStore{entries: make(map[string]Entry)}
}

func (s *memoryStore) Get(ctx context.Context, key string) (Entry, bool, error) {
	s.mu.Lock()
	entry, ok, err, hook := s.entries[key], false, s.getErr, s.getHook
	if _, exists := s.entries[key]; exists {
		ok = true
	}
	s.gets = append(s.gets, key)
	s.mu.Unlock()
	if hook != nil {
		hook(ctx, key)
	}
	return entry, ok, err
}

func (s *memoryStore) Set(ctx context.Context, key string, entry Entry) error {
	s.mu.Lock()
	err, hook := s.setErr, s.setHook
	if err == nil {
		s.entries[key] = entry
	}
	s.sets = append(s.sets, key)
	s.mu.Unlock()
	if hook != nil {
		hook(ctx, key, entry)
	}
	return err
}

func (s *memoryStore) getCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.gets)
}

func (s *memoryStore) setCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sets)
}
