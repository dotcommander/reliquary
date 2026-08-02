package cache_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/cache"
)

func ExampleNew() {
	base := exampleEmbedder{}
	store := &exampleStore{entries: make(map[string]cache.Entry)}
	cached, err := cache.New(base, store, cache.Config{Identity: "example:v1"})
	if err != nil {
		panic(err)
	}

	result, err := cached.Embed(context.Background(), embedding.Request{Inputs: []string{"hello"}})
	fmt.Println(err == nil, result.Model.Name, len(result.Vectors))
	// Output: true example 1
}

type exampleEmbedder struct{}

func (exampleEmbedder) Embed(_ context.Context, request embedding.Request) (embedding.Result, error) {
	vectors := make([]embedding.Vector, len(request.Inputs))
	for index, input := range request.Inputs {
		vectors[index] = embedding.Vector{float32(len(input))}
	}
	return embedding.Result{Model: embedding.ModelRef{Name: "example", Dim: 1}, Vectors: vectors}, nil
}

type exampleStore struct {
	entries map[string]cache.Entry
}

func (s *exampleStore) Get(_ context.Context, key string) (cache.Entry, bool, error) {
	entry, ok := s.entries[key]
	return entry, ok, nil
}

func (s *exampleStore) Set(_ context.Context, key string, entry cache.Entry) error {
	s.entries[key] = entry
	return nil
}
