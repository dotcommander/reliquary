package inmem_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/cache"
	"github.com/dotcommander/reliquary/embedding/cache/inmem"
)

func ExampleNew() {
	store := inmem.New()
	entry := cache.Entry{
		Model:  embedding.ModelRef{Name: "example", Dim: 2},
		Vector: embedding.Vector{1, 2},
	}
	if err := store.Set(context.Background(), "key", entry); err != nil {
		panic(err)
	}

	got, ok, err := store.Get(context.Background(), "key")
	fmt.Println(ok, err == nil, got.Model.Name, got.Vector)
	// Output: true true example [1 2]
}
