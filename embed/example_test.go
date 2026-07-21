package embed_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/embed"
	"github.com/dotcommander/reliquary/embedding"
)

func ExampleHashing_Embed() {
	embedder := embed.NewHashing(8)
	result, err := embedder.Embed(context.Background(), embedding.Request{Inputs: []string{"hello world"}})
	fmt.Println(len(result.Vectors), len(result.Vectors[0]), err == nil)
	// Output: 1 8 true
}
