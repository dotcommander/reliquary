package openai_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/adapter/openai"
	openaigo "github.com/openai/openai-go/v3"
)

func ExampleNew() {
	config := openai.Config{
		Model:      "text-embedding-3-small",
		Dimensions: 1536,
	}
	embedder, err := openai.New(openaigo.NewClient(), config)
	fmt.Println(embedder != nil, err == nil)
	// Output: true true
}
