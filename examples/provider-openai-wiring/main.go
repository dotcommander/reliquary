package main

import (
	"fmt"

	openaiadapter "github.com/dotcommander/reliquary/adapter/openai"
	"github.com/openai/openai-go/v3"
)

func main() {
	config := openaiadapter.Config{
		Model:      "text-embedding-3-small",
		Dimensions: 1536,
	}
	// NewClient reads OPENAI_API_KEY from the environment by default.
	embedder, err := openaiadapter.New(openai.NewClient(), config)
	if err != nil {
		panic(err)
	}
	fmt.Printf("configured %T with %d dimensions\n", embedder, config.Dimensions)
}
