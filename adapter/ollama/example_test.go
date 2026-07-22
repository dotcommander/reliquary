package ollama_test

import (
	"fmt"
	"net/http"
	"time"

	"github.com/dotcommander/reliquary/adapter/ollama"
)

func ExampleNew() {
	client := &http.Client{Timeout: 30 * time.Second}
	embedder, err := ollama.New(client, ollama.Config{
		BaseURL: "http://localhost:11434",
		Model:   "nomic-embed-text",
	})
	fmt.Println(embedder != nil, err)
	// Output: true <nil>
}
