// Command provider-ollama-wiring demonstrates configuring and wiring the Ollama
// embedding adapter into Reliquary.
//
// It shows:
// 1. Initializing caller-owned HTTP client and Ollama configuration.
// 2. Constructing the adapter/ollama Embedder.
// 3. Testing live endpoint connectivity or falling back gracefully to mock embedding.
// 4. Wiring the Ollama embedder into a Reliquary App pipeline.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/provider-ollama-wiring
package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"time"

	"github.com/dotcommander/reliquary"
	ollamaadapter "github.com/dotcommander/reliquary/adapter/ollama"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	modelName := os.Getenv("OLLAMA_MODEL")
	if modelName == "" {
		modelName = "nomic-embed-text"
	}
	const dim = 768

	httpClient := &http.Client{Timeout: 30 * time.Second}

	var embedder *ollamaadapter.Embedder
	var err error

	if baseURL != "" {
		embedder, err = ollamaadapter.New(httpClient, ollamaadapter.Config{
			BaseURL:    baseURL,
			Model:      modelName,
			Dimensions: dim,
		})
		if err != nil {
			return fmt.Errorf("new ollama adapter: %w", err)
		}
		fmt.Printf("Configured live Ollama embedder at %s (model: %s, dims: %d)\n", baseURL, modelName, dim)
	} else {
		// Start a local mock Ollama /api/embed server to demonstrate wire compatibility.
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/embed" || r.Method != http.MethodPost {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			vec := make([]float64, dim)
			vec[0] = 0.5
			vec[1] = 0.5
			var vecJSON []string
			for _, v := range vec {
				vecJSON = append(vecJSON, fmt.Sprintf("%.2f", v))
			}
			resp := fmt.Sprintf(`{"model":"%s","embeddings":[[%s]]}`, modelName, strings.Join(vecJSON, ","))
			_, _ = w.Write([]byte(resp))
		}))
		defer server.Close()

		embedder, err = ollamaadapter.New(server.Client(), ollamaadapter.Config{
			BaseURL:    server.URL,
			Model:      modelName,
			Dimensions: dim,
		})
		if err != nil {
			return fmt.Errorf("new mock ollama adapter: %w", err)
		}
		fmt.Printf("Configured demo Ollama embedder against mock server (model: %s, dims: %d)\n", modelName, dim)
	}

	// Verify embedding request
	res, err := embedder.Embed(ctx, embedding.Request{
		Model:  embedding.ModelRef{Name: modelName, Dim: dim},
		Inputs: []string{"Ollama runs open-source LLMs and embedding models locally with GPU acceleration."},
	})
	if err != nil {
		return fmt.Errorf("embed call: %w", err)
	}
	fmt.Printf("Generated %d embedding vectors (dim=%d) via Ollama contract\n", len(res.Vectors), len(res.Vectors[0]))

	// Wire into Reliquary App
	app, err := reliquary.New(
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndexIdentity(fmt.Sprintf("ollama-%s-%d", modelName, dim)),
	)
	if err != nil {
		return fmt.Errorf("new reliquary app with ollama embedder: %w", err)
	}

	docs := []document.Document{
		{
			ID:     "doc-ollama",
			Title:  "ollama-overview.md",
			Format: document.FormatMarkdown,
			Text:   "Ollama simplifies local AI operations by wrapping model weights into standard Docker-style manifests.",
		},
	}
	chunks, err := app.Ingest(ctx, docs...)
	if err != nil {
		return fmt.Errorf("ingest with ollama: %w", err)
	}
	fmt.Printf("Ingested %d document (%d chunks) into Reliquary pipeline using Ollama embedder.\n", len(docs), chunks)
	return nil
}
