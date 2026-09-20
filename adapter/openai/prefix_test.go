package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// (Prefix coverage lives here; contract behavior is covered above.)

// TestTaskPrefixes verifies that configured document and query task
// prefixes reach the provider inputs for the matching request kinds.
// Nomic-family embedding models require the `search_document: ` and
// `search_query: ` prefixes; vanilla OpenAI models run with both unset.
func TestTaskPrefixes(t *testing.T) {
	t.Parallel()

	var captured [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		captured = append(captured, request.Input)
		type row struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		}
		data := make([]row, len(request.Input))
		for i := range request.Input {
			data[i] = row{Object: "embedding", Index: i, Embedding: []float64{1, 0}}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "embedding-contract",
			"data":   data,
			"usage":  map[string]int{"prompt_tokens": len(request.Input), "total_tokens": len(request.Input)},
		})
	}))
	t.Cleanup(server.Close)

	client := openaisdk.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	embedder, err := New(client, Config{
		Model:          "embedding-contract",
		Dimensions:     2,
		DocumentPrefix: "search_document: ",
		QueryPrefix:    "search_query: ",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx := context.Background()
	batches := []struct {
		kind  embeddingcontract.Kind
		input string
	}{
		{embeddingcontract.KindDocument, "annual report"},
		{embeddingcontract.KindQuery, "annual revenue"},
	}
	for _, batch := range batches {
		request := embeddingcontract.Request{Inputs: []string{batch.input}, Kind: batch.kind}
		if _, err := embedder.Embed(ctx, request); err != nil {
			t.Fatalf("Embed(kind %d): %v", batch.kind, err)
		}
	}

	if len(captured) != 2 {
		t.Fatalf("captured %d batches, want 2", len(captured))
	}
	if captured[0][0] != "search_document: annual report" {
		t.Fatalf("document input = %q", captured[0][0])
	}
	if captured[1][0] != "search_query: annual revenue" {
		t.Fatalf("query input = %q", captured[1][0])
	}
}
