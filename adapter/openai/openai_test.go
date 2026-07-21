package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/embeddingtest"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func TestEmbeddingContract(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		type row struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		}
		data := make([]row, 0, len(request.Input))
		for i := len(request.Input) - 1; i >= 0; i-- {
			var checksum int
			for _, value := range []byte(request.Input[i]) {
				checksum += int(value)
			}
			data = append(data, row{Object: "embedding", Index: i, Embedding: []float64{float64(len(request.Input[i])), float64(checksum)}})
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "embedding-contract",
			"data":   data,
			"usage":  map[string]int{"prompt_tokens": len(request.Input), "total_tokens": len(request.Input)},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	client := openaisdk.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	embedder, err := New(client, Config{Model: "embedding-contract", Dimensions: 2})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	embeddingtest.Run(t, func() embeddingcontract.Embedder {
		return embedder
	})
}

func TestNewDoesNotPerformIOAndValidatesConfig(t *testing.T) {
	client := openaisdk.NewClient(option.WithAPIKey("unused"), option.WithBaseURL("http://127.0.0.1:1"))
	embedder, err := New(client, Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if embedder.model != defaultModel || embedder.dims != defaultDimensions {
		t.Fatalf("defaults = %#v", embedder)
	}
	if _, err := New(client, Config{Dimensions: -1}); err == nil {
		t.Fatal("New() error = nil for invalid dimensions")
	}
}

func TestEmbedMapsBatchAndOrdersVectorsByIndex(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","model":"embedding-response","data":[{"object":"embedding","index":1,"embedding":[0.3,0.4]},{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":2,"total_tokens":2}}`))
	}))
	defer server.Close()

	client := openaisdk.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	embedder, err := New(client, Config{Model: "embedding-default", Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(context.Background(), embeddingcontract.Request{
		Model:  embeddingcontract.ModelRef{Provider: "caller", Version: "2", Revision: "r1"},
		Inputs: []string{"hello\nworld", "second"},
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	inputs, ok := captured["input"].([]any)
	if !ok || len(inputs) != 2 || inputs[0] != "hello world" || captured["model"] != "embedding-default" || captured["dimensions"] != float64(2) {
		t.Fatalf("request = %#v", captured)
	}
	if result.Vectors[0][0] != float32(0.1) || result.Vectors[1][0] != float32(0.3) {
		t.Fatalf("vectors = %#v", result.Vectors)
	}
	if result.Model.Provider != "openai" || result.Model.Name != "embedding-response" || result.Model.Version != "2" || result.Model.Revision != "r1" {
		t.Fatalf("model = %#v", result.Model)
	}
}

func TestEmbedEmptyInputsAndDimensionMismatch(t *testing.T) {
	client := openaisdk.NewClient(option.WithAPIKey("unused"), option.WithBaseURL("http://127.0.0.1:1"))
	embedder, err := New(client, Config{Model: "embedding-default", Dimensions: 8})
	if err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(context.Background(), embeddingcontract.Request{
		Model: embeddingcontract.ModelRef{Provider: "caller", Version: "2", Revision: "r1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model.Provider != "openai" || result.Model.Name != "embedding-default" || result.Model.Version != "2" || result.Model.Revision != "r1" || result.Model.Dim != 8 || result.Vectors != nil {
		t.Fatalf("result = %#v", result)
	}
}
