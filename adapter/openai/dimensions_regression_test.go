package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func TestEmbeddingModelDimensionsOnWire(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name              string
		cfg               Config
		model             embeddingcontract.ModelRef
		wantModel         string
		wantDim           int
		omitDimensions    bool
		responseDim       int
		wantInvalidResult bool
	}{
		{name: "ada default", cfg: Config{Model: adaModel}, wantModel: adaModel, wantDim: 1536, omitDimensions: true},
		{name: "ada explicit native", cfg: Config{Model: adaModel, Dimensions: 1536}, wantModel: adaModel, wantDim: 1536, omitDimensions: true},
		{name: "ada request override", cfg: Config{Dimensions: 8}, model: embeddingcontract.ModelRef{Name: adaModel}, wantModel: adaModel, wantDim: 1536, omitDimensions: true},
		{name: "ada request native", model: embeddingcontract.ModelRef{Name: adaModel, Dim: 1536}, wantModel: adaModel, wantDim: 1536, omitDimensions: true},
		{name: "ada invalid returned width", cfg: Config{Model: adaModel}, wantModel: adaModel, wantDim: 1536, omitDimensions: true, responseDim: 1535, wantInvalidResult: true},
		{name: "small default", wantModel: "text-embedding-3-small", wantDim: 1536},
		{name: "large current default", cfg: Config{Model: "text-embedding-3-large"}, wantModel: "text-embedding-3-large", wantDim: 1536},
		{name: "v3 configured width", cfg: Config{Model: "text-embedding-3-large", Dimensions: 8}, wantModel: "text-embedding-3-large", wantDim: 8},
		{name: "v3 request width", cfg: Config{Dimensions: 8}, model: embeddingcontract.ModelRef{Dim: 4}, wantModel: "text-embedding-3-small", wantDim: 4},
		{name: "unknown configured width", cfg: Config{Model: "custom-embedding", Dimensions: 7}, wantModel: "custom-embedding", wantDim: 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			captured := make(chan map[string]json.RawMessage, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				captured <- request
				dim := tc.responseDim
				if dim == 0 {
					dim = tc.wantDim
				}
				vector := make([]float64, dim)
				vector[0] = 1
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(map[string]any{
					"object": "list", "model": tc.wantModel,
					"data":  []map[string]any{{"object": "embedding", "index": 0, "embedding": vector}},
					"usage": map[string]int{"prompt_tokens": 1, "total_tokens": 1},
				}); err != nil {
					t.Errorf("encode response: %v", err)
				}
			}))
			t.Cleanup(server.Close)
			client := openaisdk.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))
			embedder, err := New(client, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			result, err := embedder.Embed(context.Background(), embeddingcontract.Request{Model: tc.model, Inputs: []string{"input"}})
			if tc.wantInvalidResult {
				if !errors.Is(err, embeddingcontract.ErrInvalidResult) {
					t.Fatalf("error = %v, want invalid result", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if result.Model.Dim != tc.wantDim || len(result.Vectors) != 1 || len(result.Vectors[0]) != tc.wantDim {
					t.Fatalf("model dimension = %d, vectors = %d, want dimension %d", result.Model.Dim, len(result.Vectors), tc.wantDim)
				}
			}
			var request map[string]json.RawMessage
			select {
			case request = <-captured:
			default:
				t.Fatal("no request captured")
			}
			var model string
			if err := json.Unmarshal(request["model"], &model); err != nil || model != tc.wantModel {
				t.Fatalf("wire model = %q, error = %v, want %q", model, err, tc.wantModel)
			}
			wireDimensions, present := request["dimensions"]
			if tc.omitDimensions {
				if present {
					t.Fatalf("Ada request includes dimensions: %s", wireDimensions)
				}
			} else {
				var dim int
				if !present {
					t.Fatal("dimensions missing from request")
				}
				if err := json.Unmarshal(wireDimensions, &dim); err != nil || dim != tc.wantDim {
					t.Fatalf("wire dimension = %d, error = %v, want %d", dim, err, tc.wantDim)
				}
			}
		})
	}
}

func TestAdaRejectsIncompatibleDimensionsBeforeClient(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		http.Error(w, "unexpected call", http.StatusBadRequest)
	}))
	t.Cleanup(server.Close)
	client := openaisdk.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL), option.WithMaxRetries(0))
	for _, dim := range []int{-1, 8, 1535, 1537} {
		if _, err := New(client, Config{Model: adaModel, Dimensions: dim}); err == nil {
			t.Fatalf("New accepted incompatible Ada dimension %d", dim)
		}
	}
	embedder, err := New(client, Config{Dimensions: 8})
	if err != nil {
		t.Fatal(err)
	}
	for _, dim := range []int{-1, 8, 1535, 1537} {
		result, err := embedder.Embed(context.Background(), embeddingcontract.Request{
			Model: embeddingcontract.ModelRef{Name: adaModel, Dim: dim}, Inputs: []string{"input"},
		})
		if err == nil || result.Vectors != nil || result.Model != (embeddingcontract.ModelRef{}) {
			t.Fatalf("Embed dimension %d: result=%+v, error=%v", dim, result, err)
		}
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("client called %d times for rejected dimensions", got)
	}
}
