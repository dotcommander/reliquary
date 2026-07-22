package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/embeddingtest"
)

func TestEmbeddingContract(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request embedRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		vectors := make([]embeddingcontract.Vector, len(request.Input))
		for i, input := range request.Input {
			var checksum int
			for _, value := range []byte(input) {
				checksum += int(value)
			}
			vectors[i] = embeddingcontract.Vector{float32(len(input)), float32(checksum)}
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(embedResponse{Model: request.Model, Embeddings: vectors}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	embedder, err := New(server.Client(), Config{BaseURL: server.URL, Model: "embedding-contract", Dimensions: 2})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	embeddingtest.Run(t, func() embeddingcontract.Embedder { return embedder })
}

func TestNewValidatesConfigWithoutIO(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	client := &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected I/O")
	})}

	embedder, err := New(client, Config{BaseURL: "http://localhost:11434/", Model: " nomic-embed-text "})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("constructor performed %d HTTP calls", calls.Load())
	}
	if embedder.endpoint != "http://localhost:11434/api/embed" || embedder.model != "nomic-embed-text" || embedder.dims != 0 {
		t.Fatalf("embedder = %#v", embedder)
	}

	tests := []struct {
		name   string
		client *http.Client
		config Config
	}{
		{name: "nil client", config: Config{BaseURL: "http://localhost:11434", Model: "model"}},
		{name: "missing base URL", client: client, config: Config{Model: "model"}},
		{name: "relative base URL", client: client, config: Config{BaseURL: "localhost:11434", Model: "model"}},
		{name: "unsupported scheme", client: client, config: Config{BaseURL: "ftp://localhost", Model: "model"}},
		{name: "base URL query", client: client, config: Config{BaseURL: "http://localhost?x=1", Model: "model"}},
		{name: "blank model", client: client, config: Config{BaseURL: "http://localhost", Model: " \t"}},
		{name: "negative dimensions", client: client, config: Config{BaseURL: "http://localhost", Model: "model", Dimensions: -1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := New(test.client, test.config); err == nil {
				t.Fatal("New() error = nil")
			}
		})
	}
}

func TestEmbedWireShapeOrderAndOverrides(t *testing.T) {
	t.Parallel()

	var captured embedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/ollama/api/embed" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = io.WriteString(w, `{"model":"response-model","embeddings":[[1,2,3],[4,5,6]]}`)
	}))
	t.Cleanup(server.Close)

	embedder, err := New(server.Client(), Config{BaseURL: server.URL + "/ollama", Model: "default-model", Dimensions: 8})
	if err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(context.Background(), embeddingcontract.Request{
		Model:  embeddingcontract.ModelRef{Provider: "caller", Name: "request-model", Version: "v2", Revision: "r1", Dim: 3},
		Inputs: []string{"first", "second"},
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if captured.Model != "request-model" || captured.Dimensions != 3 || len(captured.Input) != 2 || captured.Input[0] != "first" || captured.Input[1] != "second" {
		t.Fatalf("request = %#v", captured)
	}
	if result.Model != (embeddingcontract.ModelRef{Provider: "ollama", Name: "response-model", Version: "v2", Revision: "r1", Dim: 3}) {
		t.Fatalf("model = %#v", result.Model)
	}
	if result.Vectors[0][0] != 1 || result.Vectors[1][0] != 4 {
		t.Fatalf("vectors = %#v", result.Vectors)
	}
}

func TestEmbedInfersDimensionsAndOmitsUnspecifiedDimensions(t *testing.T) {
	t.Parallel()

	var raw map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = io.WriteString(w, `{"model":"nomic","embeddings":[[0.1,0.2]]}`)
	}))
	t.Cleanup(server.Close)

	embedder, err := New(server.Client(), Config{BaseURL: server.URL, Model: "nomic"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(context.Background(), embeddingcontract.Request{Inputs: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := raw["dimensions"]; ok {
		t.Fatalf("request unexpectedly included dimensions: %#v", raw)
	}
	if result.Model.Provider != "ollama" || result.Model.Name != "nomic" || result.Model.Dim != 2 {
		t.Fatalf("model = %#v", result.Model)
	}
}

func TestEmbedEmptyInputDoesNotPerformIO(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("unexpected I/O")
	})}
	embedder, err := New(client, Config{BaseURL: "http://localhost:11434", Model: "default", Dimensions: 7})
	if err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(context.Background(), embeddingcontract.Request{Model: embeddingcontract.ModelRef{Provider: "caller", Version: "v1"}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model.Provider != "ollama" || result.Model.Name != "default" || result.Model.Version != "v1" || result.Model.Dim != 7 || result.Vectors != nil {
		t.Fatalf("result = %#v", result)
	}
}

func TestEmbedCancellation(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		<-request.Context().Done()
		return nil, request.Context().Err()
	})}
	embedder, err := New(client, Config{BaseURL: "http://localhost:11434", Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = embedder.Embed(ctx, embeddingcontract.Request{Inputs: []string{"text"}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Embed() error = %v, want context.Canceled", err)
	}
}

func TestEmbedRejectsProviderFailures(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		status        int
		body          string
		contentLength int64
	}{
		{name: "HTTP status", status: http.StatusBadGateway, body: `{"error":"offline"}`},
		{name: "invalid JSON", status: http.StatusOK, body: `{`},
		{name: "wrong count", status: http.StatusOK, body: `{"embeddings":[]}`},
		{name: "zero dimensions", status: http.StatusOK, body: `{"embeddings":[[]]}`},
		{name: "configured dimension mismatch", status: http.StatusOK, body: `{"embeddings":[[1,2]]}`},
		{name: "inconsistent dimensions", status: http.StatusOK, body: `{"embeddings":[[1,2,3],[1,2]]}`},
		{name: "nonfinite", status: http.StatusOK, body: `{"embeddings":[[1e1000,2,3]]}`},
		{name: "oversized declared response", status: http.StatusOK, body: `{}`, contentLength: maxResponseBytes + 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if test.contentLength > 0 {
					w.Header().Set("Content-Length", fmt.Sprint(test.contentLength))
				}
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			t.Cleanup(server.Close)
			embedder, err := New(server.Client(), Config{BaseURL: server.URL, Model: "model", Dimensions: 3})
			if err != nil {
				t.Fatal(err)
			}
			inputs := []string{"one"}
			if test.name == "inconsistent dimensions" {
				inputs = append(inputs, "two")
			}
			if _, err := embedder.Embed(context.Background(), embeddingcontract.Request{Inputs: inputs}); err == nil {
				t.Fatal("Embed() error = nil")
			}
		})
	}
}

func TestEmbedCapsErrorBodies(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, strings.Repeat("x", maxErrorBytes+1024))
	}))
	t.Cleanup(server.Close)
	embedder, err := New(server.Client(), Config{BaseURL: server.URL, Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = embedder.Embed(context.Background(), embeddingcontract.Request{Inputs: []string{"one"}})
	if err == nil || len(err.Error()) > maxErrorBytes+256 || !strings.HasSuffix(err.Error(), "...") {
		t.Fatalf("Embed() error length = %d, error = %.100q", len(err.Error()), err)
	}
}

func TestReadBodyReportsOversizedAndReadFailures(t *testing.T) {
	t.Parallel()

	body, oversized, err := readBody(strings.NewReader("12345"), 4)
	if err != nil || !oversized || string(body) != "1234" {
		t.Fatalf("readBody oversized = %q, %v, %v", body, oversized, err)
	}
	_, _, err = readBody(errorReader{}, 4)
	if err == nil {
		t.Fatal("readBody error = nil")
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}
