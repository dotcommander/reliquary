// Package ollama adapts Ollama's native embed API to Reliquary's embedding
// contract. Callers own the HTTP client, transport, timeout, and retry policy.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
)

const (
	maxResponseBytes = 64 << 20
	maxErrorBytes    = 64 << 10
)

// Config configures the Ollama endpoint and embedding request defaults.
type Config struct {
	BaseURL    string
	Model      string
	Dimensions int
}

// Embedder implements Reliquary's embedding contract with Ollama's native
// POST /api/embed endpoint.
type Embedder struct {
	client   *http.Client
	endpoint string
	model    string
	dims     int
}

var _ embeddingcontract.Embedder = (*Embedder)(nil)

// New validates cfg and constructs an embedder. It performs no network, file,
// environment, model-pull, or health-check I/O.
func New(client *http.Client, cfg Config) (*Embedder, error) {
	if client == nil {
		return nil, fmt.Errorf("ollama adapter: HTTP client is required")
	}

	baseURL, err := url.Parse(strings.TrimSpace(cfg.BaseURL))
	if err != nil || baseURL.Scheme == "" || baseURL.Host == "" || (baseURL.Scheme != "http" && baseURL.Scheme != "https") {
		return nil, fmt.Errorf("ollama adapter: base URL must be an absolute HTTP(S) URL")
	}
	if baseURL.User != nil || baseURL.RawQuery != "" || baseURL.Fragment != "" {
		return nil, fmt.Errorf("ollama adapter: base URL must not contain user info, a query, or a fragment")
	}

	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		return nil, fmt.Errorf("ollama adapter: model is required")
	}
	if cfg.Dimensions < 0 {
		return nil, fmt.Errorf("ollama adapter: dimensions must be nonnegative")
	}

	endpoint := strings.TrimRight(baseURL.String(), "/") + "/api/embed"
	return &Embedder{client: client, endpoint: endpoint, model: model, dims: cfg.Dimensions}, nil
}

// Embed generates one vector per input in a single provider call.
func (e *Embedder) Embed(ctx context.Context, request embeddingcontract.Request) (embeddingcontract.Result, error) {
	model := request.Model
	model.Provider = "ollama"
	if strings.TrimSpace(model.Name) == "" {
		model.Name = e.model
	}
	if model.Dim == 0 {
		model.Dim = e.dims
	}
	if model.Dim < 0 {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: dimensions must be nonnegative")
	}
	if len(request.Inputs) == 0 {
		return embeddingcontract.Result{Model: model}, nil
	}

	payload, err := json.Marshal(embedRequest{
		Model:      model.Name,
		Input:      request.Inputs,
		Dimensions: model.Dim,
	})
	if err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: encode request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint, bytes.NewReader(payload))
	if err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: create request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "application/json")

	response, err := e.client.Do(httpRequest)
	if err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: embed: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		body, truncated, readErr := readBody(response.Body, maxErrorBytes)
		if readErr != nil {
			return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: HTTP %s: read error response: %w", response.Status, readErr)
		}
		message := strings.TrimSpace(string(body))
		if truncated {
			message += "..."
		}
		if message == "" {
			return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: HTTP %s", response.Status)
		}
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: HTTP %s: %s", response.Status, message)
	}
	if response.ContentLength > maxResponseBytes {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: response exceeds %d bytes", maxResponseBytes)
	}

	body, oversized, err := readBody(response.Body, maxResponseBytes)
	if err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: read response: %w", err)
	}
	if oversized {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: response exceeds %d bytes", maxResponseBytes)
	}

	var decoded embedResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: decode response: %w", err)
	}
	if strings.TrimSpace(decoded.Model) != "" {
		model.Name = decoded.Model
	}
	model.Provider = "ollama"
	if model.Dim == 0 && len(decoded.Embeddings) > 0 {
		model.Dim = len(decoded.Embeddings[0])
	}

	result := embeddingcontract.Result{Model: model, Vectors: decoded.Embeddings}
	validationRequest := request
	validationRequest.Model = model
	if err := embeddingcontract.ValidateResult(validationRequest, result); err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("ollama adapter: %w", err)
	}
	return result, nil
}

type embedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type embedResponse struct {
	Model      string                     `json:"model"`
	Embeddings []embeddingcontract.Vector `json:"embeddings"`
}

func readBody(reader io.Reader, limit int64) ([]byte, bool, error) {
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) <= limit {
		return body, false, nil
	}
	return body[:limit], true, nil
}
