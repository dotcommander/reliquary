// Package openai adapts the official OpenAI SDK to Reliquary's embedding
// contract. Callers own client construction, credentials, transport, and retry
// policy.
package openai

import (
	"context"
	"fmt"
	"strings"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
	openaisdk "github.com/openai/openai-go/v3"
)

const (
	defaultModel      = "text-embedding-3-small"
	defaultDimensions = 1536
)

// Config configures embedding request defaults.
type Config struct {
	Model      string
	Dimensions int
}

// Embedder implements Reliquary's embedding contract with an injected OpenAI
// client.
type Embedder struct {
	client openaisdk.Client
	model  string
	dims   int
}

var _ embeddingcontract.Embedder = (*Embedder)(nil)

// New validates cfg and constructs an embedder. It performs no network, file,
// environment, or migration I/O.
func New(client openaisdk.Client, cfg Config) (*Embedder, error) {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = defaultDimensions
	}
	if cfg.Dimensions < 1 {
		return nil, fmt.Errorf("openai adapter: dimensions must be positive")
	}
	return &Embedder{client: client, model: cfg.Model, dims: cfg.Dimensions}, nil
}

// Embed generates one vector per input in a single provider call.
func (e *Embedder) Embed(ctx context.Context, request embeddingcontract.Request) (embeddingcontract.Result, error) {
	model := request.Model
	if model.Provider == "" {
		model.Provider = "openai"
	}
	if model.Name == "" {
		model.Name = e.model
	}
	if model.Dim == 0 {
		model.Dim = e.dims
	}
	if model.Dim < 1 {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: dimensions must be positive")
	}
	if len(request.Inputs) == 0 {
		return embeddingcontract.Result{Model: model}, nil
	}

	inputs := make([]string, len(request.Inputs))
	for i, input := range request.Inputs {
		inputs[i] = strings.ReplaceAll(input, "\n", " ")
	}
	response, err := e.client.Embeddings.New(ctx, openaisdk.EmbeddingNewParams{
		Input:      openaisdk.EmbeddingNewParamsInputUnion{OfArrayOfStrings: inputs},
		Model:      openaisdk.EmbeddingModel(model.Name),
		Dimensions: openaisdk.Int(int64(model.Dim)),
	})
	if err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: embed: %w", err)
	}
	if response == nil {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: empty response")
	}
	if len(response.Data) != len(inputs) {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: expected %d vectors, got %d", len(inputs), len(response.Data))
	}

	vectors := make([]embeddingcontract.Vector, len(response.Data))
	seen := make([]bool, len(response.Data))
	for _, data := range response.Data {
		if data.Index < 0 || data.Index >= int64(len(vectors)) || seen[data.Index] {
			return embeddingcontract.Result{}, fmt.Errorf("openai adapter: invalid vector index %d", data.Index)
		}
		vector := make(embeddingcontract.Vector, len(data.Embedding))
		for i, value := range data.Embedding {
			vector[i] = float32(value)
		}
		vectors[data.Index] = vector
		seen[data.Index] = true
	}
	if err := embeddingcontract.ValidateDimensions(vectors, model.Dim); err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: %w", err)
	}
	if response.Model != "" {
		model.Name = response.Model
	}
	return embeddingcontract.Result{Model: model, Vectors: vectors}, nil
}
