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
	adaModel          = "text-embedding-ada-002"
	adaDimensions     = 1536
)

// Config configures embedding request defaults.
type Config struct {
	Model      string
	Dimensions int
	// DocumentPrefix is prepended to every KindDocument input before the
	// provider call. Empty means no prefix.
	DocumentPrefix string
	// QueryPrefix is prepended to every KindQuery input before the provider
	// call. Empty means no prefix.
	QueryPrefix string
}

// Embedder implements Reliquary's embedding contract with an injected OpenAI
// client.
type Embedder struct {
	client      openaisdk.Client
	model       string
	dims        int
	docPrefix   string
	queryPrefix string
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
	if cfg.Model == adaModel && cfg.Dimensions != adaDimensions {
		return nil, fmt.Errorf("openai adapter: %s requires %d dimensions", adaModel, adaDimensions)
	}
	return &Embedder{client: client, model: cfg.Model, dims: cfg.Dimensions, docPrefix: cfg.DocumentPrefix, queryPrefix: cfg.QueryPrefix}, nil
}

// Embed generates one vector per input in a single provider call.
func (e *Embedder) Embed(ctx context.Context, request embeddingcontract.Request) (embeddingcontract.Result, error) {
	model := request.Model
	model.Provider = "openai"
	if model.Name == "" {
		model.Name = e.model
	}
	if model.Name == adaModel {
		if model.Dim != 0 && model.Dim != adaDimensions {
			return embeddingcontract.Result{}, fmt.Errorf("openai adapter: %s requires %d dimensions", adaModel, adaDimensions)
		}
		model.Dim = adaDimensions
	} else if model.Dim == 0 {
		model.Dim = e.dims
	}
	if model.Dim < 1 {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: dimensions must be positive")
	}
	if len(request.Inputs) == 0 {
		return embeddingcontract.Result{Model: model}, nil
	}

	inputs := make([]string, len(request.Inputs))
	prefix := e.docPrefix
	if request.Kind == embeddingcontract.KindQuery {
		prefix = e.queryPrefix
	}
	for i, input := range request.Inputs {
		inputs[i] = prefix + strings.ReplaceAll(input, "\n", " ")
	}
	params := openaisdk.EmbeddingNewParams{
		Input: openaisdk.EmbeddingNewParamsInputUnion{OfArrayOfStrings: inputs},
		Model: openaisdk.EmbeddingModel(model.Name),
	}
	// Ada uses its native width; only v3 and later support the dimensions field.
	if model.Name != adaModel {
		params.Dimensions = openaisdk.Int(int64(model.Dim))
	}
	response, err := e.client.Embeddings.New(ctx, params)
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
	if response.Model != "" {
		model.Name = response.Model
	}
	result := embeddingcontract.Result{Model: model, Vectors: vectors}
	validatedRequest := request
	validatedRequest.Model = model
	if err := embeddingcontract.ValidateResult(validatedRequest, result); err != nil {
		return embeddingcontract.Result{}, fmt.Errorf("openai adapter: %w", err)
	}
	return result, nil
}
