package chunking

import (
	"context"
	"errors"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/internal/validate"
)

// BatchEmbedderFromEmbedder adapts the provider-neutral embedding.Embedder
// contract to the BatchEmbedder consumed by NewSemanticChunker. model selects
// the embedding space; kind selects the task role (the zero value,
// KindDocument, is correct for chunking). Adapters that add task prefixes
// honor the kind, so pass KindDocument unless the target model requires
// otherwise.
func BatchEmbedderFromEmbedder(e embedding.Embedder, model embedding.ModelRef, kind embedding.Kind) BatchEmbedder {
	return &embedderBridge{embedder: e, model: model, kind: kind}
}

// embedderBridge is the embedding.Embedder → chunking.BatchEmbedder adapter.
type embedderBridge struct {
	embedder embedding.Embedder
	model    embedding.ModelRef
	kind     embedding.Kind
}

// EmbedBatch embeds texts through the wrapped Embedder, preserving input
// order. Empty batches short-circuit to nil, nil.
func (b *embedderBridge) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if validate.IsNil(b.embedder) {
		return nil, errors.New("chunking: batch embedder has no underlying embedding.Embedder")
	}
	if len(texts) == 0 {
		return nil, nil
	}

	result, err := b.embedder.Embed(ctx, embedding.Request{
		Model:  b.model,
		Inputs: texts,
		Kind:   b.kind,
	})
	if err != nil {
		return nil, err
	}
	if len(result.Vectors) != len(texts) {
		return nil, embedding.ErrInvalidResult
	}

	vectors := make([][]float32, len(result.Vectors))
	for i, v := range result.Vectors {
		vectors[i] = v
	}
	return vectors, nil
}
