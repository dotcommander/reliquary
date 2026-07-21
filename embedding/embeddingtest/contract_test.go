package embeddingtest

import (
	"context"
	"testing"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
)

func TestContractSuite(t *testing.T) {
	t.Parallel()

	Run(t, func() embeddingcontract.Embedder {
		return orderedEmbedder{}
	})
}

type orderedEmbedder struct{}

func (orderedEmbedder) Embed(_ context.Context, request embeddingcontract.Request) (embeddingcontract.Result, error) {
	vectors := make([]embeddingcontract.Vector, len(request.Inputs))
	for i, input := range request.Inputs {
		var firstByte float32
		if len(input) > 0 {
			firstByte = float32(input[0])
		}
		vectors[i] = embeddingcontract.Vector{float32(len(input)), firstByte}
	}
	return embeddingcontract.Result{
		Model:   embeddingcontract.ModelRef{Name: "ordered", Dim: 2},
		Vectors: vectors,
	}, nil
}
