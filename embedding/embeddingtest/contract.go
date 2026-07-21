// Package embeddingtest provides a reusable contract suite for Embedder implementations.
package embeddingtest

import (
	"context"
	"reflect"
	"testing"

	embeddingcontract "github.com/dotcommander/reliquary/embedding"
)

// Factory returns an Embedder for each contract subtest.
type Factory func() embeddingcontract.Embedder

// Run exercises the behavior required of every Embedder implementation.
func Run(t *testing.T, newEmbedder Factory) {
	t.Helper()

	t.Run("empty batch", func(t *testing.T) {
		t.Parallel()
		request := embeddingcontract.Request{}
		result := mustEmbed(t, newEmbedder(), request)
		if err := embeddingcontract.ValidateResult(request, result); err != nil {
			t.Fatalf("empty batch result: %v", err)
		}
	})

	t.Run("valid batch shape", func(t *testing.T) {
		t.Parallel()
		request := embeddingcontract.Request{Inputs: []string{"alpha", "beta", ""}}
		result := mustEmbed(t, newEmbedder(), request)
		if err := embeddingcontract.ValidateResult(request, result); err != nil {
			t.Fatalf("batch result: %v", err)
		}
	})

	t.Run("batch ordering matches individual calls", func(t *testing.T) {
		t.Parallel()
		inputs := []string{"first input", "second input", "third input"}
		request := embeddingcontract.Request{Inputs: inputs}
		batch := mustEmbed(t, newEmbedder(), request)
		if err := embeddingcontract.ValidateResult(request, batch); err != nil {
			t.Fatalf("batch result: %v", err)
		}

		for i, input := range inputs {
			individualRequest := embeddingcontract.Request{Inputs: []string{input}}
			individual := mustEmbed(t, newEmbedder(), individualRequest)
			if err := embeddingcontract.ValidateResult(individualRequest, individual); err != nil {
				t.Fatalf("individual result %d: %v", i, err)
			}
			if !reflect.DeepEqual(batch.Vectors[i], individual.Vectors[0]) {
				t.Fatalf("batch vector %d does not match the individual result", i)
			}
		}
	})
}

func mustEmbed(t *testing.T, embedder embeddingcontract.Embedder, request embeddingcontract.Request) embeddingcontract.Result {
	t.Helper()
	result, err := embedder.Embed(context.Background(), request)
	if err != nil {
		t.Fatalf("Embed(%d inputs): %v", len(request.Inputs), err)
	}
	return result
}
