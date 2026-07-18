package retrieval

import (
	"context"
	"errors"
	"fmt"

	"github.com/dotcommander/reliquary/embedding"
)

// ErrEmbeddingCountMismatch reports that result and embedding batches no longer
// describe the same candidate set.
var ErrEmbeddingCountMismatch = errors.New("retrieval: embedding count mismatch")

// EmbeddingVector converts a provider-neutral embeddings.Vector into the
// float64 vector space used by retrieval scoring.
func EmbeddingVector(v embeddings.Vector) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// EmbeddingVectors converts a batch of provider-neutral embedding vectors into
// retrieval vectors.
func EmbeddingVectors(vectors []embeddings.Vector) [][]float64 {
	out := make([][]float64, len(vectors))
	for i, v := range vectors {
		out[i] = EmbeddingVector(v)
	}
	return out
}

// EmbedResults embeds each result's Content with the embedder and attaches the
// resulting vectors in place. It is the glue between ResultsFromDocuments and
// scoring, so callers no longer hand-roll the texts -> Embed -> AttachEmbeddings loop.
func EmbedResults(ctx context.Context, e embeddings.Embedder, results []*Result) error {
	if len(results) == 0 {
		return nil
	}
	texts := make([]string, len(results))
	for i, r := range results {
		texts[i] = r.Content
	}
	embedded, err := e.Embed(ctx, embeddings.Request{Inputs: texts})
	if err != nil {
		return err
	}
	return AttachEmbeddings(results, embedded.Vectors)
}

// AttachEmbeddings copies embedding vectors onto matching retrieval results by
// index. It returns an error rather than silently dropping vectors because a
// count mismatch means the caller's scoring identity is ambiguous.
func AttachEmbeddings(results []*Result, vectors []embeddings.Vector) error {
	if len(results) != len(vectors) {
		return fmt.Errorf("%w: %d results, %d vectors", ErrEmbeddingCountMismatch, len(results), len(vectors))
	}
	for i, v := range vectors {
		if results[i] == nil {
			continue
		}
		results[i].Embedding = EmbeddingVector(v)
	}
	return nil
}

// RerankEmbedding scores results with an embeddings.Vector query.
func (s *Scorer) RerankEmbedding(queryEmbedding embeddings.Vector, queryText string, results []*Result) []*Result {
	return s.Rerank(EmbeddingVector(queryEmbedding), queryText, results)
}
