package retrieval

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// ErrInvalidRerankResult reports that a reranker returned scores that do not
// correspond one-to-one with its candidates or are not finite values in [0,1].
var ErrInvalidRerankResult = errors.New("retrieval: invalid rerank result")

// Reranker assigns an external relevance score to each candidate for a query.
// Returned scores correspond positionally to candidates.
//
// Separate Search calls may invoke the same Reranker concurrently. Implementations
// that are not concurrency-safe must provide their own synchronization.
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []*Result) ([]float64, error)
}

// ValidateRerankScores verifies that scores contains exactly one finite value
// in [0,1] for each candidate.
func ValidateRerankScores(candidateCount int, scores []float64) error {
	if len(scores) != candidateCount {
		return fmt.Errorf("%w: got %d scores for %d candidates", ErrInvalidRerankResult, len(scores), candidateCount)
	}
	for i, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return fmt.Errorf("%w: score at index %d is not finite", ErrInvalidRerankResult, i)
		}
		if score < 0 || score > 1 {
			return fmt.Errorf("%w: score at index %d is outside [0,1]: %v", ErrInvalidRerankResult, i, score)
		}
	}
	return nil
}
