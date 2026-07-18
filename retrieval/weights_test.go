package retrieval

import (
	"math"
	"testing"
)

func TestAdaptiveWeights(t *testing.T) {
	t.Parallel()

	shortWeights := Weights{Embedding: 0.45, Keyword: 0.45, Filename: 0.10}
	defaultW := DefaultWeights() // {0.63, 0.27, 0.10}
	longWeights := Weights{Embedding: 0.765, Keyword: 0.135, Filename: 0.10}

	cases := []struct {
		name       string
		tokenCount int
		want       Weights
	}{
		{"zero tokens", 0, shortWeights},
		{"two tokens", 2, shortWeights},
		{"three tokens", 3, defaultW},
		{"five tokens", 5, defaultW},
		{"six tokens", 6, longWeights},
		{"hundred tokens", 100, longWeights},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := AdaptiveWeights(tc.tokenCount)
			if got != tc.want {
				t.Errorf("AdaptiveWeights(%d) = %+v, want %+v", tc.tokenCount, got, tc.want)
			}
			sum := got.Embedding + got.Keyword + got.Filename
			if math.Abs(sum-1.0) > 1e-9 {
				t.Errorf("AdaptiveWeights(%d) weights sum = %v, want 1.0", tc.tokenCount, sum)
			}
		})
	}
}

func TestScorerScoreUsesRawWeights(t *testing.T) {
	t.Parallel()

	// calibration=false: Score applies s.weights directly without min-max calibration.
	scorer := NewScorerWithOptions(DefaultWeights(), false)

	// "alpha" and "bravo" are each >2 chars and not stopwords, so tokenize produces
	// both tokens. keywordOverlap("alpha bravo", "alpha bravo") = 1.0.
	query := "alpha bravo"
	result := &Result{
		Content:   "alpha bravo",
		Embedding: []float64{1, 0},
		// Filename intentionally empty → FilenameScore stays 0.
	}
	queryEmbedding := []float64{1, 0} // identical → Cosine64 = 1.0

	got := scorer.Score(queryEmbedding, query, result)

	// Expected: 0.63*1.0 + 0.27*1.0 + 0.10*0.0 = 0.90
	const want = 0.90
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("Score() = %v, want %v", got, want)
	}
	if math.Abs(result.CombinedScore-want) > 1e-9 {
		t.Errorf("result.CombinedScore = %v, want %v", result.CombinedScore, want)
	}
	if math.Abs(got-result.CombinedScore) > 1e-9 {
		t.Errorf("Score() return %v != result.CombinedScore %v", got, result.CombinedScore)
	}
}
