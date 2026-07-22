package retrieval

import (
	"errors"
	"math"
	"testing"
)

func TestValidateRerankScores(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		candidateCount int
		scores         []float64
		wantErr        bool
	}{
		{name: "zero candidates and nil scores", candidateCount: 0, scores: nil},
		{name: "zero candidates and empty scores", candidateCount: 0, scores: []float64{}},
		{name: "valid boundaries", candidateCount: 3, scores: []float64{0, 0.5, 1}},
		{name: "too few scores", candidateCount: 2, scores: []float64{0.5}, wantErr: true},
		{name: "too many scores", candidateCount: 1, scores: []float64{0.5, 0.6}, wantErr: true},
		{name: "scores for zero candidates", candidateCount: 0, scores: []float64{0}, wantErr: true},
		{name: "negative candidate count", candidateCount: -1, scores: nil, wantErr: true},
		{name: "NaN", candidateCount: 1, scores: []float64{math.NaN()}, wantErr: true},
		{name: "positive infinity", candidateCount: 1, scores: []float64{math.Inf(1)}, wantErr: true},
		{name: "negative infinity", candidateCount: 1, scores: []float64{math.Inf(-1)}, wantErr: true},
		{name: "below range", candidateCount: 1, scores: []float64{-0.0001}, wantErr: true},
		{name: "above range", candidateCount: 1, scores: []float64{1.0001}, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRerankScores(tt.candidateCount, tt.scores)
			if tt.wantErr {
				if !errors.Is(err, ErrInvalidRerankResult) {
					t.Fatalf("ValidateRerankScores() error = %v, want ErrInvalidRerankResult", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateRerankScores() error = %v, want nil", err)
			}
		})
	}
}
