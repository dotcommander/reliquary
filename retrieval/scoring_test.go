package retrieval

import "testing"

func TestCalibratedScoreBands(t *testing.T) {
	t.Parallel()

	strong := CalibratedScore(ScoreComponents{Semantic: 0.95, Keyword: 1, Filename: 1, Metadata: 1})
	weak := CalibratedScore(ScoreComponents{Semantic: -0.5})
	if strong <= weak {
		t.Fatalf("strong score %v <= weak score %v", strong, weak)
	}
	if Band(strong) != BandStrong {
		t.Fatalf("Band(strong) = %q, want strong", Band(strong))
	}
	if Band(weak) != BandWeak {
		t.Fatalf("Band(weak) = %q, want weak", Band(weak))
	}
}

func TestCalibratedScoreReducesCorpusVariance(t *testing.T) {
	t.Parallel()

	small := CalibratedScore(ScoreComponents{Semantic: 0.8, Keyword: 0.5})
	noisy := CalibratedScore(ScoreComponents{Semantic: 0.8, Keyword: 0.5, Filename: 0.1})
	if diff := small - noisy; diff < -0.02 || diff > 0.02 {
		t.Fatalf("calibrated score drift = %v, want stable threshold band", diff)
	}
}
