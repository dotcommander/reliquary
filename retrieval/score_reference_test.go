package retrieval

import (
	"math"
	"strings"
	"testing"
)

func TestFitScoreReferenceCleansSortsAndValidates(t *testing.T) {
	t.Parallel()

	values := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	for i := MinScoreReferenceSamples; i >= 1; i-- {
		values = append(values, float64(i))
	}
	identity := ScoreReferenceIdentity{
		ScoreVersion: "score-v1",
		ModelID:      "model-a",
		SchemaHash:   "schema-a",
		ConfigHash:   "config-a",
	}

	reference, err := FitScoreReference(values, identity)
	if err != nil {
		t.Fatalf("FitScoreReference() error = %v, want nil", err)
	}
	if reference.Version != ScoreReferenceVersion {
		t.Fatalf("Version = %d, want %d", reference.Version, ScoreReferenceVersion)
	}
	if reference.CreatedAt.IsZero() {
		t.Fatal("CreatedAt is zero")
	}
	if len(reference.Values) != MinScoreReferenceSamples {
		t.Fatalf("Values len = %d, want %d", len(reference.Values), MinScoreReferenceSamples)
	}
	for i := 1; i < len(reference.Values); i++ {
		if reference.Values[i] < reference.Values[i-1] {
			t.Fatalf("Values not sorted at %d: %v", i, reference.Values)
		}
	}
	if reference.Values[0] != 1 || reference.Values[len(reference.Values)-1] != MinScoreReferenceSamples {
		t.Fatalf("Values bounds = %v..%v, want 1..%d", reference.Values[0], reference.Values[len(reference.Values)-1], MinScoreReferenceSamples)
	}
	if err := reference.Validate(identity); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
}

func TestFitScoreReferenceRejectsSmallFiniteSample(t *testing.T) {
	t.Parallel()

	values := make([]float64, 0, MinScoreReferenceSamples)
	for i := 0; i < MinScoreReferenceSamples-1; i++ {
		values = append(values, float64(i))
	}
	values = append(values, math.NaN())
	_, err := FitScoreReference(values, ScoreReferenceIdentity{})
	if err == nil {
		t.Fatal("FitScoreReference() error = nil, want small-sample rejection")
	}
	if !strings.Contains(err.Error(), "finite samples") {
		t.Fatalf("FitScoreReference() error = %q, want finite sample message", err.Error())
	}
}

func TestScoreReferenceValidateReportsIdentityMismatch(t *testing.T) {
	t.Parallel()

	reference := ScoreReference{
		Version: ScoreReferenceVersion,
		Identity: ScoreReferenceIdentity{
			ScoreVersion: "score-v1",
			ModelID:      "model-a",
			SchemaHash:   "schema-a",
			ConfigHash:   "config-a",
		},
	}
	err := reference.Validate(ScoreReferenceIdentity{
		ScoreVersion: "score-v2",
		ModelID:      "model-b",
		SchemaHash:   "schema-b",
		ConfigHash:   "config-b",
	})
	if err == nil {
		t.Fatal("Validate() error = nil, want mismatch")
	}
	msg := err.Error()
	for _, want := range []string{
		`score_version="score-v1" expected="score-v2"`,
		`model_id="model-a" expected="model-b"`,
		`schema_hash="schema-a" expected="schema-b"`,
		`config_hash="config-a" expected="config-b"`,
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("Validate() error = %q, want substring %q", msg, want)
		}
	}
}

func TestScoreReferenceValidateZeroExpectedIsBackwardCompatible(t *testing.T) {
	t.Parallel()

	if err := (ScoreReference{}).Validate(ScoreReferenceIdentity{}); err != nil {
		t.Fatalf("zero reference Validate(zero identity) error = %v, want nil", err)
	}
}

func TestScoreReferencePercentileBoundaries(t *testing.T) {
	t.Parallel()

	reference := ScoreReference{Values: []float64{10, 20, 30, 40}}
	tests := []struct {
		score float64
		want  float64
	}{
		{score: math.NaN(), want: 0},
		{score: 0, want: 0},
		{score: 10, want: 0},
		{score: 20, want: 1.0 / 3.0},
		{score: 25, want: 1.0 / 3.0},
		{score: 40, want: 1},
		{score: 100, want: 1},
	}
	for _, tc := range tests {
		got := reference.Percentile(tc.score)
		if math.Abs(got-tc.want) > floatTol {
			t.Fatalf("Percentile(%v) = %v, want %v", tc.score, got, tc.want)
		}
	}
}

func TestScorerRerankWithReferenceMapsCombinedScoreToPercentile(t *testing.T) {
	t.Parallel()

	scorer := NewScorerWithOptions(Weights{Recency: 1}, false)
	results := []*Result{
		{ID: "weak", RecencyScore: 0.25},
		{ID: "strong", RecencyScore: 1},
	}
	reference := ScoreReference{Values: []float64{0, 0.5, 1}}

	ranked := scorer.RerankWithReference(nil, "", results, reference)
	if ranked[0].ID != "strong" {
		t.Fatalf("top ID = %q, want strong", ranked[0].ID)
	}
	if ranked[0].CombinedScore != 1 {
		t.Fatalf("strong CombinedScore = %v, want percentile 1", ranked[0].CombinedScore)
	}
	if ranked[1].CombinedScore < 0 || ranked[1].CombinedScore > 1 {
		t.Fatalf("weak CombinedScore = %v, want percentile in [0,1]", ranked[1].CombinedScore)
	}
}
