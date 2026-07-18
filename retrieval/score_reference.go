package retrieval

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"time"
)

const (
	// ScoreReferenceVersion is the current ScoreReference schema version.
	ScoreReferenceVersion = 1
	// MinScoreReferenceSamples is the minimum number of finite samples
	// FitScoreReference accepts for a reference cohort.
	MinScoreReferenceSamples = 20
)

// ScoreReferenceIdentity describes scoring inputs that invalidate a reference
// distribution when they change. Zero fields are ignored by Validate.
type ScoreReferenceIdentity struct {
	ScoreVersion string
	ModelID      string
	SchemaHash   string
	ConfigHash   string
}

// ScoreReference stores a sorted production/reference score cohort. Callers can
// persist this data in their own format and map future raw scores to stable
// cohort-relative percentiles with Percentile.
type ScoreReference struct {
	Version   int
	CreatedAt time.Time
	Identity  ScoreReferenceIdentity
	Values    []float64
}

// FitScoreReference cleans and sorts raw reference scores. It rejects cohorts
// smaller than MinScoreReferenceSamples after dropping NaN and infinite values.
func FitScoreReference(values []float64, identity ScoreReferenceIdentity) (ScoreReference, error) {
	clean := make([]float64, 0, len(values))
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			continue
		}
		clean = append(clean, value)
	}
	if len(clean) < MinScoreReferenceSamples {
		return ScoreReference{}, fmt.Errorf("retrieval: score reference needs at least %d finite samples, got %d", MinScoreReferenceSamples, len(clean))
	}
	slices.Sort(clean)
	return ScoreReference{
		Version:   ScoreReferenceVersion,
		CreatedAt: time.Now().UTC(),
		Identity:  identity,
		Values:    clean,
	}, nil
}

// Validate returns an actionable error when the reference identity does not
// match the expected scoring/model/config identity.
func (reference ScoreReference) Validate(expected ScoreReferenceIdentity) error {
	var mismatches []string
	if !expected.zero() && reference.Version != ScoreReferenceVersion {
		mismatches = append(mismatches, fmt.Sprintf("version=%d expected=%d", reference.Version, ScoreReferenceVersion))
	}
	if expected.ScoreVersion != "" && reference.Identity.ScoreVersion != expected.ScoreVersion {
		mismatches = append(mismatches, fmt.Sprintf("score_version=%q expected=%q", reference.Identity.ScoreVersion, expected.ScoreVersion))
	}
	if expected.ModelID != "" && reference.Identity.ModelID != expected.ModelID {
		mismatches = append(mismatches, fmt.Sprintf("model_id=%q expected=%q", reference.Identity.ModelID, expected.ModelID))
	}
	if expected.SchemaHash != "" && reference.Identity.SchemaHash != expected.SchemaHash {
		mismatches = append(mismatches, fmt.Sprintf("schema_hash=%q expected=%q", reference.Identity.SchemaHash, expected.SchemaHash))
	}
	if expected.ConfigHash != "" && reference.Identity.ConfigHash != expected.ConfigHash {
		mismatches = append(mismatches, fmt.Sprintf("config_hash=%q expected=%q", reference.Identity.ConfigHash, expected.ConfigHash))
	}
	if len(mismatches) > 0 {
		return fmt.Errorf("retrieval: score reference mismatch: %s", strings.Join(mismatches, "; "))
	}
	return nil
}

// Percentile maps a raw score to a stable 0..1 cohort-relative percentile. The
// minimum reference score maps to 0 and the maximum maps to 1.
func (reference ScoreReference) Percentile(score float64) float64 {
	if len(reference.Values) == 0 || math.IsNaN(score) {
		return 0
	}
	if len(reference.Values) == 1 {
		if score >= reference.Values[0] {
			return 1
		}
		return 0
	}
	count := sort.Search(len(reference.Values), func(i int) bool { return reference.Values[i] > score })
	if count <= 1 {
		return 0
	}
	if count >= len(reference.Values) {
		return 1
	}
	return float64(count-1) / float64(len(reference.Values)-1)
}

// RerankWithReference scores and sorts results, then maps each CombinedScore to
// its score-reference percentile. The caller should Validate the reference
// identity before applying it.
func (s *Scorer) RerankWithReference(queryEmbedding []float64, queryText string, results []*Result, reference ScoreReference) []*Result {
	ranked := s.Rerank(queryEmbedding, queryText, results)
	for _, result := range ranked {
		result.CombinedScore = reference.Percentile(result.CombinedScore)
	}
	slices.SortStableFunc(ranked, func(a, b *Result) int {
		return cmp.Compare(b.CombinedScore, a.CombinedScore)
	})
	return ranked
}

func (identity ScoreReferenceIdentity) zero() bool {
	return identity.ScoreVersion == "" &&
		identity.ModelID == "" &&
		identity.SchemaHash == "" &&
		identity.ConfigHash == ""
}
