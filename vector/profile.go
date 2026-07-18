package vectors

import (
	"strconv"

	"github.com/dotcommander/reliquary/internal/hash"
)

// ANNProfile describes one approximate-nearest-neighbor benchmark profile.
// It is data-only and does not imply a specific ANN implementation.
type ANNProfile struct {
	ID              string
	Kind            IndexKind
	CandidateLimit  int
	Oversampling    float64
	ExactRescore    bool
	Quantization    string
	MemoryEstimate  MemoryEstimate
	TransformDigest hash.Digest
	ProfileDigest   hash.Digest
}

// MemoryEstimate records caller-measured or estimated index memory.
type MemoryEstimate struct {
	Bytes int64
	Label string
}

// LatencySummary records deterministic benchmark sample summaries. Tests should
// compare explicit values, not wall-clock thresholds.
type LatencySummary struct {
	Samples int
	MinMS   float64
	P50MS   float64
	P95MS   float64
	MaxMS   float64
}

// ANNProfileResult records quality and latency observations for a profile.
type ANNProfileResult struct {
	Profile        ANNProfile
	RecallAtK      float64
	Latency        LatencySummary
	IndexedRows    int
	EvaluatedQuery int
}

// HashANNProfile returns a stable profile digest for ANN comparison inputs.
func HashANNProfile(profile ANNProfile) hash.Digest {
	return hash.HashIdentity(
		hash.IdentityPart{Kind: "ann_profile", ID: profile.ID},
		hash.IdentityPart{Kind: "index", Value: string(profile.Kind)},
		hash.IdentityPart{Kind: "candidate_limit", Value: intString(profile.CandidateLimit)},
		hash.IdentityPart{Kind: "oversampling", Value: floatString(profile.Oversampling)},
		hash.IdentityPart{Kind: "exact_rescore", Value: boolString(profile.ExactRescore)},
		hash.IdentityPart{Kind: "quantization", Value: profile.Quantization},
		hash.IdentityPart{Kind: "transform", Digest: profile.TransformDigest},
	)
}

func intString(value int) string {
	if value == 0 {
		return ""
	}
	return strconv.Itoa(value)
}

func floatString(value float64) string {
	if value == 0 {
		return ""
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func boolString(value bool) string {
	if !value {
		return ""
	}
	return strconv.FormatBool(value)
}
