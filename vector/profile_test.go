package vectors

import (
	"testing"

	"github.com/dotcommander/reliquary/internal/hash"
)

func TestHashANNProfileChangesForProfileFields(t *testing.T) {
	t.Parallel()

	base := ANNProfile{
		ID:              "binary-baseline",
		Kind:            IndexKindBinary,
		CandidateLimit:  100,
		Oversampling:    2,
		ExactRescore:    true,
		Quantization:    "binary",
		TransformDigest: hash.SHA256String("transform"),
	}
	baseHash := HashANNProfile(base)
	if baseHash == (hash.Digest{}) {
		t.Fatal("HashANNProfile returned zero digest")
	}

	changed := base
	changed.CandidateLimit = 200
	if got := HashANNProfile(changed); got == baseHash {
		t.Fatalf("candidate limit did not change profile digest: %s", got)
	}

	result := ANNProfileResult{
		Profile:        base,
		RecallAtK:      0.8,
		Latency:        LatencySummary{Samples: 5, P95MS: 1.25},
		IndexedRows:    1000,
		EvaluatedQuery: 10,
	}
	if result.Latency.P95MS != 1.25 || result.RecallAtK != 0.8 {
		t.Fatalf("profile result = %#v", result)
	}
}
