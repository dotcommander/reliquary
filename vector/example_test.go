package vectors_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/internal/hash"
	"github.com/dotcommander/reliquary/vector"
)

func ExampleCosine32() {
	a := []float32{1, 2, 3}
	b := []float32{1, 2, 4}

	vectors.Normalize32(a)
	vectors.Normalize32(b)

	fmt.Printf("%.3f\n", vectors.Cosine32(a, b))
	// Output: 0.991
}

func ExampleANNProfileResult() {
	transform := hash.HashIdentity(
		hash.IdentityPart{Kind: "embedding_model", ID: "demo-hash", Version: "1"},
		hash.IdentityPart{Kind: "chunker", Version: "semantic-v1", ConfigHash: "chunk-cfg"},
	)
	exact := vectors.ANNProfile{
		ID:              "exact-baseline",
		Kind:            vectors.IndexKindExact,
		CandidateLimit:  100,
		TransformDigest: transform,
	}
	binary := vectors.ANNProfile{
		ID:              "binary-screen",
		Kind:            vectors.IndexKindBinary,
		CandidateLimit:  300,
		Oversampling:    3,
		ExactRescore:    true,
		Quantization:    "median_binary",
		TransformDigest: transform,
	}
	binary.ProfileDigest = vectors.HashANNProfile(binary)

	results := []vectors.ANNProfileResult{
		{Profile: exact, RecallAtK: 1, Latency: vectors.LatencySummary{Samples: 20, P95MS: 4.2}},
		{Profile: binary, RecallAtK: 0.92, Latency: vectors.LatencySummary{Samples: 20, P95MS: 1.1}},
	}

	fmt.Println(results[0].Profile.ID, results[1].Profile.Quantization, results[1].Profile.ProfileDigest.String() != "")
	// Output: exact-baseline median_binary true
}
