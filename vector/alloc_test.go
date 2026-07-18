package vectors

import (
	"math/rand/v2"
	"testing"
)

// allocBenchVec returns a fresh vector to avoid sharing state across subtests.
func allocBenchVec(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i+1) * 0.01
	}
	return v
}

// TestBenchmarkAllocs enforces zero-allocation targets for hot-path operations.
// If any of these regress, it indicates an allocation leak that will degrade
// throughput in high-QPS retrieval paths.
//
// NOTE: Cannot use t.Parallel() — testing.AllocsPerRun panics inside parallel tests.
func TestBenchmarkAllocs(t *testing.T) {
	t.Run("DotFromBlob_512", func(t *testing.T) {
		query := benchNormalized(512)
		blob := EncodeFloat32Vec(benchNormalized(512))
		allocs := testing.AllocsPerRun(100, func() {
			DotFromBlob(query, blob)
		})
		if allocs > 0 {
			t.Errorf("DotFromBlob_512 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("Cosine32_512", func(t *testing.T) {
		a := benchNormalized(512)
		b := benchNormalized(512)
		allocs := testing.AllocsPerRun(100, func() {
			Cosine32(a, b)
		})
		if allocs > 0 {
			t.Errorf("Cosine32_512 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("Normalize32_512", func(t *testing.T) {
		allocs := testing.AllocsPerRun(100, func() {
			v := allocBenchVec(512)
			Normalize32(v)
		})
		if allocs > 0 {
			t.Errorf("Normalize32_512 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("HammingDistance_768", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(42, 0))
		a := make(BinaryVector, 12)
		b := make(BinaryVector, 12)
		for i := range a {
			a[i] = rng.Uint64()
			b[i] = rng.Uint64()
		}
		allocs := testing.AllocsPerRun(100, func() {
			HammingDistance(a, b)
		})
		if allocs > 0 {
			t.Errorf("HammingDistance_768 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("QuantizeInto_768", func(t *testing.T) {
		vec := allocBenchVec(768)
		thresholds := make([]float32, 768)
		dst := make(BinaryVector, BinaryWords(768))
		allocs := testing.AllocsPerRun(100, func() {
			if err := QuantizeInto(dst, vec, thresholds); err != nil {
				t.Fatalf("QuantizeInto_768 returned error: %v", err)
			}
		})
		if allocs > 0 {
			t.Errorf("QuantizeInto_768 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("NormSquared32_768", func(t *testing.T) {
		vec := allocBenchVec(768)
		allocs := testing.AllocsPerRun(100, func() {
			_ = NormSquared32(vec)
		})
		if allocs > 0 {
			t.Errorf("NormSquared32_768 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("IsUnit32_768", func(t *testing.T) {
		vec := allocBenchVec(768)
		normalized := make([]float32, len(vec))
		copy(normalized, vec)
		_ = Normalize32(normalized)
		allocs := testing.AllocsPerRun(100, func() {
			_ = IsUnit32(normalized, 1e-6)
		})
		if allocs > 0 {
			t.Errorf("IsUnit32_768 allocated %.1f times, want 0", allocs)
		}
	})

	t.Run("CosineDistance32_512", func(t *testing.T) {
		a := benchNormalized(512)
		b := benchNormalized(512)
		allocs := testing.AllocsPerRun(100, func() {
			_ = CosineDistance32(a, b)
		})
		if allocs > 0 {
			t.Errorf("CosineDistance32_512 allocated %.1f times, want 0", allocs)
		}
	})
}
