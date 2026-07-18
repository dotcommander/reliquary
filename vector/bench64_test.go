package vectors

import (
	"math/rand"
	"testing"
)

// benchVec64 returns a deterministic float64 vector of the given dimension.
func benchVec64(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(i+1) * 0.01
	}
	return v
}

// benchNormalized64 returns a normalized copy of a benchVec64.
func benchNormalized64(n int) []float64 {
	v := benchVec64(n)
	Normalize64(v)
	return v
}

// --- Cosine64 ---

func BenchmarkCosine64_512(b *testing.B) {
	a := benchNormalized64(512)
	c := benchNormalized64(512)
	b.ResetTimer()
	for b.Loop() {
		Cosine64(a, c)
	}
}

func BenchmarkCosine64_768(b *testing.B) {
	a := benchNormalized64(768)
	c := benchNormalized64(768)
	b.ResetTimer()
	for b.Loop() {
		Cosine64(a, c)
	}
}

// --- Normalize64 ---

func BenchmarkNormalize64_512(b *testing.B) {
	for b.Loop() {
		v := benchVec64(512)
		Normalize64(v)
	}
}

func BenchmarkNormalize64_768(b *testing.B) {
	for b.Loop() {
		v := benchVec64(768)
		Normalize64(v)
	}
}

// --- KMeans ---

func BenchmarkKMeans_100x768_k4(b *testing.B) {
	const n, dims, k = 100, 768, 4
	points := make([][]float32, n)
	for i := range points {
		points[i] = benchNormalized(dims)
	}
	rng := rand.New(rand.NewSource(42))
	b.ResetTimer()
	for b.Loop() {
		KMeans(points, k, rng)
	}
}

// --- SilhouetteScore ---

func BenchmarkSilhouetteScore_100x768_k4(b *testing.B) {
	const n, dims, k = 100, 768, 4
	points := make([][]float32, n)
	for i := range points {
		points[i] = benchNormalized(dims)
	}
	rng := rand.New(rand.NewSource(42))
	result := KMeans(points, k, rng)
	b.ResetTimer()
	for b.Loop() {
		SilhouetteScore(points, result.Assignments, result.K)
	}
}
