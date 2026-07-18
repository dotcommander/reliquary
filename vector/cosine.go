// Package vectors provides small vector math helpers for embedding and
// retrieval systems.
package vectors

import "math"

// Cosine64 computes cosine similarity between two float64 vectors.
// Returns 0 if either vector has zero magnitude.
func Cosine64(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Normalize64 L2-normalizes a float64 vector in place.
// Returns the original magnitude.
func Normalize64(v []float64) float64 {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return 0
	}
	for i := range v {
		v[i] /= norm
	}
	return norm
}

// Cosine32 computes cosine similarity between two float32 vectors.
// Returns 0 if either vector has zero magnitude.
func Cosine32(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

// Normalize32 L2-normalizes a float32 vector in place.
// Returns the original magnitude.
func Normalize32(v []float32) float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	mag := float32(math.Sqrt(float64(norm)))
	if mag == 0 {
		return 0
	}
	for i := range v {
		v[i] /= mag
	}
	return mag
}

// CosineDistance32 returns L2 distance on cosine space: 1 - cosine(a, b).
// Mismatch, empty, and zero-magnitude comparisons return 1.
func CosineDistance32(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1
	}
	sim := float64(Cosine32(a, b))
	if sim == 0 || math.IsNaN(sim) || math.IsInf(sim, 0) {
		return 1
	}
	return float32(1 - clampCosineSimilarity(sim))
}

// CosineDistance64 returns L2 distance on cosine space: 1 - cosine(a, b).
// Mismatch, empty, and zero-magnitude comparisons return 1.
func CosineDistance64(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 1
	}
	sim := Cosine64(a, b)
	if sim == 0 || math.IsNaN(sim) || math.IsInf(sim, 0) {
		return 1
	}
	return 1 - clampCosineSimilarity(sim)
}

// clampCosineSimilarity bounds cosine to [-1, 1] with NaN collapsing to -1.
func clampCosineSimilarity(cosine float64) float64 {
	switch {
	case math.IsNaN(cosine):
		return -1
	case cosine < -1:
		return -1
	case cosine > 1:
		return 1
	default:
		return cosine
	}
}
