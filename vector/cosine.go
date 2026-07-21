// Package vectors provides small vector math helpers for embedding and
// retrieval systems.
package vectors

import "math"

// Cosine64 computes cosine similarity between two float64 vectors using scaled
// accumulation to avoid overflow and underflow for finite inputs. It returns 0
// if either vector has zero magnitude.
func Cosine64(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	scaleA, sumSquaresA := scaledSumSquares64(a)
	scaleB, sumSquaresB := scaledSumSquares64(b)
	if scaleA == 0 || scaleB == 0 {
		return 0
	}

	var dot float64
	for i := range a {
		dot += (a[i] / scaleA) * (b[i] / scaleB)
	}

	return dot / math.Sqrt(sumSquaresA*sumSquaresB)
}

// Normalize64 L2-normalizes a float64 vector in place using scaled accumulation
// to avoid overflow and underflow for finite inputs. It returns the original
// magnitude, which may be +Inf when the mathematical magnitude exceeds the
// float64 range.
func Normalize64(v []float64) float64 {
	scale, sumSquares := scaledSumSquares64(v)
	if scale == 0 {
		return 0
	}
	rootSumSquares := math.Sqrt(sumSquares)
	for i := range v {
		v[i] = (v[i] / scale) / rootSumSquares
	}
	return scale * rootSumSquares
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

// Normalize32 L2-normalizes a float32 vector in place using scaled accumulation
// to avoid overflow and underflow for finite inputs. It returns the original
// magnitude, which may be +Inf when the mathematical magnitude exceeds the
// float32 range.
func Normalize32(v []float32) float32 {
	scale, sumSquares := scaledSumSquares32(v)
	if scale == 0 {
		return 0
	}
	rootSumSquares := math.Sqrt(sumSquares)
	for i := range v {
		v[i] = float32((float64(v[i]) / scale) / rootSumSquares)
	}
	return float32(scale * rootSumSquares)
}

// addScaledSquare updates a scaled sum of squares so finite values are never
// squared before they have been brought into a safe range.
func addScaledSquare(scale, sumSquares, value float64) (float64, float64) {
	if math.IsNaN(scale) || math.IsNaN(sumSquares) {
		return math.NaN(), math.NaN()
	}
	abs := math.Abs(value)
	switch {
	case math.IsNaN(abs):
		return math.NaN(), math.NaN()
	case math.IsInf(abs, 1):
		return math.Inf(1), 1
	case abs == 0:
		return scale, sumSquares
	case scale < abs:
		ratio := scale / abs
		return abs, 1 + sumSquares*ratio*ratio
	default:
		ratio := abs / scale
		return scale, sumSquares + ratio*ratio
	}
}

func scaledSumSquares32(v []float32) (scale, sumSquares float64) {
	for _, value := range v {
		scale, sumSquares = addScaledSquare(scale, sumSquares, float64(value))
	}
	return scale, sumSquares
}

func scaledSumSquares64(v []float64) (scale, sumSquares float64) {
	for _, value := range v {
		scale, sumSquares = addScaledSquare(scale, sumSquares, value)
	}
	return scale, sumSquares
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
