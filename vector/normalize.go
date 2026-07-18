package vectors

import "math"

// NormalizeTo32 returns a new L2-normalized copy of v without mutating the input.
// If v has zero magnitude, the input slice is returned unchanged (identity
// preserved) rather than a fresh zero slice.
func NormalizeTo32(v []float32) []float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	mag := float32(math.Sqrt(float64(norm)))
	if mag == 0 {
		return v
	}

	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i] / mag
	}
	return out
}

// NormalizeTo64 is the float64 twin of NormalizeTo32. Zero-magnitude input is
// returned unchanged.
func NormalizeTo64(v []float64) []float64 {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	mag := math.Sqrt(norm)
	if mag == 0 {
		return v
	}

	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] / mag
	}
	return out
}

// NormSquared32 returns the squared L2 norm of v.
// Accumulates in float64 to avoid intermediate rounding error.
func NormSquared32(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return sum
}

// NormSquared64 returns the squared L2 norm of v.
func NormSquared64(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return sum
}

// IsUnit32 validates whether v is a unit vector within tolerance.
// Returns false for invalid tolerances, empty vectors, zero vectors, and NaN/Inf inputs.
func IsUnit32(v []float32, tolerance float64) bool {
	if len(v) == 0 || tolerance < 0 || math.IsNaN(tolerance) || math.IsInf(tolerance, 0) {
		return false
	}
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	ns := NormSquared32(v)
	if ns == 0 {
		return false
	}
	return math.Abs(ns-1.0) <= tolerance
}

// IsUnit64 validates whether v is a unit vector within tolerance.
// Returns false for invalid tolerances, empty vectors, zero vectors, and NaN/Inf inputs.
func IsUnit64(v []float64, tolerance float64) bool {
	if len(v) == 0 || tolerance < 0 || math.IsNaN(tolerance) || math.IsInf(tolerance, 0) {
		return false
	}
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	ns := NormSquared64(v)
	if ns == 0 {
		return false
	}
	return math.Abs(ns-1.0) <= tolerance
}
