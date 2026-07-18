package vectors

import "math"

// Clamp01 clamps x into [0,1]. NaN and negative values collapse to 0; values
// above 1 collapse to 1. The NaN/negative check is FIRST and deliberate: a naive
// `x < 0` / `x > 1` pair lets NaN fall through (both comparisons are false),
// returning NaN and poisoning any weighted sum downstream.
func Clamp01(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return 0
	case x > 1:
		return 1
	default:
		return x
	}
}

// CosineToUnit remaps a cosine similarity in [-1,1] onto [0,1] via (x+1)/2.
// A NaN input (e.g. a zero-vector cosine) maps to 0 BEFORE the arithmetic, so it
// cannot propagate through the remap. The result is Clamp01-guarded against
// out-of-range cosine inputs.
func CosineToUnit(score float64) float64 {
	if math.IsNaN(score) {
		return 0
	}
	return Clamp01((score + 1) / 2)
}
