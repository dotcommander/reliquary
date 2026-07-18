package vectors

import "math"

// GaussianSmooth applies a 1D Gaussian kernel to scores.
// sigma controls smoothing width; typical value 1.0.
// Short vectors and invalid sigma values return a copy unchanged.
func GaussianSmooth(scores []float32, sigma float64) []float32 {
	n := len(scores)
	if n < 2 || sigma <= 0 || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		out := make([]float32, n)
		copy(out, scores)
		return out
	}

	// Build kernel over [-3σ .. 3σ] rounded to integer radius.
	radius := int(math.Ceil(3 * sigma))
	kernel := make([]float64, 2*radius+1)
	sum := 0.0
	for j := -radius; j <= radius; j++ {
		v := math.Exp(-float64(j*j) / (2 * sigma * sigma))
		kernel[j+radius] = v
		sum += v
	}
	for i := range kernel {
		kernel[i] /= sum
	}

	out := make([]float32, n)
	for i := range scores {
		var acc float64
		for j := -radius; j <= radius; j++ {
			si := i + j
			// Clamp edges instead of padding with zeros.
			if si < 0 {
				si = 0
			} else if si >= n {
				si = n - 1
			}
			acc += float64(scores[si]) * kernel[j+radius]
		}
		out[i] = float32(acc)
	}
	return out
}

// Gradient computes the discrete gradient of a score slice using central
// differences. Forward difference at index 0, backward at index n-1.
func Gradient(scores []float32) []float32 {
	n := len(scores)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	if n == 1 {
		out[0] = 0
		return out
	}
	// Forward difference for first element.
	out[0] = scores[1] - scores[0]
	// Backward difference for last element.
	out[n-1] = scores[n-1] - scores[n-2]
	// Central differences for interior.
	for i := 1; i < n-1; i++ {
		out[i] = (scores[i+1] - scores[i-1]) / 2
	}
	return out
}

// FindElbowCurvature returns the index at which the smoothed curvature peaks.
// The curvature is |d²/di²| of the Gaussian-smoothed score vector.
// minKeep guarantees a minimum index regardless of curvature shape —
// the search for the peak starts at minKeep so we never cut before that point.
// Returns len(scores)-1 when no meaningful peak is found (flat or trivially short).
func FindElbowCurvature(scores []float32, minKeep int) int {
	n := len(scores)
	if n <= minKeep {
		return n - 1
	}

	smoothed := GaussianSmooth(scores, 1.0)
	d1 := Gradient(smoothed)
	d2 := Gradient(d1)

	// Find the index with the largest |d2| at or after minKeep.
	peakIdx := n - 1
	peakVal := float32(0)
	for i := minKeep; i < n; i++ {
		v := d2[i]
		if v < 0 {
			v = -v
		}
		if v > peakVal {
			peakVal = v
			peakIdx = i
		}
	}
	return peakIdx
}
