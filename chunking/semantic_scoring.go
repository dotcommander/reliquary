package chunking

import "math"

// meanStddev computes the mean and population standard deviation of a slice.
func meanStddev(vals []float64) (float64, float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	mean := sum / float64(len(vals))

	var variance float64
	for _, v := range vals {
		d := v - mean
		variance += d * d
	}
	variance /= float64(len(vals))
	return mean, math.Sqrt(variance)
}

// isZeroVector reports whether all elements of emb are 0.
func isZeroVector(emb []float32) bool {
	for _, v := range emb {
		if v != 0 {
			return false
		}
	}
	return true
}

// meanSlice returns the arithmetic mean of a float64 slice, or 0 if empty.
func meanSlice(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// smoothSimilarities returns a smoothed copy of sims using a centered moving
// average of the given window size. Edge positions clamp to the nearest
// available neighbors. window < 2 returns sims unchanged.
func smoothSimilarities(sims []float64, window int) []float64 {
	if window < 2 || len(sims) == 0 {
		return sims
	}

	half := window / 2
	out := make([]float64, len(sims))

	for i := range sims {
		start := i - half
		if start < 0 {
			start = 0
		}
		end := i + half + 1
		if end > len(sims) {
			end = len(sims)
		}

		var sum float64
		for j := start; j < end; j++ {
			sum += sims[j]
		}
		out[i] = sum / float64(end-start)
	}
	return out
}

// findBreakpoints returns indices where consecutive similarity drops below
// mean - sensitivity*stddev. Index i means a break AFTER sentence i.
// When coherenceWindow >= 1, a break is accepted only if the mean similarity
// of the coherenceWindow pairs on each side of the candidate both exceed threshold.
func findBreakpoints(sims []float64, sensitivity float64, coherenceWindow int) []int {
	if len(sims) == 0 {
		return nil
	}

	mean, stddev := meanStddev(sims)
	threshold := mean - sensitivity*stddev

	// Clamp threshold to the meaningful cosine-similarity range.
	if threshold < 0.1 {
		threshold = 0.1
	}
	if threshold > 0.95 {
		threshold = 0.95
	}

	var breaks []int
	for i, s := range sims {
		if s >= threshold {
			continue
		}
		if coherenceWindow >= 1 && !isCoherentBreak(sims, i, threshold, coherenceWindow) {
			continue
		}
		breaks = append(breaks, i)
	}
	return breaks
}

// isCoherentBreak returns true if the left and right windows around index i
// both have mean similarity above threshold, indicating i is a genuine break
// between two coherent regions. Edge positions (i==0, i==len(sims)-1) skip
// the missing side.
func isCoherentBreak(sims []float64, i int, threshold float64, window int) bool {
	leftOK := true
	rightOK := true

	// Check left window: sims[i-window : i]
	if i > 0 {
		leftStart := i - window
		if leftStart < 0 {
			leftStart = 0
		}
		if leftStart < i {
			leftMean := meanSlice(sims[leftStart:i])
			leftOK = leftMean > threshold
		}
	}

	// Check right window: sims[i+1 : i+1+window]
	if i < len(sims)-1 {
		rightEnd := i + 1 + window
		if rightEnd > len(sims) {
			rightEnd = len(sims)
		}
		if i+1 < rightEnd {
			rightMean := meanSlice(sims[i+1 : rightEnd])
			rightOK = rightMean > threshold
		}
	}

	return leftOK && rightOK
}
