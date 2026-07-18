package vectors

import "math"

// MeanStddev returns the mean and POPULATION standard deviation (divisor N, not
// N-1) of vals. An empty slice returns (0, 0) rather than NaN. The population
// convention is deliberate so derived thresholds are reproducible across runs.
func MeanStddev(vals []float64) (mean, stddev float64) {
	n := len(vals)
	if n == 0 {
		return 0, 0
	}

	var sum float64
	for _, v := range vals {
		sum += v
	}
	mean = sum / float64(n)

	var variance float64
	for _, v := range vals {
		delta := v - mean
		variance += delta * delta
	}
	stddev = math.Sqrt(variance / float64(n))
	return mean, stddev
}

// MinMaxNormalize rescales vals into [0,1] via (v-min)/(max-min), returning a new
// slice. Degenerate input — fewer than 2 elements, or a spread (max-min) below
// 1e-10 — returns all 0.5: a (max-min) of ~0 would divide to NaN, and 0.5 keeps
// such a signal weight-neutral rather than poisoning a downstream weighted sum.
// An empty input returns an empty (non-nil) slice.
func MinMaxNormalize(vals []float64) []float64 {
	out := make([]float64, len(vals))
	if len(vals) < 2 {
		for i := range out {
			out[i] = 0.5
		}
		return out
	}

	min, max := vals[0], vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	spread := max - min
	if spread < 1e-10 {
		for i := range out {
			out[i] = 0.5
		}
		return out
	}

	for i, v := range vals {
		out[i] = (v - min) / spread
	}
	return out
}
