package vectors

import (
	"fmt"
	"math"
)

// MeanPool32 averages a set of float32 vectors componentwise and L2-normalizes the
// result, so the output is a unit vector suitable for cosine-via-dot scoring.
// Members whose length differs from the first vector's length are skipped.
// Special cases: an empty input returns nil; a single-vector input is returned
// unchanged (not normalized).
func MeanPool32(vecs [][]float32) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	if len(vecs) == 1 {
		return vecs[0]
	}

	dims := len(vecs[0])
	pooled := make([]float32, dims)
	count := 0
	for _, v := range vecs {
		if len(v) != dims {
			continue
		}
		for i := 0; i < dims; i++ {
			pooled[i] += v[i]
		}
		count++
	}
	if count == 0 {
		return pooled
	}

	inv := float32(1) / float32(count)
	for i := range pooled {
		pooled[i] *= inv
	}
	Normalize32(pooled)
	return pooled
}

// WeightedMeanPool32 computes a weighted mean of vectors, then L2-normalizes the result.
// It rejects weight mismatches, ragged vectors, and invalid weights.
func WeightedMeanPool32(vecs [][]float32, weights []float64) ([]float32, error) {
	if len(vecs) == 0 {
		return nil, nil
	}
	if len(vecs) != len(weights) {
		return nil, fmt.Errorf("weight count mismatch: len(vecs)=%d len(weights)=%d", len(vecs), len(weights))
	}

	dims := len(vecs[0])
	for i, v := range vecs {
		if len(v) != dims {
			return nil, fmt.Errorf("dimension mismatch at index %d: got %d want %d", i, len(v), dims)
		}
	}

	sum := make([]float64, dims)
	var totalWeight float64
	for i, v := range vecs {
		w := weights[i]
		if w < 0 || math.IsNaN(w) || math.IsInf(w, 0) {
			return nil, fmt.Errorf("invalid weight at index %d: %f", i, w)
		}
		if w == 0 {
			continue
		}
		totalWeight += w
		for d := range v {
			sum[d] += float64(v[d]) * w
		}
	}

	if totalWeight == 0 {
		return make([]float32, dims), nil
	}

	inv := 1 / totalWeight
	out := make([]float32, dims)
	for i := range out {
		out[i] = float32(sum[i] * inv)
	}
	Normalize32(out)
	return out, nil
}
