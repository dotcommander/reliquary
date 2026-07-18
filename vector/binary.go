package vectors

import (
	"fmt"
	"math/bits"
	"slices"
)

// BinaryVector is a packed bit representation of a float32 vector.
// For a 768-dimension embedding, this is 12 uint64s (768/64 = 12).
// For a 1024-dimension embedding, this is 16 uint64s.
type BinaryVector []uint64

// BinaryWords returns the number of uint64 words required to represent dim bits.
func BinaryWords(dim int) int {
	if dim <= 0 {
		return 0
	}
	return (dim + 63) / 64
}

// Quantize converts a float32 embedding to a BinaryVector using per-dimension
// thresholds. For dimension i: if vec[i] > thresholds[i], the bit is 1; else 0.
// Bits are packed into uint64s with dimension 0 at bit 0 of uint64[0].
// Returns an error if len(vec) != len(thresholds) — typically a sign that the
// embedder dimension changed between runs and the binary index is stale.
func Quantize(vec []float32, thresholds []float32) (BinaryVector, error) {
	if len(vec) != len(thresholds) {
		return nil, fmt.Errorf("vectors: quantize dimension mismatch: vec=%d thresholds=%d", len(vec), len(thresholds))
	}

	numWords := BinaryWords(len(vec))
	out := make(BinaryVector, numWords)
	if err := QuantizeInto(out, vec, thresholds); err != nil {
		return nil, err
	}
	return out, nil
}

// QuantizeInto encodes a float32 embedding into an existing BinaryVector buffer.
// Bits are cleared before writing to avoid stale bits from prior calls.
func QuantizeInto(dst BinaryVector, vec, thresholds []float32) error {
	if len(vec) != len(thresholds) {
		return fmt.Errorf("vectors: quantize dimension mismatch: vec=%d thresholds=%d", len(vec), len(thresholds))
	}
	expectedWords := BinaryWords(len(vec))
	if len(dst) != expectedWords {
		return fmt.Errorf("vectors: quantize destination length mismatch: got=%d expected=%d", len(dst), expectedWords)
	}

	for i := range dst {
		dst[i] = 0
	}
	for i, v := range vec {
		if v > thresholds[i] {
			dst[i/64] |= 1 << (i % 64)
		}
	}
	return nil
}

// HammingDistance returns the number of differing bits between two binary vectors.
// Returns 0 if the vectors have different lengths.
func HammingDistance(a, b BinaryVector) int {
	if len(a) != len(b) {
		return 0
	}

	var dist int
	for i := range a {
		dist += bits.OnesCount64(a[i] ^ b[i])
	}
	return dist
}

func vectorDim(vecs [][]float32) (int, error) {
	if len(vecs) == 0 {
		return 0, nil
	}
	dim := len(vecs[0])
	for i := 1; i < len(vecs); i++ {
		if len(vecs[i]) != dim {
			return 0, fmt.Errorf("vectors: compute medians dimension mismatch at index %d: got=%d expected=%d", i, len(vecs[i]), dim)
		}
	}
	return dim, nil
}

// ComputeMediansChecked computes the per-dimension median across a set of float32
// vectors with validation. Returns an error on dimension mismatch.
func ComputeMediansChecked(vecs [][]float32) ([]float32, error) {
	dim, err := vectorDim(vecs)
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, nil
	}

	if len(vecs) == 1 {
		medians := make([]float32, dim)
		copy(medians, vecs[0])
		return medians, nil
	}

	medians := make([]float32, dim)
	buf := make([]float32, 0, len(vecs))
	for d := range dim {
		buf = buf[:0]
		for _, v := range vecs {
			buf = append(buf, v[d])
		}
		slices.Sort(buf)
		medians[d] = buf[len(buf)/2]
	}

	return medians, nil
}

// ComputeMedians returns the per-dimension median across a set of float32 vectors.
// The result is suitable for use as the thresholds argument to Quantize.
// Returns nil if vectors is empty. For a single vector, returns a copy of it.
func ComputeMedians(vecs [][]float32) []float32 {
	medians, err := ComputeMediansChecked(vecs)
	if err != nil {
		return nil
	}
	return medians
}
