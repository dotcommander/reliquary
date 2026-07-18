package vectors

// Dot32 computes the dot product of two float32 vectors, accumulating in float64
// to reduce rounding error on high-dimensional inputs. For L2-normalized vectors
// the result equals cosine similarity.
func Dot32(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}
