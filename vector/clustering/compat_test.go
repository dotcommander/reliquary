package clustering

import (
	"math"
	"testing"
)

func TestCompatFunctions(t *testing.T) {
	t.Parallel()

	a := []float64{1, 0}
	b := []float64{0, 1}

	dist := EuclideanDistance(a, b)
	if math.Abs(dist-math.Sqrt(2)) > 1e-6 {
		t.Fatalf("EuclideanDistance = %v, want %v", dist, math.Sqrt(2))
	}

	c := []float64{3, 4}
	norm := NormalizeVector(c)
	if math.Abs(norm[0]-0.6) > 1e-6 || math.Abs(norm[1]-0.8) > 1e-6 {
		t.Fatalf("NormalizeVector = %v, want [0.6, 0.8]", norm)
	}

	emptyNorm := NormalizeVector(nil)
	if emptyNorm != nil {
		t.Fatalf("NormalizeVector(nil) = %v, want nil", emptyNorm)
	}
}
