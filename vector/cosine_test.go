package vectors

import (
	"math"
	"testing"
)

func approxEq32(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func approxEq64(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestCosine32_IdenticalVectors(t *testing.T) {
	t.Parallel()
	v := []float32{0.1, 0.2, 0.3, 0.4}
	got := Cosine32(v, v)
	if !approxEq32(got, 1.0, 1e-6) {
		t.Errorf("Cosine32 identical vectors: got %f, want 1.0", got)
	}
}

func TestCosine32_OrthogonalVectors(t *testing.T) {
	t.Parallel()
	a := []float32{1, 0}
	b := []float32{0, 1}
	got := Cosine32(a, b)
	if !approxEq32(got, 0.0, 1e-6) {
		t.Errorf("Cosine32 orthogonal vectors: got %f, want 0.0", got)
	}
}

func TestCosine32_OppositeVectors(t *testing.T) {
	t.Parallel()
	a := []float32{0.5, 0.5}
	b := []float32{-0.5, -0.5}
	got := Cosine32(a, b)
	if !approxEq32(got, -1.0, 1e-6) {
		t.Errorf("Cosine32 opposite vectors: got %f, want -1.0", got)
	}
}

func TestCosine32_ZeroVector(t *testing.T) {
	t.Parallel()
	a := []float32{1, 2, 3}
	b := []float32{0, 0, 0}
	got := Cosine32(a, b)
	if got != 0 {
		t.Errorf("Cosine32 zero vector: got %f, want 0.0", got)
	}
}

func TestCosine32_DifferentLengths(t *testing.T) {
	t.Parallel()
	a := []float32{1, 2}
	b := []float32{1, 2, 3}
	got := Cosine32(a, b)
	if got != 0 {
		t.Errorf("Cosine32 different lengths: got %f, want 0.0", got)
	}
}

func TestCosine64_DifferentLengths(t *testing.T) {
	t.Parallel()
	a := []float64{1, 2}
	b := []float64{1, 2, 3}
	got := Cosine64(a, b)
	if got != 0 {
		t.Errorf("Cosine64 different lengths: got %f, want 0.0", got)
	}
}

func TestCosine64_IdenticalVectors(t *testing.T) {
	t.Parallel()
	v := []float64{0.1, 0.2, 0.3, 0.4}
	got := Cosine64(v, v)
	if !approxEq64(got, 1.0, 1e-6) {
		t.Errorf("Cosine64 identical vectors: got %f, want 1.0", got)
	}
}

func TestCosine64_ZeroVector(t *testing.T) {
	t.Parallel()
	a := []float64{1, 2, 3}
	b := []float64{0, 0, 0}
	got := Cosine64(a, b)
	if got != 0 {
		t.Errorf("Cosine64 zero vector: got %f, want 0.0", got)
	}
}

func TestCosine64_ExtremeFiniteIdentical(t *testing.T) {
	t.Parallel()

	for _, v := range [][]float64{
		{math.MaxFloat64, math.MaxFloat64},
		{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
	} {
		if got := Cosine64(v, v); got != 1 {
			t.Errorf("Cosine64(%v, same): got %v, want 1", v, got)
		}
		if got := CosineDistance64(v, v); got != 0 {
			t.Errorf("CosineDistance64(%v, same): got %v, want 0", v, got)
		}
	}
}

func TestNormalize32_UnitVector(t *testing.T) {
	t.Parallel()
	v := []float32{3, 4}
	mag := Normalize32(v)
	if !approxEq32(mag, 5.0, 1e-6) {
		t.Errorf("Normalize32 magnitude: got %f, want 5.0", mag)
	}
	// Verify the vector is now unit length.
	norm := float32(0)
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	if !approxEq32(norm, 1.0, 1e-6) {
		t.Errorf("Normalize32 result not unit: got norm %f, want 1.0", norm)
	}
}

func TestNormalize64_ZeroVector(t *testing.T) {
	t.Parallel()
	v := []float64{0, 0, 0}
	mag := Normalize64(v)
	if mag != 0 {
		t.Errorf("Normalize64 zero vector: got %f, want 0.0", mag)
	}
}

func TestNormalize32_AlreadyNormalized(t *testing.T) {
	t.Parallel()
	v := []float32{1.0, 0.0}
	mag := Normalize32(v)
	if !approxEq32(mag, 1.0, 1e-6) {
		t.Errorf("Normalize32 already normalized magnitude: got %f, want 1.0", mag)
	}
	if !approxEq32(v[0], 1.0, 1e-6) {
		t.Errorf("Normalize32 already normalized v[0]: got %f, want 1.0", v[0])
	}
}

func TestNormalize32_ExtremeFinite(t *testing.T) {
	t.Parallel()

	for _, value := range []float32{math.MaxFloat32, math.SmallestNonzeroFloat32} {
		v := []float32{value, value}
		Normalize32(v)
		if got := math.Hypot(float64(v[0]), float64(v[1])); !approxEq64(got, 1, 1e-6) {
			t.Errorf("Normalize32(%v): got %v with norm %v, want unit vector", value, v, got)
		}
	}
}

func TestNormalize64_ExtremeFinite(t *testing.T) {
	t.Parallel()

	for _, value := range []float64{math.MaxFloat64, math.SmallestNonzeroFloat64} {
		v := []float64{value, value}
		Normalize64(v)
		if got := math.Hypot(v[0], v[1]); !approxEq64(got, 1, 1e-15) {
			t.Errorf("Normalize64(%v): got %v with norm %v, want unit vector", value, v, got)
		}
	}
}

func TestNormalizePreservesMixedNonFiniteBehavior(t *testing.T) {
	t.Parallel()

	for _, values := range [][]float64{
		{math.NaN(), math.Inf(1)},
		{math.Inf(1), math.NaN()},
	} {
		v64 := append([]float64(nil), values...)
		if magnitude := Normalize64(v64); !math.IsNaN(magnitude) {
			t.Errorf("Normalize64(%v) magnitude = %v, want NaN", values, magnitude)
		}

		v32 := []float32{float32(values[0]), float32(values[1])}
		if magnitude := Normalize32(v32); !math.IsNaN(float64(magnitude)) {
			t.Errorf("Normalize32(%v) magnitude = %v, want NaN", values, magnitude)
		}
	}
}

func TestCosineDistance32Identical(t *testing.T) {
	t.Parallel()

	got := CosineDistance32([]float32{1, 0}, []float32{1, 0})
	if got != 0 {
		t.Fatalf("TestCosineDistance32Identical: got %v, want 0.0", got)
	}
}

func TestCosineDistance32Orthogonal(t *testing.T) {
	t.Parallel()

	got := CosineDistance32([]float32{1, 0}, []float32{0, 1})
	if !approxEq32(got, 1.0, 1e-6) {
		t.Fatalf("TestCosineDistance32Orthogonal: got %v, want 1.0", got)
	}
}

func TestCosineDistance32Opposite(t *testing.T) {
	t.Parallel()

	got := CosineDistance32([]float32{1, 0}, []float32{-1, 0})
	if !approxEq32(got, 2.0, 1e-6) {
		t.Fatalf("TestCosineDistance32Opposite: got %v, want 2.0", got)
	}
}

func TestCosineDistance32MismatchOrZero(t *testing.T) {
	t.Parallel()

	if got := CosineDistance32([]float32{1, 0}, []float32{1, 0, 0}); got != 1 {
		t.Fatalf("TestCosineDistance32MismatchOrZero: mismatched lengths got %v, want 1.0", got)
	}
	if got := CosineDistance32(nil, []float32{}); got != 1 {
		t.Fatalf("TestCosineDistance32MismatchOrZero: empty got %v, want 1.0", got)
	}
	if got := CosineDistance32([]float32{1, 0}, []float32{0, 0}); got != 1 {
		t.Fatalf("TestCosineDistance32MismatchOrZero: zero vector got %v, want 1.0", got)
	}
}

func TestCosineDistance64Cases(t *testing.T) {
	t.Parallel()

	if got := CosineDistance64([]float64{1, 0}, []float64{1, 0}); got != 0 {
		t.Fatalf("TestCosineDistance64Cases: identical got %v, want 0.0", got)
	}
	if got := CosineDistance64([]float64{1, 0}, []float64{0, 1}); !approxEq64(got, 1.0, 1e-6) {
		t.Fatalf("TestCosineDistance64Cases: orthogonal got %v, want 1.0", got)
	}
	if got := CosineDistance64([]float64{1, 0}, []float64{-1, 0}); !approxEq64(got, 2.0, 1e-6) {
		t.Fatalf("TestCosineDistance64Cases: opposite got %v, want 2.0", got)
	}
	if got := CosineDistance64([]float64{1, 0}, []float64{1, 0, 0}); got != 1 {
		t.Fatalf("TestCosineDistance64Cases: mismatched lengths got %v, want 1.0", got)
	}
}

func TestCosineDistance32NaNInf(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	if got := CosineDistance32([]float32{nan, 1}, []float32{1, 0}); got != 1 {
		t.Fatalf("TestCosineDistance32NaNInf: NaN input got %v, want 1.0", got)
	}
	if got := CosineDistance32([]float32{inf, 1}, []float32{1, 0}); got != 1 {
		t.Fatalf("TestCosineDistance32NaNInf: Inf input got %v, want 1.0", got)
	}
}

func TestCosineDistance64NaNInf(t *testing.T) {
	t.Parallel()

	if got := CosineDistance64([]float64{math.NaN(), 1}, []float64{1, 0}); got != 1 {
		t.Fatalf("TestCosineDistance64NaNInf: NaN input got %v, want 1.0", got)
	}
	if got := CosineDistance64([]float64{math.Inf(1), 1}, []float64{1, 0}); got != 1 {
		t.Fatalf("TestCosineDistance64NaNInf: Inf input got %v, want 1.0", got)
	}
}

func TestCosineDistance64MismatchEmptyZero(t *testing.T) {
	t.Parallel()

	if got := CosineDistance64(nil, []float64{}); got != 1 {
		t.Fatalf("TestCosineDistance64MismatchEmptyZero: empty got %v, want 1.0", got)
	}
	if got := CosineDistance64([]float64{1, 0}, []float64{0, 0}); got != 1 {
		t.Fatalf("TestCosineDistance64MismatchEmptyZero: zero vector got %v, want 1.0", got)
	}
}

func TestCosineDistanceClamp(t *testing.T) {
	t.Parallel()

	if got := clampCosineSimilarity(1.0000001); got != 1 {
		t.Fatalf("TestCosineDistanceClamp: got %v, want 1", got)
	}
	if got := clampCosineSimilarity(-1.0000001); got != -1 {
		t.Fatalf("TestCosineDistanceClamp: got %v, want -1", got)
	}
	if got := clampCosineSimilarity(0.5); got != 0.5 {
		t.Fatalf("TestCosineDistanceClamp: got %v, want 0.5", got)
	}
	if got := clampCosineSimilarity(math.NaN()); got != -1 {
		t.Fatalf("TestCosineDistanceClamp: got %v, want -1", got)
	}
}
