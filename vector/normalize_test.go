package vectors

import (
	"math"
	"testing"
)

func TestNormalizeTo32_UnitOutput(t *testing.T) {
	t.Parallel()

	in := []float32{3, 4}
	got := NormalizeTo32(in)
	expected := []float32{0.6, 0.8}
	if !approxEq32(got[0], expected[0], 1e-6) {
		t.Fatalf("TestNormalizeTo32_UnitOutput: got %v, want %v", got, expected)
	}
	if !approxEq32(got[1], expected[1], 1e-6) {
		t.Fatalf("TestNormalizeTo32_UnitOutput: got %v, want %v", got, expected)
	}
	if in[0] != 3 || in[1] != 4 {
		t.Fatalf("TestNormalizeTo32_UnitOutput: got %v, want %v", in, []float32{3, 4})
	}
}

func TestNormalizeTo32_ZeroPassthrough(t *testing.T) {
	t.Parallel()

	in := []float32{0, 0, 0}
	out := NormalizeTo32(in)
	out[0] = 9
	if in[0] != 9 {
		t.Fatalf("TestNormalizeTo32_ZeroPassthrough: got %v, want backing array shared", in)
	}
}

func TestNormalizeTo64_UnitOutput(t *testing.T) {
	t.Parallel()

	in := []float64{3, 4}
	got := NormalizeTo64(in)
	if !approxEq64(got[0], 0.6, 1e-6) {
		t.Fatalf("TestNormalizeTo64_UnitOutput: got %v, want %v", got, []float64{0.6, 0.8})
	}
	if !approxEq64(got[1], 0.8, 1e-6) {
		t.Fatalf("TestNormalizeTo64_UnitOutput: got %v, want %v", got, []float64{0.6, 0.8})
	}
	if in[0] != 3 || in[1] != 4 {
		t.Fatalf("TestNormalizeTo64_UnitOutput: got %v, want %v", in, []float64{3, 4})
	}
}

func TestNormalizeTo64_ZeroPassthrough(t *testing.T) {
	t.Parallel()

	in := []float64{0, 0}
	out := NormalizeTo64(in)
	out[0] = 9
	if in[0] != 9 {
		t.Fatalf("TestNormalizeTo64_ZeroPassthrough: got %v, want backing array shared", in)
	}
}

func TestNormSquared32(t *testing.T) {
	t.Parallel()

	if got := NormSquared32([]float32{3, 4}); got != 25 {
		t.Fatalf("TestNormSquared32: got %v, want %v", got, 25)
	}
	if got := NormSquared32(nil); got != 0 {
		t.Fatalf("TestNormSquared32: got %v, want %v", got, 0)
	}
}

func TestNormSquared64(t *testing.T) {
	t.Parallel()

	if got := NormSquared64([]float64{1, 2, 2}); got != 9 {
		t.Fatalf("TestNormSquared64: got %v, want %v", got, 9)
	}
}

func TestIsUnit32(t *testing.T) {
	t.Parallel()

	if !IsUnit32([]float32{0.6, 0.8}, 1e-6) {
		t.Fatalf("TestIsUnit32: got false, want true")
	}
	if IsUnit32([]float32{3, 4}, 1e-6) {
		t.Fatalf("TestIsUnit32: got true, want false")
	}
	if IsUnit32([]float32{0, 0}, 1e-6) {
		t.Fatalf("TestIsUnit32: got true, want false")
	}
}

func TestIsUnit64Tolerance(t *testing.T) {
	t.Parallel()

	if !IsUnit64([]float64{1 + 1e-7, 0}, 1e-4) {
		t.Fatalf("TestIsUnit64Tolerance: got false, want true")
	}
	if IsUnit64([]float64{1 + 1e-3, 0}, 1e-6) {
		t.Fatalf("TestIsUnit64Tolerance: got true, want false")
	}
}

func TestIsUnitRejectsInvalidInputs(t *testing.T) {
	t.Parallel()

	if IsUnit32([]float32{1, 0}, -1) {
		t.Fatalf("TestIsUnitRejectsInvalidInputs: negative tolerance should return false")
	}
	if IsUnit32([]float32{float32(math.NaN()), 0}, 1e-6) {
		t.Fatalf("TestIsUnitRejectsInvalidInputs: NaN vector component should return false")
	}
	if IsUnit32([]float32{float32(math.Inf(1)), 0}, 1e-6) {
		t.Fatalf("TestIsUnitRejectsInvalidInputs: Inf vector component should return false")
	}
	if IsUnit32([]float32{1, 0}, math.NaN()) {
		t.Fatalf("TestIsUnitRejectsInvalidInputs: NaN tolerance should return false")
	}
	if IsUnit32([]float32{1, 0}, math.Inf(1)) {
		t.Fatalf("TestIsUnitRejectsInvalidInputs: Inf tolerance should return false")
	}
}
