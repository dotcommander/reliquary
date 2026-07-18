package vectors

import (
	"math"
	"testing"
)

func TestEuclidean64_KnownValue(t *testing.T) {
	t.Parallel()
	// distance between (0,0) and (3,4) is 5
	a := []float64{0, 0}
	b := []float64{3, 4}
	got := Euclidean64(a, b)
	if !approxEq64(got, 5.0, 1e-9) {
		t.Fatalf("Euclidean64 known value: got %f, want 5.0", got)
	}
}

func TestEuclidean64_IdenticalVectors(t *testing.T) {
	t.Parallel()
	v := []float64{1, 2, 3}
	got := Euclidean64(v, v)
	if !approxEq64(got, 0.0, 1e-9) {
		t.Fatalf("Euclidean64 identical: got %f, want 0.0", got)
	}
}

func TestEuclidean64_LengthMismatch(t *testing.T) {
	t.Parallel()
	a := []float64{1, 2}
	b := []float64{1, 2, 3}
	got := Euclidean64(a, b)
	if !math.IsInf(got, 1) {
		t.Fatalf("Euclidean64 mismatch: got %f, want +Inf", got)
	}
}

func TestEuclidean64_EmptyInput(t *testing.T) {
	t.Parallel()
	got := Euclidean64([]float64{}, []float64{})
	if !math.IsInf(got, 1) {
		t.Fatalf("Euclidean64 empty: got %f, want +Inf", got)
	}
}

func TestComputeCentroid64_KnownValue(t *testing.T) {
	t.Parallel()
	points := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	got := ComputeCentroid64(points)
	want := []float64{3, 4}
	if len(got) != len(want) {
		t.Fatalf("ComputeCentroid64: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEq64(got[i], want[i], 1e-9) {
			t.Fatalf("ComputeCentroid64: got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestComputeCentroid64_SinglePoint(t *testing.T) {
	t.Parallel()
	points := [][]float64{{7, 8, 9}}
	got := ComputeCentroid64(points)
	want := []float64{7, 8, 9}
	if len(got) != len(want) {
		t.Fatalf("ComputeCentroid64 single: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEq64(got[i], want[i], 1e-9) {
			t.Fatalf("ComputeCentroid64 single: got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestComputeCentroid64_EmptyInput(t *testing.T) {
	t.Parallel()
	got := ComputeCentroid64(nil)
	if got != nil {
		t.Fatalf("ComputeCentroid64 empty: got %v, want nil", got)
	}
}

func TestComputeCentroid64_RaggedInput(t *testing.T) {
	t.Parallel()
	got := ComputeCentroid64([][]float64{{1, 2, 3}, {1, 2}})
	if got != nil {
		t.Fatalf("ComputeCentroid64 ragged: got %v, want nil", got)
	}
}

func TestDot64_KnownValue(t *testing.T) {
	t.Parallel()
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	got := Dot64(a, b)
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if !approxEq64(got, 32.0, 1e-9) {
		t.Fatalf("Dot64 known value: got %f, want 32.0", got)
	}
}

func TestDot64_LengthMismatch(t *testing.T) {
	t.Parallel()
	a := []float64{1, 2}
	b := []float64{1, 2, 3}
	got := Dot64(a, b)
	if got != 0 {
		t.Fatalf("Dot64 mismatch: got %f, want 0", got)
	}
}

func TestDot64_OrthogonalVectors(t *testing.T) {
	t.Parallel()
	a := []float64{1, 0}
	b := []float64{0, 1}
	got := Dot64(a, b)
	if !approxEq64(got, 0.0, 1e-9) {
		t.Fatalf("Dot64 orthogonal: got %f, want 0.0", got)
	}
}
