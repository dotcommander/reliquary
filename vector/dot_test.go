package vectors

import "testing"

func TestDot32_Orthogonal(t *testing.T) {
	t.Parallel()

	a := []float32{1, 0}
	b := []float32{0, 1}
	got := Dot32(a, b)
	if got != 0 {
		t.Fatalf("TestDot32_Orthogonal: got %v, want %v", got, 0)
	}
}

func TestDot32_Parallel(t *testing.T) {
	t.Parallel()

	a := []float32{1, 0}
	b := []float32{1, 0}
	got := Dot32(a, b)
	if !approxEq64(got, 1.0, 1e-6) {
		t.Fatalf("TestDot32_Parallel: got %v, want %v", got, 1.0)
	}
}

func TestDot32_Known(t *testing.T) {
	t.Parallel()

	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	got := Dot32(a, b)
	if !approxEq64(got, 32.0, 1e-6) {
		t.Fatalf("TestDot32_Known: got %v, want %v", got, 32.0)
	}
}

func TestDot32_DifferentLengths(t *testing.T) {
	t.Parallel()

	a := []float32{1, 2}
	b := []float32{1, 2, 3}
	got := Dot32(a, b)
	if got != 0 {
		t.Fatalf("TestDot32_DifferentLengths: got %v, want %v", got, 0)
	}
}

func TestDot32_Empty(t *testing.T) {
	t.Parallel()

	a := []float32{}
	b := []float32{}
	got := Dot32(a, b)
	if got != 0 {
		t.Fatalf("TestDot32_Empty: got %v, want %v", got, 0)
	}
}
