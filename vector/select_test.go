package vectors

import (
	"math"
	"slices"
	"testing"
)

func TestTopKMaxIndices_Basic(t *testing.T) {
	t.Parallel()

	scores := []float32{0.1, 0.9, 0.5, 0.7, 0.3}
	got := TopKMaxIndices(scores, 2)
	want := []int{1, 3}
	if len(got) != len(want) {
		t.Fatalf("TestTopKMaxIndices_Basic: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("TestTopKMaxIndices_Basic: got %v, want %v", got, want)
		}
	}
}

func TestTopKMinIndices_Basic(t *testing.T) {
	t.Parallel()

	scores := []float32{0.1, 0.9, 0.5, 0.7, 0.3}
	got := TopKMinIndices(scores, 2)
	want := []int{0, 4}
	if len(got) != len(want) {
		t.Fatalf("TestTopKMinIndices_Basic: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("TestTopKMinIndices_Basic: got %v, want %v", got, want)
		}
	}
}

func TestTopK_KGreaterThanN_Clamp(t *testing.T) {
	t.Parallel()

	scores := []float32{0.2, 0.1}
	got := TopKMaxIndices(scores, 5)
	want := []int{0, 1}
	if len(got) != len(want) {
		t.Fatalf("TestTopK_KGreaterThanN_Clamp: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("TestTopK_KGreaterThanN_Clamp: got %v, want %v", got, want)
		}
	}
}

func TestTopK_KZero(t *testing.T) {
	t.Parallel()

	scores := []float32{0.2, 0.1}
	got := TopKMinIndices(scores, 0)
	if got == nil {
		t.Fatal("TestTopK_KZero: got nil, want empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("TestTopK_KZero: len(got) = %d, want 0", len(got))
	}
}

func TestTopK_Ties_StableByIndex(t *testing.T) {
	t.Parallel()

	scores := []float32{0.5, 0.5, 0.5}
	got := TopKMaxIndices(scores, 2)
	want := []int{0, 1}
	if len(got) != len(want) {
		t.Fatalf("TestTopK_Ties_StableByIndex: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("TestTopK_Ties_StableByIndex: got %v, want %v", got, want)
		}
	}
}

func TestTopK_Empty(t *testing.T) {
	t.Parallel()

	got := TopKMaxIndices([]float32{}, 3)
	if got == nil {
		t.Fatal("TestTopK_Empty: got nil, want empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("TestTopK_Empty: len(got) = %d, want 0", len(got))
	}
}

func TestTopKExcludesNaNAndOrdersInfinities(t *testing.T) {
	t.Parallel()
	scores := []float32{float32(math.NaN()), 2, float32(math.Inf(-1)), 2, float32(math.Inf(1)), float32(math.NaN())}
	for _, tc := range []struct {
		name          string
		selectIndices func([]float32, int) []int
		want          []int
	}{
		{"max", TopKMaxIndices, []int{4, 1, 3, 2}},
		{"min", TopKMinIndices, []int{2, 1, 3, 4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for _, k := range []int{1, 2, 4, 6, 10} {
				want := tc.want[:min(k, len(tc.want))]
				if got := tc.selectIndices(scores, k); !slices.Equal(got, want) {
					t.Fatalf("k=%d: got %v, want %v", k, got, want)
				}
			}
			if got := tc.selectIndices([]float32{float32(math.NaN())}, 3); len(got) != 0 {
				t.Fatalf("all NaN scores: got %v, want empty", got)
			}
		})
	}
}
