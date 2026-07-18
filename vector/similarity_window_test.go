package vectors

import (
	"testing"
)

func TestAverageSimilarity_Edges(t *testing.T) {
	t.Parallel()

	if got := AverageSimilarity(nil); got != 1 {
		t.Errorf("AverageSimilarity(nil): got %f, want 1.0", got)
	}
	if got := AverageSimilarity([][]float32{{1, 0}}); got != 1 {
		t.Errorf("AverageSimilarity single: got %f, want 1.0", got)
	}
}

func TestAverageSimilarity_Identical(t *testing.T) {
	t.Parallel()

	emb := [][]float32{{1, 0}, {2, 0}, {3, 0}}
	got := AverageSimilarity(emb)
	if !approxEq32(got, 1.0, 1e-6) {
		t.Errorf("AverageSimilarity identical-direction: got %f, want 1.0", got)
	}
}

func TestAverageSimilarity_Orthogonal(t *testing.T) {
	t.Parallel()

	emb := [][]float32{{1, 0}, {0, 1}}
	got := AverageSimilarity(emb)
	if !approxEq32(got, 0.0, 1e-6) {
		t.Errorf("AverageSimilarity orthogonal: got %f, want 0.0", got)
	}
}

func TestSlidingWindowSimilarity_Edges(t *testing.T) {
	t.Parallel()

	if got := SlidingWindowSimilarity([][]float32{{1, 0}}, 1); got == nil || len(got) != 0 {
		t.Errorf("SlidingWindowSimilarity single: got %v, want non-nil empty", got)
	}
	if got := SlidingWindowSimilarity([][]float32{{1, 0}, {0, 1}}, 0); got == nil || len(got) != 0 {
		t.Errorf("SlidingWindowSimilarity windowSize<1: got %v, want non-nil empty", got)
	}
}

func TestSlidingWindowSimilarity_Basic(t *testing.T) {
	t.Parallel()

	emb := [][]float32{{1, 0}, {1, 0}, {0, 1}, {0, 1}}
	got := SlidingWindowSimilarity(emb, 1)
	if len(got) != len(emb)-1 {
		t.Fatalf("SlidingWindowSimilarity length: got %d, want %d", len(got), len(emb)-1)
	}
	// Window 1 compares consecutive single vectors.
	if !approxEq32(got[0], 1.0, 1e-6) {
		t.Errorf("SlidingWindowSimilarity got[0]: got %f, want 1.0", got[0])
	}
	if !approxEq32(got[1], 0.0, 1e-6) {
		t.Errorf("SlidingWindowSimilarity got[1]: got %f, want 0.0", got[1])
	}
	if !approxEq32(got[2], 1.0, 1e-6) {
		t.Errorf("SlidingWindowSimilarity got[2]: got %f, want 1.0", got[2])
	}
}

func TestSlidingWindowSimilarity_Clamping(t *testing.T) {
	t.Parallel()

	emb := [][]float32{{1, 0}, {1, 0}, {1, 0}}
	// windowSize larger than len should be clamped and not panic.
	got := SlidingWindowSimilarity(emb, 100)
	if len(got) != len(emb)-1 {
		t.Fatalf("SlidingWindowSimilarity clamped length: got %d, want %d", len(got), len(emb)-1)
	}
	for i, s := range got {
		if !approxEq32(s, 1.0, 1e-6) {
			t.Errorf("SlidingWindowSimilarity clamped got[%d]: got %f, want 1.0", i, s)
		}
	}
}

func TestFindSemanticBoundaries(t *testing.T) {
	t.Parallel()

	// All above threshold -> empty (non-nil).
	allAbove := FindSemanticBoundaries([]float32{0.9, 0.95, 0.99}, 0.5)
	if allAbove == nil || len(allAbove) != 0 {
		t.Errorf("FindSemanticBoundaries all-above: got %v, want non-nil empty", allAbove)
	}

	// All below threshold -> [1..n].
	allBelow := FindSemanticBoundaries([]float32{0.1, 0.2, 0.3}, 0.5)
	want := []int{1, 2, 3}
	if len(allBelow) != len(want) {
		t.Fatalf("FindSemanticBoundaries all-below: got %v, want %v", allBelow, want)
	}
	for i := range want {
		if allBelow[i] != want[i] {
			t.Errorf("FindSemanticBoundaries all-below[%d]: got %d, want %d", i, allBelow[i], want[i])
		}
	}

	// Mixed.
	mixed := FindSemanticBoundaries([]float32{0.9, 0.1, 0.8, 0.2}, 0.5)
	wantMixed := []int{2, 4}
	if len(mixed) != len(wantMixed) {
		t.Fatalf("FindSemanticBoundaries mixed: got %v, want %v", mixed, wantMixed)
	}
	for i := range wantMixed {
		if mixed[i] != wantMixed[i] {
			t.Errorf("FindSemanticBoundaries mixed[%d]: got %d, want %d", i, mixed[i], wantMixed[i])
		}
	}
}

func TestAdaptiveThreshold_Empty(t *testing.T) {
	t.Parallel()

	if got := AdaptiveThreshold(nil); !approxEq32(got, 0.7, 1e-6) {
		t.Errorf("AdaptiveThreshold empty: got %f, want 0.7", got)
	}
}

func TestAdaptiveThreshold_ClampLow(t *testing.T) {
	t.Parallel()

	// Wide spread with low mean -> mean-stdDev clamps to 0.3.
	got := AdaptiveThreshold([]float32{0.0, 1.0, 0.0, 1.0})
	if !approxEq32(got, 0.3, 1e-6) {
		t.Errorf("AdaptiveThreshold clamp-low: got %f, want 0.3", got)
	}
}

func TestAdaptiveThreshold_ClampHigh(t *testing.T) {
	t.Parallel()

	// High mean with small spread -> mean-stdDev exceeds 0.9 -> clamps to 0.9.
	got := AdaptiveThreshold([]float32{0.98, 0.99, 1.0})
	if !approxEq32(got, 0.9, 1e-6) {
		t.Errorf("AdaptiveThreshold clamp-high: got %f, want 0.9", got)
	}
}

func TestSmoothSimilarities_Identity(t *testing.T) {
	t.Parallel()

	in := []float32{0.1, 0.2, 0.3}
	// windowSize < 1 returns input unchanged.
	got := SmoothSimilarities(in, 0)
	if len(got) != len(in) {
		t.Fatalf("SmoothSimilarities windowSize<1 length: got %d, want %d", len(got), len(in))
	}
	for i := range in {
		if got[i] != in[i] {
			t.Errorf("SmoothSimilarities windowSize<1 not identity at %d: got %f, want %f", i, got[i], in[i])
		}
	}

	// Empty input returns input unchanged.
	empty := SmoothSimilarities([]float32{}, 3)
	if len(empty) != 0 {
		t.Errorf("SmoothSimilarities empty: got %v, want empty", empty)
	}
}

func TestSmoothSimilarities_MovingAverage(t *testing.T) {
	t.Parallel()

	in := []float32{1, 2, 3}
	// windowSize 3, halfWindow 1: centered moving average with edge clamping.
	// i=0: avg(1,2)=1.5 ; i=1: avg(1,2,3)=2 ; i=2: avg(2,3)=2.5
	got := SmoothSimilarities(in, 3)
	want := []float32{1.5, 2.0, 2.5}
	if len(got) != len(want) {
		t.Fatalf("SmoothSimilarities length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEq32(got[i], want[i], 1e-6) {
			t.Errorf("SmoothSimilarities[%d]: got %f, want %f", i, got[i], want[i])
		}
	}
}
