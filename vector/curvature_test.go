package vectors

import (
	"fmt"
	"math"
	"testing"
)

// TestGaussianSmooth verifies basic smoothing properties.
func TestGaussianSmooth(t *testing.T) {
	t.Parallel()

	t.Run("single element unchanged", func(t *testing.T) {
		t.Parallel()
		in := []float32{0.7}
		got := GaussianSmooth(in, 1.0)
		assertFloat32InDelta(t, got[0], 0.7, 0.001)
	})

	t.Run("flat signal stays flat", func(t *testing.T) {
		t.Parallel()
		in := []float32{0.5, 0.5, 0.5, 0.5, 0.5}
		got := GaussianSmooth(in, 1.0)
		for i, v := range got {
			assertFloat32InDeltaf(t, v, 0.5, 0.001, "index %d", i)
		}
	})

	t.Run("output length matches input", func(t *testing.T) {
		t.Parallel()
		in := []float32{0.9, 0.7, 0.5, 0.3, 0.1}
		got := GaussianSmooth(in, 1.0)
		if len(got) != len(in) {
			t.Fatalf("len(GaussianSmooth) = %d, want %d", len(got), len(in))
		}
	})
}

func TestGaussianSmoothInvalidSigmaReturnsCopy(t *testing.T) {
	t.Parallel()

	in := []float32{0.9, 0.7, 0.2}
	for _, sigma := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		sigma := sigma
		t.Run(fmt.Sprintf("sigma_%v", sigma), func(t *testing.T) {
			t.Parallel()
			got := GaussianSmooth(in, sigma)
			if len(got) != len(in) {
				t.Fatalf("len = %d, want %d", len(got), len(in))
			}
			for i := range in {
				if got[i] != in[i] {
					t.Fatalf("got[%d] = %v, want %v", i, got[i], in[i])
				}
			}
			got[0] = 0
			if in[0] == 0 {
				t.Fatal("GaussianSmooth returned input slice, want copy")
			}
		})
	}
}

// TestGradient validates central/forward/backward differences.
func TestGradient(t *testing.T) {
	t.Parallel()

	t.Run("empty", func(t *testing.T) {
		t.Parallel()
		if got := Gradient(nil); len(got) != 0 {
			t.Fatalf("Gradient(nil) = %#v, want empty", got)
		}
	})

	t.Run("single element", func(t *testing.T) {
		t.Parallel()
		got := Gradient([]float32{1.0})
		if len(got) != 1 || got[0] != 0 {
			t.Fatalf("Gradient(single) = %#v, want [0]", got)
		}
	})

	t.Run("two elements", func(t *testing.T) {
		t.Parallel()
		// forward at 0, backward at 1
		got := Gradient([]float32{1.0, 3.0})
		assertFloat32InDelta(t, got[0], 2.0, 0.001) // forward
		assertFloat32InDelta(t, got[1], 2.0, 0.001) // backward
	})

	t.Run("length preserved", func(t *testing.T) {
		t.Parallel()
		in := []float32{0.9, 0.8, 0.7, 0.3, 0.29}
		got := Gradient(in)
		if len(got) != len(in) {
			t.Fatalf("len(Gradient) = %d, want %d", len(got), len(in))
		}
	})
}

func TestFindElbowCurvature(t *testing.T) {
	t.Parallel()

	t.Run("sharp single elbow", func(t *testing.T) {
		t.Parallel()
		scores := []float32{0.9, 0.8, 0.7, 0.3, 0.28, 0.27}
		got := FindElbowCurvature(scores, 3)
		if got != 4 {
			t.Fatalf("FindElbowCurvature = %d, want 4", got)
		}
	})

	t.Run("flat tail", func(t *testing.T) {
		t.Parallel()
		scores := []float32{0.9, 0.85, 0.83, 0.80, 0.1, 0.09}
		got := FindElbowCurvature(scores, 3)
		if got != 5 {
			t.Fatalf("FindElbowCurvature = %d, want 5", got)
		}
	})
}

func assertFloat32InDelta(t *testing.T, got, want, delta float32) {
	t.Helper()
	assertFloat32InDeltaf(t, got, want, delta, "")
}

func assertFloat32InDeltaf(t *testing.T, got, want, delta float32, format string, args ...any) {
	t.Helper()
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	if diff <= delta {
		return
	}
	if format != "" {
		t.Fatalf("%s: got %v, want %v within %v", sprintf(format, args...), got, want, delta)
	}
	t.Fatalf("got %v, want %v within %v", got, want, delta)
}

func sprintf(format string, args ...any) string {
	return fmt.Sprintf(format, args...)
}
