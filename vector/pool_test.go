package vectors

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestMeanPool32_Empty(t *testing.T) {
	t.Parallel()

	got := MeanPool32(nil)
	if got != nil {
		t.Fatalf("TestMeanPool32_Empty: got %v, want %v", got, nil)
	}
}

func TestMeanPool32_Single(t *testing.T) {
	t.Parallel()

	x := []float32{3, 4}
	got := MeanPool32([][]float32{x})
	if len(got) != len(x) {
		t.Fatalf("TestMeanPool32_Single: len(got) = %d, want %d", len(got), len(x))
	}
	if got[0] != 3 {
		t.Fatalf("TestMeanPool32_Single: got %v, want %v", got, x)
	}
	if got[1] != 4 {
		t.Fatalf("TestMeanPool32_Single: got %v, want %v", got, x)
	}
}

func TestMeanPool32_TwoUnitVectors_OutputIsUnit(t *testing.T) {
	t.Parallel()

	got := MeanPool32([][]float32{{1, 0}, {0, 1}})
	norm := float32(0)
	for _, x := range got {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	if !approxEq32(norm, 1.0, 1e-6) {
		t.Fatalf("TestMeanPool32_TwoUnitVectors_OutputIsUnit: got %v, want %v", norm, 1.0)
	}
}

func TestMeanPool32_SkipsMalformed(t *testing.T) {
	t.Parallel()

	got := MeanPool32([][]float32{{1, 0}, {0, 1}, {9, 9, 9}})
	if len(got) != 2 {
		t.Fatalf("TestMeanPool32_SkipsMalformed: len(got) = %d, want %v", len(got), 2)
	}
	expected := []float32{0.70710677, 0.70710677}
	if !approxEq32(got[0], expected[0], 1e-6) {
		t.Fatalf("TestMeanPool32_SkipsMalformed: got %v, want %v", got, expected)
	}
	if !approxEq32(got[1], expected[1], 1e-6) {
		t.Fatalf("TestMeanPool32_SkipsMalformed: got %v, want %v", got, expected)
	}
}

func TestMeanPool32_Average(t *testing.T) {
	t.Parallel()

	got := MeanPool32([][]float32{{2, 0}, {0, 2}})
	expected := []float32{0.70710677, 0.70710677}
	if !approxEq32(got[0], expected[0], 1e-6) {
		t.Fatalf("TestMeanPool32_Average: got %v, want %v", got, expected)
	}
	if !approxEq32(got[1], expected[1], 1e-6) {
		t.Fatalf("TestMeanPool32_Average: got %v, want %v", got, expected)
	}
}

func TestWeightedMeanPool32_Empty(t *testing.T) {
	t.Parallel()

	got, err := WeightedMeanPool32(nil, nil)
	if err != nil {
		t.Fatalf("TestWeightedMeanPool32_Empty: err = %v, want nil", err)
	}
	if got != nil {
		t.Fatalf("TestWeightedMeanPool32_Empty: got %v, want nil", got)
	}
}

func TestWeightedMeanPool32_WeightCountMismatch(t *testing.T) {
	t.Parallel()

	got, err := WeightedMeanPool32([][]float32{{1, 0}, {0, 1}}, []float64{1})
	if err == nil {
		t.Fatalf("TestWeightedMeanPool32_WeightCountMismatch: got %v, want error", got)
	}
	if !strings.Contains(err.Error(), "weight count mismatch") {
		t.Fatalf("TestWeightedMeanPool32_WeightCountMismatch: error = %q, want contains %q", err.Error(), "weight count mismatch")
	}
}

func TestWeightedMeanPool32_DimensionMismatch(t *testing.T) {
	t.Parallel()

	got, err := WeightedMeanPool32([][]float32{{1, 0}, {0, 1, 0}}, []float64{1, 1})
	if err == nil {
		t.Fatalf("TestWeightedMeanPool32_DimensionMismatch: got %v, want error", got)
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("TestWeightedMeanPool32_DimensionMismatch: error = %q, want contains %q", err.Error(), "dimension mismatch")
	}
}

func TestWeightedMeanPool32_InvalidWeight(t *testing.T) {
	t.Parallel()

	vecs := [][]float32{{1, 0}, {0, 1}}
	for _, weight := range []float64{-1, math.NaN(), math.Inf(1)} {
		weight := weight
		name := fmt.Sprintf("weight=%v", weight)
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			got, err := WeightedMeanPool32(vecs, []float64{weight, 1})
			if err == nil {
				t.Fatalf("TestWeightedMeanPool32_InvalidWeight: got %v, want error", got)
			}
			if !strings.Contains(err.Error(), "invalid weight") {
				t.Fatalf("TestWeightedMeanPool32_InvalidWeight: error = %q, want contains %q", err.Error(), "invalid weight")
			}
		})
	}
}

func TestWeightedMeanPool32_AllZeroWeights(t *testing.T) {
	t.Parallel()

	got, err := WeightedMeanPool32([][]float32{{1, 0}, {0, 1}}, []float64{0, 0})
	if err != nil {
		t.Fatalf("TestWeightedMeanPool32_AllZeroWeights: err = %v, want nil", err)
	}
	if len(got) != 2 {
		t.Fatalf("TestWeightedMeanPool32_AllZeroWeights: len(got) = %d, want %d", len(got), 2)
	}
	if got[0] != 0 || got[1] != 0 {
		t.Fatalf("TestWeightedMeanPool32_AllZeroWeights: got %v, want %v", got, []float32{0, 0})
	}
}

func TestWeightedMeanPool32_NormalizesOutput(t *testing.T) {
	t.Parallel()

	got, err := WeightedMeanPool32([][]float32{{1, 0}, {0, 1}}, []float64{1, 1})
	if err != nil {
		t.Fatalf("TestWeightedMeanPool32_NormalizesOutput: err = %v", err)
	}
	norm := float32(0)
	for _, x := range got {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	if !approxEq32(norm, 1.0, 1e-6) {
		t.Fatalf("TestWeightedMeanPool32_NormalizesOutput: norm = %f, want %f", norm, 1.0)
	}
}

func TestWeightedMeanPool32_WeightedCentroid(t *testing.T) {
	t.Parallel()

	got, err := WeightedMeanPool32([][]float32{{1, 0}, {0, 1}}, []float64{3, 1})
	if err != nil {
		t.Fatalf("TestWeightedMeanPool32_WeightedCentroid: err = %v", err)
	}
	if !(got[0] > got[1]) {
		t.Fatalf("TestWeightedMeanPool32_WeightedCentroid: got %v, want x > y", got)
	}
}

func TestWeightedMeanPool32_DoesNotMutateInputs(t *testing.T) {
	t.Parallel()

	input := [][]float32{{1, 0}, {0, 1}}
	inputCopy := [][]float32{append([]float32(nil), input[0]...), append([]float32(nil), input[1]...)}

	_, err := WeightedMeanPool32(input, []float64{1, 1})
	if err != nil {
		t.Fatalf("TestWeightedMeanPool32_DoesNotMutateInputs: err = %v", err)
	}
	if !approxEq32(input[0][0], inputCopy[0][0], 0) || !approxEq32(input[0][1], inputCopy[0][1], 0) {
		t.Fatalf("TestWeightedMeanPool32_DoesNotMutateInputs: input[0] mutated: got %v, want %v", input[0], inputCopy[0])
	}
	if !approxEq32(input[1][0], inputCopy[1][0], 0) || !approxEq32(input[1][1], inputCopy[1][1], 0) {
		t.Fatalf("TestWeightedMeanPool32_DoesNotMutateInputs: input[1] mutated: got %v, want %v", input[1], inputCopy[1])
	}
}
