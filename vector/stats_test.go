package vectors

import "testing"

func TestMeanStddev_Empty(t *testing.T) {
	t.Parallel()

	mean, stddev := MeanStddev(nil)
	if mean != 0 {
		t.Fatalf("TestMeanStddev_Empty: got mean %v, want %v", mean, 0)
	}
	if stddev != 0 {
		t.Fatalf("TestMeanStddev_Empty: got stddev %v, want %v", stddev, 0)
	}
}

func TestMeanStddev_Single(t *testing.T) {
	t.Parallel()

	mean, stddev := MeanStddev([]float64{5})
	if !approxEq64(mean, 5, 1e-9) {
		t.Fatalf("TestMeanStddev_Single: got mean %v, want %v", mean, 5)
	}
	if stddev != 0 {
		t.Fatalf("TestMeanStddev_Single: got stddev %v, want %v", stddev, 0)
	}
}

func TestMeanStddev_Population(t *testing.T) {
	t.Parallel()

	mean, stddev := MeanStddev([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if !approxEq64(mean, 5.0, 1e-9) {
		t.Fatalf("TestMeanStddev_Population: got mean %v, want %v", mean, 5.0)
	}
	if !approxEq64(stddev, 2.0, 1e-9) {
		t.Fatalf("TestMeanStddev_Population: got stddev %v, want %v", stddev, 2.0)
	}
}

func TestMeanStddev_TwoValues(t *testing.T) {
	t.Parallel()

	mean, stddev := MeanStddev([]float64{0, 10})
	if !approxEq64(mean, 5.0, 1e-9) {
		t.Fatalf("TestMeanStddev_TwoValues: got mean %v, want %v", mean, 5.0)
	}
	if !approxEq64(stddev, 5.0, 1e-9) {
		t.Fatalf("TestMeanStddev_TwoValues: got stddev %v, want %v", stddev, 5.0)
	}
}

func TestMinMaxNormalize(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		in   []float64
		want []float64
	}{
		{"normal_spread", []float64{0, 5, 10}, []float64{0, 0.5, 1}},
		{"all_identical", []float64{3, 3, 3}, []float64{0.5, 0.5, 0.5}},
		{"single", []float64{7}, []float64{0.5}},
		{"empty", []float64{}, []float64{}},
		{"two_values", []float64{2, 4}, []float64{0, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := MinMaxNormalize(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("MinMaxNormalize(%v) len = %d, want %d", tt.in, len(got), len(tt.want))
			}
			for i := range got {
				if !approxEq64(got[i], tt.want[i], 1e-9) {
					t.Errorf("MinMaxNormalize(%v)[%d] = %v, want %v", tt.in, i, got[i], tt.want[i])
				}
			}
		})
	}
}
