package vectors

import (
	"math"
	"testing"
)

func TestClamp01(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		in   float64
		want float64
	}{
		{"nan", math.NaN(), 0},
		{"neg_inf", math.Inf(-1), 0},
		{"pos_inf", math.Inf(1), 1},
		{"negative", -0.5, 0},
		{"zero", 0, 0},
		{"mid", 0.5, 0.5},
		{"one", 1, 1},
		{"above_one", 1.5, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := Clamp01(tt.in); got != tt.want {
				t.Errorf("Clamp01(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestCosineToUnit(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		in   float64
		want float64
	}{
		{"min", -1, 0},
		{"zero", 0, 0.5},
		{"max", 1, 1},
		{"nan", math.NaN(), 0},
		{"below_range", -2, 0},
		{"above_range", 2, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := CosineToUnit(tt.in); got != tt.want {
				t.Errorf("CosineToUnit(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
