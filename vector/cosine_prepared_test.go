package vectors

import (
	"math"
	"math/rand"
	"testing"
)

// legacyCosine64 is the pre-Prepare64 implementation, kept verbatim so tests can
// require bit-identical results from the prepared path.
func legacyCosine64(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	scaleA, sumSquaresA := scaledSumSquares64(a)
	scaleB, sumSquaresB := scaledSumSquares64(b)
	if scaleA == 0 || scaleB == 0 {
		return 0
	}
	var dot float64
	for i := range a {
		dot += (a[i] / scaleA) * (b[i] / scaleB)
	}
	return dot / math.Sqrt(sumSquaresA*sumSquaresB)
}

// randomExtremeVector64 mixes extreme magnitudes, zeros, subnormals, infinities,
// and NaNs into deterministic test vectors.
func randomExtremeVector64(r *rand.Rand, dim, iteration int) []float64 {
	v := make([]float64, dim)
	for i := range v {
		switch iteration % 7 {
		case 0:
			v[i] = r.NormFloat64() * math.Pow(10, float64(r.Intn(601)-300))
		case 1:
			v[i] = 0
		case 2:
			v[i] = 5e-324 * float64(r.Intn(3))
		case 3:
			v[i] = math.Inf(r.Intn(3) - 1)
		case 4:
			v[i] = math.NaN()
		default:
			v[i] = r.NormFloat64()
		}
	}
	return v
}

func TestCosine64Prepared_BitIdenticalToLegacy(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(1))
	for iteration := 0; iteration < 700; iteration++ {
		dim := 1 + r.Intn(64)
		a := randomExtremeVector64(r, dim, iteration)
		b := randomExtremeVector64(r, dim, iteration)
		want := legacyCosine64(a, b)
		prepared := Cosine64Prepared(a, Prepare64(a), b, Prepare64(b))
		if math.Float64bits(prepared) != math.Float64bits(want) {
			t.Fatalf("iteration %d dim %d: Cosine64Prepared = %x (%v), want legacy %x (%v)",
				iteration, dim, math.Float64bits(prepared), prepared, math.Float64bits(want), want)
		}
		if direct := Cosine64(a, b); math.Float64bits(direct) != math.Float64bits(want) {
			t.Fatalf("iteration %d dim %d: Cosine64 = %x (%v), want legacy %x (%v)",
				iteration, dim, math.Float64bits(direct), direct, math.Float64bits(want), want)
		}
	}
}

func TestCosine64Prepared_ExtremeMagnitudeCases(t *testing.T) {
	t.Parallel()
	guards := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"zero vs nonzero", []float64{0, 0, 0}, []float64{1, 2, 3}, 0},
		{"both zero", []float64{0, 0}, []float64{0, 0}, 0},
		{"both empty", []float64{}, []float64{}, 0},
		{"mismatched lengths", []float64{1, 2}, []float64{1, 2, 3}, 0},
	}
	for _, tc := range guards {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := Cosine64Prepared(tc.a, Prepare64(tc.a), tc.b, Prepare64(tc.b)); got != tc.want {
				t.Fatalf("Cosine64Prepared(%s) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}

	extremes := [][2][]float64{
		{{1e300, 1e-300, 5e-324}, {1e-300, 1e300, 1e-310}},
		{{1e-310, 2e-310, 3e-310}, {3e-310, 2e-310, 1e-310}},
		{{1e300, -1e300, 1e300}, {-1e300, 1e300, 1e300}},
	}
	for i, pair := range extremes {
		a, b := pair[0], pair[1]
		want := legacyCosine64(a, b)
		got := Cosine64Prepared(a, Prepare64(a), b, Prepare64(b))
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("extreme case %d: Cosine64Prepared = %x (%v), want legacy %x (%v)",
				i, math.Float64bits(got), got, math.Float64bits(want), want)
		}
	}
}

func TestCosine64Prepared_IdenticalVectorsNearOne(t *testing.T) {
	t.Parallel()
	v := []float64{1e-200, 1, 1e200, 0.5}
	got := Cosine64Prepared(v, Prepare64(v), v, Prepare64(v))
	if got < 1-1e-12 || got > 1+1e-12 {
		t.Fatalf("Cosine64Prepared(v, v) = %v, want ~1", got)
	}
}

func TestPrepare64_ZeroValueSemantics(t *testing.T) {
	t.Parallel()
	if got := Prepare64(nil); got != (Prepared64{}) {
		t.Fatalf("Prepare64(nil) = %+v, want zero value", got)
	}
	if got := Prepare64([]float64{0, 0, 0}); got != (Prepared64{}) {
		t.Fatalf("Prepare64(zeros) = %+v, want zero value", got)
	}
	if got := Prepare64([]float64{3, 4}); got.Scale != 4 || got.SumSquares != 1+9.0/16 {
		t.Fatalf("Prepare64({3,4}) = %+v, want scale 4 and sumSquares 1+9/16", got)
	}
}

func BenchmarkCosine64Prepared_768(b *testing.B) {
	a := benchNormalized64(768)
	c := benchNormalized64(768)
	pa, pc := Prepare64(a), Prepare64(c)
	b.ResetTimer()
	for b.Loop() {
		Cosine64Prepared(a, pa, c, pc)
	}
}

func BenchmarkLegacyCosine64_768(b *testing.B) {
	a := benchNormalized64(768)
	c := benchNormalized64(768)
	b.ResetTimer()
	for b.Loop() {
		legacyCosine64(a, c)
	}
}
