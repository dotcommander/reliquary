package vectors

import (
	"math/rand/v2"
	"strings"
	"testing"
)

func bit(bv BinaryVector, i int) bool {
	return (bv[i/64]>>uint(i%64))&1 == 1
}

func TestQuantize(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		vec        []float32
		thresholds []float32
		wantBits   []bool
	}{
		{
			name:       "zero thresholds",
			vec:        []float32{1, -1, 0.5, -0.5},
			thresholds: []float32{0, 0, 0, 0},
			wantBits:   []bool{true, false, true, false},
		},
		{
			name:       "exact threshold is unset",
			vec:        []float32{1, 0.5, 0.5},
			thresholds: []float32{1, 0.5, 0},
			wantBits:   []bool{false, false, true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := Quantize(tt.vec, tt.thresholds)
			if err != nil {
				t.Fatalf("Quantize returned error: %v", err)
			}
			if len(got) != 1 {
				t.Fatalf("len(Quantize(...)) = %d, want 1", len(got))
			}
			for i, want := range tt.wantBits {
				if bit(got, i) != want {
					t.Fatalf("bit %d = %v, want %v", i, bit(got, i), want)
				}
			}
		})
	}
}

func TestQuantizeSpansWords(t *testing.T) {
	t.Parallel()

	vec := make([]float32, 65)
	thresholds := make([]float32, 65)
	for i := range vec {
		vec[i] = 1
	}
	vec[64] = -1

	got, err := Quantize(vec, thresholds)
	if err != nil {
		t.Fatalf("Quantize returned error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(Quantize(...)) = %d, want 2", len(got))
	}
	for i := range 64 {
		if !bit(got, i) {
			t.Fatalf("bit %d = false, want true", i)
		}
	}
	if bit(got, 64) {
		t.Fatal("bit 64 = true, want false")
	}
}

func TestQuantizeDimensionMismatch(t *testing.T) {
	t.Parallel()

	got, err := Quantize([]float32{1, 2, 3}, []float32{0, 0})
	if err == nil {
		t.Fatal("Quantize returned nil error, want dimension mismatch")
	}
	if got != nil {
		t.Fatalf("Quantize returned %v, want nil vector on error", got)
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("Quantize error = %q, want dimension mismatch", err)
	}
}

func TestBinaryWords(t *testing.T) {
	t.Parallel()

	if got := BinaryWords(-1); got != 0 {
		t.Fatalf("TestBinaryWords: got %v, want %v", got, 0)
	}
	if got := BinaryWords(0); got != 0 {
		t.Fatalf("TestBinaryWords: got %v, want %v", got, 0)
	}
	if got := BinaryWords(1); got != 1 {
		t.Fatalf("TestBinaryWords: got %v, want %v", got, 1)
	}
	if got := BinaryWords(64); got != 1 {
		t.Fatalf("TestBinaryWords: got %v, want %v", got, 1)
	}
	if got := BinaryWords(65); got != 2 {
		t.Fatalf("TestBinaryWords: got %v, want %v", got, 2)
	}
	if got := BinaryWords(768); got != 12 {
		t.Fatalf("TestBinaryWords: got %v, want %v", got, 12)
	}
}

func TestQuantizeIntoMatchesQuantize(t *testing.T) {
	t.Parallel()

	vec := []float32{1, -1, 0.5, -0.5}
	thresholds := []float32{0, 0, 0, 0}
	want, err := Quantize(vec, thresholds)
	if err != nil {
		t.Fatalf("TestQuantizeIntoMatchesQuantize: Quantize returned err=%v", err)
	}

	dst := make(BinaryVector, BinaryWords(len(vec)))
	if err := QuantizeInto(dst, vec, thresholds); err != nil {
		t.Fatalf("TestQuantizeIntoMatchesQuantize: QuantizeInto returned err=%v", err)
	}

	if len(dst) != len(want) {
		t.Fatalf("TestQuantizeIntoMatchesQuantize: len(dst) = %d, want %d", len(dst), len(want))
	}
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("TestQuantizeIntoMatchesQuantize: dst[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
}

func TestQuantizeIntoClearsDestination(t *testing.T) {
	t.Parallel()

	vec := make([]float32, 65)
	thresholds := make([]float32, 65)
	vec[64] = 1
	dst := BinaryVector{^uint64(0), ^uint64(0)}

	if err := QuantizeInto(dst, vec, thresholds); err != nil {
		t.Fatalf("TestQuantizeIntoClearsDestination: QuantizeInto returned err=%v", err)
	}
	for i := range 65 {
		expected := i == 64
		if bit(dst, i) != expected {
			t.Fatalf("TestQuantizeIntoClearsDestination: bit %d = %v, want %v", i, bit(dst, i), expected)
		}
	}
}

func TestQuantizeIntoDimensionMismatch(t *testing.T) {
	t.Parallel()

	errVec := []float32{1, 2}
	thresholds := []float32{0}
	dst := make(BinaryVector, BinaryWords(len(errVec)))
	err := QuantizeInto(dst, errVec, thresholds)
	if err == nil {
		t.Fatal("TestQuantizeIntoDimensionMismatch: got nil err, want dimension mismatch")
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("TestQuantizeIntoDimensionMismatch: got err %q, want dimension mismatch", err)
	}
}

func TestQuantizeIntoDestinationLengthMismatch(t *testing.T) {
	t.Parallel()

	vec := make([]float32, 10)
	thresholds := make([]float32, 10)
	dst := make(BinaryVector, BinaryWords(len(vec))+1)
	err := QuantizeInto(dst, vec, thresholds)
	if err == nil {
		t.Fatal("TestQuantizeIntoDestinationLengthMismatch: got nil err, want destination length mismatch")
	}
	if !strings.Contains(err.Error(), "destination length mismatch") {
		t.Fatalf("TestQuantizeIntoDestinationLengthMismatch: got err %q, want destination length mismatch", err)
	}
}

func TestHammingDistance(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		a    BinaryVector
		b    BinaryVector
		want int
	}{
		{"identical", BinaryVector{0xAA}, BinaryVector{0xAA}, 0},
		{"complement byte", BinaryVector{0xAA}, BinaryVector{0x55}, 8},
		{"multiword", BinaryVector{^uint64(0), 0}, BinaryVector{^uint64(0), ^uint64(0)}, 64},
		{"different lengths", BinaryVector{0xFF}, BinaryVector{0xFF, 0xFF}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := HammingDistance(tt.a, tt.b); got != tt.want {
				t.Fatalf("HammingDistance(...) = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestComputeMedians(t *testing.T) {
	t.Parallel()

	if got := ComputeMedians(nil); got != nil {
		t.Fatalf("ComputeMedians(nil) = %v, want nil", got)
	}

	single := []float32{1, 2, 3}
	got := ComputeMedians([][]float32{single})
	want := []float32{1, 2, 3}
	assertFloat32SliceEqual(t, got, want)
	if &got[0] == &single[0] {
		t.Fatal("single vector result aliases input")
	}

	got = ComputeMedians([][]float32{
		{1, 5, 3, 7},
		{2, 4, 6, 1},
		{3, 3, 9, 4},
	})
	assertFloat32SliceEqual(t, got, []float32{2, 4, 6, 4})

	got = ComputeMedians([][]float32{
		{1, 10, 5},
		{3, 2, 8},
		{5, 6, 1},
		{7, 4, 3},
	})
	assertFloat32SliceEqual(t, got, []float32{5, 6, 5})
}

func TestComputeMediansCheckedEmpty(t *testing.T) {
	t.Parallel()

	got, err := ComputeMediansChecked(nil)
	if err != nil {
		t.Fatalf("TestComputeMediansCheckedEmpty: got err=%v, want nil", err)
	}
	if got != nil {
		t.Fatalf("TestComputeMediansCheckedEmpty: got %v, want nil", got)
	}
}

func TestComputeMediansCheckedSingleCopiesInput(t *testing.T) {
	t.Parallel()

	single := []float32{1, 2, 3}
	got, err := ComputeMediansChecked([][]float32{single})
	if err != nil {
		t.Fatalf("TestComputeMediansCheckedSingleCopiesInput: got err=%v, want nil", err)
	}
	want := []float32{1, 2, 3}
	assertFloat32SliceEqual(t, got, want)
	if &got[0] == &single[0] {
		t.Fatal("TestComputeMediansCheckedSingleCopiesInput: result aliases input")
	}
}

func TestComputeMediansCheckedValid(t *testing.T) {
	t.Parallel()

	base := [][]float32{
		{1, 5, 3, 7},
		{2, 4, 6, 1},
		{3, 3, 9, 4},
	}
	got, err := ComputeMediansChecked(base)
	if err != nil {
		t.Fatalf("TestComputeMediansCheckedValid: got err=%v, want nil", err)
	}
	assertFloat32SliceEqual(t, got, []float32{2, 4, 6, 4})
}

func TestComputeMediansCheckedDimensionMismatch(t *testing.T) {
	t.Parallel()

	got, err := ComputeMediansChecked([][]float32{{1, 2, 3}, {4, 5}})
	if err == nil {
		t.Fatal("TestComputeMediansCheckedDimensionMismatch: got nil err, want mismatch error")
	}
	if got != nil {
		t.Fatalf("TestComputeMediansCheckedDimensionMismatch: got %v, want nil", got)
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("TestComputeMediansCheckedDimensionMismatch: got error %q, want dimension mismatch", err)
	}
}

func TestComputeMediansReturnsNilOnDimensionMismatch(t *testing.T) {
	t.Parallel()

	got := ComputeMedians([][]float32{{1, 2, 3}, {4, 5}})
	if got != nil {
		t.Fatalf("TestComputeMediansReturnsNilOnDimensionMismatch: got %v, want nil", got)
	}
}

func TestQuantizeRoundtripSimilarVectorsHaveSmallHammingDistance(t *testing.T) {
	t.Parallel()

	const dimCount = 768
	rng := rand.New(rand.NewPCG(42, 0))

	base := make([]float32, dimCount)
	for i := range base {
		base[i] = rng.Float32()*2 - 1
	}
	Normalize32(base)

	near := append([]float32(nil), base...)
	for i := range near {
		near[i] += (rng.Float32() - 0.5) * 0.01
	}
	Normalize32(near)

	far := make([]float32, dimCount)
	for i := range far {
		far[i] = -base[i]
	}

	zeros := make([]float32, dimCount)
	qBase, err := Quantize(base, zeros)
	if err != nil {
		t.Fatalf("Quantize base: %v", err)
	}
	qNear, err := Quantize(near, zeros)
	if err != nil {
		t.Fatalf("Quantize near: %v", err)
	}
	qFar, err := Quantize(far, zeros)
	if err != nil {
		t.Fatalf("Quantize far: %v", err)
	}

	if got := HammingDistance(qBase, qNear); got >= dimCount/10 {
		t.Fatalf("near hamming distance = %d, want < %d", got, dimCount/10)
	}
	if got := HammingDistance(qBase, qFar); got <= dimCount*9/10 {
		t.Fatalf("far hamming distance = %d, want > %d", got, dimCount*9/10)
	}
}

func TestBinaryVector_LengthMismatch(t *testing.T) {
	t.Parallel()

	// HammingDistance returns 0 for mismatched lengths.
	short := BinaryVector{0xFF}
	long := BinaryVector{0xFF, 0xFF}
	if got := HammingDistance(short, long); got != 0 {
		t.Errorf("HammingDistance(short, long) = %d, want 0", got)
	}
	if got := HammingDistance(long, short); got != 0 {
		t.Errorf("HammingDistance(long, short) = %d, want 0", got)
	}

	// DotFromBlob returns 0 for dimension mismatch.
	query := []float32{1, 2, 3}
	blob := EncodeFloat32Vec([]float32{1, 2})
	if got := DotFromBlob(query, blob); got != 0 {
		t.Errorf("DotFromBlob(query3, blob2) = %f, want 0", got)
	}
}

func assertFloat32SliceEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d; got=%v", len(got), len(want), got)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
