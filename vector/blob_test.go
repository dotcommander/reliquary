package vectors

import (
	"encoding/hex"
	"math"
	"strings"
	"testing"
)

func TestEncodeFloat64Vec_GoldenBytes(t *testing.T) {
	t.Parallel()
	v := []float64{1.5, -0.5, 0.0}
	expectedHex := "000000000000F83F000000000000E0BF0000000000000000"
	decoded, err := hex.DecodeString(expectedHex)
	if err != nil {
		t.Fatalf("unexpected hex decode error: %v", err)
	}
	got := EncodeFloat64Vec(v)
	if len(got) != len(decoded) {
		t.Fatalf("encoded length = %d, want %d", len(got), len(decoded))
	}
	for i := range got {
		if got[i] != decoded[i] {
			t.Fatalf("byte %d mismatch: 0x%02x, want 0x%02x", i, got[i], decoded[i])
		}
	}
}

func TestEncodeFloat64Vec_RoundTrip(t *testing.T) {
	t.Parallel()
	original := []float64{1.0, -2.5, 3.14, 0.0, -0.001}
	encoded := EncodeFloat64Vec(original)
	decoded := DecodeFloat64Vec(encoded)
	if len(decoded) != len(original) {
		t.Fatalf("len(decoded) = %d, want %d", len(decoded), len(original))
	}
	for i, got := range decoded {
		if got != original[i] {
			t.Fatalf("decoded[%d] = %f, want %f", i, got, original[i])
		}
	}
}

func TestDecodeFloat64Vec_InvalidLength(t *testing.T) {
	t.Parallel()
	got := DecodeFloat64Vec([]byte{1, 2, 3})
	if got != nil {
		t.Fatalf("expected nil for invalid length, got %v", got)
	}
}

func TestDecodeFloat64Vec_NilAndEmpty(t *testing.T) {
	t.Parallel()
	if got := DecodeFloat64Vec(nil); got == nil {
		t.Fatalf("nil input should return empty slice")
	}
	if got := DecodeFloat64Vec([]byte{}); got == nil {
		t.Fatalf("empty input should return non-nil empty slice")
	}
	if len(DecodeFloat64Vec([]byte{})) != 0 {
		t.Fatalf("expected zero length for empty input")
	}
}

func TestDotFromBlob_SelfSimilarity(t *testing.T) {
	t.Parallel()
	v := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	Normalize32(v)
	blob := EncodeFloat32Vec(v)
	got := DotFromBlob(v, blob)
	if diff := math.Abs(float64(got) - 1.0); diff > 1e-5 {
		t.Errorf("dot(v, v) for normalized v = %f, want ~1.0", got)
	}
}

func TestDotFromBlob_DimensionMismatch(t *testing.T) {
	t.Parallel()
	query := []float32{1, 2, 3}
	blob := EncodeFloat32Vec([]float32{1, 2})
	if got := DotFromBlob(query, blob); got != 0 {
		t.Errorf("expected 0 for dimension mismatch, got %f", got)
	}
}

func TestDotFromBlob_NonMultipleOfFourBlob(t *testing.T) {
	t.Parallel()
	query := []float32{1, 2, 3}
	base := EncodeFloat32Vec(query) // length 4N
	for _, extra := range []int{1, 2, 3} {
		blob := make([]byte, len(base)+extra)
		copy(blob, base)
		if got := DotFromBlob(query, blob); got != 0 {
			t.Errorf("len(blob)=4N+%d: expected 0, got %f", extra, got)
		}
	}
}

func TestDotFromBlob_IdenticalNormalized(t *testing.T) {
	t.Parallel()
	v := []float32{0.5, 0.3, 0.8, 0.1}
	Normalize32(v)
	blob := EncodeFloat32Vec(v)
	got := DotFromBlob(v, blob)
	if diff := math.Abs(float64(got) - 1.0); diff > 1e-5 {
		t.Errorf("dot(v, v) for normalized v = %f, want ~1.0", got)
	}
}

func TestDotFromBlob_Orthogonal(t *testing.T) {
	t.Parallel()
	a := []float32{1, 0, 0, 0}
	b := []float32{0, 1, 0, 0}
	Normalize32(a)
	Normalize32(b)
	blob := EncodeFloat32Vec(b)
	got := DotFromBlob(a, blob)
	if math.Abs(float64(got)) > 1e-6 {
		t.Errorf("dot(orthogonal) = %f, want ~0", got)
	}
}

func TestTopKFromBlob(t *testing.T) {
	t.Parallel()

	query := []float32{1, 0}
	blobs := [][]byte{
		EncodeFloat32Vec([]float32{0.5, 0.5}),
		{1, 2, 3},
		EncodeFloat32Vec([]float32{1, 0}),
		EncodeFloat32Vec([]float32{0.25, 0}),
		EncodeFloat32Vec([]float32{1, 0}),
	}

	got := TopKFromBlob(query, blobs, 3, 0.3)
	want := []ScoredIndex{
		{Index: 2, Score: 1},
		{Index: 4, Score: 1},
		{Index: 0, Score: 0.5},
	}
	assertScoredIndexes(t, got, want)
}

func TestTopKFromBlobLimitAndEmptyInputs(t *testing.T) {
	t.Parallel()

	query := []float32{1, 0}
	blobs := [][]byte{
		EncodeFloat32Vec([]float32{0, 1}),
		EncodeFloat32Vec([]float32{1, 0}),
	}

	if got := TopKFromBlob(nil, blobs, 1, 0); len(got) != 0 {
		t.Fatalf("nil query = %#v, want empty", got)
	}
	if got := TopKFromBlob(query, blobs, 0, 0); len(got) != 0 {
		t.Fatalf("limit 0 = %#v, want empty", got)
	}

	got := TopKFromBlob(query, blobs, 10, -1)
	want := []ScoredIndex{
		{Index: 1, Score: 1},
		{Index: 0, Score: 0},
	}
	assertScoredIndexes(t, got, want)
}

func TestTopKFromBlobMinScoreFilters(t *testing.T) {
	t.Parallel()

	query := []float32{1, 0}
	blobs := [][]byte{
		EncodeFloat32Vec([]float32{0.2, 0}),
		EncodeFloat32Vec([]float32{0.8, 0}),
	}

	got := TopKFromBlob(query, blobs, 10, 0.5)
	want := []ScoredIndex{{Index: 1, Score: 0.8}}
	assertScoredIndexes(t, got, want)
}

func TestTopKFromBlobRejectsNonFiniteValues(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{float32(math.NaN()), 0}),
		EncodeFloat32Vec([]float32{1, 0}),
		EncodeFloat32Vec([]float32{float32(math.Inf(1)), 0}),
		EncodeFloat32Vec([]float32{float32(math.Inf(-1)), 0}),
		EncodeFloat32Vec([]float32{1, 0}),
		EncodeFloat32Vec([]float32{0.5, 0}),
	}
	want := []ScoredIndex{
		{Index: 1, Score: 1},
		{Index: 4, Score: 1},
		{Index: 5, Score: 0.5},
	}
	assertScoredIndexes(t, TopKFromBlob([]float32{1, 0}, blobs, 10, -1), want)

	queries := [][]float32{
		{float32(math.NaN()), 0},
		{float32(math.Inf(1)), 0},
		{float32(math.Inf(-1)), 0},
	}
	for _, query := range queries {
		if got := TopKFromBlob(query, blobs, 10, -1); got != nil {
			t.Fatalf("TopKFromBlob(%v) = %#v, want nil", query, got)
		}
	}

	overflowBlobs := [][]byte{
		EncodeFloat32Vec([]float32{math.MaxFloat32, 0}),
		EncodeFloat32Vec([]float32{0, 1}),
	}
	assertScoredIndexes(t, TopKFromBlob([]float32{2, 0}, overflowBlobs, 10, -1), []ScoredIndex{{Index: 1, Score: 0}})
}

func assertScoredIndexes(t *testing.T, got, want []ScoredIndex) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d; got %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Index != want[i].Index || got[i].Score != want[i].Score {
			t.Fatalf("result[%d] = %#v, want %#v", i, got[i], want[i])
		}
	}
}

func TestEncodeDecodeFloat32Vec_Roundtrip(t *testing.T) {
	t.Parallel()
	original := []float32{1.0, -2.5, 3.14, 0.0, -0.001}
	encoded := EncodeFloat32Vec(original)
	decoded := DecodeFloat32Vec(encoded)
	if len(decoded) != len(original) {
		t.Fatalf("len(decoded) = %d, want %d", len(decoded), len(original))
	}
	for i, got := range decoded {
		if got != original[i] {
			t.Errorf("decoded[%d] = %f, want %f", i, got, original[i])
		}
	}
}

func TestDecodeFloat32Vec_InvalidLength(t *testing.T) {
	t.Parallel()
	got := DecodeFloat32Vec([]byte{1, 2, 3})
	if got != nil {
		t.Errorf("expected nil for non-multiple-of-4 input, got %v", got)
	}
}

func TestDecodeFloat32Batch_RoundTrip(t *testing.T) {
	t.Parallel()

	vectors := [][]float32{
		{1.0, -2.5, 3.14, 0.0},
		{0.5, -0.25, 7.1},
		nil,
	}
	blobs := make([][]byte, len(vectors))
	for i, v := range vectors {
		blobs[i] = EncodeFloat32Vec(v)
	}

	decoded, err := DecodeFloat32Batch(blobs)
	if err != nil {
		t.Fatalf("DecodeFloat32Batch_RoundTrip: unexpected error: %v", err)
	}
	if len(decoded) != len(vectors) {
		t.Fatalf("DecodeFloat32Batch_RoundTrip: len(decoded) = %d, want %d", len(decoded), len(vectors))
	}
	for i, want := range vectors {
		got := decoded[i]
		if len(got) != len(want) {
			t.Fatalf("DecodeFloat32Batch_RoundTrip: decoded[%d] len = %d, want %d", i, len(got), len(want))
		}
		for j, value := range want {
			if got[j] != value {
				t.Fatalf("DecodeFloat32Batch_RoundTrip: decoded[%d][%d] = %f, want %f", i, j, got[j], value)
			}
		}
	}
}

func TestDecodeFloat32Batch_NamesBadIndex(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{1, 2, 3}),
		{1, 2, 3, 4, 5},
		EncodeFloat32Vec([]float32{0.25}),
	}
	_, err := DecodeFloat32Batch(blobs)
	if err == nil {
		t.Fatal("TestDecodeFloat32Batch_NamesBadIndex: expected error, got nil")
	}
	if got := err.Error(); got == "" || got == "error" {
		t.Fatalf("TestDecodeFloat32Batch_NamesBadIndex: unexpected empty error message")
	}
	if !strings.Contains(err.Error(), "index 1") {
		t.Fatalf("TestDecodeFloat32Batch_NamesBadIndex: error = %q, want it to name index 1", err)
	}
}

func TestDecodeFloat32Batch_Empty(t *testing.T) {
	t.Parallel()

	got, err := DecodeFloat32Batch(nil)
	if err != nil {
		t.Fatalf("TestDecodeFloat32Batch_Empty: unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("TestDecodeFloat32Batch_Empty: got nil, want non-nil empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("TestDecodeFloat32Batch_Empty: len(got) = %d, want 0", len(got))
	}
}
