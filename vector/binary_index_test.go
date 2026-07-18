package vectors

import (
	"reflect"
	"testing"
)

func TestBinaryIndex_SearchCandidates(t *testing.T) {
	t.Parallel()
	dims := 4
	v1 := []float32{1, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0}
	v3 := []float32{0, 0, 1, 0}
	v4 := []float32{0, 0, 0, 1}

	blobs := [][]byte{
		EncodeFloat32Vec(v1),
		EncodeFloat32Vec(v2),
		EncodeFloat32Vec(v3),
		EncodeFloat32Vec(v4),
	}
	groups := []string{"g1", "g2", "g3", "g4"}
	indices := []int{0, 1, 0, 0}

	idx := NewBinaryIndex(blobs, groups, indices, dims)
	if got := idx.Len(); got != 4 {
		t.Fatalf("Len() = %d, want 4", got)
	}

	candidates, ok := idx.SearchCandidates([]float32{1, 0, 0, 0})
	if !ok {
		t.Fatal("SearchCandidates ok = false")
	}
	if len(candidates) == 0 {
		t.Fatal("SearchCandidates returned no candidates")
	}

	// Clear the index
	idx.Clear()
	if got := idx.Len(); got != 0 {
		t.Fatalf("Len() after Clear = %d, want 0", got)
	}
}

func TestBinaryIndex_SearchCandidatesLimit(t *testing.T) {
	t.Parallel()

	blobs := make([][]byte, 0, 105)
	groups := make([]string, 0, 105)
	indices := make([]int, 0, 105)
	for i := 0; i < 105; i++ {
		blobs = append(blobs, EncodeFloat32Vec([]float32{float32(i), float32(105 - i)}))
		groups = append(groups, "g"+threeDigit(i))
		indices = append(indices, i)
	}

	idx := NewBinaryIndex(blobs, groups, indices, 2)

	limited, ok := idx.SearchCandidatesLimit([]float32{52, 53}, 3)
	if !ok {
		t.Fatal("SearchCandidatesLimit ok = false")
	}
	if len(limited) != 3 {
		t.Fatalf("SearchCandidatesLimit len = %d, want 3", len(limited))
	}

	defaultCandidates, ok := idx.SearchCandidates([]float32{52, 53})
	if !ok {
		t.Fatal("SearchCandidates ok = false")
	}
	explicitDefault, ok := idx.SearchCandidatesLimit([]float32{52, 53}, binaryPreFilterK)
	if !ok {
		t.Fatal("SearchCandidatesLimit explicit default ok = false")
	}
	if !reflect.DeepEqual(explicitDefault, defaultCandidates) {
		t.Fatalf("default candidates = %#v, want %#v", defaultCandidates, explicitDefault)
	}
	if len(defaultCandidates) != binaryPreFilterK {
		t.Fatalf("default candidates len = %d, want %d", len(defaultCandidates), binaryPreFilterK)
	}

	all, ok := idx.SearchCandidatesLimit([]float32{52, 53}, 200)
	if !ok {
		t.Fatal("SearchCandidatesLimit all ok = false")
	}
	if len(all) != idx.Len() {
		t.Fatalf("all candidates len = %d, want %d", len(all), idx.Len())
	}

	empty, ok := idx.SearchCandidatesLimit([]float32{52, 53}, 0)
	if !ok {
		t.Fatal("SearchCandidatesLimit empty ok = false")
	}
	if len(empty) != 0 {
		t.Fatalf("empty candidates len = %d, want 0", len(empty))
	}

	mismatched, ok := idx.SearchCandidatesLimit([]float32{1, 2, 3}, 10)
	if ok {
		t.Fatal("SearchCandidatesLimit mismatched ok = true")
	}
	if mismatched != nil {
		t.Fatalf("mismatched candidates = %#v, want nil", mismatched)
	}
}

func TestBinaryIndex_SearchCandidatesLimitTieCutoff(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{1, 1}),
		EncodeFloat32Vec([]float32{1, 1}),
		EncodeFloat32Vec([]float32{1, 1}),
	}
	groups := []string{"z", "a", "m"}
	indices := []int{0, 0, 0}

	idx := NewBinaryIndex(blobs, groups, indices, 2)
	candidates, ok := idx.SearchCandidatesLimit([]float32{1, 1}, 2)
	if !ok {
		t.Fatal("SearchCandidatesLimit ok = false")
	}
	want := []HammingCandidate{
		{Group: "a", ChunkIndex: 0, Hamming: 0},
		{Group: "m", ChunkIndex: 0, Hamming: 0},
	}
	if !reflect.DeepEqual(candidates, want) {
		t.Fatalf("candidates = %#v, want %#v", candidates, want)
	}
}

func TestBinaryIndex_SkipsRowsWithMissingGroupMetadata(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{1, 0}),
		EncodeFloat32Vec([]float32{0, 1}),
	}

	idx := NewBinaryIndex(blobs, []string{"g1"}, []int{0, 1}, 2)
	if got := idx.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}

	candidates, ok := idx.SearchCandidates([]float32{1, 0})
	if !ok {
		t.Fatal("SearchCandidates ok = false")
	}
	if len(candidates) != 1 {
		t.Fatalf("SearchCandidates len = %d, want 1", len(candidates))
	}
	if candidates[0].Group != "g1" || candidates[0].ChunkIndex != 0 {
		t.Fatalf("candidate = %+v, want g1/0", candidates[0])
	}
}

func threeDigit(n int) string {
	return string([]byte{
		byte('0' + n/100),
		byte('0' + (n/10)%10),
		byte('0' + n%10),
	})
}

func TestBinaryIndex_SkipsRowsWithMissingChunkIndexMetadata(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{1, 0}),
		EncodeFloat32Vec([]float32{0, 1}),
	}

	idx := NewBinaryIndex(blobs, []string{"g1", "g2"}, []int{0}, 2)
	if got := idx.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}

	candidates, ok := idx.SearchCandidates([]float32{1, 0})
	if !ok {
		t.Fatal("SearchCandidates ok = false")
	}
	if len(candidates) != 1 {
		t.Fatalf("SearchCandidates len = %d, want 1", len(candidates))
	}
	if candidates[0].Group != "g1" || candidates[0].ChunkIndex != 0 {
		t.Fatalf("candidate = %+v, want g1/0", candidates[0])
	}
}

func TestNewBinaryIndexCheckedReportsSkippedRows(t *testing.T) {
	t.Parallel()

	blobs := [][]byte{
		EncodeFloat32Vec([]float32{1, 0}),
		{1, 2, 3},
		EncodeFloat32Vec([]float32{1, 0, 0}),
		EncodeFloat32Vec([]float32{0, 1}),
	}
	groups := []string{"g1", "bad-blob", "wrong-dims"}
	indices := []int{0, 1, 2, 3}

	idx, report := NewBinaryIndexChecked(blobs, groups, indices, 2)
	if got := idx.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}
	if report.InputRows != 4 ||
		report.IndexedRows != 1 ||
		report.SkippedBadBlob != 1 ||
		report.SkippedMissingMetadata != 1 ||
		report.DimensionMismatch != 1 {
		t.Fatalf("report = %+v, want input=4 indexed=1 bad_blob=1 missing_metadata=1 dimension_mismatch=1", report)
	}
}
