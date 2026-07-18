package vectors

import (
	"math"
	"testing"
)

func TestExactIndex_SearchAndFilter(t *testing.T) {
	t.Parallel()
	dims := 4
	v1 := []float32{1, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0}
	v3 := []float32{0, 0, 1, 0}
	v4 := []float32{0.7071, 0.7071, 0, 0} // closer to v1 and v2

	arena := append(EncodeFloat32Vec(v1), EncodeFloat32Vec(v2)...)
	arena = append(arena, EncodeFloat32Vec(v3)...)
	arena = append(arena, EncodeFloat32Vec(v4)...)

	chunks := []IndexChunk{
		{Group: "group-a", ChunkIndex: 0, Offset: 0, Length: 16},
		{Group: "group-a", ChunkIndex: 1, Offset: 16, Length: 16},
		{Group: "group-b", ChunkIndex: 0, Offset: 32, Length: 16},
		{Group: "group-c", ChunkIndex: 0, Offset: 48, Length: 16},
	}

	idx := NewExactIndex(dims, chunks, arena)
	if got := idx.Len(); got != 4 {
		t.Fatalf("Len() = %d, want 4", got)
	}

	// Search closest to v1
	res, ok := idx.Search([]float32{1, 0, 0, 0}, 2, 0.1)
	if !ok {
		t.Fatal("Search ok = false")
	}
	if len(res) != 2 {
		t.Fatalf("Search len = %d, want 2", len(res))
	}
	if res[0].Group != "group-a" || res[0].ChunkIndex != 0 {
		t.Fatalf("Search[0] = %+v, want group-a chunk 0", res[0])
	}
	assertInDelta(t, res[0].Score, 1.0, 0.0001)

	if res[1].Group != "group-c" || res[1].ChunkIndex != 0 {
		t.Fatalf("Search[1] = %+v, want group-c chunk 0", res[1])
	}
	assertInDelta(t, res[1].Score, 0.7071, 0.0001)

	// Search Filtered to group-b and group-c only
	resFiltered, ok := idx.SearchFiltered([]float32{1, 0, 0, 0}, 2, -1.0, []string{"group-b", "group-c"})
	if !ok {
		t.Fatal("SearchFiltered ok = false")
	}
	if len(resFiltered) != 2 {
		t.Fatalf("SearchFiltered len = %d, want 2", len(resFiltered))
	}
	if resFiltered[0].Group != "group-c" {
		t.Fatalf("SearchFiltered[0].Group = %q, want group-c", resFiltered[0].Group)
	}
	assertInDelta(t, resFiltered[0].Score, 0.7071, 0.0001)
	if resFiltered[1].Group != "group-b" {
		t.Fatalf("SearchFiltered[1].Group = %q, want group-b", resFiltered[1].Group)
	}
	assertInDelta(t, resFiltered[1].Score, 0.0, 0.0001)

	// Max Pool Search
	resMaxPool, ok := idx.SearchGroupsByMaxPool([]float32{0.7071, 0.7071, 0, 0}, 2)
	if !ok {
		t.Fatal("SearchGroupsByMaxPool ok = false")
	}
	if len(resMaxPool) != 2 {
		t.Fatalf("SearchGroupsByMaxPool len = %d, want 2", len(resMaxPool))
	}
	// group-a has v4's projection: 0.7071*0.7071 + 0.7071*0.7071 = 1.0 (with v4 itself or group-a's v4 chunk)
	// group-c has group-c's chunk v4: dot with itself = 1.0
	assertInDelta(t, resMaxPool[0].Score, 1.0, 0.0001)
	assertInDelta(t, resMaxPool[1].Score, 0.7071, 0.0001)

	idx.Clear()
	if got := idx.Len(); got != 0 {
		t.Fatalf("Len() after Clear = %d, want 0", got)
	}
}

func TestExactIndex_DropsOutOfBoundsChunks(t *testing.T) {
	t.Parallel()

	dims := 4
	v1 := []float32{1, 0, 0, 0}
	arena := EncodeFloat32Vec(v1) // 16 bytes: only one in-bounds chunk fits

	chunks := []IndexChunk{
		{Group: "group-a", ChunkIndex: 0, Offset: 0, Length: 16},  // in bounds
		{Group: "group-b", ChunkIndex: 0, Offset: 16, Length: 16}, // Offset+Length > len(arena)
	}

	idx := NewExactIndex(dims, chunks, arena)
	if got := idx.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}

	res, ok := idx.Search([]float32{1, 0, 0, 0}, 5, -1.0)
	if !ok {
		t.Fatal("Search ok = false")
	}
	if len(res) != 1 {
		t.Fatalf("Search len = %d, want 1", len(res))
	}
	if res[0].Group != "group-a" {
		t.Fatalf("Search[0].Group = %q, want group-a", res[0].Group)
	}
}

func TestExactIndex_SearchFilteredInterleavedGroups(t *testing.T) {
	t.Parallel()

	vA0 := []float32{1, 0}
	vB0 := []float32{0, 1}
	vA1 := []float32{1, 0}

	arena := append(EncodeFloat32Vec(vA0), EncodeFloat32Vec(vB0)...)
	arena = append(arena, EncodeFloat32Vec(vA1)...)

	chunks := []IndexChunk{
		{Group: "group-a", ChunkIndex: 0, Offset: 0, Length: 8},
		{Group: "group-b", ChunkIndex: 0, Offset: 8, Length: 8},
		{Group: "group-a", ChunkIndex: 1, Offset: 16, Length: 8},
	}

	idx := NewExactIndex(2, chunks, arena)

	res, ok := idx.SearchFiltered([]float32{0, 1}, 10, -1.0, []string{"group-a"})
	if !ok {
		t.Fatal("SearchFiltered ok = false")
	}
	if len(res) != 2 {
		t.Fatalf("SearchFiltered len = %d, want 2", len(res))
	}
	for _, r := range res {
		if r.Group != "group-a" {
			t.Fatalf("SearchFiltered result group = %q, want group-a", r.Group)
		}
	}
}

func TestNewExactIndexCheckedReportsSkippedRows(t *testing.T) {
	t.Parallel()

	arena := EncodeFloat32Vec([]float32{1, 0})
	chunks := []IndexChunk{
		{Group: "valid", ChunkIndex: 0, Offset: 0, Length: 8},
		{Group: "bad-span", ChunkIndex: 0, Offset: 8, Length: 8},
		{Group: "wrong-dims", ChunkIndex: 0, Offset: 0, Length: 4},
	}

	idx, report := NewExactIndexChecked(2, chunks, arena)
	if got := idx.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}
	if report.InputRows != 3 || report.IndexedRows != 1 || report.SkippedBadSpan != 1 || report.DimensionMismatch != 1 {
		t.Fatalf("report = %+v, want input=3 indexed=1 bad_span=1 dimension_mismatch=1", report)
	}

	keyResults, ok := idx.SearchKeys([]float32{1, 0}, 10, -1, []IndexKey{
		{Group: "valid", ChunkIndex: 0},
		{Group: "wrong-dims", ChunkIndex: 0},
	})
	if !ok {
		t.Fatal("SearchKeys ok = false")
	}
	if len(keyResults) != 1 || keyResults[0].Group != "valid" {
		t.Fatalf("SearchKeys = %+v, want only valid chunk", keyResults)
	}

	filtered, ok := idx.SearchFiltered([]float32{1, 0}, 10, -1, []string{"valid", "wrong-dims"})
	if !ok {
		t.Fatal("SearchFiltered ok = false")
	}
	if len(filtered) != 1 || filtered[0].Group != "valid" {
		t.Fatalf("SearchFiltered = %+v, want only valid chunk", filtered)
	}
}

func TestNewExactIndexCheckedRejectsOverflowSpan(t *testing.T) {
	t.Parallel()

	arena := EncodeFloat32Vec([]float32{1})
	idx, report := NewExactIndexChecked(1, []IndexChunk{
		{Group: "overflow", ChunkIndex: 0, Offset: math.MaxInt - 1, Length: 4},
	}, arena)

	if got := idx.Len(); got != 0 {
		t.Fatalf("Len() = %d, want 0", got)
	}
	if report.InputRows != 1 || report.IndexedRows != 0 || report.SkippedBadSpan != 1 {
		t.Fatalf("report = %+v, want input=1 indexed=0 bad_span=1", report)
	}
	if results, ok := idx.Search([]float32{1}, 10, -1); ok || len(results) != 0 {
		t.Fatalf("Search() = (%+v, %v), want empty false", results, ok)
	}
}

func TestExactIndex_SearchKeys(t *testing.T) {
	t.Parallel()

	vA0 := []float32{1, 0}
	vB0 := []float32{0, 1}
	vA1 := []float32{0.9, 0.1}
	arena := append(EncodeFloat32Vec(vA0), EncodeFloat32Vec(vB0)...)
	arena = append(arena, EncodeFloat32Vec(vA1)...)

	idx := NewExactIndex(2, []IndexChunk{
		{Group: "a", ChunkIndex: 0, Offset: 0, Length: 8},
		{Group: "b", ChunkIndex: 0, Offset: 8, Length: 8},
		{Group: "a", ChunkIndex: 1, Offset: 16, Length: 8},
	}, arena)

	results, ok := idx.SearchKeys([]float32{1, 0}, 5, -1, []IndexKey{
		{Group: "a", ChunkIndex: 1},
		{Group: "missing", ChunkIndex: 0},
		{Group: "a", ChunkIndex: 1},
		{Group: "b", ChunkIndex: 0},
	})
	if !ok {
		t.Fatal("SearchKeys ok = false")
	}
	if len(results) != 2 {
		t.Fatalf("SearchKeys len = %d, want 2", len(results))
	}
	if results[0].Group != "a" || results[0].ChunkIndex != 1 {
		t.Fatalf("SearchKeys[0] = %+v, want a/1", results[0])
	}
	if results[1].Group != "b" || results[1].ChunkIndex != 0 {
		t.Fatalf("SearchKeys[1] = %+v, want b/0", results[1])
	}
}

func assertInDelta(t *testing.T, got, want, delta float64) {
	t.Helper()
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	if diff > delta {
		t.Fatalf("got %v, want %v within %v", got, want, delta)
	}
}
