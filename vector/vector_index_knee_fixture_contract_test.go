package vectors

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

func TestKneeFixturesDeterministic(t *testing.T) {
	t.Parallel()
	first := newKneeFixture(8, 7)
	second := newKneeFixture(8, 7)
	if !slices.Equal(first.arena, second.arena) {
		t.Fatal("fixture arena is not repeatable")
	}
	for index, vector := range first.vectors {
		if len(vector) != 7 {
			t.Fatalf("vector %d dimensions = %d, want 7", index, len(vector))
		}
		if !IsUnit32(vector, 1e-5) {
			t.Fatalf("vector %d is not finite and normalized: %v", index, vector)
		}
		if first.groups[index] != fmt.Sprintf("row-%08d", index) || first.chunkIndexes[index] != 0 {
			t.Fatalf("row %d identity = %q#%d", index, first.groups[index], first.chunkIndexes[index])
		}
		if !slices.Equal(first.blobs[index], second.blobs[index]) {
			t.Fatalf("vector %d encoding is not repeatable", index)
		}
	}

	firstQueries := newKneeQueries(5, 7)
	secondQueries := newKneeQueries(5, 7)
	for index := range firstQueries {
		if !slices.Equal(firstQueries[index], secondQueries[index]) {
			t.Fatalf("query %d is not repeatable", index)
		}
		if len(firstQueries[index]) != 7 || !IsUnit32(firstQueries[index], 1e-5) {
			t.Fatalf("query %d is not finite, normalized, and dimensionally valid", index)
		}
	}
	if slices.Equal(first.vectors[0], firstQueries[0]) {
		t.Fatal("corpus and query streams are not independent")
	}
}

func TestKneeSearchPaths(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{1, 0}, {0, 1}}
	blobs := [][]byte{EncodeFloat32Vec(vectors[0]), EncodeFloat32Vec(vectors[1])}
	groups := []string{"row-00000000", "row-00000001"}
	chunkIndexes := []int{0, 0}
	chunks := []IndexChunk{
		{Group: groups[0], Offset: 0, Length: len(blobs[0])},
		{Group: groups[1], Offset: len(blobs[0]), Length: len(blobs[1])},
	}
	arena := append(slices.Clone(blobs[0]), blobs[1]...)
	exact, exactReport := NewExactIndexChecked(2, chunks, arena)
	if err := validateKneeBuildReport("exact", exactReport, 2); err != nil {
		t.Fatal(err)
	}
	binary, binaryReport := NewBinaryIndexChecked(blobs, groups, chunkIndexes, 2)
	if err := validateKneeBuildReport("binary", binaryReport, 2); err != nil {
		t.Fatal(err)
	}

	query := []float32{0, 1}
	candidates, ok := binary.SearchCandidatesLimit(query, 2)
	if !ok || len(candidates) != 2 {
		t.Fatalf("binary candidates = %#v, %t; want two", candidates, ok)
	}
	if candidates[0].Hamming != candidates[1].Hamming ||
		candidates[0].Group != groups[0] ||
		candidates[1].Group != groups[1] {
		t.Fatalf("binary candidates = %#v, want tied row order", candidates)
	}
	keys := []IndexKey{
		{Group: candidates[0].Group, ChunkIndex: candidates[0].ChunkIndex},
		{Group: candidates[1].Group, ChunkIndex: candidates[1].ChunkIndex},
	}
	restricted, ok := exact.SearchKeys(query, 2, kneeMinimumSimilarity, keys)
	if !ok || len(restricted) != 2 || restricted[0].Group != groups[1] {
		t.Fatalf("SearchKeys() = %#v, %t; want second row first", restricted, ok)
	}
	full, ok := exact.Search(query, 2, kneeMinimumSimilarity)
	if !ok || len(full) != 2 || full[0].Group != groups[1] {
		t.Fatalf("Search() = %#v, %t; want second row first", full, ok)
	}
}

func TestKneeProfileMetadata(t *testing.T) {
	t.Parallel()
	exact, err := newKneeProfile(IndexKindExact, 10, 65, 0)
	if err != nil {
		t.Fatalf("newKneeProfile(exact): %v", err)
	}
	if exact.ID != "exact" ||
		exact.Kind != IndexKindExact ||
		exact.CandidateLimit != 0 ||
		exact.ExactRescore ||
		exact.Quantization != "float32" ||
		exact.MemoryEstimate != (MemoryEstimate{Bytes: 2600, Label: kneeMemoryLabel}) {
		t.Fatalf("exact profile = %#v", exact)
	}

	binary, err := newKneeProfile(IndexKindBinary, 10, 65, 7)
	if err != nil {
		t.Fatalf("newKneeProfile(binary): %v", err)
	}
	if binary.ID != "binary-candidates-7" ||
		binary.Kind != IndexKindBinary ||
		binary.CandidateLimit != 7 ||
		!binary.ExactRescore ||
		binary.Quantization != "median_binary" ||
		binary.MemoryEstimate != (MemoryEstimate{Bytes: 420, Label: kneeMemoryLabel}) {
		t.Fatalf("binary profile = %#v", binary)
	}
}

func TestKneePayloadEstimates(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		kind       IndexKind
		rows, dims int
		want       int64
		wantErr    bool
	}{
		{name: "exact formula", kind: IndexKindExact, rows: 10, dims: 65, want: 2600},
		{name: "binary formula", kind: IndexKindBinary, rows: 10, dims: 65, want: 420},
		{name: "zero rows", kind: IndexKindExact, rows: 0, dims: 1, wantErr: true},
		{name: "negative rows", kind: IndexKindExact, rows: -1, dims: 1, wantErr: true},
		{name: "zero dimensions", kind: IndexKindExact, rows: 1, dims: 0, wantErr: true},
		{name: "negative dimensions", kind: IndexKindExact, rows: 1, dims: -1, wantErr: true},
		{name: "unsupported kind", kind: IndexKind("hnsw"), rows: 1, dims: 1, wantErr: true},
		{name: "exact value product overflow", kind: IndexKindExact, rows: math.MaxInt, dims: 2, wantErr: true},
		{name: "exact byte product overflow", kind: IndexKindExact, rows: math.MaxInt, dims: 1, wantErr: true},
		{name: "binary word addition overflow", kind: IndexKindBinary, rows: 1, dims: math.MaxInt, wantErr: true},
		{name: "binary word product overflow", kind: IndexKindBinary, rows: math.MaxInt, dims: 128, wantErr: true},
		{name: "binary row byte overflow", kind: IndexKindBinary, rows: math.MaxInt, dims: 1, wantErr: true},
		{name: "binary payload addition overflow", kind: IndexKindBinary, rows: math.MaxInt / 8, dims: 64, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := kneePayloadEstimate(test.kind, test.rows, test.dims)
			if test.wantErr {
				if err == nil {
					t.Fatalf("kneePayloadEstimate() = %d, nil; want error", got)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("kneePayloadEstimate() = %d, %v; want %d, nil", got, err, test.want)
			}
		})
	}
}

func TestKneeBuildReport(t *testing.T) {
	t.Parallel()
	good := IndexBuildReport{InputRows: 2, IndexedRows: 2}
	if err := validateKneeBuildReport("exact", good, 2); err != nil {
		t.Fatalf("valid report: %v", err)
	}
	tests := []struct {
		field  string
		mutate func(*IndexBuildReport)
	}{
		{field: "InputRows", mutate: func(report *IndexBuildReport) { report.InputRows = 1 }},
		{field: "IndexedRows", mutate: func(report *IndexBuildReport) { report.IndexedRows = 1 }},
		{field: "SkippedBadSpan", mutate: func(report *IndexBuildReport) { report.SkippedBadSpan = 1 }},
		{field: "SkippedBadBlob", mutate: func(report *IndexBuildReport) { report.SkippedBadBlob = 1 }},
		{
			field:  "SkippedMissingMetadata",
			mutate: func(report *IndexBuildReport) { report.SkippedMissingMetadata = 1 },
		},
		{
			field:  "SkippedDuplicateKey",
			mutate: func(report *IndexBuildReport) { report.SkippedDuplicateKey = 1 },
		},
		{field: "DimensionMismatch", mutate: func(report *IndexBuildReport) { report.DimensionMismatch = 1 }},
		{field: "MedianError", mutate: func(report *IndexBuildReport) { report.MedianError = "bad median" }},
		{field: "QuantizeError", mutate: func(report *IndexBuildReport) { report.QuantizeError = "bad quantize" }},
	}
	for _, test := range tests {
		t.Run(test.field, func(t *testing.T) {
			t.Parallel()
			report := good
			test.mutate(&report)
			err := validateKneeBuildReport("binary-candidates-2", report, 2)
			if err == nil ||
				!strings.Contains(err.Error(), "binary-candidates-2") ||
				!strings.Contains(err.Error(), test.field) {
				t.Fatalf("error = %v, want profile and field context", err)
			}
		})
	}
}
