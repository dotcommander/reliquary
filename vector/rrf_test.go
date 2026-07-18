package vectors

import (
	"testing"
)

func TestRRF_FusesTwoLists(t *testing.T) {
	t.Parallel()

	ranked := [][]int{
		{0, 1, 2},
		{3, 1, 4},
	}
	got, max := RRF(ranked, 60)

	want := []Scored{
		{Index: 1, Score: 1.0/62.0 + 1.0/62.0},
		{Index: 0, Score: 1.0 / 61.0},
		{Index: 3, Score: 1.0 / 61.0},
		{Index: 2, Score: 1.0 / 63.0},
		{Index: 4, Score: 1.0 / 63.0},
	}

	if len(got) != len(want) {
		t.Fatalf("TestRRF_FusesTwoLists: len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Index != want[i].Index {
			t.Fatalf("TestRRF_FusesTwoLists: got[%d].Index = %d, want %d", i, got[i].Index, want[i].Index)
		}
		if !approxEq64(got[i].Score, want[i].Score, 1e-9) {
			t.Fatalf("TestRRF_FusesTwoLists: got[%d].Score = %f, want %f", i, got[i].Score, want[i].Score)
		}
	}

	if !approxEq64(max, 1.0/31.0, 1e-9) {
		t.Fatalf("TestRRF_FusesTwoLists: max = %f, want %f", max, 1.0/31.0)
	}
}

func TestRRF_KZeroEqualsDefault(t *testing.T) {
	t.Parallel()

	ranked := [][]int{{2, 3, 1}, {3, 2, 0}}
	gotDefault, maxDefault := RRF(ranked, 60)
	gotZero, maxZero := RRF(ranked, 0)

	if len(gotDefault) != len(gotZero) {
		t.Fatalf("TestRRF_KZeroEqualsDefault: len(gotDefault) = %d, len(gotZero) = %d", len(gotDefault), len(gotZero))
	}
	for i := range gotDefault {
		if gotDefault[i].Index != gotZero[i].Index {
			t.Fatalf("TestRRF_KZeroEqualsDefault: gotDefault[%d].Index = %d, gotZero[%d].Index = %d", i, gotDefault[i].Index, i, gotZero[i].Index)
		}
		if gotDefault[i].Score != gotZero[i].Score {
			t.Fatalf("TestRRF_KZeroEqualsDefault: gotDefault[%d].Score = %f, gotZero[%d].Score = %f", i, gotDefault[i].Score, i, gotZero[i].Score)
		}
	}
	if maxDefault != maxZero {
		t.Fatalf("TestRRF_KZeroEqualsDefault: maxDefault = %f, maxZero = %f", maxDefault, maxZero)
	}
}

func TestRRF_TieBreakAscendingIndex(t *testing.T) {
	t.Parallel()

	ranked := [][]int{
		{0, 1, 4},
		{1, 0, 3},
	}
	got, _ := RRF(ranked, 60)

	if len(got) < 2 {
		t.Fatalf("TestRRF_TieBreakAscendingIndex: expected at least 2 results, got %d", len(got))
	}
	// Two top results are ties between index 0 and 1.
	if got[0].Score != got[1].Score {
		t.Fatalf("TestRRF_TieBreakAscendingIndex: expected tie at top, got %f and %f", got[0].Score, got[1].Score)
	}
	if got[0].Index != 0 || got[1].Index != 1 {
		t.Fatalf("TestRRF_TieBreakAscendingIndex: first two indices = (%d, %d), want (0, 1)", got[0].Index, got[1].Index)
	}
}

func TestRRF_MaxIsFirstScore(t *testing.T) {
	t.Parallel()

	ranked := [][]int{{2, 1, 0}, {1, 3, 4}}
	got, max := RRF(ranked, 10)

	if len(got) == 0 {
		t.Fatal("TestRRF_MaxIsFirstScore: expected non-empty result")
	}
	if got[0].Score != max {
		t.Fatalf("TestRRF_MaxIsFirstScore: first score %f, max %f", got[0].Score, max)
	}
	for i, s := range got {
		if s.Score > max {
			t.Fatalf("TestRRF_MaxIsFirstScore: got[%d].Score %f exceeds max %f", i, s.Score, max)
		}
	}
}

func TestRRF_Empty(t *testing.T) {
	t.Parallel()

	gotNil, maxNil := RRF(nil, 60)
	gotEmptyList, maxEmptyList := RRF([][]int{{}}, 60)

	if gotNil == nil {
		t.Fatal("TestRRF_Empty: gotNil is nil, want non-nil empty slice")
	}
	if len(gotNil) != 0 {
		t.Fatalf("TestRRF_Empty: len(gotNil) = %d, want 0", len(gotNil))
	}
	if maxNil != 0 {
		t.Fatalf("TestRRF_Empty: maxNil = %f, want 0", maxNil)
	}

	if gotEmptyList == nil {
		t.Fatal("TestRRF_Empty: gotEmptyList is nil, want non-nil empty slice")
	}
	if len(gotEmptyList) != 0 {
		t.Fatalf("TestRRF_Empty: len(gotEmptyList) = %d, want 0", len(gotEmptyList))
	}
	if maxEmptyList != 0 {
		t.Fatalf("TestRRF_Empty: maxEmptyList = %f, want 0", maxEmptyList)
	}
}

// TestRRF_DuplicateIndexInOneListAccumulates pins that RRF does NOT dedup a
// repeated index within a single input list: each occurrence contributes
// 1/(k+rank). Index 0 appears at ranks 1 and 3 in the same list, so its score
// is 1/(60+1) + 1/(60+3).
func TestRRF_DuplicateIndexInOneListAccumulates(t *testing.T) {
	t.Parallel()

	ranked := [][]int{{0, 1, 0}}
	got, max := RRF(ranked, 60)

	want := map[int]float64{
		0: 1.0/61.0 + 1.0/63.0,
		1: 1.0 / 62.0,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for _, s := range got {
		if !approxEq64(s.Score, want[s.Index], 1e-9) {
			t.Fatalf("Index %d score = %v, want %v", s.Index, s.Score, want[s.Index])
		}
	}
	// Index 0 (accumulated twice) outranks index 1.
	if got[0].Index != 0 {
		t.Fatalf("got[0].Index = %d, want 0 (duplicate index accumulates highest)", got[0].Index)
	}
	if !approxEq64(max, 1.0/61.0+1.0/63.0, 1e-9) {
		t.Fatalf("max = %v, want %v", max, 1.0/61.0+1.0/63.0)
	}
}
