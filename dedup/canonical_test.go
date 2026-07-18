package dedup

import "testing"

func TestCanonicalizePicksMaxPerGroup(t *testing.T) {
	t.Parallel()

	groups := [][]int{
		{1, 5, 3},
		{9, 2},
		{7},
	}

	got := Canonicalize(groups, func(a, b int) bool { return a > b })

	want := []int{5, 9, 7}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestCanonicalizeTieKeepsEarliest(t *testing.T) {
	t.Parallel()

	// Distinct pointers with equal "value" — better never reports a tie as
	// preferable, so the earliest element must survive.
	a, b, c := &struct{ v int }{1}, &struct{ v int }{1}, &struct{ v int }{1}
	groups := [][]*struct{ v int }{{a, b, c}}

	got := Canonicalize(groups, func(x, y *struct{ v int }) bool { return x.v > y.v })

	if len(got) != 1 {
		t.Fatalf("len(got) = %d, want 1", len(got))
	}
	if got[0] != a {
		t.Errorf("tie did not keep earliest element")
	}
}

type hitItem struct {
	ID   string
	Hits int
}

func TestCanonicalizeWithMergeSumsHits(t *testing.T) {
	t.Parallel()

	groups := [][]*hitItem{
		{{ID: "a", Hits: 1}, {ID: "b", Hits: 4}, {ID: "c", Hits: 2}},
		{{ID: "d", Hits: 5}, {ID: "e", Hits: 3}},
	}

	got := CanonicalizeWith(
		groups,
		func(x, y *hitItem) bool { return x.Hits > y.Hits },
		func(winner, loser *hitItem) { winner.Hits += loser.Hits },
	)

	if len(got) != len(groups) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(groups))
	}
	if got[0].ID != "b" || got[0].Hits != 7 {
		t.Errorf("group 0 survivor = %+v, want {ID:b Hits:7}", *got[0])
	}
	if got[1].ID != "d" || got[1].Hits != 8 {
		t.Errorf("group 1 survivor = %+v, want {ID:d Hits:8}", *got[1])
	}
}

func TestCanonicalizeWithNilMergeMatchesCanonicalize(t *testing.T) {
	t.Parallel()

	groups := [][]int{{2, 8, 4}, {1}}
	better := func(a, b int) bool { return a > b }

	withNil := CanonicalizeWith(groups, better, nil)
	plain := Canonicalize(groups, better)

	if len(withNil) != len(plain) {
		t.Fatalf("len mismatch: %d vs %d", len(withNil), len(plain))
	}
	for i := range plain {
		if withNil[i] != plain[i] {
			t.Errorf("index %d: nil-merge %d != Canonicalize %d", i, withNil[i], plain[i])
		}
	}
}

func TestCanonicalizeSkipsEmptyAndPassesSingleton(t *testing.T) {
	t.Parallel()

	groups := [][]int{
		{},
		{42},
		{},
		{3, 9},
	}

	got := Canonicalize(groups, func(a, b int) bool { return a > b })

	want := []int{42, 9}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}
