package lexical

import (
	"math"
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary/retrieval"
	"github.com/dotcommander/reliquary/vector"
)

func TestRankByScore_DeterministicOrdering(t *testing.T) {
	t.Parallel()

	got := RankByScore([]Candidate{
		{ID: "b", Score: 2},
		{ID: "c", Score: 1},
		{ID: "a", Score: 2},
	})

	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"a", "b", "c"}) {
		t.Fatalf("RankedIDs() = %#v, want deterministic score order with ID tie-break", ids)
	}
	if got[0].Rank != 1 || got[0].ScoreSpace != ScoreSpaceProvider {
		t.Fatalf("top result = %#v, want rank and provider score-space assigned", got[0])
	}
}

func TestFromRanks_UsesRankOrderScoreSpace(t *testing.T) {
	t.Parallel()

	got := FromRanks([]string{"doc-c", "doc-a", "doc-b"}, "sqlite_fts5")
	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"doc-c", "doc-a", "doc-b"}) {
		t.Fatalf("RankedIDs() = %#v, want original rank order", ids)
	}
	if got[0].Score <= got[1].Score || got[1].Score <= got[2].Score {
		t.Fatalf("scores = %#v, want monotonically decreasing rank scores", got)
	}
	for i, item := range got {
		if item.Rank != i+1 {
			t.Fatalf("Rank = %d, want %d", item.Rank, i+1)
		}
		if item.Source != "sqlite_fts5" || item.ScoreSpace != ScoreSpaceRankOnly {
			t.Fatalf("item = %#v, want sqlite rank-only result", item)
		}
	}
}

func TestRankByProviderRank_SortsRankedRowsAndPreservesScoreSpace(t *testing.T) {
	t.Parallel()

	got := RankByProviderRank([]Result{
		{ID: "doc-b", Rank: 2, Score: 0.2, Metadata: map[string]string{"table": "docs"}},
		{ID: "doc-a", Rank: 1, Score: 0.5},
		{ID: "doc-c", Score: 0.1},
	}, "postgres_fts", ScoreSpaceProvider)

	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"doc-a", "doc-b", "doc-c"}) {
		t.Fatalf("RankedIDs() = %#v, want provider rank order", ids)
	}
	if got[1].Score != 0.2 || got[1].ScoreSpace != ScoreSpaceProvider || got[1].Metadata["table"] != "docs" {
		t.Fatalf("provider result = %#v, want score-space and metadata preserved", got[1])
	}
}

func TestFuseRRFByID_OverlappingLexicalAndVectorLists(t *testing.T) {
	t.Parallel()

	got := FuseRRFByID([]FusionInput{
		{
			Source: "bm25",
			Candidates: RankedList{
				{ID: "doc-a"},
				{ID: "doc-b"},
				{ID: "doc-c"},
			},
		},
		{
			Source: "vector",
			Candidates: RankedList{
				{ID: "doc-b"},
				{ID: "doc-a"},
				{ID: "doc-d"},
			},
		},
	}, FusionOptions{K: 60})

	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"doc-a", "doc-b", "doc-c", "doc-d"}) {
		t.Fatalf("RankedIDs() = %#v, want stable fused order", ids)
	}
	if got[0].ScoreSpace != ScoreSpaceRRF || got[0].Rank != 1 {
		t.Fatalf("top fused result = %#v, want RRF score-space and rank", got[0])
	}
}

func TestFuseRRFByID_DuplicatesMissingCandidatesAndLimit(t *testing.T) {
	t.Parallel()

	got := FuseRRFByID([]FusionInput{
		{
			Source: "bm25",
			Candidates: RankedList{
				{ID: "doc-a"},
				{ID: "doc-a"},
				{ID: ""},
				{ID: "doc-b"},
			},
		},
	}, FusionOptions{K: 1, Limit: 1})

	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"doc-a"}) {
		t.Fatalf("RankedIDs() = %#v, want duplicate ID counted once and limit applied", ids)
	}
}

func TestFuseRRFByID_StableTieOrdering(t *testing.T) {
	t.Parallel()

	got := FuseRRFByID([]FusionInput{
		{Candidates: RankedList{{ID: "b"}}},
		{Candidates: RankedList{{ID: "a"}}},
	}, FusionOptions{K: 60})

	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"a", "b"}) {
		t.Fatalf("RankedIDs() = %#v, want ID tie-break", ids)
	}
}

func TestRankedIndices_RRFInteroperabilityAndSkippedUnknowns(t *testing.T) {
	t.Parallel()

	lexicalRanks := FromRanks([]string{"doc-a", "doc-missing", "doc-b", ""}, "bm25")
	vectorRanks := []int{1, 2}
	indexByID := map[string]int{
		"doc-a": 1,
		"doc-b": 2,
	}

	indices, report := RankedIndicesWithReport(lexicalRanks, indexByID)
	fused, _ := vectors.RRF([][]int{
		indices,
		vectorRanks,
	}, 60)

	if !reflect.DeepEqual(indices, []int{1, 2}) {
		t.Fatalf("indices = %#v, want unknown and empty IDs skipped", indices)
	}
	if report.SkippedUnknown != 1 || !reflect.DeepEqual(report.UnknownIDs, []string{"doc-missing"}) {
		t.Fatalf("report = %#v, want skipped unknown ID", report)
	}
	if len(fused) != 2 || fused[0].Index != 1 || fused[1].Index != 2 {
		t.Fatalf("fused indices = %#v, want lexical/vector compatible rank order", fused)
	}
}

func TestLexicalVectorRetrievalInterop(t *testing.T) {
	t.Parallel()

	lexicalRanks := FromRanks([]string{"doc-a", "doc-b", "doc-c"}, "sqlite_fts5")
	vectorRanks := FromRanks([]string{"doc-b", "doc-a", "doc-d"}, "vector_exact")
	fused := FuseRRFByID([]FusionInput{
		{Source: "sqlite_fts5", Candidates: lexicalRanks},
		{Source: "vector_exact", Candidates: vectorRanks},
	}, FusionOptions{K: 60})

	report := retrieval.EvaluateLayers(retrieval.EvalQuery{
		ID: "q1",
		Relevant: map[string]float64{
			"doc-a": 1,
			"doc-b": 1,
		},
	}, retrieval.LayeredResults{
		Candidates: ToRetrievalResults(lexicalRanks),
		Reranked:   ToRetrievalResults(vectorRanks),
		Final:      ToRetrievalResults(fused),
	}, 2)

	if report.FinalMetrics.RecallAtK != 1 {
		t.Fatalf("Final RecallAtK = %f, want lexical/vector fused ranks to evaluate through retrieval", report.FinalMetrics.RecallAtK)
	}
}

func TestRankByScore_NaNStrictWeakOrdering(t *testing.T) {
	t.Parallel()

	items := []Candidate{
		{ID: "nan-b", Score: math.NaN()},
		{ID: "mid", Score: 1.5},
		{ID: "neg-inf", Score: math.Inf(-1)},
		{ID: "nan-a", Score: math.NaN()},
		{ID: "high", Score: 3.0},
		{ID: "pos-inf", Score: math.Inf(1)},
	}
	got := RankByScore(items)
	want := []string{"pos-inf", "high", "mid", "neg-inf", "nan-a", "nan-b"}
	if ids := RankedIDs(got); !reflect.DeepEqual(ids, want) {
		t.Fatalf("RankedIDs() = %#v, want %#v", ids, want)
	}
	for i := range items {
		if compareCandidates(items[i], items[i]) != 0 {
			t.Fatalf("compareCandidates not reflexive for %+v", items[i])
		}
		for j := range items {
			ab := compareCandidates(items[i], items[j])
			ba := compareCandidates(items[j], items[i])
			if ab != -ba {
				t.Fatalf("compareCandidates not antisymmetric for (%+v, %+v): %d vs %d", items[i], items[j], ab, ba)
			}
		}
	}
}

func TestFuseRRFByID_NonFiniteWeightsDefaultToOne(t *testing.T) {
	t.Parallel()

	for _, badWeight := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		got := FuseRRFByID([]FusionInput{
			{Source: "a", Weight: badWeight, Candidates: RankedList{{ID: "doc-1"}, {ID: "doc-2"}}},
			{Source: "b", Weight: 1.0, Candidates: RankedList{{ID: "doc-2"}, {ID: "doc-1"}}},
		}, FusionOptions{K: 60})
		want := FuseRRFByID([]FusionInput{
			{Source: "a", Weight: 1.0, Candidates: RankedList{{ID: "doc-1"}, {ID: "doc-2"}}},
			{Source: "b", Weight: 1.0, Candidates: RankedList{{ID: "doc-2"}, {ID: "doc-1"}}},
		}, FusionOptions{K: 60})
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("FuseRRFByID(weight=%v) = %#v, want %#v", badWeight, got, want)
		}
	}
}
