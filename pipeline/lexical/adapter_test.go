package lexical

import (
	"context"
	"reflect"
	"testing"
)

func TestSearcherFunc(t *testing.T) {
	t.Parallel()

	searcher := SearcherFunc(func(ctx context.Context, request SearchRequest) (RankedList, error) {
		if request.RawQuery != "alpha" || request.Limit != 2 {
			t.Fatalf("request = %#v, want raw query and limit forwarded", request)
		}
		return FromRanks([]string{"doc-a", "doc-b"}, "sqlite_fts5"), nil
	})

	got, err := searcher.SearchLexical(context.Background(), SearchRequest{RawQuery: "alpha", Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if ids := RankedIDs(got); !reflect.DeepEqual(ids, []string{"doc-a", "doc-b"}) {
		t.Fatalf("RankedIDs() = %#v, want searcher results", ids)
	}
}

func TestToRetrievalResultsWithTopics(t *testing.T) {
	t.Parallel()

	got := ToRetrievalResultsWithTopics(RankedList{
		{ID: "doc-a", Score: 2},
		{ID: "doc-b", Score: 1},
	}, map[string]string{"doc-b": "topic-b"})

	if got[0].ID != "doc-a" || got[0].Score != 2 || got[0].Topic != "" {
		t.Fatalf("got[0] = %#v, want lexical score adapted", got[0])
	}
	if got[1].Topic != "topic-b" {
		t.Fatalf("got[1].Topic = %q, want topic-b", got[1].Topic)
	}
}

func TestExplain(t *testing.T) {
	t.Parallel()

	got := Explain(RankedList{
		{ID: "doc-a", Score: 0.25, Source: "postgres_fts", ScoreSpace: ScoreSpaceProvider, Metadata: map[string]string{"schema": "app"}},
	})

	if len(got) != 1 {
		t.Fatalf("len(Explain()) = %d, want 1", len(got))
	}
	if got[0].Rank != 1 || got[0].Source != "postgres_fts" || got[0].ScoreSpace != ScoreSpaceProvider || got[0].Metadata["schema"] != "app" {
		t.Fatalf("Explanation = %#v, want source/rank/score-space/metadata", got[0])
	}
}

func TestSimulatedFTSAndLocalBM25Fixture(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{})
	docA := NewDocumentStatsFromTokens(analyzer.Analyze("alpha alpha beta"))
	docB := NewDocumentStatsFromTokens(analyzer.Analyze("alpha gamma"))
	docC := NewDocumentStatsFromTokens(analyzer.Analyze("gamma delta"))
	corpus := NewCorpusStats([]DocumentStats{docA, docB, docC})
	query := NormalizeQuery("alpha beta", analyzer)

	local := RankByScore([]Candidate{
		{ID: "doc-a", Score: BM25Score(query, docA, corpus, DefaultBM25Params()), Source: "local_bm25", ScoreSpace: ScoreSpaceLocalBM25},
		{ID: "doc-b", Score: BM25Score(query, docB, corpus, DefaultBM25Params()), Source: "local_bm25", ScoreSpace: ScoreSpaceLocalBM25},
		{ID: "doc-c", Score: BM25Score(query, docC, corpus, DefaultBM25Params()), Source: "local_bm25", ScoreSpace: ScoreSpaceLocalBM25},
	})
	sqlite := RankByOrder([]Result{
		{ID: "doc-a", Score: -0.7},
		{ID: "doc-b", Score: -0.2},
	}, "sqlite_fts5", ScoreSpaceRankOnly)
	postgres := RankByProviderRank([]Result{
		{ID: "doc-b", Rank: 2, Score: 0.2},
		{ID: "doc-a", Rank: 1, Score: 0.5},
	}, "postgres_fts", ScoreSpaceProvider)

	if local[0].ID != "doc-a" || local[0].ScoreSpace != ScoreSpaceLocalBM25 {
		t.Fatalf("local top = %#v, want doc-a local BM25", local[0])
	}
	if ids := RankedIDs(sqlite); !reflect.DeepEqual(ids, []string{"doc-a", "doc-b"}) {
		t.Fatalf("sqlite IDs = %#v, want rank-order-only adaptation", ids)
	}
	if sqlite[0].ScoreSpace != ScoreSpaceRankOnly || sqlite[0].Score <= sqlite[1].Score {
		t.Fatalf("sqlite ranks = %#v, want portable rank-order scores", sqlite)
	}
	if ids := RankedIDs(postgres); !reflect.DeepEqual(ids, []string{"doc-a", "doc-b"}) {
		t.Fatalf("postgres IDs = %#v, want provider rank adaptation", ids)
	}
	if postgres[0].Score != 0.5 || postgres[0].ScoreSpace != ScoreSpaceProvider {
		t.Fatalf("postgres top = %#v, want provider score preserved", postgres[0])
	}
}

func TestRankedIndicesAndReport(t *testing.T) {
	t.Parallel()

	list := RankedList{
		{ID: ""},
		{ID: "doc-a"},
		{ID: "doc-unknown"},
	}
	indexByID := map[string]int{"doc-a": 42}

	indices := RankedIndices(list, indexByID)
	if !reflect.DeepEqual(indices, []int{42}) {
		t.Fatalf("RankedIndices = %v, want [42]", indices)
	}

	_, report := RankedIndicesWithReport(list, indexByID)
	if report.SkippedEmpty != 1 || report.SkippedUnknown != 1 || report.OutputCount != 1 {
		t.Fatalf("report = %#v", report)
	}
}

func TestSearcherFuncNil(t *testing.T) {
	t.Parallel()

	var fn SearcherFunc
	got, err := fn.SearchLexical(context.Background(), SearchRequest{})
	if err != nil || len(got) != 0 {
		t.Fatalf("nil SearcherFunc returned %v, %v", got, err)
	}
}
