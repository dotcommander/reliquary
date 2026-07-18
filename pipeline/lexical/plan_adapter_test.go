package lexical

import (
	"testing"

	"github.com/dotcommander/reliquary/retrieval"
)

func TestToCandidateSourceReportFillsSourceAndScoreSpace(t *testing.T) {
	t.Parallel()

	list := RankedList{
		{ID: "a", Score: 2, Source: "lexical", ScoreSpace: ScoreSpaceLocalBM25},
		{ID: "b", Score: 1, Source: "lexical", ScoreSpace: ScoreSpaceLocalBM25},
	}
	got := ToCandidateSourceReport(retrieval.CandidateSource{Limit: 10}, list, map[string]string{"a": "topic"})
	if got.Source.ID != "lexical" {
		t.Fatalf("source ID = %q, want lexical", got.Source.ID)
	}
	if got.Source.ScoreSpace != string(ScoreSpaceLocalBM25) {
		t.Fatalf("score space = %q, want %q", got.Source.ScoreSpace, ScoreSpaceLocalBM25)
	}
	if len(got.Results) != 2 || got.Results[0].Topic != "topic" {
		t.Fatalf("results = %#v", got.Results)
	}
}
