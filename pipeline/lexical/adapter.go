package lexical

import (
	"context"

	"github.com/dotcommander/reliquary/retrieval"
)

// SearchRequest is the provider-neutral input for lexical adapters.
//
// Apps own query syntax, escaping, tokenizer policy, thresholds, and filters.
// RawQuery is available for SQL-backed FTS providers; Query is available when a
// caller has already normalized text through an Analyzer.
type SearchRequest struct {
	RawQuery string
	Query    Query
	Limit    int
}

// Searcher adapts an app-owned lexical backend into this package's ranked
// result contract.
type Searcher interface {
	SearchLexical(ctx context.Context, request SearchRequest) (RankedList, error)
}

// SearcherFunc adapts a function into a Searcher.
type SearcherFunc func(context.Context, SearchRequest) (RankedList, error)

// SearchLexical calls f(ctx, request).
func (f SearcherFunc) SearchLexical(ctx context.Context, request SearchRequest) (RankedList, error) {
	if f == nil {
		return RankedList{}, nil
	}
	return f(ctx, request)
}

// IndexMappingReport describes conversion from string IDs to caller-owned
// integer indices for APIs such as vectors.RRF.
type IndexMappingReport struct {
	InputCount     int
	OutputCount    int
	SkippedEmpty   int
	SkippedUnknown int
	UnknownIDs     []string
}

// RankedIndices maps ranked string IDs to integer IDs for callers such as
// vectors.RRF. Unknown and empty IDs are skipped.
func RankedIndices(list RankedList, indexByID map[string]int) []int {
	indices, _ := RankedIndicesWithReport(list, indexByID)
	return indices
}

// RankedIndicesWithReport maps ranked string IDs to integer IDs and reports
// skipped empty or unknown IDs.
func RankedIndicesWithReport(list RankedList, indexByID map[string]int) ([]int, IndexMappingReport) {
	report := IndexMappingReport{InputCount: len(list)}
	indices := make([]int, 0, len(list))
	for _, item := range list {
		if item.ID == "" {
			report.SkippedEmpty++
			continue
		}
		idx, ok := indexByID[item.ID]
		if !ok {
			report.SkippedUnknown++
			report.UnknownIDs = append(report.UnknownIDs, item.ID)
			continue
		}
		indices = append(indices, idx)
	}
	report.OutputCount = len(indices)
	return indices, report
}

// ToRetrievalResults converts lexical ranked results to retrieval metrics input.
func ToRetrievalResults(list RankedList) []retrieval.RankedResult {
	return ToRetrievalResultsWithTopics(list, nil)
}

// ToRetrievalResultsWithTopics converts lexical ranked results to retrieval
// metrics input and fills topics from caller-owned evaluation metadata.
func ToRetrievalResultsWithTopics(list RankedList, topicByID map[string]string) []retrieval.RankedResult {
	results := make([]retrieval.RankedResult, len(list))
	for i, result := range list {
		results[i] = retrieval.RankedResult{
			ID:    result.ID,
			Score: result.Score,
			Topic: topicByID[result.ID],
		}
	}
	return results
}

// ToCandidateSourceReport converts a ranked lexical list into a retrieval
// source report. Metrics are filled by retrieval.EvaluatePlan or
// retrieval.EvaluateSource.
func ToCandidateSourceReport(source retrieval.CandidateSource, list RankedList, topicByID map[string]string) retrieval.SourceReport {
	if source.ID == "" {
		for _, result := range list {
			if result.Source != "" {
				source.ID = result.Source
				break
			}
		}
	}
	if source.ScoreSpace == "" {
		for _, result := range list {
			if result.ScoreSpace != ScoreSpaceUnspecified {
				source.ScoreSpace = string(result.ScoreSpace)
				break
			}
		}
	}
	return retrieval.SourceReport{
		Source:  source,
		Results: ToRetrievalResultsWithTopics(list, topicByID),
	}
}

// Explanation is a compact, provider-neutral explanation of an adapted result.
type Explanation struct {
	ID         string
	Rank       int
	Score      float64
	Source     string
	ScoreSpace ScoreSpace
	Metadata   map[string]string
}

// Explain returns source/rank/score-space facts for a ranked list.
func Explain(list RankedList) []Explanation {
	explanations := make([]Explanation, len(list))
	for i, result := range list {
		rank := result.Rank
		if rank <= 0 {
			rank = i + 1
		}
		explanations[i] = Explanation{
			ID:         result.ID,
			Rank:       rank,
			Score:      result.Score,
			Source:     result.Source,
			ScoreSpace: result.ScoreSpace,
			Metadata:   result.Metadata,
		}
	}
	return explanations
}
