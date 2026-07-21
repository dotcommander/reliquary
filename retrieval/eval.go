package retrieval

import (
	"cmp"
	"math"
	"slices"
)

// EvalQuery describes expected relevant documents for one retrieval query.
type EvalQuery struct {
	ID         string
	Relevant   map[string]float64
	TopicByDoc map[string]string
}

// RankedResult is a scored retrieval result for metric evaluation.
type RankedResult struct {
	ID    string
	Score float64
	Topic string
}

// LayeredResults carries ranked result lists captured at each retrieval stage.
// Empty slices are valid and report zero metrics for that layer.
type LayeredResults struct {
	Candidates  []RankedResult
	Reranked    []RankedResult
	Diversified []RankedResult
	Final       []RankedResult
}

// LayerReport separates candidate generation, reranking, diversification, and
// final top-k quality so retrieval regressions can be localized to one stage.
type LayerReport struct {
	RelevantCount       int
	CandidateCount      int
	CandidateHitCount   int
	CandidateRecall     float64
	CandidateMetrics    Metrics
	RerankMetrics       Metrics
	DiversifiedMetrics  Metrics
	FinalMetrics        Metrics
	DiversityLiftAtK    int
	FinalDeltaRecallAtK float64
}

// Segmenter maps a document ID to a caller-owned evaluation segment. Returning
// an empty string excludes the document from segment summaries.
type Segmenter func(docID string) string

// SegmentMetrics summarizes retrieval quality for one segment. RelevantCount
// counts relevant documents assigned to the segment, ResultCount counts top-k
// results assigned to the segment, and HitCount counts top-k results that are
// relevant within the segment.
type SegmentMetrics struct {
	Segment       string
	Metrics       Metrics
	RelevantCount int
	ResultCount   int
	HitCount      int
}

// Metrics summarizes retrieval quality for a ranked result list.
type Metrics struct {
	RecallAtK      float64
	PrecisionAtK   float64
	MRR            float64
	NDCGAtK        float64
	UniqueTopicAtK int
}

// EvaluateLayers evaluates retrieval outputs captured at each stage.
// CandidateRecall is computed across the full candidate set, independent of k;
// the layer Metrics fields use Evaluate with the provided k. DiversityLiftAtK is
// Diversified.UniqueTopicAtK minus Rerank.UniqueTopicAtK, and
// FinalDeltaRecallAtK is Final.RecallAtK minus CandidateMetrics.RecallAtK.
func EvaluateLayers(query EvalQuery, layers LayeredResults, k int) LayerReport {
	candidates := canonicalRankedResults(layers.Candidates)
	if k <= 0 {
		return LayerReport{RelevantCount: len(query.Relevant), CandidateCount: len(candidates)}
	}
	candidateMetrics := Evaluate(query, candidates, k)
	rerankMetrics := Evaluate(query, layers.Reranked, k)
	diversifiedMetrics := Evaluate(query, layers.Diversified, k)
	finalMetrics := Evaluate(query, layers.Final, k)
	candidateHits := hitCount(query.Relevant, candidates)
	candidateRecall := 0.0
	if len(query.Relevant) > 0 {
		candidateRecall = float64(candidateHits) / float64(len(query.Relevant))
	}
	return LayerReport{
		RelevantCount:       len(query.Relevant),
		CandidateCount:      len(candidates),
		CandidateHitCount:   candidateHits,
		CandidateRecall:     candidateRecall,
		CandidateMetrics:    candidateMetrics,
		RerankMetrics:       rerankMetrics,
		DiversifiedMetrics:  diversifiedMetrics,
		FinalMetrics:        finalMetrics,
		DiversityLiftAtK:    diversifiedMetrics.UniqueTopicAtK - rerankMetrics.UniqueTopicAtK,
		FinalDeltaRecallAtK: finalMetrics.RecallAtK - candidateMetrics.RecallAtK,
	}
}

// EvaluateSegments evaluates one query by caller-provided document segments. It
// returns segments in deterministic lexical order. Segment metrics are computed
// by filtering the query relevance set and top-k result list to each segment,
// then applying Evaluate with the same k.
func EvaluateSegments(query EvalQuery, results []RankedResult, k int, segmenter Segmenter) []SegmentMetrics {
	if k <= 0 || segmenter == nil {
		return nil
	}
	results = canonicalRankedResults(results)
	limit := min(k, len(results))
	relevantBySegment := make(map[string]map[string]float64)
	for id, relevance := range query.Relevant {
		segment := segmenter(id)
		if segment == "" {
			continue
		}
		if relevantBySegment[segment] == nil {
			relevantBySegment[segment] = make(map[string]float64)
		}
		relevantBySegment[segment][id] = relevance
	}

	resultsBySegment := make(map[string][]RankedResult)
	for _, result := range results[:limit] {
		segment := segmenter(result.ID)
		if segment == "" {
			continue
		}
		resultsBySegment[segment] = append(resultsBySegment[segment], result)
	}

	segmentSet := make(map[string]struct{}, len(relevantBySegment)+len(resultsBySegment))
	for segment := range relevantBySegment {
		segmentSet[segment] = struct{}{}
	}
	for segment := range resultsBySegment {
		segmentSet[segment] = struct{}{}
	}
	if len(segmentSet) == 0 {
		return nil
	}

	segments := make([]string, 0, len(segmentSet))
	for segment := range segmentSet {
		segments = append(segments, segment)
	}
	slices.Sort(segments)

	report := make([]SegmentMetrics, 0, len(segments))
	for _, segment := range segments {
		segmentQuery := EvalQuery{
			ID:         query.ID,
			Relevant:   relevantBySegment[segment],
			TopicByDoc: query.TopicByDoc,
		}
		segmentResults := resultsBySegment[segment]
		report = append(report, SegmentMetrics{
			Segment:       segment,
			Metrics:       Evaluate(segmentQuery, segmentResults, k),
			RelevantCount: len(relevantBySegment[segment]),
			ResultCount:   len(segmentResults),
			HitCount:      hitCount(segmentQuery.Relevant, segmentResults),
		})
	}
	return report
}

func Evaluate(query EvalQuery, results []RankedResult, k int) Metrics {
	if k <= 0 {
		return Metrics{}
	}
	results = canonicalRankedResults(results)
	limit := min(k, len(results))
	if len(query.Relevant) == 0 {
		return Metrics{UniqueTopicAtK: uniqueTopics(results[:limit], query.TopicByDoc)}
	}

	hits := 0
	var reciprocalRank float64
	var dcg float64
	for i, result := range results[:limit] {
		rel := query.Relevant[result.ID]
		if rel <= 0 {
			continue
		}
		hits++
		if reciprocalRank == 0 {
			reciprocalRank = 1 / float64(i+1)
		}
		dcg += gain(rel) / math.Log2(float64(i+2))
	}

	ideal := idealDCG(query.Relevant, k)
	ndcg := 0.0
	if ideal > 0 {
		ndcg = dcg / ideal
	}

	return Metrics{
		RecallAtK:      float64(hits) / float64(len(query.Relevant)),
		PrecisionAtK:   float64(hits) / float64(k),
		MRR:            reciprocalRank,
		NDCGAtK:        ndcg,
		UniqueTopicAtK: uniqueTopics(results[:limit], query.TopicByDoc),
	}
}

func uniqueTopics(results []RankedResult, topicByDoc map[string]string) int {
	seen := make(map[string]bool)
	for _, result := range results {
		topic := result.Topic
		if topic == "" {
			topic = topicByDoc[result.ID]
		}
		if topic != "" {
			seen[topic] = true
		}
	}
	return len(seen)
}

func gain(relevance float64) float64 {
	return math.Pow(2, relevance) - 1
}

func idealDCG(relevant map[string]float64, k int) float64 {
	values := make([]float64, 0, len(relevant))
	for _, relevance := range relevant {
		if relevance > 0 {
			values = append(values, relevance)
		}
	}
	sortDesc(values)
	limit := min(k, len(values))
	total := 0.0
	for i, value := range values[:limit] {
		total += gain(value) / math.Log2(float64(i+2))
	}
	return total
}

func sortDesc(values []float64) {
	slices.SortFunc(values, func(a, b float64) int {
		return cmp.Compare(b, a)
	})
}

func hitCount(relevant map[string]float64, results []RankedResult) int {
	if len(relevant) == 0 {
		return 0
	}
	hits := 0
	for _, result := range canonicalRankedResults(results) {
		if relevant[result.ID] > 0 {
			hits++
		}
	}
	return hits
}

// canonicalRankedResults retains the first occurrence of each stable result ID.
func canonicalRankedResults(results []RankedResult) []RankedResult {
	if len(results) == 0 {
		return nil
	}
	canonical := make([]RankedResult, 0, len(results))
	seen := make(map[string]struct{}, len(results))
	for _, result := range results {
		if _, exists := seen[result.ID]; exists {
			continue
		}
		seen[result.ID] = struct{}{}
		canonical = append(canonical, result)
	}
	return canonical
}
