package retrieval

import (
	"math"
	"testing"
)

const floatTol = 1e-5

func approxEqual(t *testing.T, label string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > floatTol {
		t.Errorf("%s: got %v, want %v (diff %v)", label, got, want, math.Abs(got-want))
	}
}

// TestEvaluate covers Recall@K, Precision@K, MRR, NDCG@K hand-computed cases.
//
// NDCG formula in eval.go: gain(rel)/log2(i+2) where i is 0-based rank.
// gain(1.0) = 2^1 - 1 = 1.0
// IDCG for a single rel-1.0 doc: 1.0/log2(2) = 1.0
// DCG for that doc at rank 2 (i=1): 1.0/log2(3) ≈ 0.63093
// → NDCG ≈ 0.63093
//
// PrecisionAtK = hits/k (not hits/limit), as in eval.go line 60.
func TestEvaluate(t *testing.T) {
	t.Parallel()

	log2_3 := math.Log2(3) // ≈ 1.58496

	// IDCG for {a:1, b:1, c:1} at k=2: top-2 ideal = [1,1]
	// IDCG = gain(1)/log2(2) + gain(1)/log2(3) = 1.0 + 1/log2(3) = 1 + 1/log2_3
	// DCG for result [a-rel, x-norel] at k=2: gain(1)/log2(2) = 1.0
	// NDCG = 1.0 / (1.0 + 1/log2_3)
	ndcg3relOf2 := 1.0 / (1.0 + 1.0/log2_3)

	tests := []struct {
		name    string
		query   EvalQuery
		results []RankedResult
		k       int
		want    Metrics
	}{
		{
			name: "3 relevant docs, 1 of top-2 relevant: Recall@2=1/3, Precision@2=1/2",
			query: EvalQuery{
				ID:       "q1",
				Relevant: map[string]float64{"a": 1, "b": 1, "c": 1},
			},
			results: []RankedResult{
				{ID: "a", Score: 0.9}, // relevant
				{ID: "x", Score: 0.8}, // not relevant
				{ID: "b", Score: 0.7}, // relevant (outside top-2)
			},
			k: 2,
			want: Metrics{
				RecallAtK:    1.0 / 3.0,
				PrecisionAtK: 1.0 / 2.0,
				MRR:          1.0, // first result relevant → 1/(1)
				NDCGAtK:      ndcg3relOf2,
			},
		},
		{
			name: "first result relevant: MRR=1.0",
			query: EvalQuery{
				ID:       "q2",
				Relevant: map[string]float64{"doc1": 1},
			},
			results: []RankedResult{
				{ID: "doc1", Score: 0.9},
				{ID: "doc2", Score: 0.5},
			},
			k: 2,
			want: Metrics{
				RecallAtK:    1.0,
				PrecisionAtK: 0.5, // 1 hit / k=2
				MRR:          1.0,
				NDCGAtK:      1.0,
			},
		},
		{
			name: "first relevant at rank 2: MRR=0.5",
			query: EvalQuery{
				ID:       "q3",
				Relevant: map[string]float64{"doc2": 1},
			},
			results: []RankedResult{
				{ID: "doc1", Score: 0.9}, // not relevant
				{ID: "doc2", Score: 0.8}, // relevant at rank 2 (i=1)
			},
			k: 2,
			want: Metrics{
				RecallAtK:    1.0,
				PrecisionAtK: 0.5,
				MRR:          0.5,
				// DCG = 1.0/log2(3); IDCG = 1.0/log2(2)=1.0 → NDCG = 1/log2(3)
				NDCGAtK: 1.0 / log2_3,
			},
		},
		{
			name: "k <= 0 returns zero Metrics",
			query: EvalQuery{
				ID:       "q4",
				Relevant: map[string]float64{"doc1": 1},
			},
			results: []RankedResult{{ID: "doc1", Score: 0.9}},
			k:       0,
			want:    Metrics{},
		},
		{
			name: "empty Relevant set returns zero numeric metrics",
			query: EvalQuery{
				ID:       "q5",
				Relevant: map[string]float64{},
			},
			results: []RankedResult{
				{ID: "doc1", Score: 0.9, Topic: "t1"},
				{ID: "doc2", Score: 0.8, Topic: "t2"},
			},
			k: 2,
			// RecallAtK/PrecisionAtK/MRR/NDCGAtK all 0; UniqueTopicAtK counted
			want: Metrics{UniqueTopicAtK: 2},
		},
		{
			name: "TopicByDoc fills missing RankedResult topics",
			query: EvalQuery{
				ID:         "q-topic-map",
				Relevant:   map[string]float64{"doc1": 1, "doc2": 1},
				TopicByDoc: map[string]string{"doc1": "retrieval", "doc2": "scoring"},
			},
			results: []RankedResult{
				{ID: "doc1", Score: 0.9},
				{ID: "doc2", Score: 0.8},
			},
			k: 2,
			want: Metrics{
				RecallAtK:      1.0,
				PrecisionAtK:   1.0,
				MRR:            1.0,
				NDCGAtK:        1.0,
				UniqueTopicAtK: 2,
			},
		},
		{
			name: "k > len(results): clamped, no panic",
			query: EvalQuery{
				ID:       "q6",
				Relevant: map[string]float64{"doc1": 1},
			},
			results: []RankedResult{{ID: "doc1", Score: 0.9}},
			k:       100,
			want: Metrics{
				RecallAtK:    1.0,
				PrecisionAtK: 1.0 / 100.0, // hits/k not hits/limit
				MRR:          1.0,
				NDCGAtK:      1.0,
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := Evaluate(tc.query, tc.results, tc.k)
			approxEqual(t, "RecallAtK", got.RecallAtK, tc.want.RecallAtK)
			approxEqual(t, "PrecisionAtK", got.PrecisionAtK, tc.want.PrecisionAtK)
			approxEqual(t, "MRR", got.MRR, tc.want.MRR)
			approxEqual(t, "NDCGAtK", got.NDCGAtK, tc.want.NDCGAtK)
			if got.UniqueTopicAtK != tc.want.UniqueTopicAtK {
				t.Errorf("UniqueTopicAtK: got %d, want %d", got.UniqueTopicAtK, tc.want.UniqueTopicAtK)
			}
		})
	}
}

func TestEvaluateRecallMonotonicAsKIncreases(t *testing.T) {
	t.Parallel()

	query := EvalQuery{
		ID:       "q-monotonic",
		Relevant: map[string]float64{"rel-a": 1, "rel-b": 1, "rel-c": 1},
	}
	results := []RankedResult{
		{ID: "noise-a", Score: 0.99},
		{ID: "rel-a", Score: 0.90},
		{ID: "noise-b", Score: 0.80},
		{ID: "rel-b", Score: 0.70},
		{ID: "rel-c", Score: 0.60},
	}

	var previousRecall float64
	for k := 1; k <= len(results); k++ {
		got := Evaluate(query, results, k)
		if got.RecallAtK+floatTol < previousRecall {
			t.Fatalf("RecallAtK decreased at k=%d: got %v after %v", k, got.RecallAtK, previousRecall)
		}
		if got.RecallAtK < 0 || got.RecallAtK > 1 {
			t.Fatalf("RecallAtK at k=%d = %v, want in [0,1]", k, got.RecallAtK)
		}
		previousRecall = got.RecallAtK
	}
}

func TestEvaluateLayersLocalizesRetrievalStages(t *testing.T) {
	t.Parallel()

	query := EvalQuery{
		ID:       "q-layers",
		Relevant: map[string]float64{"rel-a": 1, "rel-b": 1, "rel-c": 1},
		TopicByDoc: map[string]string{
			"rel-a": "alpha",
			"rel-b": "beta",
			"rel-c": "gamma",
			"noise": "alpha",
		},
	}
	layers := LayeredResults{
		Candidates: []RankedResult{
			{ID: "rel-a", Score: 0.4},
			{ID: "noise", Score: 0.3},
			{ID: "rel-b", Score: 0.2},
		},
		Reranked: []RankedResult{
			{ID: "noise", Score: 0.9},
			{ID: "rel-a", Score: 0.8},
			{ID: "rel-b", Score: 0.7},
		},
		Diversified: []RankedResult{
			{ID: "rel-a", Score: 0.8},
			{ID: "rel-b", Score: 0.7},
			{ID: "noise", Score: 0.6},
		},
		Final: []RankedResult{
			{ID: "rel-a", Score: 0.8},
			{ID: "rel-b", Score: 0.7},
		},
	}

	got := EvaluateLayers(query, layers, 2)
	if got.RelevantCount != 3 || got.CandidateCount != 3 || got.CandidateHitCount != 2 {
		t.Fatalf("counts = relevant %d candidates %d hits %d, want 3/3/2", got.RelevantCount, got.CandidateCount, got.CandidateHitCount)
	}
	approxEqual(t, "CandidateRecall", got.CandidateRecall, 2.0/3.0)
	approxEqual(t, "CandidateMetrics RecallAtK", got.CandidateMetrics.RecallAtK, 1.0/3.0)
	approxEqual(t, "RerankMetrics MRR", got.RerankMetrics.MRR, 0.5)
	approxEqual(t, "FinalMetrics RecallAtK", got.FinalMetrics.RecallAtK, 2.0/3.0)
	approxEqual(t, "FinalDeltaRecallAtK", got.FinalDeltaRecallAtK, 1.0/3.0)
	if got.DiversityLiftAtK != 1 {
		t.Fatalf("DiversityLiftAtK = %d, want 1", got.DiversityLiftAtK)
	}
}

func TestEvaluateLayersEdgeCases(t *testing.T) {
	t.Parallel()

	query := EvalQuery{ID: "q-empty", Relevant: map[string]float64{"rel": 1}}
	got := EvaluateLayers(query, LayeredResults{Candidates: []RankedResult{{ID: "rel", Score: 1}}}, 0)
	if got.RelevantCount != 1 || got.CandidateCount != 1 {
		t.Fatalf("k=0 counts = relevant %d candidates %d, want 1/1", got.RelevantCount, got.CandidateCount)
	}
	if got.CandidateRecall != 0 || got.CandidateHitCount != 0 {
		t.Fatalf("k=0 candidate recall/hits = %v/%d, want 0/0", got.CandidateRecall, got.CandidateHitCount)
	}

	noRelevant := EvaluateLayers(EvalQuery{ID: "q-no-relevant"}, LayeredResults{Candidates: []RankedResult{{ID: "doc", Score: 1, Topic: "topic"}}}, 3)
	if noRelevant.RelevantCount != 0 || noRelevant.CandidateHitCount != 0 || noRelevant.CandidateRecall != 0 {
		t.Fatalf("no-relevant report = %+v, want zero relevant/hits/recall", noRelevant)
	}
	if noRelevant.CandidateMetrics.UniqueTopicAtK != 1 {
		t.Fatalf("no-relevant CandidateMetrics.UniqueTopicAtK = %d, want 1", noRelevant.CandidateMetrics.UniqueTopicAtK)
	}
}

func TestEvaluateSegments(t *testing.T) {
	t.Parallel()

	query := EvalQuery{
		ID: "q-segments",
		Relevant: map[string]float64{
			"a1": 1,
			"a2": 1,
			"b1": 1,
			"c1": 1,
		},
		TopicByDoc: map[string]string{
			"a1": "topic-a",
			"b1": "topic-b",
			"x1": "topic-x",
		},
	}
	results := []RankedResult{
		{ID: "b1", Score: 0.9},
		{ID: "x1", Score: 0.8, Topic: "topic-x"},
		{ID: "a1", Score: 0.7},
		{ID: "c1", Score: 0.6},
	}
	segmenter := func(id string) string {
		switch id {
		case "a1", "a2", "x1":
			return "alpha"
		case "b1":
			return "beta"
		case "c1":
			return "gamma"
		default:
			return ""
		}
	}

	got := EvaluateSegments(query, results, 3, segmenter)
	wants := []struct {
		segment                 string
		relevant, results, hits int
		recall, precision, mrr  float64
		uniqueTopics            int // -1 skips the UniqueTopicAtK assertion
	}{
		{segment: "alpha", relevant: 2, results: 2, hits: 1, recall: 0.5, precision: 1.0 / 3.0, mrr: 0.5, uniqueTopics: 2},
		{segment: "beta", relevant: 1, results: 1, hits: 1, recall: 1, precision: 1.0 / 3.0, mrr: 1, uniqueTopics: -1},
		{segment: "gamma", relevant: 1, results: 0, hits: 0, recall: 0, precision: 0, mrr: 0, uniqueTopics: -1},
	}
	if len(got) != len(wants) {
		t.Fatalf("EvaluateSegments len = %d, want %d; full report = %+v", len(got), len(wants), got)
	}
	for i, want := range wants {
		segment := got[i]
		if segment.Segment != want.segment {
			t.Fatalf("segment[%d] = %q, want %q; full report = %+v", i, segment.Segment, want.segment, got)
		}
		if segment.RelevantCount != want.relevant || segment.ResultCount != want.results || segment.HitCount != want.hits {
			t.Fatalf("%s counts = relevant %d results %d hits %d, want %d/%d/%d", want.segment, segment.RelevantCount, segment.ResultCount, segment.HitCount, want.relevant, want.results, want.hits)
		}
		approxEqual(t, want.segment+" RecallAtK", segment.Metrics.RecallAtK, want.recall)
		approxEqual(t, want.segment+" PrecisionAtK", segment.Metrics.PrecisionAtK, want.precision)
		approxEqual(t, want.segment+" MRR", segment.Metrics.MRR, want.mrr)
		if want.uniqueTopics >= 0 && segment.Metrics.UniqueTopicAtK != want.uniqueTopics {
			t.Fatalf("%s UniqueTopicAtK = %d, want %d", want.segment, segment.Metrics.UniqueTopicAtK, want.uniqueTopics)
		}
	}
}

func TestEvaluateSegmentsEdgeCases(t *testing.T) {
	t.Parallel()

	query := EvalQuery{ID: "q", Relevant: map[string]float64{"a": 1}}
	results := []RankedResult{{ID: "a", Score: 1}}
	segmenter := func(id string) string { return "all" }

	if got := EvaluateSegments(query, results, 0, segmenter); got != nil {
		t.Fatalf("k=0 EvaluateSegments = %+v, want nil", got)
	}
	if got := EvaluateSegments(query, results, 1, nil); got != nil {
		t.Fatalf("nil segmenter EvaluateSegments = %+v, want nil", got)
	}
	if got := EvaluateSegments(query, results, 1, func(string) string { return "" }); got != nil {
		t.Fatalf("empty segment EvaluateSegments = %+v, want nil", got)
	}

	lowN := EvaluateSegments(EvalQuery{ID: "low-n"}, []RankedResult{{ID: "z", Score: 1}}, 5, func(string) string { return "orphan" })
	if len(lowN) != 1 {
		t.Fatalf("low-N result len = %d, want 1", len(lowN))
	}
	if lowN[0].Segment != "orphan" || lowN[0].RelevantCount != 0 || lowN[0].ResultCount != 1 || lowN[0].HitCount != 0 {
		t.Fatalf("low-N segment = %+v, want orphan with relevant=0 results=1 hits=0", lowN[0])
	}
}

// TestMMREdgeCases covers boundary inputs for MMR.
func TestMMREdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("k<=0 returns nil", func(t *testing.T) {
		t.Parallel()
		items := []MMRItem{{ID: "a", Score: 0.9}}
		if got := MMR(items, 0, 0.5); got != nil {
			t.Errorf("expected nil, got %v", got)
		}
		if got := MMR(items, -1, 0.5); got != nil {
			t.Errorf("expected nil for k=-1, got %v", got)
		}
	})

	t.Run("empty items returns nil", func(t *testing.T) {
		t.Parallel()
		if got := MMR(nil, 3, 0.5); got != nil {
			t.Errorf("expected nil for nil items, got %v", got)
		}
		if got := MMR([]MMRItem{}, 3, 0.5); got != nil {
			t.Errorf("expected nil for empty items, got %v", got)
		}
	})

	t.Run("k > len(items) returns all items, no panic", func(t *testing.T) {
		t.Parallel()
		items := []MMRItem{
			{ID: "a", Score: 0.9, Embedding: []float64{1, 0}},
			{ID: "b", Score: 0.7, Embedding: []float64{0, 1}},
		}
		got := MMR(items, 100, 0.5)
		if len(got) != len(items) {
			t.Errorf("expected %d items, got %d", len(items), len(got))
		}
	})

	t.Run("lambda=0 runs without panic and returns k items", func(t *testing.T) {
		t.Parallel()
		items := []MMRItem{
			{ID: "a", Score: 0.9, Embedding: []float64{1, 0}},
			{ID: "b", Score: 0.7, Embedding: []float64{0, 1}},
			{ID: "c", Score: 0.5, Embedding: []float64{0.7, 0.7}},
		}
		got := MMR(items, 2, 0)
		if len(got) != 2 {
			t.Errorf("expected 2 items, got %d", len(got))
		}
	})

	t.Run("lambda=1 returns top-k by score order", func(t *testing.T) {
		t.Parallel()
		items := []MMRItem{
			{ID: "a", Score: 0.9, Embedding: []float64{1, 0}},
			{ID: "b", Score: 0.7, Embedding: []float64{1, 0}}, // same direction
			{ID: "c", Score: 0.5, Embedding: []float64{1, 0}},
		}
		got := MMR(items, 2, 1)
		if len(got) != 2 {
			t.Errorf("expected 2 items, got %d", len(got))
		}
		// lambda=1 → pure relevance: highest score picked first
		if got[0].ID != "a" {
			t.Errorf("expected first item to be 'a' (highest score), got %q", got[0].ID)
		}
	})
}

// TestRerankCalibrationEdgeCases covers normalizeValue/calibrateScores boundary behavior.
func TestRerankCalibrationEdgeCases(t *testing.T) {
	t.Parallel()

	scorer := NewScorerWithOptions(DefaultWeights(), false) // adaptive off: deterministic weights

	t.Run("all-equal scores produce 0.5 not NaN", func(t *testing.T) {
		t.Parallel()
		// Same embedding for every result → min==max after cosine → normalizeValue → 0.5
		emb := []float64{1, 0, 0}
		results := []*Result{
			{ID: "a", Embedding: emb, Content: "foo bar"},
			{ID: "b", Embedding: emb, Content: "foo bar"},
			{ID: "c", Embedding: emb, Content: "foo bar"},
		}
		got := scorer.Rerank(emb, "foo bar", results)
		for _, r := range got {
			if math.IsNaN(r.EmbeddingScore) {
				t.Errorf("EmbeddingScore is NaN for %q", r.ID)
			}
			if math.IsNaN(r.KeywordScore) {
				t.Errorf("KeywordScore is NaN for %q", r.ID)
			}
			if math.IsNaN(r.CombinedScore) {
				t.Errorf("CombinedScore is NaN for %q", r.ID)
			}
		}
	})

	t.Run("empty corpus returns without panic", func(t *testing.T) {
		t.Parallel()
		got := scorer.Rerank([]float64{1, 0}, "query", []*Result{})
		if len(got) != 0 {
			t.Errorf("expected empty slice, got %d items", len(got))
		}
	})

	t.Run("single item returns without panic and no NaN", func(t *testing.T) {
		t.Parallel()
		emb := []float64{0.6, 0.8}
		results := []*Result{
			{ID: "only", Embedding: emb, Content: "hello world"},
		}
		got := scorer.Rerank(emb, "hello world", results)
		if len(got) != 1 {
			t.Fatalf("expected 1 result, got %d", len(got))
		}
		r := got[0]
		if math.IsNaN(r.EmbeddingScore) || math.IsNaN(r.KeywordScore) || math.IsNaN(r.CombinedScore) {
			t.Errorf("NaN in single-item result: emb=%v kwd=%v combined=%v",
				r.EmbeddingScore, r.KeywordScore, r.CombinedScore)
		}
	})
}
