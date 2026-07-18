// Command retrieval-calibration-tune demonstrates the retrieval quality-control
// layer: captured runs, threshold checks, score-reference calibration, and
// deterministic weight tuning.
//
// Run from the repo root:
//
//	go run ./examples/retrieval-calibration-tune
package main

import (
	"fmt"

	"github.com/dotcommander/reliquary/retrieval"
)

func main() {
	fixture := retrieval.Fixture{
		ID: "runtime-search-golden",
		Queries: []retrieval.FixtureQuery{
			{
				ID:   "go-memory",
				Text: "how does Go reclaim memory",
				Judgments: []retrieval.Judgment{
					{DocID: "go-gc", Relevance: 3, Topic: "go"},
					{DocID: "go-concurrency", Relevance: 1, Topic: "go"},
					{DocID: "rust-ownership", Relevance: 1, Topic: "rust"},
				},
			},
			{
				ID:   "rust-safety",
				Text: "how does Rust enforce memory safety",
				Judgments: []retrieval.Judgment{
					{DocID: "rust-ownership", Relevance: 3, Topic: "rust"},
					{DocID: "go-gc", Relevance: 1, Topic: "go"},
				},
			},
		},
	}

	run := retrieval.Run{
		ID: "candidate-a",
		Queries: []retrieval.RunQuery{
			{
				ID: "rust-safety",
				Results: []retrieval.RankedResult{
					{ID: "rust-ownership", Score: 0.94, Topic: "rust"},
					{ID: "go-gc", Score: 0.42, Topic: "go"},
					{ID: "python-asyncio", Score: 0.20, Topic: "python"},
				},
				Stages: retrieval.LayeredResults{
					Candidates: []retrieval.RankedResult{
						{ID: "rust-ownership", Score: 0.80, Topic: "rust"},
						{ID: "go-gc", Score: 0.32, Topic: "go"},
						{ID: "python-asyncio", Score: 0.20, Topic: "python"},
					},
					Final: []retrieval.RankedResult{
						{ID: "rust-ownership", Score: 0.94, Topic: "rust"},
						{ID: "go-gc", Score: 0.42, Topic: "go"},
						{ID: "python-asyncio", Score: 0.20, Topic: "python"},
					},
				},
			},
			{
				ID: "go-memory",
				Results: []retrieval.RankedResult{
					{ID: "go-gc", Score: 0.96, Topic: "go"},
					{ID: "go-concurrency", Score: 0.53, Topic: "go"},
					{ID: "rust-ownership", Score: 0.40, Topic: "rust"},
				},
				Stages: retrieval.LayeredResults{
					Candidates: []retrieval.RankedResult{
						{ID: "go-concurrency", Score: 0.61, Topic: "go"},
						{ID: "go-gc", Score: 0.58, Topic: "go"},
						{ID: "rust-ownership", Score: 0.30, Topic: "rust"},
					},
					Final: []retrieval.RankedResult{
						{ID: "go-gc", Score: 0.96, Topic: "go"},
						{ID: "go-concurrency", Score: 0.53, Topic: "go"},
						{ID: "rust-ownership", Score: 0.40, Topic: "rust"},
					},
				},
			},
		},
	}

	report, err := retrieval.EvaluateRun(fixture, run, 3)
	if err != nil {
		panic(err)
	}
	failures := retrieval.CheckThresholds(report, retrieval.ReportThresholds{
		MinRecallAtK:    0.9,
		MinPrecisionAtK: 0.5,
		MinMRR:          0.9,
		MinNDCGAtK:      0.9,
	})
	fmt.Printf("fixture=%s run=%s queries=%d recall=%.3f ndcg=%.3f threshold_failures=%d\n",
		report.FixtureID, report.RunID, report.QueryCount, report.Metrics.RecallAtK, report.Metrics.NDCGAtK, len(failures))
	for _, q := range report.Queries {
		fmt.Printf("query=%s final_recall=%.3f candidate_recall=%.3f segments=%d\n",
			q.ID, q.Layers.FinalMetrics.RecallAtK, q.Layers.CandidateRecall, len(q.Segments))
	}

	identity := retrieval.ScoreReferenceIdentity{
		ScoreVersion: "retrieval-v1",
		ModelID:      "demo-embedder-v1",
		SchemaHash:   "docs-v1",
		ConfigHash:   "weights-v1",
	}
	reference, err := retrieval.FitScoreReference(referenceScores(), identity)
	if err != nil {
		panic(err)
	}
	if err := reference.Validate(identity); err != nil {
		panic(err)
	}

	scorer := retrieval.NewScorer(retrieval.DefaultWeights())
	candidates := []*retrieval.Result{
		{
			ID:        "go-gc",
			Content:   "The Go runtime reclaims unreachable objects with a concurrent garbage collector.",
			Filename:  "go-memory.md",
			Embedding: []float64{0.98, 0.10, 0.00},
		},
		{
			ID:        "rust-ownership",
			Content:   "Rust checks ownership and borrowing at compile time.",
			Filename:  "rust-safety.md",
			Embedding: []float64{0.20, 0.90, 0.00},
		},
	}
	ranked := scorer.RerankWithReference([]float64{1, 0, 0}, "go garbage collector memory", candidates, reference)
	fmt.Printf("calibrated_top=%s percentile=%.3f reference_samples=%d\n", ranked[0].ID, ranked[0].CombinedScore, len(reference.Values))

	tuned := retrieval.TuneWeights(tuneCases(), retrieval.TuneConfig{
		K: 2,
		Weights: []retrieval.Weights{
			{Embedding: 0.80, Keyword: 0.10, Filename: 0.10},
			{Embedding: 0.45, Keyword: 0.45, Filename: 0.10},
		},
		MMRLambdas: []float64{0.4, 0.8},
		Constraints: retrieval.TuneConstraints{
			MinRecallAtK:       0.9,
			MinNDCGAtK:         0.8,
			MinUniqueTopicsAtK: 1,
		},
	})
	fmt.Printf("tuned configs=%d has_best=%v best_embedding=%.2f best_keyword=%.2f mmr=%.1f recall=%.3f\n",
		len(tuned.Results), tuned.HasBest, tuned.Best.Weights.Embedding, tuned.Best.Weights.Keyword, tuned.Best.MMRLambda, tuned.Best.Metrics.RecallAtK)
}

func referenceScores() []float64 {
	return []float64{
		0.05, 0.08, 0.11, 0.13, 0.18,
		0.21, 0.25, 0.30, 0.34, 0.39,
		0.44, 0.48, 0.53, 0.57, 0.61,
		0.66, 0.71, 0.77, 0.83, 0.88,
		0.92, 0.96,
	}
}

func tuneCases() []retrieval.TuneCase {
	return []retrieval.TuneCase{
		{
			Query: retrieval.EvalQuery{
				ID: "go-memory",
				Relevant: map[string]float64{
					"go-gc":          3,
					"go-concurrency": 1,
				},
				TopicByDoc: map[string]string{
					"go-gc":          "go",
					"go-concurrency": "go",
					"rust-ownership": "rust",
				},
			},
			Candidates: []retrieval.TuneCandidate{
				{ID: "go-gc", Signals: retrieval.ScoreSignals{Embedding: 0.95, Keyword: 0.80, Filename: 0.70}, Embedding: []float64{1, 0, 0}, Topic: "go"},
				{ID: "go-concurrency", Signals: retrieval.ScoreSignals{Embedding: 0.70, Keyword: 0.60, Filename: 0.20}, Embedding: []float64{0.8, 0.2, 0}, Topic: "go"},
				{ID: "rust-ownership", Signals: retrieval.ScoreSignals{Embedding: 0.30, Keyword: 0.45, Filename: 0.10}, Embedding: []float64{0.1, 0.9, 0}, Topic: "rust"},
			},
		},
		{
			Query: retrieval.EvalQuery{
				ID: "rust-safety",
				Relevant: map[string]float64{
					"rust-ownership": 3,
					"go-gc":          1,
				},
				TopicByDoc: map[string]string{
					"rust-ownership": "rust",
					"go-gc":          "go",
					"python-asyncio": "python",
				},
			},
			Candidates: []retrieval.TuneCandidate{
				{ID: "rust-ownership", Signals: retrieval.ScoreSignals{Embedding: 0.93, Keyword: 0.92, Filename: 0.80}, Embedding: []float64{0, 1, 0}, Topic: "rust"},
				{ID: "go-gc", Signals: retrieval.ScoreSignals{Embedding: 0.50, Keyword: 0.35, Filename: 0.20}, Embedding: []float64{0.7, 0.2, 0}, Topic: "go"},
				{ID: "python-asyncio", Signals: retrieval.ScoreSignals{Embedding: 0.20, Keyword: 0.30, Filename: 0.10}, Embedding: []float64{0, 0, 1}, Topic: "python"},
			},
		},
	}
}
