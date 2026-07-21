# retrieval

[![Go Reference](https://pkg.go.dev/badge/github.com/dotcommander/reliquary/retrieval.svg)](https://pkg.go.dev/github.com/dotcommander/reliquary/retrieval)
[![Go Version](https://img.shields.io/badge/go-1.26-00ADD8?logo=go)](../go.mod)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hybrid scoring and reranking for local retrieval pipelines — embedding
similarity, keyword and filename overlap, MMR diversification, and IR evaluation,
all over vectors you supply.

```go
package main

import (
	"fmt"

	"github.com/dotcommander/reliquary/retrieval"
)

func main() {
	scorer := retrieval.NewScorer(retrieval.DefaultWeights())
	docVec1 := []float64{1, 0}
	docVec2 := []float64{0, 1}
	queryEmbedding := []float64{1, 0}

	results := []*retrieval.Result{
		{ID: "doc1", Content: "machine learning fundamentals", Filename: "ml-guide.md", Embedding: docVec1},
		{ID: "doc2", Content: "cooking recipes", Filename: "recipes.md", Embedding: docVec2},
	}

	ranked := scorer.Rerank(queryEmbedding, "machine learning", results)
	fmt.Println(ranked[0].ID, ranked[0].CombinedScore)
}
```

## Install

```sh
go get github.com/dotcommander/reliquary/retrieval
```

Requires Go 1.26+.

## How it fits

```
vectors   ─── cosine similarity primitives (Cosine64)
chunking  ─── text splitting strategies (NewChunker, Chunk)
               │
               ▼
          retrieval  ─── scoring · reranking · MMR · filtering · eval
```

`retrieval` sits at the top of the local-retrieval stack. It imports `vector`
for similarity math, `chunking` for text splitting, and the provider-neutral
`embedding` contract for vector adapters. You still produce embeddings
externally — with any model or provider — and pass them in. The package imposes
no constraints on how embeddings are generated.

## What's inside

| Construct | Purpose |
|---|---|
| `Scorer` / `Rerank` | Corpus-aware batch scoring with min-max calibration |
| `CalibratedScore` / `Band` | Fixed-weight single-document scoring and tiering |
| `MMR` | Maximal Marginal Relevance — relevance vs. diversity |
| `TextChunks` / `BestChunk` | Split text and pick the best-matching chunk |
| `Filter` | Path inclusion/exclusion for retrieval scans |
| `ExtractMetadata` / `MetadataScore` | Title and heading signal from path + content |
| `Evaluate` | Recall@K, Precision@K, MRR, NDCG@K, unique-topic count |
| `Fixture` / `EvaluateRun` | Golden query judgments plus aggregate, per-query, segment, and layer reports |
| `Plan` / `PlanRun` | Provider-neutral source budgets, fusion labels, stage outputs, and per-source metrics |

Evaluate a captured run against golden judgments:

```go
fixture := retrieval.Fixture{
	ID: "golden",
	Queries: []retrieval.FixtureQuery{{
		ID:        "q1",
		Text:      "machine learning",
		Judgments: []retrieval.Judgment{{DocID: "doc1", Relevance: 2, Topic: "ml"}},
	}},
}
run := retrieval.Run{
	ID: "candidate",
	Queries: []retrieval.RunQuery{{
		ID:      "q1",
		Results: []retrieval.RankedResult{{ID: "doc1", Score: 0.91}},
	}},
}

report, err := retrieval.EvaluateRun(fixture, run, 3)
if err != nil {
	panic(err)
}
fmt.Println(report.Metrics.RecallAtK)
```

`EvaluateRun` requires the run to cover every query in the fixture and rejects
unknown query IDs. Run validation rejects blank or duplicate result IDs in the
final list and every captured stage. Lower-level metric and tuning helpers
canonicalize repeated result IDs by retaining the first occurrence.

Capture staged hybrid retrieval without encoding provider query syntax:

```go
plan := retrieval.Plan{
	ID:     "hybrid",
	Fusion: retrieval.FusionModeRRF,
	Sources: []retrieval.CandidateSource{
		{ID: "lexical", ScoreSpace: "local_bm25", Limit: 50},
		{ID: "vector", ScoreSpace: "cosine", Limit: 100},
	},
}
run := retrieval.EvaluatePlan(query, plan, layers, sourceReports, 10)
fmt.Println(run.Report.CandidateRecall)
```

## Documentation

- **[Scoring & reranking guide](docs/scoring-guide.md)** — deep reference for
  every construct above, with runnable examples.
- **[API reference](https://pkg.go.dev/github.com/dotcommander/reliquary/retrieval)** — godoc.

## License

[MIT](LICENSE) © DotCommander contributors
