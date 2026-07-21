# Scoring & Reranking Guide

This is the deep reference for `github.com/dotcommander/reliquary/retrieval` —
every public construct, with runnable examples. For the shorter tour, see the
[README](../README.md).

The package produces no embeddings of its own. You generate `[]float64` vectors
externally — with any model, provider, or cache — and hand them in. Everything
here operates on vectors you already have.

## Contents

- [Reranking a batch](#reranking-a-batch)
- [Explaining score contributions](#explaining-score-contributions)
- [Calibrated single-document scoring](#calibrated-single-document-scoring)
- [Diversification with MMR](#diversification-with-mmr)
- [Chunking text](#chunking-text)
- [Filtering paths](#filtering-paths)
- [Extracting metadata](#extracting-metadata)
- [Evaluating quality](#evaluating-quality)

---

## Reranking a batch

`Scorer` is the primary entry point. It blends three signals — embedding
similarity, keyword overlap, and filename overlap — then runs min-max
calibration across the whole result set before sorting. Because calibration is
corpus-aware, the scores it produces are relative to *this batch*, not absolute.
Each signal is calibrated only across results where that signal is present, so
a missing embedding, content string, or filename does not distort another
result's range. Reranking also clears previously computed component and combined
scores before rescoring, which makes reusing `Result` values safe.

```go
scorer := retrieval.NewScorer(retrieval.DefaultWeights())

results := []*retrieval.Result{
	{ID: "a", Content: "...", Filename: "a.md", Embedding: embA},
	{ID: "b", Content: "...", Filename: "b.md", Embedding: embB},
}

// Rerank sorts in-place (descending CombinedScore) and returns the same slice.
ranked := scorer.Rerank(queryEmbedding, "search query", results)
```

After `Rerank`, every `Result` carries its component scores:

| Field | Signal |
|---|---|
| `EmbeddingScore` | `vectors.Cosine64(queryEmbedding, r.Embedding)` |
| `KeywordScore` | token overlap between query and content |
| `FilenameScore` | token overlap between query and filename |
| `CombinedScore` | weighted sum after min-max calibration |

**Default weights** (`DefaultWeights()`): Embedding 0.63, Keyword 0.27, Filename 0.10.

**Adaptive weights** are on by default for any scorer built with
`NewScorer(DefaultWeights())`. The mix shifts with query length, evaluated at
call time — short queries lean on keywords, long queries lean on the embedding:

| Query tokens | Embedding | Keyword | Filename |
|---|---|---|---|
| ≤ 2 | 0.45 | 0.45 | 0.10 |
| 3–5 | 0.63 | 0.27 | 0.10 |
| ≥ 6 | 0.765 | 0.135 | 0.10 |

Turn adaptive weighting off when you need stable, reproducible scores:

```go
scorer := retrieval.NewScorerWithOptions(retrieval.DefaultWeights(), false)
```

If you pass explicit non-default `Embedding`, `Keyword`, or `Filename` weights
to `NewScorer`, `Rerank` honors those weights instead of replacing them with the
adaptive mix. `Recency` and `Importance` weights are always caller-controlled.

Score a single result without calibration, compute filename overlap directly, or
resolve the adaptive weights for a known token count:

```go
score := scorer.Score(queryEmbedding, "query", result)
overlap := retrieval.FilenameOverlap("ml-guide.md", "machine learning")
w := retrieval.AdaptiveWeights(3) // 3 tokens → DefaultWeights
```

`RecencyFromAge(age, halfLife)` produces a finite score in `[0, 1]`. It treats
non-positive ages or half-lives as fully fresh and fails closed to `0` for NaN
or indeterminate infinite inputs.

## Explaining score contributions

Use `RerankWithTrace` when you need to explain why a result ranked where it did.
It runs the same scoring path as `Rerank`, sorts the same result slice in-place,
and returns one `ScoreTrace` per ranked result in the same order.

```go
ranked, traces := scorer.RerankWithTrace(queryEmbedding, "machine learning", results)

for i, trace := range traces {
	fmt.Printf("%d. %s score=%.3f embedding=%.3f keyword=%.3f filename=%.3f\n",
		i+1,
		ranked[i].ID,
		trace.CombinedScore,
		trace.Contributions.Embedding,
		trace.Contributions.Keyword,
		trace.Contributions.Filename,
	)
}
```

Each trace includes:

| Field | Description |
|---|---|
| `Raw` | pre-calibration embedding, keyword, and filename scores plus caller-supplied recency/importance |
| `Calibrated` | the values actually multiplied by weights after corpus calibration |
| `Weights` | the effective weights used for this rerank, including adaptive query-length weights when active |
| `Contributions` | calibrated signal × weight per signal |
| `Present` | which signals had usable inputs for that result |
| `QueryTokenCount` / `AdaptiveWeights` | query analysis used to select the effective text weights |

`ScoreTrace` is diagnostic output. Treat `CombinedScore` and the ranked result
order as the behavioral contract; use the per-signal fields to debug and tune
weights.

---

## Calibrated single-document scoring

`CalibratedScore` scores one document against fixed weights. Unlike
`Scorer.Rerank`, it applies no cross-corpus calibration — the weights are fixed
no matter how many other documents exist. Reach for it when you need a stable,
absolute score: thresholding, single-document decisions, or comparisons across
separate runs.

```go
score := retrieval.CalibratedScore(retrieval.ScoreComponents{
	Semantic: 0.95, // cosine similarity in [-1, 1]; normalized to [0, 1] internally
	Keyword:  0.80, // [0, 1]
	Filename: 1.0,  // [0, 1]
	Metadata: 0.60, // [0, 1]
})

band := retrieval.Band(score)
fmt.Println(score, band) // e.g. 0.881 strong
```

Fixed weights: Semantic × 0.62, Keyword × 0.18, Filename × 0.10, Metadata × 0.10.
The semantic input is mapped from `[-1, 1]` to `[0, 1]` before weighting.

`Band` maps the score to a human-readable tier:

| Band | Range |
|---|---|
| `BandWeak` | < 0.45 |
| `BandMedium` | 0.45 – 0.74 |
| `BandStrong` | ≥ 0.75 |

**Two pipelines, one decision.** Use `Scorer` when you have a batch and want
corpus-relative ranking. Use `CalibratedScore` when you have a single document
and need a stable, fixed-weight score.

---

## Score references and metric boundaries

`ScoreReference` maps combined retrieval scores onto a reference cohort so
thresholds can stay stable across runs. Fit it from scores produced by the same
retrieval semantics, then validate its identity before applying it:

```go
reference, err := retrieval.FitScoreReference(referenceScores, retrieval.ScoreReferenceIdentity{
	ScoreVersion: "retrieval-v1",
	ModelID:      "local-embedder-v1",
	SchemaHash:   "embedding-dim-1536",
	ConfigHash:   "weights-v1",
})
if err != nil {
	return err
}
if err := reference.Validate(retrieval.ScoreReferenceIdentity{
	ScoreVersion: "retrieval-v1",
	ModelID:      "local-embedder-v1",
	SchemaHash:   "embedding-dim-1536",
	ConfigHash:   "weights-v1",
}); err != nil {
	return err
}
ranked := scorer.RerankWithReference(queryEmbedding, "machine learning", candidates, reference)
```

A reference is stale when the score semantics change. Typical invalidation keys:

| Identity field | Invalidate when |
|---|---|
| `ScoreVersion` | score formula, calibration behavior, or feature meaning changes |
| `ModelID` | embedding model, reranker, provider, or model version changes |
| `SchemaHash` | vector dimensions, required signals, tokenizer schema, or result fields change |
| `ConfigHash` | weights, filters, MMR settings, boost policy, or reference-cohort selection changes |

Retrieval evaluation metrics answer ranking questions: did the right documents
appear, how early did they appear, did candidate generation preserve recall, and
which source/topic segments contributed? They do not measure numeric prediction
accuracy. Keep regression losses such as RMSE, RMSLE, MAE, MAPE, Huber loss, and
quantile/pinball loss in caller-owned evaluation code, where the target value
and invalid-input policy are explicit.

Likewise, keep business penalties outside `TuneWeights`. Tune retrieval weights
against ranking fixtures and constraints; apply downstream policy costs, price
losses, risk quantiles, or acquisition rules after retrieval has produced its
ranked candidate set.

---

## Diversification with MMR

`MMR` (Maximal Marginal Relevance) selects `k` items that balance relevance
against diversity, so near-duplicate results don't crowd out the list even when
they all score highly.

```go
items := []retrieval.MMRItem{
	{ID: "a1", Score: 0.99, Embedding: []float64{1, 0}, Topic: "ml"},
	{ID: "a2", Score: 0.97, Embedding: []float64{0.99, 0.1}, Topic: "ml"},
	{ID: "b1", Score: 0.90, Embedding: []float64{0, 1}, Topic: "systems"},
}

diverse := retrieval.MMR(items, 2, 0.5)
// Picks a1 (highest relevance), then b1 (most different from a1).
```

`lambda` controls the tradeoff:

| lambda | Behavior |
|---|---|
| `1.0` | pure relevance — same as top-k by score |
| `0.0` | pure diversity — maximizes spread, ignores relevance |
| `0.5` | equal weight — practical default |

`MMR` uses `vectors.Cosine64` for pairwise similarity. Items without embeddings
contribute zero similarity and are treated as maximally diverse. It returns `nil`
when `k ≤ 0` or `items` is empty.

---

## Chunking text

`TextChunks` splits a string into non-empty chunks, auto-selecting a strategy
from the content: `MarkdownAware` when markdown headings are present,
`SmartBoundary` otherwise (with a rune-based fallback if the chunker can't be
constructed).

```go
chunks := retrieval.TextChunks(content, 1200, 100) // size=1200 chars, overlap=100
```

`BestChunk` returns the highest-similarity chunk from a pre-embedded slice. If
`Similarity` is already set on a `ChunkResult` it trusts that value; otherwise it
recomputes via `vectors.Cosine64`.

```go
chunkResults := []retrieval.ChunkResult{
	{Text: "...", Embedding: emb1},
	{Text: "...", Embedding: emb2},
}

best := retrieval.BestChunk(queryEmbedding, chunkResults)
fmt.Println(best.Text, best.Similarity)
```

`ChunkResult` fields:

| Field | Description |
|---|---|
| `Text` | chunk text |
| `Embedding` | caller-supplied embedding |
| `Similarity` | cosine similarity to query; 0 means not yet computed |

For the document-to-result path, `ResultsFromDocuments` rejects blank document
IDs and duplicate IDs in one batch before chunking. `EmbedResults` rejects nil
results before calling the embedder, validates that a successful embedding batch
has one finite, positive-dimension vector per result, and does not mutate any
result when validation fails. `AttachEmbeddings` remains the lower-level helper:
it requires matching slice lengths but skips nil destinations.

---

## Filtering paths

`Filter` decides which file paths to include in a retrieval scan. `DefaultFilter`
excludes the usual non-content directories and extensions.

```go
f := retrieval.DefaultFilter()
f.Include("src/main.go")        // true
f.Include("vendor/lib/util.go") // false
f.Include("package-lock.json")  // false (.lock ext)
```

Default ignored directories: `.git`, `.hg`, `.svn`, `node_modules`, `vendor`,
`dist`, `build`, `__pycache__`. Default ignored extensions: `.lock`, `.sum`,
`.map`.

Restrict to an allowlist of extensions with `IncludeExts`:

```go
f := retrieval.Filter{
	IncludeExts: []string{".go", ".md"},
	IgnoreDirs:  retrieval.DefaultFilter().IgnoreDirs,
}
f.Include("docs/guide.md") // true
f.Include("main.py")       // false
```

Parse a comma-separated list — e.g. from a config value or CLI flag:

```go
exts := retrieval.ParseCSV(".go, .md, .txt") // []string{".go", ".md", ".txt"}
```

---

## Extracting metadata

`ExtractMetadata` reads a title and headings from file content. The title comes
from a `title:`/`Title:` frontmatter line, falling back to the filename stem.
Headings are every `#`-prefixed line, in document order.

```go
content := `title: Machine Learning Guide

# Introduction
# Core Concepts
`

meta := retrieval.ExtractMetadata("docs/ml-guide.md", content)
meta.Title    // "Machine Learning Guide"
meta.Headings // ["Introduction", "Core Concepts"]
meta.Path     // "docs/ml-guide.md"
```

`MetadataScore` returns the best weighted term overlap between a query and any
metadata field — it tests title (weight 1.0), headings joined (0.85), and
basename (0.65), and returns the highest. Feed it straight into `CalibratedScore`:

```go
score := retrieval.CalibratedScore(retrieval.ScoreComponents{
	Semantic: cosineSim,
	Keyword:  keywordOverlap,
	Filename: retrieval.FilenameOverlap(meta.Path, query),
	Metadata: retrieval.MetadataScore(query, meta),
})
```

---

## Evaluating quality

`Evaluate` computes standard IR metrics for one query against a ranked result
list.

```go
query := retrieval.EvalQuery{
	ID: "q1",
	Relevant: map[string]float64{
		"doc-a": 2.0, // graded relevance; values > 0 are relevant
		"doc-b": 1.0,
	},
	TopicByDoc: map[string]string{"doc-a": "ml", "doc-b": "systems"},
}

results := []retrieval.RankedResult{
	{ID: "doc-a", Score: 0.91, Topic: "ml"},
	{ID: "doc-c", Score: 0.80, Topic: "ml"},
	{ID: "doc-b", Score: 0.72, Topic: "systems"},
}

m := retrieval.Evaluate(query, results, 3)
fmt.Printf("Recall@3=%.2f P@3=%.2f MRR=%.2f NDCG@3=%.2f UniqueTopics=%d\n",
	m.RecallAtK, m.PrecisionAtK, m.MRR, m.NDCGAtK, m.UniqueTopicAtK)
```

`k` is the cutoff depth applied to both the result slice and the metric
denominators.

| Metric | Description |
|---|---|
| `RecallAtK` | fraction of relevant docs found in top-k |
| `PrecisionAtK` | fraction of top-k that are relevant |
| `MRR` | mean reciprocal rank of the first relevant result |
| `NDCGAtK` | normalized discounted cumulative gain (graded relevance) |
| `UniqueTopicAtK` | distinct `Topic` values in top-k; computed even when `Relevant` is empty |

`Relevant` values act as graded relevance: any value > 0 is relevant, and the
numeric value drives NDCG gain via `2^rel − 1`. MRR and NDCG return 0 when
`Relevant` is empty; `UniqueTopicAtK` is always computed.

Metric functions count only the first occurrence of each stable result ID.
`ValidateRun` is stricter: it rejects blank or duplicate IDs in the final result
list and in every captured stage. `EvaluateRun` also requires exactly one run
query for every fixture query and rejects unknown query IDs.

For golden-query regression checks, define a `Fixture` with judged documents and
evaluate a captured `Run`. The harness uses the same `Evaluate`,
`EvaluateSegments`, and `EvaluateLayers` metrics, then averages per-query metrics
into the aggregate report.

```go
fixture := retrieval.Fixture{
	ID: "golden",
	Queries: []retrieval.FixtureQuery{{
		ID:   "q1",
		Text: "machine learning",
		Judgments: []retrieval.Judgment{
			{DocID: "doc-a", Relevance: 2, Topic: "ml", Segment: "guides"},
			{DocID: "doc-b", Relevance: 1, Topic: "systems", Segment: "guides"},
		},
	}},
}

run := retrieval.Run{
	ID: "candidate",
	Queries: []retrieval.RunQuery{{
		ID: "q1",
		Results: []retrieval.RankedResult{
			{ID: "doc-a", Score: 0.91},
			{ID: "doc-c", Score: 0.80},
		},
		Stages: retrieval.StageResults{
			Candidates: []retrieval.RankedResult{{ID: "doc-a"}, {ID: "doc-b"}},
			Final:      []retrieval.RankedResult{{ID: "doc-a"}},
		},
	}},
}

report, err := retrieval.EvaluateRun(fixture, run, 2)
if err != nil {
	return err
}

failures := retrieval.CheckThresholds(report, retrieval.ReportThresholds{
	MinRecallAtK:    0.5,
	MinPrecisionAtK: 0.5,
	MinMRR:          1.0,
})
if len(failures) > 0 {
	return fmt.Errorf("retrieval quality below floor: %v", failures)
}
```

`Judgment.Segment` owns segment membership for `EvaluateRun`; result IDs without
a judged segment are excluded from segment summaries. `RunQuery.Stages` are
captured pipeline outputs only: the fixture harness does not rank, fetch, embed,
store, or mutate data.
