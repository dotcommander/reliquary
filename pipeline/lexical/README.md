# lexical

Provider-neutral lexical search contracts for analysis, BM25-compatible scoring
inputs, external FTS rank adaptation, rank fusion, and small rank fixtures.

`lexical` does not own SQLite, PostgreSQL, tokenizer extensions, migrations,
thresholds, boosts, or schema identity. Apps map their native lexical output
into stable `Result` values at the boundary.

## Local BM25

```go
analyzer := lexical.NewAnalyzer(lexical.AnalyzerOptions{MinTokenLength: 2})
query := lexical.NormalizeQuery("alpha beta", analyzer)

doc := lexical.NewDocumentStatsFromTokens(analyzer.Analyze("alpha alpha beta"))
corpus := lexical.NewCorpusStats([]lexical.DocumentStats{doc})
score := lexical.BM25Score(query, doc, corpus, lexical.DefaultBM25Params())
local := lexical.RankByScore([]lexical.Candidate{{
	ID:         "doc-a",
	Score:      score,
	Source:     "local_bm25",
	ScoreSpace: lexical.ScoreSpaceLocalBM25,
}})
_ = local
```

## SQLite FTS5 Rows

SQLite owns the table schema, tokenizer, and `bm25()` settings. Map returned
rows into rank-only results without moving those choices into `reliquary`.

```go
type fts5Row struct {
	ID    string
	Score float64 // SQLite bm25() is often lower-is-better.
}

func sqliteCandidates(rows []fts5Row) lexical.RankedList {
	results := make([]lexical.Result, 0, len(rows))
	for _, row := range rows {
		results = append(results, lexical.Result{ID: row.ID, Score: row.Score})
	}
	return lexical.RankByOrder(results, "sqlite_fts5", lexical.ScoreSpaceRankOnly)
}
```

## PostgreSQL `ts_rank` Rows

PostgreSQL owns `tsvector`, `tsquery`, dictionaries, weights, and indexes. Keep
those definitions app-local and adapt the higher-is-better rank value.

```go
type pgRankRow struct {
	ID   string
	Rank float64
}

func postgresCandidates(rows []pgRankRow) lexical.RankedList {
	results := make([]lexical.Result, 0, len(rows))
	for _, row := range rows {
		results = append(results, lexical.Result{
			ID:     row.ID,
			Score: row.Rank,
			Source: "postgres_fts",
			ScoreSpace: lexical.ScoreSpaceProvider,
		})
	}
	return lexical.RankByScore(results)
}
```

## Rank Fusion

```go
fused := lexical.FuseRRFByID([]lexical.FusionInput{
	{Source: "bm25", Candidates: lexicalCandidates},
	{Source: "vector", Candidates: vectorCandidates},
}, lexical.FusionOptions{K: 60, Limit: 20})
```

For `vectors.RRF`, keep app IDs in your own map and convert only at the boundary:

```go
indices, report := lexical.RankedIndicesWithReport(lexicalCandidates, indexByID)
_ = report.UnknownIDs
```

For retrieval evaluation, adapt lexical ranked lists without importing database
drivers or schema types:

```go
layers := retrieval.EvaluateLayers(queryFixture, retrieval.LayeredResults{
	Candidates: lexical.ToRetrievalResults(lexicalCandidates),
	Final:      lexical.ToRetrievalResults(fused),
}, 10)
_ = layers
```

`lexical.Explain` reports source, rank, score-space, score, and metadata for
debug output without defining SQL, stemming, trigram policy, or app boosts.

Cached fixtures or rank outputs become stale when tokenizer rules, BM25 params,
corpus stats, app boosts, DB FTS config, or schema/index identity changes.
