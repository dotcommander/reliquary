# Reliquary

Reliquary helps Go applications turn documents into useful context for AI
features. Start in memory, then bring your own embedder and index when you need
production storage and retrieval quality.

```sh
go get github.com/dotcommander/reliquary@v0.11.0
```

```go
package main

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
)

func main() {
	ctx := context.Background()
	app := reliquary.Quickstart()

	_, err := app.Ingest(ctx, document.Document{
		ID: "notes",
		Text: "The Go build cache lives in ~/Library/Caches/go-build.",
	})
	if err != nil {
		panic(err)
	}

	hits, err := app.Search(ctx, "where is the Go build cache?", reliquary.TopK(1))
	if err != nil {
		panic(err)
	}
	fmt.Println(hits[0].Content)
}
```

`Quickstart` uses a deterministic in-memory embedder and index. It is useful for
examples and tests, but it is not a production embedding model.

## What you can build

- Give an AI assistant useful context from project notes, READMEs, runbooks, or
  issue summaries. `App.Ingest` chunks and indexes each document; `App.Search`
  returns the best passages to add to your model prompt.
- Keep retrieval inside the current project or tenant with scalar metadata:

  ```go
  _, err := app.Ingest(ctx, document.Document{
      ID:       "runbook",
      Text:     "Restart the worker with launchctl kickstart.",
      Metadata: document.Metadata{"project": "reliquary"},
  })
  hits, err := app.Search(ctx, "restart the worker",
      reliquary.WithFilter(map[string]any{"project": "reliquary"}),
      reliquary.TopK(3),
  )
  ```

  Filters scope retrieval; they are not an authorization boundary.
- Build a less repetitive context pack by fetching more candidates than you
  return and using MMR to balance relevance with diversity:

  ```go
  hits, err := app.Search(ctx, question,
      reliquary.CandidateLimit(20),
      reliquary.TopK(5),
      reliquary.WithMMR(0.5),
  )
  ```

- Fuse independent vector and lexical rankings before TopK or MMR:

  ```go
  hits, err := app.Search(ctx, question,
      reliquary.CandidateLimit(50),
      reliquary.WithRRF(60),
      reliquary.TopK(5),
  )
  ```

  `WithRRF` asks the configured index for one vector-only ranking and one
  text-only ranking, then combines their result IDs with reciprocal rank
  fusion. It does not add BM25 or another lexical engine; your index supplies
  both rankings. Each lane receives the candidate limit independently. Values
  at or below zero and non-finite values use the standard RRF constant `60`.

- Add a per-search cross-encoder after hybrid scoring and before TopK or MMR:

  ```go
  hits, err := app.Search(ctx, question,
      reliquary.CandidateLimit(50),
      reliquary.WithReranker(bgeReranker),
      reliquary.TopK(5),
  )
  ```

  The reranker receives hybrid-ranked candidates by default or RRF-ranked
  candidates when `WithRRF` is enabled, and returns one finite score in `[0,1]`
  for each candidate. A reranker failure or malformed response fails the
  search; Reliquary does not fall back to the preceding ranking.

- Inspect how retained candidates moved through hybrid scoring, RRF, an
  external reranker, and MMR:

  ```go
  hits, err := app.Search(ctx, question,
      reliquary.CandidateLimit(50),
      reliquary.WithExplain(),
      reliquary.TopK(5),
  )
  trace := hits[0].Explain
  fmt.Println(trace.Hybrid.Raw.Keyword, trace.FinalRank)
  ```

  `Explain` is nil unless `WithExplain` is supplied. Explanations are typed,
  ephemeral result data; they are not written to the index or source metadata.
  The keyword value is token overlap computed by the hybrid scorer, not a
  backend-native BM25 score.

- Embed several questions in one call while preserving blank positions, then
  render selected passages as neutral context:

  ```go
  rows, err := app.SearchBatch(ctx, []string{question, followup})
  hits := rows[0]
  promptBlock, err := retrieval.FormatContext(hits,
      retrieval.WithHeader("[Source: %s, Lines: %d-%d]"),
      retrieval.WithMaxTokens(2048, tokenCounter),
  )
  ```

  The token counter is caller-supplied, so it can match the model that will
  consume the prompt. Context includes only a contiguous prefix of complete
  result blocks; headers and separators count toward the budget.

- Construct bounded text documents from streams with `document.FromReader`.
  Input defaults to a 16 MiB limit and must be valid UTF-8; filenames label
  documents but do not select parsers or infer formats.

- Read a local directory in deterministic, resumable batches with
  `pipeline/ingest/fs`, decode each file with its path metadata through
  `pipeline/ingest.NewRecordPipeline`, then persist mapped results with
  `pipeline/indexsink`.

## Production wiring

Production applications inject an `embedding.Embedder`, an `index.Index`, and
an explicit index identity:

```go
app, err := reliquary.New(
	reliquary.WithEmbedder(embedder),
	reliquary.WithIndex(idx),
	reliquary.WithIndexIdentity("text-embedding-3-small:v1#smart-boundary"),
)
```

The identity names the embedding space and chunking policy stored in the index.
Reliquary rejects reads and writes with a different identity even when the
vector dimensions match. Change the identity and call `ResetIndex` before a
deliberate rebuild.

Reliquary includes opt-in adapters for OpenAI and Ollama embeddings,
PostgreSQL/pgvector, and SQLite/FTS5. The Ollama adapter targets the native
`/api/embed` endpoint. Adapter constructors perform no network I/O. Database
constructors validate configuration but do not run migrations; call the
adapter's `Migrate(ctx)` explicitly. Clients, database handles, credentials,
transports, and retry policy remain caller-owned.

`App.Ingest` treats each document as a complete revision and atomically replaces
its previous chunks. Custom indexes must implement `index.Index`, including
`ReplaceDocuments`, and should run the shared `index/indextest` contract suite.
Custom embedders should run `embedding/embeddingtest`.

## Packages

| Package | Purpose |
|---|---|
| `reliquary` | High-level `App` facade, options, and constructors |
| `document` | Document value type and bounded UTF-8 reader construction |
| `embedding` | Provider-neutral `Embedder`, request, result, and vector contracts |
| `embed` | Deterministic hashing embedder for demos and tests |
| `index` | Candidate retrieval contract, in-memory implementation, and `indextest` suite |
| `chunking` | Boundary-aware text, code, sentence, and heading splitters |
| `retrieval` | Hybrid scoring, MMR, filtering, evaluation, and neutral context rendering |
| `pipeline/ingest` | Generic resumable ingestion contracts and runner |
| `pipeline/ingest/fs` | Deterministic, bounded local-directory reader |
| `pipeline/indexsink` | `pipeline/ingest` sink backed by `index.Index` |
| `pipeline/lexical` | Lexical analysis, BM25, and result fusion |
| `dedup`, `textutil`, `vector` | Retrieval primitives, vector math, quantization, and clustering |
| `adapter/openai` | OpenAI embedding adapter |
| `adapter/ollama` | Native Ollama embedding adapter |
| `adapter/postgres`, `adapter/sqlite` | Persistent candidate-retrieval adapters |

## Documentation

- [Technical specification and invariants](docs/SPECIFICATION.md)
- [Scoring, fusion, reranking, and MMR guide](retrieval/docs/scoring-guide.md)
- [Architecture and package boundaries](ARCHITECTURE.md)
- [Migration to v0.10](docs/MIGRATION-v0.10.md)
- [Migration to v0.6](docs/MIGRATION-v0.6.md)
- [Migration to v0.5](docs/MIGRATION-v0.5.md)
- [Runnable examples](examples)

## Verify

```sh
GOWORK=off go build ./...
GOWORK=off go test ./...
GOWORK=off go vet ./...
go test -race . ./index/... ./adapter/...
./scripts/check-boundaries.sh
./scripts/verify-modules.sh
```

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [MIT License](LICENSE)
