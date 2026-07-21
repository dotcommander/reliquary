# Reliquary

Reliquary helps Go applications turn documents into useful context for AI
features. Start in memory, then bring your own embedder and index when you need
production storage and retrieval quality.

```sh
go get github.com/dotcommander/reliquary@v0.10.0
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

- Feed a paginated API, object store, or file collection through
  `pipeline/ingest`, then persist mapped results with `pipeline/indexsink`.

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

Reliquary includes opt-in adapters for OpenAI embeddings, PostgreSQL/pgvector,
and SQLite/FTS5. Database constructors validate configuration but do not run
migrations; call the adapter's `Migrate(ctx)` explicitly. Clients, database
handles, credentials, transports, and retry policy remain caller-owned.

`App.Ingest` treats each document as a complete revision and atomically replaces
its previous chunks. Custom indexes must implement `index.Index`, including
`ReplaceDocuments`, and should run the shared `index/indextest` contract suite.
Custom embedders should run `embedding/embeddingtest`.

## Packages

| Package | Purpose |
|---|---|
| `reliquary` | High-level `App` facade, options, and constructors |
| `document` | Document value type |
| `embedding` | Provider-neutral `Embedder`, request, result, and vector contracts |
| `embed` | Deterministic hashing embedder for demos and tests |
| `index` | Candidate retrieval contract, in-memory implementation, and `indextest` suite |
| `chunking` | Boundary-aware text, code, sentence, and heading splitters |
| `retrieval` | Hybrid scoring, MMR diversification, filtering, and evaluation |
| `pipeline/ingest` | Generic resumable ingestion contracts and runner |
| `pipeline/indexsink` | `pipeline/ingest` sink backed by `index.Index` |
| `pipeline/lexical` | Lexical analysis, BM25, and result fusion |
| `dedup`, `textutil`, `vector` | Retrieval primitives, vector math, quantization, and clustering |
| `adapter/openai` | OpenAI embedding adapter |
| `adapter/postgres`, `adapter/sqlite` | Persistent candidate-retrieval adapters |

## Documentation

- [Technical specification and invariants](docs/SPECIFICATION.md)
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
