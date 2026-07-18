# Reliquary

Reliquary is a Go toolkit for document ingestion and retrieval. The root
facade combines chunking, embeddings, candidate retrieval, hybrid scoring, and
optional MMR reranking behind a small `App` API.

```sh
go get github.com/dotcommander/reliquary@v0.8.0
```

```go
package main

import (
	"context"
	"log"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
)

func main() {
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, document.Document{
		ID: "doc-1", Text: "Go uses a concurrent garbage collector.",
	}); err != nil {
		log.Fatal(err)
	}
	results, err := app.Search(ctx, "garbage collector", reliquary.TopK(5))
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("found %d result(s)", len(results))
}
```

This quickstart uses deterministic in-memory embeddings for demos and tests.
Production applications should inject their own embedder and index with
`New`, `WithEmbedder`, `WithIndex`, and `WithIndexIdentity`.

The index identity is required for `New` and must change whenever the embedding
space or chunking policy changes. A mismatched populated index is rejected even
when dimensions match. To rebuild intentionally, call `app.ResetIndex(ctx)`;
this permanently deletes every indexed chunk before re-ingestion. `Quickstart`
and `InMemory` supply a deterministic identity for their built-in hashing setup.

Restrict candidate retrieval by reserved fields (`id`, `document_id`, or
`filename`) or scalar metadata while retaining final reranking and MMR:

```go
results, err := app.Search(ctx, "search text",
	reliquary.WithFilter(map[string]any{"project": "reliquary"}),
	reliquary.TopK(5),
)
```

The default index is concurrency-safe and in-memory. Use `WithIndex` to inject
another implementation.

## Adapters

- `adapter/openai` adapts an injected `openai.Client` to the embedding contract.
- `adapter/postgres` provides bounded pgvector candidate retrieval.
- `adapter/sqlite` provides bounded FTS5 candidate retrieval with final ranking
  performed by Reliquary.

Database constructors validate configuration and perform no migrations. Call
the adapter's `Migrate(ctx)` method explicitly before use. Callers retain
ownership of database pools, connections, credentials, transports, and retry
policy.

## Ownership

Product memory, graph behavior, and generic application infrastructure are
intentionally outside this module. See [the v0.5 migration guide](docs/MIGRATION-v0.5.md)
for removed ownership surfaces and [the v0.6 migration guide](docs/MIGRATION-v0.6.md)
for the current public import paths.

Reliquary v0.7 uses `Index` as its only persistence seam. The deprecated
`Store` compatibility API was removed.

## Project policies

- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [MIT License](LICENSE)

## Verify

```sh
GOWORK=off go build ./...
GOWORK=off go test ./...
GOWORK=off go vet ./...
go test -race . ./index/... ./adapter/...
./scripts/check-boundaries.sh
./scripts/verify-modules.sh
```
