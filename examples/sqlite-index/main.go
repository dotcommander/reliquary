// Command sqlite-index demonstrates Reliquary's SQLite candidate index adapter.
//
// It shows:
// 1. Initializing caller-owned SQLite storage (modernc.org/sqlite).
// 2. Explicit schema migration with adapter/sqlite.Migrate.
// 3. Wiring the SQLite index into the reliquary.App pipeline.
// 4. Ingesting documents with atomic replacement and hybrid searching.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/sqlite-index
package main

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary"
	sqliteadapter "github.com/dotcommander/reliquary/adapter/sqlite"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embed"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	_ "modernc.org/sqlite"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	// Caller owns the database handle, connection pool, and file lifecycle.
	// We use an in-memory SQLite database for this self-contained demo.
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		return fmt.Errorf("open sqlite: %w", err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)

	// Construct the adapter and run its explicit schema migration.
	index, err := sqliteadapter.New(db, sqliteadapter.Config{
		Table:          "reliquary_index",
		CandidateLimit: 50,
	})
	if err != nil {
		return fmt.Errorf("new sqlite index: %w", err)
	}
	if err := index.Migrate(ctx); err != nil {
		return fmt.Errorf("migrate sqlite index: %w", err)
	}

	// Wire the index into the high-level App facade with an embedder.
	const dim = 64
	embedder := embed.NewHashing(dim)
	app, err := reliquary.New(
		reliquary.WithIndex(index),
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndexIdentity("sqlite-demo-v1"),
	)
	if err != nil {
		return fmt.Errorf("new reliquary app: %w", err)
	}

	// Ingest sample documents into SQLite.
	docs := []document.Document{
		{
			ID:     "doc-sqlite-arch",
			Title:  "sqlite-architecture.md",
			Format: document.FormatMarkdown,
			Text:   "SQLite is an in-process library implementing a self-contained, serverless, zero-configuration SQL engine.",
		},
		{
			ID:     "doc-sqlite-fts",
			Title:  "sqlite-fts5.md",
			Format: document.FormatMarkdown,
			Text:   "FTS5 is an SQLite virtual table module that provides full-text search capability with BM25 ranking.",
		},
		{
			ID:     "doc-concurrency",
			Title:  "go-concurrency.md",
			Format: document.FormatMarkdown,
			Text:   "Go concurrency uses goroutines and channels to model communicating sequential processes cleanly.",
		},
	}

	chunkCount, err := app.Ingest(ctx, docs...)
	if err != nil {
		return fmt.Errorf("ingest: %w", err)
	}
	fmt.Printf("Ingested %d documents (%d chunks) into SQLite index.\n", len(docs), chunkCount)

	// Perform hybrid search combining FTS5 lexical matching and vector cosine distance.
	query := "SQLite engine"
	results, err := app.Search(ctx, query, reliquary.TopK(2))
	if err != nil {
		return fmt.Errorf("search: %w", err)
	}

	fmt.Printf("\nSearch query: %q\n", query)
	for i, hit := range results {
		fmt.Printf("[%d] %s (score: %.3f): %s\n", i+1, hit.ID, hit.CombinedScore, strings.TrimSpace(hit.Content))
	}
	return nil
}
