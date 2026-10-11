// Command filesystem-ingest demonstrates scanning and ingesting a directory
// tree of documents into Reliquary using pipeline/ingest/fs.
//
// It shows:
// 1. Initializing ingestfs.Reader on a local directory with file filtering.
// 2. Piping discovered filesystem records through an ingest.RecordPipeline.
// 3. Sinking decoded documents into the reliquary.App indexing pipeline.
// 4. Searching across the automatically discovered and ingested files.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/filesystem-ingest
package main

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/pipeline/ingest"
	ingestfs "github.com/dotcommander/reliquary/pipeline/ingest/fs"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	// Create a temporary directory tree simulating a repo docs/ folder
	tempDir, err := os.MkdirTemp("", "reliquary-fs-demo-*")
	if err != nil {
		return fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tempDir)

	if err := createSampleDocs(tempDir); err != nil {
		return fmt.Errorf("create sample docs: %w", err)
	}

	// 1. Initialize the directory reader with Markdown-only filter
	reader, err := ingestfs.New(ingestfs.Config{
		Root:   tempDir,
		Source: "knowledge-base",
		Include: func(path string, entry fs.DirEntry) bool {
			// Include only .md files; skip hidden and draft files
			if entry.IsDir() {
				return !strings.HasPrefix(entry.Name(), ".")
			}
			return strings.HasSuffix(entry.Name(), ".md") && !strings.HasPrefix(entry.Name(), "_")
		},
	})
	if err != nil {
		return fmt.Errorf("new fs reader: %w", err)
	}

	app := reliquary.Quickstart()

	// 2. Wire decoder and sink
	var ingestedDocs []document.Document
	decoder := fsDocumentDecoder{}
	mapper := fsDocumentMapper{}
	sink := &appCollectorSink{onBatch: func(docs []document.Document) {
		ingestedDocs = append(ingestedDocs, docs...)
	}}

	pipeline := ingest.NewRecordPipeline[document.Document, document.Document](
		reader,
		decoder,
		mapper,
		sink,
	)

	// 3. Run the ingest pipeline
	report, err := pipeline.Run(ctx, ingest.Cursor{})
	if err != nil {
		return fmt.Errorf("run ingest pipeline: %w", err)
	}
	fmt.Printf("Discovered and decoded %d files (records written: %d)\n", len(ingestedDocs), report.Written)

	// 4. Ingest documents into the Reliquary App facade
	chunkCount, err := app.Ingest(ctx, ingestedDocs...)
	if err != nil {
		return fmt.Errorf("app ingest: %w", err)
	}
	fmt.Printf("Ingested %d documents (%d chunks) into Reliquary index\n\n", len(ingestedDocs), chunkCount)

	// 5. Query the indexed directory content
	query := "raft consensus leader election"
	hits, err := app.Search(ctx, query, reliquary.TopK(2))
	if err != nil {
		return fmt.Errorf("search: %w", err)
	}

	fmt.Printf("Search query: %q\n", query)
	for i, hit := range hits {
		fmt.Printf("[%d] %s (score: %.3f): %s\n", i+1, hit.ID, hit.CombinedScore, strings.TrimSpace(hit.Content))
	}
	return nil
}

type fsDocumentDecoder struct{}

func (fsDocumentDecoder) DecodeRecord(_ context.Context, record ingest.Record[[]byte]) ([]ingest.Record[document.Document], error) {
	doc, err := document.FromReader(
		record.ID,
		bytes.NewReader(record.Payload),
		document.WithFilename(record.Metadata["filename"]),
		document.WithMetadata(record.Metadata),
	)
	if err != nil {
		return nil, err
	}
	return []ingest.Record[document.Document]{{
		ID:       record.ID,
		Payload:  doc,
		Metadata: record.Metadata,
	}}, nil
}

type fsDocumentMapper struct{}

func (fsDocumentMapper) Map(_ context.Context, record ingest.Record[document.Document]) (ingest.Record[document.Document], error) {
	return record, nil
}

type appCollectorSink struct {
	onBatch func([]document.Document)
}

func (s *appCollectorSink) Write(_ context.Context, batch ingest.Batch[document.Document]) (ingest.Report, error) {
	docs := make([]document.Document, len(batch.Records))
	for i, r := range batch.Records {
		docs[i] = r.Payload
	}
	s.onBatch(docs)
	return ingest.Report{Written: len(batch.Records), Cursor: batch.Cursor}, nil
}

func createSampleDocs(root string) error {
	_ = os.MkdirAll(filepath.Join(root, "distributed"), 0o755)
	_ = os.MkdirAll(filepath.Join(root, "storage"), 0o755)

	files := map[string]string{
		"distributed/raft.md": "# Raft Protocol\nRaft is a consensus algorithm that elects a leader to manage log replication across nodes.",
		"storage/lsm.md":      "# LSM Trees\nLog-structured merge-trees append writes sequentially to write-ahead logs and SSTables.",
		"storage/draft.txt":   "This is a plain text file that will be skipped by the include filter.",
	}
	for relPath, content := range files {
		fullPath := filepath.Join(root, relPath)
		if err := os.WriteFile(fullPath, []byte(content), 0o600); err != nil {
			return err
		}
	}
	return nil
}
