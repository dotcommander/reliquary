package ingestfs_test

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/pipeline/ingest"
	"github.com/dotcommander/reliquary/pipeline/ingest/fs"
)

func Example() {
	root, err := os.MkdirTemp("", "reliquary-ingestfs-")
	if err != nil {
		fmt.Println("temp directory error")
		return
	}
	defer os.RemoveAll(root)
	if err := os.MkdirAll(filepath.Join(root, "notes"), 0o755); err != nil {
		fmt.Println("directory error")
		return
	}
	if err := os.WriteFile(filepath.Join(root, "notes", "a.txt"), []byte("hello\n"), 0o600); err != nil {
		fmt.Println("write error")
		return
	}

	reader, err := ingestfs.New(ingestfs.Config{Root: root, Source: "notes"})
	if err != nil {
		fmt.Println("reader error")
		return
	}
	sink := &documentSink{}
	pipeline := ingest.NewRecordPipeline[document.Document, document.Document](
		reader,
		documentDecoder{},
		documentMapper{},
		sink,
	)
	if _, err := pipeline.Run(context.Background(), ingest.Cursor{}); err != nil {
		fmt.Println("run error")
		return
	}

	fmt.Println(sink.records[0].ID, sink.records[0].Payload.Title)
	// Output: notes/a.txt a.txt
}

type documentDecoder struct{}

func (documentDecoder) DecodeRecord(_ context.Context, record ingest.Record[[]byte]) ([]ingest.Record[document.Document], error) {
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

type documentMapper struct{}

func (documentMapper) Map(_ context.Context, record ingest.Record[document.Document]) (ingest.Record[document.Document], error) {
	return record, nil
}

type documentSink struct {
	records []ingest.Record[document.Document]
}

func (s *documentSink) Write(_ context.Context, batch ingest.Batch[document.Document]) (ingest.Report, error) {
	s.records = append(s.records, batch.Records...)
	return ingest.Report{Written: len(batch.Records), Cursor: batch.Cursor}, nil
}
