package indexsink

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/pipeline/ingest"
	"github.com/dotcommander/reliquary/retrieval"
)

// ExampleSink shows a full resumable batch pipeline terminating at reliquary's
// Index: a byte Reader → Decoder → Mapper → indexsink.Sink. In production code,
// swap the fakes for a real source (files, object storage, a paginated API) and
// a real Index (storage/sqlite, storage/postgres, ...).
func ExampleSink() {
	idx := &fakeIndex{}
	sink, err := NewSink(idx, Config{IndexIdentity: "example-space"})
	if err != nil {
		fmt.Println("sink error:", err)
		return
	}

	pipe := ingest.NewPipeline[string, *retrieval.Result](
		exampleReader{},
		exampleDecoder{},
		exampleMapper{},
		sink,
	)

	report, err := pipe.Run(context.Background(), ingest.Cursor{Source: "example"})
	if err != nil {
		fmt.Println("run error:", err)
		return
	}

	fmt.Println("written:", report.Written)
	for _, batch := range idx.upserts {
		for _, r := range batch {
			fmt.Println("id:", r.ID)
		}
	}

	// Output:
	// written: 2
	// id: doc-1
	// id: doc-2
}

// exampleReader emits one batch of two byte records. The empty cursor token tells
// Pipeline.Run to stop after processing this batch (see ingest.Pipeline.Run).
type exampleReader struct{}

func (exampleReader) Read(_ context.Context, _ ingest.Cursor) (ingest.Batch[[]byte], error) {
	return ingest.Batch[[]byte]{
		Records: []ingest.Record[[]byte]{
			{ID: "r1", Payload: []byte("doc-1")},
			{ID: "r2", Payload: []byte("doc-2")},
		},
	}, nil
}

// exampleDecoder turns a byte payload into a single string record keyed by the
// payload itself.
type exampleDecoder struct{}

func (exampleDecoder) Decode(_ context.Context, data []byte) ([]ingest.Record[string], error) {
	s := string(data)
	return []ingest.Record[string]{{ID: s, Payload: s}}, nil
}

// exampleMapper lifts a decoded string record into a *retrieval.Result payload
// ready for Upsert.
type exampleMapper struct{}

func (exampleMapper) Map(_ context.Context, rec ingest.Record[string]) (ingest.Record[*retrieval.Result], error) {
	return ingest.Record[*retrieval.Result]{ID: rec.ID, Payload: &retrieval.Result{ID: rec.ID}}, nil
}
