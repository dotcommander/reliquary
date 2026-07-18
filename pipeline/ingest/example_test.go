package ingest_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/pipeline/ingest"
)

func ExampleBatch() {
	batch := ingest.Batch[string]{
		Records: []ingest.Record[string]{{ID: "doc-1", Payload: "body"}},
		Cursor:  ingest.Cursor{Source: "feed", Token: "next"},
	}
	fmt.Println(batch.Records[0].ID, batch.Cursor.Token)
	// Output: doc-1 next
}
