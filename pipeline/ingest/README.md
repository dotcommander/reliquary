# ingest

```go
batch := ingest.Batch[string]{
	Records: []ingest.Record[string]{{ID: "doc-1", Payload: "body"}},
	Cursor:  ingest.Cursor{Source: "feed", Token: "next"},
}
```

`ingest` supplies generic contracts for resumable ingestion flows.

It does not know how to scrape, authenticate, schedule, persist, or interpret a
source. Apps compose readers, decoders, mappers, and sinks around their own
source semantics.

Use `NewRecordPipeline` when decoding depends on the raw record ID or metadata;
the record-aware decoder owns propagation of that envelope. The existing
`NewPipeline` continues to pass payload bytes only. For local directory trees,
[`pipeline/ingest/fs`](fs) supplies a deterministic bounded reader.

For persisting pipeline output into reliquary's own storage, the
[`pipeline/indexsink`](../indexsink) package provides an `ingest.Sink` backed
by `index.Index`.
