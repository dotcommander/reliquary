# indexsink

```go
sink, err := indexsink.NewSink(idx, indexsink.Config{
	IndexIdentity: "text-embedding-3-small|smart-220-0",
})
if err != nil {
	return err
}
pipe := ingest.NewPipeline[string, *retrieval.Result](
	reader, decoder, mapper, sink,
)
report, err := pipe.Run(ctx, ingest.Cursor{Source: "feed"})
```

`indexsink` is the termination glue between `pipeline/ingest` and the `index`
contract: an `ingest.Sink[*retrieval.Result]` backed by `index.Index`, so a
resumable batch pipeline can persist into reliquary's own storage.

Use it when a `pipeline/ingest` run should share the `Index` instance wired into
an `App`. Source reading, decoding, and mapping stay caller-owned; `indexsink`
owns only the persistence step. The sink writes with merge-style `Index.Upsert`,
while `App.Ingest` atomically replaces complete document revisions with
`Index.ReplaceDocuments`. Configure the sink with the same index identity as the
App. The sink stamps missing result identities on copies and rejects conflicting
identities before writing. See `ExampleSink` for an end-to-end wiring, and
`pipeline/ingest` for the contracts.

Reference facts (verified): NewSink(index.Index, Config) returns (*Sink, error); ingest.NewPipeline[Decoded, Out](reader, decoder, mapper, sink); Sink.Write uses Index.Upsert; App.Ingest uses Index.ReplaceDocuments.
