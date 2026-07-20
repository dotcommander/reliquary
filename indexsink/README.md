# indexsink

```go
pipe := ingest.NewPipeline[string, *retrieval.Result](
	reader, decoder, mapper, indexsink.NewSink(idx), // ingest.Sink[*retrieval.Result]
)
report, err := pipe.Run(ctx, ingest.Cursor{Source: "feed"})
```

`indexsink` is the termination glue between `pipeline/ingest` and the `index`
contract: an `ingest.Sink[*retrieval.Result]` backed by `index.Index`, so a
resumable batch pipeline can persist into reliquary's own storage.

Use it when a `pipeline/ingest` run should write through the same `Index` that
`App.Ingest` uses. Source reading, decoding, and mapping stay caller-owned;
`indexsink` owns only the persistence step. See `ExampleSink` for an end-to-end
wiring, and `pipeline/ingest` for the contracts.

Reference facts (verified): NewSink(index.Index) returns ingest.Sink[*retrieval.Result]; ingest.NewPipeline[Decoded, Out](reader, decoder, mapper, sink); App.Ingest persists via the same index.Index.Upsert (rag.go:35).
