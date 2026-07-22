// Package ingest defines generic reader, decoder, mapper, sink, batch, report,
// and cursor contracts for caller-owned ingestion pipelines. Source-specific
// scraping, auth, scheduling, and persistence policy stay outside this package.
// NewRecordPipeline supports decoders that need the complete raw record
// envelope; NewPipeline retains the payload-only Decoder contract.
//
// To persist pipeline output into reliquary's own storage, use the
// pipeline/indexsink package, which provides an ingest.Sink backed by
// index.Index.
package ingest
