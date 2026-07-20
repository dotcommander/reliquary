// Package ingest defines generic reader, decoder, mapper, sink, batch, report,
// and cursor contracts for caller-owned ingestion pipelines. Source-specific
// scraping, auth, scheduling, and persistence policy stay outside this package.
//
// To persist pipeline output into reliquary's own storage, use the indexsink
// package, which provides an ingest.Sink backed by index.Index.
package ingest
