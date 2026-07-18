// Package lexical provides provider-neutral lexical search primitives.
//
// It handles deterministic token/query/stat math only: analysis, query term
// counts, local document statistics, BM25-compatible scoring inputs,
// ranked-result adapters, rank fusion, and tiny fixture evaluation. It does not
// own database schemas, SQL syntax, tokenizer extensions, stemming, trigram
// behavior, migrations, thresholds, or business boosts.
//
// SQLite FTS5, PostgreSQL full-text search, trigram indexes, external tokenizers,
// and application-specific boosts remain app-local. Treat tokenizer options,
// stop words, minimum token length, BM25 parameters, document frequencies,
// document count, average document length, and database tokenizer/schema config
// as part of the caller-owned scoring identity. Changing any of them invalidates
// cached lexical scores and rank comparisons.
//
// A full-text engine such as Bleve stays app-level (or a nested adapter), never
// a dependency of this package: an adapter maps the engine's result IDs and
// ranks into a [RankedList] (rank-only score space when raw scores are not
// comparable), leaving index mappings, analyzers, and storage app-local.
package lexical
