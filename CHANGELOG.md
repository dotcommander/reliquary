# Changelog

## v0.10.0 (2026-07-21)

### Breaking changes

- Rename the package identifier at import path `embedding` from `embeddings` to
  `embedding`. The import path and exported API remain unchanged.
- Move `indexsink` to `pipeline/indexsink` without a compatibility shim. Its
  exported API and behavior remain unchanged.

## v0.9.0 (2026-07-21)

### Breaking changes

- Require every `Index` implementation to provide atomic
  `ReplaceDocuments(ctx, []DocumentReplacement)`. `App.Ingest` now replaces
  complete document revisions atomically across its full variadic input;
  blank or duplicate document IDs are rejected before embedding.
- Version-frame `ModelRef.Identity` fields as `modelref:v1`. Existing embedding
  cache keys intentionally become cold misses; legacy keys are not read.

### Added

- Add the `indexsink` package, an `ingest.Sink` backed by `index.Index` that
  requires a non-nil index and explicit index identity, stamps missing result
  identities on copies, and rejects identity conflicts before writing.
- Add `embedding.ValidateResult`, `embedding.ErrInvalidResult`, and the reusable
  `embedding/embeddingtest` conformance suite for embedding implementations.

### Changed

- Make exact vector-index searches reject invalid query dimensions uniformly,
  and use cosine similarity for semantic chunk planning without mutating input
  vectors.
- Reject ragged or non-finite K-means/PQ training inputs and malformed PQ
  codes. PQ loads are strict, receiver-atomic, and limited to 256 MiB of
  serialized codebooks.
- Count text-analysis length thresholds in Unicode runes and make alias token
  coverage use distinct query/target occurrences with maximum fuzzy matching.

### Fixed

- Reject incomplete golden-query runs before calculating retrieval quality
  metrics or applying aggregate thresholds. Reject blank or duplicate result
  IDs in captured runs and count only the first stable ID in lower-level metric,
  source, and tuning helpers.
- Fail closed on malformed embedding batches before mutating retrieval results
  or accessing an index, reject nil ingestion results before embedding, and
  reject typed-nil embedders during application readiness checks. Typed-nil
  indexes now normalize to the default in-memory index.
- Delete documents by exact `Result.DocumentID` ownership, reject non-finite
  vectors at every Index boundary without changing data or space latches, and
  apply SQLite's candidate bound to zero-limit searches.
- Make `FindOptimalK64` return the zero invalid tuple for empty or malformed
  input and impossible ranges while preserving valid one- and two-point behavior.
- Upgrade `golang.org/x/text` to v0.39.0 to remediate GO-2026-5970.
- Preserve an index's established identity and vector dimension until explicit
  reset, including across delete-all and empty document replacement; SQLite and
  PostgreSQL persist this latch in migration-owned state tables.
- Define consistent JSON-scalar filter equality across all indexes, including
  exact finite numeric comparison, missing-versus-null semantics, and rejection
  of NaN and infinities before query embedding.
- Snapshot caller-owned arenas in both exact vector-index constructors.
- Reject blank document IDs on deletion without mutating stored rows.
- Preserve complete old document revisions when an in-memory, SQLite, or
  PostgreSQL batch replacement fails validation or storage, including result
  IDs that collide with retained documents.
- Serialize PostgreSQL first-write dimension discovery, make SQLite
  transactions rollback on panics, and match punctuation-bearing metadata keys
  exactly.
- Clear stale retrieval scores, calibrate only present signals, and correct
  ingest read/error/cursor reporting. Keep recency scoring finite for NaN and
  indeterminate infinite inputs.

## v0.8.0 (2026-07-18)

### Added

- Add a consumer-owned `Tokenizer` boundary and OpenAI-compatible
  `TiktokenTokenizer`.
- Add `FillTokenCountsWithTokenizer` and `EnforceTokenLimitsWithTokenizer` for
  exact provider/model token accounting with propagated errors.
- Add a security-reporting policy.

### Changed

- Publish only supported public adapters and remove the non-public embedding
  adapter and dependency.
- Align release automation and verification with the single root Go module.
- Update examples, installation guidance, and public documentation for direct
  root-module usage.

### Fixed

- Treat SQLite FTS5 search input as plain terms so punctuation and query
  operators cannot trigger FTS syntax errors.
- Correct stale example commands, documentation links, and package descriptions.

### Security

- Retract `v0.3.1` through `v0.7.0` because their published module archives
  included repository-only work artifacts.
- Replace imported Git history with a reviewed one-commit public history;
  legacy tags and Releases are not recreated.

## v0.7.0 (2026-07-12)

### Breaking changes
- Require `WithIndexIdentity` when constructing an `App` with `New`; this prevents same-dimension embedding models or chunking policies from silently sharing an index.

### Features
- Persist and enforce index identity in the in-memory, SQLite, and PostgreSQL indexes, with an explicit destructive `ResetIndex` rebuild path.
- Add `WithFilter` to the high-level search facade for reserved fields and scalar metadata.

### Fixes
- Correct the root facade example to use the current `document.Document` fields.

## v0.6.0 (2026-07-12)

### Breaking changes
- Finalize the flat retrieval package paths and remove the deprecated `Store`, `WithStore`, and `NewMemoryStore` compatibility API.

### Changed
- Make `Index` the only persistence seam and retain explicit caller ownership of database handles and migrations.

## v0.5.0 (2026-07-12)

### Breaking changes
- Extract memory, graph, runtime, configuration, and generic storage concerns so Reliquary owns retrieval only.

### Features
- Add explicitly constructed OpenAI, PostgreSQL/pgvector, and SQLite/FTS5 retrieval adapters.
- Retain the deprecated `Store` compatibility API for one transition release.
