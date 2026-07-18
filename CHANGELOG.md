# Changelog

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
