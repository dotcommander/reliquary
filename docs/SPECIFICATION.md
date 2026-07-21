# Reliquary Technical Specification & Invariants

This document outlines the strict behavioral contracts, storage invariants, and execution semantics enforced by Reliquary.

---

## 1. Embedder Contract & Index Identity

### Custom Embedders
- Must implement `embedding.Embedder` (`Embed(ctx, Request) (Result, error)`).
- Must return exactly **one finite, positive-dimension vector** per input text slice in the same ordinal order.
- Must pass the shared behavioral test suite in `embedding/embeddingtest`.
- Reliquary validates all embedder outputs before updating internal retrieval structures or persisting vectors to an index.

### Index Identity & Vector Dimensions
- Production applications **must** provide `WithIndexIdentity` during `reliquary.New(...)` initialization.
- The index identity tags an index with its embedding model and chunking configuration.
- Indexes **reject reads and writes** if their configured identity does not match the stored identity, even if vector dimensions are identical.
- **Identity Latching**: The first indexed non-nil result locks the identity, and the first embedded vector establishes the vector dimension.
- **Persistence Across Deletes**: Deletions and document replacements preserve the established identity and dimension, even if the index becomes completely empty.
- **Destructive Rebuild**: Calling `app.ResetIndex(ctx)` is the only operation that clears the established identity and dimension, allowing re-ingestion under a new identity or model.

---

## 2. Ingestion & Revision Semantics

### Document Revision Atomicity
- `App.Ingest(ctx, docs...)` treats every supplied `Document` as a **complete atomic revision**.
- All prior chunks belonging to the provided `Document.ID`s are replaced in a single atomic operation across the index.
- If a document produces zero chunks (e.g. empty text), all prior chunks for that document ID are deleted.
- Document IDs within a single `Ingest` batch must be non-blank and unique.

### Result Ownership & Index Writes
- `Index.Upsert` performs a merge by result ID (`Result.ID`).
- `Index.DeleteDocument` removes items by exact `Result.DocumentID` ownership (result ID string prefixes never imply document ownership).
- `ReplaceDocuments(ctx, []DocumentReplacement) error` is a mandatory atomic batch method for custom `index.Index` implementations.
- Every custom index implementation should pass the shared contract suite in `index/indextest`.
- Vectors containing `NaN` or `infinity` values are strictly rejected on writes and searches. Empty embeddings remain valid for lexical-only search backends.

---

## 3. Filtering Mechanics

### Target Fields & Scalar Matching
- Candidates can be filtered by reserved fields (`id`, `document_id`, `filename`) or custom scalar metadata (`WithFilter(...)`).
- Filtering occurs during candidate retrieval before final reranking and MMR diversification.
- **JSON-Scalar Equality**: Metadata matching operates on backend-independent equality rules:
  - Strings and booleans are type-exact.
  - Explicit `nil` filter values match only JSON `null`.
  - Finite numbers compare by exact numeric value across Go numeric types (`int`, `float64`, etc.).
  - Reserved fields match strings only.
- Compound values (arrays, maps), `NaN`, and `infinity` filter values are rejected prior to query embedding.

---

## 4. Adapter & Database Invariants

### Caller Ownership & Zero Hidden Side Effects
- Database constructors (`adapter/postgres`, `adapter/sqlite`) validate configuration parameters only.
- Adapters **never** connect automatically or execute hidden database migrations during initialization.
- Callers must explicitly call the adapter's `Migrate(ctx)` method before running queries or writes.
- Database connection pools (`pgxpool.Pool`, `sql.DB`), network transports, credentials, and retry policies remain 100% caller-owned.

### Schema & Index State Tables
- `adapter/postgres` and `adapter/sqlite` migrations create an adapter-owned index-space state table to persist index identity and vector dimension metadata across process restarts.
- `adapter/sqlite` candidate bounds apply when `IndexQuery.Limit` is zero; positive limits are never truncated below the requested count.
