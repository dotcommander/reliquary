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

### Batch Search
- `App.SearchBatch` returns one result row per input query in the same order; blank queries retain nil rows.
- Every nonblank query is submitted to the embedder in one ordered request. The complete embedding result is validated before the first index read.
- Candidate searches and optional reranker calls execute sequentially by row because `index.Index` does not promise concurrent safety. Duplicate queries remain independent reranker calls.
- Options match `App.Search`. A fresh filter map and cloned candidates isolate every query from custom-index mutation.
- Any embedding, index, reranker, cancellation, or reranker-validation failure returns `nil` and the error; partial result matrices are never returned.

### Reciprocal Rank Fusion
- `WithRRF(k)` replaces the default single-call weighted ordering with two sequential `Index.Search` calls for each nonblank query: vector populated with blank text, then text populated with a nil vector. Values `k <= 0`, `NaN`, and infinity use `60`; the last `WithRRF` option wins.
- `CandidateLimit` and a fresh filter map apply independently to each lane. The fused union may contain up to twice the candidate limit.
- Fusion uses `pipeline/lexical.FuseRRFByID`. Overlapping IDs receive both rank contributions, while duplicate IDs within one lane contribute only their first rank. Exact fused ties use ascending ID.
- Nil candidates are skipped. Blank result IDs fail the search. For overlapping IDs, the vector-lane result supplies the returned payload; lexical-only results remain eligible.
- Raw RRF scores are normalized by `nonemptyLaneCount / (k+1)` and stored in `CombinedScore`. Empty successful lanes are valid; two empty lanes return no results.
- Hybrid component fields are computed over the fused union for diagnostics. App-level `WithWeights` does not affect RRF order or final `CombinedScore`.
- An optional external reranker runs after RRF and may replace `CombinedScore`; TopK and MMR remain the final stages. Any lane error or cancellation is fatal, with no fallback or partial results.
- `WithRRF` orchestrates rankings returned by the configured index. It does not add BM25, Bleve, or another lexical backend, and it does not change the PostgreSQL adapter into a full-text-search backend.

### Search Reranking
- Default search order is candidate retrieval, hybrid scoring, optional external reranking, then TopK and optional MMR diversification. With RRF enabled, vector and lexical candidate lanes are fused before the optional external reranker. `CandidateLimit` applies during each candidate retrieval call.
- `retrieval.Reranker` receives defensive candidate clones in hybrid-ranked or RRF-ranked order with hybrid component scores populated.
- A reranker must return exactly one finite score in `[0,1]` for each candidate. Reliquary validates the complete response before changing any result.
- On success, external scores replace `CombinedScore` and results are stable-sorted descending. Equal scores preserve the preceding hybrid or RRF order; component score fields remain hybrid-owned.
- Reranker errors, cancellation, or malformed scores fail the search with no fallback or partial results. Blank queries and empty candidate sets skip reranking.
- Separate concurrent `App.Search` calls may invoke the same reranker concurrently. Reranker implementations own any required synchronization.

### Search Explainability
- `WithExplain()` is an opt-in `SearchOption` with identical behavior in `Search` and `SearchBatch`. Without it, `Result.Explain` is nil and no explanation state is allocated.
- Each returned `SearchExplanation` includes the hybrid `ScoreTrace`, the hybrid rank and whether that score owned ordering, optional RRF/reranker/MMR stage details, and a one-based final rank.
- On the default path, the hybrid score owns initial order. On the RRF path, the hybrid trace is diagnostic only and RRF owns `CombinedScore` and order. An external reranker replaces `CombinedScore`. MMR changes order only and preserves the incoming relevance score in `CombinedScore`.
- RRF details contain the effective `k`, first rank and normalized contribution from each lane, fused score, and fused rank. External-reranker details contain its input rank, returned score, and resulting rank; model-internal reasoning is outside the `retrieval.Reranker` contract.
- MMR details contain the effective clamped lambda, incoming relevance, maximum positive cosine similarity to an already selected item, relevance contribution, signed penalty `-(1-lambda) * maxSimilarity`, and selection score. Missing, orthogonal, and negatively correlated embeddings contribute zero similarity. The first selected result has zero similarity penalty.
- Explanation state is cleared at index clone/storage boundaries and attached only to final returned hits. It never mutates source `Metadata` and is not an `Index` contract or persistence field.
- Explanations cover candidates retained for facade-level ranking, not why earlier candidate-generation stages excluded other items. `ScoreTrace.Raw.Keyword` is hybrid keyword overlap, not backend-native BM25 provenance; SQLite FTS uses `MATCH` for candidate selection without exposing an FTS score.

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

### Filesystem Ingestion
- `pipeline/ingest/fs` requires an existing non-symlink directory and a nonblank source name. Relative roots become absolute without shell or `~` expansion.
- Each reader snapshots its slash-normalized regular-file paths on the first read. Hidden files are included; symlinks and special files are excluded; additions after the snapshot are ignored.
- Records are emitted in lexical path order with root-relative IDs, raw bounded bytes, and exact `source`, `path`, and `filename` metadata. Parsing and format inference remain decoder-owned.
- Cursor tokens contain the last emitted path. Resume begins after that path even if it no longer exists. A nonempty final batch and an empty tree return an empty token.
- A snapshotted file that is removed, replaced, changed, unreadable, or larger than the configured limit fails its batch without partial records. Traversal and reads honor context cancellation.
- `NewRecordPipeline` passes the complete raw record to a `RecordDecoder`; record-aware decoders explicitly own ID and metadata propagation. Existing `NewPipeline` decoders continue to receive payload bytes only.

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
- `adapter/ollama` uses the native `POST /api/embed` endpoint with a caller-owned HTTP client. Construction validates configuration without network I/O, model pulls, retries, or health checks.
- Ollama request model and dimension values may override configured defaults. Successful results always identify provider `ollama`, preserve input order, and infer an unspecified dimension from the returned vectors.
- Ollama successful response bodies are capped at 64 MiB and non-success error bodies at 64 KiB.

### Schema & Index State Tables
- `adapter/postgres` and `adapter/sqlite` migrations create an adapter-owned index-space state table to persist index identity and vector dimension metadata across process restarts.
- `adapter/sqlite` candidate bounds apply when `IndexQuery.Limit` is zero; positive limits are never truncated below the requested count.

---

## 5. Context Rendering & Reader Construction

### Neutral Context Rendering
- `retrieval.FormatContext` joins complete, non-empty result blocks with `"\n\n"` by default and adds no instructions, escaping, or prompt-injection filtering. Each block is an optional header, a newline, and the result content.
- Header templates use a single-pass grammar: `%s` selects `Filename`, then `DocumentID`, then `ID`; the first two `%d` placeholders select the inclusive start and end line; `%%` emits a literal percent. Unsupported placeholders and `%d` placeholders beyond the first two remain literal, including placeholder text inside source values. Separators are caller-controlled.
- `WithMaxTokens` accepts a caller-owned token counter. A positive limit counts the complete tentative output, including headers and separators, and retains only the contiguous prefix of blocks that fits. Counter errors and negative counts fail without partial output. Omitted limits are unbounded and never invoke a counter; explicit nonpositive limits return empty output.
- `ResultsFromDocuments` owns `reliquary.context.start_line` and `reliquary.context.end_line`. It removes caller collisions, then stores an inclusive 1-based pair when the normalized chunk text can be resolved strictly forward in the normalized source. Existing JSON metadata persistence carries these values without adapter schema changes.
- Nil results and empty content are skipped without mutating or reordering inputs.

### Bounded Text Readers
- `document.FromReader` preserves the supplied document ID, defaults to `FormatText`, and normalizes a UTF-8 BOM and line endings with `NormalizeText`.
- Inputs must be valid UTF-8 and fit within the 16 MiB default bound or a positive caller-supplied byte limit. Oversized and invalid-UTF-8 errors are classifiable with `errors.Is`.
- `WithFilename` assigns `Document.Title` only; it does not infer parsing or format. `WithMetadata` snapshots caller data and does not inject reserved filename or document-ID keys.
