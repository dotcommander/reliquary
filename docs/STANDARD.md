# Repository standard

Reliquary is retrieval-only. Core packages remain provider- and driver-neutral;
provider and database dependencies belong under `adapter/`.

Public constructors use typed configuration and injected upstream types.
Database adapter constructors perform validation only, and schema creation is
an explicit `Migrate(ctx)` operation. Candidate retrieval must remain bounded
for positive limits and deterministically order equal scores by result ID.

Every Index implementation must run `index/indextest`, and every Embedder
implementation must run `embedding/embeddingtest`. Successful embedding batches
contain exactly one positive-dimension, finite vector per input in the same
order. Changes must pass the root build, test, vet, race, boundary, module, and
whitespace checks documented in the root README.

An `Index` owns two distinct write contracts: `Upsert` merges results by result
ID, while `ReplaceDocuments` atomically replaces complete document revisions
across one batch. An empty revision deletes that document. Validation or write
failure must preserve every prior revision in the batch.

The first non-nil result establishes an Index identity and the first embedded
result establishes its vector dimension. Deletes and replacements retain both
latches even when no rows remain; only `Reset` clears them. Persistent adapters
store this state in migration-owned internal tables and backfill legacy rows.
Index writes and searches reject NaN and infinities without changing stored
data or these latches; empty embeddings remain valid for lexical-only results.
Document deletion uses exact `Result.DocumentID` equality and never infers
ownership from result ID prefixes. SQLite applies its configured/default
candidate bound when `IndexQuery.Limit` is zero and never truncates a positive
requested limit below that value.

Index filters accept only JSON scalars. Missing metadata keys never match;
explicit null, strings, and booleans are type-exact, while finite numbers use
exact JSON numeric equality across accepted Go numeric types. Reserved result
fields match strings only.
