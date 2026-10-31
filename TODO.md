# Efficiency recommendations

## Current slice: six review recommendations

- [x] Skip no-query document tokenization and unused unbudgeted context prefixes.
- [x] Score non-explained RRF without a discarded hybrid sort.
- [x] Batch PostgreSQL upserts and reuse current SQLite FTS migration state.
- [x] Snapshot only matching semantic-memory scopes.
- [x] Run the build and focused tests; review the combined diff.

Verification: `GOWORK=off GOPROXY=off go build ./...` passed. Focused tests passed for root, retrieval, adapter/postgres, adapter/sqlite, and judgment/memory. Test-fixture fixes and final source refinements were rechecked only in affected packages. Combined diff review and `git diff --check` passed. PostgreSQL uses fake-transport checks, SQLite uses disposable databases, and semantic eviction coverage includes delayed callbacks. No live PostgreSQL/provider checks, full suite, or performance benchmarks were run. SQLite retains corpus drift validation; only redundant FTS rebuilding is skipped. Semantic snapshots select live matching scopes without changing TTL/LRU behavior.

User authority: "proceed. all recs." Implementation remains uncommitted. Prior completed slice below is preserved.

## Prior slice

Scope: remove duplicate shared-index sorting, safely push down SQLite candidate bounds, and add caller-opt-in bounded PostgreSQL lexical retrieval. Preserve existing default ranking and unrelated worktree edits. MMR budgets remain caller-owned; document their tradeoff.

- [x] Remove duplicate sorting and add regression coverage.
- [x] Push SQLite bounds into SQL only for exact preselection; retain numeric postfiltering.
- [x] Explain candidate-limit versus TopK/MMR behavior.
- [x] Add opt-in PostgreSQL lexical candidate bounds and tests.
- [x] Run one build, focused tests, and consolidated diff review.

Verification: `GOWORK=off GOPROXY=off go build ./...` passed. Focused tests passed in retrieval, internal/indexutil, adapter/sqlite, and adapter/postgres. PostgreSQL checks use local SQL-shape/configuration/fake-transport tests, not a live server. No production timing, recall, or query-plan claims are made. Numeric SQLite bounds remain postfiltered; PostgreSQL full-text mode and MMR candidate budgets remain caller-controlled.
