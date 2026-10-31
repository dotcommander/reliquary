# Efficiency recommendations

## Current slice: offset mapping and bounded selection

User authority: "proceed. all recs." following the source-only efficiency review. Preserve existing uncommitted changes; no new dependencies or automatic commit.

- [x] Reuse source-offset work for ingestion line mapping and chunkservice rune projection.
- [x] Select bounded index candidates after full-pool calibration without a full sort.
- [x] Batch replacement FTS cleanup while preserving SQLite orphan recovery and rollback.
- [x] Skip unused similarity work for non-traced relevance-only MMR.
- [x] Resolve tokenization/cache-store hypotheses against current ownership and workload evidence.
- [x] Run one build, focused tests, and one consolidated diff review.

Verification: `GOWORK=off GOPROXY=off go build ./...` passed. Focused tests passed for root, chunking, retrieval, chunkservice, internal/indexutil, index/inmem, and adapter/sqlite. An initial retrieval test-fixture compile failure (integer in string metadata) was corrected; affected tests passed afterward. Consolidated review replaced the initial per-rune offset table with immutable sparse checkpoints to bound index memory; every-byte equivalence tests cover checkpoint boundaries, malformed UTF-8, multibyte interiors and newline-heavy inputs. SQLite checks use disposable databases and preserve orphan FTS cleanup and retained-ID rollback. Full suite, live providers/databases and performance benchmarks were not run; runtime gains remain unmeasured. All changes remain uncommitted.

Hypothesis disposition: no speculative feature/cache-store API added. The built-in embedding store is in-process (mutex/map, no remote round trips); the decorator already deduplicates inputs, batches provider misses, and optionally coalesces identical requests. Remote store latency is caller-specific and unmeasured. Index and facade ranking use different calibration pools and potentially different weights; token-feature reuse needs measured tokenization cost before adding a cross-layer ownership contract. These remain measurement-dependent, not implemented optimizations.

## Previous slice: six review recommendations

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
