# Architecture

Reliquary owns retrieval contracts and retrieval-specific implementations.

The dependency direction is:

```text
App facade -> ingestion/chunking/retrieval -> Index contract
                                        ^
                                        |
                         in-memory and adapter indexes
```

- `index` is the candidate-retrieval seam. `index/indextest` is the shared
  behavioral contract for every implementation.
- `chunking`, `document`, `embedding`, `retrieval`, `dedup`, `textutil`, and
  `vector` are the flat public retrieval building blocks.
- `pipeline/ingest` and `pipeline/lexical` retain their pipeline-qualified names
  because they compose multiple public building blocks.
- `indexsink` adapts `pipeline/ingest`'s Sink to the `index` contract, so a
  resumable batch pipeline can persist into reliquary's own storage.
- `adapter` contains explicitly constructed provider and database integrations.
  Adapters may depend on their upstream SDK or driver; core packages may not
  depend on adapters.
- `internal` contains hashing, validation, candidate-selection, and SQL
  implementation details shared inside this module.

The module owns no process lifecycle and performs no hidden database migration.
External clients and database handles are injected and remain caller-owned.

Product memory, graph behavior, and generic application infrastructure are
outside Reliquary's retrieval-only boundary.
