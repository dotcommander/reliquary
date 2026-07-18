# Migrating to Reliquary v0.6

Reliquary v0.6 finalizes the retrieval API paths.

| Before v0.6 | v0.6 |
| --- | --- |
| `pipeline/chunking` | `chunking` |
| `pipeline/document` | `document` |
| `pipeline/embeddings` | `embedding` |
| `pipeline/retrieval` | `retrieval` |
| `primitives/dedup` | `dedup` |
| `primitives/textutil` | `textutil` |
| `primitives/vectors` | `vector` |

`Store`, `WithStore`, and `NewMemoryStore` were removed. Implement `Index` and
construct the facade with `WithIndex`. `WithIndex(nil)` continues to select the
default in-memory index.
