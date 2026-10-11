# Reliquary Examples and Demos

Run any example from the repository root:

```sh
GOWORK=off go run ./examples/<example-dir>
```

### High-Level Facade

- **`quickstart-rag`**: Zero-config in-memory hybrid retrieval facade (`documents -> Ingest -> Search`).
- **`metadata-filtering`**: Structured metadata filtering (`WithFilter`) by JSON scalars and reserved fields (`filename`, `document_id`).
- **`search-pipeline`**: Full search pipeline with RRF fusion, external reranking, MMR diversification, and typed explanations.
- **`rag-ingest-retrieve`**: Manual hybrid retrieval pipeline with BM25, vector search, and MMR reranking.
- **`retrieval-calibration-tune`**: Retrieval evaluation and hybrid weight calibration against gold relevance sets.

### Index Providers (Storage Backends)

- **`sqlite-index`**: SQLite index adapter (`adapter/sqlite`) with FTS5 lexical BM25 matching, vector cosine distance, and schema migration.
- **`postgres-index`**: PostgreSQL index adapter (`adapter/postgres`) with pgvector and caller-owned pgxpool connection management.

### Embedder Providers & Caching

- **`provider-openai-wiring`**: OpenAI embedding adapter (`adapter/openai`) configuration with dimension bounds.
- **`provider-ollama-wiring`**: Ollama embedding adapter (`adapter/ollama`) with native `/api/embed` batch vectors and endpoint fallback.
- **`embed-server-quickstart`**: Local embeddings server adapter with nomic task prefixes and Matryoshka dimensions.
- **`embedding-cache`**: Transparent embedding cache decorator (`embedding/cache` + `inmem.Store`) eliminating redundant provider calls.

### Judgment Providers (Evaluation & Reasoning)

- **`judgment-quickstart`**: Typed Choice/Score/Noul evaluation core with TTL memory cache and reflex-to-deliberate escalation.
- **`judgment-semantic-quickstart`**: Semantic judgment memory scanning cosine similarity over vectors with local embedding models.
- **`provider-jev-judge`**: Jev typed evaluation adapter (`adapter/jev`) mapping Choice, Score, and Noul primitives.
- **`provider-structjudge-wiring`**: Schema-forced structured LLM judge adapter (`adapter/structjudge`) with JSON schema completion validation.

### Retrieval, Chunking, & Ingest Pipelines

- **`filesystem-ingest`**: Batch directory tree scanner (`pipeline/ingest/fs`) with file filtering and pipeline streaming into Reliquary.
- **`chunking-strategies`**: HeadingAware, SentenceBoundary, SmartBoundary, and Recursive text chunking strategies.
- **`dedup-pipeline`**: Ingestion deduplication using NormalizedHash (exact/whitespace) and SimHash (near-duplicate clustering).
- **`vector-clustering-pq`**: K-Means vector clustering (`vector/clustering`) and Product Quantization (`vector/pq`) vector compression.
