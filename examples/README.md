# Retrieval examples

Run an example from the repository root:

```sh
GOWORK=off go run ./examples/quickstart-rag
```

- `quickstart-rag`: in-memory retrieval facade.
- `search-pipeline`: batch search with RRF, external reranking, MMR, and typed explanations.
- `provider-openai-wiring`: injected OpenAI embedding adapter.
- `postgres-index`: explicit PostgreSQL retrieval migration.
- `rag-ingest-retrieve`: manual hybrid retrieval and MMR.
- `retrieval-calibration-tune`: retrieval evaluation and weight tuning.

Run the complete facade search pipeline after the quickstart:

```sh
GOWORK=off go run ./examples/search-pipeline
```
