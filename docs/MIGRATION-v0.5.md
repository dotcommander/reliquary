# Migrating to Reliquary v0.5

Reliquary v0.5 narrows the repository to retrieval. Use this ownership map for
packages removed from v0.4.

| Removed surface | Owner or replacement |
| --- | --- |
| `memory/...` | Application-owned memory behavior |
| `graph/...` | Memory product packages, or application-owned graph behavior |
| `contracts/llm` | Application-owned provider contracts |
| `config/...` | Application-owned configuration |
| `runtime/cli`, `runtime/web` | Application-owned CLI and web behavior |
| `storage/postgres`, `storage/sqlite`, `storage/migrate`, `storage/redis` | Caller-selected database and migration libraries |
| `platform/fs`, logging | Standard-library or application-owned infrastructure |
| repository-local tooling | Removed without replacement |
| generic events, workflow, media, web-fetch, registry, and testkit packages | Removed without replacement |

Retrieval integrations now live at their final paths:

- `github.com/dotcommander/reliquary/adapter/openai`
- `github.com/dotcommander/reliquary/adapter/postgres`
- `github.com/dotcommander/reliquary/adapter/sqlite`

PostgreSQL and SQLite adapters require an explicit `Migrate(ctx)` call. Their
constructors do not connect or create schema.

`Store`, `WithStore`, and `NewMemoryStore` remain as deprecated compatibility
APIs for v0.5 only. Implement `Index` and use `WithIndex` before upgrading to
v0.6.
