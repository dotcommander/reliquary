#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"

fail() {
	echo "boundary check: $1" >&2
	exit 1
}

core_imports=$(go list -f '{{.ImportPath}} {{join .Imports " "}}' . ./chunking/... ./dedup/... ./document/... ./embed/... ./embedding/... ./index/... ./pipeline/... ./retrieval/... ./textutil/... ./vector/... ./contracts/... 2>/dev/null)
if printf '%s\n' "$core_imports" | rg -q 'github.com/dotcommander/reliquary/adapter/|github.com/jackc/pgx|github.com/openai/openai-go|github.com/pgvector/pgvector-go|modernc.org/sqlite'; then
	fail "core packages must not import adapters or provider/database drivers"
fi

if rg -n '^package main$' --glob '*.go' --glob '!examples/**' . >/dev/null; then
	fail "package main is allowed only under examples"
fi

if rg -n 'github.com/dotcommander/reliquary/(memory|graph|config|runtime|storage|tools/[^/]+|contracts/(llm|events|media|observability|storage|webfetch|workflow))([/" ]|$)' --glob '*.go' --glob '*.md' --glob '!docs/MIGRATION-v0.5.md' . >/dev/null; then
	fail "stale removed Reliquary import path found"
fi

if rg -n 'github.com/dotcommander/reliquary/(pipeline/(chunking|document|embeddings|retrieval)|primitives/(dedup|textutil|vectors))([/" ]|$)' --glob '*.go' --glob '*.md' --glob '!docs/MIGRATION-v0.6.md' . >/dev/null; then
	fail "stale pre-v0.6 Reliquary import path found"
fi

echo "boundary check: ok"
