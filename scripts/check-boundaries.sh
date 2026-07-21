#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"

fail() {
	echo "boundary check: $1" >&2
	exit 1
}

module=$(GOWORK=off go list -m -f '{{.Path}}') || fail "could not identify the root module"
[ -n "$module" ] || fail "root module path is empty"

packages=$(GOWORK=off go list -f '{{.ImportPath}}' ./...) || fail "could not discover module packages"
core_packages=
for package in $packages; do
	case "$package" in
		"$module"/adapter|"$module"/adapter/*|"$module"/examples|"$module"/examples/*)
			;;
		*)
			core_packages="$core_packages $package"
			;;
	esac
done
[ -n "$core_packages" ] || fail "no core packages discovered"

# Package paths cannot contain whitespace, so intentional word splitting keeps
# this POSIX-shell compatible while checking every core package's full closure.
core_dependencies=$(GOWORK=off go list -deps -f '{{.ImportPath}}' $core_packages) || fail "could not inspect core package dependencies"
for dependency in $core_dependencies; do
	case "$dependency" in
		"$module"/adapter|"$module"/adapter/*|github.com/jackc/pgx|github.com/jackc/pgx/*|github.com/openai/openai-go|github.com/openai/openai-go/*|github.com/pgvector/pgvector-go|github.com/pgvector/pgvector-go/*|modernc.org/sqlite|modernc.org/sqlite/*)
			fail "core packages must not depend on adapters or provider/database drivers"
			;;
	esac
done

if rg -n '^package main$' --glob '*.go' --glob '!examples/**' . >/dev/null; then
	fail "package main is allowed only under examples"
fi

if rg -n 'github.com/dotcommander/reliquary/(memory|graph|config|runtime|storage|tools/[^/]+|contracts/(llm|events|media|observability|storage|webfetch|workflow))([/" ]|$)' --glob '*.go' --glob '*.md' --glob '!docs/MIGRATION-v0.5.md' . >/dev/null; then
	fail "stale removed Reliquary import path found"
fi

if rg -n 'github.com/dotcommander/reliquary/(pipeline/(chunking|document|embeddings|retrieval)|primitives/(dedup|textutil|vectors))([/" ]|$)' --glob '*.go' --glob '*.md' --glob '!docs/MIGRATION-v0.6.md' . >/dev/null; then
	fail "stale pre-v0.6 Reliquary import path found"
fi

if rg -n 'github.com/dotcommander/reliquary/indexsink([[:space:][:punct:]]|$)' --glob '*.go' --glob '*.md' --glob '!docs/MIGRATION-v0.10.md' . >/dev/null; then
	fail "retired indexsink import path found"
fi

echo "boundary check: ok"
