#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
tmp=${TMPDIR:-/tmp}/reliquary-boundary-test.$$
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

fail() {
	echo "check_boundaries_test.sh: $*" >&2
	exit 1
}

new_repo() {
	repo=$1
	mkdir -p "$repo/scripts" "$repo/adapter/postgres" "$repo/examples/demo" "$repo/deps/pgx" "$repo/deps/wrapper"
	cp "$script_dir/check-boundaries.sh" "$repo/scripts/check-boundaries.sh"
	cat > "$repo/go.mod" <<'EOF'
module github.com/dotcommander/reliquary

go 1.26.0

require (
	example.invalid/wrapper v0.0.0
	github.com/jackc/pgx/v5 v5.0.0
)

replace example.invalid/wrapper => ./deps/wrapper

replace github.com/jackc/pgx/v5 => ./deps/pgx
EOF
	cat > "$repo/deps/pgx/go.mod" <<'EOF'
module github.com/jackc/pgx/v5

go 1.26.0
EOF
	cat > "$repo/deps/pgx/pgx.go" <<'EOF'
package pgx
EOF
	cat > "$repo/deps/wrapper/go.mod" <<'EOF'
module example.invalid/wrapper

go 1.26.0

require github.com/jackc/pgx/v5 v5.0.0
EOF
	cat > "$repo/deps/wrapper/wrapper.go" <<'EOF'
package wrapper

import _ "github.com/jackc/pgx/v5"
EOF
	cat > "$repo/adapter/postgres/postgres.go" <<'EOF'
package postgres

import _ "github.com/jackc/pgx/v5"
EOF
	cat > "$repo/examples/demo/main.go" <<'EOF'
package main

import _ "github.com/jackc/pgx/v5"
EOF
}

assert_rejected() {
	repo=$1
	case_name=$2
	if (cd "$repo" && ./scripts/check-boundaries.sh) >"$repo/output" 2>&1; then
		fail "$case_name unexpectedly passed"
	fi
	if ! rg -q 'core packages must not depend on adapters or provider/database drivers' "$repo/output"; then
		fail "$case_name did not report the dependency boundary"
	fi
}

mkdir -p "$tmp"

repo=$tmp/clean
new_repo "$repo"
cat > "$repo/reliquary.go" <<'EOF'
package reliquary
EOF
if ! (cd "$repo" && ./scripts/check-boundaries.sh) >"$repo/output" 2>&1; then
	fail "clean module failed: $(tail -n 1 "$repo/output")"
fi

repo=$tmp/indexsink
new_repo "$repo"
cat > "$repo/reliquary.go" <<'EOF'
package reliquary
EOF
mkdir -p "$repo/indexsink"
cat > "$repo/indexsink/sink.go" <<'EOF'
package indexsink

import _ "github.com/jackc/pgx/v5"
EOF
assert_rejected "$repo" "indexsink direct dependency"

repo=$tmp/internal
new_repo "$repo"
cat > "$repo/reliquary.go" <<'EOF'
package reliquary
EOF
mkdir -p "$repo/internal/store"
cat > "$repo/internal/store/store.go" <<'EOF'
package store

import _ "github.com/jackc/pgx/v5"
EOF
assert_rejected "$repo" "internal direct dependency"

repo=$tmp/transitive
new_repo "$repo"
cat > "$repo/reliquary.go" <<'EOF'
package reliquary

import _ "example.invalid/wrapper"
EOF
assert_rejected "$repo" "transitive wrapper dependency"

echo "check_boundaries_test.sh: PASS"
