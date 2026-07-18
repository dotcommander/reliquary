#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"

modules=$(find . -name go.mod -not -path './.git/*' -print)
[ "$modules" = "./go.mod" ] || {
	echo "verify modules: expected one root module, found:" >&2
	printf '%s\n' "$modules" >&2
	exit 1
}

GOWORK=off go mod verify
GOWORK=off go test ./...
GOWORK=off go vet ./...
echo "verify modules: ok"
