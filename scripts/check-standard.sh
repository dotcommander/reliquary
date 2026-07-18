#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"

GOWORK=off go build ./...
GOWORK=off go test ./...
GOWORK=off go vet ./...
./scripts/check-boundaries.sh
./scripts/verify-modules.sh
git diff --check
