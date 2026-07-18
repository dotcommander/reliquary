#!/bin/sh
set -eu

script=$(CDPATH= cd -- "$(dirname "$0")" && pwd)/$(basename "$0")

usage() {
	echo "usage: release.sh plan <vX.Y.Z>" >&2
	echo "       release.sh apply <plan.json>" >&2
	exit 2
}

[ "$#" -ge 1 ] || usage
command=$1
shift

case "$command" in
plan)
	[ "$#" -eq 1 ] || usage
	python3 - "$1" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

SEMVER = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")


def fail(message):
    print(f"release.sh: {message}", file=sys.stderr)
    raise SystemExit(1)


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


root = Path.cwd()
go_mod = root / "go.mod"
if not go_mod.is_file() or not (root / ".git").exists():
    fail("run from the reliquary repository root")

version = sys.argv[1]
if not SEMVER.fullmatch(version):
    fail(f"invalid version: {version}")

module_path = next(
    (line.split()[1] for line in go_mod.read_text().splitlines() if line.startswith("module ")),
    None,
)
if module_path is None:
    fail("missing module directive in go.mod")

if subprocess.run(
    ["git", "rev-parse", "-q", "--verify", f"refs/tags/{version}"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0:
    fail(f"tag already exists: {version}")

plan = {
    "schema": "reliquary-release-plan/v1",
    "head": git("rev-parse", "HEAD"),
    "modules": [
        {
            "dir": ".",
            "path": module_path,
            "version": version,
            "tag": version,
            "dependencies": [],
            "requirement_updates": [],
        }
    ],
}
json.dump(plan, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY
	;;
apply)
	[ "$#" -eq 1 ] || usage
	plan=$1
	[ -f "$plan" ] || { echo "release.sh: plan file not found: $plan" >&2; exit 1; }
	[ -f go.mod ] && [ -d .git ] || { echo "release.sh: run from the reliquary repository root" >&2; exit 1; }

	plan_abs=$(cd "$(dirname "$plan")" && pwd)/$(basename "$plan")
	plan_rel=$(python3 - "$plan_abs" <<'PY'
from pathlib import Path
import sys
try:
    print(Path(sys.argv[1]).resolve().relative_to(Path.cwd().resolve()))
except ValueError:
    pass
PY
	)
	dirty=$(git status --porcelain --untracked-files=all | awk -v plan="$plan_rel" '
		plan == "" || substr($0, 4) != plan || substr($0, 1, 2) != "??"
	')
	if [ -n "$dirty" ]; then
		echo "release.sh: working tree must be clean except for the untracked plan file" >&2
		printf '%s\n' "$dirty" >&2
		exit 1
	fi

	# Validate the immutable release identity before creating a disposable
	# worktree or changing the live checkout.
	python3 - "$plan_abs" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

SEMVER = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")


def fail(message):
    print(f"release.sh: {message}", file=sys.stderr)
    raise SystemExit(1)


try:
    plan = json.loads(Path(sys.argv[1]).read_text())
except (OSError, json.JSONDecodeError) as error:
    fail(f"invalid plan: {error}")
if plan.get("schema") != "reliquary-release-plan/v1":
    fail("unsupported plan schema")

head = subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE
).stdout.strip()
if plan.get("head") != head:
    fail(f"plan HEAD {plan.get('head')} does not match checkout HEAD {head}")

modules = plan.get("modules")
if not isinstance(modules, list) or len(modules) != 1:
    fail("plan must contain exactly the root module")
module = modules[0]
if module.get("dir") != ".":
    fail("plan module must be the repository root")
version = module.get("version")
if not isinstance(version, str) or not SEMVER.fullmatch(version):
    fail("invalid version in plan")
if module.get("tag") != version:
    fail(f"invalid tag in plan: {module.get('tag')}")
if module.get("dependencies") != [] or module.get("requirement_updates") != []:
    fail("single-module plan must not contain internal dependency updates")

module_path = next(
    (line.split()[1] for line in Path("go.mod").read_text().splitlines() if line.startswith("module ")),
    None,
)
if module_path != module.get("path"):
    fail("module path changed for repository root")
if subprocess.run(
    ["git", "rev-parse", "-q", "--verify", f"refs/tags/{version}"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0:
    fail(f"tag already exists: {version}")
PY

	# Re-plan and require exact semantic equality so a hand-edited plan cannot
	# change the tag, module identity, or source commit.
	python3 - "$plan_abs" "$script" <<'PY'
import json
import subprocess
import sys
from pathlib import Path


def fail(message):
    print(f"release.sh: {message}", file=sys.stderr)
    raise SystemExit(1)


try:
    supplied = json.loads(Path(sys.argv[1]).read_text())
    version = supplied["modules"][0]["version"]
except (OSError, json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
    fail(f"invalid plan: {error}")
result = subprocess.run(
    [sys.argv[2], "plan", version],
    text=True,
    stdout=subprocess.PIPE,
)
if result.returncode != 0:
    raise SystemExit(result.returncode)
if supplied != json.loads(result.stdout):
    fail("plan does not match the current repository")
PY

	preflight=$(mktemp -d "${TMPDIR:-/tmp}/reliquary-release.XXXXXX")
	cleanup() {
		git worktree remove --force "$preflight" >/dev/null 2>&1 || true
		rm -rf "$preflight"
	}
	trap cleanup EXIT HUP INT TERM

	git worktree add --detach -q "$preflight" HEAD
	(
		cd "$preflight"
		export GOWORK=off
		go mod tidy
		go mod verify
		go test -race ./...
		go build ./...
		go vet ./...
		./scripts/check-boundaries.sh
		./scripts/verify-modules.sh
	)

	# Copy the exact verified module state back. A tidy-only change is committed
	# before tagging so the tag always names the state that passed preflight.
	for name in go.mod go.sum; do
		if [ -f "$preflight/$name" ]; then
			cp "$preflight/$name" "$name"
		elif [ -e "$name" ]; then
			rm "$name"
		fi
		if [ -e "$name" ] || git ls-files --error-unmatch "$name" >/dev/null 2>&1; then
			git add -A -- "$name"
		fi
	done
	if ! git diff --quiet --cached; then
		version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["modules"][0]["version"])' "$plan_abs")
		git commit -m "release: $version"
	fi

	version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["modules"][0]["version"])' "$plan_abs")
	git tag -a "$version" -m "Release $version"
	printf '%s\n' "$version"
	echo "release.sh: local commit and tag created; push intentionally not run"
	;;
*) usage ;;
esac
