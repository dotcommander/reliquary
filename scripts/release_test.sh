#!/bin/sh
set -eu

release_script=$(CDPATH= cd -- "$(dirname "$0")" && pwd)/release.sh
release_script_dir=$(dirname "$release_script")
tmp=${TMPDIR:-/tmp}/reliquary-release-test.$$
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

fail() {
	echo "release_test.sh: $*" >&2
	exit 1
}

new_repo() {
	repo=$1
	mkdir -p "$repo/bin" "$repo/scripts"
	git -C "$repo" init -q
	git -C "$repo" config user.name "Reliquary Release Test"
	git -C "$repo" config user.email "release-test@example.invalid"

	cat > "$repo/go.mod" <<'EOF'
module github.com/dotcommander/reliquary

go 1.26.0
EOF
	cat > "$repo/main.go" <<'EOF'
package reliquary
EOF
	cat > "$repo/bin/go" <<'EOF'
#!/bin/sh
if [ "${RELEASE_TEST_TIDY_CHANGE:-}" = 1 ] && [ "$1 $2" = "mod tidy" ]; then
	printf '\n// release-test tidy change\n' >> go.mod
fi
exit 0
EOF
	chmod +x "$repo/bin/go"
	cp "$release_script_dir/check-boundaries.sh" "$repo/scripts/check-boundaries.sh"
	cp "$release_script_dir/verify-modules.sh" "$repo/scripts/verify-modules.sh"
	git -C "$repo" add go.mod main.go bin/go scripts
	git -C "$repo" commit -q -m initial
}

assert_pristine() {
	repo=$1
	head=$2
	[ "$(git -C "$repo" rev-parse HEAD)" = "$head" ] || fail "failed preflight created a commit"
	git -C "$repo" diff --quiet || fail "failed preflight rewrote tracked files"
	git -C "$repo" diff --cached --quiet || fail "failed preflight staged files"
}

mkdir -p "$tmp"

# Planning the actual one-module shape is read-only and records the root module.
repo=$tmp/plan
new_repo "$repo"
before=$(git -C "$repo" status --porcelain)
(cd "$repo" && "$release_script" plan v0.8.0 > plan.json)
[ "$(git -C "$repo" status --porcelain | sed '/?? plan.json/d')" = "$before" ] || fail "plan mutated repository"
python3 - "$repo/plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
assert plan["schema"] == "reliquary-release-plan/v1"
assert len(plan["modules"]) == 1
module = plan["modules"][0]
assert module == {
    "dependencies": [],
    "dir": ".",
    "path": "github.com/dotcommander/reliquary",
    "requirement_updates": [],
    "tag": "v0.8.0",
    "version": "v0.8.0",
}
PY

# Invalid versions and existing root tags are rejected without mutation.
repo=$tmp/rejected
new_repo "$repo"
head=$(git -C "$repo" rev-parse HEAD)
if (cd "$repo" && "$release_script" plan 0.8.0 >plan.json 2>error); then
	fail "planner accepted a non-v semver"
fi
rg -q 'invalid version: 0.8.0' "$repo/error" || fail "invalid-version error missing"
git -C "$repo" tag -a v0.8.0 -m collision
if (cd "$repo" && "$release_script" plan v0.8.0 >plan.json 2>error); then
	fail "planner accepted a tag collision"
fi
rg -q 'tag already exists: v0.8.0' "$repo/error" || fail "collision error missing"
assert_pristine "$repo" "$head"

# Apply rejects a plan for an older commit before any verification or mutation.
repo=$tmp/stale
new_repo "$repo"
(cd "$repo" && "$release_script" plan v0.8.0 > plan.json)
printf '\n' >> "$repo/main.go"
git -C "$repo" add main.go
git -C "$repo" commit -q -m advance
head=$(git -C "$repo" rev-parse HEAD)
if (cd "$repo" && PATH="$repo/bin:$PATH" "$release_script" apply plan.json >"$tmp/stale-output" 2>"$tmp/stale-error"); then
	fail "apply accepted a stale plan"
fi
rg -q 'does not match checkout HEAD' "$tmp/stale-error" || fail "stale-plan error missing"
assert_pristine "$repo" "$head"

# Failed disposable verification leaves the live repository and tag set untouched.
repo=$tmp/verify-failure
new_repo "$repo"
cat > "$repo/bin/go" <<'EOF'
#!/bin/sh
exit 1
EOF
chmod +x "$repo/bin/go"
git -C "$repo" add bin/go
git -C "$repo" commit -q --amend --no-edit
(cd "$repo" && "$release_script" plan v0.8.0 > plan.json)
head=$(git -C "$repo" rev-parse HEAD)
if (cd "$repo" && PATH="$repo/bin:$PATH" "$release_script" apply plan.json >"$tmp/verify-output" 2>"$tmp/verify-error"); then
	fail "apply accepted failed verification"
fi
assert_pristine "$repo" "$head"
if git -C "$repo" rev-parse -q --verify refs/tags/v0.8.0 >/dev/null; then
	fail "failed verification created a tag"
fi

# Repository-specific verification rejects a nested module that generic root
# package tests would skip.
repo=$tmp/nested-module
new_repo "$repo"
mkdir -p "$repo/nested"
cat > "$repo/nested/go.mod" <<'EOF'
module example.invalid/nested

go 1.26.0
EOF
cat > "$repo/nested/nested.go" <<'EOF'
package nested
EOF
git -C "$repo" add nested
git -C "$repo" commit -q -m 'add nested module'
(cd "$repo" && "$release_script" plan v0.8.0 > plan.json)
head=$(git -C "$repo" rev-parse HEAD)
if (cd "$repo" && PATH="$repo/bin:$PATH" "$release_script" apply plan.json >"$tmp/nested-output" 2>"$tmp/nested-error"); then
	fail "apply accepted a nested module"
fi
rg -q 'expected one root module' "$tmp/nested-error" || fail "nested-module error missing"
assert_pristine "$repo" "$head"
if git -C "$repo" rev-parse -q --verify refs/tags/v0.8.0 >/dev/null; then
	fail "nested module created a tag"
fi

# A clean verified module tags the planned commit without manufacturing a commit.
repo=$tmp/apply
new_repo "$repo"
(cd "$repo" && "$release_script" plan v0.8.0 > plan.json)
head=$(git -C "$repo" rev-parse HEAD)
(
	cd "$repo"
	PATH="$repo/bin:$PATH" "$release_script" apply plan.json >"$tmp/apply-output"
)
[ "$(git -C "$repo" rev-parse HEAD)" = "$head" ] || fail "clean apply created an unnecessary commit"
[ "$(git -C "$repo" rev-list -n 1 v0.8.0)" = "$head" ] || fail "tag does not point at planned commit"
rg -q '^v0.8.0$' "$tmp/apply-output" || fail "apply output omitted tag"

# If tidy changes module files, apply commits the verified state before tagging it.
repo=$tmp/tidy-change
new_repo "$repo"
(cd "$repo" && "$release_script" plan v0.8.0 > plan.json)
head=$(git -C "$repo" rev-parse HEAD)
(
	cd "$repo"
	RELEASE_TEST_TIDY_CHANGE=1 PATH="$repo/bin:$PATH" "$release_script" apply plan.json >"$tmp/tidy-output"
)
new_head=$(git -C "$repo" rev-parse HEAD)
[ "$new_head" != "$head" ] || fail "tidy change was not committed"
[ "$(git -C "$repo" rev-list -n 1 v0.8.0)" = "$new_head" ] || fail "tag does not include tidy commit"
rg -q 'release-test tidy change' "$repo/go.mod" || fail "verified tidy state was not copied back"

echo "release_test.sh: PASS"
