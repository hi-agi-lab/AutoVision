#!/usr/bin/env bash
# Build (and optionally push) a single-commit public snapshot.
#
# The public repository carries release snapshots only. The private history
# contains pre-remediation commits and must never be pushed (see the pre-push
# hook and docs/release_process.md). This script creates a root commit from
# the exact tree of --ref, verifies that the tree is unchanged and that the
# commit has no parent, then optionally pushes the branch and annotated tag.
#
# Usage:
#   scripts/publish_clean_cut.sh [--ref HEAD] [--branch release-public] \
#       [--tag vX.Y.Z] [--remote <name> --push]
#
# Examples:
#   scripts/publish_clean_cut.sh                    # local snapshot + tag
#   scripts/publish_clean_cut.sh --remote publish --push
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

REF="HEAD"
BRANCH="release-public"
TAG=""
REMOTE=""
PUSH=0

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ref)
            [ $# -ge 2 ] || { echo "--ref needs a value" >&2; exit 2; }
            REF="$2"
            shift 2
            ;;
        --branch)
            [ $# -ge 2 ] || { echo "--branch needs a value" >&2; exit 2; }
            BRANCH="$2"
            shift 2
            ;;
        --tag)
            [ $# -ge 2 ] || { echo "--tag needs a value" >&2; exit 2; }
            TAG="$2"
            shift 2
            ;;
        --remote)
            [ $# -ge 2 ] || { echo "--remote needs a value" >&2; exit 2; }
            REMOTE="$2"
            shift 2
            ;;
        --push)
            PUSH=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -n "$(git status --porcelain)" ]; then
    echo "worktree is not clean; commit the release tree before snapshotting" >&2
    exit 1
fi

REF_COMMIT=$(git rev-parse --verify "$REF^{commit}")
VERSION=$(git show "$REF_COMMIT:src/autovision/_version.py" | sed -n 's/^__version__ = "\([^"]*\)"/\1/p')
[ -n "$VERSION" ] || { echo "cannot read version from $REF:src/autovision/_version.py" >&2; exit 1; }
TAG=${TAG:-v$VERSION}
if [ "$TAG" != "v$VERSION" ]; then
    echo "tag '$TAG' does not match package version 'v$VERSION'" >&2
    exit 1
fi

tracked_ignored=$(git ls-files -ci --exclude-standard || true)
if [ -n "$tracked_ignored" ]; then
    echo "tracked files match .gitignore; remove them from the index first:" >&2
    echo "$tracked_ignored" >&2
    exit 1
fi

TREE=$(git rev-parse "$REF_COMMIT^{tree}")

if git ls-tree -r --name-only "$REF_COMMIT" | grep -q '^\.autovision-cache/'; then
    echo "cache artifacts are present in the snapshot tree" >&2
    exit 1
fi

COMMIT=$(git commit-tree "$TREE" -m "AutoVision $TAG (public snapshot)")
git branch -f "$BRANCH" "$COMMIT"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    existing=$(git rev-parse "refs/tags/$TAG^{commit}")
    if [ "$existing" != "$COMMIT" ]; then
        echo "tag $TAG already points at $existing, not $COMMIT" >&2
        exit 1
    fi
else
    git tag -a "$TAG" -m "AutoVision $TAG" "$COMMIT"
fi

parent_count=$(git rev-list --parents -n1 "$COMMIT" | awk '{print NF}')
if [ "$parent_count" -ne 1 ]; then
    echo "snapshot commit unexpectedly has parents" >&2
    exit 1
fi
if ! git diff --quiet "$REF_COMMIT" "$COMMIT"; then
    echo "snapshot tree differs from $REF" >&2
    exit 1
fi

echo "snapshot commit: $COMMIT"
echo "tree:            $TREE"
echo "branch:          $BRANCH"
echo "tag:             $TAG"
echo "parents:         none"

if [ "$PUSH" -eq 1 ]; then
    [ -n "$REMOTE" ] || { echo "--remote is required with --push" >&2; exit 2; }
    if ! git push "$REMOTE" "$BRANCH:main"; then
        echo "push rejected. If this replaces a previous public snapshot, use:" >&2
        echo "  git push --force-with-lease $REMOTE $BRANCH:main" >&2
        exit 1
    fi
    git push "$REMOTE" "refs/tags/$TAG"
    echo "pushed $BRANCH:main and $TAG to $REMOTE"
else
    echo "not pushed; re-run with --remote <name> --push after verification"
fi
