#!/bin/bash
# Commit what is staged, with the pre-commit hooks, while other edits exist: pre-commit's own stash of
# unstaged changes needs a writable ~/.cache, which this cluster's home is not. Copies every modified
# file to a scratch dir, resets the worktree to the index, commits, then copies the files back.
# Usage: commit_staged.sh REPO -m SUBJECT [-m BODY]   (extra args go to git commit; -s is added)
set -uo pipefail
repo=$1; shift
S=$(mktemp -d "${TMPDIR:-/tmp}/commit_staged.XXXX")
cd "$repo"
git diff --name-only > "$S/files.txt"
while read -r f; do mkdir -p "$S/$(dirname "$f")"; cp "$f" "$S/$f"; done < "$S/files.txt"
git checkout -q -- $(cat "$S/files.txt")
git commit -q -s "$@"; rc=$?
while read -r f; do cp "$S/$f" "$f"; done < "$S/files.txt"
echo "commit rc=$rc; restored $(wc -l < "$S/files.txt") files from $S"
exit $rc
