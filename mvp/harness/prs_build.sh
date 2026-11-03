#!/bin/bash
# Build and unit-test PR commits of the stack in order, each on the incremental build tree: /checkout (the
# prs-tree worktree) is moved to each commit in turn. Usage: prs_build.sh N:SHA [N:SHA ...]
for item in "$@"; do
    n=${item%%:*}; sha=${item#*:}
    git -c safe.directory='*' -C /checkout checkout -q --detach "$sha" || { echo "CHECKOUT_FAILED $sha"; exit 1; }
    echo "##### PR $n $sha"
    bash /art/build_and_test.sh "pr$n-$sha" || { echo "PR_FAILED $n"; exit 1; }
done
echo PRS_DONE
