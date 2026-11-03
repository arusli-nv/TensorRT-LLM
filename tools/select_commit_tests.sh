#!/bin/bash
# Per-commit unit-test selection: the test files a commit touches, plus every test file under tests/unittest (at that
# commit) that imports a changed product module of the FT core (FOCUS: moe_load_balancer, peer_loss_recovery,
# mpi_session, executor/recovery/*, nvlink_one_sided), plus every test file naming a function whose def line the
# commit changes in tensorrt_llm/ or calls as a method on a changed line (names < 8 chars or dunder skipped). A module matches on its dotted path, or on its basename (a
# recovery module's basename only after 'recovery.' or in an import from the recovery package). Usage: select_commit_tests.sh REPO COMMIT
set -euo pipefail
repo=$1; c=$2
focus='(moe_load_balancer|peer_loss_recovery|mpi_session|executor/recovery/[^/]+|nvlink_one_sided)\.py$'
generic='^(__init__|config|utils|common|base|types|interface|api|_utils|logger|mapping|version)$'
{
    git -C "$repo" diff-tree --no-commit-id --name-only -r --diff-filter=AMR "$c" -- 'tests/unittest/*test_*.py'
    git -C "$repo" diff-tree --no-commit-id --name-only -r --diff-filter=AMR "$c" -- 'tensorrt_llm/*.py' | while read -r f; do
        [[ $f =~ $focus ]] || continue
        mod=${f%.py}; mod=${mod%/__init__}; dotted=${mod//\//.}; base=${mod##*/}
        pat="\\b${dotted//./\\.}\\b"
        if [[ $f == */recovery/* ]]; then
            pat="$pat|\\brecovery(\\.| import .*\\b)${base}\\b"   # recovery.<base> or from ...recovery import <base>
        elif ! [[ $base =~ $generic ]]; then
            pat="$pat|\\b${base}\\b"
        fi
        git -C "$repo" grep -l -E "$pat" "$c" -- 'tests/unittest/*test_*.py' | sed "s/^$c://"
    done
    # Functions whose def lines change, and methods called on changed lines (a call gaining an argument breaks a
    # test that asserts the call or fakes the method).
    git -C "$repo" show --format= "$c" -- 'tensorrt_llm/*.py' | grep -E '^[-+][^-+]' |
        grep -oE '(def [A-Za-z_][A-Za-z0-9_]*|\.[A-Za-z_][A-Za-z0-9_]*\()' | sed -E 's/^def //; s/^\.//; s/\($//' | sort -u |
        grep -vE '^__' | awk 'length($0) >= 8' | while read -r name; do
            git -C "$repo" grep -l -w "$name" "$c" -- 'tests/unittest/*test_*.py' | sed "s/^$c://"
        done
} | sort -u | while read -r t; do git -C "$repo" cat-file -e "$c:$t" 2>/dev/null && echo "$t"; done
