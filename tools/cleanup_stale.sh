#!/bin/bash
# Remove stale git worktrees and dead caches of the WideEP FT work (run by the user from their terminal: the Claude
# sandbox cannot write .git/worktrees). Prints the plan; acts only with --yes.
# Removing a worktree keeps its branch; only uncommitted changes in it would be lost, so a worktree with changes is
# skipped (listed) unless --force-dirty is given.
# Usage: bash cleanup_stale.sh [--yes] [--force-dirty]
set -uo pipefail
repo=/tmp/claude-161238/work/repo
scratch=/tmp/claude-161238
yes= force_dirty=
for arg in "$@"; do
    case $arg in --yes) yes=1 ;; --force-dirty) force_dirty=1 ;; *) echo "unknown argument $arg"; exit 2 ;; esac
done

# Worktrees still in use: the main clone, v3 (the v5 base, branch of record), v2 and its PR split, the rsg-v2 PR stack.
keep=(
    "$repo"
    "$scratch/wt-v3"
    "$scratch/wt-v2"
    "$scratch/wt-v2cpp"
    "$scratch/wt-v2py"
    "$scratch/work/wt-rsg-E"
    "$scratch/work/wt-rsg-K"
    "$scratch/work/wt-rsg-R"
    "$scratch/work/wt-rsg-X"
)
# Dead caches and scratch: profile caches, old nsys reports, mktemp scratch, pip caches, doc backups.
dead=(
    "$scratch/nsys_ab_cache" "$scratch/ftprof32" "$scratch/ftprof" "$scratch/nsys" "$scratch/nsys-arusli"
    "$scratch"/tmp.* "$scratch"/pip-ephem-wheel-cache-* "$scratch"/prready.* "$scratch/rs"
    "$scratch"/*.nsys-rep "$scratch"/DESIGN*.before* "$scratch"/*.diff "$scratch/job.pre-hostcache2"
)

is_kept() { local k; for k in "${keep[@]}"; do [[ $1 == "$k" ]] && return 0; done; return 1; }

remove_wt=() dirty=()
while read -r path; do
    is_kept "$path" && continue
    if [[ -d $path && -n $(git -C "$path" status --porcelain 2>/dev/null | grep -v '^??') ]]; then
        dirty+=("$path")
        [[ -n $force_dirty ]] && remove_wt+=("$path")
    else
        remove_wt+=("$path")
    fi
done < <(git -C "$repo" worktree list --porcelain | awk '/^worktree /{print $2}')

echo "KEEP (${#keep[@]} worktrees):"; printf '  %s\n' "${keep[@]}"
echo "REMOVE (${#remove_wt[@]} worktrees; branches stay):"; printf '  %s\n' "${remove_wt[@]}"
if (( ${#dirty[@]} )); then
    echo "WITH UNCOMMITTED CHANGES (${#dirty[@]}; $([[ -n $force_dirty ]] && echo removed || echo 'skipped, --force-dirty removes')):"
    printf '  %s\n' "${dirty[@]}"
fi
echo "DELETE (caches and scratch):"
for d in "${dead[@]}"; do [[ -e $d ]] && echo "  $(du -sh "$d" 2>/dev/null | cut -f1)  $d"; done
[[ -n $yes ]] || { echo "Plan only. Run with --yes to act."; exit 0; }

for path in "${remove_wt[@]}"; do git -C "$repo" worktree remove --force "$path" && echo "removed $path"; done
git -C "$repo" worktree prune
for d in "${dead[@]}"; do [[ -e $d ]] && rm -rf -- "$d" && echo "deleted $d"; done
echo "Done. Worktrees left: $(git -C "$repo" worktree list | wc -l)"
