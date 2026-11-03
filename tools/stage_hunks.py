"""Stage only the hunks of one file's working-tree diff whose added or removed lines match a regex.

Non-interactive `git add -p`: prints the selected hunks and applies them to the index.
Usage: python3 stage_hunks.py REPO FILE REGEX
"""

import re
import subprocess
import sys

repo, path, pattern = sys.argv[1:4]
diff = subprocess.run(["git", "-C", repo, "diff", "-U3", "--", path], check=True, capture_output=True,
                      text=True).stdout
header, *hunks = re.split(r"(?m)^(?=@@ )", diff)
chosen = [h for h in hunks if any(re.search(pattern, line) for line in h.splitlines()[1:]
                                  if line[:1] in "+-")]
if not chosen:
    sys.exit(f"no hunk of {path} matches {pattern!r}")
patch = header + "".join(chosen)
print(patch)
subprocess.run(["git", "-C", repo, "apply", "--cached", "-"], input=patch, text=True, check=True)
