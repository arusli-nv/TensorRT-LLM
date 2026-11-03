"""Drop a directory's files from this node's page cache (posix_fadvise DONTNEED); print time and bytes.

Usage: python3 drop_page_cache.py DIR
"""

import os
import sys
import time


def main(root: str) -> None:
    start, files, size = time.time(), 0, 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            with open(path, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
            size += os.path.getsize(path)
    print(f"DROP_PAGE_CACHE start={start:.3f} s={time.time() - start:.2f} files={files} GiB={size / 2**30:.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
