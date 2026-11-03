"""Pytest plugin: deselect the tests listed in known_failures.txt beside this file (load with -p known_failures)."""
import pathlib

_IDS = [
    line.split("#")[0].strip()
    for line in pathlib.Path(__file__).with_name("known_failures.txt").read_text().splitlines()
]
_IDS = [test_id for test_id in _IDS if test_id]


def pytest_collection_modifyitems(config, items):
    known = [item for item in items if any(item.nodeid.endswith(test_id) for test_id in _IDS)]
    if known:
        config.hook.pytest_deselected(items=known)
        items[:] = [item for item in items if item not in known]
