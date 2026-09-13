"""No test may evict a ``palinode.*`` module from ``sys.modules``.

Deleting a package module and re-importing it produces a second module
object. Every module that already did ``from palinode.x import thing`` keeps
the first one; every later import gets the second. Singletons split, monkey-
patches land on the wrong copy, and the failure shows up in an unrelated test
file as something like ``ECONNREFUSED``: five session-end integration tests
failed whenever ``test_session_end_timeout.py`` ran first, until its import
checks moved to a subprocess.

A fresh import that must not leak belongs in a subprocess. ``importlib.reload``
on an already-imported module keeps identity and is not gated here; the
established ``importlib.reload(palinode.api.server)`` fixture pattern is
self-contained.
"""
from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).parent

#: ``del sys.modules[...]``, ``sys.modules.pop(...)``, or an assignment into
#: ``sys.modules`` — any of them on a palinode module. The name check is on
#: the same line or inside a loop whose filter names palinode, so the pattern
#: matches the literal and the loop form the original offender used.
_EVICTION = re.compile(r"(del\s+sys\.modules\[|sys\.modules\.pop\(|sys\.modules\[[^\]]+\]\s*=)")


def _offenders() -> list[str]:
    hits: list[str] = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not _EVICTION.search(line):
                continue
            window = "\n".join(lines[max(0, i - 4): i + 1])
            if "palinode" in window:
                hits.append(f"{path.relative_to(TESTS_DIR.parent)}:{i + 1}: {line.strip()}")
    return hits


def test_no_test_evicts_palinode_modules_from_sys_modules() -> None:
    offenders = _offenders()
    assert not offenders, (
        "Tests that evict palinode modules from sys.modules split module identity "
        "for the rest of the session. Run the fresh import in a "
        "subprocess instead:\n  " + "\n  ".join(offenders)
    )


def test_guard_matches_the_original_offender_shape(tmp_path: Path) -> None:
    """The loop form of the original offender, where the filter naming palinode sits a few
    lines above the ``del``, must be caught."""
    sample = (
        "for mod in list(sys.modules):\n"
        "    if 'palinode.cli._api' in mod:\n"
        "        del sys.modules[mod]\n"
    )
    lines = sample.splitlines()
    caught = False
    for i, line in enumerate(lines):
        if _EVICTION.search(line) and "palinode" in "\n".join(lines[max(0, i - 4): i + 1]):
            caught = True
    assert caught
