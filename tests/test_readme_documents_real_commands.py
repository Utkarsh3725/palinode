"""The README must not name a CLI command that does not exist.

Documentation and implementation for the ``dream`` alias arrive from two
different places — the alias from a contributor PR on the public repo, the
positioning paragraph from here — so the ordering between them is a thing a
human has to remember, right up until it is a thing CI enforces.

This is the same shape as the ``server.json`` <-> ``mcp.py`` metadata pin: two
artifacts that must agree, bound by a test rather than by discipline. A README
that documents a command the CLI does not have is the same defect class the
repo keeps finding elsewhere — a surface confidently describing behaviour that
is not there.

Scoped deliberately narrowly. This checks commands the README presents as
runnable ``palinode <verb>`` invocations in prose, not every string in every
table, because a table of planned or third-party commands is a different thing
from a sentence telling a reader what to type.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from palinode.cli import main as cli_root


README = Path(__file__).resolve().parent.parent / "README.md"

#: Commands the README describes in prose as something the reader can run.
#: Add a verb here when the README starts telling people to type it.
DOCUMENTED_IN_PROSE = ("dream",)

#: A skip here would let a README edit that drops the verb turn the test green
#: silently. Failing forces the change to be deliberate: either restore the
#: sentence, or remove the verb from DOCUMENTED_IN_PROSE in the same change.
_NOT_DOCUMENTED = (
    "README no longer documents `palinode {verb}` in prose. If the command "
    "was removed on purpose, remove it from DOCUMENTED_IN_PROSE in this test "
    "deliberately; otherwise restore the README sentence."
)


def _registered_commands() -> set[str]:
    """Every command name the CLI actually exposes, aliases included."""
    names: set[str] = set()
    for attr in ("commands", "_commands"):
        registry = getattr(cli_root, attr, None)
        if isinstance(registry, dict):
            names.update(registry.keys())
    return names


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("verb", DOCUMENTED_IN_PROSE)
def test_readme_command_is_registered(verb: str) -> None:
    """If the README says ``palinode <verb>``, the CLI must answer to it."""
    mentioned = re.search(rf"`palinode {re.escape(verb)}`", _readme_text())
    assert mentioned, _NOT_DOCUMENTED.format(verb=verb)

    assert verb in _registered_commands(), (
        f"README documents `palinode {verb}` but the CLI has no such command. "
        f"Either the docs landed before the implementation, or the command was "
        f"removed and the docs were not. Registered: "
        f"{sorted(_registered_commands())}"
    )


@pytest.mark.parametrize("verb", DOCUMENTED_IN_PROSE)
def test_documented_command_actually_runs(verb: str) -> None:
    """Registered is not the same as working — ``--help`` must succeed.

    Cheap end-to-end check that the alias is wired to a real callback rather
    than registered as a name with nothing behind it.
    """
    assert re.search(rf"`palinode {re.escape(verb)}`", _readme_text()), (
        _NOT_DOCUMENTED.format(verb=verb)
    )

    result = CliRunner().invoke(cli_root, [verb, "--help"])
    assert result.exit_code == 0, (
        f"`palinode {verb} --help` exited {result.exit_code}:\n{result.output}"
    )
