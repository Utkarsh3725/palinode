"""Gist-pointer discipline: a `core: true` memory is an index entry, not a dump.

Core memories are injected at every session start, so their size is spent
whether or not anyone wanted it. The lint check flags the ones that have grown
past ``context.core_gist_max_chars`` and carries the remediation with the
finding, so whichever surface renders it also says what to do.

Real files under ``tmp_path``; the lint pass reads markdown, not the DB.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from palinode.cli.lint import api_client, lint as lint_command
from palinode.core.config import config
from palinode.core.lint import CORE_GIST_REMEDIATION, run_lint_pass


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.context, "core_gist_max_chars", 200)
    (tmp_path / "insights").mkdir()
    return tmp_path


def _seed(memory_dir, name: str, body: str, core: bool = True) -> None:
    (memory_dir / "insights" / name).write_text(
        f"---\nid: insights-{name[:-3]}\ncategory: insights\ntype: Insight\n"
        f"core: {'true' if core else 'false'}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_a_core_memory_over_gist_size_is_flagged_with_the_remediation(memory_dir):
    _seed(memory_dir, "bloated.md", "x" * 400)
    [finding] = run_lint_pass()["oversized_core"]
    assert finding["file"] == "insights/bloated.md"
    assert finding["chars"] == 400
    assert finding["tokens"] == 100
    assert finding["limit"] == 200
    assert finding["remediation"] == CORE_GIST_REMEDIATION
    assert "gist + pointer" in CORE_GIST_REMEDIATION


def test_a_gist_sized_core_memory_is_not_flagged(memory_dir):
    _seed(memory_dir, "tight.md", "The rule, in one line. Detail: insights/full.md")
    assert run_lint_pass()["oversized_core"] == []


def test_a_large_non_core_memory_is_not_flagged(memory_dir):
    _seed(memory_dir, "reference.md", "x" * 4000, core=False)
    assert run_lint_pass()["oversized_core"] == []


def test_zero_disables_the_check(memory_dir, monkeypatch):
    monkeypatch.setattr(config.context, "core_gist_max_chars", 0)
    _seed(memory_dir, "bloated.md", "x" * 4000)
    assert run_lint_pass()["oversized_core"] == []


def test_the_cli_renders_the_finding_and_its_remediation(memory_dir, monkeypatch):
    _seed(memory_dir, "bloated.md", "x" * 400)
    monkeypatch.setattr(api_client, "lint", lambda **_kwargs: run_lint_pass())
    result = CliRunner().invoke(lint_command, ["--format", "text"])
    assert result.exit_code == 0
    # Rich wraps at the console width; compare on the unwrapped text.
    flat = " ".join(result.output.split())
    assert "Core memories over gist size (1)" in flat
    assert "insights/bloated.md: 400 chars" in flat
    assert CORE_GIST_REMEDIATION in flat


def test_the_cli_says_so_when_every_core_memory_is_gist_sized(memory_dir, monkeypatch):
    _seed(memory_dir, "tight.md", "One line, with a pointer.")
    monkeypatch.setattr(api_client, "lint", lambda **_kwargs: run_lint_pass())
    result = CliRunner().invoke(lint_command, ["--format", "text"])
    assert "All core memories are gist-sized" in result.output
