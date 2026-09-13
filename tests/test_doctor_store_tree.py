"""Tests for the ``store_tree_clean`` doctor check.

Real ``tmp_path`` git repositories, real git: the check answers "are there
uncommitted files in the store?", and a mocked ``git status`` would assert the
parsing, not the answer.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from palinode.core.config import Config
from palinode.diagnostics.checks.store_tree import (
    _DIRTY_WARN_THRESHOLD,
    _count_dirty,
    store_tree_clean,
)
from palinode.diagnostics.registry import all_checks
from palinode.diagnostics.types import DoctorContext


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False,
    )


@pytest.fixture(autouse=True)
def hermetic_git(monkeypatch):
    for var in (
        "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL", "EMAIL", "GIT_DIR", "GIT_WORK_TREE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _ctx(memory_dir: Path) -> DoctorContext:
    cfg = Config(memory_dir=str(memory_dir), db_path=str(memory_dir / ".palinode.db"))
    return DoctorContext(config=cfg)


def _init_repo(path: Path) -> None:
    assert _git(path, "init", "-q").returncode == 0
    _git(path, "config", "user.name", "Palinode Tests")
    _git(path, "config", "user.email", "tests@example.com")
    (path / "README.md").write_text("store\n")
    _git(path, "add", "README.md")
    assert _git(path, "commit", "-qm", "initial").returncode == 0


def test_registered_as_fast_check() -> None:
    names = {fn.__name__: tags for fn, tags in all_checks()}
    assert "store_tree_clean" in names
    assert "fast" in names["store_tree_clean"]


def test_not_a_repo_is_info(tmp_path: Path) -> None:
    result = store_tree_clean(_ctx(tmp_path))
    assert result.severity == "info"
    assert result.passed is True
    assert "not a git repository" in result.message


def test_clean_tree_passes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    result = store_tree_clean(_ctx(tmp_path))
    assert result.passed is True
    assert "clean" in result.message


def test_few_dirty_files_pass_with_count(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("edited\n")
    result = store_tree_clean(_ctx(tmp_path))
    assert result.passed is True
    assert "1 uncommitted file" in result.message
    assert "1 modified, 0 untracked" in result.message


def test_backlog_over_threshold_warns_with_one_step_remediation(tmp_path: Path) -> None:
    """The dogfood shape: hundreds of committed memories each given one
    uncommitted frontmatter line. Modified, not untracked."""
    _init_repo(tmp_path)
    projects = tmp_path / "projects"
    projects.mkdir()
    n = _DIRTY_WARN_THRESHOLD + 2
    for i in range(n):
        (projects / f"m{i}.md").write_text(f"---\nid: m{i}\n---\nbody\n")
    _git(tmp_path, "add", "projects")
    assert _git(tmp_path, "commit", "-qm", "memories").returncode == 0
    for i in range(n):
        p = projects / f"m{i}.md"
        p.write_text(p.read_text().replace("---\nbody", 'description: "x"\n---\nbody'))

    result = store_tree_clean(_ctx(tmp_path))
    assert result.severity == "warn"
    assert result.passed is False
    assert f"{n} uncommitted files" in result.message
    assert f"{n} modified, 0 untracked" in result.message
    assert result.remediation is not None
    assert "git -C" in result.remediation and "add -A --" in result.remediation


def test_untracked_files_count_separately(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "insights").mkdir()
    for i in range(_DIRTY_WARN_THRESHOLD + 1):
        (tmp_path / "insights" / f"u{i}.md").write_text("---\n---\nnew\n")
    result = store_tree_clean(_ctx(tmp_path))
    assert result.passed is False
    assert f"0 modified, {_DIRTY_WARN_THRESHOLD + 1} untracked" in result.message


def test_count_dirty_parses_porcelain() -> None:
    porcelain = " M a.md\n?? b.md\nA  c.md\n\n?? d/\n"
    assert _count_dirty(porcelain) == (2, 2)
