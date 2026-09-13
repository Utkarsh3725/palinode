"""``POST /generate-summaries`` commits what it writes.

The deferred description (and the core summary) land through a bare
read-modify-write of frontmatter; before this the original save was committed
and the later injection was not, so a store with a slow CHAT host accumulated
one dirty file per save — 572 on the dogfood store over two weeks — while the
save path kept reporting ``git_committed: true``.

Real git, real ``tmp_path`` store, nothing about the commit mocked: what is
under test is whether a commit exists afterwards.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import palinode.api.routers.memory as _memory_router
import palinode.api.server as _server_mod
from palinode.api.server import app
from palinode.core.config import config


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    )
    return res.stdout


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


@pytest.fixture()
def git_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A store that is a real git repo with auto_commit on — a provisioned store."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", True)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "backfill-test@example.invalid")
    _git(tmp_path, "config", "user.name", "Backfill Test")
    # Not a .md file: the API's startup guard refuses a store that has memory
    # files but no database yet.
    # A provisioned store ignores its own database; the test app creates one.
    (tmp_path / ".gitignore").write_text(".palinode.db*\n")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


@pytest.fixture()
def client(git_store: Path):
    from palinode.api import server as srv
    srv._rate_counters.clear()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    srv._rate_counters.clear()


def _commit_count(store: Path) -> int:
    return int(_git(store, "rev-list", "--count", "HEAD").strip())


def _seed(store: Path, n: int, core: bool = False) -> list[Path]:
    """``n`` committed memories with no description — the deferred-save shape."""
    sub = store / "insights"
    sub.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = sub / f"m{i}.md"
        core_line = "core: true\n" if core else ""
        p.write_text(f"---\nid: m{i}\ntype: Insight\n{core_line}---\nBody of memory {i}.\n")
        paths.append(p)
    _git(store, "add", "insights")
    _git(store, "commit", "-qm", "seed")
    return paths


def test_backfill_commits_the_descriptions_it_writes(client, git_store: Path) -> None:
    _seed(git_store, 3)
    before = _commit_count(git_store)

    with patch("palinode.api.server._generate_description", return_value="Filled in."):
        res = client.post("/generate-summaries")

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["descriptions_generated"] == 3
    assert body["commits"] == 1
    assert body["files_committed"] == 3
    assert _commit_count(git_store) == before + 1
    assert _git(git_store, "status", "--porcelain") == ""
    subject = _git(git_store, "log", "-1", "--pretty=%s").strip()
    assert subject.startswith(f"{config.git.commit_prefix} backfill: 3 descriptions (palinode ")
    # Exactly the touched files, nothing else swept in.
    names = _git(git_store, "show", "--name-only", "--pretty=", "HEAD").split()
    assert sorted(names) == [f"insights/m{i}.md" for i in range(3)]


def test_backfill_commits_in_batches(client, git_store: Path, monkeypatch) -> None:
    """A long walk against a slow host must not strand everything until the end."""
    monkeypatch.setattr(_memory_router, "_BACKFILL_COMMIT_BATCH", 4)
    _seed(git_store, 10)
    before = _commit_count(git_store)

    with patch("palinode.api.server._generate_description", return_value="Filled in."):
        res = client.post("/generate-summaries")

    body = res.json()
    assert body["descriptions_generated"] == 10
    assert body["commits"] == 3  # 4 + 4 + 2
    assert body["files_committed"] == 10
    assert _commit_count(git_store) == before + 3
    assert _git(git_store, "status", "--porcelain") == ""


def test_summary_and_description_on_one_file_is_one_commit(client, git_store: Path) -> None:
    _seed(git_store, 1, core=True)
    before = _commit_count(git_store)

    with patch("palinode.api.server._generate_description", return_value="Desc."), \
            patch("palinode.api.server._generate_summary", return_value="Sum."):
        res = client.post("/generate-summaries")

    body = res.json()
    assert body["descriptions_generated"] == 1
    assert body["summaries_generated"] == 1
    assert body["commits"] == 1
    assert body["files_committed"] == 1
    assert _commit_count(git_store) == before + 1
    subject = _git(git_store, "log", "-1", "--pretty=%s").strip()
    assert "1 description, 1 summary" in subject
    text = (git_store / "insights" / "m0.md").read_text()
    assert 'description: "Desc."' in text and 'summary: "Sum."' in text


def test_nothing_written_means_no_commit(client, git_store: Path) -> None:
    _seed(git_store, 2)
    before = _commit_count(git_store)
    with patch("palinode.api.server._generate_description",
               return_value=_server_mod._DESCRIPTION_DEFERRED):
        res = client.post("/generate-summaries")
    body = res.json()
    assert body["descriptions_generated"] == 0
    assert body["commits"] == 0
    assert body["files_committed"] == 0
    assert _commit_count(git_store) == before


def test_auto_commit_off_writes_but_does_not_commit(client, git_store: Path, monkeypatch) -> None:
    """The operator who commits on their own schedule gets the file, not a commit."""
    monkeypatch.setattr(config.git, "auto_commit", False)
    paths = _seed(git_store, 2)
    before = _commit_count(git_store)
    with patch("palinode.api.server._generate_description", return_value="Filled in."):
        res = client.post("/generate-summaries")
    body = res.json()
    assert body["descriptions_generated"] == 2
    assert body["commits"] == 0
    assert body["files_committed"] == 0
    assert _commit_count(git_store) == before
    assert all('description: "Filled in."' in p.read_text() for p in paths)
    assert len(_git(git_store, "status", "--porcelain").splitlines()) == 2


def test_backfill_commit_message_shapes() -> None:
    msg = _memory_router._backfill_commit_message({"description": 1, "summary": 0})
    assert msg.startswith(f"{config.git.commit_prefix} backfill: 1 description (palinode ")
    msg = _memory_router._backfill_commit_message({"description": 12, "summary": 2})
    assert "12 descriptions, 2 summaries" in msg
