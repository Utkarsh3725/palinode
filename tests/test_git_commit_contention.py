"""``try_commit_memory_files`` under contention for ``.git/index.lock``.

Two writers can commit against one store at once: threads inside the API
(save path, backfill, a CLI-driven bootstrap through the REST layer) and the
watcher process alongside it. Git serialises them with ``index.lock`` and
fails the loser with exit 128; the helper used to treat that as terminal, so
the loser's file stayed dirty. On the dogfood store a whole-store
``bootstrap-ids`` racing the watcher stranded 82 files this way.

Real git, real ``tmp_path`` repos. The stale-lock case shrinks the backoff
so the test stays fast without mocking the retry away.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from palinode.core import git_tools
from palinode.core.config import config


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


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.git, "auto_commit", True)
    assert _git(tmp_path, "init", "-q").returncode == 0
    _git(tmp_path, "config", "user.name", "Contention Test")
    _git(tmp_path, "config", "user.email", "contention@example.invalid")
    (tmp_path / ".gitkeep").write_text("")
    _git(tmp_path, "add", ".gitkeep")
    assert _git(tmp_path, "commit", "-qm", "initial").returncode == 0
    return tmp_path


def _commit_count(repo: Path) -> int:
    return int(_git(repo, "rev-list", "--count", "HEAD").stdout.strip())


def test_concurrent_threads_all_land(repo: Path) -> None:
    """Sixteen threads, sixteen files, sixteen commits, clean tree."""
    (repo / "projects").mkdir()
    n = 16
    paths = []
    for i in range(n):
        p = repo / "projects" / f"m{i}.md"
        p.write_text(f"---\nid: m{i}\n---\nbody {i}\n")
        paths.append(str(p))
    before = _commit_count(repo)
    outcomes: list[git_tools.CommitOutcome] = [None] * n  # type: ignore[list-item]
    start = threading.Barrier(n)

    def worker(i: int) -> None:
        start.wait()
        outcomes[i] = git_tools.try_commit_memory_files([paths[i]], f"commit m{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(o.committed for o in outcomes), [o.error for o in outcomes if not o.committed]
    assert _commit_count(repo) == before + n
    assert _git(repo, "status", "--porcelain").stdout == ""


def test_cross_process_lock_is_waited_out(repo: Path, monkeypatch) -> None:
    """A lock held by *another* process clears mid-retry; the commit lands.

    Simulated by a thread that removes a hand-placed ``index.lock`` after a
    short hold — the same shape as the watcher finishing its commit.
    """
    monkeypatch.setattr(git_tools, "_INDEX_LOCK_BACKOFF", (0.02, 0.02, 0.02, 0.02, 0.02))
    p = repo / "note.md"
    p.write_text("---\nid: note\n---\nbody\n")
    lock = repo / ".git" / "index.lock"
    lock.write_text("")
    releaser = threading.Timer(0.05, lambda: lock.unlink(missing_ok=True))
    releaser.start()
    try:
        outcome = git_tools.try_commit_memory_files([str(p)], "after the lock clears")
    finally:
        releaser.cancel()
    assert outcome == git_tools.CommitOutcome(True), outcome
    assert _git(repo, "status", "--porcelain").stdout == ""


def test_stale_lock_is_still_reported_after_bounded_retries(repo: Path, monkeypatch, caplog) -> None:
    """A lock nobody releases (a crashed git) must not hang or be swallowed."""
    monkeypatch.setattr(git_tools, "_INDEX_LOCK_BACKOFF", (0.01, 0.01, 0.01, 0.01, 0.01))
    p = repo / "note.md"
    p.write_text("---\nid: note\n---\nbody\n")
    (repo / ".git" / "index.lock").write_text("")
    with caplog.at_level("DEBUG", logger="palinode.git_tools"):
        outcome = git_tools.try_commit_memory_files([str(p)], "never lands")
    assert outcome.committed is False
    assert outcome.error is not None and "index.lock" in outcome.error
    retries = [r for r in caplog.records if "hit index.lock" in r.getMessage()]
    assert len(retries) == git_tools._INDEX_LOCK_RETRIES


def test_non_lock_exit_128_is_not_retried(tmp_path: Path, monkeypatch, caplog) -> None:
    """Not a repository is exit 128 too; it must fail once, immediately."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.git, "auto_commit", True)
    p = tmp_path / "note.md"
    p.write_text("body\n")
    with caplog.at_level("DEBUG", logger="palinode.git_tools"):
        outcome = git_tools.try_commit_memory_files([str(p)], "no repo")
    assert outcome.committed is False
    assert outcome.error is not None and "not a git repository" in outcome.error
    assert not [r for r in caplog.records if "hit index.lock" in r.getMessage()]


def test_is_index_lock_collision_signature() -> None:
    lock = subprocess.CompletedProcess([], 128, "", "fatal: Unable to create '.git/index.lock': File exists.")
    norepo = subprocess.CompletedProcess([], 128, "", "fatal: not a git repository")
    ident = subprocess.CompletedProcess([], 128, "", "Author identity unknown")
    assert git_tools._is_index_lock_collision(lock)
    assert not git_tools._is_index_lock_collision(norepo)
    assert not git_tools._is_index_lock_collision(ident)
