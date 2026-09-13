"""The ``projection_current`` doctor check reports migration progress.

Real SQLite under ``tmp_path`` with the minimal ``chunks`` shape the check
reads; no mocking.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from palinode.core.config import Config
from palinode.core.projection import PROJECTION_VERSION
from palinode.diagnostics.runner import run_one
from palinode.diagnostics.types import DoctorContext


def _ctx(db_path: Path) -> DoctorContext:
    return DoctorContext(config=Config(memory_dir=str(db_path.parent), db_path=str(db_path)))


def _schema(db_path: Path, *, with_column: bool = True) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    cols = "id TEXT PRIMARY KEY, file_path TEXT, content TEXT"
    if with_column:
        cols += ", projected_hash TEXT, projection_version INTEGER"
    con.execute(f"CREATE TABLE chunks ({cols})")
    con.commit()
    return con


def _insert(con: sqlite3.Connection, n: int, version: int | None) -> None:
    for i in range(n):
        con.execute(
            "INSERT INTO chunks (id, file_path, content, projection_version) VALUES (?, ?, ?, ?)",
            (f"{version}-{i}", "/m/x.md", "text", version),
        )
    con.commit()


def test_missing_db_passes(tmp_path):
    result = run_one(_ctx(tmp_path / "missing.db"), "projection_current")
    assert result.passed and "does not exist" in result.message


def test_empty_store_passes(tmp_path):
    db = tmp_path / ".palinode.db"
    _schema(db).close()
    result = run_one(_ctx(db), "projection_current")
    assert result.passed and "empty" in result.message


def test_all_chunks_current_passes_with_count(tmp_path):
    db = tmp_path / ".palinode.db"
    con = _schema(db)
    _insert(con, 3, PROJECTION_VERSION)
    con.close()
    result = run_one(_ctx(db), "projection_current")
    assert result.passed
    assert "All 3 indexed chunks" in result.message
    assert f"v{PROJECTION_VERSION}" in result.message


def test_chunks_behind_warn_with_progress_counts(tmp_path):
    db = tmp_path / ".palinode.db"
    con = _schema(db)
    _insert(con, 2, PROJECTION_VERSION)
    _insert(con, 3, None)
    _insert(con, 1, PROJECTION_VERSION - 1)
    con.close()
    result = run_one(_ctx(db), "projection_current")
    assert not result.passed and result.severity == "warn"
    assert "4 of 6 indexed chunks" in result.message
    assert "palinode reindex" in result.remediation
    assert "behind         : 4" in result.remediation


def test_schema_without_the_column_reports_every_chunk_behind(tmp_path):
    db = tmp_path / ".palinode.db"
    con = _schema(db, with_column=False)
    con.execute("INSERT INTO chunks (id, file_path, content) VALUES ('a', '/m/x.md', 't')")
    con.commit()
    con.close()
    result = run_one(_ctx(db), "projection_current")
    assert not result.passed and "1 of 1 indexed chunks" in result.message


def test_check_is_registered_as_fast():
    from palinode.diagnostics import checks  # noqa: F401  (registration side effect)
    from palinode.diagnostics.registry import all_checks

    registered = {fn.__name__: tags for fn, tags in all_checks()}
    assert "projection_current" in registered
    assert "fast" in registered["projection_current"]
