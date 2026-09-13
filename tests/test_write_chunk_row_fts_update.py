"""Overwriting a chunk's content must drop its old FTS5 tokens.

``chunks_fts`` is an external-content FTS5 table: the only way to remove a
document's tokens is the ``'delete'`` command fed the values *as indexed*.
``write_chunk_row`` overwrote the ``chunks`` row first, so the fallback
delete was handed the new content and the old tokens stayed behind — a term
that no longer appeared in a chunk kept matching it. Surfaced by the
current-text projection's migration (a re-derived chunk still matched the
retired wording), but it applied to every edit of an indexed file.

Real SQLite under ``tmp_path``; no mocking.
"""
from __future__ import annotations

import pytest

from palinode.core import store
from palinode.core.config import config

_VEC = [0.1] * 1024


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    store.init_db()
    return tmp_path


def _write(chunk_id: str, text: str, embedding: list[float] | None = _VEC) -> None:
    with store.transaction() as db:
        store.write_chunk_row(
            db.cursor(),
            chunk_id=chunk_id, file_path="/m/a.md", section_id="root",
            category="projects", content=text, metadata_json="{}",
            content_hash=f"h:{text}", meta_hash="m", created_at="", last_updated="",
            embedding=embedding or [],
        )


def _docsize_rows() -> int:
    db = store.get_db()
    try:
        return db.execute("SELECT count(*) FROM chunks_fts_docsize").fetchone()[0]
    finally:
        db.close()


def test_updated_content_no_longer_matches_its_old_terms(tmp_store):
    _write("c1", "the sky is green")
    assert [r["content"] for r in store.search_fts("green")] == ["the sky is green"]

    _write("c1", "the sky is blue")
    assert store.search_fts("green") == []
    assert [r["content"] for r in store.search_fts("blue")] == ["the sky is blue"]
    assert _docsize_rows() == 1


def test_repeated_rewrites_keep_one_fts_document(tmp_store):
    for word in ("alpha", "beta", "gamma", "delta"):
        _write("c1", f"word {word}")
    assert _docsize_rows() == 1
    assert store.search_fts("alpha") == [] and store.search_fts("beta") == []
    assert [r["content"] for r in store.search_fts("delta")] == ["word delta"]


def test_fts_only_row_updates_cleanly_too(tmp_store):
    """The deferred (vector-less) path goes through the same FTS handling."""
    _write("c1", "the sky is green", embedding=None)
    _write("c1", "the sky is blue", embedding=None)
    assert store.search_fts("green") == []
    assert [r["content"] for r in store.search_fts("blue")] == ["the sky is blue"]


def test_fts_only_rewrite_drops_the_previous_vector(tmp_store):
    """A vector-less rewrite (cold embedder) of a row that *had* a vector must
    not keep it: it described the old text, and leaving it would make the row
    look embedded so the planner never re-embeds the new text."""
    _write("c1", "the sky is green")
    _write("c1", "the sky is blue", embedding=None)
    db = store.get_db()
    try:
        assert db.execute("SELECT 1 FROM chunks_vec WHERE id = 'c1'").fetchone() is None
    finally:
        db.close()
    assert [r["content"] for r in store.search_fts("blue")] == ["the sky is blue"]


def test_row_the_index_never_held_is_not_deleted_from_fts(tmp_store):
    """A chunks row with no FTS document (out-of-sync store) must not trigger
    a 'delete' for values FTS never indexed; it is simply indexed fresh."""
    with store.transaction() as db:
        db.execute(
            "INSERT INTO chunks (id, file_path, section_id, category, content) "
            "VALUES ('c1', '/m/a.md', 'root', 'projects', 'orphan text')"
        )
    assert store.search_fts("orphan") == []
    assert _docsize_rows() == 0

    _write("c1", "now indexed")
    assert [r["content"] for r in store.search_fts("indexed")] == ["now indexed"]
    assert _docsize_rows() == 1
