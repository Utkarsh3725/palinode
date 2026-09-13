"""The index is derived from the projected current text — end to end.

Real files under ``tmp_path``, the real executor (``apply_operations``), real
SQLite through the reconcile seam, real FTS5. The embedder is the only double,
and it *records* what it was asked to embed: the proof that the vector arm
sees the projected text is the list of inputs, since a fake vector cannot
tell old wording from new.
"""
from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from palinode.consolidation.executor import apply_operations
from palinode.core import parser, store
from palinode.core.config import config
from palinode.core.hashing import stable_md5_hexdigest
from palinode.core.projection import PROJECTION_VERSION
from palinode.indexer import reconcile

_VEC = [0.03] * 1024

SEED = (
    "---\n"
    "id: demo\n"
    "category: projects\n"
    "status: active\n"
    "---\n\n"
    "# Demo\n\n"
    "- Use endpoint A. <!-- fact:endpoint -->\n"
    "- The sky is green. <!-- fact:sky -->\n"
    "- Deploys go through CI. <!-- fact:ci -->\n"
)


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


def _seed(tmp_store) -> str:
    path = tmp_store / "projects" / "demo.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(SEED, encoding="utf-8")
    return str(path)


def _retire(path: str) -> str:
    """Real executor: SUPERSEDE one fact, RETRACT another. Returns the new content."""
    stats = apply_operations(path, [
        {"op": "SUPERSEDE", "id": "endpoint", "new_text": "Use endpoint B.",
         "rationale": "Explicit replacement"},
        {"op": "RETRACT", "id": "sky", "rationale": "never true",
         "falsified_by": "projects/demo"},
    ])
    assert stats["superseded"] == 1 and stats["retracted"] == 1, stats
    with open(path, encoding="utf-8") as f:
        return f.read()


def _reconcile(path: str, content: str, seen: list[str] | None = None):
    def _embed(text: str, backend: str = "local") -> list[float]:
        if seen is not None:
            seen.append(text)
        return _VEC

    with patch("palinode.core.embedder.embed", side_effect=_embed):
        return reconcile.reconcile(path, content)


def _rows(path: str) -> list[tuple]:
    db = store.get_db()
    try:
        rows = db.execute(
            "SELECT id, section_id, content, content_hash, projected_hash, "
            "projection_version FROM chunks WHERE file_path = ? ORDER BY id",
            (path,),
        ).fetchall()
        return [tuple(r) for r in rows]
    finally:
        db.close()


def _write_legacy_rows(path: str, content: str) -> None:
    """A pre-projection store: raw section text as the chunk, no stamp."""
    metadata, sections = parser.parse_markdown(content)
    with store.transaction() as db:
        cur = db.cursor()
        for sec in sections:
            store.write_chunk_row(
                cur,
                chunk_id=stable_md5_hexdigest(f"{path}#{sec['section_id']}"),
                file_path=path,
                section_id=sec["section_id"],
                category="projects",
                content=sec["content"],
                metadata_json="{}",
                content_hash=hashlib.sha256(sec["content"].encode()).hexdigest(),
                meta_hash=store.meta_hash(metadata),
                created_at="", last_updated="",
                embedding=_VEC,
            )


# ── acceptance: the old assertion does not rank or render; history stays ─────


def test_retired_facts_do_not_reach_fts_or_the_embedder(tmp_store):
    path = _seed(tmp_store)
    content = _retire(path)
    history = tmp_store / "projects" / "demo-history.md"
    history_before = history.read_bytes()
    assert b"Superseded (" in history_before and b"Retracted (" in history_before

    seen: list[str] = []
    diff = _reconcile(path, content, seen)
    assert diff.committed and diff.written == 1 and diff.embed_failures == 0

    # The raw file still carries both tombstones and the sidecar is untouched.
    assert "~~Use endpoint A.~~ [superseded" in content
    assert "~~The sky is green.~~ [RETRACTED" in content
    assert history.read_bytes() == history_before

    # Vector arm: the embedder was fed the projected text only.
    assert len(seen) == 1
    assert "Use endpoint A." not in seen[0] and "sky is green" not in seen[0]
    assert "Use endpoint B." in seen[0] and "Deploys go through CI." in seen[0]

    # Keyword arm: the retracted wording finds nothing; the superseded wording
    # finds only its successor.
    assert store.search_fts("green") == []
    hits = store.search_fts("endpoint")
    assert len(hits) == 1
    assert "Use endpoint B." in hits[0]["content"]
    assert "Use endpoint A." not in hits[0]["content"]

    # Hybrid search + freshness: the snippet is the successor, the raw-hash
    # agreement is valid, and the chunk is current (nothing retired in it).
    merged = store.search_hybrid("endpoint", _VEC, top_k=5, threshold=0.0)
    assert len(merged) == 1
    hit = merged[0]
    assert "Use endpoint A." not in hit["content"] and "Use endpoint B." in hit["content"]
    assert hit["freshness"] == "valid"
    assert hit["currency"] == "current"


def test_stored_hashes_are_in_separate_domains(tmp_store):
    path = _seed(tmp_store)
    content = _retire(path)
    _reconcile(path, content)
    (_, _, stored_content, content_hash, projected_hash, version), = _rows(path)

    _, sections = parser.parse_markdown(content)
    raw = sections[0]["content"]
    assert content_hash == hashlib.sha256(raw.encode()).hexdigest()
    assert projected_hash == hashlib.sha256(stored_content.encode()).hexdigest()
    assert projected_hash != content_hash
    assert version == PROJECTION_VERSION
    assert "~~" in raw and "~~" not in stored_content


def test_fresh_rebuild_and_incremental_reconcile_converge(tmp_store, monkeypatch):
    path = _seed(tmp_store)
    content = _retire(path)

    # Incremental: a store indexed before the projection existed.
    _write_legacy_rows(path, content)
    legacy = _rows(path)
    assert legacy[0][5] is None and "Use endpoint A." in legacy[0][2]
    assert store.search_fts("green") != []  # the legacy tokens are live
    diff = _reconcile(path, content)
    assert diff.reprojected == 1 and diff.written == 0 and diff.stamped == 0
    incremental = _rows(path)
    assert reconcile.plan(reconcile.derive(path, content)).is_noop
    assert store.search_fts("green") == []  # and gone once re-derived

    # Fresh: an empty database, same file.
    monkeypatch.setattr(config, "db_path", str(tmp_store / "fresh.db"))
    store.init_db()
    diff = _reconcile(path, content)
    assert diff.written == 1
    fresh = _rows(path)

    assert incremental == fresh
    assert incremental[0][5] == PROJECTION_VERSION


def test_chunk_on_an_older_projection_version_is_rederived(tmp_store):
    path = _seed(tmp_store)
    content = _retire(path)
    _reconcile(path, content)
    (chunk_id, *_), = _rows(path)

    # Stamp an older version: derived text and vector still right → stamp only.
    with store.transaction() as db:
        db.execute("UPDATE chunks SET projection_version = 0 WHERE id = ?", (chunk_id,))
    p = reconcile.plan(reconcile.derive(path, content))
    assert [s.chunk_id for s in p.stamp_only] == [chunk_id] and not p.to_index
    diff = _reconcile(path, content)
    assert diff.stamped == 1 and diff.reprojected == 0
    assert _rows(path)[0][5] == PROJECTION_VERSION
    assert reconcile.plan(reconcile.derive(path, content)).is_noop

    # Older version *and* stale derived text → the text is rewritten.
    _, sections = parser.parse_markdown(content)
    with store.transaction() as db:
        db.execute(
            "UPDATE chunks SET projection_version = 0, content = ? WHERE id = ?",
            (sections[0]["content"], chunk_id),
        )
    p = reconcile.plan(reconcile.derive(path, content))
    assert [(pw.section.chunk_id, pw.reason) for pw in p.to_index] == [
        (chunk_id, reconcile.REPROJECT)
    ]
    diff = _reconcile(path, content)
    assert diff.reprojected == 1
    row = _rows(path)[0]
    assert row[5] == PROJECTION_VERSION and "Use endpoint A." not in row[2]


def test_unstamped_row_with_nothing_to_project_is_stamped_without_reembedding(tmp_store):
    path = _seed(tmp_store)  # no tombstones
    _write_legacy_rows(path, SEED)
    assert _rows(path)[0][5] is None

    def _no_embed(text: str, backend: str = "local") -> list[float]:
        raise AssertionError("a stamp-only reconcile must not embed")

    with patch("palinode.core.embedder.embed", side_effect=_no_embed):
        diff = reconcile.reconcile(path, SEED)
    assert diff.committed and diff.stamped == 1
    assert diff.reprojected == 0 and diff.written == 0 and diff.reembedded == 0
    (_, _, stored, content_hash, projected_hash, version), = _rows(path)
    assert version == PROJECTION_VERSION and projected_hash == content_hash
    assert reconcile.plan(reconcile.derive(path, SEED)).is_noop


def test_reprojection_under_a_cold_embedder_stays_observable(tmp_store, monkeypatch):
    path = _seed(tmp_store)
    content = _retire(path)
    _write_legacy_rows(path, content)

    monkeypatch.setattr(reconcile, "_embeds_deferred", lambda client: True)
    diff = reconcile.reconcile(path, content)
    assert diff.committed and diff.deferred and diff.reprojected == 1
    assert diff.error is not None and diff.error.startswith("embed deferred")
    assert not diff.vec_ok

    # Keyword-searchable on the projected text now; the vector is pending and
    # is exactly what the next warm pass re-embeds.
    assert store.search_fts("green") == []
    assert _rows(path)[0][5] == PROJECTION_VERSION
    p = reconcile.plan(reconcile.derive(path, content))
    assert [pw.reason for pw in p.to_index] == [reconcile.REEMBED]


def test_index_file_reports_the_migration_counters(tmp_store):
    from palinode.indexer.index_file import index_file

    path = _seed(tmp_store)
    content = _retire(path)
    _write_legacy_rows(path, content)
    with patch("palinode.core.embedder.embed", return_value=_VEC):
        result = index_file(path)
    assert result["chunks_reprojected"] == 1 and result["chunks_stamped"] == 0
    assert result["embedded"] and result["error"] is None
