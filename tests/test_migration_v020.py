"""v0.20 migration acceptance: an upgraded store and a clean one agree, and a
rollback revives nothing.

Two release claims are asserted here, and they are the two an operator has to
take on trust otherwise:

1. **A store indexed by pre-projection code, then reindexed, is
   byte-for-byte the store a fresh index of the same files produces.** The
   migration is a rebuild of derived state, so "upgrade in place" and "delete
   the database and start over" must not be two different stores. Simulated
   the only honest way available: real files, the real executor writing the
   real tombstones, then chunk rows written the way the pre-projection indexer
   wrote them (raw section text, no projection stamp), then the same per-file
   entry point ``palinode reindex`` drives.

2. **Downgrading does not revive expired authority or retired conclusions.**
   Retirement lives in markdown and git, not in the index: the two nullable
   columns the upgrade adds are ignored by older code, and a full rebuild
   *from the files alone* re-derives the same retired verdicts. This is the
   test ``docs/UPGRADING-v0.20.md`` points at for that claim.

Real SQLite under ``tmp_path`` throughout — no database double. The embedder
is the only stand-in, because a fake vector cannot tell old wording from new
and the keyword arm can.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from palinode.consolidation.executor import apply_operations
from palinode.core import parser, store
from palinode.core.config import config
from palinode.core.hashing import stable_md5_hexdigest
from palinode.core.lifecycle import eligibility
from palinode.core.projection import PROJECTION_VERSION
from palinode.indexer import index_file, reconcile

_VEC = [0.05] * 1024

#: A project note whose facts the executor retires in place — the case the
#: projection exists for.
DEMO = """---
id: demo
category: projects
status: active
---

# Demo

- Deploy target is Nomad. <!-- fact:target -->
- The staging cluster is in eu-west-1. <!-- fact:staging -->
- Backups run nightly. <!-- fact:backups -->
"""

#: Archived in place: still under decisions/, retired by its own frontmatter.
ARCHIVED_IN_PLACE = """---
id: db-choice
category: decisions
status: archived
superseded_by: decisions/db-choice-v2
---

# Database choice

- We will use MySQL for the ledger.
"""

#: Retired by location: the weekly pass moved it without rewriting a byte, so
#: its frontmatter still claims it is live.
ARCHIVED_BY_PATH = """---
id: old-note
category: daily
status: active
---

# March standup

- The ingest pipeline is owned by the platform team.
"""

#: Expired authority: an acting memory whose grant lapsed.
EXPIRED = """---
id: oncall-grant
category: projects
status: active
expires_at: 2020-01-01T00:00:00Z
---

# Oncall grant

- Riley may approve production deploys.
"""

#: Nothing retired anywhere in it — the control that keeps the others honest.
PLAIN = """---
id: plain
category: insights
status: active
---

# Plain

- Hybrid search fuses BM25 and vector ranks.
"""

SEEDS: dict[str, str] = {
    "projects/demo.md": DEMO,
    "decisions/db-choice.md": ARCHIVED_IN_PLACE,
    "archive/2026-03-01.md": ARCHIVED_BY_PATH,
    "projects/oncall-grant.md": EXPIRED,
    "insights/plain.md": PLAIN,
}

_ROW_COLUMNS = (
    "file_path, section_id, content, content_hash, projected_hash, projection_version"
)


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    return tmp_path


def _seed_files(root: Path) -> list[str]:
    """Write the corpus and let the real executor retire two of demo's facts."""
    paths: list[str] = []
    for rel, text in SEEDS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))

    demo = str(root / "projects" / "demo.md")
    stats = apply_operations(demo, [
        {"op": "SUPERSEDE", "id": "target", "new_text": "Deploy target is Kubernetes.",
         "rationale": "Explicit replacement recorded in the migration notes"},
        {"op": "RETRACT", "id": "staging", "rationale": "the cluster was never in eu-west-1",
         "falsified_by": "projects/demo"},
    ])
    assert stats["superseded"] == 1 and stats["retracted"] == 1, stats

    raw = (root / "projects" / "demo.md").read_text(encoding="utf-8")
    assert "~~Deploy target is Nomad.~~ [superseded" in raw
    assert "~~The staging cluster is in eu-west-1.~~ [RETRACTED" in raw
    # -history.md is a new file the executor wrote; it is part of the store.
    return sorted(str(p) for p in root.rglob("*.md"))


def _embed(text: str, backend: str = "local") -> list[float]:
    return _VEC


def _index_all(paths: list[str]) -> list[dict]:
    """The per-file unit ``palinode reindex`` drives, over the whole store."""
    with patch("palinode.core.embedder.embed", side_effect=_embed):
        return [index_file.index_file(p) for p in paths]


def _write_legacy_rows(paths: list[str]) -> None:
    """Chunk rows as the pre-projection indexer wrote them.

    Raw section text as the chunk body, no ``projected_hash`` and no
    ``projection_version`` — the two columns did not exist, so on an upgraded
    schema they are NULL.
    """
    for path in paths:
        content = Path(path).read_text(encoding="utf-8")
        metadata, sections = parser.parse_markdown(content)
        category = str(metadata.get("category") or "general")
        with store.transaction() as db:
            cur = db.cursor()
            for sec in sections:
                store.write_chunk_row(
                    cur,
                    chunk_id=stable_md5_hexdigest(f"{path}#{sec['section_id']}"),
                    file_path=path,
                    section_id=sec["section_id"],
                    category=category,
                    content=sec["content"],
                    metadata_json=json.dumps(metadata, default=str),
                    content_hash=hashlib.sha256(sec["content"].encode()).hexdigest(),
                    meta_hash=store.meta_hash(metadata),
                    created_at="",
                    last_updated="",
                    embedding=_VEC,
                )


def _rows() -> list[tuple]:
    db = store.get_db()
    try:
        return [
            tuple(r)
            for r in db.execute(
                f"SELECT {_ROW_COLUMNS} FROM chunks ORDER BY file_path, section_id"
            ).fetchall()
        ]
    finally:
        db.close()


# ── 1. upgraded store == clean store ────────────────────────────────────────


def test_an_upgraded_store_reindexes_to_exactly_the_clean_store(tmp_store, monkeypatch):
    paths = _seed_files(tmp_store)

    # A store indexed before the projection existed.
    _write_legacy_rows(paths)
    legacy = _rows()
    assert legacy, "the legacy fixture indexed nothing"
    assert all(row[5] is None for row in legacy), "legacy rows must carry no stamp"
    assert store.search_fts("eu-west-1"), "the retired wording is live pre-migration"

    # The migration: one reindex pass.
    results = _index_all(paths)
    assert all(r["error"] is None for r in results), [r["error"] for r in results]
    assert sum(r["chunks_reprojected"] + r["chunks_stamped"] for r in results) == len(legacy)
    upgraded = _rows()

    # A clean store: empty database, same files, nothing else.
    monkeypatch.setattr(config, "db_path", str(tmp_store / "clean.db"))
    store.init_db()
    assert _rows() == []
    _index_all(paths)
    clean = _rows()

    assert upgraded == clean
    assert {row[5] for row in upgraded} == {PROJECTION_VERSION}
    # Every projected hash agrees, which is the claim stated as such.
    assert [row[4] for row in upgraded] == [row[4] for row in clean]
    assert all(row[4] is not None for row in upgraded)


def test_the_migrated_index_answers_the_same_way_the_clean_one_does(tmp_store, monkeypatch):
    paths = _seed_files(tmp_store)
    _write_legacy_rows(paths)
    _index_all(paths)

    def _answers() -> dict[str, list[str]]:
        return {
            term: sorted(hit["content"] for hit in store.search_fts(term))
            for term in ("Nomad", "eu-west-1", "Kubernetes", "Backups", "MySQL")
        }

    upgraded = _answers()
    monkeypatch.setattr(config, "db_path", str(tmp_store / "clean.db"))
    store.init_db()
    _index_all(paths)

    assert upgraded == _answers()
    # And the retired wording is in neither: the migration removed the old FTS
    # tokens rather than leaving them beside the successor.
    assert upgraded["Nomad"] == [] and upgraded["eu-west-1"] == []
    assert upgraded["Kubernetes"] and upgraded["Backups"]


def test_the_raw_files_and_their_history_are_untouched_by_the_migration(tmp_store):
    paths = _seed_files(tmp_store)
    before = {p: Path(p).read_bytes() for p in paths}
    history = tmp_store / "projects" / "demo-history.md"
    assert history.exists(), "the executor's sidecar is part of what must survive"

    _write_legacy_rows(paths)
    _index_all(paths)

    assert {p: Path(p).read_bytes() for p in paths} == before
    # The retired wording is still readable from the file, which is the whole
    # point of projecting the index rather than rewriting the source.
    demo = (tmp_store / "projects" / "demo.md").read_text(encoding="utf-8")
    assert "Deploy target is Nomad." in demo and "eu-west-1" in demo
    assert "Superseded (" in history.read_text(encoding="utf-8")


# ── 2. rollback ─────────────────────────────────────────────────────────────


def test_the_added_columns_are_nullable_so_older_code_can_still_write(tmp_store):
    """A downgrade leaves the columns in place; old code never names them."""
    db = store.get_db()
    try:
        info = {row[1]: row for row in db.execute("PRAGMA table_info(chunks)").fetchall()}
    finally:
        db.close()
    for column in ("projected_hash", "projection_version"):
        assert column in info, f"{column} must exist after the upgrade"
        assert info[column][3] == 0, f"{column} must be nullable for a rollback"
        assert info[column][4] is None, f"{column} must have no default"

    # Old code calls the writer without the projection arguments at all.
    paths = _seed_files(tmp_store)
    _write_legacy_rows(paths)
    rows = _rows()
    assert rows and all(row[4] is None and row[5] is None for row in rows)
    # And the rows it wrote are readable and searchable, stamp or no stamp.
    assert store.search_fts("Backups")


def test_rollback_does_not_revive_expired_authority_or_retired_conclusions(tmp_store):
    """Retirement lives in markdown and git, so a rebuilt index re-derives it.

    The downgrade case, played out: the index is thrown away and rebuilt by
    code that writes no projection stamp (the pre-upgrade indexer). Every
    record that was retired is still retired, because the verdict is read from
    the file's live frontmatter and its path — never from the index.
    """
    paths = _seed_files(tmp_store)
    _index_all(paths)

    expected = {
        "decisions/db-choice.md": ("retired", "status:archived"),
        "archive/2026-03-01.md": ("retired", "path:archive"),
        "projects/oncall-grant.md": ("retired", "expired"),
        "insights/plain.md": ("current", "status:active"),
    }

    def _verdicts() -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for rel in expected:
            path = tmp_store / rel
            meta, _ = parser.parse_markdown(path.read_text(encoding="utf-8"))
            elig = eligibility(meta, path=str(path))
            out[rel] = (elig.state, elig.reason)
        return out

    assert _verdicts() == expected

    # Throw the index away entirely and rebuild it the pre-upgrade way.
    Path(config.db_path).unlink()
    store.init_db()
    _write_legacy_rows(paths)
    assert _rows(), "the rebuild indexed nothing"

    assert _verdicts() == expected

    # Search agrees, and the note retired *by location* is the sharp case: its
    # frontmatter still says `status: active`, so no status filter excludes it
    # — the index serves it, and the lifecycle verdict retires it anyway.
    hits = [h for h in store.search_hybrid("ingest pipeline", _VEC, top_k=5, threshold=0.0)
            if h["file_path"].endswith("archive/2026-03-01.md")]
    assert hits, "history must stay reachable after the rebuild"
    assert hits[0]["currency"] == "retired"
    assert hits[0]["currency_reason"] == "path:archive"

    # The archived decision keeps its declared retirement and stays out of
    # default recall, while the file itself is still there to be read.
    assert not [h for h in store.search_hybrid("MySQL ledger", _VEC, top_k=5, threshold=0.0)
                if h["file_path"].endswith("decisions/db-choice.md")]
    assert "MySQL" in (tmp_store / "decisions" / "db-choice.md").read_text(encoding="utf-8")


def test_a_rebuild_from_files_alone_reproduces_the_projected_rows(tmp_store):
    """The index is derived state: deleting it loses nothing but time."""
    paths = _seed_files(tmp_store)
    _index_all(paths)
    before = _rows()
    assert before

    Path(config.db_path).unlink()
    store.init_db()
    _index_all(paths)

    assert _rows() == before
    assert reconcile.plan(
        reconcile.derive(
            str(tmp_store / "projects" / "demo.md"),
            (tmp_store / "projects" / "demo.md").read_text(encoding="utf-8"),
        )
    ).is_noop
