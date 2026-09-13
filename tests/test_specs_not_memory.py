"""The store's own prompts under ``specs/prompts/`` are not memory.

A provisioned store keeps its editable consolidation prompts at
``specs/prompts/*.md``. They carry ``id``/``task``/``model``/``active``/
``version`` frontmatter, which is enough to look like a memory to a walk that
tests only the first path segment against a set naming the legacy top-level
``prompts`` directory — ``parts[0]`` is ``specs``, so every such walk let them
through. One shared predicate (``palinode.core.skip_dirs.is_skipped_path``)
now answers "is this a memory file?" for every surface that had its own copy
of that set, matching on every directory segment.

One test per surface, each against a tmp store holding a real memory beside a
prompt file in its shipped shape.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from palinode.api.server import app
from palinode.core.config import config
from palinode.core.skip_dirs import ALWAYS_SKIP, is_skipped_path

client = TestClient(app)


def _write(path: Path, frontmatter: str, body: str = "Body text.\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}---\n\n{body}")
    return path


def _write_prompt(store: Path, name: str = "compaction", extra: str = "") -> Path:
    """A store prompt in its shipped shape (see ``specs/prompts/compaction.md``)."""
    return _write(
        store / "specs" / "prompts" / f"{name}.md",
        f"id: prompt-{name}\n"
        f"name: {name}\n"
        f"task: {name}\n"
        'model: "*"\n'
        "version: 4\n"
        "active: true\n" + extra,
        body="# Compaction Prompt\n\nYou are a memory compaction engine.\n",
    )


def _write_memory(store: Path, rel: str = "decisions/keep.md", extra: str = "") -> Path:
    return _write(
        store / rel,
        "id: decision-keep\n"
        "title: Keep This\n"
        "type: Decision\n"
        "category: decisions\n"
        "description: a real memory\n"
        "created_at: 2026-09-01T00:00:00Z\n"
        "last_updated: 2026-09-01T00:00:00Z\n" + extra,
    )


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp store: one real memory, one store prompt under ``specs/prompts``."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    _write_memory(tmp_path)
    _write_prompt(tmp_path)
    return tmp_path


# ── the shared predicate ─────────────────────────────────────────────────────


def test_skipped_on_every_segment_not_just_the_first():
    assert is_skipped_path("specs/prompts/compaction.md")
    assert is_skipped_path("specs/amr.md")
    assert is_skipped_path("foo/specs/bar.md")
    assert is_skipped_path("logs/deep/nested/note.md")


def test_whole_segments_only():
    # Prefixes are not segments, and the filename is not a directory.
    assert not is_skipped_path("specsheet/x.md")
    assert not is_skipped_path("decisions/specs-review.md")
    assert not is_skipped_path("specs.md")
    assert not is_skipped_path("logs.md")
    assert not is_skipped_path("keep.md")


def test_native_separator_matches():
    assert is_skipped_path(os.path.join("specs", "prompts", "compaction.md"))
    assert not is_skipped_path(os.path.join("decisions", "keep.md"))


def test_extra_adds_and_never_subtracts():
    # `daily` is a per-surface decision — the digest reads it, /list does not.
    assert not is_skipped_path("daily/2026-09-01.md")
    assert is_skipped_path("daily/2026-09-01.md", {"daily"})
    # An extras set that omits a never-memory dir cannot re-admit it.
    assert is_skipped_path("specs/prompts/compaction.md", {"daily"})
    assert "specs" in ALWAYS_SKIP and "daily" not in ALWAYS_SKIP


# ── GET /list, and the SessionStart hook's core_only injection ───────────────


def test_list_excludes_store_prompts(store: Path):
    res = client.get("/list")
    assert res.status_code == 200
    files = [row["file"] for row in res.json()]
    assert "decisions/keep.md" in files
    assert not any("prompts" in f for f in files), files


def test_list_core_only_excludes_store_prompts(store: Path):
    """The shipped SessionStart hook injects ``GET /list?core_only=true``.

    The prompt here declares ``core: true`` — not its shipped frontmatter, but
    the point is that the skip is by path, not by the prompt happening to lack
    the flag.
    """
    _write_prompt(store, "extraction", extra="core: true\n")
    _write_memory(store, "decisions/core.md", extra="core: true\n")

    res = client.get("/list", params={"core_only": "true"})
    assert res.status_code == 200
    files = [row["file"] for row in res.json()]
    assert "decisions/core.md" in files
    assert not any("prompts" in f for f in files), files


def test_collect_memory_files_excludes_store_prompts(store: Path):
    from palinode.api.routers.memory import collect_memory_files

    files = {row["file"] for row in collect_memory_files()}
    assert "decisions/keep.md" in files
    assert not any("prompts" in f for f in files), files


def test_caller_supplied_skip_dirs_cannot_re_admit_prompts(store: Path):
    """``skip_dirs=`` replaces the surface's own dirs, never the shared floor."""
    from palinode.api.routers.memory import collect_memory_files

    files = {row["file"] for row in collect_memory_files(skip_dirs=set())}
    assert "decisions/keep.md" in files
    assert not any("prompts" in f for f in files), files


# ── provenance UI ────────────────────────────────────────────────────────────


def test_ui_does_not_browse_store_prompts(store: Path):
    from palinode.api.ui.views import is_browsable_memory, scan_memory_files

    assert is_browsable_memory("decisions/keep.md")
    assert not is_browsable_memory("specs/prompts/compaction.md")

    paths = {row["path"] for row in scan_memory_files()}
    assert "decisions/keep.md" in paths
    assert not any("prompts" in p for p in paths), paths


# ── advisory project review ──────────────────────────────────────────────────


def test_review_does_not_queue_store_prompts(store: Path):
    """A prompt tagged with the project is still out of the review's scope."""
    from palinode.core import review as review_mod

    _write_prompt(store, "consolidation", extra="entities: [project/alpha]\n")
    _write_memory(store, "decisions/alpha.md", extra="entities: [project/alpha]\n")

    scope = review_mod._scope_files("project/alpha")
    assert scope == {os.path.join("decisions", "alpha.md")}

    out = review_mod.run_review("alpha")
    named = [
        review_mod._finding_file(item)
        for findings in out["findings"].values()
        for item in findings
    ]
    assert not any("prompts" in (f or "") for f in named), named


# ── session-start context digest ─────────────────────────────────────────────


def test_context_digest_does_not_select_store_prompts(store: Path):
    from palinode.core.context_prime import build_context_digest

    _write_prompt(store, "digest", extra="core: true\n")
    _write_memory(store, "decisions/core.md", extra="core: true\n")

    digest = build_context_digest()
    selected = {
        row["file"]
        for section in ("core_memories", "recent_decisions", "open_action_items",
                        "recent_snapshots")
        for row in digest[section]
    }
    assert "decisions/core.md" in selected
    assert not any("prompts" in f for f in selected), selected


# ── backed_by propagation ────────────────────────────────────────────────────


def test_propagate_does_not_reach_store_prompts(store: Path):
    from palinode.consolidation.propagate import find_dependents

    _write_prompt(store, "update", extra="backed_by: [decisions/keep]\n")
    _write_memory(store, "insights/dependent.md", extra="backed_by: [decisions/keep]\n")

    found = find_dependents(["decisions/keep"], base_dir=str(store))
    assert str(store / "insights" / "dependent.md") in found
    assert not any("prompts" in p for p in found), found


# ── the indexer ──────────────────────────────────────────────────────────────


def test_watcher_does_not_index_store_prompts(store: Path):
    # ``config.palinode_dir`` is ``memory_dir``, which the fixture points here.
    from palinode.indexer.watcher import PalinodeHandler

    handler = PalinodeHandler()
    try:
        assert handler.is_valid_file(str(store / "decisions" / "keep.md"))
        assert not handler.is_valid_file(str(store / "specs" / "prompts" / "compaction.md"))
    finally:
        handler.shutdown()


def test_watcher_indexes_a_store_that_lives_under_a_specs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The skip set is matched relative to the store root, not on the whole path."""
    from palinode.indexer.watcher import PalinodeHandler

    store_root = tmp_path / "specs" / "palinode"
    _write_memory(store_root)
    monkeypatch.setattr(config, "memory_dir", str(store_root))

    handler = PalinodeHandler()
    try:
        assert handler.is_valid_file(str(store_root / "decisions" / "keep.md"))
    finally:
        handler.shutdown()


def test_reindex_gcs_prompt_chunks_a_previous_index_left_behind(
    store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A store indexed before this fix loses the prompt chunks on ``reindex``.

    Reproduces the reindex pass's own two steps against a real database — the
    walk filtered through ``is_valid_file``, then ``gc_orphaned_chunks`` over
    exactly what the walk yielded — without needing the embedder.
    """
    import glob

    from palinode.core import store as store_mod
    from palinode.indexer.watcher import PalinodeHandler
    from tests._store_helpers import upsert_chunks

    monkeypatch.setattr(config, "db_path", str(store / ".palinode.db"))
    monkeypatch.setattr(store_mod, "_db_checked", False)
    # The store already has its memory files — this is the "indexed before the
    # fix" case, not a misconfigured db_path.
    monkeypatch.setenv("PALINODE_ALLOW_FRESH_DB", "1")
    store_mod.init_db()

    memory_path = str(store / "decisions" / "keep.md")
    prompt_path = str(store / "specs" / "prompts" / "compaction.md")
    dimensions = config.embeddings.primary.dimensions
    upsert_chunks(
        [
            {
                "id": "keep-1",
                "file_path": memory_path,
                "section_id": "root",
                "category": "decisions",
                "content": "kept content",
                "metadata": {},
                "embedding": [0.0] * dimensions,
            },
            {
                "id": "prompt-1",
                "file_path": prompt_path,
                "section_id": "root",
                "category": "specs",
                "content": "you are a memory compaction engine",
                "metadata": {},
                "embedding": [0.0] * dimensions,
            },
        ],
        skip_unchanged=False,
    )

    handler = PalinodeHandler()
    try:
        files = [
            fp
            for fp in glob.glob(os.path.join(config.palinode_dir, "**/*.md"), recursive=True)
            if handler.is_valid_file(fp)
        ]
    finally:
        handler.shutdown()
    assert memory_path in files
    assert prompt_path not in files

    paths_removed, chunks_removed = store_mod.gc_orphaned_chunks(files)
    assert (paths_removed, chunks_removed) == (1, 1)

    db = store_mod.get_db()
    try:
        remaining = [
            row["file_path"] for row in db.execute("SELECT file_path FROM chunks")
        ]
    finally:
        db.close()
    assert remaining == [memory_path]
