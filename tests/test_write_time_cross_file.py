"""Write-time contradiction check: cross-file targets and revision preconditions.

The checker (``runner._check_contradictions``) retrieves candidate memories
across every stored file, but ``executor.apply_operations`` is per-file and
the write-time applier used to call it on the just-saved file only. This
module drives the real boundary end to end — ``save_memory`` → inline index →
``schedule_contradiction_check`` → retrieval → proposal → ``_translate_ops`` →
``_route_ops`` → ``apply_operations`` → git commit → reindex — with two memory
files, A saved and indexed first, then B saved so the check runs against B with
A's chunk in the candidate set. Only the model is faked (at the same
``_call_llm_with_fallback`` seam the live path calls) and the embedder returns
a fixed vector so every chunk is a candidate. Real SQLite, real git, real
files under ``tmp_path``.

Cases, in order: (a) a proposal naming a fact that lives in A; (b) naming an
id present in both A and B; (c) naming an id in neither; (d) A's target line
edited on disk between proposal and application; (e) a DELETE with no
replacement text, on an age-eligible and on an identity document; (f) an
UPDATE with no replacement text.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import types
from collections import Counter
from unittest.mock import patch

import pytest

from palinode.consolidation import write_time
from palinode.core import store
from palinode.core.config import config
from palinode.core.save import save_memory

_VECTOR = [0.1] * 1024

_ALPHA = "- Alpha fact one <!-- fact:alpha-f1 -->\n- Shared fact <!-- fact:shared-id -->\n"
_BETA = "- Beta fact one <!-- fact:beta-f1 -->\n- Shared fact <!-- fact:shared-id -->\n"


def _fake_embed(text: str, backend: str = "local") -> list[float]:
    return list(_VECTOR)


class _FakeModel:
    """Stands in for ``runner._call_llm_with_fallback``.

    ``response`` is returned verbatim; ``prompts`` records every user prompt so
    a test can assert what the candidate set contained; ``before_reply`` runs
    inside the call, i.e. after the candidates were retrieved and before the
    proposal reaches the applier — the window case (d) needs.
    """

    def __init__(self) -> None:
        self.response = json.dumps({"operation": "NOOP"})
        self.prompts: list[str] = []
        self.before_reply = None

    def __call__(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        self.prompts.append(user_prompt)
        if self.before_reply is not None:
            self.before_reply()
        return self.response, "fake-model"

    def propose(self, **op) -> None:
        self.response = json.dumps(op)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Git-backed memory_dir with a real SQLite store, fake embedder, fake model."""
    root = str(tmp_path)
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(["git", "-C", root, "config", "user.email", "t@t.test"], check=True)
    subprocess.run(["git", "-C", root, "config", "user.name", "test"], check=True)

    monkeypatch.setattr(config, "memory_dir", root)
    monkeypatch.setattr(config, "db_path", os.path.join(root, ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", True)
    monkeypatch.setattr(config.auto_summary, "enabled", False)
    monkeypatch.setattr(config.consolidation.write_time, "enabled", True)
    monkeypatch.setattr(config.consolidation.write_time, "queue_max_size", 10)
    monkeypatch.setattr(write_time, "_queue", None)

    store.init_db()
    # update.md is what lets _check_contradictions reach its model call
    # instead of short-circuiting to ADD.
    prompts = tmp_path / "specs" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "update.md").write_text("Return the operation as JSON.\n")

    model = _FakeModel()
    with patch("palinode.core.embedder.embed", side_effect=_fake_embed), \
            patch("palinode.consolidation.runner._call_llm_with_fallback", model):
        yield types.SimpleNamespace(root=root, model=model)


def _git(root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, check=True,
    ).stdout


def _save(content: str, slug: str, *, sync: bool = True, type: str = "Decision") -> dict:
    return save_memory(content=content, type=type, slug=slug, sync=sync)


def _body(path: str) -> str:
    text = open(path, encoding="utf-8").read()
    return text.split("---\n", 2)[2].strip() + "\n"


def _index_rows(path: str) -> list[dict]:
    db = store.get_db()
    try:
        rows = db.execute(
            "SELECT file_path, section_id, content, content_hash FROM chunks "
            "WHERE file_path = ?",
            (path,),
        ).fetchall()
    finally:
        db.close()
    return [dict(r) for r in rows]


def _freshness(path: str) -> set[str]:
    return {r["freshness"] for r in store.check_freshness(_index_rows(path))}


def _seed(env, *, type: str = "Decision") -> tuple[str, str]:
    """Save A (model says NOOP), then arm the model for B's save. Returns A's
    path and the git HEAD after A so tests can diff the commits B produced."""
    a = _save(_ALPHA, "alpha", type=type)
    assert a["indexed"] and a["git_committed"], a
    env.model.prompts.clear()
    return a["file_path"], _git(env.root, "rev-parse", "HEAD").strip()


def _commits_since(root: str, base: str) -> list[tuple[str, list[str]]]:
    """[(subject, [files]), …] newest first, for commits after ``base``."""
    out = _git(root, "log", f"{base}..HEAD", "--format=%x00%s", "--name-only")
    commits: list[tuple[str, list[str]]] = []
    for block in out.split("\x00")[1:]:
        subject, *files = [ln for ln in block.splitlines() if ln.strip()]
        commits.append((subject, files))
    return commits


# ── (a) target fact lives in another file ──────────────────────────────────


def test_cross_file_update_routes_to_owning_file(env):
    """A proposal naming A's fact, generated while saving B, is applied to A —
    not dropped as unmatched against B, and B is not touched."""
    a_path, base = _seed(env)
    env.model.propose(operation="UPDATE", target_id="alpha-f1",
                      new_text="Alpha fact one (revised)")

    b = _save(_BETA, "beta")

    # Evidence the retrieval spans files: A's chunk was in front of the model.
    assert len(env.model.prompts) == 1
    assert "decisions/alpha.md" in env.model.prompts[0]

    stats = b["write_time_check"]["applied_stats"]
    assert stats["updated"] == 1
    assert stats["unmatched"] == 0
    assert stats["ambiguous_rejected"] == 0
    assert stats["stale_rejected"] == 0
    assert stats["translation_skipped"] == 0
    # The routing input never leaks into the returned proposal.
    assert "candidates" not in b["write_time_check"]["operations"][0]

    assert "- Alpha fact one (revised) <!-- fact:alpha-f1 -->" in _body(a_path)
    assert "Shared fact <!-- fact:shared-id -->" in _body(a_path)
    assert _body(b["file_path"]) == _BETA

    # Provenance: the dedup pass is its own commit, on A, after B's auto-save.
    commits = _commits_since(env.root, base)
    assert commits[0] == (f"{config.git.commit_prefix} write-time dedup: decisions/alpha.md",
                          ["decisions/alpha.md"])
    assert commits[1][0] == f"{config.git.commit_prefix} auto-save: decisions/beta.md"
    assert _git(env.root, "status", "--porcelain", "--", "decisions/") == ""

    # Index reconciliation: A's row was re-indexed in-process, not left stale
    # for the watcher.
    assert _freshness(a_path) == {"valid"}
    assert any("(revised)" in r["content"] for r in _index_rows(a_path))
    assert _freshness(b["file_path"]) == {"valid"}


def test_cross_file_delete_supersedes_in_owning_file_with_history(env):
    """A write-time DELETE (→ SUPERSEDE) on A's fact tombstones A's line,
    writes A's history sibling, and commits both together."""
    a_path, base = _seed(env)
    env.model.propose(operation="DELETE", target_id="alpha-f1",
                      new_text="Alpha fact one, corrected by beta",
                      reason="contradicted by beta")

    b = _save(_BETA, "beta")

    stats = b["write_time_check"]["applied_stats"]
    assert stats["superseded"] == 1
    assert stats["unmatched"] == 0

    body = _body(a_path)
    assert "~~Alpha fact one~~ [superseded" in body
    assert "- Alpha fact one, corrected by beta <!-- fact:supersedes-alpha-f1 -->" in body
    # The successor line is the op's own text — B's body never lands in A.
    assert "Beta fact one" not in body
    assert "fact:beta-f1" not in body
    assert _body(b["file_path"]) == _BETA

    history = os.path.join(env.root, "decisions", "alpha-history.md")
    assert os.path.exists(history)
    assert "Superseded" in open(history).read()
    assert "contradicted by beta" in open(history).read()
    assert not os.path.exists(os.path.join(env.root, "decisions", "beta-history.md"))

    subject, files = _commits_since(env.root, base)[0]
    assert subject == f"{config.git.commit_prefix} write-time dedup: decisions/alpha.md"
    assert sorted(files) == ["decisions/alpha-history.md", "decisions/alpha.md"]
    assert _git(env.root, "status", "--porcelain", "--", "decisions/") == ""
    assert _freshness(a_path) == {"valid"}


# ── (b) the id exists in both files ────────────────────────────────────────


def test_colliding_id_is_rejected_not_applied_to_saved_file(env):
    """When both A and B carry the target id the proposal is rejected outright:
    neither file changes, no history is written, no dedup commit is made."""
    a_path, base = _seed(env)
    env.model.propose(operation="DELETE", target_id="shared-id",
                      new_text="Shared fact, revised", reason="contradicted")

    b = _save(_BETA, "beta")

    stats = b["write_time_check"]["applied_stats"]
    assert stats["ambiguous_rejected"] == 1
    # Rejected before the executor ran, so its counters are absent — nothing
    # was superseded or reported unmatched anywhere.
    assert stats.get("superseded", 0) == 0
    assert stats.get("unmatched", 0) == 0

    assert _body(a_path) == _ALPHA
    assert _body(b["file_path"]) == _BETA
    assert not any(n.endswith("-history.md") for n in os.listdir(os.path.join(env.root, "decisions")))
    assert [s for s, _ in _commits_since(env.root, base)] == [
        f"{config.git.commit_prefix} auto-save: decisions/beta.md",
    ]
    assert _git(env.root, "status", "--porcelain", "--", "decisions/") == ""


# ── (c) the id exists in neither file ──────────────────────────────────────


def test_unknown_id_is_unmatched_with_no_mutation(env):
    """An id no candidate carries falls back to the saved file, where the
    executor reports it unmatched. Nothing is written or committed."""
    a_path, base = _seed(env)
    env.model.propose(operation="UPDATE", target_id="ghost", new_text="x")

    b = _save(_BETA, "beta")

    stats = b["write_time_check"]["applied_stats"]
    assert stats["unmatched"] == 1
    assert stats["updated"] == 0
    assert stats["ambiguous_rejected"] == 0
    assert stats["stale_rejected"] == 0

    assert _body(a_path) == _ALPHA
    assert _body(b["file_path"]) == _BETA
    assert [s for s, _ in _commits_since(env.root, base)] == [
        f"{config.git.commit_prefix} auto-save: decisions/beta.md",
    ]
    assert _freshness(a_path) == {"valid"}


# ── (d) the target changed between proposal and application ────────────────


def test_stale_target_is_rejected_not_overwritten(env):
    """A's target line is edited on disk after the candidates were retrieved
    and before the proposal is applied. The edit wins; the proposal is
    rejected on the stale section hash, not applied last-write-wins."""
    a_path, base = _seed(env)
    env.model.propose(operation="UPDATE", target_id="alpha-f1",
                      new_text="Alpha fact one (revised)")

    def edit_a_on_disk() -> None:
        text = open(a_path, encoding="utf-8").read()
        open(a_path, "w", encoding="utf-8").write(
            text.replace("Alpha fact one <!-- fact:alpha-f1 -->",
                         "Alpha fact one EDITED <!-- fact:alpha-f1 -->")
        )

    env.model.before_reply = edit_a_on_disk

    b = _save(_BETA, "beta")

    stats = b["write_time_check"]["applied_stats"]
    assert stats["stale_rejected"] == 1
    assert stats.get("updated", 0) == 0
    assert stats.get("unmatched", 0) == 0

    assert "- Alpha fact one EDITED <!-- fact:alpha-f1 -->" in _body(a_path)
    assert "(revised)" not in _body(a_path)
    assert _body(b["file_path"]) == _BETA
    assert [s for s, _ in _commits_since(env.root, base)] == [
        f"{config.git.commit_prefix} auto-save: decisions/beta.md",
    ]
    # The precondition was judged from the index's section hash: the row is
    # still the pre-edit one (nothing was applied, so nothing was reindexed).
    assert _freshness(a_path) == {"stale"}


# ── the saved file's own facts keep working ────────────────────────────────


def test_saved_file_own_fact_still_targets_saved_file(env):
    """B's own chunk is a candidate too; a proposal naming B's fact routes to B
    — the pre-existing single-file behaviour, now by ownership rather than
    by default."""
    a_path, base = _seed(env)
    env.model.propose(operation="UPDATE", target_id="beta-f1",
                      new_text="Beta fact one (revised)")

    b = _save(_BETA, "beta")

    stats = b["write_time_check"]["applied_stats"]
    assert stats["updated"] == 1
    assert stats["ambiguous_rejected"] == 0
    assert "- Beta fact one (revised) <!-- fact:beta-f1 -->" in _body(b["file_path"])
    assert _body(a_path) == _ALPHA
    assert _commits_since(env.root, base)[0] == (
        f"{config.git.commit_prefix} write-time dedup: decisions/beta.md",
        ["decisions/beta.md"],
    )
    assert _freshness(b["file_path"]) == {"valid"}


# ── partial failure and the disk-marker path ───────────────────────────────


def test_executor_failure_leaves_no_partial_state(env):
    """If the executor raises, the save has still landed, nothing else is
    written or committed, and the stats carry only the routing counters."""
    a_path, base = _seed(env)
    env.model.propose(operation="UPDATE", target_id="alpha-f1",
                      new_text="Alpha fact one (revised)")

    with patch("palinode.consolidation.executor.apply_operations",
               side_effect=RuntimeError("executor exploded")):
        b = _save(_BETA, "beta")

    assert b["save_outcome"] == "created"
    assert b["write_time_check"]["applied_stats"] == {
        "translation_skipped": 0, "ambiguous_rejected": 0, "stale_rejected": 0,
    }
    assert _body(a_path) == _ALPHA
    assert [s for s, _ in _commits_since(env.root, base)] == [
        f"{config.git.commit_prefix} auto-save: decisions/beta.md",
    ]
    assert _git(env.root, "status", "--porcelain", "--", "decisions/") == ""


def test_pending_marker_is_swept_and_routed_through_the_worker(env):
    """A save from a context with no event loop leaves a disk marker; the
    startup sweep re-enqueues it and the worker applies the routed op to A.
    The marker is consumed and nothing is left pending."""
    a_path, base = _seed(env)
    env.model.propose(operation="UPDATE", target_id="alpha-f1",
                      new_text="Alpha fact one (revised)")

    b = _save(_BETA, "beta", sync=False)
    assert "write_time_check" not in b
    pending = write_time._pending_dir()
    assert [n for n in os.listdir(pending) if n.endswith(".json")]
    assert _body(a_path) == _ALPHA  # nothing applied yet

    async def drain() -> None:
        state = types.SimpleNamespace()
        await write_time.start_worker(state)  # sweeps, then starts the loop
        await asyncio.wait_for(write_time._get_queue().join(), timeout=30)
        await write_time.stop_worker(state)

    asyncio.run(drain())

    assert [n for n in os.listdir(pending) if n.endswith(".json")] == []
    assert "- Alpha fact one (revised) <!-- fact:alpha-f1 -->" in _body(a_path)
    assert _body(b["file_path"]) == _BETA
    assert _commits_since(env.root, base)[0] == (
        f"{config.git.commit_prefix} write-time dedup: decisions/alpha.md",
        ["decisions/alpha.md"],
    )
    assert _freshness(a_path) == {"valid"}


# ── (e) DELETE with no replacement text ────────────────────────────────────
#
# A text-less DELETE used to become a SUPERSEDE whose `new_text` fell back to
# the saved item's content — the whole body of B — so a multi-line save was
# inserted into A as one fact line carrying every one of B's fact ids. It is
# now an ARCHIVE: the target is retired into history, nothing is inserted.

_BETA_MULTI = (
    "- Beta fact one <!-- fact:beta-f1 -->\n"
    "- Beta fact two <!-- fact:beta-f2 -->\n"
    "- Beta fact three <!-- fact:beta-f3 -->\n"
)


def _fact_marker_counts(root: str) -> Counter[str]:
    """How many times each ``<!-- fact:ID -->`` marker occurs across every
    ``.md`` under ``root`` (history siblings included) — an id seen twice is
    a duplicate, whichever file(s) carry it."""
    counts: Counter[str] = Counter()
    for dirpath, _dirs, files in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        for name in files:
            if name.endswith(".md"):
                text = open(os.path.join(dirpath, name), encoding="utf-8").read()
                counts.update(re.findall(r"<!-- fact:([^\s]+) -->", text))
    return counts


def test_textless_delete_archives_target_without_inserting_saved_body(env):
    """A multi-line B save with a DELETE naming A's fact and no replacement
    text: A's fact is archived into its history sibling, A gains no line, and
    no fact id is duplicated anywhere in the store."""
    a_path, base = _seed(env)
    env.model.propose(operation="DELETE", target_id="alpha-f1",
                      reason="retired by beta")

    b = _save(_BETA_MULTI, "beta")

    stats = b["write_time_check"]["applied_stats"]
    assert stats["archived"] == 1
    assert stats["superseded"] == 0
    assert stats["unmatched"] == 0
    assert stats["protected_rejected"] == 0

    body = _body(a_path)
    assert body == "- Shared fact <!-- fact:shared-id -->\n"
    assert "Beta fact" not in body
    assert _body(b["file_path"]) == _BETA_MULTI

    history = os.path.join(env.root, "decisions", "alpha-history.md")
    assert os.path.exists(history)
    history_text = open(history, encoding="utf-8").read()
    assert "Archived: Alpha fact one (reason: retired by beta)" in history_text
    assert "Beta fact" not in history_text

    # Every fact id in the store occurs exactly once: alpha-f1 only in the
    # history entry, B's three ids only in B.
    counts = _fact_marker_counts(env.root)
    assert counts == {
        "alpha-f1": 1, "shared-id": 1, "beta-f1": 1, "beta-f2": 1, "beta-f3": 1,
    }

    subject, files = _commits_since(env.root, base)[0]
    assert subject == f"{config.git.commit_prefix} write-time dedup: decisions/alpha.md"
    assert sorted(files) == ["decisions/alpha-history.md", "decisions/alpha.md"]
    assert _git(env.root, "status", "--porcelain", "--", "decisions/") == ""
    assert _freshness(a_path) == {"valid"}
    assert not any("alpha-f1" in r["content"] for r in _index_rows(a_path))


def test_textless_delete_on_identity_document_is_rejected_untouched(env, caplog):
    """The same DELETE against a fact in ``people/`` (superseded-only): the
    executor's ADR-020 guard rejects the ARCHIVE — it names no
    ``superseded_by`` — and the target is byte-identical, with no history,
    no dedup commit, and the reason in the log."""
    a_path, base = _seed(env, type="PersonMemory")
    assert a_path.endswith(os.path.join("people", "alpha.md"))
    before = open(a_path, "rb").read()
    env.model.propose(operation="DELETE", target_id="alpha-f1",
                      reason="retired by beta")

    with caplog.at_level(logging.WARNING):
        b = _save(_BETA_MULTI, "beta", type="PersonMemory")

    assert "people/alpha.md" in env.model.prompts[0]
    stats = b["write_time_check"]["applied_stats"]
    assert stats["protected_rejected"] == 1
    assert stats["archived"] == 0
    assert stats["superseded"] == 0
    assert stats["unmatched"] == 0

    assert open(a_path, "rb").read() == before
    assert _body(b["file_path"]) == _BETA_MULTI
    assert not os.path.exists(os.path.join(env.root, "people", "alpha-history.md"))
    assert [s for s, _ in _commits_since(env.root, base)] == [
        f"{config.git.commit_prefix} auto-save: people/beta.md",
    ]
    assert _git(env.root, "status", "--porcelain", "--", "people/") == ""
    # The frontmatter signal (type: PersonMemory) outranks the people/ path
    # rule in retirement.classify — the log names the rule that decided it.
    assert (
        "ARCHIVE rejected by retirement_policy=superseded-only guard "
        "(type:PersonMemory): age/staleness is not a retirement reason"
        in caplog.text
    )
    assert _freshness(a_path) == {"valid"}


def test_translate_ops_textless_delete_never_carries_saved_body():
    """Unit pin on the translation step alone: a DELETE without ``new_text``
    becomes ``{"op": "ARCHIVE", "id": ..., "reason": ...}`` and the saved
    item's (multi-line) content appears nowhere in the translated op."""
    ops = [
        {
            "operation": "DELETE",
            "target_id": "alpha-f1",
            "reason": "retired by beta",
            "item": {"id": "decisions-beta", "content": _BETA_MULTI},
        }
    ]

    translated = write_time._translate_ops(ops, "/tmp/decisions/beta.md")

    assert translated == [
        {"op": "ARCHIVE", "id": "alpha-f1", "reason": "retired by beta"},
    ]
    assert "beta-f" not in json.dumps(translated)
    assert "Beta fact" not in json.dumps(translated)


# ── (f) UPDATE with no replacement text ────────────────────────────────────
#
# A text-less UPDATE used to fall back to the saved item's content — the whole
# body of B — as `new_text`, rewriting A's target line as B's entire body and
# duplicating every fact id it contained. It is malformed: nothing is
# translated, the skip is counted, and the target is untouched.


def test_textless_update_is_skipped_and_target_untouched(env, caplog):
    """A multi-line B save with an UPDATE naming A's fact and no replacement
    text: nothing reaches the executor, A is byte-identical, no fact id is
    duplicated anywhere, the skip is counted, and the warning names the id."""
    a_path, base = _seed(env)
    before = open(a_path, "rb").read()
    env.model.propose(operation="UPDATE", target_id="alpha-f1",
                      reason="revised by beta")

    with caplog.at_level(logging.WARNING):
        b = _save(_BETA_MULTI, "beta")

    assert "decisions/alpha.md" in env.model.prompts[0]
    # Nothing was routed, so the translator's counter is the whole report.
    assert b["write_time_check"]["applied_stats"] == {"translation_skipped": 1}

    assert open(a_path, "rb").read() == before
    assert _body(b["file_path"]) == _BETA_MULTI
    assert _fact_marker_counts(env.root) == {
        "alpha-f1": 1, "shared-id": 1, "beta-f1": 1, "beta-f2": 1, "beta-f3": 1,
    }
    assert not os.path.exists(os.path.join(env.root, "decisions", "alpha-history.md"))
    assert [s for s, _ in _commits_since(env.root, base)] == [
        f"{config.git.commit_prefix} auto-save: decisions/beta.md",
    ]
    assert _git(env.root, "status", "--porcelain", "--", "decisions/") == ""
    assert _freshness(a_path) == {"valid"}

    assert (
        "write-time: UPDATE skipped — fact id='alpha-f1' carries no new_text"
        in caplog.text
    )


def test_translate_ops_textless_update_produces_no_op(caplog):
    """Unit pin on the translation step alone: an UPDATE without ``new_text``
    translates to nothing — the saved item's (multi-line) content is never
    promoted to replacement text — and the warning names the target id."""
    ops = [
        {
            "operation": "UPDATE",
            "target_id": "alpha-f1",
            "reason": "revised by beta",
            "item": {"id": "decisions-beta", "content": _BETA_MULTI},
        }
    ]

    with caplog.at_level(logging.WARNING):
        translated = write_time._translate_ops(ops, "/tmp/decisions/beta.md")

    assert translated == []
    assert "fact id='alpha-f1'" in caplog.text
