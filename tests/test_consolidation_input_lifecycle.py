"""Consolidation input goes through the shared lifecycle classifier.

The defect as filed: ``_get_decisions_for_project`` withheld only ``status:
superseded``, so a decision archived in place still reached the model under
``ACTIVE_DECISIONS (governing this project)``; and facts the executor had
already retired in place reached ``EXISTING_FACTS`` indistinguishable from
current ones. The review probe captured both at the propose seam.

Now ``ACTIVE_DECISIONS`` excludes every retired decision (archived in place,
superseded, deprecated, retracted, expired, ``superseded_by``) while unmarked
legacy decisions stay in; and ``EXISTING_FACTS`` holds only current facts, with
the retired ones in their own ``RETIRED_FACTS`` block — same ids, same text,
same markers, so nothing about source identity or history is lost.

Every test captures the **actual propose input** (the assembled user prompt
handed to the LLM seam) and asserts on it. The end-to-end scenarios use real
files and a real SQLite store under ``tmp_path``; the embedder is the only
thing stubbed, and the executor is exercised as shipped.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from palinode.consolidation import runner
from palinode.core import store
from palinode.core.config import config
from palinode.core.context_prime import build_context_digest
from palinode.core.lifecycle import eligibility

PROMPT = "# Compaction Prompt\nReturn the operations JSON array.\n"
NOTE = [{"date": "2026-09-10", "content": "Check the current endpoint."}]
PAST = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
FUTURE = (datetime.now(UTC) + timedelta(days=30)).isoformat()
_VEC = [0.05] * 1024


@pytest.fixture
def project_store(tmp_path, monkeypatch):
    """Project ``demo`` with a tagged status doc and the prompt the runner reads."""
    monkeypatch.delenv("PALINODE_PROJECT", raising=False)
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.git, "auto_commit", False)
    monkeypatch.setattr(config.context, "auto_detect", True)
    prompts = tmp_path / "specs" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "compaction.md").write_text(PROMPT, encoding="utf-8")
    (tmp_path / "decisions").mkdir()
    (tmp_path / "projects").mkdir()
    (tmp_path / "projects" / "demo.md").write_text(
        "---\nid: projects-demo\n---\n\n"
        "- Endpoint B. <!-- fact:endpoint-b -->\n"
        "- Deploys go through the CI gate. <!-- fact:ci-gate -->\n",
        encoding="utf-8",
    )
    return tmp_path


def _decision(root, slug: str, body: str, **fm: Any) -> None:
    import yaml

    meta = {"id": f"decisions-{slug}", "name": slug, "entities": ["project/demo"], **fm}
    (root / "decisions" / f"{slug}.md").write_text(
        f"---\n{yaml.safe_dump(meta, default_flow_style=False)}---\n\n{body}\n",
        encoding="utf-8",
    )


def _capture() -> tuple[dict[str, str], Any]:
    seen: dict[str, str] = {}

    def _fake(system_prompt: str, user_prompt: str) -> tuple[str, str]:
        seen["user"] = user_prompt
        return "[]", "stub"

    return seen, _fake


def _sections(prompt: str) -> dict[str, str]:
    """Split the captured prompt into its ``## HEADING`` sections by name."""
    out: dict[str, str] = {}
    current = None
    for line in prompt.splitlines():
        if line.startswith("## "):
            current = line[3:].split(" (", 1)[0].strip()
            out[current] = ""
        elif current is not None:
            out[current] += line + "\n"
    return out


def _propose(project_id: str = "demo", notes=NOTE) -> str:
    seen, fake = _capture()
    ops, model = runner._consolidate_project(project_id, notes, llm_fn=fake)
    assert model == "stub" and ops == []
    return seen["user"]


# ── ACTIVE_DECISIONS eligibility ─────────────────────────────────────────────


def test_archived_in_place_decision_is_not_governing(project_store):
    _decision(project_store, "archived", "Use retired endpoint A.", status="archived")
    _decision(project_store, "active", "Use current endpoint B.", status="active")
    _decision(project_store, "superseded", "Use superseded endpoint Z.", status="superseded")
    prompt = _propose()
    decisions = _sections(prompt)["ACTIVE_DECISIONS"]
    assert "Use current endpoint B." in decisions
    assert "Use retired endpoint A." not in decisions
    assert "Use superseded endpoint Z." not in decisions
    assert "decisions/archived" not in prompt


@pytest.mark.parametrize(
    "fm",
    [
        {"status": "deprecated"},
        {"status": "retracted"},
        {"lifecycle": "archived"},
        {"status": "active", "superseded_by": "decisions/other"},
        {"status": "active", "expires_at": PAST},
    ],
    ids=["deprecated", "retracted", "ku-lifecycle", "superseded_by", "expired"],
)
def test_every_retired_signal_withholds_the_decision(project_store, fm):
    _decision(project_store, "retired", "Retired constraint.", **fm)
    _decision(project_store, "live", "Live constraint.", status="active")
    decisions = _sections(_propose())["ACTIVE_DECISIONS"]
    assert "Live constraint." in decisions
    assert "Retired constraint." not in decisions


def test_unmarked_legacy_and_unexpired_decisions_remain_governing(project_store):
    _decision(project_store, "legacy", "Legacy constraint with no status.")
    _decision(project_store, "licensed", "Licensed constraint.", status="active", expires_at=FUTURE)
    decisions = _sections(_propose())["ACTIVE_DECISIONS"]
    assert "Legacy constraint with no status." in decisions
    assert "Licensed constraint." in decisions


def test_guard_context_excludes_retired_decision_refs(project_store):
    """What the guard counts as "in the prompt" matches what the model saw."""
    _decision(project_store, "archived", "Retired.", status="archived")
    _decision(project_store, "active", "Live.", status="active")
    assembled = runner._assemble_prompt("demo", NOTE)
    assert assembled is not None
    assert assembled.context.decision_refs == frozenset({"decisions/active"})


def test_no_governing_decisions_omits_the_section(project_store):
    _decision(project_store, "archived", "Retired.", status="archived")
    assert "ACTIVE_DECISIONS" not in _propose()


# ── EXISTING_FACTS vs RETIRED_FACTS ──────────────────────────────────────────


def test_retired_facts_are_distinguished_without_losing_identity(project_store):
    (project_store / "projects" / "demo.md").write_text(
        "---\nid: projects-demo\n---\n\n"
        "- ~~Endpoint A.~~ [superseded 2026-09-10] <!-- fact:old -->\n"
        "- Endpoint B. <!-- fact:supersedes-old -->\n"
        "- ~~The sky is green.~~ [RETRACTED 2026-09-10 — never true] <!-- fact:sky -->\n"
        "- Deploys go through the CI gate. <!-- fact:ci-gate -->\n",
        encoding="utf-8",
    )
    prompt = _propose()
    sections = _sections(prompt)

    assert "## EXISTING_FACTS (2 facts from demo.md)" in prompt
    current = sections["EXISTING_FACTS"]
    assert "[supersedes-old] Endpoint B." in current
    assert "[ci-gate] Deploys go through the CI gate." in current
    assert "~~" not in current and "[old]" not in current and "[sky]" not in current

    assert "## RETIRED_FACTS (2 superseded or retracted — history, not current state)" in prompt
    retired = sections["RETIRED_FACTS"]
    # Same ids, same text, same markers: identity and history intact.
    assert "[old] ~~Endpoint A.~~ [superseded 2026-09-10]" in retired
    assert "[sky] ~~The sky is green.~~ [RETRACTED 2026-09-10 — never true]" in retired
    assert "Endpoint B." not in retired and "CI gate" not in retired

    # Section order: current facts, then history, then constraints, then notes.
    order = list(sections)
    assert order == ["EXISTING_FACTS", "RETIRED_FACTS", "RECENT_NOTES"]

    # Every id shown is in context for the guard — history included, so a
    # RETRACT may cite a superseded fact as its evidence.
    assembled = runner._assemble_prompt("demo", NOTE)
    assert assembled.context.fact_ids == frozenset({"old", "supersedes-old", "sky", "ci-gate"})


def test_no_retired_facts_means_no_retired_block(project_store):
    prompt = _propose()
    assert "RETIRED_FACTS" not in prompt
    assert "## EXISTING_FACTS (2 facts from demo.md)" in prompt


def test_nightly_pass_uses_the_same_split(project_store):
    (project_store / "specs" / "prompts" / "nightly-consolidation.md").write_text(PROMPT, encoding="utf-8")
    (project_store / "projects" / "demo.md").write_text(
        "---\nid: projects-demo\n---\n\n"
        "- ~~Endpoint A.~~ [superseded 2026-09-10] <!-- fact:old -->\n"
        "- Endpoint B. <!-- fact:supersedes-old -->\n",
        encoding="utf-8",
    )
    seen, fake = _capture()
    runner._consolidate_project("demo", NOTE, is_nightly=True, llm_fn=fake)
    sections = _sections(seen["user"])
    assert "[supersedes-old] Endpoint B." in sections["EXISTING_FACTS"]
    assert "[old] ~~Endpoint A.~~" in sections["RETIRED_FACTS"]


# ── retired by location: a note that was moved under archive/ ────────────────


def _archived(root, rel: str, body: str, **fm: Any) -> str:
    """A record sitting under ``archive/`` whose frontmatter still says active.

    That is what the weekly pass leaves behind: ``_archive_daily_notes`` moves
    the file and does not rewrite a byte of it, so nothing but its location
    says it was retired.
    """
    import yaml

    meta = {"entities": ["project/demo"], "status": "active", **fm}
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\n{yaml.safe_dump(meta, default_flow_style=False)}---\n\n{body}\n",
        encoding="utf-8",
    )
    return rel


def test_a_decision_moved_under_archive_is_not_governing(project_store):
    rel = _archived(
        project_store, "archive/decisions/endpoint-a.md", "Use retired endpoint A.",
        id="decisions-endpoint-a", name="endpoint-a",
    )
    _decision(project_store, "ci-gate", "Deploys go through the CI gate.", status="active")
    decisions = _sections(_propose())["ACTIVE_DECISIONS"]
    assert "Deploys go through the CI gate." in decisions
    assert "Use retired endpoint A." not in decisions
    # The decision scan does not descend into archive/, and the classifier the
    # scan consults agrees about the file it never reaches — so the two cannot
    # drift apart if the scan is ever widened.
    assert eligibility({"status": "active"}, path=rel).reason == "path:archive"


def test_an_archived_note_never_primes_the_digest(project_store):
    """Frontmatter says active and ``core: true``; the path says retired.

    The digest's own scan already skips ``archive/``, so this holds twice
    over — pinned here so the guarantee does not depend on which of the two
    enforces it.
    """
    _archived(
        project_store, "archive/2026/2026-03-01.md",
        "The activity gate cadence is hourly.",
        id="daily-2026-03-01", type="Decision", core=True, date="2026-03-01",
    )
    _decision(project_store, "ci-gate", "Deploys go through the CI gate.",
              status="active", date="2026-09-01")
    digest = build_context_digest(project="demo")
    selected = {
        row["file"]
        for key in ("core_memories", "recent_decisions", "open_action_items", "recent_snapshots")
        for row in digest[key]
    }
    assert "archive/2026/2026-03-01.md" not in selected
    assert "decisions/ci-gate.md" in selected


# ── end to end: real store, real executor, captured propose input ────────────


@pytest.fixture
def live_store(project_store, monkeypatch):
    """Real SQLite under tmp_path; embedder stubbed; the content scanner too."""
    monkeypatch.setenv("PALINODE_ALLOW_FRESH_DB", "1")
    monkeypatch.setattr(config, "db_path", str(project_store / ".palinode.db"))
    store.init_db()
    with (
        patch("palinode.core.embedder.embed", return_value=list(_VEC)),
        patch("palinode.core.store.scan_memory_content", return_value=(True, "OK")),
    ):
        yield project_store


def _index(root, rel: str) -> None:
    from palinode.indexer.index_file import index_file

    result = index_file(str(root / rel))
    assert result["error"] is None, result


def _recalled_files(top_k: int = 20) -> set[str]:
    """Memory-relative paths of what default recall returns (``exclude_status`` applied)."""
    import os

    hits = store.search(list(_VEC), top_k=top_k, threshold=0.0)
    return {
        os.path.relpath(r["file_path"], config.memory_dir)
        if os.path.isabs(r["file_path"]) else r["file_path"]
        for r in hits
    }


def test_archive_then_compact_then_recall(live_store):
    """archive → compact → recall: the archived decision governs nothing and
    is recalled nowhere, while its history stays inspectable."""
    from palinode.consolidation.archive import archive_memory

    _decision(live_store, "endpoint-a", "Use endpoint A.", status="active",
              created_at="2026-08-01T00:00:00Z")
    _decision(live_store, "ci-gate", "Deploys go through the CI gate.", status="active",
              created_at="2026-08-02T00:00:00Z")
    _index(live_store, "decisions/endpoint-a.md")
    _index(live_store, "decisions/ci-gate.md")
    assert "decisions/endpoint-a.md" in _recalled_files()

    # Before: both govern, both prime.
    before = _sections(_propose())["ACTIVE_DECISIONS"]
    assert "Use endpoint A." in before and "CI gate" in before
    assert "decisions/endpoint-a.md" in {
        r["file"] for r in build_context_digest(project="demo")["recent_decisions"]
    }

    result = archive_memory("decisions/endpoint-a.md", reason="endpoint A retired")
    assert result["status"] == "archived"

    # Compact: the archived decision is withheld; the unaffected one stays.
    after = _propose()
    decisions = _sections(after)["ACTIVE_DECISIONS"]
    assert "Use endpoint A." not in decisions
    assert "Deploys go through the CI gate." in decisions
    assert "[endpoint-b] Endpoint B." in _sections(after)["EXISTING_FACTS"]

    # Recall: the digest no longer presents it; search no longer returns it;
    # the file and its history sibling are still there to inspect.
    digest = build_context_digest(project="demo")
    assert [r["file"] for r in digest["recent_decisions"]] == ["decisions/ci-gate.md"]
    assert "decisions/endpoint-a.md" not in _recalled_files()
    assert (live_store / "decisions" / "endpoint-a.md").exists()
    history = (live_store / "decisions" / "endpoint-a-history.md").read_text(encoding="utf-8")
    assert "endpoint A retired" in history


def test_explicit_replacement_then_repeated_compaction_then_recall(live_store):
    """A → B replacement, at both the decision and the fact level, survives
    repeated compaction passes: B governs and primes, A is history in both
    places, unaffected claims are untouched, and the propose input is stable."""
    from palinode.consolidation.archive import archive_memory
    from palinode.consolidation.executor import apply_operations

    # Decision A, replaced explicitly by decision B.
    _decision(live_store, "endpoint-a", "Use endpoint A.", status="active",
              created_at="2026-08-01T00:00:00Z")
    _decision(live_store, "endpoint-b", "Use endpoint B.", status="active",
              created_at="2026-09-01T00:00:00Z")
    _index(live_store, "decisions/endpoint-a.md")
    _index(live_store, "decisions/endpoint-b.md")
    archive_memory("decisions/endpoint-a.md", superseded_by="decisions/endpoint-b")

    # Fact A in the status doc, replaced explicitly by the executor (as shipped).
    target = live_store / "projects" / "demo.md"
    target.write_text(
        "---\nid: projects-demo\n---\n\n"
        "- Endpoint A. <!-- fact:endpoint -->\n"
        "- Deploys go through the CI gate. <!-- fact:ci-gate -->\n",
        encoding="utf-8",
    )
    stats = apply_operations(str(target), [
        {"op": "SUPERSEDE", "id": "endpoint", "new_text": "Endpoint B.",
         "reason": "explicit replacement"},
    ])
    assert stats["superseded"] == 1

    # Three compaction passes, each proposing nothing; capture each input.
    prompts = [_propose() for _ in range(3)]
    assert prompts[0] == prompts[1] == prompts[2], "repeated compaction must not drift"
    sections = _sections(prompts[0])

    decisions = sections["ACTIVE_DECISIONS"]
    assert "Use endpoint B." in decisions and "Use endpoint A." not in decisions
    assert "[supersedes-endpoint] Endpoint B." in sections["EXISTING_FACTS"]
    assert "[ci-gate] Deploys go through the CI gate." in sections["EXISTING_FACTS"]
    assert "[endpoint] ~~Endpoint A.~~ [superseded " in sections["RETIRED_FACTS"]
    assert "Endpoint A." not in sections["EXISTING_FACTS"]

    # Recall: B primes, A does not, and A is still on disk with its trail.
    digest = build_context_digest(project="demo")
    assert [r["file"] for r in digest["recent_decisions"]] == ["decisions/endpoint-b.md"]
    assert "decisions/endpoint-a.md" not in _recalled_files()
    assert "decisions/endpoint-b.md" in _recalled_files()
    assert "Superseded by decisions/endpoint-b" in (
        live_store / "decisions" / "endpoint-a-history.md"
    ).read_text(encoding="utf-8")
    assert "Superseded" in (live_store / "projects" / "demo-history.md").read_text(encoding="utf-8")
    # The status doc itself still carries the tombstone: nothing was deleted.
    assert "~~Endpoint A.~~" in target.read_text(encoding="utf-8")


def test_an_archived_note_stays_findable_and_searches_as_retired(live_store):
    """History stays indexed; only the lifecycle verdict changes.

    The dogfood failure: an archived March daily note came back from search
    with ``currency: unmarked`` and was presented as a usable current
    assertion.
    """
    rel = _archived(
        live_store, "archive/2026/2026-03-01.md",
        "The activity gate cadence is hourly.",
        id="daily-2026-03-01", date="2026-03-01",
    )
    _index(live_store, rel)

    hits = store.search_hybrid(
        "activity gate cadence", list(_VEC), top_k=10, threshold=0.0, record_access=False,
    )
    hit = next((h for h in hits if h["file_path"].endswith(rel)), None)
    assert hit is not None, "an archived note must stay findable"
    assert hit["currency"] == "retired"
    assert hit["currency_reason"] == "path:archive"
