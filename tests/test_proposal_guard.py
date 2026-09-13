"""The propose-side guard: uncited RETRACT → PROPOSE_CONTRADICTS, footer ops rejected.

Two field observations on the production model drive these tests. A RETRACT
with the rationale "known to be incorrect" and nothing in context to show it
(the rig smoke), and a RETRACT whose rationale described one fact but whose
``id`` was the ``- [[notes]]`` wikilink under the auto-footer, which the
executor applied (the dogfood weekly).

Unit tests drive :func:`guard_operations` directly; the runner tests drive
the real ``run_consolidation`` / ``run_nightly`` path with a fake at the
propose seam on a real ``tmp_path`` store, and assert on what reached the
executor and what the run summary reports.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
import pytest

from palinode.consolidation import runner
from palinode.consolidation.proposal_guard import (
    GUARD_STATS,
    PromptContext,
    footer_fact_ids,
    guard_operations,
)
from palinode.core.config import config

FOOTER_MARKER = "<!-- palinode-auto-footer -->"


# ── unit: guard_operations ──────────────────────────────────────────────────


def _ctx(**overrides) -> PromptContext:
    base = dict(
        fact_ids=frozenset({"port-6340", "embed-bge", "notes-31435a", "notes-3a5443"}),
        footer_fact_ids=frozenset({"notes-3a5443"}),
        decision_refs=frozenset({"decisions/api-port-6341"}),
        note_refs=("daily/2026-09-03", "daily/2026-09-04"),
    )
    base.update(overrides)
    return PromptContext(**base)


def _retract(fact_id: str = "embed-bge", **fields) -> dict:
    return {"op": "RETRACT", "id": fact_id, "reason": "bge-m3 is known to be incorrect", **fields}


def test_uncited_retract_is_downgraded_to_propose_contradicts(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="palinode.consolidation.guard"):
        ops, stats = guard_operations([_retract()], _ctx(), target="projects/x.md")

    assert stats == {"retract_downgraded": 1, "footer_op_rejected": 0}
    assert ops == [{
        "op": "PROPOSE_CONTRADICTS",
        "id": "embed-bge",
        "contradicts": ["daily/2026-09-04"],
        "rationale": (
            "bge-m3 is known to be incorrect "
            "(downgraded from RETRACT: no in-context evidence cited)"
        ),
    }]
    assert "RETRACT downgraded to PROPOSE_CONTRADICTS" in caplog.text
    assert "embed-bge" in caplog.text


def test_retract_citing_an_existing_fact_id_passes_through_unchanged() -> None:
    op = _retract(falsified_by="port-6340")
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert ops == [op]
    assert stats == {"retract_downgraded": 0, "footer_op_rejected": 0}


def test_retract_citing_a_rendered_decision_ref_passes_through() -> None:
    op = _retract(falsified_by="decisions/api-port-6341")
    ops, _ = guard_operations([op], _ctx(), target="t")
    assert ops == [op]


@pytest.mark.parametrize("spelling", ["daily/2026-09-03.md", "./daily/2026-09-03", " daily/2026-09-03 "])
def test_citation_spellings_the_executor_accepts_resolve_too(spelling: str) -> None:
    op = _retract(falsified_by=spelling)
    ops, _ = guard_operations([op], _ctx(), target="t")
    assert ops == [op]


def test_retract_citing_a_ref_that_was_not_in_the_prompt_is_downgraded() -> None:
    """A citation is necessary evidence; one the model never saw is no evidence."""
    op = _retract(falsified_by="decisions/embedding-model")
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert stats["retract_downgraded"] == 1
    assert ops[0]["op"] == "PROPOSE_CONTRADICTS"


def test_a_wrong_falsified_by_is_not_rescued_by_the_rationale() -> None:
    """The field is the explicit claim; when present it is checked as given."""
    op = _retract(falsified_by="decisions/nope", reason="port-6340 shows it false")
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert stats["retract_downgraded"] == 1
    assert ops[0]["op"] == "PROPOSE_CONTRADICTS"


def test_rationale_naming_an_in_context_fact_id_is_the_fallback() -> None:
    op = _retract(reason="contradicted by port-6340, the observed value")
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert ops == [op]
    assert stats["retract_downgraded"] == 0


def test_rationale_naming_a_note_ref_is_the_fallback() -> None:
    op = _retract(reason="daily/2026-09-03.md records the actual model")
    ops, _ = guard_operations([op], _ctx(), target="t")
    assert ops == [op]


def test_the_retracted_fact_does_not_count_as_its_own_evidence() -> None:
    op = _retract(reason="embed-bge is wrong")
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert stats["retract_downgraded"] == 1
    assert ops[0]["op"] == "PROPOSE_CONTRADICTS"


def test_partial_token_matches_do_not_resolve() -> None:
    """`port-6340` must not be found inside `port-63400` or `xport-6340`."""
    op = _retract(reason="see port-63400 and xport-6340")
    _, stats = guard_operations([op], _ctx(), target="t")
    assert stats["retract_downgraded"] == 1


def test_uncited_retract_with_no_note_ref_is_dropped_not_applied(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="palinode.consolidation.guard"):
        ops, stats = guard_operations([_retract()], _ctx(note_refs=()), target="t")
    assert ops == []
    assert stats["retract_downgraded"] == 1
    assert "RETRACT dropped" in caplog.text


def test_downgraded_op_rationale_without_a_reason_is_just_the_note() -> None:
    op = {"op": "RETRACT", "id": "embed-bge"}
    ops, _ = guard_operations([op], _ctx(), target="t")
    assert ops[0]["rationale"] == "(downgraded from RETRACT: no in-context evidence cited)"


@pytest.mark.parametrize("kind", ["RETRACT", "ARCHIVE", "SUPERSEDE"])
def test_retiring_op_aimed_at_a_footer_fact_is_rejected(kind: str, caplog) -> None:
    op = {"op": kind, "id": "notes-3a5443", "new_text": "x",
          "reason": "the ingest line is conflated with the host bullet"}
    with caplog.at_level(logging.WARNING, logger="palinode.consolidation.guard"):
        ops, stats = guard_operations([op], _ctx(), target="projects/notes.md")
    assert ops == []
    assert stats == {"retract_downgraded": 0, "footer_op_rejected": 1}
    assert f"{kind} rejected" in caplog.text
    assert "auto-footer" in caplog.text


def test_footer_rejection_precedes_the_citation_check() -> None:
    """A cited RETRACT at a footer id is still a footer op."""
    op = _retract("notes-3a5443", falsified_by="notes-31435a")
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert ops == []
    assert stats["footer_op_rejected"] == 1


def test_update_on_a_footer_fact_is_not_the_guards_business() -> None:
    op = {"op": "UPDATE", "id": "notes-3a5443", "new_text": "- [[notes]]"}
    ops, stats = guard_operations([op], _ctx(), target="t")
    assert ops == [op]
    assert stats == {"retract_downgraded": 0, "footer_op_rejected": 0}


def test_other_ops_and_order_are_preserved() -> None:
    ops_in = [
        {"op": "UPDATE", "id": "port-6340", "new_text": "y"},
        _retract(),
        {"op": "ARCHIVE", "id": "port-6340"},
        "not a dict",
    ]
    ops, _ = guard_operations(ops_in, _ctx(), target="t")
    assert [o if isinstance(o, str) else o["op"] for o in ops] == [
        "UPDATE", "PROPOSE_CONTRADICTS", "ARCHIVE", "not a dict",
    ]


def test_stats_always_carry_every_guard_key() -> None:
    _, stats = guard_operations([], _ctx(), target="t")
    assert set(stats) == set(GUARD_STATS) == {"retract_downgraded", "footer_op_rejected"}


# ── unit: footer_fact_ids ───────────────────────────────────────────────────


def test_footer_fact_ids_are_the_bullets_after_the_marker() -> None:
    body = (
        "# Notes\n\n- ingests captures from voice notes <!-- fact:notes-31435a -->\n\n"
        f"## See also\n{FOOTER_MARKER}\n- [[notes]] <!-- fact:notes-3a5443 -->\n"
        "- [[host-a]] <!-- fact:notes-9 -->\n"
    )
    assert footer_fact_ids(body) == {"notes-3a5443", "notes-9"}


def test_no_marker_means_no_footer_facts() -> None:
    body = "## See also\n- [[notes]] <!-- fact:notes-3a5443 -->\n"
    assert footer_fact_ids(body) == frozenset()


# ── runner: the guard sits between the model and the executor ───────────────


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@pytest.fixture
def store(tmp_path, monkeypatch) -> Path:
    """The rig's second seed, roughly: a decision, a later-dated observed fact
    that disagrees with it, a note that says nothing about which is right, and
    an auto-footer with a tagged wikilink."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.git, "auto_commit", False)
    for sub in ("projects", "decisions", "daily", "specs/prompts"):
        (tmp_path / sub).mkdir(parents=True)
    (tmp_path / "specs" / "prompts" / "compaction.md").write_text(
        "Return consolidation operations as a JSON array.\n", encoding="utf-8"
    )
    (tmp_path / "specs" / "prompts" / "nightly-consolidation.md").write_text(
        "Return consolidation operations as a JSON array.\n", encoding="utf-8"
    )
    (tmp_path / "projects" / "rig.md").write_text(
        "---\nid: projects-rig\ncategory: project\n---\n\n# Rig\n\n"
        "- [2026-09-03] API listens on 6340 (observed). <!-- fact:port-6340 -->\n"
        "- [2026-09-04] Embedding model is bge-m3 (observed). <!-- fact:embed-bge -->\n\n"
        f"## See also\n{FOOTER_MARKER}\n- [[rig]] <!-- fact:rig-footer -->\n",
        encoding="utf-8",
    )
    (tmp_path / "decisions" / "api-port-6341.md").write_text(
        "---\nid: decisions-api-port-6341\nname: api-port-6341\n"
        "entities:\n  - project/rig\n---\n\n[2026-08-15] API listens on 6341.\n",
        encoding="utf-8",
    )
    (tmp_path / "daily" / f"{_today()}-rig.md").write_text(
        "---\nid: rig-note\ncategory: daily\n---\n\n"
        "Checked project/rig; nothing decided about the port or the model.\n",
        encoding="utf-8",
    )
    return tmp_path


def _llm(ops: list[dict]):
    seen: dict[str, str] = {}

    def _fn(system_prompt: str, user_prompt: str) -> tuple[str, str]:
        seen["user"] = user_prompt
        return json.dumps(ops), "fake-model"

    _fn.seen = seen  # type: ignore[attr-defined]
    return _fn


UNCITED = {"op": "RETRACT", "id": "embed-bge", "reason": "bge-m3 is known to be incorrect"}


def test_uncited_retract_reaches_the_executor_as_a_contradicts_link(store, caplog) -> None:
    target = store / "projects" / "rig.md"
    llm = _llm([UNCITED])

    with caplog.at_level(logging.WARNING):
        result = runner.run_consolidation(llm_fn=llm)

    assert result["retract_downgraded"] == 1
    assert result["footer_op_rejected"] == 0
    assert result.get("retracted", 0) == 0
    assert result["contradicts_proposed"] == 1
    assert "RETRACT downgraded to PROPOSE_CONTRADICTS" in caplog.text

    post = frontmatter.load(target)
    assert post.metadata["contradicts"] == [f"daily/{_today()}-rig"]
    assert "~~" not in post.content, "the RETRACT was applied"
    assert "<!-- fact:embed-bge -->" in post.content
    assert "downgraded from RETRACT" in post.content, "the audit log names the downgrade"


def test_the_prompt_shows_each_note_ref_so_the_model_can_cite_it(store) -> None:
    llm = _llm([])
    runner.run_consolidation(llm_fn=llm)
    assert f"### {_today()} (ref: daily/{_today()}-rig)" in llm.seen["user"]
    assert "(ref: decisions/api-port-6341)" in llm.seen["user"]


def test_retract_citing_a_fact_in_existing_facts_is_applied(store) -> None:
    target = store / "projects" / "rig.md"
    cited = dict(UNCITED, falsified_by="port-6340")

    result = runner.run_consolidation(llm_fn=_llm([cited]))

    assert result["retract_downgraded"] == 0
    assert result["retracted"] == 1
    assert "~~" in frontmatter.load(target).content


def test_retract_citing_the_rendered_decision_is_applied(store) -> None:
    cited = dict(UNCITED, falsified_by="decisions/api-port-6341")
    result = runner.run_consolidation(llm_fn=_llm([cited]))
    assert result["retract_downgraded"] == 0
    assert result["retracted"] == 1


def test_retract_citing_a_ref_not_in_the_prompt_is_downgraded(store) -> None:
    cited = dict(UNCITED, falsified_by="decisions/embedding-model")
    result = runner.run_consolidation(llm_fn=_llm([cited]))
    assert result["retract_downgraded"] == 1
    assert result.get("retracted", 0) == 0
    assert result["contradicts_proposed"] == 1


@pytest.mark.parametrize("kind", ["RETRACT", "ARCHIVE"])
def test_retiring_op_aimed_at_the_footer_is_rejected(store, kind, caplog) -> None:
    target = store / "projects" / "rig.md"
    before = target.read_text(encoding="utf-8")
    op = {"op": kind, "id": "rig-footer", "falsified_by": "port-6340",
          "reason": "the embedding line is wrong"}

    with caplog.at_level(logging.WARNING):
        result = runner.run_consolidation(llm_fn=_llm([op]))

    assert result["footer_op_rejected"] == 1
    assert result["retract_downgraded"] == 0
    assert "auto-footer" in caplog.text
    assert target.read_text(encoding="utf-8") == before
    assert result["projects_all_ops_filtered"] == ["rig"]


def test_dry_run_previews_the_downgrade_and_reports_the_count(store) -> None:
    target = store / "projects" / "rig.md"
    before = target.read_text(encoding="utf-8")

    result = runner.run_consolidation(dry_run=True, llm_fn=_llm([UNCITED]))

    assert result["dry_run"] is True
    assert result["retract_downgraded"] == 1
    assert result["footer_op_rejected"] == 0
    assert [c["type"] for c in result["proposed_changes"]] == ["PROPOSE_CONTRADICTS"]
    assert target.read_text(encoding="utf-8") == before


def test_nightly_runs_the_guard_before_its_allowed_ops_filter(store) -> None:
    """Nightly's allowed_ops has no RETRACT; the guard turns an uncited one
    into the PROPOSE_CONTRADICTS the filter admits."""
    target = store / "projects" / "rig.md"
    assert "RETRACT" not in config.consolidation.nightly.allowed_ops

    result = runner.run_nightly(llm_fn=_llm([UNCITED]))

    assert result["retract_downgraded"] == 1
    assert result["footer_op_rejected"] == 0
    assert result["contradicts_proposed"] == 1
    assert result.get("retracted", 0) == 0
    assert frontmatter.load(target).metadata["contradicts"] == [f"daily/{_today()}-rig"]


def test_nightly_rejects_footer_ops_too(store) -> None:
    op = {"op": "SUPERSEDE", "id": "rig-footer", "new_text": "x", "reason": "y"}
    result = runner.run_nightly(llm_fn=_llm([op]))
    assert result["footer_op_rejected"] == 1
    assert result["projects_all_ops_filtered"] == ["rig"]


def test_quiet_pass_reports_zero_for_both_counts(store) -> None:
    # Nightly first: the weekly pass retires the note, and a nightly with no
    # notes returns its early "no_new_notes" shape instead of a summary.
    for result in (
        runner.run_nightly(dry_run=True, llm_fn=_llm([])),
        runner.run_consolidation(llm_fn=_llm([])),
    ):
        assert result["retract_downgraded"] == 0
        assert result["footer_op_rejected"] == 0
