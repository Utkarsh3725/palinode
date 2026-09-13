"""`ARCHIVE_BEFORE`: one operation retires a dated range of status log lines.

The weekly pass on a six-month status backlog is honest and still cannot
finish: 449 facts in, ~430 of them stale dated session lines, an ARCHIVE with a
rationale for each, ~60 KB of output. No cap carries it, so the proposal is
truncated and nothing is retired at all. A range op makes the proposal's size
follow the number of *reasons* rather than the number of facts.

It is a retiring op and is guarded like one: refused on a living
(`update_policy: replace`) document, refused on a `superseded-only` document
(a date range is an age argument, and there is no field that could make it
anything else), refused by the propose-side guard when aimed at the
auto-footer, and absent from the nightly pass's `allowed_ops`.

Real files under ``tmp_path``, the real executor and the real runner.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from palinode.consolidation import runner
from palinode.consolidation.executor import apply_operations
from palinode.consolidation.proposal_guard import PromptContext, guard_operations
from palinode.core.config import config

LINES = (
    "- [2026-01-05] January session. <!-- fact:jan -->\n"
    "- [2026-03-09] March session. <!-- fact:mar -->\n"
    "- [2026-06-01] June session. <!-- fact:jun -->\n"
    "- An undated curated fact. <!-- fact:curated -->\n"
)


@pytest.fixture(autouse=True)
def _memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    return tmp_path


def _target(tmp_path: Path, *, frontmatter: str = "", body: str = LINES,
            name: str = "proj-status.md") -> Path:
    path = tmp_path / "projects" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: projects-proj-status\ncategory: project\n{frontmatter}---\n\n"
        f"# Proj Status\n\n{body}",
        encoding="utf-8",
    )
    return path


def _range_op(before: str, **extra) -> dict:
    return {"op": "ARCHIVE_BEFORE", "before": before,
            "reason": "stale session lines", **extra}


# ---------------------------------------------------------------------------
# What it archives
# ---------------------------------------------------------------------------

def test_archives_exactly_the_lines_strictly_before_the_date(tmp_path) -> None:
    target = _target(tmp_path)

    stats = apply_operations(str(target), [_range_op("2026-06-01")])

    text = target.read_text(encoding="utf-8")
    assert stats["archived"] == 2
    assert stats["archived_by_range"] == 2
    assert "<!-- fact:jan -->" not in text and "<!-- fact:mar -->" not in text
    # The boundary date itself is kept, and an undated fact is not a log line.
    assert "<!-- fact:jun -->" in text
    assert "<!-- fact:curated -->" in text


def test_every_archived_line_is_kept_verbatim_in_history(tmp_path) -> None:
    target = _target(tmp_path)

    apply_operations(str(target), [_range_op("2026-06-01")])

    history = (tmp_path / "projects" / "proj-history.md").read_text(encoding="utf-8")
    assert "January session." in history
    assert "March session." in history
    assert "June session." not in history
    assert "stale session lines" in history


@pytest.mark.parametrize("before", ["", "June 2026", "2026-13-45", "2026", None])
def test_a_malformed_date_is_rejected_counted_and_logged(
    tmp_path, caplog, before
) -> None:
    target = _target(tmp_path)
    before_bytes = target.read_bytes()

    with caplog.at_level(logging.WARNING):
        stats = apply_operations(str(target), [{"op": "ARCHIVE_BEFORE", "before": before}])

    assert stats["archived"] == 0
    assert stats["unmatched"] == 1
    assert target.read_bytes() == before_bytes
    assert "ARCHIVE_BEFORE dropped" in caplog.text


def test_a_range_matching_nothing_is_unmatched_not_a_silent_success(
    tmp_path, caplog
) -> None:
    target = _target(tmp_path)

    with caplog.at_level(logging.WARNING):
        stats = apply_operations(str(target), [_range_op("2020-01-01")])

    assert stats["archived"] == 0
    assert stats["archived_by_range"] == 0
    assert stats["unmatched"] == 1
    assert "ARCHIVE_BEFORE unmatched" in caplog.text


def test_the_range_sees_what_earlier_operations_in_the_call_did(tmp_path) -> None:
    """Ops apply sequentially; a line archived first is not archived twice."""
    target = _target(tmp_path)

    stats = apply_operations(str(target), [
        {"op": "ARCHIVE", "id": "jan", "rationale": "named individually"},
        _range_op("2026-06-01"),
    ])

    assert stats["archived"] == 2
    assert stats["archived_by_range"] == 1


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------

def test_rejected_on_a_superseded_only_document(tmp_path, caplog) -> None:
    target = _target(tmp_path, frontmatter="retirement_policy: superseded-only\n")
    before = target.read_bytes()

    with caplog.at_level(logging.WARNING):
        stats = apply_operations(str(target), [_range_op("2026-06-01")])

    assert stats["protected_rejected"] == 1
    assert stats["archived"] == 0
    assert target.read_bytes() == before
    assert "a date range is an age argument" in caplog.text


def test_a_superseded_by_field_does_not_buy_a_range_past_the_guard(tmp_path) -> None:
    """Unlike a single ARCHIVE: a range names no fact, so it can name no successor."""
    target = _target(tmp_path, frontmatter="retirement_policy: superseded-only\n")

    stats = apply_operations(
        str(target), [_range_op("2026-06-01", superseded_by="projects/proj")]
    )

    assert stats["protected_rejected"] == 1
    assert stats["archived"] == 0


def test_rejected_on_a_living_replace_document(tmp_path, caplog) -> None:
    target = _target(tmp_path, frontmatter="update_policy: replace\n")

    with caplog.at_level(logging.WARNING):
        stats = apply_operations(str(target), [_range_op("2026-06-01")])

    assert stats["protected_rejected"] == 1
    assert stats["archived"] == 0
    assert "update_policy=replace" in caplog.text


def test_propose_side_guard_rejects_a_range_aimed_at_the_footer(caplog) -> None:
    """A range carries a date, not an id — one that carries a footer id is
    mis-aimed in exactly the way the footer rule exists for."""
    context = PromptContext(
        fact_ids=frozenset({"footer-1"}),
        footer_fact_ids=frozenset({"footer-1"}),
        decision_refs=frozenset(),
        note_refs=("daily/2026-09-12",),
    )

    with caplog.at_level(logging.WARNING):
        kept, stats = guard_operations(
            [_range_op("2026-06-01", id="footer-1")], context, target="proj-status.md",
        )

    assert kept == []
    assert stats["footer_op_rejected"] == 1


# ---------------------------------------------------------------------------
# Through the passes
# ---------------------------------------------------------------------------

@pytest.fixture
def pass_store(tmp_path, monkeypatch):
    for sub in ("projects", "daily", "specs/prompts"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "specs" / "prompts" / "compaction.md").write_text(
        "Return consolidation operations as a JSON array.\n", encoding="utf-8"
    )
    (tmp_path / "specs" / "prompts" / "nightly-consolidation.md").write_text(
        "Return nightly operations as a JSON array.\n", encoding="utf-8"
    )
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    (tmp_path / "daily" / f"{today}.md").write_text(
        f"---\nid: daily-{today}\ncategory: daily\nentities:\n- project/proj\n---\n\n"
        "Worked on project/proj today.\n",
        encoding="utf-8",
    )
    # The deterministic sweep is a separate mechanism with its own tests; keep
    # it out of the way so these measure what the *model* proposed.
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 0)
    return tmp_path


def _proposing(op: dict):
    payload = json.dumps([op])
    return lambda _s, _u: (payload, "fake-model")


def test_the_weekly_pass_applies_a_proposed_range(pass_store) -> None:
    target = _target(pass_store)

    result = runner.run_consolidation(llm_fn=_proposing(_range_op("2026-06-01")))

    assert result["archived"] == 2
    assert result["archived_by_range"] == 2
    text = target.read_text(encoding="utf-8")
    assert "<!-- fact:jan -->" not in text
    assert "- [ARCHIVE_BEFORE] before 2026-06-01: stale session lines" in text


def test_a_range_that_retired_nothing_writes_no_log_line(pass_store) -> None:
    """Like a MERGE that merged nothing: the log records what happened."""
    target = _target(pass_store)

    result = runner.run_consolidation(llm_fn=_proposing(_range_op("2020-01-01")))

    assert result["archived_by_range"] == 0
    assert "ARCHIVE_BEFORE" not in target.read_text(encoding="utf-8")


def test_the_dry_run_preview_lists_the_ids_the_range_would_retire(pass_store) -> None:
    target = _target(pass_store)
    before = target.read_bytes()

    result = runner.run_consolidation(
        dry_run=True, llm_fn=_proposing(_range_op("2026-06-01"))
    )

    (change,) = result["proposed_changes"]
    assert change["type"] == "ARCHIVE_BEFORE"
    assert change["before"] == "2026-06-01"
    assert change["ids"] == ["jan", "mar"]
    assert change["count"] == 2
    assert target.read_bytes() == before


def test_the_nightly_pass_filters_the_range_op_out(pass_store) -> None:
    """Visible as a filtered group in the run summary, not as a quiet week."""
    target = _target(pass_store)

    result = runner.run_nightly(llm_fn=_proposing(_range_op("2026-06-01")))

    assert result["projects_all_ops_filtered"] == ["proj"]
    assert result.get("archived", 0) == 0
    assert "<!-- fact:jan -->" in target.read_text(encoding="utf-8")


def test_an_operator_can_take_the_range_op_away_from_the_weekly_pass(
    pass_store, monkeypatch
) -> None:
    monkeypatch.setattr(
        config.consolidation, "allowed_ops",
        [op for op in config.consolidation.allowed_ops if op != "ARCHIVE_BEFORE"],
    )
    target = _target(pass_store)

    result = runner.run_consolidation(llm_fn=_proposing(_range_op("2026-06-01")))

    assert result["projects_all_ops_filtered"] == ["proj"]
    assert "<!-- fact:jan -->" in target.read_text(encoding="utf-8")


def test_a_relative_range_retires_only_the_backlog(pass_store) -> None:
    """The shape a model actually proposes: a date it computed from today."""
    cutoff = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
    old = (datetime.now(UTC) - timedelta(days=120)).strftime("%Y-%m-%d")
    recent = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
    target = _target(pass_store, body=(
        f"- [{old}] Old session. <!-- fact:old -->\n"
        f"- [{recent}] Recent session. <!-- fact:recent -->\n"
    ))

    result = runner.run_consolidation(llm_fn=_proposing(_range_op(cutoff)))

    text = target.read_text(encoding="utf-8")
    assert result["archived_by_range"] == 1
    assert "<!-- fact:old -->" not in text
    assert "<!-- fact:recent -->" in text
