"""Deterministic age retirement of stale status log lines, before the model runs.

A ``projects/<slug>-status.md`` fed by ``POST /session-end`` gains one
``- [YYYY-MM-DD] …`` line per session. On the dogfood store that reached 449
tagged facts, nearly all of them stale session lines, and the weekly pass could
not digest it: doing what rule 4 asks — archive aggressively for status — meant
~430 ARCHIVE ops with a rationale each, ~60 KB, past every workable token cap.
The pass failed, so *nothing* was retired, and the next week it failed again.

Retiring a line because it is older than a configured window is arithmetic, not
judgement, so the runner does it before it builds the prompt: real ARCHIVE ops
through the real executor, the verbatim line kept in the ``-history.md``
sibling, one commit and one Consolidation Log line of its own.

Real files under ``tmp_path``, the real runner→executor path, a fake only at
the propose seam (``llm_fn``) — no mocked store, no mocked executor.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from palinode.consolidation import runner
from palinode.consolidation.log_lines import dated_log_lines, older_than
from palinode.core.config import config
from palinode.core.parser import split_frontmatter


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")


def _status_doc(body: str, *, frontmatter: str = "") -> str:
    return (
        "---\n"
        "id: projects-proj-status\n"
        "category: project\n"
        "entities:\n"
        "- project/proj\n"
        f"{frontmatter}"
        "---\n\n"
        "# Proj Status\n\n"
        "## Current Work\n\n"
        f"{body}\n"
    )


def _log_line(days: int, index: int) -> str:
    return (
        f"- [{_days_ago(days)}] Session {index}: shipped something. "
        f"(1 decision → daily/{_days_ago(days)}.md) <!-- fact:s{index:04d} -->"
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A memory dir the weekly pass can run against, with today's daily note."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    for sub in ("projects", "daily", "specs/prompts"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "specs" / "prompts" / "compaction.md").write_text(
        "Return consolidation operations as a JSON array.\n", encoding="utf-8"
    )
    (tmp_path / "daily" / f"{_today()}.md").write_text(
        f"---\nid: daily-{_today()}\ncategory: daily\nentities:\n- project/proj\n---\n\n"
        "Worked on project/proj today.\n",
        encoding="utf-8",
    )
    return tmp_path


def _write_target(store: Path, body: str, *, frontmatter: str = "",
                  name: str = "proj-status.md") -> Path:
    target = store / "projects" / name
    target.write_text(_status_doc(body, frontmatter=frontmatter), encoding="utf-8")
    return target


def _no_ops(_system: str, _user: str) -> tuple[str, str]:
    """A model that proposes nothing — isolates the deterministic sweep."""
    return "[]", "fake-model"


def _recording_llm(seen: list[str]):
    def _fn(_system: str, user_prompt: str) -> tuple[str, str]:
        seen.append(user_prompt)
        return "[]", "fake-model"
    return _fn


def _ids(text: str) -> set[str]:
    _, body = split_frontmatter(text)
    return {line.fact_id for line in dated_log_lines(body)}


# ---------------------------------------------------------------------------
# The recognizer
# ---------------------------------------------------------------------------

def test_recognizer_takes_dated_tagged_bullets_and_nothing_else() -> None:
    body = (
        "## Current Work\n\n"
        "- [2026-03-04] A session line. <!-- fact:a -->\n"
        "- An undated curated fact. <!-- fact:b -->\n"
        "- Shipped on [2026-03-04], which is not a log line. <!-- fact:c -->\n"
        "- [2026-03-04] A line with no fact marker\n"
        "- [2026-13-45] Not a date at all. <!-- fact:d -->\n"
        "- ~~[2026-03-04] Already retired.~~ [superseded 2026-09-01] <!-- fact:e -->\n"
        "\n```markdown\n"
        "- [2026-03-04] A documentation example. <!-- fact:f -->\n"
        "```\n"
    )
    assert [line.fact_id for line in dated_log_lines(body)] == ["a"]


def test_dated_lines_inside_the_consolidation_log_are_recognized() -> None:
    """Where session-end actually puts them — recognition is by shape, not section.

    `POST /session-end` appends to the *end of the file*, and
    `## Consolidation Log` is the last section a status document has, so every
    dated session line lands inside it. A recognizer that skipped that section
    would retire nothing at all on the document this mechanism exists for.
    """
    body = (
        "## Current Work\n\n"
        "- [2026-03-04] A session line above the log. <!-- fact:above -->\n\n"
        "## Consolidation Log\n\n"
        "- _[log elided] 12 operation line(s) across 3 date block(s) — "
        "2026-01-01 → 2026-02-02. Full detail in git history._\n\n"
        "- [2026-03-05] A session line inside the log. <!-- fact:inside -->\n"
        "### 2026-03-06\n"
        "- [UPDATE] supersedes-thing-dbba20: rewrote it\n"
        "- [2026-03-06] A session line inside a date block. <!-- fact:in-block -->\n"
    )
    assert [line.fact_id for line in dated_log_lines(body)] == [
        "above", "inside", "in-block",
    ]


def test_operation_records_and_the_elision_bullet_are_never_log_lines() -> None:
    """The audit trail *of* a retirement; retiring it would be the log eating itself."""
    body = (
        "- _[log elided] 291 operation line(s) across 48 date block(s) — "
        "2026-03-31 → 2026-06-28. Full detail in git history._\n"
        "- [UPDATE] thing-1: rewrote it\n"
        "- [SUPERSEDE] thing-2: replaced\n"
        "- [ARCHIVE] thing-3: stale\n"
        "- [RETRACT] thing-4: wrong\n"
        "- [MERGE] thing-5, thing-6: duplicated\n"
        "- [ARCHIVE_BEFORE] before 2026-06-01: age-retention\n"
        "- [KEEP] thing-7: still true\n"
    )
    assert dated_log_lines(body) == []


def test_recognizer_stops_at_the_auto_footer() -> None:
    from palinode.core.embedding_preprocess import AUTO_FOOTER_MARKER

    body = (
        "- [2026-03-04] A session line. <!-- fact:a -->\n"
        f"{AUTO_FOOTER_MARKER}\n"
        "## See also\n\n"
        "- [2026-03-04] [[some/link]] <!-- fact:footer-1 -->\n"
    )
    assert [line.fact_id for line in dated_log_lines(body)] == ["a"]


def test_older_than_is_strict() -> None:
    body = (
        "- [2026-03-04] on the boundary <!-- fact:a -->\n"
        "- [2026-03-03] before it <!-- fact:b -->\n"
    )
    cutoff = datetime(2026, 3, 4, tzinfo=UTC)
    assert [line.fact_id for line in older_than(body, cutoff)] == ["b"]


# ---------------------------------------------------------------------------
# The runner-side sweep
# ---------------------------------------------------------------------------

def test_only_lines_older_than_the_window_are_retired(store, monkeypatch) -> None:
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    target = _write_target(store, "\n".join([
        _log_line(200, 1), _log_line(120, 2), _log_line(91, 3),
        _log_line(89, 4), _log_line(1, 5),
    ]))

    result = runner.run_consolidation(llm_fn=_no_ops)

    assert result["age_retired"] == 3
    assert _ids(target.read_text(encoding="utf-8")) == {"s0004", "s0005"}


def test_recent_undated_footer_and_fenced_content_survive(store, monkeypatch) -> None:
    from palinode.core.embedding_preprocess import AUTO_FOOTER_MARKER

    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    body = "\n".join([
        _log_line(200, 1),
        "- An undated architectural fact. <!-- fact:undated -->",
        "- [2026-01-01] fenced, in a code sample:",
        "",
        "```markdown",
        "- [2020-01-01] A documentation example. <!-- fact:fenced -->",
        "```",
        "",
        "## Consolidation Log",
        "",
        "- _[log elided] 4 operation line(s) across 1 date block(s) — "
        "2026-01-01 → 2026-01-02. Full detail in git history._",
        "",
        "## See also",
        "",
        AUTO_FOOTER_MARKER,
        "- [2020-01-01] [[projects/other]] <!-- fact:footer -->",
    ])
    target = _write_target(store, body)

    result = runner.run_consolidation(llm_fn=_no_ops)

    text = target.read_text(encoding="utf-8")
    assert result["age_retired"] == 1
    assert "<!-- fact:s0001 -->" not in text
    for survivor in ("undated", "fenced", "footer"):
        assert f"<!-- fact:{survivor} -->" in text
    assert "_[log elided] 4 operation line(s)" in text


@pytest.mark.parametrize(
    "name,frontmatter",
    [
        # A project's profile document is identity, inferred from its path.
        ("proj.md", ""),
        # A status document that declares itself protected wins over the path.
        ("proj-status.md", "retirement_policy: superseded-only\n"),
    ],
)
def test_a_superseded_only_target_is_never_swept(
    store, monkeypatch, caplog, name, frontmatter
) -> None:
    """And the executor's ADR-020 guard never fires, because nothing is proposed.

    The sweep asks the same classifier the guard reads, so a protected document
    produces no ops at all — a rejection count would mean the two had drifted.
    """
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    target = _write_target(
        store, _log_line(200, 1), frontmatter=frontmatter, name=name,
    )

    with caplog.at_level(logging.WARNING):
        result = runner.run_consolidation(llm_fn=_no_ops)

    assert result["age_retired"] == 0
    assert "<!-- fact:s0001 -->" in target.read_text(encoding="utf-8")
    assert "retirement_policy=superseded-only" not in caplog.text
    assert result.get("protected_rejected", 0) == 0


def test_zero_disables_the_sweep(store, monkeypatch) -> None:
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 0)
    target = _write_target(store, _log_line(500, 1))

    result = runner.run_consolidation(llm_fn=_no_ops)

    assert result["age_retired"] == 0
    assert "<!-- fact:s0001 -->" in target.read_text(encoding="utf-8")


def test_dry_run_counts_without_applying(store, monkeypatch) -> None:
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    target = _write_target(store, "\n".join([_log_line(200, 1), _log_line(1, 2)]))
    before = target.read_bytes()

    result = runner.run_consolidation(dry_run=True, llm_fn=_no_ops)

    assert result["age_retired"] == 1
    assert target.read_bytes() == before
    assert not (store / "projects" / "proj-history.md").exists()


def test_retired_lines_are_kept_verbatim_in_the_history_sibling(
    store, monkeypatch
) -> None:
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    _write_target(store, _log_line(200, 1))

    runner.run_consolidation(llm_fn=_no_ops)

    history = (store / "projects" / "proj-history.md").read_text(encoding="utf-8")
    assert "Session 1: shipped something." in history
    assert "status: archived" in history
    assert "age-retention" in history, "the history entry names the actor"
    assert "older than 90 days" in history


def test_the_sweep_commits_on_its_own_and_logs_one_range_line(
    store, monkeypatch
) -> None:
    """Its own commit, and a single Consolidation Log line naming the range.

    One line per retired fact would need eliding to stay readable, which is the
    log reporting the sweep as noise.
    """
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    monkeypatch.setattr(config.git, "auto_commit", True)
    target = _write_target(store, "\n".join(_log_line(200 - i, i) for i in range(1, 6)))
    for args in (
        ["init"], ["config", "user.email", "t@example.com"],
        ["config", "user.name", "T"], ["add", "."], ["commit", "-m", "seed"],
    ):
        subprocess.run(["git", *args], cwd=store, check=True, capture_output=True)

    result = runner.run_consolidation(llm_fn=_no_ops)

    assert result["age_retired"] == 5
    subjects = subprocess.run(
        ["git", "log", "--format=%s"], cwd=store,
        check=True, capture_output=True, text=True,
    ).stdout
    assert "age-retention: 5 status log line(s) older than 90d" in subjects

    log_lines = [
        line for line in target.read_text(encoding="utf-8").splitlines()
        if line.startswith("- [ARCHIVE")
    ]
    assert len(log_lines) == 1
    assert log_lines[0].startswith("- [ARCHIVE_BEFORE] before ")
    assert "retired 5 status log line(s) older than 90 days" in log_lines[0]


#: A verbatim-shaped excerpt of `projects/palinode-status.md` on the dogfood
#: host (ids anonymised, dates made relative): the elision bullet, dated
#: session lines appended straight into the log section, and a `### <date>`
#: block of operation records interleaved among them. This is the layout the
#: first version of the recognizer got wrong — it excluded the whole
#: `## Consolidation Log` section, which is where every one of the real
#: document's 453 dated fact lines lives.
_REAL_LAYOUT = [
    "## Consolidation Log",
    "",
    "- _[log elided] 291 operation line(s) across 48 date block(s) — "
    "2026-03-31 → 2026-06-28. Full detail in git history._",
    "",
    "- [{d200}] Test session: launched Palinode v0.5.0, indexed the store. "
    "<!-- fact:proj-status-84109d -->",
    "- [{d160}] CLI parity: added read and session-end commands "
    "<!-- fact:proj-status-5f93e9 -->",
    "### {d120}",
    "- [UPDATE] proj-status-4d5e6f: rewrote the endpoint fact",
    "- [ARCHIVE] proj-status-84109d: superseded by the v0.8 layout",
    "- [{d120}] Shipped the v0.8.16 systemd reconciliation. "
    "<!-- fact:proj-status-1a2b3c -->",
    "- [{d10}] Audited the public repository for scrub drift. "
    "<!-- fact:proj-status-4d5e6f -->",
    "- [{d2}] Landed the retrieval receipt. <!-- fact:proj-status-7g8h9i -->",
]


def test_the_real_document_layout_retires_exactly_its_stale_session_lines(
    store, monkeypatch
) -> None:
    """Regression for the layout the recognizer must actually handle.

    Everything below `## Consolidation Log` and nothing else: the two lines
    outside the 90-day window and the one inside a `### <date>` block go; the
    recent lines, the elision bullet, the operation records and the block
    heading all stay.
    """
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    dates = {"d200": _days_ago(200), "d160": _days_ago(160), "d120": _days_ago(120),
             "d10": _days_ago(10), "d2": _days_ago(2)}
    target = _write_target(
        store, "\n".join(line.format(**dates) for line in _REAL_LAYOUT),
    )

    result = runner.run_consolidation(llm_fn=_no_ops)

    text = target.read_text(encoding="utf-8")
    assert result["age_retired"] == 3
    for retired in ("84109d", "5f93e9", "1a2b3c"):
        assert f"<!-- fact:proj-status-{retired} -->" not in text
    for kept in ("4d5e6f", "7g8h9i"):
        assert f"<!-- fact:proj-status-{kept} -->" in text
    # The log's own furniture is not a log line and survives intact.
    assert "_[log elided] 291 operation line(s)" in text
    assert "- [UPDATE] proj-status-4d5e6f: rewrote the endpoint fact" in text
    assert "- [ARCHIVE] proj-status-84109d: superseded by the v0.8 layout" in text
    assert f"### {dates['d120']}" in text, "an emptied-around date block keeps its heading"

    history = (store / "projects" / "proj-history.md").read_text(encoding="utf-8")
    assert "Test session: launched Palinode v0.5.0" in history
    assert "Shipped the v0.8.16 systemd reconciliation." in history
    assert "Audited the public repository" not in history


def test_the_swept_real_layout_still_satisfies_repair_status(
    store, monkeypatch
) -> None:
    """Removing session lines must not leave a log `palinode repair-status` rewrites.

    They parse as *raw* items outside any `### <date>` block, so the bounding
    and elision logic never counted them and cannot miss them.
    """
    from palinode.consolidation import status_doc

    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    dates = {"d200": _days_ago(200), "d160": _days_ago(160), "d120": _days_ago(120),
             "d10": _days_ago(10), "d2": _days_ago(2)}
    target = _write_target(
        store, "\n".join(line.format(**dates) for line in _REAL_LAYOUT),
    )

    runner.run_consolidation(llm_fn=_no_ops)
    swept = target.read_text(encoding="utf-8")
    history_ids = status_doc.fact_ids(
        (store / "projects" / "proj-history.md").read_text(encoding="utf-8")
    )
    repaired, report = status_doc.repair_status_doc(swept, extra_known_ids=history_ids)

    assert report["log_lines_elided"] == 0
    assert report["log_ids_unresolved"] == 0
    assert "- [UPDATE] proj-status-4d5e6f" in repaired
    assert f"### {dates['d120']}" in repaired
    assert "_[log elided] 291 operation line(s)" in repaired


def test_the_model_is_shown_only_what_the_sweep_left(store, monkeypatch) -> None:
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    _write_target(store, "\n".join([
        _log_line(200, 1), _log_line(150, 2), _log_line(3, 3),
    ]))
    seen: list[str] = []

    runner.run_consolidation(llm_fn=_recording_llm(seen))

    assert len(seen) == 1
    assert "s0003" in seen[0]
    assert "s0001" not in seen[0] and "s0002" not in seen[0]
    assert "EXISTING_FACTS (1 facts" in seen[0]


def test_a_document_swept_empty_is_a_no_op_group_not_a_failure(
    store, monkeypatch
) -> None:
    """Every fact retired by age leaves nothing to compact — a quiet group."""
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    _write_target(store, _log_line(200, 1))

    result = runner.run_consolidation(llm_fn=_no_ops)

    assert result["status"] == "success"
    assert result["age_retired"] == 1
    assert result["projects_no_ops"] == ["proj"]


def test_the_nightly_pass_never_age_retires(store, monkeypatch) -> None:
    """Nightly stays UPDATE/SUPERSEDE-shaped: no sweep, no `age_retired` key."""
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    target = _write_target(store, _log_line(200, 1))

    result = runner.run_nightly(llm_fn=_no_ops)

    assert "age_retired" not in result
    assert "<!-- fact:s0001 -->" in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The measurement — a replica of the document that could not be digested
# ---------------------------------------------------------------------------

#: The dogfood shape, from the issue: 449 tagged facts, ~430 of them dated
#: session-end lines across six months, the rest current curated facts.
BACKLOG_LINES = 400   # older than the 90-day window
RECENT_LINES = 30     # inside it
CURRENT_FACTS = 19    # undated, curated
TOTAL_FACTS = BACKLOG_LINES + RECENT_LINES + CURRENT_FACTS

#: One `### <date>` block of operation records per this many session lines, so
#: the fixture interleaves them the way a real document does.
_BLOCK_EVERY = 40


def _dogfood_body() -> str:
    """The real layout: curated facts above, every session line *inside* the log.

    Mirrors `projects/palinode-status.md` on the dogfood host — curated bullets
    under `## Current Work`, then `## Consolidation Log` holding the elision
    bullet, the dated session lines session-end appended to the end of the
    file, and `### <date>` blocks of operation records interleaved among them.
    """
    lines = [
        f"- Curated fact {i}: the architecture still holds. <!-- fact:c{i:04d} -->"
        for i in range(CURRENT_FACTS)
    ]
    lines += [
        "",
        "## Consolidation Log",
        "",
        "- _[log elided] 291 operation line(s) across 48 date block(s) — "
        "2026-03-31 → 2026-06-28. Full detail in git history._",
        "",
    ]
    sessions = (
        [(180 - (i * 85) // BACKLOG_LINES, i) for i in range(BACKLOG_LINES)]
        + [(89 - (i * 88) // RECENT_LINES, BACKLOG_LINES + i) for i in range(RECENT_LINES)]
    )
    for position, (days, index) in enumerate(sessions):
        if position and position % _BLOCK_EVERY == 0:
            lines += [
                f"### {_days_ago(days)}",
                f"- [UPDATE] supersedes-proj-status-{index:04d}: rewrote a fact",
                f"- [ARCHIVE] proj-status-{index:04d}: stale milestone",
            ]
        lines.append(_log_line(days, index))
    return "\n".join(lines)


def test_449_fact_backlog_is_reduced_before_the_model_sees_it(
    store, monkeypatch
) -> None:
    """The measurement the issue asks for, on a replica of the real document.

    Before: 449 facts in EXISTING_FACTS. After the sweep: only the facts inside
    the retention window plus the undated curated ones — far below the ~200
    threshold at which an honest per-fact ARCHIVE proposal overruns the cap.
    """
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    target = _write_target(store, _dogfood_body())
    assert len(_ids(target.read_text(encoding="utf-8"))) == BACKLOG_LINES + RECENT_LINES

    unswept: list[str] = []
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 0)
    runner.run_consolidation(dry_run=True, llm_fn=_recording_llm(unswept))
    assert f"EXISTING_FACTS ({TOTAL_FACTS} facts" in unswept[0]

    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 90)
    swept: list[str] = []
    result = runner.run_consolidation(llm_fn=_recording_llm(swept))

    assert result["age_retired"] == BACKLOG_LINES
    shown = int(swept[0].split("EXISTING_FACTS (", 1)[1].split(" facts", 1)[0])
    assert shown == RECENT_LINES + CURRENT_FACTS
    assert shown < 100, "the whole point: the model is shown the window, not the backlog"
    # The prompt shrinks with it — this is the number the token cap cares about.
    assert len(swept[0]) < len(unswept[0]) / 5


def test_one_archive_before_can_do_what_the_sweep_does(store, monkeypatch) -> None:
    """The model-facing half of the same fix, measured on the same document.

    With the sweep disabled, a single `ARCHIVE_BEFORE` — 1 operation, ~120
    characters — retires the same backlog that needed ~400 ARCHIVE ops.
    """
    monkeypatch.setattr(config.consolidation, "status_log_retention_days", 0)
    target = _write_target(store, _dogfood_body())
    before = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
    proposal = json.dumps([{
        "op": "ARCHIVE_BEFORE",
        "before": before,
        "reason": "six months of session lines, superseded by later releases",
    }])
    assert len(proposal) < 200

    result = runner.run_consolidation(llm_fn=lambda s, u: (proposal, "fake-model"))

    assert result["archived"] == BACKLOG_LINES
    assert result["archived_by_range"] == BACKLOG_LINES
    assert result["age_retired"] == 0
    assert len(_ids(target.read_text(encoding="utf-8"))) == RECENT_LINES
