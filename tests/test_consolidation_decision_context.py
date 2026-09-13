"""The compaction prompt must carry the decisions governing the project.

``_get_decisions_for_project`` existed since the original runner and was never
called — one occurrence in the whole package, the ``def`` itself. So
``_consolidate_project`` proposed operations against a project's status doc with
no knowledge of that project's decisions, and could propose an UPDATE or
SUPERSEDE contradicting a live decision with nothing to notice: the executor
validates operation *shape*, not agreement with the decision record.

Decisions are supplied as **constraints**, not as material to compact. They are
not in EXISTING_FACTS and no operation targets them, which is why these tests
assert on what reaches the prompt rather than on what the executor writes.
"""
from __future__ import annotations

import pytest

from palinode.consolidation import runner
from palinode.core.config import config


PROMPT = """# Compaction Prompt
Return the operations JSON array.
"""


def _decision(tmp_path, slug: str, *, project: str, body: str, status: str | None = None):
    d = tmp_path / "decisions"
    d.mkdir(parents=True, exist_ok=True)
    front = f"id: decisions-{slug}\nname: {slug}\nentities:\n  - project/{project}\n"
    if status:
        front += f"status: {status}\n"
    (d / f"{slug}.md").write_text(f"---\n{front}---\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def project_store(tmp_path, monkeypatch):
    """A project with one tagged fact, plus the prompt the runner reads."""
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))

    prompts = tmp_path / "specs" / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "compaction.md").write_text(PROMPT, encoding="utf-8")

    projects = tmp_path / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    (projects / "alpha.md").write_text(
        "---\nid: projects-alpha\n---\n\n"
        "- The parser runs single-threaded. <!-- fact:alpha-parser -->\n",
        encoding="utf-8",
    )
    return tmp_path


def _capture_prompt(monkeypatch) -> dict:
    """Run the propose seam with a fake LLM and keep the user prompt."""
    seen: dict = {}

    def _fake(system_prompt: str, user_prompt: str) -> tuple[str, str]:
        seen["system"] = system_prompt
        seen["user"] = user_prompt
        return "[]", "stub"

    return seen, _fake


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_active_decisions_reach_the_prompt(project_store, monkeypatch):
    """The fix: a decision governing this project is in the user prompt."""
    _decision(
        project_store,
        "single-writer",
        project="alpha",
        body="All writes go through one choke point. No direct file writes.",
    )
    seen, fake = _capture_prompt(monkeypatch)

    runner._consolidate_project("alpha", [{"date": "2026-08-05", "content": "note"}], llm_fn=fake)

    assert "ACTIVE_DECISIONS" in seen["user"], "decision section missing from prompt"
    assert "one choke point" in seen["user"], "decision body never reached the model"


def test_superseded_decisions_are_withheld(project_store, monkeypatch):
    """A superseded decision is exactly what must NOT be treated as binding."""
    _decision(
        project_store, "live-rule", project="alpha", body="Use the new pipeline."
    )
    _decision(
        project_store,
        "dead-rule",
        project="alpha",
        body="Use the retired pipeline.",
        status="superseded",
    )
    seen, fake = _capture_prompt(monkeypatch)

    runner._consolidate_project("alpha", [{"date": "2026-08-05", "content": "n"}], llm_fn=fake)

    assert "Use the new pipeline" in seen["user"]
    assert "retired pipeline" not in seen["user"], (
        "a superseded decision was presented to the compactor as in force"
    )


def test_other_projects_decisions_are_not_included(project_store, monkeypatch):
    """Scoped by `entities`, so an unrelated project's rules stay out."""
    _decision(project_store, "mine", project="alpha", body="Alpha rule.")
    _decision(project_store, "theirs", project="beta", body="Beta rule.")
    seen, fake = _capture_prompt(monkeypatch)

    runner._consolidate_project("alpha", [{"date": "2026-08-05", "content": "n"}], llm_fn=fake)

    assert "Alpha rule" in seen["user"]
    assert "Beta rule" not in seen["user"]


# ---------------------------------------------------------------------------
# Shape and safety
# ---------------------------------------------------------------------------

def test_no_decisions_means_no_empty_heading(project_store, monkeypatch):
    """Omit the section rather than send an empty header — tokens and clarity."""
    seen, fake = _capture_prompt(monkeypatch)

    runner._consolidate_project("alpha", [{"date": "2026-08-05", "content": "n"}], llm_fn=fake)

    assert "ACTIVE_DECISIONS" not in seen["user"]
    assert "EXISTING_FACTS" in seen["user"], "the rest of the prompt is intact"


def test_decision_context_is_budgeted(project_store, monkeypatch):
    """A long decision record must not crowd out the notes being compacted."""
    for i in range(40):
        _decision(
            project_store, f"rule-{i:02d}", project="alpha", body="x" * 400
        )
    seen, fake = _capture_prompt(monkeypatch)

    runner._consolidate_project("alpha", [{"date": "2026-08-05", "content": "n"}], llm_fn=fake)

    section = seen["user"].split("ACTIVE_DECISIONS", 1)[1].split("## RECENT_NOTES", 1)[0]
    assert len(section) < runner.MAX_DECISIONS_CHARS + 500, "decision budget not enforced"
    assert "more decision(s) not shown" in section, "truncation must be visible"


def test_unreadable_decisions_do_not_stop_compaction(project_store, monkeypatch):
    """Context is an improvement to the proposal, never a precondition for it."""
    def _boom(_project_id):
        raise OSError("decisions/ is unreadable")

    monkeypatch.setattr(runner, "_get_decisions_for_project", _boom)
    seen, fake = _capture_prompt(monkeypatch)

    ops = runner._consolidate_project(
        "alpha", [{"date": "2026-08-05", "content": "n"}], llm_fn=fake
    )

    assert "EXISTING_FACTS" in seen["user"], "compaction ran without decision context"
    assert ops == ([], "stub") or isinstance(ops, tuple)


def test_the_helper_is_no_longer_dead_code():
    """Pins the wiring itself.

    The defect was not a broken function — it was a correct function nobody
    called. A test on behaviour alone would pass again if someone unwired it and
    the prompt silently lost its constraints, so the call is asserted directly.
    """
    import inspect

    source = inspect.getsource(runner._render_active_decisions)
    assert "_get_decisions_for_project" in source
    assert "_render_active_decisions" in inspect.getsource(runner._format_active_decisions)

    # The prompt is built in _assemble_prompt, which _consolidate_project calls.
    builder = inspect.getsource(runner._assemble_prompt)
    assert "_render_active_decisions" in builder, (
        "_assemble_prompt no longer loads decision context"
    )
    assert "_assemble_prompt" in inspect.getsource(runner._consolidate_project), (
        "_consolidate_project no longer builds the prompt through _assemble_prompt"
    )
