"""The session-start digest selects and qualifies through the shared lifecycle
classifier — a retired record is never presented as current, and a usable
record's ``contradicts`` / ``stale_backing`` / ``epistemic`` reach every
surface the digest is rendered on.

The recall defect as filed: ``build_context_digest`` selected a ``status:
archived`` decision still under ``decisions/`` as a recent decision, and
``_digest_row`` dropped the qualifiers, so a contested, unverified snapshot
became an unqualified startup summary. The research probe reproduced both.

Regression cases: archived-in-place decision; retired core memory; retired
and qualified ProjectSnapshot; expired acting state (injected clock); active
conflict; stale backing; declared epistemic; unmarked legacy record. Then the
same rows through REST JSON, the MCP ``palinode_session_init`` text and the
CLI ``palinode prime`` text, asserting on the rendered output of each.

Real frontmatter files under ``tmp_path``; no DB (the digest is frontmatter-
only by design).
"""
from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from palinode.api.server import app
from palinode.core import expiry
from palinode.core.config import config
from palinode.core.context_prime import (
    DIGEST_QUALIFIER_KEYS,
    build_context_digest,
    format_context_digest,
)

client = TestClient(app)

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
PAST = (NOW - timedelta(hours=1)).isoformat()
FUTURE = (NOW + timedelta(days=30)).isoformat()


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("PALINODE_PROJECT", raising=False)
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config.context, "auto_detect", True)
    expiry._REPORTED.clear()
    yield tmp_path
    expiry._REPORTED.clear()


def _seed(memory_dir, rel: str, meta: dict[str, Any], body: str = "body"):
    p = memory_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\n{yaml.safe_dump(meta, default_flow_style=False)}---\n\n{body}\n",
        encoding="utf-8",
    )
    return p


def _files(section: list[dict[str, Any]]) -> list[str]:
    return [row["file"] for row in section]


def _all_files(digest: dict[str, Any]) -> set[str]:
    return set(
        _files(digest["core_memories"])
        + _files(digest["recent_decisions"])
        + _files(digest["open_action_items"])
        + _files(digest["recent_snapshots"])
    )


# ── the probe's scenario, on real files ──────────────────────────────────────


@pytest.fixture
def probe_store(memory_dir):
    """The research probe's three records, written to disk."""
    _seed(memory_dir, "decisions/retired-endpoint.md", {
        "type": "Decision", "entities": ["project/demo"], "status": "archived",
        "superseded_by": "decisions/current-endpoint", "description": "Use endpoint A",
        "created_at": "2026-09-01T00:00:00Z",
    })
    _seed(memory_dir, "decisions/current-endpoint.md", {
        "type": "Decision", "entities": ["project/demo"], "status": "active",
        "description": "Use endpoint B", "created_at": "2026-08-01T00:00:00Z",
    })
    _seed(memory_dir, "projects/contested-state.md", {
        "type": "ProjectSnapshot", "entities": ["project/demo"],
        "contradicts": ["decisions/current-endpoint"],
        "stale_backing": [{"ref": "decisions/retired-endpoint", "op": "archive"}],
        "epistemic": "unverified", "description": "Use endpoint A",
    })
    return memory_dir


def test_archived_in_place_decision_is_not_a_recent_decision(probe_store):
    digest = build_context_digest(project="demo", now=NOW)
    assert _files(digest["recent_decisions"]) == ["decisions/current-endpoint.md"]
    assert "decisions/retired-endpoint.md" not in _all_files(digest)


def test_qualified_snapshot_keeps_its_qualifiers_in_the_row(probe_store):
    digest = build_context_digest(project="demo", now=NOW)
    [row] = digest["recent_snapshots"]
    assert row["file"] == "projects/contested-state.md"
    assert row["contradicts"] == ["decisions/current-endpoint"]
    assert row["stale_backing"] == ["decisions/retired-endpoint"]
    assert row["epistemic"] == "unverified"


def test_rendered_digest_carries_conflict_stale_support_and_epistemic(probe_store):
    text = format_context_digest(build_context_digest(project="demo", now=NOW))
    assert "retired-endpoint.md" not in text
    line = next(ln for ln in text.splitlines() if "contested-state.md" in ln)
    assert "⚠ contradicts: decisions/current-endpoint" in line
    assert "⚠ stale backing: decisions/retired-endpoint" in line
    assert "epistemic: unverified" in line


# ── regression cases, one per record kind ────────────────────────────────────


def test_retired_core_memory_is_withheld(memory_dir):
    _seed(memory_dir, "insights/live.md", {"type": "Insight", "core": True, "title": "Live"})
    _seed(memory_dir, "insights/archived.md",
          {"type": "Insight", "core": True, "status": "archived", "title": "Gone"})
    _seed(memory_dir, "insights/deprecated.md",
          {"type": "Insight", "core": True, "lifecycle": "deprecated", "title": "Old"})
    _seed(memory_dir, "insights/replaced.md",
          {"type": "Insight", "core": True, "superseded_by": "insights/live", "title": "Replaced"})
    digest = build_context_digest(now=NOW)
    assert _files(digest["core_memories"]) == ["insights/live.md"]


def test_expired_acting_state_is_withheld_by_the_injected_clock(memory_dir, caplog):
    _seed(memory_dir, "insights/licensed.md",
          {"type": "Insight", "core": True, "expires_at": FUTURE, "title": "Licensed"})
    _seed(memory_dir, "insights/lapsed.md",
          {"type": "Insight", "core": True, "expires_at": PAST, "title": "Lapsed"})
    _seed(memory_dir, "decisions/lapsed-decision.md",
          {"type": "Decision", "entities": ["project/p"], "expires_at": PAST, "title": "Lapsed D"})
    _seed(memory_dir, "decisions/live-decision.md",
          {"type": "Decision", "entities": ["project/p"], "title": "Live D"})

    import logging

    with caplog.at_level(logging.WARNING, logger="palinode.expiry"):
        digest = build_context_digest(project="p", now=NOW)
        build_context_digest(project="p", now=NOW)
    assert _files(digest["core_memories"]) == ["insights/licensed.md"]
    assert _files(digest["recent_decisions"]) == ["decisions/live-decision.md"]
    # The acting-state lapse is still reported, and still once per process.
    hits = [r for r in caplog.records if "core memory insights/lapsed.md expired" in r.getMessage()]
    assert len(hits) == 1

    # Turn the clock back and the same files are all eligible again.
    earlier = build_context_digest(project="p", now=NOW - timedelta(days=2))
    assert set(_files(earlier["core_memories"])) == {"insights/licensed.md", "insights/lapsed.md"}


def test_retired_snapshot_and_action_item_are_withheld(memory_dir):
    _seed(memory_dir, "projects/wrap-live.md",
          {"type": "ProjectSnapshot", "entities": ["project/p"], "title": "Live wrap"})
    _seed(memory_dir, "projects/wrap-archived.md",
          {"type": "ProjectSnapshot", "entities": ["project/p"], "status": "archived",
           "title": "Archived wrap"})
    _seed(memory_dir, "inbox/open.md",
          {"type": "ActionItem", "entities": ["project/p"], "title": "Open"})
    _seed(memory_dir, "inbox/done.md",
          {"type": "ActionItem", "entities": ["project/p"], "status": "done", "title": "Done"})
    _seed(memory_dir, "inbox/archived.md",
          {"type": "ActionItem", "entities": ["project/p"], "status": "archived", "title": "Archived"})
    digest = build_context_digest(project="p", now=NOW)
    assert _files(digest["recent_snapshots"]) == ["projects/wrap-live.md"]
    assert _files(digest["open_action_items"]) == ["inbox/open.md"]


def test_active_conflict_stale_backing_and_epistemic_each_survive_alone(memory_dir):
    _seed(memory_dir, "decisions/contested.md",
          {"type": "Decision", "entities": ["project/p"], "title": "Contested",
           "contradicts": ["insights/other"]})
    _seed(memory_dir, "decisions/shaky.md",
          {"type": "Decision", "entities": ["project/p"], "title": "Shaky",
           "stale_backing": [{"ref": "insights/withdrawn"}]})
    _seed(memory_dir, "insights/guess.md",
          {"type": "Insight", "core": True, "title": "Guess", "epistemic": "inference"})
    digest = build_context_digest(project="p", now=NOW)
    rows = {r["file"]: r for r in digest["recent_decisions"] + digest["core_memories"]}
    assert rows["decisions/contested.md"]["contradicts"] == ["insights/other"]
    assert "stale_backing" not in rows["decisions/contested.md"]
    assert rows["decisions/shaky.md"]["stale_backing"] == ["insights/withdrawn"]
    assert rows["insights/guess.md"]["epistemic"] == "inference"

    text = format_context_digest(digest)
    assert "[⚠ contradicts: insights/other]" in text
    assert "[⚠ stale backing: insights/withdrawn]" in text
    assert "[epistemic: inference]" in text


def test_unmarked_legacy_record_is_usable_and_stays_unmarked(memory_dir):
    _seed(memory_dir, "decisions/legacy.md",
          {"type": "Decision", "entities": ["project/p"], "title": "Legacy"})
    _seed(memory_dir, "insights/legacy-core.md", {"core": True, "title": "Legacy core"})
    digest = build_context_digest(project="p", now=NOW)
    [decision] = digest["recent_decisions"]
    [core] = digest["core_memories"]
    # Present, and with no qualifier keys at all: absence is not a claim.
    assert set(decision) == {"file", "summary"}
    assert set(core) == {"file", "summary"}
    text = format_context_digest(digest)
    assert "- [decisions/legacy.md] Legacy\n" in text + "\n"
    assert "epistemic" not in text


# ── ordering: dates, not touches ─────────────────────────────────────────────


def test_recent_orders_by_effective_date_not_mtime(memory_dir):
    newest = _seed(memory_dir, "decisions/newest.md",
                   {"type": "Decision", "entities": ["project/p"], "title": "Newest",
                    "created_at": "2026-09-01T00:00:00Z"})
    oldest = _seed(memory_dir, "decisions/oldest.md",
                   {"type": "Decision", "entities": ["project/p"], "title": "Oldest",
                    "created_at": "2026-01-01T00:00:00Z"})
    undated = _seed(memory_dir, "decisions/undated.md",
                    {"type": "Decision", "entities": ["project/p"], "title": "Undated"})
    # Touch the oldest and the undated file *after* everything else: a touch
    # is not a new effective decision, so neither climbs above `newest`.
    os.utime(newest, (1000, 1000))
    os.utime(oldest, (9000, 9000))
    os.utime(undated, (9999, 9999))
    digest = build_context_digest(project="p", now=NOW)
    assert _files(digest["recent_decisions"]) == [
        "decisions/newest.md", "decisions/oldest.md", "decisions/undated.md",
    ]


def test_undated_records_keep_a_deterministic_mtime_order_among_themselves(memory_dir):
    a = _seed(memory_dir, "projects/wrap-a.md",
              {"type": "ProjectSnapshot", "entities": ["project/p"], "title": "A"})
    b = _seed(memory_dir, "projects/wrap-b.md",
              {"type": "ProjectSnapshot", "entities": ["project/p"], "title": "B"})
    os.utime(a, (2000, 2000))
    os.utime(b, (1000, 1000))
    digest = build_context_digest(project="p", now=NOW)
    assert _files(digest["recent_snapshots"]) == ["projects/wrap-a.md", "projects/wrap-b.md"]


# ── surface parity: REST JSON, MCP text, CLI text ────────────────────────────


def _assert_equivalent_semantics(text: str, *, wrapped: bool = False) -> None:
    """What every rendered surface must say about the probe store.

    ``wrapped``: the CLI prints through a Rich console that soft-wraps long
    lines at the terminal width, so its output is flattened to one line per
    row before the row-level checks.
    """
    assert "decisions/retired-endpoint.md" not in text
    assert "[decisions/current-endpoint.md]" in text
    if wrapped:
        text = " ".join(text.split())
        rows = text.split("- [")
    else:
        rows = text.splitlines()
    line = next(ln for ln in rows if "contested-state.md" in ln)
    assert "⚠ contradicts: decisions/current-endpoint" in line
    assert "⚠ stale backing: decisions/retired-endpoint" in line
    assert "epistemic: unverified" in line


def test_rest_prime_carries_the_qualifier_keys_and_withholds_retired(probe_store):
    res = client.post("/context/prime", json={"project": "demo"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert _files(data["recent_decisions"]) == ["decisions/current-endpoint.md"]
    [row] = data["recent_snapshots"]
    assert {k for k in DIGEST_QUALIFIER_KEYS} <= set(row)
    assert row["contradicts"] == ["decisions/current-endpoint"]
    assert row["stale_backing"] == ["decisions/retired-endpoint"]
    assert row["epistemic"] == "unverified"
    # Unqualified rows carry none of the keys.
    [decision] = data["recent_decisions"]
    assert not set(DIGEST_QUALIFIER_KEYS) & set(decision)


@pytest.mark.asyncio
async def test_mcp_session_init_renders_equivalent_semantics(probe_store, monkeypatch):
    import palinode.mcp as mcp

    async def _post_via_testclient(path, json=None, timeout=30.0):
        return client.post(path, json=json or {})

    monkeypatch.setattr(mcp, "_post", _post_via_testclient)
    result = await mcp._dispatch_tool("palinode_session_init", {"project": "demo"})
    _assert_equivalent_semantics(result[0].text)


def test_cli_prime_renders_equivalent_semantics(probe_store):
    from click.testing import CliRunner

    from palinode.cli import _api

    prime_mod = importlib.import_module("palinode.cli.prime")
    digest = client.post("/context/prime", json={"project": "demo"}).json()

    class _Client:
        def post(self, path, json=None, timeout=None):
            class _R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return digest

            return _R()

    fake = _api.PalinodeAPI.__new__(_api.PalinodeAPI)
    fake.client = _Client()
    with patch.object(prime_mod, "api_client", fake):
        result = CliRunner().invoke(prime_mod.prime, ["--format", "text", "-p", "demo"])
    assert result.exit_code == 0, result.output
    _assert_equivalent_semantics(result.output, wrapped=True)
