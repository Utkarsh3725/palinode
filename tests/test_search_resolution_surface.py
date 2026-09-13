"""``resolution`` on the search surfaces — one decision, three renderings.

The policy is decided once, server-side, and each surface only formats it, so
this file asserts what that buys: the REST block, the MCP text and the CLI
lines report the same outcome and the same reasons for the same hit. Real
SQLite under ``tmp_path`` through ``TestClient``; no DB mocking.
"""
from __future__ import annotations

import hashlib
import math
import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from palinode.api.server import app
from palinode.cli.search import _resolution_lines
from palinode.core import store
from palinode.core.config import config
from palinode.indexer import reconcile
from palinode.mcp import _format_results

_DIM = 1024


def _bow_embed(text: str, backend: str = "local") -> list[float]:
    vec = [0.0] * _DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    with patch("palinode.core.embedder.embed", side_effect=_bow_embed):
        yield tmp_path


def _write(mem, rel: str, body: str, **meta) -> None:
    import yaml

    path = mem / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(meta, default_flow_style=False, sort_keys=False)
    content = f"---\n{fm}---\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    assert reconcile.reconcile(str(path), content).committed


@pytest.fixture()
def client(mem):
    with TestClient(app) as c:
        yield c


def _replaced(mem) -> None:
    """A replaced record that still ranks first, and its standing successor."""
    _write(mem, "decisions/db.md", "# DB\n\nWe use Postgres as the primary database.",
           type="Decision", status="active", superseded_by="decisions/db-v2",
           date="2026-01-05", entities=["project/shop"])
    _write(mem, "decisions/db-v2.md", "# DB v2\n\nThe primary database is SQLite now.",
           type="Decision", status="active", date="2026-09-01", entities=["project/shop"])


def _conflicted(mem) -> None:
    _write(mem, "decisions/deploy.md", "# Deploy\n\nDeploys go to the VPS.",
           type="Decision", status="active", date="2026-01-05", entities=["project/shop"])
    _write(mem, "observations/deploy-seen.md", "# Deploy seen\n\nDeploys are going to k8s.",
           epistemic="fact", status="active", date="2026-09-01",
           entities=["project/shop"], contradicts=["decisions/deploy"])


def test_resolution_is_opt_in_and_additive(client, mem):
    _replaced(mem)
    body = {"query": "primary database Postgres", "limit": 1, "threshold": 0.0}

    def _stable(rows):
        return [{k: v for k, v in r.items()
                 if k not in ("recall_count", "last_recalled", "importance")} for r in rows]

    plain = _stable(client.post("/search", json=body).json())
    assert plain and "resolution" not in plain[0] and "evidence" not in plain[0]
    assert plain == _stable(client.post("/search", json=body | {"resolve": "none"}).json())

    hit = _stable(client.post("/search", json=body | {"resolve": "linked"}).json())[0]
    assert hit["rel_path"] == "decisions/db.md"
    block = hit["resolution"]
    assert set(block) == {"outcome", "current", "sides", "support", "reasons", "qualifiers"}
    assert block["outcome"] == "supported_current"
    assert block["current"]["ref"] == "decisions/db-v2"
    assert "explicit_replacement" in block["reasons"]
    # The hit itself is untouched: every key it had without `resolve` is intact.
    for k, v in plain[0].items():
        assert hit[k] == v


def test_every_surface_reports_the_same_outcome_and_reasons(client, mem):
    _replaced(mem)
    rows = client.post("/search", json={
        "query": "primary database Postgres", "limit": 1, "threshold": 0.0, "resolve": "linked",
    }).json()
    block = rows[0]["resolution"]

    mcp_text = _format_results(rows)
    cli_text = "\n".join(_resolution_lines(block))
    for rendering in (mcp_text, cli_text):
        assert "current" in rendering
        assert "decisions/db-v2" in rendering
        for reason in block["reasons"]:
            assert reason in rendering

    # Deterministic: the same request twice decides the same way.
    again = client.post("/search", json={
        "query": "primary database Postgres", "limit": 1, "threshold": 0.0, "resolve": "linked",
    }).json()
    assert again[0]["resolution"] == block


def test_a_conflict_keeps_both_sides_on_every_surface(client, mem):
    _conflicted(mem)
    rows = client.post("/search", json={
        "query": "Deploys go to the VPS", "limit": 1, "threshold": 0.0, "resolve": "linked",
    }).json()
    block = rows[0]["resolution"]
    assert block["outcome"] == "unresolved_conflict"
    assert block["current"] is None
    assert "policy_implementation_mismatch" in block["reasons"]

    refs = {s["ref"] for s in block["sides"]}
    assert refs == {"decisions/deploy", "observations/deploy-seen"}
    for rendering in (_format_results(rows), "\n".join(_resolution_lines(block))):
        assert "unresolved conflict" in rendering
        for ref in refs:
            assert ref in rendering


def test_renderers_stay_silent_without_a_resolution():
    row = {"file_path": "/store/decisions/db.md", "score": 0.9, "snippet": "x", "metadata": {}}
    assert "⇒" not in _format_results([row])
    assert _resolution_lines(None) == []


def test_recency_branch_resolves_too(client, mem):
    _replaced(mem)
    hits = client.post("/search", json={"query": "", "limit": 5, "resolve": "linked"}).json()
    assert hits and all("resolution" in h for h in hits)
