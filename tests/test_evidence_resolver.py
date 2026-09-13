"""Bounded evidence resolution (``palinode.core.evidence``) — read-only closure.

Real files under ``tmp_path``, real SQLite through the reconcile seam, real
FTS5, real git for the no-mutation check. The embedder is the only double and
it is deterministic: a hashed bag-of-words vector, so "semantically similar"
means "shares words" and the vector arm ranks the same way on every run.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from palinode.core import evidence as ev
from palinode.core import store
from palinode.core.config import config
from palinode.core.evidence import EvidenceBudget, RequestCache, resolve_evidence
from palinode.core.scope import ScopeChain
from palinode.indexer import reconcile

_DIM = 1024
_NOW = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)


def _bow_embed(text: str, backend: str = "local") -> list[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalised."""
    vec = [0.0] * _DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    """Git-backed memory dir with a real index and the deterministic embedder."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", False)
    store.init_db()
    with patch("palinode.core.embedder.embed", side_effect=_bow_embed):
        yield tmp_path


def _doc(body: str, **meta) -> str:
    import yaml

    fm = yaml.safe_dump(meta, default_flow_style=False, sort_keys=False)
    return f"---\n{fm}---\n\n{body}\n"


def _write(mem, rel: str, body: str, *, index: bool = True, **meta) -> str:
    """Write ``rel`` with the given frontmatter and (by default) index it."""
    path = mem / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _doc(body, **meta)
    path.write_text(content, encoding="utf-8")
    if index:
        diff = reconcile.reconcile(str(path), content)
        assert diff.committed, diff
    return str(path)


def _seed_row(mem, rel: str) -> dict:
    """The search-result shape for ``rel`` straight from the index."""
    db = store.get_db()
    try:
        row = db.execute(
            "SELECT id, file_path, section_id, content_hash FROM chunks WHERE file_path = ?",
            (str(mem / rel),),
        ).fetchone()
    finally:
        db.close()
    assert row is not None, rel
    return {"id": row["id"], "file_path": row["file_path"],
            "section_id": row["section_id"], "content_hash": row["content_hash"]}


def _refs(records) -> list[str]:
    return [r.ref for r in records]


def _dump(result) -> str:
    import json

    return json.dumps([s.to_dict() for s in result.seeds])


# ── link traversal ───────────────────────────────────────────────────────────


def test_forward_and_reverse_contradicts(mem):
    _write(mem, "insights/a.md", "# A\n\nThroughput is 900 per hour.",
           status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nThroughput is 400 per hour.", status="active")
    # c names a; a does not name c — reachable only through the reverse edge.
    _write(mem, "insights/c.md", "# C\n\nThroughput was never measured.",
           status="active", contradicts=["insights/a"])

    res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked", now=_NOW)
    seed = res.seeds[0]
    by_ref = {r.ref: r for r in seed.conflicts}
    assert set(by_ref) == {"insights/b", "insights/c"}
    assert by_ref["insights/b"].direction == "forward"
    assert by_ref["insights/c"].direction == "reverse"
    assert by_ref["insights/c"].via == "insights/a"
    assert by_ref["insights/b"].currency == "current"
    assert by_ref["insights/b"].freshness == "valid"
    assert "Throughput is 400" in by_ref["insights/b"].excerpt
    assert seed.coverage() == {"status": "partial", "reasons": ["fallback_disabled"]}


def test_backed_by_forward_and_dependents_reverse(mem):
    _write(mem, "insights/claim.md", "# Claim\n\nLatency is 20 ms.",
           status="active", backed_by=["research/paper"])
    _write(mem, "research/paper.md", "# Paper\n\nMeasured latency of 20 ms.", status="active")
    _write(mem, "decisions/act.md", "# Act\n\nWe act on the claim.",
           status="active", backed_by=["insights/claim"])

    seed = resolve_evidence([_seed_row(mem, "insights/claim.md")], mode="linked").seeds[0]
    by_ref = {r.ref: r for r in seed.support}
    assert by_ref["research/paper"].direction == "forward"
    assert by_ref["decisions/act"].direction == "reverse"


def test_replacement_chain_surfaces_successor_without_presenting_retired_as_current(mem):
    _write(mem, "decisions/db.md", "# DB\n\nWe use Postgres.",
           status="active", superseded_by="decisions/db-v2")
    _write(mem, "decisions/db-v2.md", "# DB v2\n\nWe use SQLite.",
           status="active", superseded_by="decisions/db-v3")
    _write(mem, "decisions/db-v3.md", "# DB v3\n\nWe use SQLite with WAL.",
           status="active", date="2026-09-01")

    rows = store.check_freshness([_seed_row(mem, "decisions/db.md") | {"content": "We use Postgres."}])
    assert rows[0]["currency"] == "retired"  # the seed itself, from the search path

    seed = resolve_evidence(rows, mode="linked", now=_NOW).seeds[0]
    chain = [(r.ref, r.depth, r.currency, r.direction) for r in seed.replacements]
    assert chain == [
        ("decisions/db-v2", 1, "retired", "forward"),
        ("decisions/db-v3", 2, "current", "forward"),
    ]
    assert seed.replacements[1].effective_at.startswith("2026-09-01")


def test_reverse_replacement_finds_predecessor(mem):
    _write(mem, "decisions/old.md", "# Old\n\nUse endpoint A.",
           status="active", superseded_by="decisions/new")
    _write(mem, "decisions/new.md", "# New\n\nUse endpoint B.", status="active")

    seed = resolve_evidence([_seed_row(mem, "decisions/new.md")], mode="linked").seeds[0]
    assert [(r.ref, r.direction, r.currency) for r in seed.replacements] == [
        ("decisions/old", "reverse", "retired"),
    ]


def test_cycle_terminates_and_each_edge_once(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nbeta", status="active", contradicts=["insights/c"])
    _write(mem, "insights/c.md", "# C\n\ngamma", status="active", contradicts=["insights/a"])

    res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked",
                           budget=EvidenceBudget(max_depth=5, max_edges=100))
    seed = res.seeds[0]
    refs = _refs(seed.conflicts)
    assert sorted(refs) == ["insights/b", "insights/c"]
    assert len(refs) == len(set(refs)), "a record was reported twice"
    # a→b forward and c→a reverse close the ring; b→c is already reported.
    assert res.stats["edges_followed"] == 2
    assert seed.coverage()["reasons"] == ["fallback_disabled"]


def test_reciprocal_backlink_is_one_record(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nbeta", status="active", contradicts=["insights/a"])
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked").seeds[0]
    assert _refs(seed.conflicts) == ["insights/b"]


def test_seed_is_never_its_own_evidence_and_other_seeds_are_not_discovered(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha shared", status="active",
           contradicts=["insights/b"], entities=["project/x"])
    _write(mem, "insights/b.md", "# B\n\nbeta shared", status="active",
           contradicts=["insights/a"], entities=["project/x"])
    rows = [_seed_row(mem, "insights/a.md"), _seed_row(mem, "insights/b.md")]
    res = resolve_evidence(rows, mode="full")
    for seed in res.seeds:
        assert seed.seed_ref not in _refs(seed.records())
        assert not seed.discovered  # the other seed is already in the result list


# ── missing / hidden / scope ─────────────────────────────────────────────────


def test_missing_target_is_a_reason_not_a_record(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active",
           contradicts=["insights/nope", "../../etc/passwd"])
    res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked")
    seed = res.seeds[0]
    assert seed.conflicts == []
    assert "target_missing" in seed.reasons
    assert "nope" not in _dump(res) and "passwd" not in _dump(res)


def test_hidden_target_leaks_nothing(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/secret"])
    _write(mem, "insights/secret.md", "# Secret Title Zebra\n\nzebra body",
           status="active", visibility="private", contradicts=["insights/a"])
    res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked", chain=None)
    seed = res.seeds[0]
    assert seed.conflicts == []
    assert "target_hidden" in seed.reasons
    text = _dump(res)
    assert "Zebra" not in text and "secret" not in text and "zebra" not in text
    # The count is not revealed either: one hidden target or three read the same.
    _write(mem, "insights/secret2.md", "# Second\n\nzz", status="active",
           visibility="private", contradicts=["insights/a"])
    res2 = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked", chain=None)
    assert res2.seeds[0].coverage() == seed.coverage()


def test_scope_mismatch_is_distinguished_from_hidden(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active",
           contradicts=["insights/other", "insights/mine"])
    _write(mem, "insights/other.md", "# Other\n\nother", status="active", scope="project/other")
    _write(mem, "insights/mine.md", "# Mine\n\nmine", status="active", scope="project/mine")
    chain = ScopeChain(project="mine")
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked", chain=chain).seeds[0]
    assert _refs(seed.conflicts) == ["insights/mine"]
    assert "scope_mismatch" in seed.reasons and "target_hidden" not in seed.reasons


def test_hidden_fallback_candidate_is_a_reason_only(mem):
    _write(mem, "decisions/a.md", "# A\n\nwe chose alpha", status="active", entities=["project/x"])
    _write(mem, "decisions/private.md", "# Private Quokka\n\nquokka", status="active",
           entities=["project/x"], visibility="private")
    res = resolve_evidence([_seed_row(mem, "decisions/a.md")], mode="full", chain=None)
    assert "target_hidden" in res.seeds[0].reasons
    assert "uokka" not in _dump(res)


# ── index lag and concurrent edits ───────────────────────────────────────────


def test_reverse_edge_the_index_reports_but_the_file_dropped_is_index_lag(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active")
    b = _write(mem, "insights/b.md", "# B\n\nbeta", status="active", contradicts=["insights/a"])
    # The link is removed on disk; the index still carries it.
    open(b, "w", encoding="utf-8").write(_doc("# B\n\nbeta", status="active"))
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked").seeds[0]
    assert seed.conflicts == []
    assert "index_lag" in seed.reasons


def test_expanded_record_whose_body_changed_is_marked_stale_not_silently_used(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/b"])
    b = _write(mem, "insights/b.md", "# B\n\nold body", status="active")
    open(b, "w", encoding="utf-8").write(_doc("# B\n\nnew body", status="active"))
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked").seeds[0]
    rec = seed.conflicts[0]
    assert rec.freshness == "stale"
    assert "new body" in rec.excerpt  # live text, flagged — not the indexed text unflagged
    assert "index_lag" in seed.reasons


def test_seed_edited_between_search_and_expansion_is_a_revision_mismatch(mem):
    a = _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nbeta", status="active")
    row = _seed_row(mem, "insights/a.md")  # hash as the search saw it
    open(a, "w", encoding="utf-8").write(_doc("# A\n\nalpha edited", status="active",
                                              contradicts=["insights/b"]))
    seed = resolve_evidence([row], mode="linked").seeds[0]
    assert seed.seed_freshness == "stale"
    assert "index_lag" in seed.reasons
    assert _refs(seed.conflicts) == ["insights/b"]  # links still read from the live file


def test_unindexed_record_reports_unknown_freshness(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nbeta", status="active", index=False)
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked").seeds[0]
    assert seed.conflicts[0].freshness == "unknown"


# ── budgets ──────────────────────────────────────────────────────────────────


def _chain(mem, n: int) -> None:
    for i in range(n):
        meta = {"status": "active"}
        if i + 1 < n:
            meta["superseded_by"] = f"decisions/v{i + 1}"
        _write(mem, f"decisions/v{i}.md", f"# v{i}\n\nversion {i}", **meta)


def test_replacement_chain_limit_is_separate_from_edge_limit(mem):
    _chain(mem, 6)
    row = _seed_row(mem, "decisions/v0.md")

    short = resolve_evidence([row], mode="linked",
                             budget=EvidenceBudget(max_replacement_chain=2, max_edges=48)).seeds[0]
    assert _refs(short.replacements) == ["decisions/v1", "decisions/v2"]
    assert "budget_exhausted:replacement_chain" in short.reasons
    assert "budget_exhausted:edges" not in short.reasons

    full = resolve_evidence([row], mode="linked",
                            budget=EvidenceBudget(max_replacement_chain=10, max_edges=48)).seeds[0]
    assert _refs(full.replacements) == [f"decisions/v{i}" for i in range(1, 6)]
    assert "budget_exhausted:replacement_chain" not in full.reasons

    edges = resolve_evidence([row], mode="linked",
                             budget=EvidenceBudget(max_replacement_chain=10, max_edges=1)).seeds[0]
    assert _refs(edges.replacements) == ["decisions/v1"]
    assert "budget_exhausted:edges" in edges.reasons
    assert "budget_exhausted:replacement_chain" not in edges.reasons


def test_file_budget_is_named_when_exhausted(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active",
           contradicts=[f"insights/t{i}" for i in range(4)])
    for i in range(4):
        _write(mem, f"insights/t{i}.md", f"# T{i}\n\nt{i}", status="active")
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked",
                            budget=EvidenceBudget(max_files=2)).seeds[0]
    assert len(seed.conflicts) == 2
    assert "budget_exhausted:files" in seed.reasons


def test_depth_limit_is_named_only_when_something_is_cut_off(mem):
    _write(mem, "insights/a.md", "# A\n\na", status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nb", status="active", contradicts=["insights/c"])
    _write(mem, "insights/c.md", "# C\n\nc", status="active", contradicts=["insights/d"])
    _write(mem, "insights/d.md", "# D\n\nd", status="active")
    row = _seed_row(mem, "insights/a.md")
    cut = resolve_evidence([row], mode="linked", budget=EvidenceBudget(max_depth=2)).seeds[0]
    assert _refs(cut.conflicts) == ["insights/b", "insights/c"]
    assert "budget_exhausted:depth" in cut.reasons
    whole = resolve_evidence([row], mode="linked", budget=EvidenceBudget(max_depth=3)).seeds[0]
    assert _refs(whole.conflicts) == ["insights/b", "insights/c", "insights/d"]
    assert "budget_exhausted:depth" not in whole.reasons


def test_fallback_budgets_are_separate_and_named(mem):
    for i in range(3):
        _write(mem, f"decisions/s{i}.md", f"# S{i}\n\nseed {i} about widgets",
               status="active", entities=["project/w"])
    for i in range(6):
        _write(mem, f"decisions/c{i}.md", f"# C{i}\n\ncorrection {i} about widgets",
               status="active", entities=["project/w"])
    rows = [_seed_row(mem, f"decisions/s{i}.md") for i in range(3)]

    reads = resolve_evidence(rows, mode="full",
                             budget=EvidenceBudget(fallback_max_reads=2, fallback_max_queries=99))
    assert reads.stats["fallback_reads"] == 2
    assert any("budget_exhausted:fallback_reads" in s.reasons for s in reads.seeds)
    assert not any("budget_exhausted:fallback_queries" in s.reasons for s in reads.seeds)

    queries = resolve_evidence(rows, mode="full",
                               budget=EvidenceBudget(fallback_max_queries=1, fallback_max_reads=99))
    assert queries.stats["fallback_queries"] == 1
    assert any("budget_exhausted:fallback_queries" in s.reasons for s in queries.seeds)


def test_mode_none_touches_nothing(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active", contradicts=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nbeta", status="active")
    cache = RequestCache()
    res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="none", request_cache=cache)
    assert res.seeds[0].records() == [] and res.seeds[0].reasons == set()
    assert cache.files_read == 0


def test_unknown_mode_is_rejected(mem):
    with pytest.raises(ValueError):
        resolve_evidence([], mode="everything")


def test_request_cache_reads_each_file_once(mem):
    _write(mem, "insights/a.md", "# A\n\na", status="active",
           contradicts=["insights/b"], backed_by=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nb", status="active", contradicts=["insights/a"])
    cache = RequestCache()
    res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked", request_cache=cache)
    assert res.stats["files_read"] == 2  # a (seed) and b, once each
    assert res.stats["reverse_lookups"] == 1


# ── unlinked discovery: positive and negative controls ───────────────────────


def test_unlinked_correction_is_discovered_only_with_fallback(mem):
    _write(mem, "decisions/cache.md", "# Cache\n\nWe standardised on Redis for the cache.",
           status="active", entities=["project/shop"], date="2026-03-01")
    _write(mem, "decisions/cache-later.md",
           "# Cache revisited\n\nDropped the managed key-value service; the shop now keeps its cache in-process.",
           status="active", entities=["project/shop"], date="2026-08-01")
    _write(mem, "decisions/unrelated.md", "# Unrelated\n\nWe picked a logo colour.",
           status="active", entities=["project/brand"], date="2026-08-02")
    row = _seed_row(mem, "decisions/cache.md")

    linked = resolve_evidence([row], mode="linked").seeds[0]
    assert linked.records() == []
    assert linked.coverage() == {"status": "partial", "reasons": ["fallback_disabled"]}

    full = resolve_evidence([row], mode="full").seeds[0]
    found = {r.ref: r for r in full.discovered}
    assert "decisions/cache-later" in found
    assert found["decisions/cache-later"].relation == "entity"
    assert found["decisions/cache-later"].direction == "discovered"
    assert found["decisions/cache-later"].effective_at.startswith("2026-08-01")
    assert "decisions/unrelated" not in found  # negative control: other subject
    assert "fallback_disabled" not in full.reasons


def test_newest_decision_with_no_relation_is_found_by_neighbour_query(mem):
    # No entities, no links, no shared identifier in the title — only the body overlaps.
    _write(mem, "decisions/deploy.md", "# Deploy\n\nDeploys go through the nightly pipeline.",
           status="active", date="2026-01-01")
    _write(mem, "decisions/pipeline-change.md",
           "# Pipeline change\n\nDeploys no longer go through the nightly pipeline; they run on merge.",
           status="active", date="2026-09-01")
    _write(mem, "insights/cats.md", "# Cats\n\nThe office cat prefers the sunny desk.", status="active")
    row = _seed_row(mem, "decisions/deploy.md")
    full = resolve_evidence([row], mode="full").seeds[0]
    found = {r.ref: r for r in full.discovered}
    assert "decisions/pipeline-change" in found
    assert found["decisions/pipeline-change"].relation in ("keyword", "neighbor")
    assert "insights/cats" not in found


def test_cited_sources_and_claim_anchors_are_exact_lookups(mem):
    _write(mem, "research/paper.md", "# Paper\n\nThe measured figure is 20 ms.", status="active")
    _write(mem, "insights/claim.md", "# Claim\n\nLatency is 20 ms.", status="active",
           sources=[{"ref": "research/paper.md", "quote": "measured figure is 20 ms"}],
           claims=[{"claim_id": "abc", "text": "Latency is 20 ms.",
                    "source_id": "research/paper.md",
                    "span": {"quote": "measured figure is 20 ms"}}])
    res = resolve_evidence([_seed_row(mem, "insights/claim.md")], mode="full",
                           budget=EvidenceBudget(fallback_max_queries=0))
    full = res.seeds[0]
    assert [(r.ref, r.relation) for r in full.discovered] == [("research/paper", "sources")]
    assert res.stats["fallback_queries"] == 0  # exact refs cost no query


# ── no mutation ──────────────────────────────────────────────────────────────


def test_resolution_never_writes_or_commits(mem):
    _write(mem, "insights/a.md", "# A\n\nalpha", status="active",
           contradicts=["insights/b"], entities=["project/x"])
    _write(mem, "insights/b.md", "# B\n\nbeta", status="active", superseded_by="insights/c")
    _write(mem, "insights/c.md", "# C\n\ngamma", status="active", entities=["project/x"])
    subprocess.run(["git", "-C", str(mem), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(mem), "commit", "-q", "-m", "seed"], check=True)
    before = {p: os.stat(p).st_mtime_ns for p in mem.rglob("*.md")}
    head = subprocess.run(["git", "-C", str(mem), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout

    def _boom(*a, **k):
        raise AssertionError("evidence resolution must not write")

    with patch("palinode.core.git_tools.write_memory_file", _boom), \
         patch("palinode.core.git_tools.commit_memory_file", _boom), \
         patch("palinode.core.git_tools.commit_memory_files", _boom), \
         patch("palinode.core.store.record_recall", _boom):
        res = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="full")
    assert res.seeds[0].records()
    after = {p: os.stat(p).st_mtime_ns for p in mem.rglob("*.md")}
    assert after == before
    status = subprocess.run(["git", "-C", str(mem), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    assert status.strip() == ""
    assert subprocess.run(["git", "-C", str(mem), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout == head


# ── held-out recall on a seeded corpus ───────────────────────────────────────

# Six correction / conflict pairs. Each pair shares a subject; the second side
# is written to rank below a small top-k for the pair's query, either because
# it uses different vocabulary or because noise files share the query's words.
# Three pairs carry an explicit link (one forward contradicts, one reverse
# contradicts, one superseded_by); three carry none and can be reached only by
# discovery (shared entity; shared identifier in the body; body overlap only).
_PAIRS = [
    # (seed rel, seed body, second rel, second body, entities, link, query)
    ("decisions/db.md", "We use Postgres as the primary database.",
     "decisions/db-move.md", "The primary datastore moved off the managed relational service to SQLite.",
     ["project/shop"], ("contradicts", "forward"), "primary database"),
    ("decisions/auth.md", "Sessions are stored in signed cookies.",
     "decisions/auth-fix.md", "Signed cookie sessions were dropped for server-side tokens.",
     ["project/shop"], ("contradicts", "reverse"), "session cookies"),
    ("decisions/queue.md", "Background jobs run on the beanstalk queue.",
     "decisions/queue-v2.md", "Background jobs now run on the built-in scheduler.",
     ["project/shop"], ("superseded_by", "forward"), "background jobs queue"),
    ("decisions/cdn.md", "Static assets are served from the edge cache.",
     "decisions/cdn-off.md", "Static assets come straight from the origin again; the edge layer was removed.",
     ["project/site"], None, "static assets edge cache"),
    ("insights/perf.md", "Throughput is 900 units per hour on the widget line.",
     "insights/perf-remeasure.md", "Remeasured widget line throughput at 400 units per hour.",
     [], None, "widget line throughput"),
    ("decisions/deploy.md", "Deploys go through the nightly pipeline.",
     "decisions/deploy-change.md", "Deploys no longer go through the nightly pipeline; they ship on merge.",
     [], None, "deploy nightly pipeline"),
]


def _seed_corpus(mem) -> None:
    for seed_rel, seed_body, second_rel, second_body, ents, link, _q in _PAIRS:
        seed_meta = {"status": "active", "date": "2026-01-01"}
        second_meta = {"status": "active", "date": "2026-08-01"}
        if ents:
            seed_meta["entities"] = list(ents)
            second_meta["entities"] = list(ents)
        if link == ("contradicts", "forward"):
            seed_meta["contradicts"] = [second_rel[:-3]]
        elif link == ("contradicts", "reverse"):
            second_meta["contradicts"] = [seed_rel[:-3]]
        elif link == ("superseded_by", "forward"):
            seed_meta["superseded_by"] = second_rel[:-3]
        _write(mem, seed_rel, f"# {seed_rel}\n\n{seed_body}", **seed_meta)
        _write(mem, second_rel, f"# {second_rel}\n\n{second_body}", **second_meta)
    # Noise: near-restatements of each seed, so for a query in the seed's own
    # words the seed ranks first, the restatements fill the window, and the
    # second side — which says the same thing in other words — falls below it.
    for i, (_s, seed_body, _r, _rb, _e, _l, _q) in enumerate(_PAIRS):
        for j in range(4):
            _write(mem, f"insights/noise-{i}-{j}.md",
                   f"# Note {i}-{j}\n\nNoted in passing (item {j}): {seed_body}",
                   status="active", date="2026-02-01")


def _search(query: str, limit: int) -> list[dict]:
    return store.search_hybrid(query, _bow_embed(query), top_k=limit, threshold=0.0,
                               record_access=False)


def test_held_out_recall_improves_with_resolver(mem, capsys):
    _seed_corpus(mem)
    assert len(list(mem.rglob("*.md"))) >= 30
    limit = 2
    hits_off = hits_linked = hits_full = 0
    linked_reachable = 0
    for seed_rel, seed_body, second_rel, _rb, _e, link, _q in _PAIRS:
        second_ref = second_rel[:-3]
        query = seed_body
        results = _search(query, limit)
        seed_rows = [r for r in results if r["file_path"] == str(mem / seed_rel)]
        assert seed_rows, f"seed {seed_rel} must be in the top-{limit} for {query!r}"
        in_top = any(r["file_path"] == str(mem / second_rel) for r in results)
        hits_off += in_top
        linked = resolve_evidence(seed_rows, mode="linked").seeds[0]
        linked_found = second_ref in _refs(linked.records())
        hits_linked += in_top or linked_found
        linked_reachable += linked_found
        full = resolve_evidence(seed_rows, mode="full").seeds[0]
        hits_full += in_top or second_ref in _refs(full.records())
        if link is None:
            assert not linked_found, "unlinked pair must not be reachable by links alone"
    n = len(_PAIRS)
    with capsys.disabled():
        print(f"\nheld-out second-side recall@{limit}: off={hits_off}/{n} "
              f"linked={hits_linked}/{n} full={hits_full}/{n}; "
              f"explicit-link coverage={linked_reachable}/{n}")
    assert hits_off < hits_full
    assert hits_linked >= hits_off
    assert hits_full == n
    assert linked_reachable == 3  # the three pairs the corpus links explicitly


def test_coverage_vocabulary_is_closed(mem):
    _write(mem, "insights/a.md", "# A\n\na", status="active", contradicts=["insights/nope"])
    seed = resolve_evidence([_seed_row(mem, "insights/a.md")], mode="linked").seeds[0]
    assert seed.reasons <= ev.COVERAGE_REASONS
