"""Revision-aware stale backing — two-hop support checks and revalidation receipts.

The correctness conditions this pins:

* A source that stops standing reaches its dependent at read time, at one hop
  and at two, and says which kind of loss it was — *withdrawn* (retired, and
  uncertain) or *disproven* (falsified, with the evidence named). Neither ever
  produces the opposite claim or the older retired value.
* Only an explicitly declared ``backing_policy`` lets the checker conclude
  anything from a multi-source list. A legacy list stays advisory.
* A backing counts as revalidated only against the source's *current* raw
  revision. A formatting-only re-save of the dependent certifies nothing.
* Caching is per request and only reusable under the delivery-receipt contract:
  a changed supporting revision, a changed caller scope, a changed policy
  version or a crossed time boundary all invalidate — the last of those with
  no file having changed at all.
* Persistent work goes through the marker/executor/git path that already
  exists. Markdown and git reconstruct the state; a failure stays observable.

Real files under ``tmp_path``, real SQLite through the reconcile seam, real
git, the real executor and the real on-demand archive/restore paths. The
embedder is the only double.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from unittest.mock import patch

import frontmatter
import pytest
import yaml

from palinode.consolidation import propagate, write_time
from palinode.consolidation.archive import archive_memory, restore_memory
from palinode.consolidation.executor import apply_operations
from palinode.core import evidence as ev
from palinode.core import resolution as res
from palinode.core import revalidation as rv
from palinode.core import store
from palinode.core.config import config
from palinode.core.receipt import PolicyVersion, reuse_key
from palinode.indexer import reconcile

_DIM = 1024
_NOW = datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC)


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
    """Git-backed memory dir, real index, deterministic embedder, pinned clock."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    monkeypatch.setattr(config, "db_path", str(tmp_path / ".palinode.db"))
    monkeypatch.setattr(config.git, "auto_commit", True)
    monkeypatch.setattr(propagate, "_utc_now", lambda: _NOW)
    store.init_db()
    with patch("palinode.core.embedder.embed", side_effect=_bow_embed):
        yield tmp_path


def _doc(body: str, **meta) -> str:
    fm = yaml.safe_dump(meta, default_flow_style=False, sort_keys=False)
    return f"---\n{fm}---\n\n{body}\n"


def _write(mem, rel: str, body: str, *, index: bool = True, **meta) -> str:
    path = mem / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _doc(body, **meta)
    path.write_text(content, encoding="utf-8")
    if index:
        diff = reconcile.reconcile(str(path), content)
        assert diff.committed, diff
    return str(path)


def _commit(mem, message: str = "seed") -> None:
    subprocess.run(["git", "-C", str(mem), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(mem), "commit", "-q", "-m", message], check=True)


def _meta(path: str) -> dict:
    return frontmatter.load(path).metadata


def _revision(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _stale_refs(path: str) -> list[str]:
    return [e["ref"] for e in propagate.parse_stale_backing(_meta(path))]


def _seed_row(mem, rel: str) -> dict:
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


def _check(mem, rel: str, *, now=_NOW, **kwargs) -> rv.SupportCheck:
    """``check_support`` for one file, reading the rest of the store from disk."""
    path = str(mem / rel)
    return rv.check_support(
        rv.normalize_ref(rel), _meta(path),
        read=rv.disk_reader(str(mem), now=now), now=now,
        revision=_revision(path), **kwargs,
    )


# ── the vocabularies stay pinned to one another ──────────────────────────────


def test_the_two_copies_of_each_shared_constant_agree() -> None:
    """Three modules spell these out rather than import them; none may drift."""
    assert rv.BACKING_POLICIES == res.BACKING_POLICIES
    assert rv.SUPPORT_WITHDRAWN == res.SUPPORT_WITHDRAWN
    assert rv.SUPPORT_DISPROVEN == res.SUPPORT_DISPROVEN
    assert {res.SUPPORT_WITHDRAWN, res.SUPPORT_DISPROVEN} <= res.REASONS
    assert rv.BUDGET_SUPPORT_HOPS in ev.COVERAGE_REASONS
    assert rv.MARKER_KIND == write_time._REVALIDATE_KIND


# ── withdrawn support vs a disproven conclusion ──────────────────────────────


def test_retraction_without_replacement_is_uncertainty_not_the_opposite_claim(mem):
    """The acceptance case: a source retracted with nothing to replace it."""
    _write(mem, "insights/throughput.md", "# Throughput\n\nThroughput is 900 per hour.",
           status="retracted", type="Insight")
    # The opposing measurement exists and is perfectly current. It is not the
    # answer either: nothing elects it, and nothing may.
    _write(mem, "insights/counter.md", "# Counter\n\nThroughput is 400 per hour.",
           status="active", type="Insight")
    _write(mem, "decisions/capacity.md", "# Capacity\n\nPlan for 900 per hour.",
           status="active", type="Decision", backed_by=["insights/throughput"])

    check = _check(mem, "decisions/capacity.md")
    assert [(f.ref, f.hop, f.reason) for f in check.findings] == [
        ("insights/throughput", 1, rv.SUPPORT_WITHDRAWN)
    ]
    assert check.status == rv.STATUS_ADVISORY  # no declared policy: advisory

    result = ev.resolve_evidence([_seed_row(mem, "decisions/capacity.md")],
                                 mode="linked", now=_NOW)
    seed = result.seeds[0]
    resolution = res.resolve(seed.seed_meta, seed, now=_NOW)
    assert resolution.outcome == res.OUTCOME_INSUFFICIENT
    assert res.SUPPORT_WITHDRAWN in resolution.reasons
    assert res.SUPPORT_DISPROVEN not in resolution.reasons
    assert resolution.current is None
    assert "stale_backing:insights/throughput@1:support_withdrawn" in resolution.qualifiers
    # Neither the opposite claim nor the retired source's own value is served.
    assert "insights/counter" not in [s.ref for s in resolution.sides if s.ref]
    assert "400" not in str(resolution.to_dict())


def test_a_falsified_source_disproves_its_dependent_without_electing_anything(mem):
    """``falsified_by`` is the evidence that separates disproven from withdrawn."""
    _write(mem, "insights/refuted.md", "# Refuted\n\nThe cache is write-through.",
           status="retracted", falsified_by=["insights/measured"], type="Insight")
    _write(mem, "decisions/tuning.md", "# Tuning\n\nTune for a write-through cache.",
           status="active", type="Decision", backed_by=["insights/refuted"])

    check = _check(mem, "decisions/tuning.md")
    assert [f.reason for f in check.findings] == [rv.SUPPORT_DISPROVEN]
    assert check.disproven_refs() == frozenset({"insights/refuted"})

    result = ev.resolve_evidence([_seed_row(mem, "decisions/tuning.md")],
                                 mode="linked", now=_NOW)
    resolution = res.resolve(result.seeds[0].seed_meta, result.seeds[0], now=_NOW)
    assert resolution.outcome == res.OUTCOME_INSUFFICIENT
    assert res.SUPPORT_DISPROVEN in resolution.reasons
    assert res.SUPPORT_WITHDRAWN not in resolution.reasons
    assert resolution.current is None


def test_an_archived_source_withdraws_support_it_does_not_disprove_it(mem):
    _write(mem, "insights/parked.md", "# Parked\n\nA measurement, archived.",
           status="archived", type="Insight")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on the parked measurement.",
           status="active", type="Decision", backed_by=["insights/parked"])

    check = _check(mem, "decisions/rests.md")
    assert [(f.reason, f.detail) for f in check.findings] == [
        (rv.SUPPORT_WITHDRAWN, "status:archived")
    ]


def test_a_source_retired_by_location_is_withdrawn_support(mem):
    """A note under ``archive/`` keeps its frontmatter; the path is the statement."""
    _write(mem, "archive/old-note.md", "# Old\n\nStill says active.", status="active")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on the archived note.",
           status="active", type="Decision", backed_by=["archive/old-note"])

    check = _check(mem, "decisions/rests.md")
    assert [(f.ref, f.reason, f.detail) for f in check.findings] == [
        ("archive/old-note", rv.SUPPORT_WITHDRAWN, "path:archive")
    ]


def test_a_missing_source_is_coverage_never_a_withdrawal(mem):
    """Absence is not a retirement: a partial store must not read as one."""
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on a paper not in the store.",
           status="active", type="Decision", backed_by=["research/paper"])

    check = _check(mem, "decisions/rests.md")
    assert check.findings == ()
    assert check.reasons == (rv.TARGET_MISSING,)


def test_a_standing_source_supports_what_cites_it(mem):
    _write(mem, "insights/holds.md", "# Holds\n\nStill true.", status="active")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.",
           status="active", type="Decision", backed_by=["insights/holds"])

    result = ev.resolve_evidence([_seed_row(mem, "decisions/rests.md")],
                                 mode="linked", now=_NOW)
    resolution = res.resolve(result.seeds[0].seed_meta, result.seeds[0], now=_NOW)
    assert resolution.outcome == res.OUTCOME_SUPPORTED
    assert not [q for q in resolution.qualifiers if q.startswith("stale_backing")]


# ── all-of vs any-of vs the legacy advisory list ─────────────────────────────


@pytest.mark.parametrize(
    "policy,expected_status,expected_outcome",
    [
        ("all-of", rv.STATUS_UNSUPPORTED, res.OUTCOME_INSUFFICIENT),
        ("any-of", rv.STATUS_STILL_SUPPORTED, res.OUTCOME_SUPPORTED),
        (None, rv.STATUS_ADVISORY, res.OUTCOME_SUPPORTED),
    ],
)
def test_one_withdrawn_source_of_two_under_each_policy(
    mem, policy, expected_status, expected_outcome
):
    """Only an explicit policy lets the checker conclude from a multi-source list."""
    _write(mem, "insights/gone.md", "# Gone\n\nRetired.", status="archived")
    _write(mem, "insights/holds.md", "# Holds\n\nStanding.", status="active")
    extra = {"backing_policy": policy} if policy else {}
    _write(mem, "decisions/two.md", "# Two\n\nRests on both.", status="active",
           type="Decision", backed_by=["insights/gone", "insights/holds"], **extra)

    check = _check(mem, "decisions/two.md")
    assert check.policy == (policy or rv.POLICY_ADVISORY)
    assert check.status == expected_status
    # Whatever the policy, the withdrawal itself is always reported.
    assert [f.ref for f in check.findings] == ["insights/gone"]

    result = ev.resolve_evidence([_seed_row(mem, "decisions/two.md")],
                                 mode="linked", now=_NOW)
    seed = result.seeds[0]
    resolution = res.resolve(seed.seed_meta, seed, now=_NOW)
    assert resolution.outcome == expected_outcome
    # Advisory and any-of still stand, and still carry the qualification.
    assert "stale_backing:insights/gone@1:support_withdrawn" in resolution.qualifiers


def test_all_of_with_every_source_standing_is_the_only_fully_supported(mem):
    _write(mem, "insights/one.md", "# One\n\nStanding.", status="active")
    _write(mem, "insights/two.md", "# Two\n\nStanding.", status="active")
    _write(mem, "decisions/both.md", "# Both\n\nRests on both.", status="active",
           type="Decision", backing_policy="all-of",
           backed_by=["insights/one", "insights/two"])

    assert _check(mem, "decisions/both.md").status == rv.STATUS_FULLY_SUPPORTED


def test_any_of_falls_only_when_every_source_is_gone(mem):
    _write(mem, "insights/one.md", "# One\n\nGone.", status="archived")
    _write(mem, "insights/two.md", "# Two\n\nGone.", status="archived")
    _write(mem, "decisions/either.md", "# Either\n\nEither will do.", status="active",
           type="Decision", backing_policy="any-of",
           backed_by=["insights/one", "insights/two"])

    check = _check(mem, "decisions/either.md")
    assert check.status == rv.STATUS_UNSUPPORTED
    result = ev.resolve_evidence([_seed_row(mem, "decisions/either.md")],
                                 mode="linked", now=_NOW)
    resolution = res.resolve(result.seeds[0].seed_meta, result.seeds[0], now=_NOW)
    assert resolution.outcome == res.OUTCOME_INSUFFICIENT
    assert res.SUPPORT_WITHDRAWN in resolution.reasons


def test_an_unreadable_policy_falls_back_to_advisory(mem):
    """A policy that cannot be read must not make the checker more confident."""
    _write(mem, "insights/gone.md", "# Gone\n\nRetired.", status="archived")
    _write(mem, "insights/holds.md", "# Holds\n\nStanding.", status="active")
    _write(mem, "decisions/odd.md", "# Odd\n\nRests on both.", status="active",
           type="Decision", backing_policy=["all-of"],
           backed_by=["insights/gone", "insights/holds"])

    assert _check(mem, "decisions/odd.md").policy == rv.POLICY_ADVISORY


# ── the second hop ───────────────────────────────────────────────────────────


def test_retiring_the_root_qualifies_the_second_hop_record(mem):
    """The defect: a conclusion two hops out stayed unqualified until maintenance."""
    _write(mem, "research/root.md", "# Root\n\nThe original measurement.", status="active")
    _write(mem, "insights/middle.md", "# Middle\n\nRests on the root.",
           status="active", backed_by=["research/root"])
    _write(mem, "decisions/leaf.md", "# Leaf\n\nRests on the middle.",
           status="active", type="Decision", backed_by=["insights/middle"])
    _commit(mem)

    assert _check(mem, "decisions/leaf.md").findings == ()

    archive_memory("research/root.md", reason="withdrawn by its author")

    # Propagation reached the middle (one hop) and nothing else.
    assert _stale_refs(str(mem / "insights/middle.md")) == ["research/root"]
    assert _stale_refs(str(mem / "decisions/leaf.md")) == []

    check = _check(mem, "decisions/leaf.md")
    assert [(f.ref, f.hop, f.reason, f.via) for f in check.findings] == [
        ("research/root", 2, rv.SUPPORT_WITHDRAWN, "insights/middle")
    ]

    result = ev.resolve_evidence([_seed_row(mem, "decisions/leaf.md")],
                                 mode="linked", now=_NOW)
    resolution = res.resolve(result.seeds[0].seed_meta, result.seeds[0], now=_NOW)
    assert "stale_backing:research/root@2:support_withdrawn" in resolution.qualifiers
    # A second-hop loss qualifies; it does not by itself retire the leaf's own
    # declared support, which is one standing record.
    assert resolution.outcome == res.OUTCOME_SUPPORTED


def test_restoring_the_root_clears_the_second_hop_flag_without_rewriting_prose(mem):
    _write(mem, "research/root.md", "# Root\n\nThe original measurement.", status="active")
    _write(mem, "insights/middle.md", "# Middle\n\nRests on the root.",
           status="active", backed_by=["research/root"])
    leaf = _write(mem, "decisions/leaf.md", "# Leaf\n\nRests on the middle.",
                  status="active", type="Decision", backed_by=["insights/middle"])
    _commit(mem)
    before = open(leaf, encoding="utf-8").read()

    archive_memory("research/root.md", reason="parked")
    assert _check(mem, "decisions/leaf.md").findings != ()

    restore_memory("research/root.md", reason="needed again")

    assert _check(mem, "decisions/leaf.md").findings == ()
    # Nobody rewrote the leaf on the way out or the way back.
    assert open(leaf, encoding="utf-8").read() == before


def test_the_walk_stops_at_the_hop_bound_and_says_so(mem):
    _write(mem, "research/root.md", "# Root\n\nRetired.", status="archived")
    _write(mem, "insights/middle.md", "# Middle\n\nRests on the root.",
           status="active", backed_by=["research/root"])
    _write(mem, "decisions/leaf.md", "# Leaf\n\nRests on the middle.",
           status="active", backed_by=["insights/middle"])

    check = _check(mem, "decisions/leaf.md", max_hops=1)
    assert check.findings == ()
    assert rv.BUDGET_SUPPORT_HOPS in check.reasons


def test_a_cycle_terminates_and_is_reported_once(mem):
    _write(mem, "insights/a.md", "# A\n\nRests on b.", status="active",
           backed_by=["insights/b"])
    _write(mem, "insights/b.md", "# B\n\nRests on a.", status="archived",
           backed_by=["insights/a"])

    check = _check(mem, "insights/a.md", max_hops=8)
    assert [(f.ref, f.hop) for f in check.findings] == [("insights/b", 1)]
    assert rv.BUDGET_SUPPORT_HOPS not in check.reasons


def test_the_read_budget_stops_the_walk_and_names_itself(mem):
    for i in range(4):
        _write(mem, f"insights/s{i}.md", f"# S{i}\n\nRetired.", status="archived")
    _write(mem, "decisions/many.md", "# Many\n\nRests on four.", status="active",
           backed_by=[f"insights/s{i}" for i in range(4)])

    check = _check(mem, "decisions/many.md", max_reads=2)
    assert len(check.findings) == 2
    assert rv.BUDGET_SUPPORT_HOPS in check.reasons


def test_the_budget_reason_reaches_the_seed_coverage(mem):
    _write(mem, "research/root.md", "# Root\n\nRetired.", status="archived")
    _write(mem, "insights/middle.md", "# Middle\n\nRests on the root.",
           status="active", backed_by=["research/root"])
    _write(mem, "decisions/leaf.md", "# Leaf\n\nRests on the middle.",
           status="active", backed_by=["insights/middle"])

    budget = ev.EvidenceBudget(max_support_hops=1)
    result = ev.resolve_evidence([_seed_row(mem, "decisions/leaf.md")],
                                 mode="linked", budget=budget, now=_NOW)
    assert rv.BUDGET_SUPPORT_HOPS in result.seeds[0].coverage()["reasons"]
    assert set(result.seeds[0].reasons) <= ev.COVERAGE_REASONS


# ── revalidation receipts ────────────────────────────────────────────────────


def test_a_receipt_counts_only_against_the_sources_current_revision(mem):
    source = _write(mem, "insights/source.md", "# Source\n\nMeasured at 900.",
                    status="active")
    stale_revision = _revision(source)
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on the source.",
           status="active", backed_by=["insights/source"],
           revalidated=[rv.build_revalidation("insights/source", stale_revision,
                                              at=_NOW)])

    assert _check(mem, "decisions/rests.md").findings == ()

    # The source is edited. The receipt now names a revision that is not the
    # one on disk, and the dependent says so.
    _write(mem, "insights/source.md", "# Source\n\nRemeasured at 1200.", status="active")
    check = _check(mem, "decisions/rests.md")
    assert [(f.ref, f.reason) for f in check.findings] == [
        ("insights/source", rv.SUPPORT_REVISION_CHANGED)
    ]
    # A moved source is a prompt to re-verify, never a withdrawal: the record
    # still stands, carrying the qualification.
    result = ev.resolve_evidence([_seed_row(mem, "decisions/rests.md")],
                                 mode="linked", now=_NOW)
    resolution = res.resolve(result.seeds[0].seed_meta, result.seeds[0], now=_NOW)
    assert resolution.outcome == res.OUTCOME_SUPPORTED
    assert any(q.startswith("stale_backing:insights/source@1:support_revision_changed")
               for q in resolution.qualifiers)


def test_a_record_with_no_receipt_is_never_reported_as_changed(mem):
    """Nothing was recorded, so nothing can have changed since. Unknown stays unknown."""
    _write(mem, "insights/source.md", "# Source\n\nEdited many times.", status="active")
    _write(mem, "decisions/legacy.md", "# Legacy\n\nA record from before receipts.",
           status="active", backed_by=["insights/source"])

    assert _check(mem, "decisions/legacy.md").findings == ()


def test_a_receipt_never_revives_a_withdrawn_source(mem):
    _write(mem, "insights/source.md", "# Source\n\nRetired.", status="archived")
    revision = _revision(str(mem / "insights/source.md"))
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.", status="active",
           backed_by=["insights/source"],
           revalidated=[rv.build_revalidation("insights/source", revision, at=_NOW)])

    assert [f.reason for f in _check(mem, "decisions/rests.md").findings] == [
        rv.SUPPORT_WITHDRAWN
    ]


def test_recording_a_receipt_is_idempotent_per_source_revision(mem):
    path = _write(mem, "decisions/rests.md", "# Rests\n\nBody.", status="active",
                  backed_by=["insights/source"])
    content = open(path, encoding="utf-8").read()

    first = rv.merge_revalidation_into_content(
        content, rv.build_revalidation("insights/source", "abc123", at=_NOW))
    assert rv.revalidated_revisions(frontmatter.loads(first).metadata) == {
        "insights/source": "abc123"
    }
    # Same source revision → no new entry, and byte-identical content.
    again = rv.merge_revalidation_into_content(
        first, rv.build_revalidation("insights/source", "abc123", at=_NOW))
    assert again == first
    # A new revision replaces that source's entry rather than piling up.
    moved = rv.merge_revalidation_into_content(
        first, rv.build_revalidation("insights/source", "def456", at=_NOW))
    assert len(rv.parse_revalidations(frontmatter.loads(moved).metadata)) == 1
    assert rv.revalidated_revisions(frontmatter.loads(moved).metadata) == {
        "insights/source": "def456"
    }
    # The body is never touched by a receipt.
    assert frontmatter.loads(moved).content.strip() == "# Rests\n\nBody."


def test_a_malformed_receipt_certifies_nothing(mem):
    """A receipt naming no revision is not a receipt."""
    meta = {"revalidated": [{"ref": "insights/source"}, "nonsense",
                            {"revision": "abc"}, {"ref": "x/y", "revision": "def"}]}
    assert rv.revalidated_revisions(meta) == {"x/y": "def"}


# ── the persisted half: the marker, the applier, git ─────────────────────────


def test_the_applier_records_findings_through_git_without_touching_prose(mem):
    _write(mem, "insights/source.md", "# Source\n\nRetired.", status="archived")
    dep = _write(mem, "decisions/rests.md", "# Rests\n\nThe body nobody may rewrite.",
                 status="active", type="Decision", backed_by=["insights/source"])
    _commit(mem)

    stats = rv.apply_revalidation(dep, now=_NOW)

    assert stats == {"flagged": 1, "cleared": 0, "skipped": 0}
    entry = propagate.parse_stale_backing(_meta(dep))[0]
    assert entry["ref"] == "insights/source"
    assert entry["op"] == rv.REVALIDATE_CHECK_OP
    assert entry["hop"] == 1
    assert entry["reason"].startswith(rv.SUPPORT_WITHDRAWN)
    assert frontmatter.load(dep).content.strip() == "# Rests\n\nThe body nobody may rewrite."
    assert _meta(dep)["status"] == "active"  # never retired by a check

    # Markdown and git reconstruct it: one commit, the file in it, nothing dirty.
    log = subprocess.run(["git", "-C", str(mem), "log", "--format=%s"],
                         capture_output=True, text=True, check=True).stdout
    assert "backing revalidation: decisions/rests" in log.splitlines()[0]
    dirty = subprocess.run(["git", "-C", str(mem), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout
    assert "decisions/rests.md" not in dirty

    # Idempotent: a second application of the same state writes nothing.
    assert rv.apply_revalidation(dep, now=_NOW) == {"flagged": 0, "cleared": 0, "skipped": 0}


def test_a_matching_revision_receipt_clears_the_flag_a_formatting_resave_does_not(mem):
    """The convention becomes evidence: only a revision-matched receipt clears."""
    source = _write(mem, "insights/source.md",
                    "# Source\n\n- [2026-01-01] Throughput is 900 <!-- fact:f1 -->\n",
                    status="active", type="Insight")
    dep = _write(mem, "decisions/rests.md", "# Rests\n\nPlan for 900.",
                 status="active", type="Decision", backed_by=["insights/source"])
    _commit(mem)

    # The real executor retires a fact in the source; propagation flags the
    # dependent exactly as it does today.
    apply_operations(source, [{"op": "SUPERSEDE", "id": "f1",
                               "new_text": "Throughput is 1200", "reason": "remeasured"}])
    assert _stale_refs(dep) == ["insights/source"]

    # A formatting-only re-save of the dependent certifies nothing: the flag
    # survives the applier, because the dependent's bytes say nothing about
    # the source.
    body = open(dep, encoding="utf-8").read()
    with open(dep, "w", encoding="utf-8") as fh:
        fh.write(body.replace("Plan for 900.", "Plan for 900.  \n"))
    assert rv.apply_revalidation(dep, now=_NOW)["cleared"] == 0
    assert _stale_refs(dep) == ["insights/source"]

    # A receipt naming a *stale* revision does not clear it either.
    with open(dep, encoding="utf-8") as fh:
        content = fh.read()
    with open(dep, "w", encoding="utf-8") as fh:
        fh.write(rv.merge_revalidation_into_content(
            content, rv.build_revalidation("insights/source", "0" * 64, at=_NOW)))
    assert rv.apply_revalidation(dep, now=_NOW)["cleared"] == 0
    assert _stale_refs(dep) == ["insights/source"]

    # A receipt naming the source's current revision does.
    with open(dep, encoding="utf-8") as fh:
        content = fh.read()
    with open(dep, "w", encoding="utf-8") as fh:
        fh.write(rv.merge_revalidation_into_content(
            content, rv.build_revalidation("insights/source", _revision(source), at=_NOW)))
    assert rv.apply_revalidation(dep, now=_NOW)["cleared"] == 1
    assert _stale_refs(dep) == []
    assert "stale_backing" not in _meta(dep)
    assert frontmatter.load(dep).content.strip().startswith("# Rests")


def test_a_source_change_while_the_dependent_is_archived_mutates_nothing(mem):
    _write(mem, "insights/source.md", "# Source\n\nStanding.", status="active")
    dep = _write(mem, "decisions/parked.md", "# Parked\n\nRests on the source.",
                 status="active", type="Decision", backed_by=["insights/source"])
    _commit(mem)

    assert archive_memory("decisions/parked.md", reason="parked")["status"] == "archived"
    archive_memory("insights/source.md", reason="withdrawn")

    # Propagation skips archived dependents, and so does the applier: an
    # archived record asserts nothing, so there is nothing to qualify.
    before = open(dep, encoding="utf-8").read()
    assert rv.apply_revalidation(dep, now=_NOW) == {"flagged": 0, "cleared": 0, "skipped": 1}
    assert open(dep, encoding="utf-8").read() == before

    restore_memory("decisions/parked.md", reason="needed again")

    # On the way back the check runs and qualifies it. (The restore's own
    # re-check already flags this one; the applier is idempotent per ref.)
    assert _stale_refs(dep) == ["insights/source"]
    assert [f.reason for f in _check(mem, "decisions/parked.md").findings] == [
        rv.SUPPORT_WITHDRAWN
    ]
    assert rv.apply_revalidation(dep, now=_NOW)["flagged"] == 0


def test_a_resave_that_drops_the_flag_does_not_stop_the_check(mem):
    """The filed defect: a re-save cleared the flag as if the source were verified."""
    _write(mem, "insights/source.md", "# Source\n\nRetired.", status="archived")
    dep = _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.", status="active",
                 type="Decision", backed_by=["insights/source"],
                 stale_backing=[{"ref": "insights/source", "op": "archive"}])
    _commit(mem)

    # What every save surface does: rebuild the frontmatter from its inputs,
    # which drops the flag.
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.   ", status="active",
           type="Decision", backed_by=["insights/source"])
    assert _stale_refs(dep) == []

    # The source is still retired, so the read-time check still says so, and
    # the applier records it again.
    assert [f.reason for f in _check(mem, "decisions/rests.md").findings] == [
        rv.SUPPORT_WITHDRAWN
    ]
    assert rv.apply_revalidation(dep, now=_NOW)["flagged"] == 1
    assert _stale_refs(dep) == ["insights/source"]


def test_the_sweep_enqueues_a_marker_the_worker_applies(mem):
    """Persistent work rides the existing marker queue, not a new path."""
    _write(mem, "insights/source.md", "# Source\n\nRetired.", status="archived")
    dep = _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.",
                 status="active", type="Decision", backed_by=["insights/source"])
    _commit(mem)

    enqueued = rv.sweep_revalidations(str(mem), now=_NOW)
    assert enqueued == ["decisions/rests.md"]

    pending = sorted((mem / ".palinode" / "pending").glob("*.json"))
    assert len(pending) == 1
    import json

    job = json.loads(pending[0].read_text())
    assert job["item"]["kind"] == rv.MARKER_KIND
    assert job["item"]["findings"][0]["reason"] == rv.SUPPORT_WITHDRAWN
    assert os.path.realpath(job["file_path"]) == os.path.realpath(dep)

    # The worker's own entry point applies it — deterministically, no LLM.
    result = write_time._run_check_and_apply(job["file_path"], job["item"])
    assert result["operations"] == []
    assert result["applied_stats"] == {"flagged": 1, "cleared": 0, "skipped": 0}
    assert _stale_refs(dep) == ["insights/source"]

    # Nothing left to do: the sweep does not re-enqueue a recorded finding.
    assert rv.sweep_revalidations(str(mem), now=_NOW) == []


def test_a_marker_for_a_vanished_target_is_reported_not_swallowed(mem):
    stats = rv.apply_revalidation(str(mem / "decisions/gone.md"),
                                  {"kind": rv.MARKER_KIND}, now=_NOW)
    assert stats == {"flagged": 0, "cleared": 0, "skipped": 1}


def test_the_sweep_skips_archived_records_and_records_with_no_backing(mem):
    _write(mem, "insights/source.md", "# Source\n\nRetired.", status="archived")
    _write(mem, "decisions/parked.md", "# Parked\n\nRests on it.", status="archived",
           backed_by=["insights/source"])
    _write(mem, "decisions/plain.md", "# Plain\n\nCites nothing.", status="active")

    assert rv.sweep_revalidations(str(mem), now=_NOW) == []


# ── the cache: per request, and only under the reuse contract ────────────────


def _view(ref: str, revision: str, **meta) -> rv.SourceView:
    return rv.SourceView.of(ref, meta, revision=revision, now=_NOW)


def test_a_supporting_revision_change_inside_one_request_is_not_served_warm():
    """Seed, resolve, edit the source through the seam, resolve again."""
    state = {"insights/source": _view("insights/source", "rev-1", status="active")}

    def _read(ref: str):
        return state.get(rv.normalize_ref(ref))

    cache = rv.SupportCache()
    meta = {"backed_by": ["insights/source"]}
    first = rv.resolve_support("decisions/rests", meta, read=_read, now=_NOW,
                               revision="dep-1", cache=cache)
    assert first.findings == ()
    assert cache.misses == 1

    # The source is archived mid-request. The warm entry names its old
    # revision, so it cannot be reused — and the new answer is the true one.
    state["insights/source"] = _view("insights/source", "rev-2", status="archived")
    second = rv.resolve_support("decisions/rests", meta, read=_read, now=_NOW,
                                revision="dep-1", cache=cache)
    assert [f.reason for f in second.findings] == [rv.SUPPORT_WITHDRAWN]
    assert cache.hits == 0 and cache.misses == 2

    # Unchanged state is served warm — the cache is a cache.
    third = rv.resolve_support("decisions/rests", meta, read=_read, now=_NOW,
                               revision="dep-1", cache=cache)
    assert third is second
    assert cache.hits == 1


def test_an_unchanged_file_across_the_noon_boundary_is_re_evaluated():
    """Warm at 11:59 with an ``expires_at`` at 12:00; consume at 12:01."""
    noon = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)
    before = datetime(2026, 9, 12, 11, 59, 0, tzinfo=UTC)
    after = datetime(2026, 9, 12, 12, 1, 0, tzinfo=UTC)
    source = {"status": "active", "expires_at": noon.isoformat()}

    def _read(ref: str):
        # The bytes never change: same revision at 11:59 and at 12:01.
        return rv.SourceView.of("insights/grant", source, revision="rev-1",
                                now=_pinned_clock[0])

    _pinned_clock = [before]
    cache = rv.SupportCache()
    meta = {"backed_by": ["insights/grant"]}

    warm = rv.resolve_support("decisions/acting", meta, read=_read, now=before,
                              revision="dep-1", cache=cache)
    assert warm.findings == ()  # the grant is in force at 11:59

    _pinned_clock[0] = after
    cold = rv.resolve_support("decisions/acting", meta, read=_read, now=after,
                              revision="dep-1", cache=cache)
    assert [(f.reason, f.detail) for f in cold.findings] == [
        (rv.SUPPORT_WITHDRAWN, "expired")
    ]
    assert cache.hits == 0  # the window differs; nothing was served warm

    # The two keys differ for exactly that reason: same scope, same policy,
    # same revisions, different window.
    warm_key = cache.key_for(warm, now=before, scope=(), policy_version="p/1")
    cold_key = cache.key_for(cold, now=after, scope=(), policy_version="p/1")
    assert warm_key != cold_key
    assert reuse_key(scope=(), query_scope={"ref": "decisions/acting"},
                     policy_version="p/1", revisions=cold.inputs,
                     window=(None, noon.isoformat())) != reuse_key(
        scope=(), query_scope={"ref": "decisions/acting"}, policy_version="p/1",
        revisions=cold.inputs, window=(noon.isoformat(), None))

    # Historical evidence stays readable: the expired grant is still a record
    # with content, not a hole.
    assert _read("insights/grant").meta["expires_at"] == noon.isoformat()


def test_cached_evidence_never_crosses_a_caller_or_policy_boundary():
    state = {"insights/source": _view("insights/source", "rev-1", status="active")}

    def _read(ref: str):
        return state.get(rv.normalize_ref(ref))

    cache = rv.SupportCache()
    meta = {"backed_by": ["insights/source"]}
    rv.resolve_support("decisions/rests", meta, read=_read, now=_NOW, revision="dep-1",
                       cache=cache, scope=("project/alpha",), policy_version="p/1")

    # A different caller scope is a different delivery, warm entry or not.
    assert cache.get("decisions/rests", read=_read, now=_NOW,
                     scope=("project/beta",), policy_version="p/1") is None
    # So is a policy version the config moved under.
    assert cache.get("decisions/rests", read=_read, now=_NOW,
                     scope=("project/alpha",), policy_version="p/2") is None
    # The original caller, unchanged, is served.
    assert cache.get("decisions/rests", read=_read, now=_NOW,
                     scope=("project/alpha",), policy_version="p/1") is not None


def test_an_unknown_revision_is_never_shown_to_still_match():
    state = {"insights/source": _view("insights/source", None, status="active")}

    def _read(ref: str):
        return state.get(rv.normalize_ref(ref))

    cache = rv.SupportCache()
    rv.resolve_support("decisions/rests", {"backed_by": ["insights/source"]},
                       read=_read, now=_NOW, revision="dep-1", cache=cache)
    assert cache.get("decisions/rests", read=_read, now=_NOW) is None


def test_the_policy_version_of_a_real_delivery_is_the_receipt_one():
    """The cache keys on the same policy identity a delivery receipt records."""
    current = PolicyVersion.current()
    cache = rv.SupportCache()
    check = rv.check_support("decisions/rests", {}, read=lambda ref: None, now=_NOW)
    assert cache.key_for(check, now=_NOW, scope=(), policy_version=current) == \
        cache.key_for(check, now=_NOW, scope=(), policy_version=current.as_str())


def test_a_hidden_source_contributes_coverage_and_nothing_else(mem):
    """A source the requester may not see is never reported as a retirement."""
    _write(mem, "insights/secret.md", "# Secret\n\nRetired, and not yours to see.",
           status="archived", visibility="private")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on something.",
           status="active", backed_by=["insights/secret"])

    from palinode.core.scope import ScopeChain

    result = ev.resolve_evidence([_seed_row(mem, "decisions/rests.md")], mode="linked",
                                 chain=ScopeChain(project="project/other"), now=_NOW)
    seed = result.seeds[0]
    check = seed.support_checks["decisions/rests"]
    assert check.findings == ()
    assert ev.TARGET_HIDDEN in seed.reasons
    resolution = res.resolve(seed.seed_meta, seed, now=_NOW)
    assert res.COVERAGE_PARTIAL in resolution.reasons
    # The seed's own frontmatter names the ref, and always did. What the
    # hidden record *says* — its title, its body, its retirement — does not
    # reach the requester through the check.
    assert "not yours to see" not in str(resolution.to_dict())
    assert not [q for q in resolution.qualifiers if q.startswith("stale_backing")]


def test_the_evidence_layer_checks_a_replacement_that_could_stand(mem):
    """The record that would stand must have standing backing of its own."""
    _write(mem, "insights/gone.md", "# Gone\n\nRetired.", status="archived")
    _write(mem, "decisions/new.md", "# New\n\nThe replacement.", status="active",
           type="Decision", backed_by=["insights/gone"])
    _write(mem, "decisions/old.md", "# Old\n\nThe replaced record.", status="active",
           type="Decision", superseded_by="decisions/new")

    result = ev.resolve_evidence([_seed_row(mem, "decisions/old.md")],
                                 mode="linked", now=_NOW)
    seed = result.seeds[0]
    assert "decisions/new" in seed.support_checks
    resolution = res.resolve(seed.seed_meta, seed, now=_NOW)
    # Not the replacement (its support is gone) and emphatically not the
    # record it replaced.
    assert resolution.outcome == res.OUTCOME_INSUFFICIENT
    assert res.SUPPORT_WITHDRAWN in resolution.reasons
    assert resolution.current is None


def test_the_support_check_adds_no_field_to_the_evidence_payload(mem):
    """The qualification travels in the resolution block, not a new key."""
    _write(mem, "insights/gone.md", "# Gone\n\nRetired.", status="archived")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.", status="active",
           backed_by=["insights/gone"])

    result = ev.resolve_evidence([_seed_row(mem, "decisions/rests.md")],
                                 mode="linked", now=_NOW)
    assert set(result.seeds[0].to_dict()) == {
        "replacements", "conflicts", "support", "discovered", "seed_freshness",
        "coverage",
    }


def test_the_check_reads_nothing_beyond_the_request_file_budget(mem):
    """Its reads are charged to ``max_files`` like every other expansion."""
    for i in range(3):
        _write(mem, f"insights/s{i}.md", f"# S{i}\n\nRetired.", status="archived")
    _write(mem, "decisions/many.md", "# Many\n\nRests on three.", status="active",
           backed_by=[f"insights/s{i}" for i in range(3)])

    budget = ev.EvidenceBudget(max_files=1, max_edges=0)
    result = ev.resolve_evidence([_seed_row(mem, "decisions/many.md")],
                                 mode="linked", budget=budget, now=_NOW)
    seed = result.seeds[0]
    assert ev.BUDGET_FILES in seed.reasons
    assert len(seed.support_checks["decisions/many"].findings) <= 1


# ── lint reports what the check found ────────────────────────────────────────


def test_lint_reports_a_second_hop_finding_propagation_never_wrote(mem):
    from palinode.core.lint import run_lint_pass

    _write(mem, "research/root.md", "# Root\n\nRetired.", status="archived")
    _write(mem, "insights/middle.md", "# Middle\n\nRests on the root.",
           status="active", backed_by=["research/root"])
    _write(mem, "decisions/leaf.md", "# Leaf\n\nRests on the middle.",
           status="active", backed_by=["insights/middle"])

    findings = {f["file"]: f for f in run_lint_pass()["stale_backing"]}
    leaf = findings[os.path.join("decisions", "leaf.md")]
    assert [e["ref"] for e in leaf["stale_backing"]] == ["research/root"]
    assert leaf["stale_backing"][0]["op"] == rv.REVALIDATE_CHECK_OP
    assert leaf["stale_backing"][0]["hop"] == 2
    assert leaf["support"]["status"] == rv.STATUS_ADVISORY
    assert leaf["support"]["findings"][0]["reason"] == rv.SUPPORT_WITHDRAWN


def test_lint_does_not_double_report_a_ref_already_flagged(mem):
    from palinode.core.lint import run_lint_pass

    _write(mem, "insights/source.md", "# Source\n\nRetired.", status="archived")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on it.", status="active",
           backed_by=["insights/source"],
           stale_backing=[{"ref": "insights/source", "op": "archive"}])

    findings = {f["file"]: f for f in run_lint_pass()["stale_backing"]}
    entries = findings[os.path.join("decisions", "rests.md")]["stale_backing"]
    assert [(e["ref"], e["op"]) for e in entries] == [("insights/source", "archive")]


def test_lint_reports_nothing_for_a_store_whose_backing_all_stands(mem):
    from palinode.core.lint import run_lint_pass

    _write(mem, "insights/source.md", "# Source\n\nStanding.", status="active")
    _write(mem, "decisions/rests.md", "# Rests\n\nRests on it, and on a paper.",
           status="active", backed_by=["insights/source", "research/paper"])

    assert run_lint_pass()["stale_backing"] == []
