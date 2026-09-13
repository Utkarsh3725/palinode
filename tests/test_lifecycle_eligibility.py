"""``palinode.core.lifecycle`` — the one eligibility classifier recall and
consolidation input share.

Pure-function coverage: every existing lifecycle signal maps to the right
state and reason; qualifiers ride along; the injected clock decides
``expires_at``; the ordering fallback never treats a file touch as a date;
and the retired-fact recognizer agrees with what the executor actually
writes (pinned against the executor's real output, so the writer and this
reader cannot drift without a test saying so).
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from palinode.core.config import config
from palinode.core.lifecycle import (
    ARCHIVE_SEGMENT,
    EFFECTIVE_DATE_FIELDS,
    RETIRED_STATUSES,
    Eligibility,
    effective_at,
    eligibility,
    is_retired_fact_text,
    order_key,
)

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
PAST = (NOW - timedelta(hours=1)).isoformat()
FUTURE = (NOW + timedelta(days=30)).isoformat()


# ── states ───────────────────────────────────────────────────────────────────


def test_unmarked_record_is_usable_and_stays_unmarked():
    e = eligibility({"type": "Decision", "title": "legacy"}, now=NOW)
    assert e.state == "unmarked"
    assert e.reason == "unmarked"
    assert e.usable and not e.retired
    assert e.epistemic is None  # absent epistemic is not promoted to anything


def test_active_status_is_current():
    e = eligibility({"status": "active"}, now=NOW)
    assert e.state == "current" and e.reason == "status:active"


@pytest.mark.parametrize("value", sorted(RETIRED_STATUSES))
def test_retired_statuses_retire(value):
    e = eligibility({"status": value}, now=NOW)
    assert e.retired and e.reason == f"status:{value}"


def test_status_is_case_insensitive():
    assert eligibility({"status": "Archived"}, now=NOW).reason == "status:archived"


def test_ku_lifecycle_field_mirrors_status():
    assert eligibility({"lifecycle": "deprecated"}, now=NOW).reason == "lifecycle:deprecated"
    assert eligibility({"lifecycle": "active"}, now=NOW).state == "current"


def test_incident_statuses_are_current_not_retired():
    for value in ("open", "monitoring", "resolved"):
        assert eligibility({"status": value}, now=NOW).state == "current"


def test_superseded_by_retires_even_with_active_status():
    """An explicit replacement outranks whatever status the record still claims."""
    e = eligibility(
        {"status": "active", "superseded_by": "decisions/current"}, now=NOW
    )
    assert e.retired and e.reason == "superseded_by"
    assert e.superseded_by == "decisions/current"


def test_empty_superseded_by_is_not_a_replacement():
    assert eligibility({"superseded_by": ""}, now=NOW).state == "unmarked"


def test_archived_in_place_under_decisions_is_retired():
    """The recall probe's case: ``status: archived`` while still under ``decisions/``."""
    e = eligibility(
        {"type": "Decision", "status": "archived", "superseded_by": "decisions/b"},
        path="decisions/a.md", now=NOW,
    )
    assert e.retired and e.reason == "status:archived"


def test_non_dict_metadata_is_unmarked():
    assert eligibility(None, now=NOW).state == "unmarked"  # type: ignore[arg-type]
    assert eligibility([], now=NOW).state == "unmarked"  # type: ignore[arg-type]


# ── retired by location (``archive/``) ───────────────────────────────────────


@pytest.mark.parametrize("rel", ["archive/2026/2026-03-01.md", "archive/x.md"])
def test_a_record_under_archive_is_retired_by_its_path(rel):
    """The weekly pass moves a note and leaves its bytes alone, so its
    location is the only thing that says it was retired."""
    e = eligibility({"type": "Note", "title": "a March daily note"}, path=rel, now=NOW)
    assert e.retired and e.reason == f"path:{ARCHIVE_SEGMENT}"


def test_the_path_outranks_frontmatter_that_still_claims_active():
    e = eligibility({"status": "active"}, path="archive/2026/x.md", now=NOW)
    assert e.retired and e.reason == "path:archive"
    assert eligibility({"lifecycle": "active"}, path="archive/x.md", now=NOW).retired


def test_a_declared_retirement_under_archive_keeps_its_own_reason():
    """Same verdict either way; the declaration is the more specific answer to why."""
    assert eligibility({"status": "archived"}, path="archive/x.md", now=NOW).reason \
        == "status:archived"
    assert eligibility({"status": "active", "expires_at": PAST}, path="archive/x.md", now=NOW).reason \
        == "expired"


def test_the_path_outranks_superseded_by():
    e = eligibility({"superseded_by": "decisions/b"}, path="archive/x.md", now=NOW)
    assert e.retired and e.reason == "path:archive"
    assert e.superseded_by == "decisions/b"  # the pointer still rides along


@pytest.mark.parametrize(
    "rel",
    [
        "archives/x.md",
        "x/my-archive.md",
        "my-archive/x.md",
        "archive.md",
        "projects/archive.md",
        "daily/2026-03-01.md",
    ],
)
def test_only_a_whole_archive_directory_segment_counts(rel):
    assert eligibility({}, path=rel, now=NOW).state == "unmarked"


def test_an_absolute_path_is_read_relative_to_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    assert eligibility({}, path=str(tmp_path / "archive" / "2026" / "x.md"), now=NOW).reason \
        == "path:archive"
    assert eligibility({}, path=str(tmp_path / "daily" / "x.md"), now=NOW).state == "unmarked"


def test_a_store_living_under_an_archive_directory_is_not_wholly_retired(tmp_path, monkeypatch):
    """Reading the path as memory-relative first is what keeps the rule off the store root."""
    root = tmp_path / "archive" / "store"
    monkeypatch.setattr(config, "memory_dir", str(root))
    assert eligibility({}, path=str(root / "projects" / "demo.md"), now=NOW).state == "unmarked"
    assert eligibility({}, path=str(root / "archive" / "2026" / "x.md"), now=NOW).reason \
        == "path:archive"


def test_without_a_path_the_record_is_judged_by_frontmatter_alone():
    assert eligibility({}, now=NOW).state == "unmarked"
    assert eligibility({}, path="", now=NOW).state == "unmarked"
    assert eligibility({"status": "active"}, now=NOW).state == "current"


# ── expiry with the injected clock ───────────────────────────────────────────


def test_expired_acting_state_is_retired_by_the_injected_clock():
    e = eligibility({"core": True, "expires_at": PAST}, now=NOW)
    assert e.retired and e.reason == "expired"
    assert e.expires_at == PAST


def test_unexpired_record_keeps_its_declared_state():
    assert eligibility({"status": "active", "expires_at": FUTURE}, now=NOW).state == "current"
    assert eligibility({"expires_at": FUTURE}, now=NOW).state == "unmarked"


def test_expiry_is_at_or_before_now():
    assert eligibility({"expires_at": NOW.isoformat()}, now=NOW).retired


def test_malformed_expires_at_never_retires():
    """Same rule as the acting-state gate: a typo cannot silently disarm a record."""
    e = eligibility({"status": "active", "expires_at": "next tuesday"}, now=NOW)
    assert e.state == "current"


def test_expiry_outranks_a_current_status():
    assert eligibility({"status": "active", "expires_at": PAST}, now=NOW).reason == "expired"


# ── qualifiers ───────────────────────────────────────────────────────────────


def test_qualifiers_ride_along_on_a_usable_record():
    e = eligibility(
        {
            "type": "ProjectSnapshot",
            "contradicts": ["decisions/current-endpoint"],
            "stale_backing": [{"ref": "decisions/retired-endpoint", "op": "archive"}],
            "epistemic": "unverified",
        },
        now=NOW,
    )
    assert e.usable
    assert e.contradicts == ("decisions/current-endpoint",)
    assert e.stale_backing == ("decisions/retired-endpoint",)
    assert e.epistemic == "unverified"
    assert e.qualified


def test_qualifiers_soft_fail_on_malformed_shapes():
    e = eligibility(
        {
            "contradicts": "decisions/one",  # a bare string is one ref
            "stale_backing": [{"no_ref": 1}, "junk", {"ref": " decisions/two "}],
            "epistemic": "  ",
        },
        now=NOW,
    )
    assert e.contradicts == ("decisions/one",)
    assert e.stale_backing == ("decisions/two",)
    assert e.epistemic is None
    assert eligibility({"contradicts": {"not": "a list"}}, now=NOW).contradicts == ()


def test_unqualified_record_reports_not_qualified():
    assert not eligibility({"status": "active"}, now=NOW).qualified


def test_result_is_frozen():
    e = eligibility({}, now=NOW)
    with pytest.raises(AttributeError):
        e.state = "retired"  # type: ignore[misc]
    assert isinstance(e, Eligibility)


# ── ordering fallback ────────────────────────────────────────────────────────


def test_effective_date_field_order():
    assert EFFECTIVE_DATE_FIELDS == ("date", "last_updated", "created_at")


def test_declared_date_wins_over_save_stamps():
    meta = {
        "date": "2026-01-01",
        "last_updated": "2026-06-01T00:00:00Z",
        "created_at": "2026-03-01T00:00:00Z",
    }
    assert effective_at(meta) == datetime(2026, 1, 1, tzinfo=UTC)


def test_last_updated_then_created_at():
    assert effective_at({"last_updated": "2026-06-01T00:00:00Z", "created_at": "2026-03-01T00:00:00Z"}) \
        == datetime(2026, 6, 1, tzinfo=UTC)
    assert effective_at({"created_at": "2026-03-01T00:00:00Z"}) == datetime(2026, 3, 1, tzinfo=UTC)


def test_yaml_bare_date_and_datetime_values_parse():
    assert effective_at({"date": date(2026, 9, 10)}) == datetime(2026, 9, 10, tzinfo=UTC)
    assert effective_at({"created_at": datetime(2026, 9, 10, 8, 0)}) == datetime(2026, 9, 10, 8, 0, tzinfo=UTC)


def test_malformed_date_falls_through_to_the_next_field():
    assert effective_at({"date": "sometime", "created_at": "2026-03-01T00:00:00Z"}) \
        == datetime(2026, 3, 1, tzinfo=UTC)


def test_undated_record_has_no_effective_moment():
    assert effective_at({"title": "x"}) is None
    assert effective_at({"date": 20260910}) is None


def test_order_key_dated_ranks_above_undated_whatever_the_mtime():
    """A file touch is not a new effective decision."""
    dated = eligibility({"created_at": "2020-01-01T00:00:00Z"}, now=NOW)
    undated = eligibility({}, now=NOW)
    assert order_key(dated, mtime=0.0) > order_key(undated, mtime=9e12)


def test_order_key_newest_dated_first_and_mtime_only_breaks_undated_ties():
    older = eligibility({"created_at": "2026-01-01T00:00:00Z"}, now=NOW)
    newer = eligibility({"created_at": "2026-02-01T00:00:00Z"}, now=NOW)
    assert order_key(newer, mtime=1.0) > order_key(older, mtime=9e12)
    undated = eligibility({}, now=NOW)
    assert order_key(undated, mtime=2.0) > order_key(undated, mtime=1.0)


# ── retired-fact recognizer, pinned to the executor's real output ────────────


def test_recognizer_matches_what_the_executor_writes(tmp_path, monkeypatch):
    from palinode.consolidation import executor

    monkeypatch.setattr("palinode.core.config.config.git.auto_commit", False)
    target = tmp_path / "projects" / "demo.md"
    target.parent.mkdir()
    body = (
        "- Use endpoint A. <!-- fact:endpoint -->\n"
        "- The sky is green. <!-- fact:sky -->\n"
    )
    superseded = executor._supersede_fact(
        body, "endpoint", "Use endpoint B.", "Explicit replacement", str(target)
    )
    retracted = executor._retract_fact(
        superseded, "sky", "never true", str(target)
    )
    from palinode.consolidation.fact_ids import FACT_LINE_RE

    harvested = {
        m.group(2): m.group(1).strip() for m in FACT_LINE_RE.finditer(retracted)
    }
    assert set(harvested) == {"endpoint", "supersedes-endpoint", "sky"}
    assert is_retired_fact_text(harvested["endpoint"])          # ~~…~~ [superseded YYYY-MM-DD]
    assert is_retired_fact_text(harvested["sky"])               # ~~…~~ [RETRACTED YYYY-MM-DD — reason]
    assert not is_retired_fact_text(harvested["supersedes-endpoint"])


def test_recognizer_matches_the_mention_level_retract_marker():
    assert is_retired_fact_text("~~Alice lives in Paris~~ [RETRACTED 2026-09-10 r:0badc0de].")


def test_recognizer_is_conservative_about_user_strikethrough():
    """Only Palinode's own marker retires a fact; prose strikethrough does not."""
    assert not is_retired_fact_text("~~old idea~~ we decided against this")
    assert not is_retired_fact_text("We ~~thought~~ knew the answer")
    assert not is_retired_fact_text("Not struck [superseded 2026-09-10]")
    # A mid-bullet mention strike leaves a current fact with a struck span.
    assert not is_retired_fact_text("Alice, ~~who lives in Paris~~ [RETRACTED 2026-09-10 r:0badc0de]., is here")
    assert not is_retired_fact_text("")
