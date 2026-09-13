"""The injection packer: what fits, what is demoted, and what may never be lost.

The contract under test is not "stay under N chars" — a blind slice does that.
It is that a payload trimmed to fit never manufactures certainty: no unit is
split, no kept unit loses a qualifier, no conflict group loses one side, and
every omission is reported rather than silent.

Pure-function tests: no store, no config, no clock.
"""
from __future__ import annotations

import logging

import pytest

from palinode.core.packing import (
    CHARS_PER_TOKEN,
    KIND_ASSERTION,
    KIND_CONFLICT,
    KIND_CONTESTED_PARTIAL,
    KIND_GIST,
    KIND_INSUFFICIENCY,
    PRIORITY_ASSERTION,
    PRIORITY_BACKGROUND,
    PRIORITY_CONFLICT,
    PRIORITY_INSUFFICIENCY,
    SURFACE_PER_TURN,
    SURFACE_STARTUP,
    Budget,
    Packed,
    Unit,
    budget_from_config,
    estimate_tokens,
    pack,
)


def _assertion(text: str, priority: int = PRIORITY_ASSERTION, **kw) -> Unit:
    return Unit(kind=KIND_ASSERTION, text=text, priority=priority, **kw)


def _conflict(text: str, refs: tuple[str, ...], **kw) -> Unit:
    return Unit(
        kind=KIND_CONFLICT,
        text=text,
        priority=kw.pop("priority", PRIORITY_CONFLICT),
        refs=refs,
        qualifiers=kw.pop("qualifiers", ("contradicts:" + refs[-1],)),
        **kw,
    )


# ── measurement ──────────────────────────────────────────────────────────────


def test_token_estimate_is_ceiling_of_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * CHARS_PER_TOKEN) == 1
    assert estimate_tokens("a" * (CHARS_PER_TOKEN + 1)) == 2


def test_unit_char_cost_includes_the_joining_newline():
    unit = _assertion("abcd")
    assert unit.chars == len("abcd") + 1
    assert pack([unit, _assertion("efgh")], Budget()).text == "abcd\nefgh"


# ── priority order ───────────────────────────────────────────────────────────


def test_packs_in_priority_order_and_keeps_input_order_within_a_rung():
    units = [
        _assertion("background", priority=PRIORITY_BACKGROUND),
        _assertion("first", priority=PRIORITY_ASSERTION),
        _assertion("second", priority=PRIORITY_ASSERTION),
    ]
    packed = pack(units, Budget())
    assert [u.text for u in packed.units] == ["first", "second", "background"]


def test_priority_decides_who_gets_the_space():
    # Room for one line of six chars plus its newline.
    budget = Budget(max_chars=7)
    units = [
        _assertion("second", priority=PRIORITY_BACKGROUND),
        _assertion("first!", priority=PRIORITY_ASSERTION),
    ]
    packed = pack(units, budget)
    assert [u.text for u in packed.units] == ["first!"]
    assert [u.text for u in packed.omitted] == ["second"]


# ── never split, never strip ─────────────────────────────────────────────────


def test_a_unit_that_does_not_fit_is_omitted_whole_never_sliced():
    packed = pack([_assertion("a" * 100)], Budget(max_chars=40))
    assert packed.units == ()
    assert packed.text == ""
    assert [u.text for u in packed.omitted] == ["a" * 100]


def test_a_qualified_unit_may_not_carry_a_gist():
    with pytest.raises(ValueError, match="qualifiers"):
        _assertion("claim [epistemic: unverified]", qualifiers=("epistemic:unverified",),
                   gist="claim")


def test_conflict_and_insufficiency_units_may_not_carry_a_gist():
    with pytest.raises(ValueError, match="may not carry a gist"):
        Unit(kind=KIND_CONFLICT, text="A vs B", gist="A")
    with pytest.raises(ValueError, match="may not carry a gist"):
        Unit(kind=KIND_INSUFFICIENCY, text="not enough evidence", gist="unknown")


def test_kept_units_keep_every_qualifier_they_carried():
    qualified = _assertion(
        "- [a.md] A [epistemic: unverified]",
        qualifiers=("epistemic:unverified",),
    )
    plain = _assertion("- [b.md] B — detail", gist="- [b.md] B")
    packed = pack([qualified, plain], Budget(max_chars=50))
    kept = {u.text for u in packed.units}
    assert "- [a.md] A [epistemic: unverified]" in kept
    kept_qualified = next(u for u in packed.units if u.qualifiers)
    assert kept_qualified.qualifiers == ("epistemic:unverified",)
    assert "epistemic: unverified" in kept_qualified.text
    # The plain unit is the one that had to give — demoted, not stripped.
    assert "- [b.md] B" in kept
    assert [u.text for u in packed.demoted] == ["- [b.md] B — detail"]


def test_demotion_uses_the_gist_and_keeps_the_pointer():
    packed = pack(
        [_assertion("- [a.md] Title — a long explanation", gist="- [a.md] Title")],
        Budget(max_chars=20),
    )
    [kept] = packed.units
    assert kept.kind == KIND_GIST
    assert kept.text == "- [a.md] Title"
    assert packed.omitted == ()


# ── conflicts: whole, or an explicit stub ────────────────────────────────────


def test_a_conflict_group_that_fits_is_kept_whole():
    group = _conflict("A (a.md) vs B (b.md)", refs=("a.md", "b.md"))
    packed = pack([group], Budget(max_chars=100))
    assert packed.units == (group,)
    assert packed.truncated_reason is None


def test_a_conflict_group_that_does_not_fit_becomes_a_stub_that_keeps_refs():
    group = _conflict("A (a.md) vs B (b.md) " + "x" * 200, refs=("a.md", "b.md"))
    packed = pack([group], Budget(max_chars=80))
    [stub] = packed.units
    assert stub.kind == KIND_CONTESTED_PARTIAL
    assert stub.text == "⚠ 1 conflict omitted for budget — see a.md, b.md"
    assert stub.refs == ("a.md", "b.md")
    assert packed.omitted == (group,)
    assert packed.truncated_reason


def test_one_stub_aggregates_every_omitted_conflict_group():
    groups = [
        _conflict("A vs B " + "x" * 200, refs=("a.md", "b.md")),
        _conflict("C vs D " + "y" * 200, refs=("c.md", "b.md")),
    ]
    packed = pack(groups, Budget(max_chars=100))
    [stub] = packed.units
    assert stub.text == "⚠ 2 conflicts omitted for budget — see a.md, b.md, c.md"


def test_the_stub_evicts_background_rather_than_going_unsaid():
    conflict = _conflict("A vs B " + "x" * 200, refs=("a.md", "b.md"))
    filler = _assertion("z" * 40, priority=PRIORITY_BACKGROUND)
    packed = pack([conflict, filler], Budget(max_chars=60))
    assert [u.kind for u in packed.units] == [KIND_CONTESTED_PARTIAL]
    assert {u.text for u in packed.omitted} == {conflict.text, filler.text}
    assert packed.chars <= 60


def test_a_budget_too_small_even_for_the_stub_says_so():
    conflict = _conflict("A vs B " + "x" * 200, refs=("a.md", "b.md"))
    packed = pack([conflict], Budget(max_chars=10))
    assert packed.units == ()
    assert "contested-partial stub" in packed.truncated_reason


# ── both caps ────────────────────────────────────────────────────────────────


def test_the_char_cap_is_enforced():
    units = [_assertion("a" * 20) for _ in range(10)]
    packed = pack(units, Budget(max_chars=63))
    assert len(packed.units) == 3
    assert len(packed.text) <= 63


def test_the_token_cap_is_enforced_independently_of_the_char_cap():
    units = [_assertion("a" * 20) for _ in range(10)]
    packed = pack(units, Budget(max_tokens=10))
    assert len(packed.units) == 2  # 5 estimated tokens each
    assert packed.tokens <= 10
    assert "max_tokens" in packed.truncated_reason


def test_reserved_scaffolding_is_spent_before_any_unit():
    units = [_assertion("a" * 20)]
    assert pack(units, Budget(max_chars=21)).units
    assert pack(units, Budget(max_chars=21, reserved_chars=10)).units == ()


def test_an_unset_budget_keeps_everything():
    units = [_assertion("a" * 5000) for _ in range(20)]
    packed = pack(units, Budget())
    assert len(packed.units) == 20
    assert packed.omitted == ()
    assert packed.truncated_reason is None
    assert packed.complete


# ── reporting ────────────────────────────────────────────────────────────────


def test_omission_is_reported_in_the_result_and_logged_at_warning(caplog):
    units = [_assertion("a" * 20) for _ in range(5)]
    with caplog.at_level(logging.WARNING, logger="palinode.packing"):
        packed = pack(units, Budget(max_chars=42))
    assert len(packed.omitted) == 3
    assert "3 of 5 units omitted for budget" in packed.truncated_reason
    assert not packed.complete
    assert any("over budget" in r.message for r in caplog.records)


def test_nothing_omitted_means_no_reason_and_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="palinode.packing"):
        packed = pack([_assertion("short")], Budget(max_chars=500))
    assert packed.truncated_reason is None
    assert caplog.records == []


# ── determinism ──────────────────────────────────────────────────────────────


def test_pack_is_deterministic_for_the_same_inputs():
    units = [_assertion(f"row {i}", priority=PRIORITY_ASSERTION + i % 3) for i in range(12)]
    budget = Budget(max_chars=40, max_tokens=9)
    first = pack(units, budget)
    second = pack(units, budget)
    assert first == second


def test_repacking_a_packed_payload_is_a_fixed_point():
    units = [
        _assertion("- [a.md] A — detail", gist="- [a.md] A"),
        _conflict("A vs B " + "x" * 100, refs=("a.md", "b.md")),
        _assertion("- [c.md] C", priority=PRIORITY_BACKGROUND),
    ]
    budget = Budget(max_chars=70)
    once = pack(units, budget)
    twice = pack(once.units, budget)
    assert twice.units == once.units
    assert twice.text == once.text


def test_insufficiency_notes_are_ordinary_whole_units():
    note = Unit(
        kind=KIND_INSUFFICIENCY,
        text="no eligible evidence for X — see a.md",
        priority=PRIORITY_INSUFFICIENCY,
        refs=("a.md",),
        qualifiers=("no_eligible_evidence",),
    )
    assert pack([note], Budget(max_chars=100)).units == (note,)
    dropped = pack([note], Budget(max_chars=5))
    assert dropped.units == ()
    assert dropped.omitted == (note,)


def test_packed_defaults_are_an_empty_complete_pack():
    assert Packed().text == ""
    assert Packed().complete


# ── config-driven budgets, separate per surface ──────────────────────────────


def test_startup_and_per_turn_budgets_come_from_different_keys(monkeypatch):
    from palinode.core.config import config

    monkeypatch.setattr(config.context, "injection_max_chars", 111)
    monkeypatch.setattr(config.context, "injection_max_tokens", 11)
    monkeypatch.setattr(config.context, "recall_max_chars", 222)
    monkeypatch.setattr(config.context, "recall_max_tokens", 22)

    startup = budget_from_config(SURFACE_STARTUP)
    per_turn = budget_from_config(SURFACE_PER_TURN, reserved_chars=5)

    assert (startup.max_chars, startup.max_tokens) == (111, 11)
    assert (per_turn.max_chars, per_turn.max_tokens) == (222, 22)
    assert per_turn.reserved_chars == 5


def test_zero_on_both_caps_is_an_unlimited_budget(monkeypatch):
    from palinode.core.config import config

    monkeypatch.setattr(config.context, "injection_max_chars", 0)
    monkeypatch.setattr(config.context, "injection_max_tokens", 0)
    assert budget_from_config(SURFACE_STARTUP).unlimited


def test_an_unknown_surface_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="unknown injection surface"):
        budget_from_config("whenever")
