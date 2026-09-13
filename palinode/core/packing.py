"""Deterministic packing of injected context into a size budget.

Injection payloads — the session-start digest, the per-turn recall block, any
future bundle — are assembled from *units*: indivisible pieces of text that a
reader must receive whole or not at all. This module decides which units fit a
budget and which do not, and it does so under one rule:

    **Truncation may remove information. It may never manufacture certainty.**

A payload trimmed to fit must never leave a contested claim looking settled, an
unverified claim looking verified, or an absence looking like a checked "no".
Concretely:

- A unit is never split. Half a row is a row whose qualification may be missing.
- A kept unit never loses a qualifier, because the qualifier is part of its
  text. A unit that carries qualifications may not even be demoted to its
  shorter ``gist`` form — demotion is for plain units only, and constructing a
  qualified unit with a ``gist`` is an error.
- A conflict group is whole-or-stub. When it does not fit, the pack emits one
  explicit contested-partial stub — ``⚠ N conflicts omitted for budget — see
  <refs>`` — that keeps every source pointer, so the reader knows a conflict
  exists and where to look. Dropping one side of a conflict silently is the
  failure this module exists to prevent.
- Over-budget is visible: :attr:`Packed.omitted`, :attr:`Packed.truncated_reason`
  and a WARNING log line, never a silent slice.

Priority decides who gets first refusal on the remaining space (the allocation
order from the context-engineering review: current assertions, then conflict
alternatives, then explicit-insufficiency notes, then background). It does not
mean a lower-priority unit is never kept after a higher-priority one was
refused — packing continues past a unit that does not fit, so a small
background unit may ride along behind a large refused one. That is deliberate:
the refused unit's absence is reported either way, and leaving the space empty
buys nothing.

Pure: no I/O, no clock, no config read inside :func:`pack`. Same units and
budget in, same ``Packed`` out; re-packing a ``Packed``'s own units is a
fixed point (same units, same text). :func:`budget_from_config` is the one
config-reading helper, and it reads the loaded singleton at call time.

The unit vocabulary (``kind``, ``priority``, ``refs``, ``qualifiers``, ``text``)
is deliberately general: the current-state digest is the first caller, and the
context bundle's own budget step is meant to route through this same function
rather than grow a second packing rule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

logger = logging.getLogger("palinode.packing")

#: Characters per token for :func:`estimate_tokens`. This is an ESTIMATE, not a
#: tokenizer: Palinode ships no tokenizer, and adding one (tiktoken is
#: model-specific and none of the local models use it) to decide how many rows
#: fit a digest is a dependency bought for nothing. 4 chars/token is the usual
#: English-prose ratio and errs high on identifier-dense text like file paths —
#: which is the safe direction for a cap. Anything reported in tokens by this
#: module is an estimate and is named as one.
CHARS_PER_TOKEN = 4

#: The newline that joins one unit to the next, charged per unit so a pack's
#: char total matches the length of the string a caller renders from it.
JOIN_CHARS = 1

# ── unit kinds ───────────────────────────────────────────────────────────────
#: A current assertion — a digest row, a bundle fact. May carry a ``gist``.
KIND_ASSERTION = "assertion"
#: A conflict group: every visible side plus the reasons, in one unit. Whole or
#: stub; never demoted, never split. Mirrors ``resolution.OUTCOME_CONFLICT``.
KIND_CONFLICT = "conflict"
#: An explicit "not enough evidence" note (``resolution.OUTCOME_INSUFFICIENT``).
#: Carries qualifications by nature, so it is never demoted — silently dropping
#: it turns "unknown" back into "nothing to say".
KIND_INSUFFICIENCY = "insufficiency"
#: A unit demoted to its gist-plus-pointer form by :func:`pack`.
KIND_GIST = "gist"
#: The stub :func:`pack` emits for conflict groups that did not fit.
KIND_CONTESTED_PARTIAL = "contested_partial"

# ── priorities ───────────────────────────────────────────────────────────────
#: Lower packs first. Callers may use any int; these are the default rungs.
PRIORITY_ASSERTION = 10
PRIORITY_CONFLICT = 20
PRIORITY_INSUFFICIENCY = 30
PRIORITY_BACKGROUND = 40

#: Kinds that must never be demoted to a gist, whatever the caller passes.
_NEVER_DEMOTED = frozenset({KIND_CONFLICT, KIND_INSUFFICIENCY})


def estimate_tokens(text: str) -> int:
    """Estimated token count for *text* — ``ceil(len / CHARS_PER_TOKEN)``.

    An estimate, deterministic and monotonic in length. Never a tokenizer's
    answer; see :data:`CHARS_PER_TOKEN`.
    """
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


@dataclass(frozen=True)
class Unit:
    """One indivisible thing to inject.

    ``text`` is the rendered line (or block) exactly as the caller will emit it,
    qualifiers included — which is what makes "never strip a qualifier from a
    kept unit" structural rather than a rule someone has to remember.

    ``qualifiers`` names those qualifications for reporting and for the demotion
    guard; it is not rendered by this module. ``refs`` are the source pointers
    that survive into the contested-partial stub when a conflict group does not
    fit. ``gist`` is an optional shorter rendering (gist plus pointer) used when
    the full text does not fit; only plain, unqualified assertions may carry
    one. ``payload`` is an opaque caller handle (a digest row, a bundle entry)
    that :func:`pack` never inspects.
    """

    kind: str
    text: str
    priority: int = PRIORITY_BACKGROUND
    refs: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()
    gist: str | None = None
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.gist is None:
            return
        if self.qualifiers:
            raise ValueError(
                f"unit with qualifiers may not carry a gist: {self.qualifiers!r} "
                "— demoting it would strip the qualification that makes the "
                "claim honest"
            )
        if self.kind in _NEVER_DEMOTED:
            raise ValueError(f"{self.kind} units may not carry a gist")

    @property
    def chars(self) -> int:
        return len(self.text) + JOIN_CHARS

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass(frozen=True)
class Budget:
    """A payload ceiling in characters and in estimated tokens.

    Both caps are enforced; ``0`` (or negative) means that cap is off, and a
    budget with both off is :attr:`unlimited` — the packer then keeps every
    unit, which is how a caller's own line/count bounds remain the only bound.

    ``reserved_*`` is fixed scaffolding the caller will render around the units
    (headers, a trailing hint) and that the units therefore may not spend.
    """

    max_chars: int = 0
    max_tokens: int = 0
    reserved_chars: int = 0
    reserved_tokens: int = 0

    @property
    def unlimited(self) -> bool:
        return self.max_chars <= 0 and self.max_tokens <= 0


@dataclass(frozen=True)
class Packed:
    """What fit, what did not, and why.

    ``units`` is what to render, in packing order. ``omitted`` holds the units
    that did not fit (originals, not stubs) and ``demoted`` the originals of
    units kept in gist form. ``truncated_reason`` is ``None`` exactly when
    nothing was omitted.

    ``limits`` names the caps that actually bound this pack — ``max_chars``,
    ``max_tokens``, or both — in the order they first bit. Empty when nothing
    was refused. It is the same information ``truncated_reason`` states in
    prose, carried structurally so a caller can map it onto its own coverage
    vocabulary instead of parsing a sentence.
    """

    units: tuple[Unit, ...] = ()
    omitted: tuple[Unit, ...] = ()
    demoted: tuple[Unit, ...] = ()
    truncated_reason: str | None = None
    chars: int = 0
    tokens: int = 0
    limits: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The kept units joined by the newline their char cost accounts for."""
        return "\n".join(u.text for u in self.units)

    @property
    def complete(self) -> bool:
        return not self.omitted


def _ordered(units: Sequence[Unit]) -> list[Unit]:
    """Stable priority order: lower priority first, input order within a rung."""
    return [u for _, u in sorted(enumerate(units), key=lambda p: (p[1].priority, p[0]))]


def _contested_stub(conflicts: Sequence[Unit]) -> Unit:
    """The explicit partial result for conflict groups that did not fit.

    Keeps every source pointer the omitted groups carried: the reader loses the
    detail, not the knowledge that a conflict exists or the way to reach it.
    Both sides of one conflict name each other, so the same record arrives twice
    — once as a path and once as the bare ref the other side links it by. They
    are deduplicated to one pointer (first spelling wins) so the stub lists each
    record once instead of once per direction.
    """
    refs: list[str] = []
    seen: set[str] = set()
    for unit in conflicts:
        for ref in unit.refs:
            key = ref[:-3] if ref.endswith(".md") else ref
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    where = ", ".join(refs) if refs else "no source pointers recorded"
    plural = "conflict" if len(conflicts) == 1 else "conflicts"
    return Unit(
        kind=KIND_CONTESTED_PARTIAL,
        text=f"⚠ {len(conflicts)} {plural} omitted for budget — see {where}",
        priority=PRIORITY_CONFLICT,
        refs=tuple(refs),
    )


def pack(units: Sequence[Unit], budget: Budget) -> Packed:
    """Fit *units* into *budget*, preserving conflicts and qualifications.

    Units are admitted in stable priority order. A unit that does not fit is
    demoted to its ``gist`` when it has one and the gist fits, otherwise
    omitted. When any omitted unit is a conflict group, one contested-partial
    stub is appended, evicting the lowest-priority kept units if that is what
    it takes to make room — an explicit "there is a conflict here, here is
    where" outranks one more background line.

    Returns a :class:`Packed`; logs a WARNING whenever anything was omitted or
    demoted. An unlimited budget keeps everything.
    """
    ordered = _ordered(units)
    if budget.unlimited:
        chars = sum(u.chars for u in ordered) + budget.reserved_chars
        tokens = sum(u.tokens for u in ordered) + budget.reserved_tokens
        return Packed(units=tuple(ordered), chars=chars, tokens=tokens)

    kept: list[Unit] = []
    #: The original behind each kept unit — itself, or the unit a kept gist was
    #: demoted from. Parallel to ``kept`` so an eviction reports what was lost
    #: rather than the shortened form it briefly wore.
    kept_origin: list[Unit] = []
    omitted: list[Unit] = []
    demoted: list[Unit] = []
    chars = budget.reserved_chars
    tokens = budget.reserved_tokens
    limits: list[str] = []

    def fits(unit: Unit) -> str | None:
        """``None`` when *unit* fits, else the name of the cap it would break."""
        if 0 < budget.max_chars < chars + unit.chars:
            return "max_chars"
        if 0 < budget.max_tokens < tokens + unit.tokens:
            return "max_tokens"
        return None

    for unit in ordered:
        broke = fits(unit)
        if broke is None:
            kept.append(unit)
            kept_origin.append(unit)
            chars += unit.chars
            tokens += unit.tokens
            continue
        if broke not in limits:
            limits.append(broke)
        if unit.gist is not None:
            gist_unit = replace(unit, kind=KIND_GIST, text=unit.gist, gist=None)
            if fits(gist_unit) is None:
                kept.append(gist_unit)
                kept_origin.append(unit)
                demoted.append(unit)
                chars += gist_unit.chars
                tokens += gist_unit.tokens
                continue
        omitted.append(unit)

    stub_dropped = False
    if any(u.kind == KIND_CONFLICT for u in omitted):
        while True:
            stub = _contested_stub([u for u in omitted if u.kind == KIND_CONFLICT])
            if fits(stub) is None:
                kept.append(stub)
                kept_origin.append(stub)
                chars += stub.chars
                tokens += stub.tokens
                break
            if not kept:
                # The budget cannot hold even the statement that something was
                # withheld. Nothing is rendered rather than a settled-looking
                # fragment; the reason below says so.
                stub_dropped = True
                break
            evicted = kept.pop()
            original = kept_origin.pop()
            chars -= evicted.chars
            tokens -= evicted.tokens
            if original is not evicted:
                demoted.remove(original)
            omitted.append(original)

    reason = None
    if omitted:
        caps = ", ".join(limits) or "max_chars"
        reason = (
            f"{len(omitted)} of {len(ordered)} units omitted for budget "
            f"({caps}; chars {chars}/{budget.max_chars}, "
            f"tokens {tokens}/{budget.max_tokens})"
        )
        if stub_dropped:
            reason += "; budget too small even for the contested-partial stub"

    if omitted or demoted:
        logger.warning(
            "injection payload over budget op=pack kept=%d omitted=%d demoted=%d "
            "chars=%d/%d tokens=%d/%d reason=%s",
            len(kept),
            len(omitted),
            len(demoted),
            chars,
            budget.max_chars,
            tokens,
            budget.max_tokens,
            reason,
        )

    return Packed(
        units=tuple(kept),
        omitted=tuple(omitted),
        demoted=tuple(demoted),
        truncated_reason=reason,
        chars=chars,
        tokens=tokens,
        limits=tuple(limits),
    )


#: The two injection surfaces, budgeted separately on purpose: the startup
#: payload is paid once and can afford orientation, while the per-turn recall
#: block is paid on every message and competes with the user's own turn. One
#: shared number would be wrong for one of them.
SURFACE_STARTUP = "startup"
SURFACE_PER_TURN = "per_turn"


def budget_from_config(
    surface: str = SURFACE_STARTUP,
    *,
    reserved_chars: int = 0,
    reserved_tokens: int = 0,
) -> Budget:
    """The configured :class:`Budget` for an injection surface.

    ``startup`` reads ``context.injection_max_chars`` / ``injection_max_tokens``
    (session-start core injection: ``/context/prime``, ``palinode_session_init``,
    ``palinode prime``); ``per_turn`` reads ``context.recall_max_chars`` /
    ``recall_max_tokens`` (the per-message recall block).
    """
    from palinode.core.config import config

    if surface == SURFACE_PER_TURN:
        max_chars = config.context.recall_max_chars
        max_tokens = config.context.recall_max_tokens
    elif surface == SURFACE_STARTUP:
        max_chars = config.context.injection_max_chars
        max_tokens = config.context.injection_max_tokens
    else:
        raise ValueError(f"unknown injection surface: {surface!r}")
    return Budget(
        max_chars=max_chars,
        max_tokens=max_tokens,
        reserved_chars=reserved_chars,
        reserved_tokens=reserved_tokens,
    )
