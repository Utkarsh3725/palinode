"""Propose-side guard on retiring operations, run before the executor sees them.

The executor (:mod:`palinode.consolidation.executor`) is the dispose-side
check: it validates each op's fields against the document's declared regime.
It never sees the *prompt*, so it cannot tell whether the memory a RETRACT
names as its evidence was ever in front of the model. That is this module's
one job: the runner builds a :class:`PromptContext` from exactly what it put
in the prompt — the EXISTING_FACTS ids, the ACTIVE_DECISIONS refs, the
RECENT_NOTES refs — and :func:`guard_operations` checks each retiring op
against it. Deterministic, no LLM, no prose judgement.

Two rules, both from field observations on the production model:

* **An uncited RETRACT is downgraded to PROPOSE_CONTRADICTS.** The model was
  seen retracting a later-dated observation with the rationale "known to be
  incorrect" and nothing in context to show it. A citation is necessary
  evidence, not proof: the guard checks only that ``falsified_by`` (or, when
  the field is absent, a memory ref or fact id named in the rationale)
  resolves to something that was in the prompt. It resolves nothing further.
  The downgrade keeps the signal — the world disagrees with this fact — as a
  reviewable link instead of a tombstone, and counts it as
  ``retract_downgraded``. When no well-formed memory ref can be linked, the op
  is dropped under the same count; the RETRACT is never applied.

* **A retiring op aimed at the auto-footer is rejected.** The ``## See also``
  block under ``<!-- palinode-auto-footer -->`` holds ``[[wikilinks]]`` that
  carry fact ids like any bullet, so the model can — and did — aim a RETRACT
  whose rationale described one fact at the id of a footer link. Footer links
  are navigation, not claims; nothing there is ever retired by a proposal.
  Counted as ``footer_op_rejected``.

Everything else passes through untouched, in order.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from palinode.consolidation.fact_ids import FACT_LINE_RE
from palinode.consolidation.op_parse import op_kind, op_reason
from palinode.core.embedding_preprocess import AUTO_FOOTER_MARKER
from palinode.core.typed_links import TypedLinkError, normalize_link_refs

logger = logging.getLogger("palinode.consolidation.guard")

#: Ops that retire a fact's current text. MERGE also retires, but it names
#: several ids and never aims at a single wrong one; the footer rule is about
#: a single mis-aimed id. ARCHIVE_BEFORE names a date rather than an id and
#: its own recognizer stops at the footer marker, so the footer rule can only
#: fire on one carrying a stray ``id`` — which is exactly the mis-aim the rule
#: exists for, and is why it is in the set rather than exempted from it.
RETIRING_OPS = frozenset({"RETRACT", "ARCHIVE", "ARCHIVE_BEFORE", "SUPERSEDE"})

#: Stats this guard adds to a run summary, always present so a quiet pass
#: reports ``0`` rather than omitting the key.
GUARD_STATS: tuple[str, ...] = ("retract_downgraded", "footer_op_rejected")

_DOWNGRADE_NOTE = "(downgraded from RETRACT: no in-context evidence cited)"


@dataclass(frozen=True)
class PromptContext:
    """What one compaction prompt actually contained, as citable identifiers.

    Built by the runner from the same rendering it sent, so a decision the
    budget truncated out of ACTIVE_DECISIONS, or a note that fell off the end
    of RECENT_NOTES, is not in here either. ``note_refs`` keeps rendering
    order; the last one is the group's latest note and the downgrade's
    fallback link.
    """

    fact_ids: frozenset[str]
    footer_fact_ids: frozenset[str]
    decision_refs: frozenset[str]
    note_refs: tuple[str, ...]

    @property
    def memory_refs(self) -> frozenset[str]:
        return self.decision_refs | frozenset(self.note_refs)

    @property
    def fallback_ref(self) -> str | None:
        return self.note_refs[-1] if self.note_refs else None

    def resolves(self, citation: str) -> bool:
        """Does *citation* name a fact id or memory ref that was in the prompt?

        Accepts the ``category/slug.md`` spelling the executor's own tests use
        for ``falsified_by`` as equivalent to ``category/slug``.
        """
        ref = _canonical_ref(citation)
        return bool(ref) and (ref in self.fact_ids or ref in self.memory_refs)

    def cited_in(self, text: str, *, excluding: str | None = None) -> str | None:
        """First in-context identifier named as a whole token in *text*.

        The fallback for a RETRACT with no ``falsified_by``. Only identifiers
        that were in the prompt are searched for, so a match is by
        construction in context; the op's own id is excluded because a fact
        does not falsify itself.
        """
        for ref in sorted(self.memory_refs):
            if _named_in(ref, text, allow_md=True):
                return ref
        for fid in sorted(self.fact_ids):
            if fid != excluding and _named_in(fid, text):
                return fid
        return None


def _canonical_ref(value: object) -> str:
    ref = str(value or "").strip()
    if ref.startswith("./"):
        ref = ref[2:]
    if ref.endswith(".md"):
        ref = ref[:-3]
    return ref


def _named_in(identifier: str, text: str, *, allow_md: bool = False) -> bool:
    tail = r"(?:\.md)?" if allow_md else ""
    pattern = rf"(?<![\w./-]){re.escape(identifier)}{tail}(?![\w./-])"
    return re.search(pattern, text) is not None


def footer_fact_ids(body: str) -> frozenset[str]:
    """Fact ids on bullets at or after the auto-footer marker in *body*.

    The marker is the save path's contract for "everything from here on is
    generated navigation" — the same boundary the indexer strips before
    embedding and the mention-level retract refuses to strike.
    """
    idx = body.find(AUTO_FOOTER_MARKER)
    if idx == -1:
        return frozenset()
    return frozenset(m.group(2) for m in FACT_LINE_RE.finditer(body[idx:]))


def guard_operations(
    operations: list[dict],
    context: PromptContext,
    *,
    target: str,
) -> tuple[list[dict], dict[str, int]]:
    """Apply the two rules to *operations*; return ``(kept_ops, stats)``.

    ``stats`` always carries every key in :data:`GUARD_STATS`. Non-dict entries
    pass through for the executor to log, exactly as they would have.
    """
    stats = {key: 0 for key in GUARD_STATS}
    kept: list[dict] = []
    for op in operations:
        if not isinstance(op, dict):
            kept.append(op)
            continue
        kind = op_kind(op) or "KEEP"
        fact_id = str(op.get("id") or "")

        if kind in RETIRING_OPS and fact_id in context.footer_fact_ids:
            logger.warning(
                "%s rejected: fact id=%r is in the auto-footer block of %s — "
                "footer wikilinks are navigation, not claims (rationale: %r)",
                kind, fact_id, target, op_reason(op),
            )
            stats["footer_op_rejected"] += 1
            continue

        if kind != "RETRACT":
            kept.append(op)
            continue

        evidence = _evidence(op, context, fact_id)
        if evidence is not None:
            kept.append(op)
            continue

        downgraded = _downgrade(op, context, fact_id)
        stats["retract_downgraded"] += 1
        if downgraded is None:
            logger.warning(
                "RETRACT dropped: no in-context evidence cited and no memory ref "
                "to link the conflict to — id=%r on %s (falsified_by=%r, "
                "rationale=%r)",
                fact_id, target, op.get("falsified_by"), op_reason(op),
            )
            continue
        logger.warning(
            "RETRACT downgraded to PROPOSE_CONTRADICTS: no in-context evidence "
            "cited — id=%r on %s (falsified_by=%r, rationale=%r); linking to %s",
            fact_id, target, op.get("falsified_by"), op_reason(op),
            downgraded["contradicts"][0],
        )
        kept.append(downgraded)
    return kept, stats


def _evidence(op: dict, context: PromptContext, fact_id: str) -> str | None:
    """The in-context identifier this RETRACT cites, or ``None``.

    ``falsified_by`` is the explicit claim and is checked as given — a field
    that names something outside the prompt is an uncited RETRACT, whatever
    the rationale says. Only when the field is absent is the rationale scanned.
    """
    cited = str(op.get("falsified_by") or "").strip()
    if cited:
        return _canonical_ref(cited) if context.resolves(cited) else None
    return context.cited_in(op_reason(op), excluding=fact_id)


def _downgrade(op: dict, context: PromptContext, fact_id: str) -> dict | None:
    """The PROPOSE_CONTRADICTS op standing in for an uncited RETRACT.

    Links to the group's latest note — the memory the runner already holds
    for this pass and the closest thing to "what disagreed". ``None`` when no
    well-formed memory ref exists to link to, in which case the caller drops
    the op rather than apply a RETRACT nothing supports.
    """
    ref = context.fallback_ref
    if not ref:
        return None
    try:
        refs = normalize_link_refs([ref], "contradicts")
    except TypedLinkError:
        return None
    if not refs:
        return None
    reason = op_reason(op)
    return {
        "op": "PROPOSE_CONTRADICTS",
        "id": fact_id,
        "contradicts": refs,
        "rationale": f"{reason} {_DOWNGRADE_NOTE}".strip(),
    }
