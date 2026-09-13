"""Scoring: four stages, kept apart, plus the rates a release gate quotes.

The four stages are scored **separately and in order**, because "the answer was
wrong" is not a finding — *where the information was lost* is:

``detection``
    Every ref the oracle names as evidence appears somewhere in the delivered
    context. An arm that never surfaced the correction cannot have dispositioned
    it.
``disposition``
    The reader's current / contested / unknown matches the oracle's.
``presentation``
    No retired value sits in a slot the reader reads as current; when the
    oracle says contested, both sides are in the payload; every qualifier the
    oracle requires is carried.
``behavior``
    The reader answered with the expected value, or abstained when abstention
    was the expected act.

:func:`failure_stage` names the **first** stage that failed, which is what the
per-family failure table is built from.

Confidence intervals are a percentile bootstrap over **episodes**, never over
questions: questions inside one episode share a store, a transition and a
vocabulary, so resampling them as if they were independent would report an
interval several times too narrow.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

#: Bootstrap resamples. 2000 is enough for a stable 95% percentile interval at
#: these episode counts and cheap enough to run in the report step.
BOOTSTRAP_RESAMPLES = 2000

#: The interval the report quotes.
CONFIDENCE = 0.95

STAGES: tuple[str, ...] = ("detection", "disposition", "presentation", "behavior")


def vocabulary(question: Any) -> tuple[str, ...]:
    """Every candidate value this question could be answered with."""
    values = [question.answer, *question.not_current, *question.sides]
    return tuple(dict.fromkeys(v for v in values if v))


@dataclass
class QuestionScore:
    """One (episode, question, arm) result."""

    episode_id: str
    family: str
    control: str
    split: str
    arm: str
    ask: str
    expected_disposition: str
    observed_disposition: str
    expected_answer: str | None
    observed_answer: str | None
    detection: bool
    #: Detection measured on the *structured* payload rather than the rendered
    #: text — ``None`` for an arm that has no structured payload.
    detection_payload: bool | None
    disposition: bool
    presentation: bool
    behavior: bool
    stale_current: bool
    false_resolution: bool
    appropriate_abstention: bool | None
    source_correct: bool | None
    permission_violation: bool
    deadline_fallback: bool
    #: For the matched-budget control only: did the top-k ladder actually reach
    #: the bundle's token total? ``None`` for every other arm.
    budget_matched: bool | None
    tokens: int
    output_tokens: int
    file_reads: int
    latency_ms: float
    note: str | None = None

    @property
    def failure_stage(self) -> str | None:
        for stage in STAGES:
            if not getattr(self, stage):
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        out["failure_stage"] = self.failure_stage
        return out


def _current_region(context: str, resolved: bool) -> str:
    """The part of *context* a reader reads as the standing answer.

    For the resolved grammar that is the ``Current (n):`` section. For unmarked
    hits it is the whole payload — an unqualified excerpt *is* presented as
    current, which is the entire output-contract argument.
    """
    if not resolved:
        return context
    from bench.current_state.reader import SECTION_RE

    region: list[str] = []
    active = False
    for line in context.splitlines():
        header = SECTION_RE.match(line.strip())
        if header:
            active = header.group(1).lower() == "current"
            continue
        if active:
            region.append(line)
    return "\n".join(region)


def score_question(
    episode: Any, question: Any, delivery: Any, answer: Any
) -> QuestionScore:
    """Score one delivery + reader decision against the oracle."""
    from palinode.core.packing import estimate_tokens

    context = delivery.context
    seen = set(answer.seen_refs)
    detection = set(question.evidence) <= seen if question.evidence else True
    payload_refs = delivery.extra.get("payload_refs")
    detection_payload = (
        set(question.evidence) <= set(payload_refs)
        if question.evidence and payload_refs is not None
        else None
    )

    disposition_ok = answer.disposition == question.disposition

    region = _current_region(context, answer.resolved_grammar).lower()
    retired_clean = not any(v.lower() in region for v in question.not_current)
    sides_ok = True
    if question.disposition == "contested":
        from bench.current_state.reader import OMITTED_CONFLICT_RE

        lowered = context.lower()
        # Both sides in the payload, or — when the output budget dropped the
        # group — the notice that names it by ref. Budget pressure may make a
        # contested answer smaller; the one thing it may not do is make it look
        # settled, and requiring the full sides here would score the correct
        # behaviour as a presentation failure.
        sides_ok = all(v.lower() in lowered for v in question.sides) or (
            bool(OMITTED_CONFLICT_RE.search(context))
            and set(question.evidence) <= set(answer.seen_refs)
        )
    qualifiers_ok = all(q.lower() in context.lower() for q in question.qualifiers)
    presentation = retired_clean and sides_ok and qualifiers_ok

    expected = question.answer
    observed = answer.value
    behavior = (
        (expected is None and observed is None)
        or (expected is not None and observed is not None
            and expected.lower() == observed.lower())
    )

    stale_current = bool(
        answer.disposition == "current"
        and observed is not None
        and any(observed.lower() == v.lower() for v in question.not_current)
    )
    false_resolution = answer.disposition == "current" and question.disposition != "current"
    abstention = (
        answer.disposition == question.disposition
        if question.disposition in ("contested", "unknown")
        else None
    )
    source_correct = (
        set(question.evidence) <= set(answer.cited_refs)
        if question.evidence and question.disposition != "unknown"
        else None
    )
    violation = any(marker in context for marker in episode.forbidden)

    return QuestionScore(
        episode_id=episode.id,
        family=episode.family,
        control=episode.control,
        split=episode.split,
        arm=delivery.arm,
        ask=question.ask,
        expected_disposition=question.disposition,
        observed_disposition=answer.disposition,
        expected_answer=expected,
        observed_answer=observed,
        detection=detection,
        detection_payload=detection_payload,
        disposition=disposition_ok,
        presentation=presentation,
        behavior=behavior,
        stale_current=stale_current,
        false_resolution=false_resolution,
        appropriate_abstention=abstention,
        source_correct=source_correct,
        permission_violation=violation,
        deadline_fallback=bool(answer.deadline_fallback),
        budget_matched=delivery.extra.get("matched"),
        tokens=delivery.tokens,
        output_tokens=estimate_tokens(answer.rendered()),
        file_reads=delivery.file_reads,
        latency_ms=delivery.latency_ms,
        note=question.note,
    )


# ── aggregation ──────────────────────────────────────────────────────────────


@dataclass
class EpisodeScore:
    """Per-episode roll-up — the unit the bootstrap resamples."""

    episode_id: str
    family: str
    control: str
    split: str
    arm: str
    questions: list[QuestionScore] = field(default_factory=list)
    before_ok: bool | None = None

    def rate(self, attribute: str) -> float:
        values = [bool(getattr(q, attribute)) for q in self.questions]
        return statistics.fmean(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family": self.family,
            "control": self.control,
            "split": self.split,
            "arm": self.arm,
            "before_ok": self.before_ok,
            "questions": [q.to_dict() for q in self.questions],
        }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, round(pct * (len(ordered) - 1))))
    return ordered[rank]


def bootstrap_ci(
    episodes: Sequence[EpisodeScore],
    metric: Callable[[EpisodeScore], float | None],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20260912,
) -> dict[str, Any]:
    """Percentile bootstrap over episodes for one per-episode metric.

    ``metric`` returns ``None`` for an episode the metric does not apply to
    (a family with no contested question has no abstention rate), and those
    episodes are dropped from the denominator rather than counted as zero.
    """
    values = [v for v in (metric(e) for e in episodes) if v is not None]
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    means: list[float] = []
    size = len(values)
    for _ in range(resamples):
        means.append(statistics.fmean(rng.choices(values, k=size)))
    tail = (1.0 - CONFIDENCE) / 2.0
    return {
        "n": size,
        "mean": statistics.fmean(values),
        "lo": _percentile(means, tail),
        "hi": _percentile(means, 1.0 - tail),
    }


def summarize(
    episodes: Sequence[EpisodeScore], *, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Every headline number for one arm, with intervals and denominators."""
    questions = [q for e in episodes for q in e.questions]

    def per_episode(attribute: str) -> Callable[[EpisodeScore], float | None]:
        def metric(episode: EpisodeScore) -> float | None:
            return episode.rate(attribute) if episode.questions else None
        return metric

    def optional(attribute: str) -> Callable[[EpisodeScore], float | None]:
        def metric(episode: EpisodeScore) -> float | None:
            values = [
                bool(getattr(q, attribute))
                for q in episode.questions
                if getattr(q, attribute) is not None
            ]
            return statistics.fmean(values) if values else None
        return metric

    def negative_only(episode: EpisodeScore) -> float | None:
        if episode.control != "negative" or not episode.questions:
            return None
        return episode.rate("behavior")

    latencies = [q.latency_ms for q in questions]
    return {
        "episodes": len(episodes),
        "questions": len(questions),
        "detection": bootstrap_ci(episodes, per_episode("detection"), resamples=resamples),
        "disposition": bootstrap_ci(episodes, per_episode("disposition"), resamples=resamples),
        "presentation": bootstrap_ci(episodes, per_episode("presentation"), resamples=resamples),
        "behavior": bootstrap_ci(episodes, per_episode("behavior"), resamples=resamples),
        "stale_current": bootstrap_ci(episodes, per_episode("stale_current"), resamples=resamples),
        "false_resolution": bootstrap_ci(
            episodes, per_episode("false_resolution"), resamples=resamples
        ),
        "appropriate_abstention": bootstrap_ci(
            episodes, optional("appropriate_abstention"), resamples=resamples
        ),
        "source_correct": bootstrap_ci(episodes, optional("source_correct"), resamples=resamples),
        "detection_payload": bootstrap_ci(
            episodes, optional("detection_payload"), resamples=resamples
        ),
        "unaffected_fact_retention": bootstrap_ci(episodes, negative_only, resamples=resamples),
        "before_state_pass": _before_state(episodes),
        "permission_violations": sum(1 for q in questions if q.permission_violation),
        "deadline_fallbacks": sum(1 for q in questions if q.deadline_fallback),
        "budget_matched": {
            "n": sum(1 for q in questions if q.budget_matched is not None),
            "reached": sum(1 for q in questions if q.budget_matched),
        },
        "tokens": _cost(q.tokens for q in questions),
        "output_tokens": _cost(q.output_tokens for q in questions),
        "file_reads": _cost(q.file_reads for q in questions),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 2),
            "p95": round(_percentile(latencies, 0.95), 2),
            "total": round(sum(latencies), 1),
        },
        "failure_stages": _stage_counts(questions),
        "families": _family_table(questions),
    }


def _before_state(episodes: Sequence[EpisodeScore]) -> dict[str, Any]:
    checked = [e for e in episodes if e.before_ok is not None]
    return {
        "n": len(checked),
        "passed": sum(1 for e in checked if e.before_ok),
    }


def _cost(values: Iterable[int]) -> dict[str, Any]:
    items = list(values)
    if not items:
        return {"total": 0, "mean": 0.0, "p95": 0}
    return {
        "total": sum(items),
        "mean": round(statistics.fmean(items), 1),
        "p95": int(_percentile([float(v) for v in items], 0.95)),
    }


def _stage_counts(questions: Sequence[QuestionScore]) -> dict[str, int]:
    counts = {stage: 0 for stage in STAGES}
    counts["passed"] = 0
    for question in questions:
        stage = question.failure_stage
        counts["passed" if stage is None else stage] += 1
    return counts


def _family_table(questions: Sequence[QuestionScore]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for question in questions:
        bucket = table.setdefault(
            question.family,
            {"n": 0, "detection": 0, "disposition": 0, "presentation": 0,
             "behavior": 0, "stale_current": 0},
        )
        bucket["n"] += 1
        for key in ("detection", "disposition", "presentation", "behavior", "stale_current"):
            bucket[key] += int(bool(getattr(question, key)))
    return table


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE",
    "STAGES",
    "EpisodeScore",
    "QuestionScore",
    "bootstrap_ci",
    "score_question",
    "summarize",
    "vocabulary",
]
