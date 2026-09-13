"""The versioned event corpus: episodes, oracles, held-out variants, coverage.

An **episode** is a sequence of *events* (saves, explicit replacements,
retractions, archives, consolidation passes, clock advances, imports, restores)
followed by one or more *questions*, each carrying a **deterministic oracle**:
the expected disposition, the expected current value, the retired values that
must never be presented as current, and the evidence refs that should be
delivered. Nothing in an oracle is a model's opinion — every expected answer
follows mechanically from the events, which is why this corpus needs no
semantic judge.

Two properties make the numbers mean something:

**Positive and negative controls per family.** Every scenario family carries a
positive episode (the transition happened) and a negative one (structurally the
same, no transition). A system that refuses every question, or forgets every
old fact, passes the positives and fails the negatives. The coverage gate
(:func:`coverage_gate`) fails if either side of any family disappears.

**Before-state checks.** A withdrawal case where the answer becomes unknown is
a trivial pass for an arm that knows nothing. An episode may declare a ``before``
checkpoint: at that event index the same arm must already answer the question
correctly. The harness withholds disposition credit on the final question when
the before-state check failed.

Held-out evaluation is generated, not authored: :func:`load_corpus` derives one
variant of every base episode by substituting an unseen project name and
paraphrasing the question with a fixed template set, under a fixed seed. The
dev split is what the corpus was written against; the held-out split is what
the report leads with.

One family is present and deliberately unscored. Claim-validity / as-of
semantics need the temporal-assertion schema that is deferred to the next
release, so its episodes carry the scoped disposition
:data:`DEFERRED_DISPOSITION` rather than being dropped: a family that vanishes
from the table is indistinguishable from a family nobody thought about.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import yaml

#: Bumped when the episode schema or the authored content changes in a way
#: that invalidates comparison with an earlier run's numbers.
CORPUS_VERSION = 1

#: The scoped disposition recorded for the claim-validity / as-of family. Not
#: one of the three outcomes; the harness reports these episodes and scores
#: nothing from them.
DEFERRED_DISPOSITION = "deferred:temporal-assertions"

#: The three scored dispositions, matching the resolution policy's outcomes
#: (``supported_current`` / ``unresolved_conflict`` / ``insufficient_evidence``)
#: in the vocabulary a reader uses.
DISPOSITIONS: tuple[str, ...] = ("current", "contested", "unknown")

#: The scenario families, in the order the research lists them. The key is the
#: corpus id; the value is the required result, quoted from the research table
#: so the corpus and the design cannot drift on what a family is *for*.
FAMILIES: dict[str, str] = {
    "explicit_replacement": "B is current; A remains reachable as history",
    "proposal_after_decision": "C does not silently replace B",
    "delayed_import": "Ingestion order does not determine effective order",
    "future_effective": "Old value remains applicable until the transition",
    "staging_vs_production": "Keep both within their respective scopes",
    "policy_vs_deployment": "Report a mismatch, not a fabricated winner",
    "multi_claim_file": "Preserve the unaffected claims",
    "retraction_without_replacement": "Dependent answer becomes uncertain",
    "declared_derivation": "Derive only the declared consequence",
    "multi_source_two_hop": "Avoid both over-invalidation and stale certainty",
    "unlinked_correction": "Recover via links or structured identity",
    "archived_in_place": "Exclude from current search and session context",
    "retired_beside_active": "No retired assertion presented as current",
    "hash_valid_obsolete": "Integrity does not become a truth label",
    "touch_or_regeneration": "Does not create a new effective decision date",
    "missing_target_or_cycle": "Return bounded uncertainty",
    "hidden_linked_source": "No data or title leak",
    "concurrent_edit_index_lag": "Retry or disclose partial freshness",
    "restore_or_deletion": "No automatic resurrection of retired conclusions",
    "tight_token_budget": "Preserve the conflict or decline a settled answer",
    "claim_validity_as_of": "Deferred: needs the temporal-assertion schema",
}

#: The family whose episodes are recorded and not scored (see the module doc).
DEFERRED_FAMILY = "claim_validity_as_of"

#: Families whose correct answer follows from a *mechanical* signal the
#: resolution policy acts on — an explicit ``superseded_by`` chain, a declared
#: retirement, an ``expires_at`` on the clock, or a location-based archive.
#: These are the families a release may claim, and the only ones the CI test
#: asserts an arm ordering on.
MECHANICAL_FAMILIES: frozenset[str] = frozenset({
    "explicit_replacement",
    "archived_in_place",
    "retired_beside_active",
    "future_effective",
    "retraction_without_replacement",
    "restore_or_deletion",
})

#: Project names the authored (dev-split) episodes use.
DEV_PROJECTS: tuple[str, ...] = ("atlas", "beacon", "cinder")

#: Project names reserved for the held-out split. No authored episode names
#: one, so a held-out episode is genuinely an unseen subject.
HELD_OUT_PROJECTS: tuple[str, ...] = ("quarry", "riptide", "solstice")

#: Paraphrase templates for held-out questions. ``{q}`` is the authored
#: question with its leading interrogative removed where one exists; the
#: templates deliberately share few content words with the records, which is
#: what makes the held-out split a harder retrieval problem than the dev split.
PARAPHRASE_TEMPLATES: tuple[str, ...] = (
    "as of today, {q}",
    "what is the standing answer: {q}",
    "remind me {q}",
    "current state please — {q}",
)

_DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "episodes.yaml")


# ── schema ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    """One question and its deterministic oracle."""

    ask: str
    #: ``current`` | ``contested`` | ``unknown`` | :data:`DEFERRED_DISPOSITION`.
    disposition: str
    #: The value a correct reader answers with, or ``None`` when the correct
    #: behavior is to abstain.
    answer: str | None = None
    #: Values that must never be rendered in a current slot.
    not_current: tuple[str, ...] = ()
    #: Refs the delivered context must mention for detection to count.
    evidence: tuple[str, ...] = ()
    #: Both sides that must be present when the oracle disposition is
    #: ``contested``. Defaults to ``not_current`` plus ``answer``.
    sides: tuple[str, ...] = ()
    #: Qualifier substrings the delivered text must carry (``epistemic:``,
    #: ``contradicts:``, ``expires_at:``…).
    qualifiers: tuple[str, ...] = ()
    #: Free-text note carried into the report for a known policy gap.
    note: str | None = None

    @property
    def scored(self) -> bool:
        return self.disposition in DISPOSITIONS


@dataclass(frozen=True)
class BeforeCheck:
    """The pre-transition question an arm must already be able to answer."""

    #: Index into ``events``: the check runs after this many events have been
    #: applied (``at: 2`` → after the first two events).
    at: int
    ask: str
    answer: str


@dataclass(frozen=True)
class Episode:
    """One authored (or derived) scenario."""

    id: str
    family: str
    control: str                       # "positive" | "negative"
    project: str
    events: tuple[dict[str, Any], ...]
    questions: tuple[Question, ...]
    split: str = "dev"                 # "dev" | "held_out"
    before: BeforeCheck | None = None
    tags: tuple[str, ...] = ()
    #: Strings that must never appear in any arm's delivered context — a hidden
    #: record's title and body. Checked as a hard invariant, not a score.
    forbidden: tuple[str, ...] = ()
    #: Output budget overrides for the bundle arm, when the episode is about
    #: budget pressure.
    budget: dict[str, int] = field(default_factory=dict)
    #: Base episode id this was derived from (held-out split only).
    derived_from: str | None = None

    @property
    def mechanical(self) -> bool:
        return "mechanical" in self.tags or self.family in MECHANICAL_FAMILIES

    @property
    def consolidating(self) -> bool:
        """Does this episode run at least one consolidation pass?"""
        return any(e.get("op") == "consolidate" for e in self.events)


@dataclass(frozen=True)
class Corpus:
    """A loaded corpus: episodes plus the provenance of the load."""

    version: int
    episodes: tuple[Episode, ...]
    seed: int
    source: str

    def split(self, name: str) -> tuple[Episode, ...]:
        return tuple(e for e in self.episodes if e.split == name)


# ── loading ──────────────────────────────────────────────────────────────────


def _question(raw: dict[str, Any]) -> Question:
    sides = tuple(raw.get("sides") or ())
    if not sides and raw.get("disposition") == "contested":
        sides = tuple(
            v for v in (raw.get("answer"), *(raw.get("not_current") or ())) if v
        )
    return Question(
        ask=raw["ask"],
        disposition=raw["disposition"],
        answer=raw.get("answer"),
        not_current=tuple(raw.get("not_current") or ()),
        evidence=tuple(raw.get("evidence") or ()),
        sides=sides,
        qualifiers=tuple(raw.get("qualifiers") or ()),
        note=raw.get("note"),
    )


def _episode(raw: dict[str, Any]) -> Episode:
    before = raw.get("before")
    return Episode(
        id=raw["id"],
        family=raw["family"],
        control=raw["control"],
        project=raw["project"],
        events=tuple(raw["events"]),
        questions=tuple(_question(q) for q in raw["questions"]),
        before=BeforeCheck(at=before["at"], ask=before["ask"], answer=before["answer"])
        if before
        else None,
        tags=tuple(raw.get("tags") or ()),
        forbidden=tuple(raw.get("forbidden") or ()),
        budget=dict(raw.get("budget") or {}),
    )


def _substitute(value: Any, mapping: dict[str, str]) -> Any:
    """Rewrite every authored project name in *value* using *mapping*."""
    if isinstance(value, str):
        out = value
        for old, new in mapping.items():
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [_substitute(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, mapping) for k, v in value.items()}
    return value


def _paraphrase(ask: str, rng: random.Random) -> str:
    """A held-out phrasing of *ask*, chosen deterministically from the seed."""
    stripped = ask
    for lead in ("which ", "what ", "who ", "where ", "how "):
        if stripped.lower().startswith(lead):
            stripped = stripped[len(lead):]
            break
    return rng.choice(PARAPHRASE_TEMPLATES).format(q=stripped)


def _held_out(base: Episode, rng: random.Random) -> Episode:
    """Derive one held-out variant: unseen project, paraphrased questions.

    The *events* keep their structure and their vocabulary except for the
    project name, so the transition being measured is identical; only the
    subject and the wording of the question are unseen. That isolates
    generalization from memorization of a particular phrasing.
    """
    new_project = HELD_OUT_PROJECTS[
        DEV_PROJECTS.index(base.project) % len(HELD_OUT_PROJECTS)
    ]
    mapping = {base.project: new_project}
    events = tuple(_substitute(dict(e), mapping) for e in base.events)
    questions = tuple(
        Question(
            ask=_paraphrase(_substitute(q.ask, mapping), rng),
            disposition=q.disposition,
            answer=_substitute(q.answer, mapping),
            not_current=tuple(_substitute(list(q.not_current), mapping)),
            evidence=tuple(_substitute(list(q.evidence), mapping)),
            sides=tuple(_substitute(list(q.sides), mapping)),
            qualifiers=tuple(_substitute(list(q.qualifiers), mapping)),
            note=q.note,
        )
        for q in base.questions
    )
    before = (
        BeforeCheck(
            at=base.before.at,
            ask=_paraphrase(_substitute(base.before.ask, mapping), rng),
            answer=_substitute(base.before.answer, mapping),
        )
        if base.before
        else None
    )
    return Episode(
        id=f"{base.id}--heldout",
        family=base.family,
        control=base.control,
        project=new_project,
        events=events,
        questions=questions,
        split="held_out",
        before=before,
        tags=base.tags,
        forbidden=tuple(_substitute(list(base.forbidden), mapping)),
        budget=dict(base.budget),
        derived_from=base.id,
    )


def load_corpus(path: str | None = None, *, seed: int = 20260912) -> Corpus:
    """Load the authored episodes and derive the held-out split.

    Deterministic in ``(data file, seed)``: the held-out projects and
    paraphrases are chosen from a seeded RNG, so two runs compare like for
    like. Raises ``ValueError`` on a corpus that fails structural validation —
    an unknown family, a disposition outside the vocabulary, or a before-check
    pointing past the end of its event list.
    """
    source = path or _DEFAULT_DATA
    with open(source, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    version = int(raw.get("corpus_version", 0))
    if version != CORPUS_VERSION:
        raise ValueError(
            f"corpus_version {version} != {CORPUS_VERSION}; the harness and the "
            "data file must agree, because scores are only comparable within a version"
        )

    base = [_episode(e) for e in raw["episodes"]]
    _validate(base)

    rng = random.Random(seed)
    derived = [_held_out(e, rng) for e in base]
    return Corpus(
        version=version,
        episodes=tuple(base + derived),
        seed=seed,
        source=source,
    )


def _validate(episodes: list[Episode]) -> None:
    problems: list[str] = []
    seen: set[str] = set()
    for ep in episodes:
        if ep.id in seen:
            problems.append(f"{ep.id}: duplicate episode id")
        seen.add(ep.id)
        if ep.family not in FAMILIES:
            problems.append(f"{ep.id}: unknown family {ep.family!r}")
        if ep.control not in ("positive", "negative"):
            problems.append(f"{ep.id}: control must be positive|negative")
        if ep.project not in DEV_PROJECTS:
            problems.append(
                f"{ep.id}: project {ep.project!r} is not one of {DEV_PROJECTS} "
                "(held-out names are reserved for derived episodes)"
            )
        if not ep.questions:
            problems.append(f"{ep.id}: no questions")
        for q in ep.questions:
            if q.disposition not in DISPOSITIONS and q.disposition != DEFERRED_DISPOSITION:
                problems.append(f"{ep.id}: bad disposition {q.disposition!r}")
            if q.disposition == "current" and not q.answer:
                problems.append(f"{ep.id}: a 'current' oracle needs an answer")
            # YAML will happily hand back a float for ``answer: 0.018``; the
            # reader matches values as substrings and would crash on one.
            for value in (q.answer, *q.not_current, *q.sides, *q.qualifiers):
                if value is not None and not isinstance(value, str):
                    problems.append(
                        f"{ep.id}: value {value!r} is {type(value).__name__}, not a "
                        "string — quote it in the data file"
                    )
            if q.disposition in ("contested", "unknown") and q.answer:
                problems.append(
                    f"{ep.id}: a {q.disposition!r} oracle must not name a current answer"
                )
        if ep.before is not None and not 0 < ep.before.at <= len(ep.events):
            problems.append(f"{ep.id}: before.at={ep.before.at} is outside the event list")
    if problems:
        raise ValueError("corpus validation failed:\n  " + "\n  ".join(problems))


# ── coverage gate ────────────────────────────────────────────────────────────


def coverage_gate(corpus: Corpus, *, split: str | None = None) -> dict[str, Any]:
    """Per-family positive/negative counts and what is missing.

    The gate the acceptance criterion names: *every* required family has both
    controls. ``ok`` is False if any family has zero of either, which is the
    failure mode worth catching — a family silently dropped during a corpus
    edit looks exactly like a family nobody wrote.
    """
    episodes = corpus.episodes if split is None else corpus.split(split)
    counts = {
        family: {"positive": 0, "negative": 0, "questions": 0, "scored_questions": 0}
        for family in FAMILIES
    }
    for ep in episodes:
        bucket = counts[ep.family]
        bucket[ep.control] += 1
        bucket["questions"] += len(ep.questions)
        bucket["scored_questions"] += sum(1 for q in ep.questions if q.scored)

    missing = [
        f"{family}:{control}"
        for family, bucket in counts.items()
        for control in ("positive", "negative")
        if bucket[control] == 0
    ]
    return {
        "split": split or "all",
        "families": counts,
        "missing": missing,
        "episodes": len(episodes),
        "ok": not missing,
    }


__all__ = [
    "CORPUS_VERSION",
    "DEFERRED_DISPOSITION",
    "DEFERRED_FAMILY",
    "DEV_PROJECTS",
    "DISPOSITIONS",
    "FAMILIES",
    "HELD_OUT_PROJECTS",
    "MECHANICAL_FAMILIES",
    "BeforeCheck",
    "Corpus",
    "Episode",
    "Question",
    "coverage_gate",
    "load_corpus",
]
