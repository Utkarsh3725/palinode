"""Running the corpus: arms, controls, invariants, and the release fixtures.

Three things happen here that are not scores.

**Hard invariants.** Some properties are not allowed to be a percentage. A read
arm that mutates memory, a delivery that names a record the requester may not
see, or a retired value presented as current in a *mechanical* case is a
failure of the run, not a point off a rate. They are asserted
(:class:`InvariantFailure`) and the run stops being an acceptance run if one
fires.

**Rebuild / replay.** The contract is that markdown and git are authoritative
and SQLite is rebuildable. So every episode's store is copied, its index
deleted and rebuilt from the files alone, and the bundle re-resolved: the same
records must be selected at the same source revisions, and the receipt must
name those same revisions. That is what "explainable receipts" means
operationally — the delivery can be re-derived from the files.

**Controls.** The matched-budget control (does reading more material do what
resolution does?) is per question. The raw-evidence control (does rewriting
memory help or hurt?) is per episode: the same events, the same reader, the
same questions, once with the consolidation passes and once without them.

Arms are read-only, so one world serves all of them. That is not an
optimization — it is the reason the mutation invariant can be checked at all:
the arms run against an identical store and any divergence in ``HEAD`` or the
porcelain status between them is attributable to the arm that ran.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

from bench.current_state import arms as arms_mod
from bench.current_state import reader as reader_mod
from bench.current_state import scoring
from bench.current_state import world as world_mod
from bench.current_state.corpus import (
    DEFERRED_DISPOSITION,
    DEFERRED_FAMILY,
    FAMILIES as FAMILY_NAMES,
    MECHANICAL_FAMILIES,
    Corpus,
    Episode,
    coverage_gate,
)


class InvariantFailure(AssertionError):
    """A property that may not be a percentage was violated."""


@dataclass
class RunResult:
    """Everything one run produced."""

    corpus_version: int
    corpus_source: str
    seed: int
    coverage: dict[str, Any]
    arms: dict[str, list[scoring.EpisodeScore]] = field(default_factory=dict)
    invariants: list[dict[str, Any]] = field(default_factory=list)
    replay: list[dict[str, Any]] = field(default_factory=list)
    controls: dict[str, Any] = field(default_factory=dict)
    fixtures: dict[str, Any] = field(default_factory=dict)
    llm_reader: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def violations(self) -> list[dict[str, Any]]:
        return [i for i in self.invariants if not i["ok"]]


# ── one episode ──────────────────────────────────────────────────────────────


def _deliver(arm: str, world: Any, question_text: str, episode: Episode,
             bundle_tokens: int | None) -> arms_mod.Delivery:
    if arm == "baseline":
        return arms_mod.baseline(world, question_text)
    if arm == "projection":
        return arms_mod.projection(world, question_text)
    if arm == "bounded_evidence":
        return arms_mod.bounded_evidence(world, question_text)
    if arm == "bundle":
        return arms_mod.bundle(world, question_text, budget=episode.budget)
    if arm == "matched_budget":
        return arms_mod.matched_budget(
            world, question_text, target_tokens=bundle_tokens or 0
        )
    raise ValueError(f"unknown arm {arm!r}")


def _read(delivery: arms_mod.Delivery, question: Any, *,
          llm: bool = False) -> reader_mod.Answer:
    vocab = scoring.vocabulary(question)
    if llm:
        return reader_mod.llm_read(delivery.context, vocab, question.ask)
    return reader_mod.read(delivery.context, vocab)


def _guarded(world: Any, label: str, run: Callable[[], Any],
             invariants: list[dict[str, Any]]) -> Any:
    """Run a read arm and prove it changed nothing."""
    head_before, dirty_before = world.head(), world.dirty()
    result = run()
    head_after, dirty_after = world.head(), world.dirty()
    invariants.append({
        "invariant": "no_read_triggered_mutation",
        "where": label,
        "ok": head_before == head_after and dirty_before == dirty_after,
        "detail": {"head_before": head_before[:12], "head_after": head_after[:12],
                   "dirty_after": dirty_after},
    })
    return result


def run_episode(
    episode: Episode,
    root: str,
    *,
    arm_names: Sequence[str] = arms_mod.ARMS,
    invariants: list[dict[str, Any]] | None = None,
    skip_ops: tuple[str, ...] = (),
) -> tuple[dict[str, scoring.EpisodeScore], Any]:
    """Replay one episode and score every arm against every question.

    Returns the per-arm scores and the materialized world, so a caller that
    needs the store afterwards (the rebuild/replay check) works from the same
    object rather than reconstructing one with a different clock.
    """
    checks = invariants if invariants is not None else []
    scores = {
        arm: scoring.EpisodeScore(
            episode_id=episode.id, family=episode.family, control=episode.control,
            split=episode.split, arm=arm,
        )
        for arm in arm_names
    }

    world = world_mod.World(root, start=world_mod._episode_start(episode))
    world.setup()

    for position, event in enumerate(episode.events, start=1):
        if event.get("op") in skip_ops:
            continue
        world.apply(event)
        if episode.before is not None and episode.before.at == position:
            _before_check(world, episode, arm_names, scores, checks)

    for question in episode.questions:
        if question.disposition == DEFERRED_DISPOSITION:
            continue
        bundle_tokens: int | None = None
        for arm in arm_names:
            delivery = _guarded(
                world, f"{episode.id}:{arm}",
                # Bound as defaults: the lambda is called immediately, but a
                # late-binding closure over the loop variables is a bug waiting
                # for someone to defer the call.
                lambda arm=arm, q=question, budget=bundle_tokens: _deliver(
                    arm, world, q.ask, episode, budget
                ),
                checks,
            )
            if arm == "bundle":
                bundle_tokens = delivery.tokens
            answer = _read(delivery, question)
            score = scoring.score_question(episode, question, delivery, answer)
            scores[arm].questions.append(score)
            _record_hard_invariants(episode, question, delivery, score, checks)

    return scores, world


def _before_check(world: Any, episode: Episode, arm_names: Sequence[str],
                  scores: dict[str, scoring.EpisodeScore],
                  checks: list[dict[str, Any]]) -> None:
    """Could this arm answer the question *before* the transition?

    Without this an arm that forgets everything passes every withdrawal case
    by answering "unknown" after a retraction it never knew about.
    """
    from bench.current_state.corpus import Question

    probe = Question(
        ask=episode.before.ask,
        disposition="current",
        answer=episode.before.answer,
    )
    for arm in arm_names:
        delivery = _guarded(
            world, f"{episode.id}:{arm}:before",
            lambda arm=arm: _deliver(arm, world, probe.ask, episode, None),
            checks,
        )
        answer = _read(delivery, probe)
        scores[arm].before_ok = bool(
            answer.value and answer.value.lower() == probe.answer.lower()
        )


def _record_hard_invariants(episode: Episode, question: Any,
                            delivery: arms_mod.Delivery, score: scoring.QuestionScore,
                            checks: list[dict[str, Any]]) -> None:
    if episode.forbidden:
        checks.append({
            "invariant": "no_unauthorized_disclosure",
            "where": f"{episode.id}:{delivery.arm}",
            "ok": not score.permission_violation,
            "detail": {"markers": list(episode.forbidden)},
        })
    if delivery.arm == "bundle" and episode.family in MECHANICAL_FAMILIES:
        checks.append({
            "invariant": "no_retired_as_current",
            "where": f"{episode.id}:{delivery.arm}",
            "ok": not score.stale_current,
            "detail": {"observed": score.observed_answer,
                       "retired": list(question.not_current)},
        })


# ── rebuild / replay ─────────────────────────────────────────────────────────


def replay_check(episode: Episode, world: Any, replay_root: str) -> dict[str, Any]:
    """Drop the index on a copy, rebuild from files, re-resolve, compare.

    Equivalence is checked where it is falsifiable: which records the bundle
    selects, the raw source revision each one was delivered at, and the
    revisions the receipt recorded. A bundle that selected the same refs at
    different revisions would mean the projection is not a pure function of the
    file, which is the property this exists to test.

    The *same* clock is used on both sides. Re-deriving one would make a
    difference in ``now`` indistinguishable from a difference in the rebuild.
    """
    if os.path.exists(replay_root):
        shutil.rmtree(replay_root)
    shutil.copytree(world.root, replay_root)

    question = episode.questions[0]
    world.attach()
    before = arms_mod.bundle(world, question.ask, budget=episode.budget)

    replayed = world_mod.World(replay_root, start=world.now)
    replayed.attach()
    replayed.apply({"op": "drop_index"})
    after = arms_mod.bundle(replayed, question.ask, budget=episode.budget)

    same_selection = before.extra.get("selected") == after.extra.get("selected")
    same_revisions = before.extra.get("source_revisions") == after.extra.get("source_revisions")
    receipt_revisions = _receipt_revisions(after.extra.get("receipt"))
    receipts_explain = all(
        after.extra.get("source_revisions", {}).get(ref) == revision
        for ref, revision in receipt_revisions.items()
        if ref in after.extra.get("source_revisions", {})
    )
    # An episode that deliberately left the index behind the file must NOT
    # replay identically — the rebuild is supposed to catch the file up. There
    # the passing condition is inverted, which is the strongest available test
    # that the index really is derived state.
    expect_divergence = "stale_index" in episode.tags
    return {
        "episode_id": episode.id,
        "ok": (
            same_selection
            and receipts_explain
            and (same_revisions is not expect_divergence)
        ),
        "same_selection": same_selection,
        "same_revisions": same_revisions,
        "expected_divergence": expect_divergence,
        "receipts_explain_revisions": receipts_explain,
        "selected": after.extra.get("selected"),
        "receipt_records": len(receipt_revisions),
    }


def _receipt_revisions(receipt: dict[str, Any] | None) -> dict[str, str]:
    """``ref → revision`` from a delivery receipt's public ``supplied`` list."""
    if not isinstance(receipt, dict):
        return {}
    out: dict[str, str] = {}
    for row in receipt.get("supplied") or []:
        if not isinstance(row, dict):
            continue
        ref, revision = row.get("ref"), row.get("revision")
        if isinstance(ref, str) and isinstance(revision, str):
            out[ref] = revision
    return out


# ── the whole run ────────────────────────────────────────────────────────────


def run(
    corpus: Corpus,
    *,
    arm_names: Sequence[str] = arms_mod.ARMS,
    limit: int | None = None,
    splits: Sequence[str] = ("dev", "held_out"),
    replay: bool = True,
    controls: bool = True,
    fixtures: bool = True,
    resamples: int = scoring.BOOTSTRAP_RESAMPLES,
    workdir: str | None = None,
) -> RunResult:
    """Run the corpus end to end and return everything the report needs."""
    episodes = [e for e in corpus.episodes if e.split in splits]
    if limit is not None:
        episodes = _balanced_slice(episodes, limit)

    result = RunResult(
        corpus_version=corpus.version,
        corpus_source=corpus.source,
        seed=corpus.seed,
        coverage=coverage_gate(corpus, split="dev"),
        environment=_environment(),
    )
    result.arms = {arm: [] for arm in arm_names}

    tmp = workdir or tempfile.mkdtemp(prefix="palinode-current-state-")
    created = workdir is None
    try:
        with world_mod.deterministic_embedder():
            for episode in episodes:
                root = os.path.join(tmp, episode.id)
                scores, built = run_episode(
                    episode, root, arm_names=arm_names, invariants=result.invariants
                )
                for arm, score in scores.items():
                    # An episode whose questions are all deferred contributes
                    # coverage, not a rate: folding an empty episode in as a
                    # zero would understate every arm by the same amount and
                    # still be wrong.
                    if score.questions:
                        result.arms[arm].append(score)
                if replay:
                    result.replay.append(
                        replay_check(episode, built, os.path.join(tmp, f"{episode.id}-replay"))
                    )
                if not _keep_worlds():
                    shutil.rmtree(root, ignore_errors=True)
                    shutil.rmtree(os.path.join(tmp, f"{episode.id}-replay"), ignore_errors=True)

            if controls:
                result.controls = _run_controls(
                    episodes, tmp, arm_names=arm_names, resamples=resamples
                )
            if fixtures:
                result.fixtures = run_release_fixtures(corpus, tmp)
    finally:
        if created and not _keep_worlds():
            shutil.rmtree(tmp, ignore_errors=True)

    result.invariants.extend(_replay_invariants(result.replay))
    return result


def _keep_worlds() -> bool:
    return os.environ.get("PALINODE_BENCH_KEEP_WORLDS") == "1"


def _balanced_slice(episodes: Sequence[Episode], limit: int) -> list[Episode]:
    """A slice that keeps one positive and one negative per family as far as it
    goes — a slice that happened to be all positives would make the CI
    assertions meaningless."""
    by_family: dict[tuple[str, str], list[Episode]] = {}
    for episode in episodes:
        by_family.setdefault((episode.family, episode.control), []).append(episode)
    chosen: list[Episode] = []
    for key in sorted(by_family):
        chosen.append(by_family[key][0])
        if len(chosen) >= limit:
            break
    return chosen


def _replay_invariants(replays: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "invariant": "rebuild_replay_equivalent",
        "where": r["episode_id"],
        "ok": r["ok"],
        "detail": {k: r[k] for k in ("same_selection", "same_revisions",
                                     "expected_divergence",
                                     "receipts_explain_revisions")},
    } for r in replays]


def _environment() -> dict[str, Any]:
    import platform

    from palinode.core.config import config

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedder": "deterministic-hashed-bow",
        "embedding_dims": int(config.embeddings.primary.dimensions),
    }


# ── controls ─────────────────────────────────────────────────────────────────


def _run_controls(episodes: Sequence[Episode], tmp: str, *,
                  arm_names: Sequence[str], resamples: int) -> dict[str, Any]:
    """Raw-evidence retention versus consolidation, on matched episodes."""
    matched = [e for e in episodes if e.consolidating]
    control_arms = [a for a in arm_names if a in ("baseline", "bundle")] or ["bundle"]
    consolidated: dict[str, list[scoring.EpisodeScore]] = {a: [] for a in control_arms}
    raw: dict[str, list[scoring.EpisodeScore]] = {a: [] for a in control_arms}

    for episode in matched:
        with_pass, _ = run_episode(
            episode, os.path.join(tmp, f"{episode.id}-consolidated"),
            arm_names=control_arms,
        )
        without, _ = run_episode(
            episode, os.path.join(tmp, f"{episode.id}-raw"),
            arm_names=control_arms, skip_ops=("consolidate",),
        )
        for arm in control_arms:
            consolidated[arm].append(with_pass[arm])
            raw[arm].append(without[arm])
        if not _keep_worlds():
            for suffix in ("consolidated", "raw"):
                shutil.rmtree(os.path.join(tmp, f"{episode.id}-{suffix}"), ignore_errors=True)

    return {
        "raw_vs_consolidation": {
            "episodes": [e.id for e in matched],
            "consolidated": {
                a: scoring.summarize(v, resamples=resamples) for a, v in consolidated.items()
            },
            "raw_retention": {
                a: scoring.summarize(v, resamples=resamples) for a, v in raw.items()
            },
        },
        "repeated_consolidation": [
            e.id for e in episodes if "repeated_consolidation" in e.tags
        ],
    }


# ── release fixtures ─────────────────────────────────────────────────────────


def run_release_fixtures(corpus: Corpus, tmp: str) -> dict[str, Any]:
    """The end-to-end acceptance fixtures, including the *actual* hook script.

    A fixture is tagged in the corpus, not written twice: the same events the
    scored run replays are replayed here, and then the delivery goes through a
    live API server and the shipped ``UserPromptSubmit`` script so what the
    reader sees is what a session would actually have been handed.
    """
    tagged = {
        tag: [e for e in corpus.split("dev") if tag in e.tags]
        for tag in ("release_fixture", "repeated_consolidation", "lineage")
    }
    out: dict[str, Any] = {"hook": [], "lineage": [], "repeated_consolidation": []}

    hook_dir = os.path.join(tmp, "hook")
    os.makedirs(hook_dir, exist_ok=True)
    script = arms_mod.hook_script_path(hook_dir)

    for episode in tagged["release_fixture"]:
        root = os.path.join(tmp, f"fixture-{episode.id}")
        world = world_mod.World(root, start=world_mod._episode_start(episode))
        world.setup()
        for event in episode.events:
            world.apply(event)
        question = episode.questions[0]
        env = {
            "PALINODE_HOOK_RECALL_TRIGGERS": "0",
            "PALINODE_HOOK_RESOLVE_DEADLINE": str(episode.budget.get("deadline_ms", 5000)),
            "PALINODE_HOOK_RECALL_MAX_CHARS": str(episode.budget.get("hook_max_chars", 3000)),
        }
        try:
            with arms_mod.live_api() as url:
                delivery = arms_mod.hook_context(
                    question.ask, api_url=url, script=script, env=env
                )
            answer = _read(delivery, question)
            score = scoring.score_question(episode, question, delivery, answer)
            out["hook"].append({
                "episode_id": episode.id,
                "ask": question.ask,
                "expected": {"disposition": question.disposition, "answer": question.answer},
                "observed": {"disposition": answer.disposition, "answer": answer.value,
                             "sides": list(answer.sides), "refs": list(answer.cited_refs)},
                "ok": score.disposition and score.behavior and score.presentation,
                "detection": score.detection,
                "context_chars": len(delivery.context),
                "hook_exit": delivery.extra.get("hook_exit"),
                "injected": delivery.context,
            })
        except Exception as exc:  # noqa: BLE001 — a fixture that cannot run is reported
            out["hook"].append({
                "episode_id": episode.id, "ok": False, "error": f"{type(exc).__name__}: {exc}",
            })
        if not _keep_worlds():
            shutil.rmtree(root, ignore_errors=True)

    for episode in tagged["lineage"]:
        out["lineage"].append(_lineage_fixture(episode, os.path.join(tmp, f"lineage-{episode.id}")))
    for episode in tagged["repeated_consolidation"]:
        out["repeated_consolidation"].append(
            _repeat_fixture(episode, os.path.join(tmp, f"repeat-{episode.id}"))
        )
    return out


def _lineage_fixture(episode: Episode, root: str) -> dict[str, Any]:
    """One observation copied into several summaries is one lineage group.

    Checked on the resolution layer, because that is where grouping lives: the
    bundle reports refs, the policy reports origins.
    """
    from palinode.core.evidence import resolve_evidence
    from palinode.core.resolution import resolve

    world = world_mod.World(root, start=world_mod._episode_start(episode))
    world.setup()
    for event in episode.events:
        world.apply(event)
    rows, _ = arms_mod.seed_rows(episode.questions[0].ask, arms_mod.DEFAULT_TOP_K)
    evidence = resolve_evidence(rows, mode="full", now=world.now)
    groups: list[dict[str, Any]] = []
    for seed in evidence.seeds:
        resolution = resolve(seed.seed_meta, seed, now=world.now)
        for group in resolution.support:
            groups.append({
                "origin": group.origin,
                "origin_kind": group.origin_kind,
                "members": [m.ref for m in group.members],
            })
    anchored = [g for g in groups if g["origin_kind"] in ("claim", "source", "backed_by")]
    largest = max((len(g["members"]) for g in anchored), default=0)

    # The same grouping as the shipped delivery receipt reports it, so the
    # fixture checks the surface a consumer actually sees, not only the policy.
    delivered = arms_mod.bundle(world, episode.questions[0].ask, budget=episode.budget)
    receipt_lineage = [
        g for g in ((delivered.extra.get("receipt") or {}).get("lineage") or [])
        if g.get("status") == "known"
    ]
    if not _keep_worlds():
        shutil.rmtree(root, ignore_errors=True)
    return {
        "episode_id": episode.id,
        "ok": len(anchored) >= 1 and largest >= 2,
        "anchored_groups": anchored,
        "copies_in_largest_group": largest,
        "receipt_lineage_groups": receipt_lineage,
    }


def _repeat_fixture(episode: Episode, root: str) -> dict[str, Any]:
    """Three consolidation cycles, a delayed import and an explicit restore.

    Asserts the two halves that matter together: a retired value never comes
    back as current, and an unrelated claim in the same store stays answerable.
    A system that survives the first by forgetting everything fails the second.
    """
    world = world_mod.World(root, start=world_mod._episode_start(episode))
    world.setup()
    for event in episode.events:
        world.apply(event)
    per_question = []
    for question in episode.questions:
        delivery = arms_mod.bundle(world, question.ask, budget=episode.budget)
        answer = _read(delivery, question)
        score = scoring.score_question(episode, question, delivery, answer)
        per_question.append({
            "ask": question.ask,
            "expected": question.answer,
            "observed": answer.value,
            "disposition_ok": score.disposition,
            "behavior_ok": score.behavior,
            "stale_current": score.stale_current,
        })
    if not _keep_worlds():
        shutil.rmtree(root, ignore_errors=True)
    return {
        "episode_id": episode.id,
        "ok": all(q["behavior_ok"] and not q["stale_current"] for q in per_question),
        "questions": per_question,
    }


# ── the optional model-backed reader arm ─────────────────────────────────────


def run_llm_reader(corpus: Corpus, result: RunResult, *,
                   per_family: int = 1, require_coverage: bool = True,
                   workdir: str | None = None) -> dict[str, Any]:
    """Run the model-backed reader over a family-stratified sample.

    Stratified by family on purpose. A model-backed reader is the only
    non-deterministic component this harness can contain, and the rule it has
    to obey is the one the write-time contradiction judge broke: a judge
    validated on clean isolated pairs and then trusted on a real conversation.
    So the sample is drawn one episode per scored family, from the same
    realistic mixed-claim records the scored run uses — never a purpose-built
    pair set — and with ``require_coverage`` (the default) a family with zero
    coverage means **no number is produced at all** rather than a number
    quietly computed over the families that happened to be sampled.

    If the endpoint cannot be reached the arm is reported **unrun**, with the
    transport error. An unrun arm is a result; an invented one is not.
    """
    scored_families = {f for f in FAMILY_NAMES if f != DEFERRED_FAMILY}
    sample: list[Episode] = []
    seen: set[str] = set()
    for episode in corpus.split("dev"):
        key = episode.family
        if sum(1 for e in sample if e.family == key) >= per_family:
            continue
        if not any(q.disposition != DEFERRED_DISPOSITION for q in episode.questions):
            continue
        sample.append(episode)
        seen.add(key)

    tmp = workdir or tempfile.mkdtemp(prefix="palinode-current-state-llm-")
    rows: list[dict[str, Any]] = []
    try:
        with world_mod.deterministic_embedder():
            for episode in sample:
                world = world_mod.World(
                    os.path.join(tmp, f"llm-{episode.id}"),
                    start=world_mod._episode_start(episode),
                )
                world.setup()
                for event in episode.events:
                    world.apply(event)
                for question in episode.questions:
                    if question.disposition == DEFERRED_DISPOSITION:
                        continue
                    delivery = arms_mod.bundle(
                        world, question.ask, budget=episode.budget
                    )
                    rule = _read(delivery, question)
                    try:
                        model = _read(delivery, question, llm=True)
                    except reader_mod.LlmReaderUnavailable as exc:
                        return {
                            "status": "unreachable",
                            "detail": str(exc),
                            "families_sampled": sorted(seen),
                        }
                    rows.append({
                        "episode_id": episode.id,
                        "family": episode.family,
                        "expected": question.disposition,
                        "rule": rule.disposition,
                        "model": model.disposition,
                        "rule_value": rule.value,
                        "model_value": model.value,
                    })
                shutil.rmtree(world.root, ignore_errors=True)
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        return {"status": "empty", "families_sampled": sorted(seen)}

    missing = sorted(scored_families - seen)
    base = {
        "model": os.environ.get(reader_mod.ENV_MODEL),
        "questions": len(rows),
        "families_sampled": sorted(seen & scored_families),
        "families_missing": missing,
        "families_required": len(scored_families),
        "disagreements": [r for r in rows if r["rule"] != r["model"]],
        "rows": rows,
    }
    if missing and require_coverage:
        return {
            **base,
            "status": "coverage_gap",
            "detail": (
                "the sample covered no episode of "
                + ", ".join(missing)
                + "; no agreement number is reported for a reader with zero "
                  "coverage on a scored family"
            ),
        }
    agree_disposition = sum(1 for r in rows if r["rule"] == r["model"]) / len(rows)
    agree_value = sum(
        1 for r in rows
        if (r["rule_value"] or "").lower() == (r["model_value"] or "").lower()
    ) / len(rows)
    return {
        **base,
        "status": "run",
        "agreement_disposition": round(agree_disposition, 3),
        "agreement_value": round(agree_value, 3),
    }


__all__ = [
    "InvariantFailure",
    "RunResult",
    "run_llm_reader",
    "replay_check",
    "run",
    "run_episode",
    "run_release_fixtures",
]
