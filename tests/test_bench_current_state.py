"""The current-state evaluation harness, exercised in bounded time.

What CI has to protect is not the headline number — that lives in the
acceptance report and moves with the corpus — but the machinery that makes the
number mean anything:

* the corpus loads, validates, and every scenario family still has both
  controls (the coverage gate);
* the held-out split is derived deterministically and uses unseen projects;
* an episode materializes through the **real** store, executor, consolidation
  runner and resolution path, and every arm reads that store without changing
  it;
* the mechanical invariants hold on a slice — no read-triggered mutation, no
  unauthorized disclosure, no retired value presented as current;
* the bundle arm beats the plain-search baseline on disposition for the
  mechanical families, which is the one ordering the release claims.

The full run (88 episodes, five arms, the live-server fixtures) is marked
``slow``. Everything else here is a small balanced slice.
"""
from __future__ import annotations

import os

import pytest

from bench.current_state import arms as arms_mod
from bench.current_state import harness, reader, scoring
from bench.current_state import world as world_mod
from bench.current_state.corpus import (
    CORPUS_VERSION,
    DEFERRED_DISPOSITION,
    DEV_PROJECTS,
    FAMILIES,
    HELD_OUT_PROJECTS,
    MECHANICAL_FAMILIES,
    coverage_gate,
    load_corpus,
)

#: Small enough for CI, wide enough that the mechanical assertion has both a
#: positive and a negative control of more than one family in it.
SLICE = 10


def _config_snapshot():
    from palinode.core import store
    from palinode.core.config import config

    return (
        config.memory_dir,
        config.db_path,
        store._db_checked,
        config.git.auto_commit,
        os.environ.get("PALINODE_ALLOW_FRESH_DB"),
    )


def _config_restore(snap) -> None:
    from palinode.core import store
    from palinode.core.config import config

    (config.memory_dir, config.db_path, store._db_checked,
     config.git.auto_commit, fresh) = snap
    if fresh is None:
        os.environ.pop("PALINODE_ALLOW_FRESH_DB", None)
    else:
        os.environ["PALINODE_ALLOW_FRESH_DB"] = fresh


@pytest.fixture(autouse=True, scope="module")
def _restore_global_config_for_module():
    """Restore the process-wide state ``point_config_at`` mutates.

    Module-scoped **and autouse** deliberately: a higher-scoped fixture is set
    up before a function-scoped one, so a function-scoped snapshot taken after
    the module-scoped ``sliced_run`` had already re-pointed the config would
    "restore" the mutated values and leak them into the rest of the suite.
    ``git.auto_commit`` is in the snapshot for the same reason — the bench
    harness turns it off, and three unrelated suites assert that commits happen.
    """
    snap = _config_snapshot()
    try:
        yield
    finally:
        _config_restore(snap)


@pytest.fixture(autouse=True)
def _restore_global_config():
    """Per-test restore, so one test in this file cannot bleed into the next."""
    snap = _config_snapshot()
    try:
        yield
    finally:
        _config_restore(snap)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def sliced_run(corpus, tmp_path_factory):
    """One bounded run, shared by the assertions that read its output."""
    workdir = str(tmp_path_factory.mktemp("current-state"))
    return harness.run(
        corpus,
        limit=SLICE,
        splits=("dev",),
        replay=True,
        controls=False,
        fixtures=False,
        resamples=200,
        workdir=workdir,
    )


# ── corpus ────────────────────────────────────────────────────────────────


def test_corpus_version_and_size(corpus):
    assert corpus.version == CORPUS_VERSION
    authored = corpus.split("dev")
    # The design calls for roughly 80–120 episodes across both splits.
    assert 80 <= len(corpus.episodes) <= 120, len(corpus.episodes)
    assert len(authored) == len(corpus.split("held_out"))


def test_every_family_has_both_controls(corpus):
    """The coverage gate. A family that vanished must fail the build."""
    gate = coverage_gate(corpus, split="dev")
    assert gate["ok"], f"families missing a control: {gate['missing']}"
    assert set(gate["families"]) == set(FAMILIES)


def test_coverage_gate_fails_when_a_family_loses_a_control(corpus):
    """The gate is a gate — prove it can say no."""
    from dataclasses import replace

    from bench.current_state.corpus import Corpus

    victim = next(e for e in corpus.split("dev") if e.control == "negative")
    thinned = Corpus(
        version=corpus.version,
        episodes=tuple(
            e for e in corpus.episodes
            if not (e.family == victim.family and e.control == "negative")
        ),
        seed=corpus.seed,
        source=corpus.source,
    )
    gate = coverage_gate(thinned, split="dev")
    assert not gate["ok"]
    assert f"{victim.family}:negative" in gate["missing"]
    assert replace(victim, id="unused").family == victim.family


def test_held_out_split_uses_unseen_projects_and_is_deterministic(corpus):
    held = corpus.split("held_out")
    assert held
    assert {e.project for e in held} <= set(HELD_OUT_PROJECTS)
    assert not ({e.project for e in held} & set(DEV_PROJECTS))
    again = load_corpus(seed=corpus.seed)
    assert [e.questions[0].ask for e in again.split("held_out")] == [
        e.questions[0].ask for e in held
    ]


def test_deferred_family_is_recorded_and_never_scored(corpus):
    deferred = [
        q for e in corpus.episodes for q in e.questions
        if q.disposition == DEFERRED_DISPOSITION
    ]
    assert deferred, "the deferred claim-validity family must stay in the corpus"
    assert all(not q.scored for q in deferred)


# ── the world drives the real pipeline ────────────────────────────────────


def test_an_episode_materializes_through_the_real_executor(tmp_path, corpus):
    """A consolidation episode really runs the runner, the guard and the executor.

    Asserted on the file, not on a return value: the retired wording is struck
    in place, the successor is on the page, and the index carries the projected
    text rather than the tombstone.
    """
    from palinode.core import store

    episode = next(e for e in corpus.split("dev") if e.id == "explicit-replacement-pos")
    with world_mod.deterministic_embedder():
        world = world_mod.World(str(tmp_path / "w"),
                                start=world_mod._episode_start(episode))
        world.setup()
        for event in episode.events:
            world.apply(event)

        raw = world.read("projects/atlas-status")
        assert "~~[2026-01-10] The atlas cache backend is Memcached.~~" in raw
        assert "The atlas cache backend is Redis." in raw

        db = store.get_db()
        try:
            rows = db.execute(
                "SELECT content FROM chunks WHERE file_path LIKE '%atlas-status.md'"
            ).fetchall()
        finally:
            db.close()
        indexed = "\n".join(r["content"] for r in rows)
        assert "Redis" in indexed
        assert "Memcached" not in indexed, "the projection must drop the tombstone"


def test_a_consolidation_event_that_changes_nothing_is_loud(tmp_path, corpus):
    """A pass asked to apply ops that reach no target raises, never no-ops."""
    with world_mod.deterministic_embedder():
        world = world_mod.World(str(tmp_path / "w"))
        world.setup()
        with pytest.raises(ValueError, match="compacted no project"):
            world.apply({"op": "consolidate", "pass": "nightly",
                         "ops": [{"op": "SUPERSEDE", "id": "nope",
                                  "new_text": "x", "rationale": "y"}]})


# ── the run ───────────────────────────────────────────────────────────────


def test_slice_runs_every_arm_over_every_question(sliced_run):
    assert set(sliced_run.arms) == set(arms_mod.ARMS)
    counts = {arm: sum(len(e.questions) for e in eps)
              for arm, eps in sliced_run.arms.items()}
    assert len(set(counts.values())) == 1, (
        f"arms must answer the same questions: {counts}"
    )
    assert all(v > 0 for v in counts.values())


def test_mechanical_invariants_hold(sliced_run):
    """No read-triggered mutation, no disclosure, no retired-as-current."""
    assert sliced_run.invariants, "the run recorded no invariant checks at all"
    kinds = {c["invariant"] for c in sliced_run.invariants}
    assert "no_read_triggered_mutation" in kinds
    assert not sliced_run.violations, sliced_run.violations


def test_rebuild_replay_is_equivalent(sliced_run):
    """Dropping the index and rebuilding from the files re-derives the delivery."""
    assert sliced_run.replay
    for entry in sliced_run.replay:
        assert entry["ok"], entry
        assert entry["receipts_explain_revisions"], entry


def test_bundle_beats_baseline_on_disposition_for_mechanical_families(sliced_run):
    """The one arm ordering the release claims, on the families it claims."""
    def mechanical(arm: str) -> float:
        episodes = [e for e in sliced_run.arms[arm] if e.family in MECHANICAL_FAMILIES]
        assert episodes, f"the slice contains no mechanical family for {arm}"
        return scoring.summarize(episodes, resamples=100)["disposition"]["mean"]

    bundle, baseline = mechanical("bundle"), mechanical("baseline")
    assert bundle >= baseline, f"bundle {bundle} < baseline {baseline}"
    assert scoring.summarize(
        [e for e in sliced_run.arms["bundle"] if e.family in MECHANICAL_FAMILIES],
        resamples=100,
    )["stale_current"]["mean"] == 0.0


def test_retired_wording_never_reaches_the_bundle_reader(tmp_path, corpus):
    """The motivating case, asserted directly rather than as a rate.

    A fact retired in place still sits in the file next to its replacement.
    The baseline reads the file and takes the retired value; the bundle reads
    the projection and takes the current one.
    """
    episode = next(e for e in corpus.split("dev") if e.id == "retired-beside-active-pos")
    question = episode.questions[0]
    vocab = scoring.vocabulary(question)
    with world_mod.deterministic_embedder():
        world = world_mod.World(str(tmp_path / "w"),
                                start=world_mod._episode_start(episode))
        world.setup()
        for event in episode.events:
            world.apply(event)

        baseline = reader.read(arms_mod.baseline(world, question.ask).context, vocab)
        bundled = reader.read(arms_mod.bundle(world, question.ask).context, vocab)

    assert baseline.value == "vim keybindings", "the raw-text arm should be fooled"
    assert bundled.value is None and bundled.disposition == "unknown"


def test_before_state_checks_ran_and_passed(sliced_run):
    """An arm that knew nothing beforehand gets no credit for abstaining after."""
    checked = [
        e for eps in sliced_run.arms.values() for e in eps if e.before_ok is not None
    ]
    assert checked, "no episode in the slice declared a before-state check"
    assert all(e.before_ok for e in checked), [
        e.episode_id for e in checked if not e.before_ok
    ]


def test_report_renders_from_a_run(sliced_run):
    from bench.current_state import report

    payload = report.summarize_run(sliced_run, resamples=100, splits=("dev",))
    rendered = report.render(payload)
    assert "Scenario-family coverage (gate)" in rendered
    assert "Release claims supported / not supported" in rendered
    assert "### Not supported" in rendered
    assert "Model-backed reader arm" in rendered


# ── the full run ──────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_full_corpus_run(tmp_path):
    """Both splits, every arm, the controls and the live-server fixtures.

    Roughly a minute of wall clock — well past the suite-wide 60 s timeout,
    which is why it carries its own and is deselected by default.
    """
    from bench.current_state import report

    corpus = load_corpus()
    result = harness.run(
        corpus, splits=("dev", "held_out"), resamples=300,
        workdir=str(tmp_path / "full"),
    )
    assert not result.violations, result.violations
    payload = report.summarize_run(result, resamples=300)
    assert payload["coverage"]["ok"]
    hooks = payload["fixtures"]["hook"]
    assert hooks, "the release fixtures did not run"
    assert all(entry.get("ok") for entry in hooks), hooks
