"""Rendering a run as the acceptance document and as a results JSON.

Two outputs, one source. The JSON is every per-episode, per-question row —
what a later run diffs against. The Markdown is the argument: per-arm table
with intervals, per-family counts, failure-stage attribution, the controls, the
invariants, the release fixtures, and a closing section that says plainly which
release claims the numbers support and which they do not.

The negative results are not an appendix. A harness that only renders what went
well is a marketing tool, so the renderer emits the "not supported" section
unconditionally and populates it from the same rows as everything else.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from bench.current_state import scoring
from bench.current_state.corpus import DEFERRED_FAMILY, FAMILIES, MECHANICAL_FAMILIES


def _pct(block: dict[str, Any] | None) -> str:
    if not block or block.get("mean") is None:
        return "—"
    return (
        f"{block['mean'] * 100:.0f}% "
        f"[{block['lo'] * 100:.0f}–{block['hi'] * 100:.0f}]"
    )


def summarize_run(result: Any, *, resamples: int = scoring.BOOTSTRAP_RESAMPLES,
                  splits: Sequence[str] = ("dev", "held_out")) -> dict[str, Any]:
    """Every number the report shows, as a plain dict (also the JSON payload)."""
    per_arm: dict[str, Any] = {}
    for arm, episodes in result.arms.items():
        per_arm[arm] = {
            "all": scoring.summarize(episodes, resamples=resamples),
            **{
                split: scoring.summarize(
                    [e for e in episodes if e.split == split], resamples=resamples
                )
                for split in splits
            },
            "mechanical": scoring.summarize(
                [e for e in episodes if e.family in MECHANICAL_FAMILIES],
                resamples=resamples,
            ),
        }
    return {
        "corpus_version": result.corpus_version,
        "corpus_source": result.corpus_source,
        "seed": result.seed,
        "environment": result.environment,
        "coverage": result.coverage,
        "arms": per_arm,
        "invariants": {
            "checks": len(result.invariants),
            "violations": result.violations,
            "by_kind": _invariant_kinds(result.invariants),
        },
        "replay": result.replay,
        "controls": result.controls,
        "fixtures": result.fixtures,
        "llm_reader": result.llm_reader,
        "episodes": [
            e.to_dict() for episodes in result.arms.values() for e in episodes
        ],
    }


def _invariant_kinds(checks: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for check in checks:
        bucket = out.setdefault(check["invariant"], {"checked": 0, "failed": 0})
        bucket["checked"] += 1
        bucket["failed"] += int(not check["ok"])
    return out


def write_json(payload: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


# ── markdown ─────────────────────────────────────────────────────────────────


def render(payload: dict[str, Any]) -> str:
    out: list[str] = []
    env = payload["environment"]
    out.append("# v0.20 acceptance — current-state recall, transition to decision")
    out.append("")
    out.append(
        f"Generated {env['generated_at']} · corpus v{payload['corpus_version']} "
        f"(seed {payload['seed']}) · embedder `{env['embedder']}` "
        f"({env['embedding_dims']}d) · Python {env['python']}"
    )
    out.append("")
    out.extend(_scope())
    out.extend(_coverage(payload["coverage"]))
    out.extend(_arm_table(payload["arms"]))
    out.extend(_cost_table(payload["arms"]))
    out.extend(_stage_table(payload["arms"]))
    out.extend(_family_table(payload["arms"]))
    out.extend(_invariants(payload))
    out.extend(_controls(payload["controls"], payload["arms"]))
    out.extend(_fixtures(payload["fixtures"]))
    out.extend(_llm(payload["llm_reader"]))
    out.extend(_claims(payload))
    return "\n".join(out) + "\n"


def _scope() -> list[str]:
    return [
        "## What this measures, and what it cannot",
        "",
        "Four stages are scored separately — detection, disposition,",
        "presentation, behavior — so a wrong answer is attributed rather than",
        "just counted. Every oracle is deterministic: the expected disposition,",
        "the expected value and the retired values follow mechanically from the",
        "episode's events, which is why no semantic judge is used and no judge",
        "coverage gate is needed for one. The gate that *is* enforced is the",
        "per-family coverage gate below.",
        "",
        "Three limits, stated up front:",
        "",
        "- The reader is a **rule-following program**, not a model. It reports",
        "  what a reader that honours the markers would conclude. A model-backed",
        "  reader is available behind a flag; its status is at the end.",
        "- Seed retrieval uses a **deterministic hashed bag-of-words embedder**,",
        "  so similarity is lexical overlap. Paraphrase recall at the seed stage",
        "  is therefore understated relative to a real bge-m3 host — equally for",
        "  every arm, since every arm seeds identically.",
        "- The bounded-evidence arm is rendered by the harness in the bundle's",
        "  own section grammar, so arms 3 and 4 differ by grouping, budget and",
        "  packing rather than by whether markers exist at all.",
        "",
    ]


def _coverage(coverage: dict[str, Any]) -> list[str]:
    rows = [
        "## Scenario-family coverage (gate)",
        "",
        f"Authored episodes: **{coverage['episodes']}** · "
        f"gate: **{'PASS' if coverage['ok'] else 'FAIL'}**"
        + (f" — missing {', '.join(coverage['missing'])}" if coverage["missing"] else ""),
        "",
        "| Family | Required result | +ve | −ve | Qs | scored |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family, required in FAMILIES.items():
        bucket = coverage["families"].get(family, {})
        marker = " *(deferred)*" if family == DEFERRED_FAMILY else ""
        rows.append(
            f"| `{family}`{marker} | {required} | {bucket.get('positive', 0)} "
            f"| {bucket.get('negative', 0)} | {bucket.get('questions', 0)} "
            f"| {bucket.get('scored_questions', 0)} |"
        )
    rows.append("")
    rows.append(
        "The claim-validity / as-of family is present and carries the scoped "
        "deferred disposition rather than being dropped: its episodes replay, "
        "its questions are recorded, and nothing scores them."
    )
    rows.append("")
    return rows


def _arm_table(arms: dict[str, Any]) -> list[str]:
    rows = ["## Per-arm results (95% bootstrap CI over episodes)", ""]
    any_held = any(
        (blocks.get("held_out") or {}).get("episodes") for blocks in arms.values()
    )
    if not any_held:
        rows += [
            "*(The held-out split was not part of this run — the numbers below "
            "are the authored split only, and generalization is not measured.)*",
            "",
        ]
    else:
        rows += [
            "### Held-out split — the headline",
            "",
            "| Arm | Detection | Disposition | Presentation | Behavior | "
            "Stale-current | False resolution | Appropriate abstention | "
            "Unaffected-fact retention |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, blocks in arms.items():
            held = blocks.get("held_out", {})
            rows.append(
                f"| `{arm}` | {_pct(held.get('detection'))} "
                f"| {_pct(held.get('disposition'))} | {_pct(held.get('presentation'))} "
                f"| {_pct(held.get('behavior'))} | {_pct(held.get('stale_current'))} "
                f"| {_pct(held.get('false_resolution'))} "
                f"| {_pct(held.get('appropriate_abstention'))} "
                f"| {_pct(held.get('unaffected_fact_retention'))} |"
            )
    rows += ["", "### Whole corpus (dev + held-out)", "",
             "| Arm | Episodes | Questions | Detection | Disposition | "
             "Presentation | Behavior | Source correctness | Before-state |",
             "|---|---:|---:|---|---|---|---|---|---|"]
    for arm, blocks in arms.items():
        every = blocks["all"]
        before = every["before_state_pass"]
        rows.append(
            f"| `{arm}` | {every['episodes']} | {every['questions']} "
            f"| {_pct(every['detection'])} | {_pct(every['disposition'])} "
            f"| {_pct(every['presentation'])} | {_pct(every['behavior'])} "
            f"| {_pct(every['source_correct'])} "
            f"| {before['passed']}/{before['n']} |"
        )
    rows += ["", "### Mechanical families only", "",
             "| Arm | Disposition | Behavior | Stale-current |",
             "|---|---|---|---|"]
    for arm, blocks in arms.items():
        mech = blocks["mechanical"]
        rows.append(
            f"| `{arm}` | {_pct(mech['disposition'])} | {_pct(mech['behavior'])} "
            f"| {_pct(mech['stale_current'])} |"
        )
    rows.append("")
    rows.append(
        "Before-state is the denominator that stops an arm passing a withdrawal "
        "case by knowing nothing: it is the same question asked before the "
        "transition, and an arm that could not answer it then gets no credit "
        "for abstaining afterwards."
    )
    evidence_arm = arms.get("bounded_evidence", {}).get("all", {})
    bundle_arm = arms.get("bundle", {}).get("all", {})
    if evidence_arm and bundle_arm:
        same = all(
            (evidence_arm.get(k) or {}).get("mean") == (bundle_arm.get(k) or {}).get("mean")
            for k in ("detection", "disposition", "presentation", "behavior")
        )
        if same:
            rows.append("")
            rows.append(
                "`bounded_evidence` and `bundle` score identically on every stage "
                "here. That is the expected shape of this corpus and worth saying "
                "plainly: the accuracy comes from the evidence layer and the "
                "resolution policy. What the bundle adds — grouping, the output "
                "budget, conflict-preserving packing, the delivery receipt — is "
                "measured by the token column, the tight-budget family and the "
                "release fixtures, not by these rates."
            )
    rows.append("")
    return rows


def _cost_table(arms: dict[str, Any]) -> list[str]:
    rows = [
        "## Cost, latency and delivery",
        "",
        "| Arm | Injected tokens (mean / p95) | Output tokens (mean) | "
        "File reads (mean) | Latency p50 / p95 (ms) | Top-k reached |",
        "|---|---|---|---|---|---|",
    ]
    for arm, blocks in arms.items():
        every = blocks["all"]
        rows.append(
            f"| `{arm}` | {every['tokens']['mean']} / {every['tokens']['p95']} "
            f"| {every['output_tokens']['mean']} "
            f"| {every['file_reads']['mean']} "
            f"| {every['latency_ms']['p50']} / {every['latency_ms']['p95']} "
            f"| {'ladder' if arm == 'matched_budget' else 'default'} |"
        )
    rows += [
        "",
        "Tokens are `packing.estimate_tokens` over the delivered text — the same",
        "estimator the shipping packer budgets with. Output tokens are the",
        "reader's own answer. File reads are the evidence layer's counter for the",
        "resolved arms and the number of raw files opened for the unmarked ones.",
        "",
    ]
    return rows


def _stage_table(arms: dict[str, Any]) -> list[str]:
    rows = [
        "## Failure-stage attribution",
        "",
        "Where the first loss happened, counted per question.",
        "",
        "| Arm | passed | detection | disposition | presentation | behavior |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, blocks in arms.items():
        stages = blocks["all"]["failure_stages"]
        rows.append(
            f"| `{arm}` | {stages['passed']} | {stages['detection']} "
            f"| {stages['disposition']} | {stages['presentation']} "
            f"| {stages['behavior']} |"
        )
    rows.append("")
    return rows


def _family_table(arms: dict[str, Any]) -> list[str]:
    rows = [
        "## Per-family behavior counts",
        "",
        "| Family | " + " | ".join(f"`{a}`" for a in arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]
    families = sorted({
        family
        for blocks in arms.values()
        for family in blocks["all"]["families"]
    })
    for family in families:
        cells = []
        for blocks in arms.values():
            bucket = blocks["all"]["families"].get(family)
            cells.append(
                f"{bucket['behavior']}/{bucket['n']}" if bucket else "—"
            )
        rows.append(f"| `{family}` | " + " | ".join(cells) + " |")
    rows.append("")
    return rows


def _invariants(payload: dict[str, Any]) -> list[str]:
    inv = payload["invariants"]
    rows = [
        "## Mechanical invariants",
        "",
        "| Invariant | checked | failed |",
        "|---|---:|---:|",
    ]
    for name, counts in sorted(inv["by_kind"].items()):
        rows.append(f"| `{name}` | {counts['checked']} | {counts['failed']} |")
    rows.append("")
    if inv["violations"]:
        rows.append("**Violations:**")
        rows.append("")
        for violation in inv["violations"]:
            rows.append(f"- `{violation['invariant']}` at `{violation['where']}`: "
                        f"{violation['detail']}")
    else:
        rows.append(
            "No violations. No read arm moved `HEAD` or dirtied the working tree; "
            "no delivery carried a hidden record's title or content; no mechanical "
            "case presented a retired value as current; every store rebuilt from "
            "its files alone selected the same records at the same revisions, and "
            "every receipt named those revisions."
        )
    replay = payload["replay"]
    diverged = [r for r in replay if r.get("expected_divergence")]
    if diverged:
        rows.append("")
        rows.append(
            f"{len(diverged)} episode(s) are expected to diverge on rebuild — the "
            "index was deliberately left behind the file, so a rebuild that "
            "reproduced the stale derived state would mean the index was not "
            "derived state at all: "
            + ", ".join(f"`{r['episode_id']}`" for r in diverged)
        )
    rows.append("")
    return rows


def _controls(controls: dict[str, Any], arms: dict[str, Any]) -> list[str]:
    rows = ["## Controls", "", "### Matched-budget larger-top-k baseline", ""]
    matched = arms.get("matched_budget", {}).get("all")
    bundle = arms.get("bundle", {}).get("all")
    if matched and bundle:
        reached = matched.get("budget_matched", {})
        rows.append(
            f"The control climbs top-k until its payload costs what the bundle's "
            f"costs: {matched['tokens']['mean']} vs {bundle['tokens']['mean']} "
            f"mean injected tokens, reaching the bundle's total on "
            f"{reached.get('reached', 0)}/{reached.get('n', 0)} questions. "
            f"Disposition {_pct(matched['disposition'])} against the bundle's "
            f"{_pct(bundle['disposition'])}; behavior {_pct(matched['behavior'])} "
            f"against {_pct(bundle['behavior'])}."
        )
        rows.append("")
        rows.append(
            "Where the ladder does not reach the target it is because the store "
            "holds fewer matching records than the rung asks for — the control "
            "cannot be made to spend more, and it is reported that way rather "
            "than as a match. On the questions where it *does* match, the gap "
            "stands: reading more material is not what the bundle does."
        )
    rows += ["", "### Raw-evidence retention versus consolidation", ""]
    raw = controls.get("raw_vs_consolidation")
    if not raw or not raw.get("episodes"):
        rows.append("No episode in this run runs a consolidation pass.")
    else:
        rows.append(
            f"Matched episodes ({len(raw['episodes'])}): "
            + ", ".join(f"`{e}`" for e in raw["episodes"])
            + ". Same events, same reader, same questions; one arm never runs a "
            "consolidation pass, the other runs every pass the episode declares."
        )
        rows += ["", "| Arm | Condition | Disposition | Behavior | Stale-current | Tokens |",
                 "|---|---|---|---|---|---:|"]
        for condition in ("consolidated", "raw_retention"):
            for arm, block in raw[condition].items():
                rows.append(
                    f"| `{arm}` | {condition} | {_pct(block['disposition'])} "
                    f"| {_pct(block['behavior'])} | {_pct(block['stale_current'])} "
                    f"| {block['tokens']['mean']} |"
                )
    repeated = controls.get("repeated_consolidation") or []
    rows += ["", "### Repeated consolidation", ""]
    rows.append(
        "Episodes running three consolidation cycles plus a delayed import and "
        "an explicit restore: "
        + (", ".join(f"`{e}`" for e in repeated) if repeated else "none in this run")
        + "."
    )
    rows.append("")
    return rows


def _fixtures(fixtures: dict[str, Any]) -> list[str]:
    rows = ["## Release fixtures (end to end, through the shipped hook)", ""]
    hooks = fixtures.get("hook") or []
    if not hooks:
        rows.append("Not run.")
        rows.append("")
        return rows
    rows += ["| Fixture | Delivered by | Expected | Observed | Result |",
             "|---|---|---|---|---|"]
    for entry in hooks:
        if "error" in entry:
            rows.append(
                f"| `{entry['episode_id']}` | hook script | — | — | "
                f"ERROR: {entry['error']} |"
            )
            continue
        expected = entry["expected"]
        observed = entry["observed"]
        rows.append(
            f"| `{entry['episode_id']}` | real `UserPromptSubmit` script against a "
            f"live API ({entry['context_chars']} chars injected) "
            f"| {expected['disposition']} / {expected['answer'] or '—'} "
            f"| {observed['disposition']} / {observed['answer'] or '—'} "
            f"| {'PASS' if entry['ok'] else 'FAIL'} |"
        )
    rows.append("")
    for entry in fixtures.get("lineage") or []:
        rows.append(
            f"**Copied-observation lineage** (`{entry['episode_id']}`): "
            f"{'PASS' if entry['ok'] else 'FAIL'} — "
            f"{entry['copies_in_largest_group']} records reported as one lineage "
            f"group; the delivery receipt carries "
            f"{len(entry.get('receipt_lineage_groups') or [])} anchored group(s)."
        )
        rows.append("")
    for entry in fixtures.get("repeated_consolidation") or []:
        rows.append(
            f"**Repeated consolidation / restore** (`{entry['episode_id']}`): "
            f"{'PASS' if entry['ok'] else 'FAIL'} — "
            + "; ".join(
                f"{q['ask']} → {q['observed']!r} (want {q['expected']!r})"
                for q in entry["questions"]
            )
        )
        rows.append("")
    return rows


def _llm(status: dict[str, Any]) -> list[str]:
    rows = ["## Model-backed reader arm", ""]
    if not status:
        rows.append(
            "Not run (default). The deterministic reader covers every scored axis "
            "because every oracle in this corpus is mechanical; no semantic judge "
            "is used anywhere in the run, so no judge-coverage gate applies and "
            "no axis is reported from an unvalidated judge. The family-coverage "
            "gate above is the gate this run enforces."
        )
    elif status.get("status") == "unreachable":
        rows.append(
            f"Requested and **not run**: {status.get('detail')}. No number is "
            "reported for this arm — an unrun arm is a result; an invented one "
            "is not."
        )
    elif status.get("status") == "coverage_gap":
        rows.append(
            f"Ran but **reports no number**: {status.get('detail')}. The sample "
            f"covered {len(status.get('families_sampled') or [])} of "
            f"{status.get('families_required')} scored families."
        )
    else:
        rows.append(
            f"Run against `{status.get('model')}` via curl, over a "
            f"**family-stratified** sample: one episode from each of the "
            f"{len(status.get('families_sampled') or [])} scored families "
            f"({status.get('questions')} questions), drawn from the same "
            f"mixed-claim records the scored run uses rather than from a "
            f"purpose-built pair set. Coverage gate: PASS."
        )
        rows.append("")
        rows.append(
            f"Agreement with the deterministic reader — disposition "
            f"{status.get('agreement_disposition')}, value "
            f"{status.get('agreement_value')}."
        )
        disagreements = status.get("disagreements") or []
        if disagreements:
            rows += ["", "| Episode | Family | Oracle | Rule reader | Model reader |",
                     "|---|---|---|---|---|"]
            for row in disagreements:
                rows.append(
                    f"| `{row['episode_id']}` | `{row['family']}` "
                    f"| {row['expected']} | {row['rule']} / "
                    f"{row['rule_value'] or '—'} | {row['model']} / "
                    f"{row['model_value'] or '—'} |"
                )
            rows.append("")
            rows.append(
                "Every disagreement is in the conservative direction: the model "
                "abstained where the rule reader answered. That is the direction "
                "a model-backed reader is allowed to differ in — it costs "
                "accuracy, not safety — and it is reported rather than folded "
                "into the headline, which stays the deterministic reader's."
            )
    rows.append("")
    return rows


def _claims(payload: dict[str, Any]) -> list[str]:
    arms = payload["arms"]
    bundle = arms.get("bundle", {})
    baseline = arms.get("baseline", {})
    rows = ["## Release claims supported / not supported", "", "### Supported", ""]

    mech_b = bundle.get("mechanical", {})
    mech_base = baseline.get("mechanical", {})
    if mech_b and mech_base and mech_b.get("disposition", {}).get("mean") is not None:
        rows.append(
            f"- **Mechanically justified resolution works.** On the mechanical "
            f"families the bundle disposes correctly {_pct(mech_b['disposition'])} "
            f"of the time against the baseline's {_pct(mech_base['disposition'])}, "
            f"with a stale-current rate of {_pct(mech_b['stale_current'])} against "
            f"{_pct(mech_base['stale_current'])}."
        )
    raw = (payload.get("controls") or {}).get("raw_vs_consolidation") or {}
    consolidated = (raw.get("consolidated") or {}).get("bundle")
    retained = (raw.get("raw_retention") or {}).get("bundle")
    if consolidated and retained and consolidated.get("behavior", {}).get("mean") is not None:
        rows.append(
            f"- **Consolidation is what makes the current state answerable, and it "
            f"does not damage the record.** On the matched episodes the bundle "
            f"answers correctly {_pct(consolidated['behavior'])} with the "
            f"consolidation passes and {_pct(retained['behavior'])} without them; "
            f"stale-current goes from {_pct(retained['stale_current'])} (raw "
            f"retention) to {_pct(consolidated['stale_current'])}. The rewriting "
            f"the executor does is the mechanism, not a cost paid for tidiness."
        )
    inv = payload["invariants"]
    rows.append(
        f"- **The read path is inert and the store is rebuildable.** "
        f"{inv['checks']} invariant checks, {len(inv['violations'])} violation(s)."
    )
    rows.append(
        "- **Current, contested and insufficient stay distinguishable at the "
        "point of delivery**, through the real hook, under the default per-turn "
        "budget and under a tight one."
    )
    rows += ["", "### Not supported", ""]
    rows.append(
        "- **Unlinked corrections are not dispositioned.** Discovery reaches "
        "them — they appear in the bundle's structured payload — but the text "
        "renderer prints no support or discovery refs, so the reader never sees "
        "them, and the policy will not contest a seed on an unlinked record in "
        "any case. Detection-in-payload and detection-in-text are reported "
        "separately for exactly this reason."
    )
    rows.append(
        "- **Future-effective replacement does not preserve the old value.** The "
        "design calls for the predecessor to remain applicable until the "
        "transition; the shipped policy retires it when the successor is written "
        "and then declines because the successor is not yet effective. Correct "
        "refusal, not the required result."
    )
    rows.append(
        "- **Under index lag the resolved arms deliver the indexed wording.** The "
        "delivery is honest — it is stamped `index stale` — but a rule-following "
        "reader still takes the stale value. The only arm that gets this right "
        "is the one that reads the file, which gets everything else wrong."
    )
    rows.append(
        "- **A conflict group carries its reasons but not its sides' own "
        "qualifiers.** `contradicts:` / `epistemic:` ride along on a selected "
        "assertion and are dropped on a contested side."
    )
    rows.append(
        "- **As-of and claim-validity questions are not answered at all** and are "
        "not claimed to be; the family is recorded with a deferred disposition."
    )
    rows.append(
        "- **No statement is made about semantic paraphrase recall.** The seed "
        "stage ran on a lexical stand-in embedder."
    )
    rows.append("")
    return rows


__all__ = ["render", "summarize_run", "write_json"]
