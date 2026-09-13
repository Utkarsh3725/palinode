"""CLI for the current-state evaluation.

    python -m bench.current_state --full --out results.json --report report.md
    python -m bench.current_state --slice 6          # the bounded CI slice
    python -m bench.current_state --coverage-only    # just the family gate

``--require-coverage`` turns the family gate into an exit code, which is what a
release gate wants: a family that quietly disappeared from the corpus is
indistinguishable from a family nobody wrote, and neither may ship.
"""
from __future__ import annotations

import argparse
import logging
import sys

from bench.current_state import arms as arms_mod
from bench.current_state import harness, report
from bench.current_state.corpus import coverage_gate, load_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.current_state")
    parser.add_argument("--full", action="store_true",
                        help="run every episode in both splits with all controls")
    parser.add_argument("--slice", type=int, default=None, metavar="N",
                        help="run a balanced N-episode slice (the CI shape)")
    parser.add_argument("--corpus", default=None, help="path to an episodes YAML")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--out", default=None, help="results JSON destination")
    parser.add_argument("--report", default=None, help="Markdown report destination")
    parser.add_argument("--resamples", type=int, default=2000,
                        help="bootstrap resamples per interval")
    parser.add_argument("--coverage-only", action="store_true",
                        help="print the family-coverage gate and exit")
    parser.add_argument("--require-coverage", action="store_true",
                        help="exit non-zero when any family lacks either control")
    parser.add_argument("--no-fixtures", action="store_true",
                        help="skip the live-server hook fixtures")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip the rebuild/replay check")
    parser.add_argument("--llm-reader", action="store_true",
                        help="also run the model-backed reader (off by default; "
                             "needs PALINODE_BENCH_LLM_URL and _MODEL)")
    parser.add_argument("--keep-worlds", action="store_true",
                        help="leave each episode's store on disk for inspection")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.disable(logging.WARNING)

    corpus = load_corpus(args.corpus, seed=args.seed)
    gate = coverage_gate(corpus, split="dev")

    if args.coverage_only or args.require_coverage:
        print(f"corpus v{corpus.version}: {gate['episodes']} authored episodes, "
              f"{len(corpus.episodes)} with the held-out split")
        print(f"family coverage: {'PASS' if gate['ok'] else 'FAIL'}")
        for name in gate["missing"]:
            print(f"  MISSING {name}")
        if args.coverage_only:
            return 0 if gate["ok"] else 1
        if not gate["ok"]:
            return 1

    if args.keep_worlds:
        import os
        os.environ["PALINODE_BENCH_KEEP_WORLDS"] = "1"

    splits = ("dev", "held_out") if args.full or args.slice is None else ("dev",)
    result = harness.run(
        corpus,
        arm_names=arms_mod.ARMS,
        limit=args.slice,
        splits=splits,
        replay=not args.no_replay,
        controls=True,
        fixtures=not args.no_fixtures,
        resamples=args.resamples,
    )
    if args.llm_reader:
        # The coverage requirement is not optional for a model-backed reader:
        # it is the one non-deterministic component the harness can contain,
        # and a number computed over whichever families happened to be sampled
        # is the failure this rule exists to prevent.
        result.llm_reader = harness.run_llm_reader(
            corpus, result, require_coverage=True
        )

    payload = report.summarize_run(result, resamples=args.resamples, splits=splits)
    if args.out:
        report.write_json(payload, args.out)
        print(f"results → {args.out}", file=sys.stderr)
    rendered = report.render(payload)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        print(f"report  → {args.report}", file=sys.stderr)
    else:
        print(rendered)

    return 1 if result.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
