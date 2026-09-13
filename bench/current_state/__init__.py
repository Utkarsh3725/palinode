"""Current-state recall evaluation — from memory transitions to agent decisions.

Retrieval success is not resolution quality. This package measures the four
stages that sit between "a record changed" and "an agent acted correctly",
separately, so a failure can be attributed:

1. **Detection** — did the arm surface the correction or conflicting evidence?
2. **Disposition** — current / contested / unknown, decided correctly?
3. **Presentation** — did the delivered text preserve the disposition, keep a
   retired value out of the current slot, and carry the qualifiers?
4. **Behavior** — did a reader consuming only that text choose the right value,
   or abstain when it should?

Everything here is deterministic. The corpus is a versioned event stream
(:mod:`bench.current_state.corpus`) replayed through the **real** store, the
real executor, the real consolidation runner (with a scripted proposal seam),
and the real resolution/bundle path (:mod:`bench.current_state.world`). The
reader is a rule-based program (:mod:`bench.current_state.reader`), not a
model, so "agent behavior" is a measured number rather than a sampled one. An
optional model-backed reader exists behind a flag and is off by default.

Run it::

    python -m bench.current_state --full --out results.json --report report.md
    python -m bench.current_state --slice 6        # the bounded CI slice

The acceptance report this produces is an internal document; the harness and
its corpus are what ship.
"""
from bench.current_state.corpus import (
    CORPUS_VERSION,
    FAMILIES,
    Episode,
    coverage_gate,
    load_corpus,
)

__all__ = [
    "CORPUS_VERSION",
    "FAMILIES",
    "Episode",
    "coverage_gate",
    "load_corpus",
]
