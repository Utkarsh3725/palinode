"""The reader: what an agent does with the text an arm injected.

"Did the agent decide correctly?" is the stage the research says nobody
measures, and measuring it with a model makes the number a property of the
model. So the default reader here is a **program**: it consumes only the
delivered context, applies three rules, and returns a decision.

    1. If the context marks a claim contested, abstain and list the sides.
    2. Otherwise, if it marks a value current, answer with that value.
    3. Otherwise abstain as unknown.

Rule 1 outranks rule 2 deliberately: a reader that answers from the current
slot while a conflict sits underneath is exactly the failure the resolution
layer exists to prevent, and a scoring reader that did the same would hide it.

An arm that delivers **unmarked** text — plain search hits, which is what
recall looked like before projection and resolution — has no markers for rules
1–3 to fire on. For that shape the reader does what an agent demonstrably does
with a ranked list: it takes the top hit and states its value. That is not a
strawman; it is the behavior the whole output-contract argument is about, and
it is why an unmarked arm can score a *stale-current answer* at all.

The optional model-backed reader (:func:`llm_read`) exists for the one thing a
program cannot stand in for — whether a real model honors the markers — and is
off unless asked for. It is driven through ``curl`` rather than a Python HTTP
client on purpose: this harness has been run on a host whose Python could not
route to the inference endpoint while ``curl`` could, and a reader that fails
to connect must report that, never fabricate.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence

#: Section labels the resolved grammar emits (``bundle._section_label``). The
#: bounded-evidence arm renders the same labels, so the reader's rules are a
#: property of the *contract*, not of one surface.
SECTION_RE = re.compile(
    r"^(Current|Contested|Replaced|Unknown)\s*\(\d+\)", re.IGNORECASE
)

#: The resolved payload's own heading. Needed alongside :data:`SECTION_RE`
#: because a bundle can have no section at all — a conflict group that did not
#: fit the budget leaves the heading, the omitted-conflict notice and nothing
#: else, and reading that as unmarked hits is how a contested claim would be
#: mistaken for silence.
RESOLVED_HEADING_RE = re.compile(r"Resolved from memory", re.IGNORECASE)

#: A memory ref as every surface renders it: bracketed, and always carrying a
#: category segment. Requiring the ``/`` is what keeps the currency stamp
#: (``[current · 2026-01-10]``) from being read as a source pointer.
REF_RE = re.compile(r"\[([A-Za-z0-9][\w.-]*(?:/[\w.-]+)+)\]")

#: A ref written bare rather than bracketed. Used only on the omitted-conflict
#: notice, which names the group's refs in prose because there is no unit left
#: to hang them on.
BARE_REF_RE = re.compile(r"(?<![\w/\[])([a-z][\w.-]*(?:/[\w.-]+)+)")

#: The bundle's own notice for a conflict group that did not fit the budget.
#: A reader that ignored it would read a budget-shrunk answer as settled.
OMITTED_CONFLICT_RE = re.compile(r"Still contested, omitted for budget", re.IGNORECASE)

#: What the bundle says when it resolved and found nothing.
NOTHING_RE = re.compile(r"Nothing in memory answers this", re.IGNORECASE)

#: The per-turn hook's marker for "resolution missed its deadline; these are
#: unresolved hits". Recorded, not acted on — the point is to measure how often
#: a turn falls back.
DEADLINE_MARKER_RE = re.compile(r"resolution unavailable \(deadline\)", re.IGNORECASE)


@dataclass(frozen=True)
class Answer:
    """One reader decision, plus what it read it from."""

    disposition: str                 # current | contested | unknown
    value: str | None
    sides: tuple[str, ...] = ()
    #: Refs on the unit the answer was read from — what the reader would cite.
    cited_refs: tuple[str, ...] = ()
    #: Every ref anywhere in the delivered context — the detection denominator.
    seen_refs: tuple[str, ...] = ()
    #: True when the context used the resolved grammar (section labels).
    resolved_grammar: bool = False
    #: True when the context carried the per-turn deadline fallback marker.
    deadline_fallback: bool = False

    @property
    def abstained(self) -> bool:
        return self.value is None

    def rendered(self) -> str:
        """The answer as the agent would state it — the output-token payload."""
        if self.disposition == "contested":
            return (
                "I can't settle this: memory holds "
                + " and ".join(self.sides or ("two conflicting records",))
                + f" as open sides ({', '.join(self.cited_refs) or 'no refs'})."
            )
        if self.disposition == "unknown":
            return "Memory does not hold a settled answer to that."
        return f"{self.value} ({', '.join(self.cited_refs) or 'no refs'})."


# ── block splitting ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Block:
    """One rendered unit: a leading line plus its indented continuation lines."""

    section: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _blocks(context: str) -> tuple[list[_Block], bool]:
    """Split *context* into units, tagged with the section they sit under."""
    section = "unmarked"
    resolved = False
    blocks: list[_Block] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(_Block(section, tuple(current)))
            current = []

    for line in context.splitlines():
        header = SECTION_RE.match(line.strip())
        if header:
            flush()
            section = header.group(1).lower()
            resolved = True
            continue
        if not line.strip():
            flush()
            continue
        if line.startswith((" ", "\t")) and current:
            current.append(line)
            continue
        flush()
        current = [line]
    flush()
    return blocks, resolved or bool(RESOLVED_HEADING_RE.search(context))


def _values_in(text: str, vocabulary: Sequence[str]) -> list[str]:
    """Vocabulary values present in *text*, in order of first appearance."""
    lowered = text.lower()
    found = [(lowered.find(v.lower()), v) for v in vocabulary if v.lower() in lowered]
    return [v for _, v in sorted(found)]


def _refs_in(text: str) -> list[str]:
    refs = REF_RE.findall(text)
    for line in text.splitlines():
        if OMITTED_CONFLICT_RE.search(line):
            refs.extend(BARE_REF_RE.findall(line))
    return list(dict.fromkeys(refs))


# ── the deterministic reader ─────────────────────────────────────────────────


def read(context: str, vocabulary: Sequence[str]) -> Answer:
    """Decide from *context* alone, using only the values in *vocabulary*.

    ``vocabulary`` is the episode's candidate values (the current one and every
    retired one). Restricting attention to them is what makes the reader a
    decision procedure rather than an extraction model: the question of *which
    value* is never in doubt, only which one the context presents as standing.
    """
    blocks, resolved = _blocks(context)
    seen_refs = tuple(_refs_in(context))
    deadline = bool(DEADLINE_MARKER_RE.search(context))

    if not resolved:
        # Unmarked hits: the top-ranked unit that mentions a candidate value is
        # what an agent reads as the answer.
        for block in blocks:
            values = _values_in(block.text, vocabulary)
            if values:
                return Answer(
                    disposition="current",
                    value=values[0],
                    cited_refs=tuple(_refs_in(block.text)),
                    seen_refs=seen_refs,
                    resolved_grammar=False,
                    deadline_fallback=deadline,
                )
        return Answer(
            disposition="unknown", value=None, seen_refs=seen_refs,
            resolved_grammar=False, deadline_fallback=deadline,
        )

    contested_blocks = [b for b in blocks if b.section == "contested"]
    contested_values: list[str] = []
    contested_refs: list[str] = []
    for block in contested_blocks:
        contested_values.extend(_values_in(block.text, vocabulary))
        contested_refs.extend(_refs_in(block.text))

    if OMITTED_CONFLICT_RE.search(context):
        # A conflict the budget dropped. The bundle names it by ref precisely
        # so silence cannot make it look settled — so the reader must not
        # settle either, even though no side's text is in front of it.
        notice = next(
            (b for b in blocks if OMITTED_CONFLICT_RE.search(b.text)), None
        )
        return Answer(
            disposition="contested",
            value=None,
            sides=tuple(dict.fromkeys(contested_values)),
            cited_refs=tuple(dict.fromkeys(
                contested_refs + (_refs_in(notice.text) if notice else [])
            )),
            seen_refs=seen_refs,
            resolved_grammar=True,
            deadline_fallback=deadline,
        )

    if contested_values:
        return Answer(
            disposition="contested",
            value=None,
            sides=tuple(dict.fromkeys(contested_values)),
            cited_refs=tuple(dict.fromkeys(contested_refs)),
            seen_refs=seen_refs,
            resolved_grammar=True,
            deadline_fallback=deadline,
        )

    for block in blocks:
        if block.section != "current":
            continue
        values = _values_in(block.text, vocabulary)
        if values:
            return Answer(
                disposition="current",
                value=values[0],
                cited_refs=tuple(_refs_in(block.text)),
                seen_refs=seen_refs,
                resolved_grammar=True,
                deadline_fallback=deadline,
            )

    return Answer(
        disposition="unknown", value=None, seen_refs=seen_refs,
        resolved_grammar=True, deadline_fallback=deadline,
    )


# ── the optional model-backed reader ─────────────────────────────────────────


class LlmReaderUnavailable(RuntimeError):
    """The configured inference endpoint could not be reached or parsed."""


#: Endpoint and model come from the environment. No default host is compiled
#: in: a benchmark that silently points at somebody's box is worse than one
#: that refuses to run.
ENV_URL = "PALINODE_BENCH_LLM_URL"
ENV_MODEL = "PALINODE_BENCH_LLM_MODEL"

_LLM_SYSTEM = (
    "You answer strictly from the memory context given to you. "
    "Reply with one JSON object and nothing else: "
    '{"disposition": "current"|"contested"|"unknown", "value": <string|null>, '
    '"sides": [<string>, ...]}. '
    "Use 'current' only when the context marks one value as the standing answer. "
    "Use 'contested' when the context shows an unresolved conflict; list the sides. "
    "Use 'unknown' when the context does not settle the question. Never guess."
)


def llm_read(context: str, vocabulary: Sequence[str], question: str, *, timeout: int = 60) -> Answer:
    """Ask the configured model to play the same role as :func:`read`.

    Driven through ``curl`` (see the module doc). Raises
    :class:`LlmReaderUnavailable` rather than returning a fabricated answer on
    any transport or parse failure — an unrun arm is a reportable result; an
    invented one is not.
    """
    url = os.environ.get(ENV_URL)
    model = os.environ.get(ENV_MODEL)
    if not url or not model:
        raise LlmReaderUnavailable(
            f"set {ENV_URL} and {ENV_MODEL} to run the model-backed reader"
        )
    if not shutil.which("curl"):
        raise LlmReaderUnavailable("curl is not on PATH")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Candidate values: {', '.join(vocabulary)}\n\n"
                    f"Memory context:\n{context}"
                ),
            },
        ],
    }
    try:
        proc = subprocess.run(
            [
                "curl", "-sS", "-f", "-X", "POST",
                f"{url.rstrip('/')}/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "--max-time", str(timeout),
                "-d", json.dumps(payload),
            ],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise LlmReaderUnavailable(f"curl to {url} failed: {exc}") from exc

    try:
        body = json.loads(proc.stdout)
        text = body["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        decided = json.loads(match.group() if match else text)
    except Exception as exc:  # noqa: BLE001 — any parse failure is "unavailable"
        raise LlmReaderUnavailable(
            f"could not parse a decision from the model response: {exc}"
        ) from exc

    disposition = str(decided.get("disposition") or "unknown").lower()
    if disposition not in ("current", "contested", "unknown"):
        disposition = "unknown"
    value = decided.get("value")
    return Answer(
        disposition=disposition,
        value=str(value) if disposition == "current" and value else None,
        sides=tuple(str(s) for s in (decided.get("sides") or [])),
        cited_refs=tuple(_refs_in(context)),
        seen_refs=tuple(_refs_in(context)),
        resolved_grammar=bool(RESOLVED_HEADING_RE.search(context)),
        deadline_fallback=bool(DEADLINE_MARKER_RE.search(context)),
    )


__all__ = [
    "ENV_MODEL",
    "ENV_URL",
    "Answer",
    "LlmReaderUnavailable",
    "llm_read",
    "read",
]
