"""Current-text projection — the text the index is derived from.

The consolidation executor retires a fact *in place*: the old wording stays
in the active document, struck through and followed by Palinode's own marker
(``~~old~~ [superseded YYYY-MM-DD]``, ``~~old~~ [RETRACTED YYYY-MM-DD …]``),
with the successor on the next line. That is the right shape for the file —
the history is visible, reviewable and reversible by hand — but it is the
wrong shape for retrieval: the obsolete wording sat beside the current one in
every chunk, so it still ranked in BM25 and in the vector index and rendered
in snippets, and file-level archive filtering could not help because the file
is not archived.

This module is the pure projection the indexer applies to each section before
the text reaches FTS or the embedder. It removes **only** what the executor's
own recognizers (:mod:`palinode.core.lifecycle`) identify as a retirement
marker; everything else is preserved byte-for-byte:

* A line that :func:`~palinode.core.lifecycle.is_retired_fact_text` recognises
  (with its list marker stripped) is removed whole — the fact-level SUPERSEDE
  and RETRACT tombstones, and a mention-level strike that is the entire item.
* A mention-level strike that sits *inside* otherwise-current text
  (:data:`~palinode.core.lifecycle.RETIRED_MENTION_RE`, the ``r:<id>`` form
  ``consolidation.retract`` writes mid-paragraph) removes just that span; the
  surrounding sentences stay. The line goes only if nothing but its list
  marker is left. The recognizer is deliberately line-anchored, so this is
  the one place the two forms are told apart.
* Fenced code (backtick or tilde) and inline code spans are content, never
  tombstones — a fenced example of a tombstone is a literal example.
* Ordinary Markdown strikethrough, a marker without the strike, a marker with
  a malformed date, or any other near-miss is not a tombstone and stays.
* Frontmatter is untouched; a document is split with the same lossless
  ``split_frontmatter`` the executor uses.

It is deterministic and idempotent: projecting a projection changes nothing.
Raw markdown on disk is never altered — the projection exists only in the
derived index rows, alongside the raw section hash, so the retired wording is
still retrievable from the file itself, its ``-history.md`` sibling, ``git
log`` and ``palinode_blame``. :data:`PROJECTION_VERSION` is stored with every
derived row; bumping it invalidates the derived text without any raw byte
having changed (see ``indexer.reconcile``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from palinode.core.lifecycle import RETIRED_MENTION_RE, is_retired_fact_text
from palinode.core.parser import split_frontmatter

#: Version of the projection rules. Stored on every derived chunk; a row on
#: another version (or none) is re-derived on the next reconcile even when the
#: raw section bytes are unchanged. Bump when the rules change.
PROJECTION_VERSION = 1

# A markdown list marker at the head of a line, captured with its indentation
# so an affected item can be reassembled with the marker it came with.
_LIST_MARKER_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")

# A fenced-code delimiter. Same shape ``consolidation.retract`` refuses to
# strike inside; a closing fence uses the same character as its opener.
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

# An inline code span. Text inside one is a literal, not a marker.
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# A struck mention with the whitespace that joined it to the previous
# sentence, so removing it closes the gap instead of leaving a double space.
_MENTION_SPAN_RE = re.compile(r"\s*" + RETIRED_MENTION_RE.pattern)


@dataclass(frozen=True)
class Projected:
    """Result of :func:`project_current_text`."""

    #: The current-state text — what the index is derived from.
    text: str
    #: :data:`PROJECTION_VERSION` the text was produced under.
    version: int
    #: What was removed, in document order: a whole line (without its line
    #: ending) or a mention span. For tests and observability.
    removed: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def _strip_mention_spans(text: str) -> tuple[str, list[str]]:
    """Remove every struck mention outside inline code from one line."""
    removed: list[str] = []

    def _sub(segment: str) -> str:
        def _take(m: re.Match[str]) -> str:
            removed.append(m.group().strip())
            return ""
        return _MENTION_SPAN_RE.sub(_take, segment)

    pieces: list[str] = []
    pos = 0
    for code in _INLINE_CODE_RE.finditer(text):
        pieces.append(_sub(text[pos:code.start()]))
        pieces.append(code.group())
        pos = code.end()
    pieces.append(_sub(text[pos:]))
    return "".join(pieces), removed


def project_current_text(markdown: str) -> Projected:
    """Project ``markdown`` to its current-state text. Pure; see the module doc."""
    head, body = split_frontmatter(markdown)
    out: list[str] = []
    removed: list[str] = []
    fence: str | None = None

    for line in body.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        ending = line[len(bare):]
        stripped = bare.strip()

        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue

        list_match = _LIST_MARKER_RE.match(bare)
        prefix = list_match.group(1) if list_match else ""
        rest = bare[len(prefix):]

        if RETIRED_MENTION_RE.search(rest):
            lead = rest[: len(rest) - len(rest.lstrip())]
            projected, spans = _strip_mention_spans(rest[len(lead):])
            removed.extend(spans)
            projected = projected.lstrip()
            if not projected.strip():
                continue
            out.append(f"{prefix}{lead}{projected}{ending}")
            continue

        if is_retired_fact_text(rest):
            removed.append(bare)
            continue

        out.append(line)

    return Projected(
        text=head + "".join(out),
        version=PROJECTION_VERSION,
        removed=tuple(removed),
    )
