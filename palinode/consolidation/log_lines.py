"""Recognize the dated log lines a status document accumulates.

A ``projects/<slug>-status.md`` is a longitudinal index: ``POST /session-end``
appends one line per session, rendered by
``palinode.api.routers.session._status_line`` and stamped with a fact marker on
the way in::

    - [2026-03-04] Shipped the retrieval receipt. (2 decisions → daily/2026-03-04.md) <!-- fact:palinode-status-9c1f02 -->

Six months of those is a backlog no compaction prompt can digest: the
dogfood store reached 449 tagged facts, ~430 of them stale session lines, and
an honest ARCHIVE-per-fact proposal ran to ~60 KB — past any token cap the
model will serve. Retiring them is deterministic work (a date is arithmetic,
not judgement), so it belongs here rather than in a proposal.

**Where those lines actually live matters, and is not where you would guess.**
The session-end writer appends to the *end of the file*, and ``## Consolidation
Log`` is the last section a status document has — so the dated session lines
land **inside** it, interleaved with the ``### <date>`` blocks of
``- [UPDATE] <id>: …`` operation records that ``_update_status_summary``
writes, and below the ``_[log elided]_`` bullet. On the real dogfood document
all 453 dated fact lines are under that heading and none are above it::

    ## Consolidation Log

    - _[log elided] 291 operation line(s) across 48 date block(s) — 2026-03-31 → 2026-06-28. Full detail in git history._

    - [2026-03-31] Test session: launched Palinode v0.5.0, … <!-- fact:…-84109d -->
    - [2026-04-04] CLI parity: added read and session-end commands <!-- fact:…-5f93e9 -->
    ### 2026-06-28
    - [UPDATE] supersedes-…-dbba20:
    - [2026-06-28] Shipped the entire v0.8.16 … <!-- fact:… -->

So recognition is by **shape, never by section**: a recognizer that excluded
the log section would retire nothing at all on the one document this exists
for. The two kinds of line that share that section with the session lines are
excluded by what they look like — an operation record opens with an op word
rather than a date, and the elision bullet opens with ``_[log elided]``.

This module is the single recognizer both retirement paths read, so the
runner's age sweep and the executor's ``ARCHIVE_BEFORE`` cannot drift on
what a log line is:

* :func:`dated_log_lines` — every recognized line in a document *body*.
* :func:`older_than` — those strictly older than a moment.

**What counts as a dated log line**, precisely:

* a markdown list item in the document **body** (frontmatter is excluded by the
  caller, which passes a body), anywhere in it,
* whose text begins — immediately after the ``-``/``*`` marker — with a
  ``[YYYY-MM-DD]`` tag that :func:`palinode.core.lifecycle.parse_moment` can
  read as a real date,
* and which carries an executor-addressable ``<!-- fact:id -->`` marker.

**What does not**, each for its own reason:

* an operation record — ``- [UPDATE] <id>: …``, ``- [ARCHIVE] …``, any
  ``- [OP_WORD] …`` — which is the audit trail *of* a retirement; retiring the
  record of an archive would be the log eating itself,
* the ``- _[log elided] N operation line(s) …_`` bullet, for the same reason,
* a bullet with no date tag — an undated curated fact is not a log line,
* a bullet whose date is not at the start (``- Shipped on [2026-03-04]``) — the
  rendering is the contract, not any date anywhere in the text,
* a retired-in-place line (``- ~~…~~ [superseded 2026-09-10] <!-- fact:x -->``)
  — the strike comes before the date, so a tombstone is never re-retired,
* anything inside a fenced code block — a sample line in documentation is not
  a fact,
* anything at or after ``<!-- palinode-auto-footer -->`` — footer wikilinks are
  navigation, the same boundary the propose-side guard refuses to retire at.

``### <date>`` headings are not list items and are never touched. A block whose
operation records all survive keeps its heading whatever this removes from
around it: :mod:`palinode.consolidation.status_doc` parses a session line as a
*raw* item outside any block, so removing one cannot empty a block or disturb
the log's bounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from palinode.core.embedding_preprocess import AUTO_FOOTER_MARKER
from palinode.core.lifecycle import parse_moment

#: One dated log line. Deliberately anchored: the date tag must follow the list
#: marker directly, which is what ``_status_line`` renders and what the
#: executor's own nightly-MERGE date check (``_extract_fact_date``) reads.
LOG_LINE_RE = re.compile(
    r"^[ \t]*[-*][ \t]+\[(?P<date>\d{4}-\d{2}-\d{2})\]"
    r"(?P<text>.*?)<!-- fact:(?P<id>\S+) -->"
)

#: An operation record in the ``## Consolidation Log`` — ``- [UPDATE] id: why``.
#: The bracket holds an op word, not a date, so it cannot match
#: :data:`LOG_LINE_RE`; checked explicitly so the exclusion is a stated rule
#: rather than a side effect of one regex's character class.
OP_RECORD_RE = re.compile(r"^[ \t]*[-*][ \t]+\[[A-Z][A-Z_]*\]")

#: The cumulative ``- _[log elided] N operation line(s) …_`` bullet. It carries
#: dates, but in the middle of the line and about lines that are already gone.
ELISION_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]+_?\[log elided\]")

#: A markdown fence, either flavour. Toggles recognition off while open.
_FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~)")


@dataclass(frozen=True)
class LogLine:
    """A recognized dated log line: its id, its date, and where it sits."""

    fact_id: str
    date: str
    moment: datetime
    text: str
    line_number: int


def dated_log_lines(body: str) -> list[LogLine]:
    """Every dated status log line in *body*, in document order.

    *body* is a document body — pass
    :func:`palinode.core.parser.split_frontmatter`'s second element, never the
    whole file: a YAML frontmatter list entry uses the same ``- item`` syntax,
    and the whole point of the body/frontmatter split elsewhere in
    consolidation is that it is never mistaken for a fact.
    """
    found: list[LogLine] = []
    in_fence = False

    for number, line in enumerate(body.splitlines(), start=1):
        if AUTO_FOOTER_MARKER in line:
            break
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if OP_RECORD_RE.match(line) or ELISION_BULLET_RE.match(line):
            continue
        match = LOG_LINE_RE.match(line)
        if not match:
            continue
        moment = parse_moment(match.group("date"))
        if moment is None:
            # `\d{4}-\d{2}-\d{2}` matches 2026-13-45; a date that is not a date
            # is not a log line, and nothing downstream has to re-check it.
            continue
        found.append(LogLine(
            fact_id=match.group("id"),
            date=match.group("date"),
            moment=moment,
            text=match.group("text").strip(),
            line_number=number,
        ))
    return found


def older_than(body: str, cutoff: datetime) -> list[LogLine]:
    """Dated log lines in *body* strictly older than *cutoff*.

    Strict, in both callers: an ``ARCHIVE_BEFORE`` names the first date it
    keeps, and an age window of N days keeps a line dated exactly N days ago.
    """
    return [line for line in dated_log_lines(body) if line.moment < cutoff]
