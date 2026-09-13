"""Session-start context digest (ADR-012 Layer 4; the /context/prime core).

One bounded, deterministic digest of "what should a fresh session know":
the resolved project scope, `core: true` memories, recently-modified
decisions, open action items, and the most recent project snapshots (the
`/wrap` "pick up where you left off" notes) — built from frontmatter reads
only (no embeds, no LLM, no network), so it is safe on the session cold-start
path.

Every row is selected and qualified through :mod:`palinode.core.lifecycle`,
the eligibility classifier the consolidation runner also uses: a retired
record (archived in place, superseded, deprecated, retracted, or past its
``expires_at``) is never presented as current, and a usable record's
``contradicts`` / ``stale_backing`` / ``epistemic`` qualifiers ride into its
row so a contested or unverified snapshot cannot become an unqualified
startup summary. "Recent" orders by the record's effective date (declared
``date``, else the save stamps); a file's mtime is not a new effective
decision and only breaks ties among undated records.

Serves every surface per ADR-010 parity:

- ``POST /context/prime`` — the endpoint the Claude Code SessionStart hook
  already calls (shipped forward-compat; this module makes it live).
- ``palinode_session_init`` — the MCP tool for MCP-only harnesses (Claude
  Desktop, Codex CLI, Gemini CLI), ADR-012 §3.4's discoverable
  "first call you should make".
- ``palinode prime`` — the CLI.

Scope discipline (the ADR's critical constraint): project-scoped rows are
returned ONLY when a project actually resolves — explicit ``project`` arg,
else ``cwd`` basename through ``config.context.project_map`` /
``auto_detect`` (the same ADR-008 resolution the ambient search boost uses).
With no resolvable project (e.g. Claude Desktop, which has no CWD), the
digest degrades to core memories only, clearly labelled — it never guesses
a project and never bleeds another project's context.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import Any

from palinode.core.config import config
from palinode.core.expiry import core_has_expired
from palinode.core.lifecycle import Eligibility, eligibility, order_key
from palinode.core.packing import (
    KIND_ASSERTION,
    KIND_CONFLICT,
    KIND_CONTESTED_PARTIAL,
    KIND_GIST,
    PRIORITY_ASSERTION,
    SURFACE_STARTUP,
    Budget,
    Unit,
    budget_from_config,
    estimate_tokens,
    pack,
)
from palinode.core.skip_dirs import is_skipped_path

#: Bounded digest sizes — the digest rides the session cold-start path.
MAX_CORE_MEMORIES = 10
MAX_RECENT_DECISIONS = 5
MAX_OPEN_ACTION_ITEMS = 5
#: Snapshots accrete fast (one per /wrap), so keep this tight — a fresh
#: session wants the last couple of "where I left off" notes, not a history.
MAX_RECENT_SNAPSHOTS = 3
#: Hard cap on any single digest line (title + description).
MAX_LINE_CHARS = 200
#: Row keys a qualified record carries in addition to ``file`` and ``summary``.
#: Present only when the record declares them — an absent ``epistemic`` is
#: unmarked and stays absent. Every surface (REST JSON, MCP text, CLI text)
#: carries exactly these; ``tests/test_context_prime_lifecycle.py`` pins it.
DIGEST_QUALIFIER_KEYS = ("contradicts", "stale_backing", "epistemic")

PALINODE_HINT = (
    "Memory available. Call palinode_search before answering questions about "
    "prior decisions or project state; save decisions with palinode_save "
    "(include the rationale); call palinode_session_end before the session ends."
)

# On top of the never-memory dirs every surface skips
# (``skip_dirs.ALWAYS_SKIP`` — ``logs``, ``.obsidian`` and the store's own
# ``specs/prompts`` copies). Unlike the /list browse surface, `inbox` is NOT
# skipped — ActionItems live there, and open action items are one of the
# digest's three sections.
_SKIP_DIRS = frozenset({"daily", "archive"})


def resolve_project(cwd: str | None = None, project: str | None = None) -> str | None:
    """Resolve a project entity ref, or None when no project can be named.

    Explicit ``project`` wins (bare slugs gain the ``project/`` prefix), then
    the ``PALINODE_PROJECT`` env var, then the ``cwd`` basename through
    ``config.context.project_map`` and ``auto_detect`` — the same ADR-008
    resolution order ``mcp.py:_resolve_context()`` uses. Never guesses when
    none is usable.
    """
    if project:
        return project if "/" in project else f"project/{project}"
    env = os.environ.get("PALINODE_PROJECT")
    if env:
        return env if "/" in env else f"project/{env}"
    if not cwd:
        return None
    basename = os.path.basename(os.path.normpath(cwd))
    if not basename:
        return None
    mapped = config.context.project_map.get(basename)
    if mapped:
        return mapped if "/" in mapped else f"project/{mapped}"
    if config.context.auto_detect:
        return f"project/{basename}"
    return None


def _scan_memories(base_dir: str) -> list[dict[str, Any]]:
    """Frontmatter scan of the memory dir (skip-dirs excluded), mtime attached."""
    from palinode.core import parser

    out: list[dict[str, Any]] = []
    for filepath in glob.glob(os.path.join(base_dir, "**/*.md"), recursive=True):
        rel = os.path.relpath(filepath, base_dir)
        if is_skipped_path(rel, _SKIP_DIRS):
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                meta, _ = parser.parse_frontmatter(f.read())
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        out.append({"file": rel, "meta": meta, "mtime": os.path.getmtime(filepath)})
    return out


def _digest_row(entry: dict[str, Any]) -> dict[str, Any]:
    meta = entry["meta"]
    title = str(meta.get("title") or "").strip()
    if not title:
        title = os.path.splitext(os.path.basename(entry["file"]))[0]
    description = str(meta.get("description") or "").strip()
    line = f"{title} — {description}" if description else title
    row: dict[str, Any] = {"file": entry["file"], "summary": line[:MAX_LINE_CHARS]}
    elig: Eligibility = entry["elig"]
    if elig.contradicts:
        row["contradicts"] = list(elig.contradicts)
    if elig.stale_backing:
        row["stale_backing"] = list(elig.stale_backing)
    if elig.epistemic:
        row["epistemic"] = elig.epistemic
    return row


def _row_qualifiers(row: dict[str, Any]) -> str:
    """Render a row's qualifiers as the bracketed labels search results use.

    ``⚠ contradicts:`` / ``⚠ stale backing:`` mirror the ``palinode_search``
    result renderer so an agent reads one vocabulary across surfaces. Empty
    when the row carries none.
    """
    bits: list[str] = []
    if row.get("contradicts"):
        bits.append("⚠ contradicts: " + ", ".join(row["contradicts"]))
    if row.get("stale_backing"):
        bits.append("⚠ stale backing: " + ", ".join(row["stale_backing"]))
    if row.get("epistemic"):
        bits.append(f"epistemic: {row['epistemic']}")
    return " [" + " | ".join(bits) + "]" if bits else ""


def _digest_line(row: dict[str, Any]) -> str:
    """The one rendered line a digest row becomes, qualifiers included.

    Shared by the renderer and the budget packer so the packer measures exactly
    the string the reader will see — a cap computed on anything else is a cap
    on the wrong thing.
    """
    return f"- [{row['file']}] {row['summary']}{_row_qualifiers(row)}"


#: Digest sections in render order, each with its packing priority. Every row
#: is a current assertion, so they share :data:`PRIORITY_ASSERTION` and the
#: offset only preserves the digest's own ordering: a fresh session wants the
#: "where I left off" snapshot before an open action item. What makes a
#: contested row whole-or-stub is its unit *kind*, not its priority.
_SECTION_PRIORITIES: tuple[tuple[str, str, int], ...] = (
    ("recent_snapshots", "Recent snapshots", PRIORITY_ASSERTION),
    ("core_memories", "Core memories", PRIORITY_ASSERTION + 1),
    ("recent_decisions", "Recent decisions", PRIORITY_ASSERTION + 2),
    ("open_action_items", "Open action items", PRIORITY_ASSERTION + 3),
)


def _row_qualifier_labels(row: dict[str, Any]) -> tuple[str, ...]:
    """The row's qualifications in ``resolution``'s ``key:ref`` vocabulary.

    Reporting and the packer's demotion guard only — a unit that carries any of
    these may not be demoted to a gist, because the gist would drop them.
    """
    out: list[str] = []
    out.extend(f"contradicts:{ref}" for ref in row.get("contradicts", ()))
    out.extend(f"stale_backing:{ref}" for ref in row.get("stale_backing", ()))
    if row.get("epistemic"):
        out.append(f"epistemic:{row['epistemic']}")
    return tuple(out)


def _row_gist(row: dict[str, Any]) -> str | None:
    """The row's gist-plus-pointer form: the title alone, keeping the path.

    ``None`` when there is nothing to demote (the summary is already just the
    title), so a demotion never costs a reader a line without buying space.
    """
    head = str(row["summary"]).split(" — ", 1)[0]
    return head if head != row["summary"] else None


def _digest_heading(digest: dict[str, Any]) -> str:
    if digest.get("project"):
        return f"## Session context: {digest['project']}"
    return "## Session context (no project resolved — core memories only)"


def _scaffold_cost(digest: dict[str, Any]) -> tuple[int, int]:
    """Chars and estimated tokens the rendering spends outside the rows.

    The heading, one ``###`` per section that currently has rows, the blank
    line and the memory-contract hint. An upper bound — a section whose rows
    are all omitted loses its heading too — and reserving it is what keeps the
    cap a cap on the *rendered* payload rather than on the rows alone.
    """
    scaffold = [_digest_heading(digest)]
    scaffold += [f"### {label}" for key, label, _ in _SECTION_PRIORITIES if digest.get(key)]
    scaffold += ["", str(digest.get("_palinode_hint", PALINODE_HINT))]
    return (
        sum(len(line) + 1 for line in scaffold),
        sum(estimate_tokens(line) for line in scaffold),
    )


def _apply_budget(digest: dict[str, Any], budget: Budget) -> dict[str, Any]:
    """Trim the digest to *budget*, in place, preserving conflicts.

    An unlimited budget returns the digest untouched — the ``MAX_*`` count
    bounds and ``MAX_LINE_CHARS`` above stay the only bound and the rendering
    is byte-identical to the unbudgeted one. Otherwise every row becomes a
    packing unit and :func:`palinode.core.packing.pack` decides: kept whole,
    demoted to its title-plus-path gist (plain rows only — a qualified row is
    never shortened), or omitted. A contested row that does not fit leaves the
    explicit ``_contested_partial`` stub behind, so a conflict never leaves one
    side looking settled.
    """
    if budget.unlimited:
        return digest

    units: list[Unit] = []
    gists: dict[tuple[str, int], str] = {}
    for key, _label, priority in _SECTION_PRIORITIES:
        for index, row in enumerate(digest.get(key, [])):
            qualifiers = _row_qualifier_labels(row)
            gist_summary = _row_gist(row) if not qualifiers else None
            payload = (key, index)
            if gist_summary is not None:
                gists[payload] = gist_summary
            units.append(Unit(
                kind=KIND_CONFLICT if row.get("contradicts") else KIND_ASSERTION,
                text=_digest_line(row),
                priority=priority,
                refs=(row["file"], *row.get("contradicts", ())),
                qualifiers=qualifiers,
                gist=(
                    _digest_line({**row, "summary": gist_summary})
                    if gist_summary is not None else None
                ),
                payload=payload,
            ))

    packed = pack(units, budget)
    kept = {u.payload: u for u in packed.units if u.payload is not None}
    for key, _label, _priority in _SECTION_PRIORITIES:
        rows = []
        for index, row in enumerate(digest.get(key, [])):
            unit = kept.get((key, index))
            if unit is None:
                continue
            if unit.kind == KIND_GIST:
                row = {**row, "summary": gists[(key, index)]}
            rows.append(row)
        if key in digest:
            digest[key] = rows

    stub = next((u for u in packed.units if u.kind == KIND_CONTESTED_PARTIAL), None)
    if stub is not None:
        digest["_contested_partial"] = stub.text
    digest["_budget"] = {
        "max_chars": budget.max_chars,
        "max_tokens": budget.max_tokens,
        "chars": packed.chars,
        "tokens": packed.tokens,
        "omitted": len(packed.omitted),
        "demoted": len(packed.demoted),
        "truncated_reason": packed.truncated_reason,
    }
    return digest


def _most_recent_first(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries, key=lambda m: order_key(m["elig"], m["mtime"]), reverse=True
    )


def _has_entity(meta: dict[str, Any], entity: str) -> bool:
    entities = meta.get("entities")
    return isinstance(entities, list) and entity in entities


def build_context_digest(
    cwd: str | None = None,
    project: str | None = None,
    scope_chain: Any | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the bounded session-start digest for the resolved scope.

    ``now`` is the lifecycle clock (``expires_at``); ``None`` reads the wall
    clock. Injected so eligibility is testable without touching time.

    ``scope_chain`` (a :class:`palinode.core.scope.ScopeChain`) enables
    ADR-009 scoped mode: memories not visible on the chain are dropped from
    every digest section — an off-chain explicit ``scope:`` (Layer 1), a
    ``private`` memory whose owner is off-chain, or a ``restricted`` memory
    whose ``access`` the chain doesn't intersect. Memories without an
    explicit scope always pass (ADR-009 §7).

    ``None`` = classic mode: scope isolation is off (all ``core: true``
    memories, the pre-slice-4 behavior) but ``private``/``restricted``
    memories are still withheld — classic is a *selection* mode, and it was
    never meant to be a way around access control.
    """
    resolved = resolve_project(cwd=cwd, project=project)
    memories = _scan_memories(config.memory_dir)
    # Always route through the choke point; `meta` here is live frontmatter
    # (just parsed by _scan_memories), so this costs no extra read.
    from palinode.core.visibility import is_visible

    memories = [
        m for m in memories
        if is_visible(scope_chain, m["file"], metadata=m["meta"])
    ]

    # Lifecycle eligibility, once per record, from the same live frontmatter.
    # A retired record leaves every section here; a usable one carries its
    # qualifiers into its row. Authority monotonicity for core memories is a
    # special case of this — `core_has_expired` is still consulted so the
    # lapse is reported once per process, as it is on the /list path.
    for m in memories:
        m["elig"] = eligibility(m["meta"], path=m["file"], now=now)
        if m["meta"].get("core") is True:
            core_has_expired(m["file"], m["meta"], now)
    memories = [m for m in memories if m["elig"].usable]

    core = _most_recent_first([m for m in memories if m["meta"].get("core") is True])

    recent_decisions: list[dict[str, Any]] = []
    open_action_items: list[dict[str, Any]] = []
    recent_snapshots: list[dict[str, Any]] = []
    if resolved:
        scoped = [m for m in memories if _has_entity(m["meta"], resolved)]
        decisions = [
            m for m in scoped
            if m["meta"].get("type") == "Decision" or m["file"].startswith("decisions/")
        ]
        recent_decisions = _most_recent_first(decisions)[:MAX_RECENT_DECISIONS]

        # Retirement is already applied above; "open" additionally excludes
        # the completion states, which are this section's own notion, not a
        # lifecycle one.
        actions = [
            m for m in scoped
            if m["meta"].get("type") == "ActionItem"
            and str(m["meta"].get("status") or "").lower() not in ("done", "resolved")
        ]
        open_action_items = _most_recent_first(actions)[:MAX_OPEN_ACTION_ITEMS]

        # ProjectSnapshots are the session_end / palinode_save wrap notes.
        # Select by type only, NOT by a projects/ path OR (as decisions does
        # with decisions/): projects/ is a mixed dir — it also holds the
        # append-only <project>-status.md logs — so a path fallback would
        # surface non-snapshot files.
        snapshots = [
            m for m in scoped if m["meta"].get("type") == "ProjectSnapshot"
        ]
        recent_snapshots = _most_recent_first(snapshots)[:MAX_RECENT_SNAPSHOTS]

    # A ProjectSnapshot flagged core: true satisfies both filters. Its
    # purpose-built home is the Recent snapshots section (which leads the
    # digest), so drop it from Core memories rather than render it twice.
    snapshot_files = {m["file"] for m in recent_snapshots}
    core = [m for m in core if m["file"] not in snapshot_files]

    digest = {
        "project": resolved,
        "core_memories": [_digest_row(m) for m in core[:MAX_CORE_MEMORIES]],
        "recent_decisions": [_digest_row(m) for m in recent_decisions],
        "open_action_items": [_digest_row(m) for m in open_action_items],
        "recent_snapshots": [_digest_row(m) for m in recent_snapshots],
        "_palinode_hint": PALINODE_HINT,
    }
    # The count bounds above are the selection; the budget is the ceiling on
    # what the selection renders to. Both apply — the packer takes the already
    # bounded rows as its input rather than truncating a second time.
    reserved_chars, reserved_tokens = _scaffold_cost(digest)
    return _apply_budget(digest, budget_from_config(
        SURFACE_STARTUP,
        reserved_chars=reserved_chars,
        reserved_tokens=reserved_tokens,
    ))


def format_context_digest(digest: dict[str, Any]) -> str:
    """Render the digest as compact text (shared by the MCP and CLI surfaces).

    Renders what the budget left: rows already trimmed or demoted by
    :func:`_apply_budget`, plus the ``_contested_partial`` stub when conflict
    groups were withheld. With no budget configured nothing was trimmed and the
    output is the unbudgeted rendering, byte for byte.
    """
    lines: list[str] = [_digest_heading(digest)]
    # Snapshots first: the /wrap "where did I leave off" note is the single
    # most useful thing a resuming session can see.
    sections = tuple(
        (label, digest.get(key, [])) for key, label, _ in _SECTION_PRIORITIES
    )
    for label, rows in sections:
        if rows:
            lines.append(f"### {label}")
            for row in rows:
                lines.append(_digest_line(row))
    if digest.get("_contested_partial"):
        lines.append(str(digest["_contested_partial"]))
    if len(lines) == 1:
        withheld = (digest.get("_budget") or {}).get("omitted") or 0
        if withheld:
            # "(no memories in scope yet)" would be a lie: memories were found
            # and the budget could not afford to name them. This line is
            # scaffolding like the heading and the hint — a budget too small to
            # hold it is too small to be honest, and saying so beats claiming an
            # empty store.
            lines.append(
                f"⚠ {withheld} memories withheld for budget — call palinode_search"
            )
        else:
            lines.append("(no memories in scope yet)")
    lines.append("")
    lines.append(digest.get("_palinode_hint", PALINODE_HINT))
    return "\n".join(lines)
