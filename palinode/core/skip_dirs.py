"""One definition of "this path is not a memory file".

Five surfaces walked the store with their own copy of a skip-dir set — the
``GET /list`` listing path (and therefore the SessionStart hook's ``core_only``
injection), the provenance UI, the advisory project review, the session-start
context digest, and ``backed_by`` propagation. Each named the legacy top-level
``prompts`` directory; none named ``specs``. The store keeps its editable
consolidation prompts at ``specs/prompts/*.md`` (see
:data:`palinode.prompts.STORE_PROMPTS_SUBPATH`), and those files carry
``id``/``task``/``model``/``active``/``version`` frontmatter — enough to look
like a memory to every one of those walks. So a provisioned store listed,
browsed, reviewed, primed and propagated into its own prompt files.

Two failure modes, one fix:

- **Only the first segment was tested.** ``specs/prompts/compaction.md`` has
  ``parts[0] == "specs"``, so naming ``prompts`` never helped. The mechanical
  cross-linker hit this first and fixed it by testing *every* directory
  segment; :func:`is_skipped_path` has the same shape.
- **Five copies of the set.** Adding a directory meant finding all five.

:data:`ALWAYS_SKIP` is deliberately narrow: it holds only the directories that
are not memory on *any* surface. ``daily``, ``archive`` and ``inbox`` are not
in it, because whether they are memory depends on who is asking — the digest
reads ``inbox`` (open ActionItems are one of its three sections) while ``/list``
skips it, and ``archive`` stays searchable everywhere. Those stay per-surface
``extra`` arguments, so a caller's existing behaviour is exactly preserved.

Not adopted here (each documents its own reason): the consolidation runner,
which already skipped ``specs``; ``core.lint`` and ``lint.contradictions``,
which lint ``daily``/``inbox`` on purpose and whose sets are narrower than any
of these; and ``core.cross_refs``, which skips only the ``prompts`` segment
inside ``specs/`` so a hand-written ``specs/amr.md`` stays linkable — it shares
the every-segment *shape*, not this set.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

#: Directories that never hold memory files, on any surface. Matched against
#: every directory segment of a path, not just the first.
ALWAYS_SKIP: frozenset[str] = frozenset(
    {"specs", "prompts", "logs", ".palinode", ".obsidian", ".git"}
)


def is_skipped_path(rel_path: str, extra: Iterable[str] = frozenset()) -> bool:
    """True when any *directory* segment of ``rel_path`` is a non-memory dir.

    ``rel_path`` is relative to the memory dir, with either separator
    (``specs/prompts/compaction.md`` and its Windows spelling both match).
    ``extra`` adds the caller's surface-specific directories to
    :data:`ALWAYS_SKIP` — it never subtracts from it.

    Only directory segments count, so a memory named ``logs.md`` or
    ``specs.md`` is still a memory, and a top-level ``specsheet/`` is
    untouched (whole segments, not prefixes).
    """
    segments = rel_path.replace(os.sep, "/").split("/")[:-1]
    if not segments:
        return False
    skip = ALWAYS_SKIP | frozenset(extra)
    return any(segment in skip for segment in segments)
