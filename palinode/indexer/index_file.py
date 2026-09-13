"""Palinode shared indexer entry point.

Thin wrapper over :mod:`palinode.indexer.reconcile`. Historically this module
held the whole "parse markdown -> embed sections -> upsert chunks" pipeline
inline; that logic now lives behind the derive/plan/apply seam in ``reconcile``
so a file's chunks, vectors, FTS tokens, metadata and entity rows move in one
transaction, a frontmatter-only edit is seen, and a changed or
removed entity ref no longer orphans a row.

``index_file`` is kept as the small, stable entry the watcher
(``palinode.indexer.watcher``) and ``POST /save`` (``api.routers.memory``)
already call, returning the same result dict shape so those callers are
untouched.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from palinode.indexer import reconcile

logger = logging.getLogger("palinode.indexer")


def index_file(filepath: str, *, content: str | None = None) -> dict[str, Any]:
    """Reconcile one markdown file's derived state with the DB.

    Args:
        filepath: absolute path to the .md file (must exist on disk).
        content: optional pre-read file content. If omitted, reads from disk.

    Returns:
        dict with keys:
            * ``embedded`` (bool): True iff the reconcile committed with vectors
              (or was a no-op); False on cold-defer or a fail-closed abort.
            * ``chunks_written`` (int): chunks newly written (new/changed body).
            * ``chunks_unchanged`` (int): sections already fully indexed.
            * ``chunks_reembedded`` (int): FTS-only rows re-indexed once a vector
              became available.
            * ``chunks_reprojected`` (int): rows whose derived text was
              re-derived under the current projection version.
            * ``chunks_stamped`` (int): rows that only received the projection
              stamp (derived text and vector already correct).
            * ``chunks_deleted`` (int): obsolete rows pruned for this file.
            * ``indexed_vec`` / ``indexed_fts`` (bool): per-index health.
            * ``error`` (str | None): one-line failure/deferral reason, if any.
    """
    result: dict[str, Any] = {
        "embedded": False,
        "indexed_vec": True,
        "indexed_fts": True,
        "chunks_written": 0,
        "chunks_unchanged": 0,
        "chunks_reembedded": 0,
        "chunks_reprojected": 0,
        "chunks_stamped": 0,
        "chunks_deleted": 0,
        "error": None,
    }

    if not os.path.exists(filepath):
        result["error"] = "file not found"
        return result

    if content is None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(
                "index read failed op=index file_path=%s error=%r",
                filepath, str(e),
            )
            result["error"] = f"read failed: {e}"
            return result

    diff = reconcile.reconcile(filepath, content)
    result.update(
        embedded=diff.committed and not diff.deferred and diff.embed_failures == 0,
        indexed_vec=diff.vec_ok,
        indexed_fts=diff.fts_ok,
        chunks_written=diff.written,
        chunks_unchanged=diff.unchanged,
        chunks_reembedded=diff.reembedded,
        chunks_reprojected=diff.reprojected,
        chunks_stamped=diff.stamped,
        chunks_deleted=diff.deleted,
        error=diff.error,
    )
    return result
