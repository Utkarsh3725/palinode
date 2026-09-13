"""``POST /resolve`` — the bounded-resolution operation over HTTP.

One request, one qualified bundle: what stands, what replaced what, what is
still contested, and what the store will not answer for. The whole decision
lives in :mod:`palinode.core.bundle`, so this router is a thin translation of
the canonical parameters (ADR-010) into a :class:`~palinode.core.bundle.
BundleRequest` and back out again. Read-only: nothing here writes a file, a
link, a commit, or recall metadata.

``session_id`` is accepted beyond the canonical parameter set for the same
reason ``/context/prime`` accepts it — harness hooks send it on every call —
and it is used only to resolve the requester's scope chain.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from palinode.api._util import _safe_500
from palinode.api.rate_limit import _RATE_LIMIT_SEARCH, _check_rate_limit
from palinode.api.routers.search import _resolve_scope_chain
from palinode.core.bundle import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_ITEMS,
    BundleBudget,
    BundleRequest,
    build_bundle,
)
from palinode.core.parity import RESOLVE_INTENTS

logger = logging.getLogger("palinode.api")
router = APIRouter()


class ResolveRequest(BaseModel):
    """One bounded-resolution request.

    ``query`` and ``ref`` are the two ways in — a natural-language question or
    an exact memory ref (a fact id: the path without ``.md``). At least one is
    required. ``context`` are refs the caller already holds: each becomes a
    seed of its own, so context a caller is carrying is *checked* rather than
    assumed current. ``intent`` is reserved at ``current_state``; as-of /
    known-at questions are deliberately not offered yet.
    """

    query: str | None = None
    ref: str | None = None
    context: list[str] | None = None
    intent: Literal[*RESOLVE_INTENTS] | None = None
    # Output budget. Whole units are dropped in priority order when it bites;
    # a conflict group is never split, only ever dropped and reported.
    max_items: int | None = Field(default=None, ge=0, le=50)
    max_chars: int | None = Field(default=None, ge=0, le=20000)
    session_id: str | None = None


@router.post("/resolve")
def resolve_api(req: ResolveRequest, request: Request = None) -> dict[str, Any]:
    """Resolve a question (or an exact record) against current memory.

    Returns the bundle: ``selected`` / ``replaced`` / ``conflicts`` /
    ``insufficient``, with ``coverage``, ``source_revisions``, the delivery
    ``receipt`` (and its ``receipt_ref``) and the rendered ``text`` every
    surface shares. The receipt costs no lookup and writes no log row — like
    ``/context/prime``, this endpoint has never written to the retrieval log.
    """
    if request:
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip, "search", _RATE_LIMIT_SEARCH):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        bundle_request = BundleRequest(
            query=req.query,
            ref=req.ref,
            context=tuple(req.context or ()),
            intent=req.intent or "current_state",
            budget=BundleBudget(
                max_items=DEFAULT_MAX_ITEMS if req.max_items is None else req.max_items,
                max_chars=DEFAULT_MAX_CHARS if req.max_chars is None else req.max_chars,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        # ADR-009 Layer 2: session identity only. `context` here is memory
        # refs, not the ambient entity refs `/search` boosts on, so it is not
        # a source of project scope.
        chain = _resolve_scope_chain(session_id=req.session_id)
        return build_bundle(bundle_request, chain=chain).to_dict()
    except Exception as e:
        raise _safe_500(e, "Resolve failed")
