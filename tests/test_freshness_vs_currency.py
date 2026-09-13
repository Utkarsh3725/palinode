"""Index freshness is not factual currency.

``check_freshness`` compares a chunk's stored section hash to the file on disk.
That answers one question — does the index agree with the source? — and a
chunk that still carries a superseded fact's ``~~struck~~ [superseded …]``
tombstone answers it ``valid``: the index faithfully reflects the file. The MCP
renderer used to show that as ``✓ valid``, which read as validation of the
assertion.

Three questions, three fields, three labels:

* ``freshness`` — index/source agreement (unchanged field and vocabulary).
* ``span_integrity`` — are the record's cited quotes still in their sources?
* ``currency`` — is the assertion still in force (live lifecycle + chunk text)?

Real files under ``tmp_path``; the renderers are exercised on the annotated
results so the labels are asserted where a reader sees them.
"""
from __future__ import annotations

import hashlib
import os
import re
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from palinode.api.ui.views import run_search
from palinode.cli.search import search as cli_search
from palinode.core import parser as _parser
from palinode.core.config import config
from palinode.core.lifecycle import contains_retired_fact_text
from palinode.core.quote_verify import quote_hash
from palinode.core.store import check_freshness
from palinode.mcp import _format_results

TOMBSTONE = (
    "- ~~Use endpoint A.~~ [superseded 2026-09-12] <!-- fact:endpoint -->\n"
    "- Use endpoint B. <!-- fact:supersedes-endpoint -->\n"
)
PLAIN = "- Use endpoint B. <!-- fact:endpoint -->\n"

# Wording a matching hash must never be rendered with. ``current`` as a word
# (not a substring of ``currency``, which is a field name, not a label).
_FORBIDDEN_NEXT_TO_AGREEMENT = re.compile(r"verified|\bcurrent\b|✓", re.IGNORECASE)


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    return tmp_path


def _write(memory_dir, rel: str, frontmatter: str, body: str) -> str:
    p = memory_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")
    return str(p)


def _result(path: str, body: str, *, hashed: bool = True, **extra) -> dict:
    """A search hit for ``path`` whose stored hash is the indexed body's hash."""
    _, sections = _parser.parse_markdown(f"---\n---\n\n{body}")
    section = sections[0]
    r = {
        "file_path": path,
        "section_id": section["section_id"],
        "content": section["content"],
        "snippet": section["content"].strip(),
        "score": 0.9,
        "metadata": {},
    }
    if hashed:
        r["content_hash"] = hashlib.sha256(section["content"].encode()).hexdigest()
    r.update(extra)
    return r


def _cli(result: dict) -> str:
    # The API hands the CLI a memory-relative title; an absolute tmp path would
    # wrap the 80-column rich console and split a label across lines.
    hit = dict(result, file=os.path.relpath(result["file_path"], config.memory_dir))
    # The CLI asks for the delivery receipt, so the client returns a pair.
    with patch("palinode.cli.search.api_client.search", return_value=([hit], None)):
        out = CliRunner().invoke(cli_search, ["anything", "--format", "text"])
    assert out.exit_code == 0, out.output
    # The rich console soft-wraps at 80 columns; the labels are asserted as
    # text, not as terminal layout.
    return re.sub(r"\s+", " ", out.output)


# ── the six acceptance cases ──────────────────────────────────────────────────

def test_matches_source_and_current(memory_dir):
    path = _write(memory_dir, "decisions/endpoint.md", "status: active\n", PLAIN)
    r = check_freshness([_result(path, PLAIN)])[0]

    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("valid", "unanchored", "current")
    assert r["currency_reason"] == "status:active"

    mcp = _format_results([r])
    assert "[index matches source]" in mcp
    assert "retired" not in mcp and "contested" not in mcp
    assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(mcp)

    cli = _cli(r)
    assert "[index matches source]" in cli
    assert "retired" not in cli and "contested" not in cli
    assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(cli)


def test_stale_index_after_source_edit(memory_dir):
    path = _write(memory_dir, "decisions/endpoint.md", "status: active\n", PLAIN)
    r = _result(path, PLAIN)
    # Edited after indexing: the stored hash no longer matches.
    _write(memory_dir, "decisions/endpoint.md", "status: active\n", "- Use endpoint C.\n")
    r = check_freshness([r])[0]

    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("stale", "unanchored", "current")

    mcp = _format_results([r])
    assert "[⚠ index stale]" in mcp and "index matches source" not in mcp
    cli = _cli(r)
    assert "[⚠ index stale]" in cli


def test_unknown_hash_still_classifies_currency(memory_dir):
    path = _write(memory_dir, "decisions/endpoint.md", "status: active\n", PLAIN)
    r = check_freshness([_result(path, PLAIN, hashed=False)])[0]

    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("unknown", "unanchored", "current")

    mcp = _format_results([r])
    assert "index" not in mcp  # unknown agreement renders no badge, as before
    assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(mcp)
    cli = _cli(r)
    assert "index" not in cli


def test_retired_tombstone_in_chunk_matches_source_but_is_retired(memory_dir):
    """The probe's mixed chunk: agreement valid, currency retired, side by side."""
    path = _write(memory_dir, "decisions/endpoint.md", "status: active\n", TOMBSTONE)
    r = check_freshness([_result(path, TOMBSTONE)])[0]

    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("valid", "unanchored", "retired")
    assert r["currency_reason"] == "retired fact text"

    mcp = _format_results([r])
    assert "[index matches source]" in mcp
    assert "⚠ retired: retired fact text" in mcp
    assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(mcp)

    cli = _cli(r)
    assert "[index matches source]" in cli
    assert "[⚠ retired: retired fact text]" in cli
    assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(cli)


def test_retired_by_live_frontmatter(memory_dir):
    path = _write(
        memory_dir, "decisions/endpoint.md",
        "status: archived\nsuperseded_by: decisions/endpoint-b\n", PLAIN,
    )
    # Indexed metadata still says active — the live file decides.
    r = check_freshness([_result(path, PLAIN, metadata={"status": "active"})])[0]

    assert (r["freshness"], r["currency"]) == ("valid", "retired")
    assert r["currency_reason"] == "superseded_by: decisions/endpoint-b"

    mcp = _format_results([r])
    assert "[index matches source]" in mcp
    assert "⚠ retired: superseded_by: decisions/endpoint-b" in mcp
    assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(mcp)
    assert "[⚠ retired: superseded_by: decisions/endpoint-b]" in _cli(r)


def test_contested_assertion(memory_dir):
    path = _write(
        memory_dir, "decisions/endpoint.md",
        "status: active\ncontradicts:\n  - decisions/endpoint-b\n", PLAIN,
    )
    meta = {"status": "active", "contradicts": ["decisions/endpoint-b"]}
    r = check_freshness([_result(path, PLAIN, metadata=meta)])[0]

    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("valid", "unanchored", "contested")
    assert r["currency_reason"] == "contradicts: decisions/endpoint-b"

    mcp = _format_results([r])
    assert "[index matches source]" in mcp
    # The refs label is the contested label; the bare word is not duplicated.
    assert "⚠ contradicts: decisions/endpoint-b" in mcp
    assert "⚠ contested" not in mcp
    assert "retired" not in mcp

    cli = _cli(r)
    assert "[index matches source]" in cli and "[⚠ contested]" in cli


def test_contested_when_indexed_metadata_lags_the_file(memory_dir):
    """Live frontmatter has the conflict, chunks.metadata does not: still labelled."""
    path = _write(
        memory_dir, "decisions/endpoint.md",
        "status: active\ncontradicts:\n  - decisions/endpoint-b\n", PLAIN,
    )
    r = check_freshness([_result(path, PLAIN)])[0]
    assert r["currency"] == "contested"
    assert "⚠ contested" in _format_results([r])


def test_unmarked_epistemic_state_renders_no_label(memory_dir):
    path = _write(memory_dir, "decisions/endpoint.md", "type: Decision\n", PLAIN)
    r = check_freshness([_result(path, PLAIN)])[0]

    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("valid", "unanchored", "unmarked")
    assert r["currency_reason"] == "unmarked"

    for out in (_format_results([r]), _cli(r)):
        assert "[index matches source]" in out
        assert "unmarked" not in out
        assert "epistemic" not in out and "inference" not in out and "unverified" not in out
        assert "retired" not in out and "contested" not in out
        assert not _FORBIDDEN_NEXT_TO_AGREEMENT.search(out)


# ── span integrity ────────────────────────────────────────────────────────────

def _anchored(memory_dir, quote: str, *, stored_hash: str | None = None) -> str:
    h = stored_hash if stored_hash is not None else quote_hash(quote)
    fm = f'status: active\nsources:\n  - ref: research/paper.md\n    quote: "{quote}"\n    quote_hash: "{h}"\n'
    return _write(memory_dir, "insights/claim.md", fm, PLAIN)


def test_span_integrity_ok_when_quote_still_in_cited_source(memory_dir):
    _write(memory_dir, "research/paper.md", "type: ResearchRef\n", "The exact cited passage.\n")
    path = _anchored(memory_dir, "The exact cited passage.")
    r = check_freshness([_result(path, PLAIN)])[0]
    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("valid", "ok", "current")
    assert "[cited quote found in source]" in _format_results([r])
    assert "[cited quote found in source]" in _cli(r)


def test_span_integrity_drifted_when_cited_source_changed(memory_dir):
    _write(memory_dir, "research/paper.md", "type: ResearchRef\n", "Something else now.\n")
    path = _anchored(memory_dir, "The exact cited passage.")
    r = check_freshness([_result(path, PLAIN)])[0]
    # The memory file itself is unchanged: agreement valid, span broken.
    assert (r["freshness"], r["span_integrity"]) == ("valid", "source_drifted")
    assert "[⚠ cited quote: source_drifted]" in _format_results([r])
    assert "[⚠ cited quote: source_drifted]" in _cli(r)


def test_span_integrity_missing_source_and_tampered_anchor(memory_dir):
    path = _anchored(memory_dir, "The exact cited passage.")
    r = check_freshness([_result(path, PLAIN)])[0]
    assert r["span_integrity"] == "source_missing"

    _write(memory_dir, "research/paper.md", "type: ResearchRef\n", "The exact cited passage.\n")
    path = _anchored(memory_dir, "The exact cited passage.", stored_hash="sha256:" + "0" * 64)
    r = check_freshness([_result(path, PLAIN)])[0]
    assert r["span_integrity"] == "anchor_tampered"


# ── edges ─────────────────────────────────────────────────────────────────────

def test_missing_source_file(memory_dir):
    r = check_freshness([_result(str(memory_dir / "decisions/gone.md"), PLAIN)])[0]
    assert (r["freshness"], r["span_integrity"], r["currency"]) == ("stale", "unanchored", "unmarked")
    assert r["currency_reason"] == "source missing"


def test_chunks_metadata_is_never_the_currency_source(memory_dir):
    """A retired flag that exists only in the lagging index does not retire the hit."""
    path = _write(memory_dir, "decisions/endpoint.md", "status: active\n", PLAIN)
    r = check_freshness([_result(path, PLAIN, metadata={"status": "archived"})])[0]
    assert r["currency"] == "current"


def test_probe_shape_via_executor_and_indexer(memory_dir):
    """The research probe's case, on a real file: executor output → index derivation
    → the derived text carries only the successor; check_freshness compares the
    *raw* section hash and reports agreement valid and currency current. The
    tombstone stays in the file, and raw text handed in still reads as retired."""
    from palinode.consolidation.executor import _supersede_fact
    from palinode.indexer.reconcile import derive

    with patch("palinode.consolidation.executor.append_to_history"):
        revised = _supersede_fact(
            "- Use endpoint A. <!-- fact:endpoint -->\n", "endpoint",
            "Use endpoint B.", "Explicit replacement", "demo.md",
        )
    path = str(memory_dir / "projects" / "demo.md")
    (memory_dir / "projects").mkdir()
    markdown = "---\nstatus: active\n---\n\n" + revised
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    section = derive(path, markdown).sections[0]
    assert "Use endpoint A." not in section.content  # projected out of the index
    assert "Use endpoint B." in section.content
    r = check_freshness([{
        "file_path": path,
        "section_id": section.section_id,
        "content": section.content,
        "snippet": section.content,
        "content_hash": section.content_hash,  # raw domain, as the indexer stores it
        "metadata": {},
    }])[0]
    assert (r["freshness"], r["currency"]) == ("valid", "current")

    with open(path, encoding="utf-8") as f:
        assert "~~Use endpoint A.~~ [superseded" in f.read()
    raw = check_freshness([{
        "file_path": path,
        "section_id": section.section_id,
        "content": revised,
        "content_hash": section.content_hash,
        "metadata": {},
    }])[0]
    assert (raw["freshness"], raw["currency"]) == ("valid", "retired")


def test_ui_search_row_carries_agreement_and_currency():
    hits = [{
        "file_path": "/store/decisions/endpoint.md", "snippet": "x", "score": 0.5,
        "metadata": {"type": "Decision"},
        "freshness": "valid", "currency": "retired", "currency_reason": "retired fact text",
    }]
    out = run_search("endpoint", lambda q: hits, lambda p: p.removeprefix("/store/"))
    row = out["results"][0]
    assert row["index_agreement"] == "valid"
    assert row["currency"] == "retired"
    assert row["currency_reason"] == "retired fact text"
    assert "freshness" not in row  # not the memory list's age-based key


@pytest.mark.parametrize("text, expected", [
    (TOMBSTONE, True),
    ("- Use endpoint B. <!-- fact:supersedes-endpoint -->\n", False),
    ("* ~~Sky is green~~ [RETRACTED 2026-09-10 — falsified] <!-- fact:sky -->", True),
    ("1. ~~old~~ [superseded 2026-09-10] <!-- fact:n -->", True),
    ("- We ~~thought~~ knew the answer\n- ~~old idea~~ we decided against this", False),
    ("", False),
])
def test_contains_retired_fact_text(text, expected):
    assert contains_retired_fact_text(text) is expected
