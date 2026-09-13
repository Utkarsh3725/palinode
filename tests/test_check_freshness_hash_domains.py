"""``check_freshness`` must compare like with like.

Two different hashes are both called ``content_hash``:

* ``chunks.content_hash`` — SHA-256 of **one section's body**, written by the
  indexer.
* frontmatter ``content_hash:`` — SHA-256 of the **whole request body** as
  submitted, written by the save path.

``check_freshness`` fell back to the second when the first was NULL and then
compared it against a freshly computed per-section hash. That can never match,
so every row predating the column reported ``stale`` unconditionally — the
exact bug the function's docstring claims to have fixed, reintroduced through
the fallback.

A missing hash is unknowable. ``unknown`` is the honest answer; a false
``stale`` sends a caller chasing a staleness that does not exist.

A third hash joined them: ``chunks.projected_hash``, over the *derived*
current-text projection the index is built from. It lives in its own column,
next to the version that produced it, and is never a comparand here either:
``content_hash`` stays over the raw section, so freshness and the quote-anchor
verifier keep targeting raw source bytes.
"""
from __future__ import annotations

import hashlib

import pytest

from palinode.core import store
from palinode.core.config import config
from palinode.core.projection import PROJECTION_VERSION, project_current_text
from palinode.core.quote_verify import quote_hash, verify_source_anchors
from palinode.indexer.reconcile import derive

_TOMBSTONE = "- ~~Old value lives here.~~ [superseded 2026-09-12] <!-- fact:a -->\n"
_SUCCESSOR = "- New value lives here. <!-- fact:supersedes-a -->\n"

_PAD = "Filler sentence to clear the single-chunk threshold. " * 20


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    # `palinode_dir` is a read-only property and is only consulted to resolve
    # *relative* result paths; every fixture here passes an absolute one.
    monkeypatch.setattr(config, "memory_dir", str(tmp_path))
    return tmp_path


def _write(memory_dir, name: str, body: str) -> str:
    p = memory_dir / name
    p.write_text(f"---\nid: x\ncategory: projects\n---\n\n{body}\n", encoding="utf-8")
    return str(p)


def test_null_column_reports_unknown_not_stale(memory_dir):
    """The regression. A pre-column row carries only the frontmatter hash; the
    old fallback compared it to a per-section hash and always lost."""
    path = _write(memory_dir, "a.md", f"# X\n\nA fact.\n{_PAD}")
    whole_file_hash = hashlib.sha256(open(path).read().encode()).hexdigest()

    results = [{
        "file_path": path,
        "section_id": "root",
        "content_hash": None,                          # pre-column row
        "metadata": {"content_hash": whole_file_hash},  # frontmatter, wrong domain
    }]
    out = store.check_freshness(results)
    assert out[0]["freshness"] == "unknown", out[0]


def test_frontmatter_hash_is_never_consulted(memory_dir):
    """Even a frontmatter hash that happens to be present must not be treated
    as a comparand — it answers a different question."""
    path = _write(memory_dir, "b.md", f"# X\n\nA fact.\n{_PAD}")
    results = [{
        "file_path": path,
        "section_id": "root",
        "content_hash": None,
        "metadata": {"content_hash": "0" * 64},
    }]
    assert store.check_freshness(results)[0]["freshness"] == "unknown"


def test_matching_section_hash_still_reports_valid(memory_dir):
    """The fix must not break the case that worked: a real column hash over the
    real section content."""
    body = f"# X\n\nA fact.\n{_PAD}"
    path = _write(memory_dir, "c.md", body)

    _, sections = store._parser.parse_markdown(open(path).read())
    section = sections[0]
    real = hashlib.sha256(section["content"].encode()).hexdigest()

    results = [{
        "file_path": path,
        "section_id": section["section_id"],
        "content_hash": real,
        "metadata": {},
    }]
    assert store.check_freshness(results)[0]["freshness"] == "valid"


def test_genuinely_changed_section_still_reports_stale(memory_dir):
    """And the other case that worked: a real hash that no longer matches."""
    path = _write(memory_dir, "d.md", f"# X\n\nA fact.\n{_PAD}")
    results = [{
        "file_path": path,
        "section_id": "root",
        "content_hash": hashlib.sha256(b"something else entirely").hexdigest(),
        "metadata": {},
    }]
    assert store.check_freshness(results)[0]["freshness"] == "stale"


# ── raw vs projected ──────────────────────────────────────────────────────────


def test_projection_leaves_the_raw_hash_alone(memory_dir):
    """The indexer's ``content_hash`` is over the raw section; the projected
    text gets its own hash and version. Only the raw one is a comparand."""
    path = _write(memory_dir, "e.md", f"# X\n\n{_TOMBSTONE}{_SUCCESSOR}{_PAD}")
    content = open(path, encoding="utf-8").read()
    _, sections = store._parser.parse_markdown(content)
    raw_hash = hashlib.sha256(sections[0]["content"].encode()).hexdigest()

    sec = derive(path, content).sections[0]
    assert sec.content_hash == raw_hash
    assert sec.projected_hash == hashlib.sha256(sec.content.encode()).hexdigest()
    assert sec.projected_hash != raw_hash
    assert sec.projection_version == PROJECTION_VERSION
    assert "Old value" not in sec.content and "New value" in sec.content

    base = {"file_path": path, "section_id": sec.section_id, "content": sec.content, "metadata": {}}
    assert store.check_freshness([{**base, "content_hash": sec.content_hash}])[0]["freshness"] == "valid"
    # The projected hash answers a different question and can never match.
    assert store.check_freshness([{**base, "content_hash": sec.projected_hash}])[0]["freshness"] == "stale"


def test_projected_hash_equals_raw_hash_when_nothing_is_projected(memory_dir):
    path = _write(memory_dir, "f.md", f"# X\n\n{_SUCCESSOR}{_PAD}")
    content = open(path, encoding="utf-8").read()
    sec = derive(path, content).sections[0]
    assert sec.projected_hash == sec.content_hash
    assert sec.projection_version == PROJECTION_VERSION


def test_anchor_verification_still_targets_raw_bytes_of_a_retired_line(memory_dir):
    """A quote that lives inside a tombstoned line verifies against the file —
    the projection exists only in the index, never in what the verifier reads."""
    path = _write(memory_dir, "src.md", f"# X\n\n{_TOMBSTONE}{_SUCCESSOR}")
    quote = "Old value lives here."
    anchors = [{"ref": "src.md", "quote": quote, "quote_hash": quote_hash(quote)}]
    (result,) = verify_source_anchors(anchors, str(memory_dir))
    assert result.ok and not result.partial

    content = open(path, encoding="utf-8").read()
    assert quote not in project_current_text(content).text
