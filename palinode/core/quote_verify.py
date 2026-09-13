"""Source-quote integrity verification (provenance Q2, the quote-hash source-anchor work).

The deterministic, offline, network-free core of the quote-drift check requested in
public issue the OpenClaw-profile migration plan (Q2): given a claim that cites a quote
from a source, confirm the quote (a) is internally consistent with its recorded hash and
(b) still appears in the cited source. This is the analog of yopedia's Phase C
anchor-verifier lint.

Scope boundary: this module is the *primitive* only — pure verification logic plus a
thin file reader over the proposed ``sources:`` anchor shape. It does NOT define the
capture path or commit the live frontmatter schema (that needs sign-off; see the
quote-hash source-anchor work). It is intentionally a plain integrity check — verifying
that a cited quote still matches its source involves no cryptographic signing or
attestation, so it has no dependency on the separate memory-attestation work.

Proposed anchor shape (per the quote-hash source-anchor work, not yet a captured schema)::

    sources:
      - ref: research/some-paper.md # path under memory_dir
        quote: "the exact cited passage"
        quote_hash: "sha256:<hex of normalize_quote(quote)>"
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from enum import Enum

from palinode.core.hashing import stable_md5_hexdigest
from palinode.core.parser import parse_markdown

# Smart punctuation → ASCII, so a quote copied through a renderer still matches
# the source. Determinism is the whole point: the same logical text must always
# produce the same normalized form and therefore the same hash.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "…": "...",
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_MAP))
_WS_RE = re.compile(r"\s+")


def normalize_quote(text: str) -> str:
    """Canonicalize a quote for stable hashing and substring matching.

    Folds smart punctuation to ASCII, collapses all whitespace runs to single
    spaces, and strips. Idempotent.
    """
    folded = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group()], text)
    return _WS_RE.sub(" ", folded).strip()


#: Algorithm used for newly written quote hashes.
DEFAULT_QUOTE_HASH_ALGORITHM = "sha256"

#: Algorithm a bare, unprefixed digest is assumed to use. Palinode <= 0.9.8 wrote
#: bare MD5 hex (it reused the dedup hasher), so an unprefixed digest in an
#: existing store is always MD5.
LEGACY_QUOTE_HASH_ALGORITHM = "md5"

_HEX_RE = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)


class UnsupportedHashAlgorithm(ValueError):
    """Raised when a stored quote_hash names an algorithm we cannot compute."""


def _digest(text: str, algorithm: str) -> str:
    """Bare hex digest of ``text`` under ``algorithm`` (no prefix)."""
    if algorithm == "md5":
        # usedforsecurity=False, via the shared helper — this is an integrity
        # check for accidental drift, not an adversarial one, and the flag keeps
        # it usable under FIPS.
        return stable_md5_hexdigest(text)
    if algorithm == "sha256":
        return hashlib.sha256(text.encode()).hexdigest()
    raise UnsupportedHashAlgorithm(f"unsupported quote_hash algorithm: {algorithm!r}")


def quote_hash(text: str, algorithm: str = DEFAULT_QUOTE_HASH_ALGORITHM) -> str:
    """Stable, algorithm-prefixed hash of a quote's normalized form.

    Returns ``"<algorithm>:<hex>"`` — e.g. ``"sha256:029a3c30…"``.

    The prefix is load-bearing. Palinode <= 0.9.8 emitted a bare MD5 digest,
    which cannot be migrated: given ``"a1b2c3…"`` there is no way to tell which
    algorithm produced it, so the store could never move off MD5 without
    invalidating every existing anchor. Prefixing makes the algorithm an
    explicit, per-anchor fact, so old and new anchors coexist and verify
    correctly side by side.

    Never compare the output of this function to a stored hash with ``==`` —
    a stored hash may use a different algorithm. Use :func:`quote_hash_matches`.
    """
    return f"{algorithm}:{_digest(normalize_quote(text), algorithm)}"


def parse_quote_hash(stored: str) -> tuple[str, str]:
    """Split a stored quote_hash into ``(algorithm, hexdigest)``.

    A bare, unprefixed digest is resolved **by length** — 32 hex characters is
    MD5 (the pre-0.10 form), 64 is SHA-256. Any other length is rejected rather
    than guessed at: assuming MD5 for a digest that cannot be MD5 would
    guarantee a false ``anchor_tampered`` on verify, which is worse than an
    explicit error.

    Raises :class:`UnsupportedHashAlgorithm` when the prefix names an algorithm
    we cannot compute, when an unprefixed value is not hex, or when a bare
    digest's length matches no supported algorithm.
    """
    value = (stored or "").strip()
    if ":" in value:
        algorithm, _, digest = value.partition(":")
        algorithm = algorithm.strip().lower()
        digest = digest.strip().lower()
        if algorithm not in ("md5", "sha256"):
            raise UnsupportedHashAlgorithm(
                f"unsupported quote_hash algorithm: {algorithm!r}"
            )
        return algorithm, digest
    if not _HEX_RE.match(value):
        raise UnsupportedHashAlgorithm(f"malformed quote_hash: {value!r}")
    if len(value) == 32:
        return LEGACY_QUOTE_HASH_ALGORITHM, value.lower()
    if len(value) == 64:
        return "sha256", value.lower()
    raise UnsupportedHashAlgorithm(
        f"bare quote_hash of length {len(value)} matches no supported algorithm "
        "(32 hex = md5, 64 = sha256); emit '<algorithm>:<hex>'"
    )


def quote_hash_matches(quote: str, stored: str) -> bool:
    """Does ``stored`` match ``quote``, under ``stored``'s own algorithm?

    This is the comparison every caller wants. Recomputing with the *default*
    algorithm and testing equality would report every legacy MD5 anchor as
    tampered the moment the default changed — the regression this indirection
    exists to prevent.

    Raises :class:`UnsupportedHashAlgorithm` if ``stored`` is unusable.
    """
    algorithm, digest = parse_quote_hash(stored)
    return _digest(normalize_quote(quote), algorithm) == digest


class QuoteStatus(str, Enum):
    OK = "ok"
    ANCHOR_TAMPERED = "anchor_tampered"  # stored hash != hash(stored quote)
    SOURCE_DRIFTED = "source_drifted"    # quote no longer present in source
    SOURCE_MISSING = "source_missing"    # cited source file does not exist


@dataclass
class VerifyResult:
    status: QuoteStatus
    ref: str
    expected_hash: str = ""
    actual_hash: str = ""
    message: str = ""
    #: True when no stored hash existed, so only the drift axis was checkable.
    #: Distinguishes "verified" from "verified as far as was possible" — a
    #: hash-less ``ok`` must not read as stronger than it is.
    partial: bool = False

    @property
    def ok(self) -> bool:
        return self.status is QuoteStatus.OK


def verify_quote(quote: str, expected_hash: str, source_text: str, ref: str = "") -> VerifyResult:
    """Verify one quote anchor against its source. Pure; no I/O.

    Two independent failure axes:
      - ANCHOR_TAMPERED: the stored ``expected_hash`` does not match the hash of
        the stored ``quote`` — the anchor was edited without re-hashing.
      - SOURCE_DRIFTED: the quote no longer appears in the source — the source
        changed out from under the claim.
    """
    if expected_hash:
        try:
            algorithm, _ = parse_quote_hash(expected_hash)
            matches = quote_hash_matches(quote, expected_hash)
        except UnsupportedHashAlgorithm as exc:
            # An anchor we cannot compute is an unusable anchor: report it as
            # tampered rather than silently passing it.
            return VerifyResult(
                QuoteStatus.ANCHOR_TAMPERED, ref, expected_hash, "", str(exc),
            )
        # Echo the recomputed digest in the STORED algorithm so the two halves
        # of the result are comparable in an error message.
        actual = quote_hash(quote, algorithm)
        if not matches:
            return VerifyResult(
                QuoteStatus.ANCHOR_TAMPERED, ref, expected_hash, actual,
                "stored quote_hash does not match hash of stored quote",
            )
    else:
        actual = quote_hash(quote)
    # No stored hash means the anchor-integrity axis was undecidable — only
    # drift was checked. Flag the result partial so an ok here is not read as
    # stronger than it is (and ANCHOR_TAMPERED is unreachable on this path:
    # there is no stored hash to disagree with the quote).
    partial = not expected_hash
    if normalize_quote(quote) not in normalize_quote(source_text):
        return VerifyResult(
            QuoteStatus.SOURCE_DRIFTED, ref, expected_hash, actual,
            "cited quote not found in source", partial=partial,
        )
    return VerifyResult(QuoteStatus.OK, ref, expected_hash, actual, partial=partial)


def verify_memory_sources(file_path: str, memory_dir: str) -> list[VerifyResult]:
    """Verify every ``sources:`` anchor in one memory file.

    Returns ``[]`` when the file carries no ``sources:`` anchors — the check is
    a clean no-op on today's corpus, so it is safe to run before any anchors
    are captured.
    """
    full_path = file_path if os.path.isabs(file_path) else os.path.join(memory_dir, file_path)
    with open(full_path, encoding="utf-8") as f:
        metadata, _ = parse_markdown(f.read())
    return verify_source_anchors(metadata.get("sources"), memory_dir)


def verify_source_anchors(sources: object, memory_dir: str) -> list[VerifyResult]:
    """Verify already-parsed ``sources:`` anchors against the files they cite.

    The file-free half of :func:`verify_memory_sources`, for callers that hold
    the frontmatter already (the search freshness annotation parses each hit's
    file once and must not read it again). ``[]`` when there are no anchors.
    """
    if not isinstance(sources, list):
        return []

    results: list[VerifyResult] = []
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("ref", ""))
        quote = str(entry.get("quote", ""))
        expected = str(entry.get("quote_hash", ""))
        src_path = os.path.join(memory_dir, ref)
        if not ref or not os.path.exists(src_path):
            results.append(VerifyResult(
                QuoteStatus.SOURCE_MISSING, ref, expected, "",
                f"cited source not found: {ref or '(empty ref)'}",
            ))
            continue
        with open(src_path, encoding="utf-8") as sf:
            results.append(verify_quote(quote, expected, sf.read(), ref))
    return results
