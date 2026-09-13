"""Materializing an episode: real files, real index, real executor, real git.

An episode's events are replayed through the production write paths, never
through a fixture that imitates them:

* a save writes markdown under a throwaway ``PALINODE_DIR`` and runs the real
  ``index_file`` pipeline (parse → projection → SHA-256 dedup → embed → FTS +
  vector upsert);
* an explicit replacement goes through ``consolidation.archive.archive_memory``,
  which is what writes ``status: archived`` + ``superseded_by`` and pushes the
  status into the index;
* fact-level supersession, archival and retraction go through
  ``consolidation.executor.apply_operations`` — the deterministic op executor
  itself;
* a consolidation pass goes through ``consolidation.runner.run_nightly`` /
  ``run_consolidation`` with a **scripted** ``llm_fn``. The proposal seam is
  the only thing replaced, and it is replaced with a constant, so the corpus
  contains no model output and no network call.

Two things the harness needs that production reads from the wall clock are
injected instead: the evaluation clock (``World.now``, handed to
``build_bundle`` / ``resolve_evidence`` / ``eligibility``) and the consolidation
runner's own ``_utc_now``, patched for the duration of a pass so a dated corpus
does not decay as the calendar moves.

The embedder is deterministic by default: a hashed bag-of-words vector, so
cosine similarity is real lexical overlap rather than a random projection, and
a run reproduces byte-for-byte on a host with no Ollama. That is a **stated
limitation**, not a hidden one — paraphrase recall at the seed stage is
lexical here, and :mod:`bench.current_state.report` says so. Pass
``embedder="ollama"`` to measure against the real one.

Git is real too, because one of the invariants is "a read arm mutates nothing":
every event is committed, and the harness compares ``HEAD`` and the porcelain
status before and after each read arm.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from bench import harness as bench_harness

#: Vector dimension fallback when the configured embedding model is unknown.
_FALLBACK_DIMS = 1024

#: Words dropped from the deterministic embedder's bag. Small on purpose — the
#: point is lexical overlap, not a tuned retrieval model.
_STOPWORDS = frozenset(
    "a an the and or of to in for on is are was were be been it its this that "
    "we our you your they their i as at by with from what which who how does do "
    "did no not".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ── deterministic embedder ───────────────────────────────────────────────────


def hashed_embedding(text: str, dims: int) -> list[float]:
    """A deterministic bag-of-words vector: hash each token into a bucket.

    Cosine between two of these is a real lexical-overlap measure, which is
    what makes the hybrid arm meaningful without a model. It is **not** a
    semantic embedding: two paraphrases with no shared content words are
    orthogonal here and would not be under bge-m3.
    """
    counts: dict[int, float] = {}
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % dims
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    vector = [0.0] * dims
    if not counts:
        return vector
    norm = math.sqrt(sum(v * v for v in counts.values()))
    for bucket, value in counts.items():
        vector[bucket] = value / norm
    return vector


@contextlib.contextmanager
def deterministic_embedder() -> Iterator[None]:
    """Install :func:`hashed_embedding` as the process embedder for a block.

    Both entry points are replaced: ``embed`` (the query path) and
    ``embed_many`` (the batch the reconcile pass prefers). Patching only the
    first leaves the ingest path reaching for Ollama and aborting the
    reconcile, which is a silently empty index rather than a loud failure.
    """
    from palinode.core import embedder as embedder_mod
    from palinode.core.config import config
    from palinode.core.ollama_client import get_ollama_client

    try:
        dims = int(config.embeddings.primary.dimensions)
    except Exception:
        dims = _FALLBACK_DIMS

    client = get_ollama_client()
    original = embedder_mod.embed
    original_many = embedder_mod.embed_many
    original_probe = client.probe_embed
    embedder_mod.embed = lambda text: hashed_embedding(text, dims)  # type: ignore[assignment]
    embedder_mod.embed_many = (  # type: ignore[assignment]
        lambda texts: [hashed_embedding(t, dims) for t in texts]
    )
    client.probe_embed = lambda **kwargs: True  # type: ignore[method-assign]
    try:
        yield
    finally:
        embedder_mod.embed = original  # type: ignore[assignment]
        embedder_mod.embed_many = original_many  # type: ignore[assignment]
        client.probe_embed = original_probe  # type: ignore[method-assign]


# ── clock injection ──────────────────────────────────────────────────────────


@contextlib.contextmanager
def frozen_consolidation_clock(now: datetime) -> Iterator[None]:
    """Freeze the consolidation layer's own wall-clock reads at *now*.

    The runner filters daily notes by ``_utc_now() - lookback_days``, the
    executor stamps tombstones with today's date, and the ``backed_by``
    propagation stamps each ``stale_backing`` entry with ``at:``. All three are
    wall-clock reads in production; all three must follow the episode's clock,
    or a corpus authored in one month stops consolidating in the next and the
    store's bytes differ between two runs of the same corpus.
    """
    from palinode.consolidation import activity_gate, executor, propagate, runner

    targets = [(runner, "_utc_now"), (activity_gate, "_utc_now"),
               (executor, "_utc_now"), (propagate, "_utc_now")]
    originals = [(mod, name, getattr(mod, name)) for mod, name in targets]
    for mod, name in targets:
        setattr(mod, name, lambda: now)
    try:
        yield
    finally:
        for mod, name, original in originals:
            setattr(mod, name, original)


# ── the world ────────────────────────────────────────────────────────────────


@dataclass
class EventOutcome:
    """What one event did — carried into the report when a pass was a no-op."""

    op: str
    detail: dict[str, Any] = field(default_factory=dict)


class World:
    """One episode's store: files, index, git history, and an evaluation clock.

    Construct, :meth:`setup`, then :meth:`apply` events in order. Every write
    path is the production one; the only substitutions are the embedder (see
    the module doc) and the consolidation proposal seam.
    """

    def __init__(self, root: str, *, start: datetime | None = None) -> None:
        self.root = root
        self.now = start or datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        self.outcomes: list[EventOutcome] = []
        self._committed = 0

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        """Point the global config here, create the schema, init git."""
        os.makedirs(self.root, exist_ok=True)
        # Resolve symlinks before anything records the path. On macOS a temp
        # dir is ``/var/folders/...`` whose real path is ``/private/var/...``,
        # and ``path_guard`` compares realpaths: an unresolved root makes every
        # write through the guard look like a traversal attempt.
        self.root = os.path.realpath(self.root)
        bench_harness.point_config_at(self.root)
        bench_harness.init_store()
        self._git_init()

    def attach(self) -> None:
        """Point the global config at an already-materialized store.

        What :meth:`setup` does minus the schema and the git init — used for a
        copied store whose files and history are already on disk.
        """
        self.root = os.path.realpath(self.root)
        bench_harness.point_config_at(self.root)

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=check
        )

    def _git_init(self) -> None:
        Path(self.root, ".gitignore").write_text(
            ".palinode.db\n.palinode.db-*\n.palinode/\n", encoding="utf-8"
        )
        self._git("init", "-q")
        self._git("config", "user.email", "bench@example.invalid")
        self._git("config", "user.name", "current-state bench")
        self._git("config", "commit.gpgsign", "false")
        self.commit("corpus: initial state")

    def commit(self, message: str) -> None:
        """Commit everything tracked. Empty commits are skipped, not forced."""
        self._git("add", "-A")
        status = self._git("status", "--porcelain")
        if not status.stdout.strip():
            return
        self._git("commit", "-q", "-m", message)
        self._committed += 1

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.strip()

    def dirty(self) -> str:
        return self._git("status", "--porcelain").stdout.strip()

    # -- helpers ------------------------------------------------------------

    def rel_path(self, ref: str) -> str:
        """The memory-relative path for a ref.

        The path guard rejects absolute paths outright, so every op that goes
        through ``resolve_memory_ref`` (archive, retract) is handed this, not
        :meth:`abs_path`.
        """
        return ref if ref.endswith(".md") else f"{ref}.md"

    def abs_path(self, ref: str) -> str:
        return os.path.join(self.root, self.rel_path(ref))

    def index(self, ref: str) -> dict[str, Any]:
        from palinode.indexer.index_file import index_file

        return index_file(self.abs_path(ref))

    def reindex_all(self) -> int:
        from palinode.indexer.index_file import index_file

        count = 0
        for path in sorted(Path(self.root).rglob("*.md")):
            index_file(str(path))
            count += 1
        return count

    def _write(self, ref: str, meta: dict[str, Any], body: str) -> str:
        path = self.abs_path(ref)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        front = yaml.dump(meta, default_flow_style=False, sort_keys=True)
        Path(path).write_text(f"---\n{front}---\n\n{body.rstrip()}\n", encoding="utf-8")
        return path

    def read(self, ref: str) -> str:
        return Path(self.abs_path(ref)).read_text(encoding="utf-8")

    def _meta_of(self, ref: str) -> tuple[dict[str, Any], str]:
        from palinode.core.parser import split_frontmatter

        raw = self.read(ref)
        head, body = split_frontmatter(raw)
        meta = yaml.safe_load(head.strip().strip("-").strip() or "{}") or {}
        return meta, body

    # -- events -------------------------------------------------------------

    def apply(self, event: dict[str, Any]) -> EventOutcome:
        """Apply one corpus event. Unknown ops raise — silence would be a lie."""
        op = event.get("op")
        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            raise ValueError(f"unknown corpus event op: {op!r}")
        # Time passes as memories are written: an event dated later than the
        # current clock moves it forward before the event is applied, so a
        # record is never written into its own future and a tombstone carries
        # the date the change actually happened. `advance_clock` is the only
        # thing that can move it the other way.
        declared = event.get("date")
        if isinstance(declared, str):
            moment = datetime.fromisoformat(declared)
            moment = moment if moment.tzinfo else moment.replace(tzinfo=UTC, hour=9)
            self.now = max(self.now, moment)
        outcome = handler(event)
        self.commit(f"{op}: {event.get('ref') or event.get('to') or ''}".strip(": "))
        self.outcomes.append(outcome)
        return outcome

    def _op_save(self, event: dict[str, Any]) -> EventOutcome:
        ref = event["ref"]
        category = ref.split("/", 1)[0]
        meta: dict[str, Any] = {
            "id": ref.replace("/", "-"),
            "category": category,
            "type": event.get("type", "Insight"),
            "date": event.get("date", self.now.date().isoformat()),
            "created_at": event.get("date", self.now.date().isoformat()),
            "last_updated": event.get("date", self.now.date().isoformat()),
        }
        if event.get("title"):
            meta["title"] = event["title"]
        meta.update(event.get("fm") or {})
        title = event.get("title") or ref.rsplit("/", 1)[-1].replace("-", " ")
        body = f"# {title}\n\n{event['body'].strip()}\n"
        self._write(ref, meta, body)
        result = self.index(ref)
        return EventOutcome("save", {"ref": ref, "chunks": result["chunks_written"]})

    def _op_set_frontmatter(self, event: dict[str, Any]) -> EventOutcome:
        """Edit frontmatter only. ``reindex: false`` reproduces the real index
        lag: ``index_file`` skips the chunk upsert when the section body hash is
        unchanged, so a frontmatter-only edit leaves ``chunks.metadata`` stale."""
        ref = event["ref"]
        meta, body = self._meta_of(ref)
        for key, value in (event.get("fm") or {}).items():
            if value is None:
                meta.pop(key, None)
            else:
                meta[key] = value
        self._write(ref, meta, body.strip())
        if event.get("reindex", True):
            self.index(ref)
        return EventOutcome("set_frontmatter", {"ref": ref})

    def _op_edit_body(self, event: dict[str, Any]) -> EventOutcome:
        ref = event["ref"]
        meta, _ = self._meta_of(ref)
        title = meta.get("title") or ref.rsplit("/", 1)[-1].replace("-", " ")
        self._write(ref, meta, f"# {title}\n\n{event['body'].strip()}\n")
        if event.get("reindex", True):
            self.index(ref)
        return EventOutcome("edit_body", {"ref": ref, "reindexed": event.get("reindex", True)})

    def _op_touch(self, event: dict[str, Any]) -> EventOutcome:
        """Rewrite identical bytes and bump the file mtime — a regenerated
        summary. Must not become a new effective decision date."""
        ref = event["ref"]
        path = self.abs_path(ref)
        raw = Path(path).read_text(encoding="utf-8")
        Path(path).write_text(raw, encoding="utf-8")
        os.utime(path, None)
        self.index(ref)
        return EventOutcome("touch", {"ref": ref})

    def _op_supersede_record(self, event: dict[str, Any]) -> EventOutcome:
        from palinode.consolidation.archive import archive_memory

        with frozen_consolidation_clock(self.now):
            result = archive_memory(
                self.rel_path(event["ref"]),
                reason=event.get("reason", "explicitly replaced"),
                superseded_by=event["by"],
            )
        self.index(event["ref"])
        return EventOutcome("supersede_record", {"ref": event["ref"], "status": result["status"]})

    def _op_archive_record(self, event: dict[str, Any]) -> EventOutcome:
        from palinode.consolidation.archive import archive_memory

        with frozen_consolidation_clock(self.now):
            result = archive_memory(
                self.rel_path(event["ref"]), reason=event.get("reason", "retired")
            )
        self.index(event["ref"])
        return EventOutcome("archive_record", {"ref": event["ref"], "status": result["status"]})

    def _op_move_to_archive(self, event: dict[str, Any]) -> EventOutcome:
        """Retire by *location* only — what the weekly pass does to a daily note:
        the bytes and the frontmatter are untouched, the move is the statement."""
        from palinode.core import store

        ref = event["ref"]
        src = self.abs_path(ref)
        dst_ref = f"archive/{ref}"
        dst = self.abs_path(dst_ref)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        store.delete_file_chunks(src)
        self.index(dst_ref)
        return EventOutcome("move_to_archive", {"ref": ref, "to": dst_ref})

    def _op_restore_from_archive(self, event: dict[str, Any]) -> EventOutcome:
        """An explicit restore: move the file back, frontmatter untouched."""
        from palinode.core import store

        ref = event["ref"]
        src = self.abs_path(f"archive/{ref}")
        dst = self.abs_path(ref)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        store.delete_file_chunks(src)
        self.index(ref)
        return EventOutcome("restore_from_archive", {"ref": ref})

    def _op_delete(self, event: dict[str, Any]) -> EventOutcome:
        from palinode.core import store

        path = self.abs_path(event["ref"])
        store.delete_file_chunks(path)
        os.remove(path)
        return EventOutcome("delete", {"ref": event["ref"]})

    def _fact_ops(self, event: dict[str, Any], ops: list[dict[str, Any]]) -> EventOutcome:
        from palinode.consolidation.executor import apply_operations

        with frozen_consolidation_clock(self.now):
            stats = apply_operations(self.abs_path(event["ref"]), ops)
        self.index(event["ref"])
        if stats["unmatched"]:
            raise ValueError(
                f"{event['ref']}: {stats['unmatched']} executor op(s) matched nothing "
                f"({ops!r}) — the corpus event did not do what it claims"
            )
        return EventOutcome(event["op"], {"ref": event["ref"], "stats": stats})

    def _op_supersede_fact(self, event: dict[str, Any]) -> EventOutcome:
        return self._fact_ops(event, [{
            "op": "SUPERSEDE",
            "id": event["fact"],
            "new_text": event["new_text"],
            "rationale": event.get("reason", "corrected"),
        }])

    def _op_archive_fact(self, event: dict[str, Any]) -> EventOutcome:
        op: dict[str, Any] = {
            "op": "ARCHIVE", "id": event["fact"],
            "reason": event.get("reason", "retired"),
        }
        if event.get("superseded_by"):
            op["superseded_by"] = event["superseded_by"]
        return self._fact_ops(event, [op])

    def _op_retract_fact(self, event: dict[str, Any]) -> EventOutcome:
        op: dict[str, Any] = {
            "op": "RETRACT", "id": event["fact"],
            "reason": event.get("reason", "known incorrect"),
        }
        if event.get("falsified_by"):
            op["falsified_by"] = event["falsified_by"]
        return self._fact_ops(event, [op])

    def _op_retract_mention(self, event: dict[str, Any]) -> EventOutcome:
        from palinode.consolidation.retract import retract_mentions

        with frozen_consolidation_clock(self.now):
            result = retract_mentions(
                self.rel_path(event["ref"]),
                event["pref"],
                superseded_by=event.get("superseded_by"),
            )
        if result["status"] not in ("retracted", "already_retracted"):
            raise ValueError(
                f"{event['ref']}: retract_mentions returned {result['status']!r} — "
                "the corpus event matched nothing"
            )
        self.index(event["ref"])
        return EventOutcome("retract_mention", {"ref": event["ref"], "status": result["status"]})

    def _op_consolidate(self, event: dict[str, Any]) -> EventOutcome:
        """A real consolidation pass with a scripted proposal.

        ``llm_fn`` is the runner's documented propose seam and is the *only*
        thing replaced: the prompt is still assembled, the proposal guard still
        runs, the ops filter still applies, and the executor still applies what
        survives. A pass that was asked to change something and changed nothing
        raises, because a silently no-op consolidation would quietly turn every
        downstream oracle into a test of the un-consolidated store.
        """
        from palinode.consolidation import runner

        ops = json.dumps(event.get("ops") or [])

        def scripted(system_prompt: str, user_prompt: str) -> tuple[str, str]:
            return ops, "scripted"

        which = event.get("pass", "weekly")
        with frozen_consolidation_clock(self.now):
            if which == "nightly":
                result = runner.run_nightly(llm_fn=scripted)
            else:
                result = runner.run_consolidation(llm_fn=scripted)
        self.reindex_all()
        if event.get("ops") and not result.get("projects_compacted"):
            raise ValueError(
                f"consolidate({which}) compacted no project: {result!r} — the corpus "
                "event expected the scripted ops to reach a target document"
            )
        return EventOutcome("consolidate", {"pass": which, "result": result})

    def _op_advance_clock(self, event: dict[str, Any]) -> EventOutcome:
        raw = str(event["to"])
        moment = datetime.fromisoformat(raw)
        self.now = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
        return EventOutcome("advance_clock", {"to": self.now.isoformat()})

    def _op_reindex(self, event: dict[str, Any]) -> EventOutcome:
        return EventOutcome("reindex", {"files": self.reindex_all()})

    def _op_drop_index(self, event: dict[str, Any]) -> EventOutcome:
        """Delete the derived state and rebuild it — the rebuild/replay case."""
        from palinode.core import store

        store._db_checked = False
        db_path = os.path.join(self.root, ".palinode.db")
        for suffix in ("", "-wal", "-shm", "-journal"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(db_path + suffix)
        bench_harness.init_store()
        return EventOutcome("drop_index", {"files": self.reindex_all()})


# ── building a world from an episode ─────────────────────────────────────────


def build(
    episode: Any,
    root: str,
    *,
    until: int | None = None,
    skip_ops: tuple[str, ...] = (),
    on_checkpoint: Callable[[World, int], None] | None = None,
) -> World:
    """Replay *episode*'s events into a fresh world at *root*.

    ``until`` stops after that many events (the before-state checkpoint);
    ``skip_ops`` drops whole event kinds, which is how the raw-evidence control
    runs the *same* episode with consolidation never happening.
    """
    world = World(root, start=_episode_start(episode))
    world.setup()
    for position, event in enumerate(episode.events, start=1):
        if until is not None and position > until:
            break
        if event.get("op") in skip_ops:
            continue
        world.apply(event)
        if on_checkpoint is not None:
            on_checkpoint(world, position)
    return world


def _episode_start(episode: Any) -> datetime:
    """The episode's clock at t0: the earliest date any event declares."""
    dates = [
        str(e["date"]) for e in episode.events if isinstance(e.get("date"), (str,))
    ]
    if not dates:
        return datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    earliest = min(dates)
    moment = datetime.fromisoformat(earliest)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC, hour=9)


__all__ = [
    "EventOutcome",
    "World",
    "build",
    "deterministic_embedder",
    "frozen_consolidation_clock",
    "hashed_embedding",
]
