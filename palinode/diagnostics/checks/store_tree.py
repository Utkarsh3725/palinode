"""
Check: store_tree_clean

Counts modified and untracked files in ``memory_dir``'s working tree via
``git status --porcelain``. Every store write is supposed to be committed
with provenance, so a dirty tree means some writer skipped the commit — the
description backfill did exactly that for two weeks and 572 files before
anyone looked, because ``git_remote_health`` reports unpushed *commits* and
says nothing about uncommitted *files*.

Read-only: ``git status`` never touches the index or the tree.

Severity: warn above ``_DIRTY_WARN_THRESHOLD``. Pass with the count below
it — an operator's in-progress config edit is not drift. Info when
``memory_dir`` is not a git repository.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from palinode.diagnostics.registry import register
from palinode.diagnostics.types import CheckResult, DoctorContext

_NAME = "store_tree_clean"
_GIT_TIMEOUT = 10  # seconds — a status walk of a few thousand files

#: Dirty files tolerated before the check warns. One or two is an operator
#: mid-edit; ten means a writer is skipping its commit.
_DIRTY_WARN_THRESHOLD = 10


def _run_git(args: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", cwd, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _count_dirty(porcelain: str) -> tuple[int, int]:
    """(modified_or_staged, untracked) from ``git status --porcelain`` output."""
    modified = 0
    untracked = 0
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if line.startswith("??"):
            untracked += 1
        else:
            modified += 1
    return modified, untracked


@register(tags=("fast",))
def store_tree_clean(ctx: DoctorContext) -> CheckResult:
    """Check that ``memory_dir``'s git working tree has no uncommitted files.

    Reports:
    - Pass: tree clean, or dirty count at or below the threshold
    - Warn: more than the threshold of modified/untracked files
    - Info: memory_dir is not a git repository, or git is unavailable
    """
    memory_dir = str(Path(ctx.config.memory_dir).expanduser().resolve())

    try:
        repo = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=memory_dir, timeout=3)
    except FileNotFoundError:
        return CheckResult(
            name=_NAME,
            severity="info",
            passed=True,
            message="git binary not found; cannot inspect the store's working tree.",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name=_NAME,
            severity="info",
            passed=True,
            message="git rev-parse timed out in memory_dir; skipping tree check.",
        )

    if repo.returncode != 0 or repo.stdout.strip() != "true":
        return CheckResult(
            name=_NAME,
            severity="info",
            passed=True,
            message=f"memory_dir is not a git repository: {memory_dir}",
        )

    try:
        status = _run_git(
            ["status", "--porcelain", "--untracked-files=all"],
            cwd=memory_dir,
            timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name=_NAME,
            severity="info",
            passed=True,
            message=f"git status timed out after {_GIT_TIMEOUT}s; skipping tree check.",
        )

    if status.returncode != 0:
        detail = (status.stderr.strip().splitlines() or ["(no output)"])[0]
        return CheckResult(
            name=_NAME,
            severity="info",
            passed=True,
            message=f"git status failed in memory_dir ({detail}); skipping tree check.",
        )

    modified, untracked = _count_dirty(status.stdout)
    total = modified + untracked
    counts = f"{modified} modified, {untracked} untracked"

    if total > _DIRTY_WARN_THRESHOLD:
        return CheckResult(
            name=_NAME,
            severity="warn",
            passed=False,
            message=(
                f"{total} uncommitted files in the memory store ({counts}). "
                "Some writer is skipping its commit; git status is no longer a "
                "'what changed' signal and a checkout or stash would discard the drift."
            ),
            remediation=(
                f"Inspect with 'git -C {memory_dir} status --short'. To commit the "
                f"backlog in one step: 'git -C {memory_dir} add -A -- people projects "
                "decisions insights research inbox daily && "
                f"git -C {memory_dir} commit -m \"palinode: commit uncommitted store writes\"'. "
                "If the count keeps growing, find the writer: every store write must "
                "commit through palinode.core.git_tools."
            ),
        )

    if total:
        return CheckResult(
            name=_NAME,
            severity="warn",
            passed=True,
            message=f"Memory store tree has {total} uncommitted file(s) ({counts}) — within tolerance.",
        )

    return CheckResult(
        name=_NAME,
        severity="warn",
        passed=True,
        message="Memory store working tree is clean.",
    )
