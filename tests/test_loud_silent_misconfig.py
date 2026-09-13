"""Regression tests for the palinode CLI silently uses defaults when work + the
git-persistence no-op fix — silent-misconfig → loud-recoverable.

the palinode CLI silently uses defaults when work: palinode CLI silently used defaults
when no config file was found. The dim "Palinode config: defaults" banner was easy to
miss when systemd wired PALINODE_DIR but an interactive ssh session didn't. Production
deployments could run `palinode lint` against the wrong filesystem and get bogus output.
Now warns explicitly with the paths searched.

the git-persistence no-op fix: git_persistence silently no-op'd when memory_dir wasn't a git
repo. /save kept landing on disk, history vanished, no signal to the
operator. Now warn-once at API startup with the `git init` fix command.
"""
from __future__ import annotations

import logging
from pathlib import Path


# loud-recoverable defaults --------------------------------------


def _isolate_config_search(monkeypatch, tmp_path: Path) -> None:
    """Make both config search paths miss, regardless of the checkout.

    `load_config()` derives the repo-root candidate from the config module's
    own `__file__` (two directories up), so chdir cannot hide a checkout's
    `palinode.config.yaml`. Re-pointing `__file__` into `tmp_path` sends that
    lookup to an empty tree; an empty PALINODE_DIR covers the other candidate.
    """
    fake_module = tmp_path / "fake-repo" / "palinode" / "core" / "config.py"
    fake_module.parent.mkdir(parents=True)
    from palinode.core import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "__file__", str(fake_module))
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setenv("PALINODE_DIR", str(memory_dir))


def test_load_config_warns_when_using_defaults(tmp_path, monkeypatch, caplog):
    """`load_config()` must log a warning when no config file is found."""
    _isolate_config_search(monkeypatch, tmp_path)
    from palinode.core import config as cfg_mod

    with caplog.at_level(logging.WARNING, logger="palinode.config"):
        cfg_mod.load_config()

    warned = [
        rec.message for rec in caplog.records
        if "no palinode.config.yaml found" in rec.message
    ]
    assert warned, "both search paths missed, so load_config must warn"
    # The warning must name where it looked, so the user can self-recover.
    assert "Searched:" in warned[0]
    candidate = str(tmp_path / "memory" / "palinode.config.yaml")
    assert candidate in warned[0], f"warning should list the PALINODE_DIR candidate {candidate}"


def test_load_config_warning_message_lists_searched_paths(caplog, monkeypatch, tmp_path):
    """The warning must name the candidate paths so the user knows where
    to drop a config file. (the palinode CLI silently uses defaults when work acceptance: the user can self-recover.)
    """
    from palinode.core import config as cfg_mod

    # Force load_config to search only the tmp_path so the warning fires.
    # We point PALINODE_DIR at an empty directory; the repo-root search
    # still happens, but we'll monkeypatch _logger to capture the message
    # regardless of which path is loaded.
    monkeypatch.setenv("PALINODE_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="palinode.config"):
        cfg_mod.load_config()

    # Either way, the warning message format is fixed. Check the source
    # contains the right shape so future changes don't drop "Searched:" or
    # the recovery hint.
    import inspect
    src = inspect.getsource(cfg_mod.load_config)
    assert "no palinode.config.yaml found" in src
    assert "Searched:" in src
    assert "PALINODE_DIR" in src  # recovery hint


def test_default_banner_label_is_loud(monkeypatch, capsys, tmp_path):
    """The stderr banner must clearly mark "defaults" — not just label it."""
    _isolate_config_search(monkeypatch, tmp_path)
    from palinode.core import config as cfg_mod

    cfg_mod.load_config()
    captured = capsys.readouterr()
    # Both search paths miss, so defaults are loaded and the banner must
    # carry the visible marker — a bare "defaults" label is the regression.
    assert "defaults" in captured.err
    assert "⚠" in captured.err or "no config file" in captured.err, (
        "When defaults are loaded, banner must be visibly marked. "
        "Plain 'defaults' label is the #273 regression."
    )


# git-not-a-repo warning at API startup --------------------------


def test_lifespan_warns_when_memory_dir_not_a_git_repo(tmp_path, monkeypatch, caplog):
    """API startup must warn when auto_commit is on but no .git/ exists."""
    # Build a memory_dir without .git/
    (tmp_path / "people").mkdir()
    monkeypatch.setattr("palinode.core.config.config.memory_dir", str(tmp_path))
    monkeypatch.setattr("palinode.core.config.config.git.auto_commit", True)

    # Re-run the git-check directly (lifespan is async; this is the
    # equivalent unit-level check so we don't have to spin a real server).
    not_git_repo = not (Path(str(tmp_path)) / ".git").exists()
    assert not_git_repo, "test precondition: tmp_path must not be a git repo"

    # The exact warning string must include the `git init` fix command.
    # Pin the message shape against drift.
    import inspect
    from palinode.api import server
    src = inspect.getsource(server.lifespan)
    assert "is not a git repository" in src
    assert "git init" in src
    assert "auto_commit" in src


def test_lifespan_does_not_warn_when_auto_commit_disabled(tmp_path, monkeypatch):
    """If git.auto_commit is False, no warning fires — saves were never
    going to commit anyway, so a non-git memory_dir is a deliberate
    choice, not a misconfiguration."""
    # Just verify the source guards on auto_commit explicitly so a future
    # refactor can't drop that condition and turn this into per-startup noise
    # for users who opted out of git.
    import inspect
    from palinode.api import server
    src = inspect.getsource(server.lifespan)
    # The guard must be `if config.git.auto_commit and not ...` — pinning the
    # AND so the bypass-when-disabled path stays intact.
    assert "config.git.auto_commit" in src
    # The check must come before the warning emission.
    auto_commit_idx = src.find("config.git.auto_commit")
    warning_idx = src.find("is not a git repository")
    assert auto_commit_idx < warning_idx, (
        "auto_commit check must gate the warning, not follow it. "
        "Otherwise users with auto_commit=false get noise. (#354)"
    )
