# Contributing to Palinode

Thanks for looking. Palinode is a small project with a specific shape, and the
fastest way to have a good time contributing is to know that shape first.

## What this project is trying to be

Palinode is the **reference implementation of auditable agent memory**: memory where
every remembered fact is a line in a git-versioned markdown file that you can read,
`diff`, `blame`, and roll back — and where each stored claim can carry an explicit
epistemic status, typed links to the evidence that backs or contradicts it, and a
verifiable quote-level citation to its source.

That phrase — *reference implementation* — is doing real work. Palinode is not trying
to out-scale hosted memory services, and it is not competing for the largest install
base. It is trying to be the clearest, most correct, most inspectable expression of a
particular idea about how agent memory should work. **Contributions that make it
clearer or more correct are worth more here than contributions that make it bigger.**

## Ways to contribute

- **Report a bug.** Include your OS, Python version, `palinode --version`, and the
  output of `palinode doctor` if the daemon is involved.
- **Improve documentation.** Docs PRs are always welcome and are the lowest-friction
  way to start. If a setup step confused you, that is a bug in the docs.
- **Pick up an issue.** Issues labelled `good first issue` are scoped to be
  self-contained. `help wanted` means the maintainer would genuinely like a hand.
  Comment on the one you want and it is yours — and **start with one.** Holding
  several at a time usually means several sit still, and a claimed issue nobody is
  moving is worse than an unclaimed one.
- **Report a security issue.** Please do not open a public issue — see
  [`SECURITY.md`](SECURITY.md).

If you are planning something substantial, **open an issue before writing the code.**
Palinode is maintained by one person and has a deliberately narrow scope; the worst
outcome for both of us is you finishing a large PR that does not fit.

**There is no payment or bounty programme.** Everything here is volunteer work, and
`good first issue` marks a good way into the codebase rather than a listing. Offers to
implement an issue for a fee get declined — politely, and every time. Saying so here so
nobody spends their time writing one.

## Before you start: scope

Things that are likely to be accepted:

- Bug fixes with a regression test
- Documentation and error-message improvements
- Platform/compatibility fixes (other OSes, Python versions, MCP clients)
- Performance work with a benchmark showing the difference
- Making an existing capability available on a surface that is missing it

Things that are likely to be declined, and why:

- **Speculative abstractions.** Interfaces, plugin systems, or config knobs added for
  a use case nobody has yet. If it has one caller, it does not need an abstraction.
- **New required runtime dependencies.** Palinode runs on a directory of markdown, one
  SQLite file, and an embedding endpoint. Adding Postgres, Redis, or a message queue
  to the required path is a hard no.
- **Anything that lets a model write to disk directly.** See the invariants below.
- **Features that only work with one vendor's cloud.**

## Development setup

```bash
git clone https://github.com/phasespace-labs/palinode
cd palinode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires **Python 3.11+**.

Run the tests:

```bash
pytest                    # unit + integration; live tests are excluded by default
pytest -m "not slow"      # skip tests that need a running embedding endpoint
```

Lint and static security analysis, both of which CI gates on:

```bash
ruff check palinode/ tests/ scripts/
bandit -r palinode/ -ll
```

The ruff rule set is deliberately narrow — bug classes only (`F`, `B`, `E7`, `E9`),
no formatting opinions. Please do not widen it in a feature PR; the reasoning for
each excluded rule is commented in `pyproject.toml`.

### Cite issues by URL, not by number

One gate surprises people, so it is worth knowing before CI tells you: **an issue
reference in code has to be a full URL, not a bare `#123`.**

Palinode is developed in a private repository and synced here, so the same `#123` means
different things in the two trackers. A bare number is unfollowable for a public reader
and can resolve to a different issue entirely. The guard cannot tell which tracker you
meant — including when you meant this one — so it asks everyone for the qualified form.

```python
"""Fixes #123."""                                                    # fails CI
"""Fixes https://github.com/phasespace-labs/palinode/issues/123."""  # fine
```

Or name the thing instead — "fixes the frontmatter split" — and skip the reference
entirely. Either passes.

It applies to docstrings, comments, CLI `--help` text, MCP tool and parameter
descriptions, CI workflow comments, and `.gitattributes`/`.gitignore`. Files under
`docs/` are not scanned, so a changelog entry cites issues normally. The rule lives in
`tests/test_no_issue_refs_user_surface.py`, which pins the distinction with its own
tests.

More detail, including a gotcha about editable installs inside `git worktree`
checkouts, is in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## The invariants

These are the rules that make Palinode what it is. A PR that violates one will be
asked to change regardless of how good the rest of it is — so it is worth reading
this section before you write code, not after.

**1. Files are the source of truth.** Markdown + YAML frontmatter under the memory
directory is canonical. The SQLite index is a *derived* artifact and must always be
rebuildable from the files alone. If a change makes the index authoritative for
anything, that is a design bug. The test: if you delete the database, nothing the user
cares about should be lost.

**2. Writes must be validated.** Memory changes must pass the applicable path,
content, and retirement checks and use the existing write functions. A new path
that skips those checks will be rejected.

**3. Every write is committed with provenance.** A memory that changed without a
commit recording who changed it and why is not auditable, which defeats the purpose.

**4. All surfaces stay at parity.** MCP server, REST API, CLI, and plugin should
expose the same capabilities. If you add something to one, either add it to the others
or say in the PR why it genuinely belongs to only one.

**5. Never mock the database in tests.** Use a real SQLite database against `tmp_path`.
Mocked stores hide exactly the bugs this project needs to catch. Integration tests that
mock the store will be asked to use the real thing.

**6. Validate every user-supplied path.** Reject `..` traversal and symlinks that
escape the memory directory. This is user data on a real filesystem.

## Submitting a pull request

1. Branch from `main`.
2. Write a test that fails before your change and passes after it.
3. Run `pytest`, `ruff`, and `bandit` locally — CI runs all three.
4. **Add a CHANGELOG entry.** Put a bullet under the existing `## Unreleased` heading
   in `docs/CHANGELOG.md` — the topmost section — in the right Added / Changed / Fixed /
   Removed / Security bucket. Add it under an existing heading — please do not create a
   new one, as the file uses a union merge strategy and duplicate headings survive
   merges silently. And never add it to an already-released version's section, even if
   similar bullets there look like the natural place: released sections are immutable
   records, they are hash-frozen, and CI will fail the PR.
   Genuinely non-shipping PRs (docs, CI) can note `skip-changelog` instead.
5. Describe *why* in the PR body, not just what. The diff shows what.

Small, focused PRs get reviewed faster than large ones. If your change is large,
splitting it is almost always worth the effort.

## Commit messages

Conventional-commit prefixes (`fix:`, `feat:`, `docs:`, `test:`, `refactor:`, `chore:`)
are used throughout the history. Beyond the prefix, write a body that explains the
reasoning — this project treats commit messages as documentation, and several of them
are the only record of why something is the way it is.

## A note on history and cadence

Palinode has one maintainer, so review latency varies. The published git history is
release-granular rather than commit-by-commit, which means `git log` here reads as a
sequence of releases rather than a development diary. Neither of these affects your
contribution — PRs are reviewed and merged normally — but both surprise people who
expect a large-team rhythm, so they are worth saying out loud.

## Code of conduct

Be decent. Assume good faith, critique the work rather than the person, and accept
that "no, and here is why" is a legitimate and common answer to a feature proposal.
Behaviour that makes the project worse to participate in gets you removed from it.

## Licence

Palinode is MIT-licensed. By contributing, you agree that your contributions are
licensed under the same terms.
