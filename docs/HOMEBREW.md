# Install and operate Palinode with Homebrew

Homebrew manages Palinode's Python environment and exposes `palinode`,
`palinode-api`, `palinode-watcher`, `palinode-mcp`, and the HTTP MCP entry point.
You need Git with your user name and email configured so memory writes can be
committed. Your memory directory is independent of the installed package.

## Install and start

For local search, [install and start Ollama](https://formulae.brew.sh/formula/ollama)
and download the embedding model (about 1.2 GB):

```sh
brew install ollama
brew services start ollama
ollama pull bge-m3
```

Skip this step if Ollama is already running with `bge-m3`, or configure another
supported embedding endpoint. Then install Palinode:

```sh
brew install phasespace-labs/palinode/palinode
palinode --version
mkdir -p ~/.palinode
git -C ~/.palinode init
PALINODE_DIR=~/.palinode palinode start
```

Keep that terminal open: `palinode start` runs the API and watcher in the
foreground. In a second terminal:

```sh
export PALINODE_DIR="$HOME/.palinode"
palinode doctor
palinode save --type Decision "We chose SQLite for the sample project's cache because it needs no separate database service."
palinode search "SQLite cache"
```

You can inspect the saved memory at <http://127.0.0.1:6340/ui/>. See the
[inspector guide](UI.md) for access details and its read-only behavior.

Saves persist without an embedding service, but the search API returns HTTP 503
until the embedder is reachable; `palinode resolve` falls back to keyword-only
matching and reports `degraded:keyword_only` in its coverage. A chat model is
optional for consolidation.
Use `palinode doctor` to distinguish an unavailable embedding service from other
setup problems.

## Connect an editor

For Claude Code:

```sh
claude mcp add --transport stdio --env PALINODE_DIR="$HOME/.palinode" \
  palinode -- "$(brew --prefix palinode)/bin/palinode-mcp"
```

For another editor, use `palinode mcp-config --stdio` and replace the generated
command with the absolute path printed by:

```sh
echo "$(brew --prefix palinode)/bin/palinode-mcp"
```

Also set `PALINODE_DIR` in the server entry's `env` to the absolute path printed
by `echo "$HOME/.palinode"`; GUI clients do not expand `~` or inherit this terminal's
environment reliably. Follow that client's [configuration and restart recipe](MCP-INSTALL-RECIPES.md).
The absolute path avoids depending on a GUI application's shell PATH. From your
project directory, `palinode init` adds the relevant memory instructions and
supported hooks. [Integration behavior](HARNESSES.md) explains which actions run
automatically for each client. Confirm that your agent can call `palinode_status`
and recall the sample decision you saved above.

## Upgrade

Stop foreground services with Ctrl-C, then:

```sh
brew update
brew upgrade phasespace-labs/palinode/palinode
palinode --version
PALINODE_DIR=~/.palinode palinode start
```

Run `palinode doctor` from the other terminal after startup. Read the release's
compatibility notes for any store migration or prompt refresh; a package upgrade
does not replace your existing store configuration or prompt edits.

For services managed by launchd or systemd, stop and restart them through that
manager instead. Use the stable path under `brew --prefix palinode` for installed
executables, rather than a version-specific Cellar path. The formula does not
register a `brew services` service. See the
[macOS](../deploy/launchd/README.md) and [Linux](../deploy/systemd/README.md) guides
for persistent service setup.

Homebrew follows public stable releases through reviewed tap updates. If the
source release is newer than your formula, check the
[tap update PRs](https://github.com/phasespace-labs/homebrew-palinode/pulls) and
[update workflow](https://github.com/phasespace-labs/homebrew-palinode/actions/workflows/update-release.yml).
The source release and its Homebrew distribution are separate steps; `brew update`
cannot fetch a formula update that has not been merged yet.
