# palinode-plugin-core

The shared TypeScript core for Palinode's harness plugins (`plugins/pi`,
`plugins/cline`). Extracted on the third plugin, per the rule recorded in
ADR-019 §4: two implementations do not reveal which parts are common and
which are harness-shaped; three do.

It contains exactly what the plugins duplicated before extraction, and
nothing hook-shaped:

| Export | What it is |
|--------|------------|
| `configFromEnv(env, overrides)` | The shared knobs (`PALINODE_API_URL`, `PALINODE_API_TOKEN`, `PALINODE_HOOK_RECALL_*`, `PALINODE_HOOK_INJECT_*`, `PALINODE_HOOK_MIN_MESSAGES`) plus `PALINODE_HOOK_RECALL_PROFILE`; harness config overrides the env |
| `PROFILES` | Recall profiles (`coding`, `monitoring`, `investigation`, `writing`, `conversation`, `minimal`, `off`) — the OpenClaw plugin's vocabulary, expressed as which channels are on |
| `apiJson(cfg, fetch, path, init)` | The fail-open REST client: bearer, timeout, HTTP ≥ 400 → `null`, never throws |
| `buildRecallContext(prompt, cfg, fetch)` | Per-turn recall: fired triggers + the memory channel (bounded resolution, falling back to strict-threshold search) → one bounded text block, or `null` |
| `resolveBundle(prompt, cfg, fetch, maxChars)` | One `POST /resolve` under the per-turn deadline: the qualified bundle, or `null` when it did not arrive in time |
| `RESOLUTION_DEADLINE_MARKER` | The line the fallback carries so unresolved hits are never read as a resolved answer |
| `trimToUnitBoundary(text, maxChars)` | The only trim on the injection path: cuts between units, drops a contested block whole and names it with the server's own stub |
| `buildCoreDigest(cfg, fetch, cwd, sessionId)` | Session-start priming: warm `/context/prime`, digest of `core: true` memories, or `null` |
| `buildSessionCapture(entries, cfg, origin)` / `postSessionCapture` | The capture-floor payload for `/session-end`, over Pi- or Cline-shaped message entries |

## Routing: which path gets the qualified answer

Three ways memory reaches a session, and this core keeps them apart:

| When | What runs | Budget |
|------|-----------|--------|
| **Session start** (`buildCoreDigest`) | Ordinary priming: warm `/context/prime`, digest the `core: true` memories. **No resolution** — a startup digest is orientation, not an answer, and there is no question yet to resolve. | Its own payload, `coreMaxChars`, `timeoutMs` |
| **Per turn** (`buildRecallContext`) | Bounded resolution: `POST /resolve` for the prompt, injected as the rendered bundle. | `resolveDeadlineMs` (**250 ms**, `PALINODE_HOOK_RESOLVE_DEADLINE`) and the injection cap minus the preamble |
| **Explicit follow-up** | `palinode_resolve` / `palinode_search` / `palinode_read` as tools, agent-initiated. Injection is a starting point; these are the way to the rest. | No deadline |

Past the deadline the turn falls back to today's search payload — byte for
byte — prefixed with `RESOLUTION_DEADLINE_MARKER`. The marker is the point: a
silent fallback would hand the model an unchecked hit with the authority of a
resolved answer. `PALINODE_HOOK_RESOLVE=0` opts out entirely and restores the
pre-resolution behaviour with no marker, because nothing then claims to have
resolved anything.

The injection cap is spent in one direction only: the bundle is requested at
`maxChars − preamble`, and if the total still overruns, the **trigger** section
is trimmed. The bundle is never sliced — the server packs conflicts whole, and
cutting the text afterwards is exactly how one side of a conflict goes missing.
Under ~300 characters of room the channel says nothing at all.

### No client-side slicing

There is **no `text.slice(0, cap)` anywhere on the injection path.** Both
payload builders end in `trimToUnitBoundary(text, cap)` instead:

- the cut lands between lines, never inside one, so a row cannot arrive with
  its qualifiers (or its path) missing;
- a block is a line plus the indented lines under it, so a row and its
  qualifiers travel together;
- a block that does not fit is dropped **whole**, and if it was a contested
  one (`⚠ contradicts`, a `Contested (…)` section, the bundle's `Still
  contested` notice) the same stub the server emits takes its place —
  `⚠ N conflicts omitted for budget — see <refs>` — evicting further kept
  blocks to make room, because naming a conflict outranks one more ordinary
  row;
- below room for even that stub, nothing is rendered: a fragment that fit by
  dropping the notice would read as settled.

The server already packs both payloads to a budget (`max_chars` on `/resolve`;
`context.injection_max_*` on the prime path), so this is the last line of
defence rather than the budget itself — but it is the line that used to be a
plain slice, and a plain slice is how a payload packed honestly arrives
dishonest.

## The one invariant this core owns

Everything this module produces for injection is a **message body**. There
is no function here that yields a system prompt, and no binding may route
these strings into one (ADR-019 §4). Model providers cache the prompt
as a strict prefix — tools, then system, then messages — so per-turn content
in the system prompt invalidates the whole cached prefix every turn and
costs more than the recall saves. Bindings append a message after the cached
prefix instead, and each binding's own test suite pins that.

## How it is consumed

Not published to npm. Each plugin compiles the core into its own `dist/`
(`tsconfig` `rootDir: ".."`, `include: ["src/**/*.ts", "../core/src/**/*.ts"]`),
so an installed plugin has no runtime dependency on this directory. Import
it by relative path: `import { ... } from "../../core/src/index.js"`.

```bash
npm install
npm test        # vitest — pure functions over an injected fetch
```
