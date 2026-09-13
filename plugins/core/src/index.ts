/**
 * palinode-plugin-core — the shared TypeScript core for Palinode's harness
 * plugins (ADR-019 delivery adapters; extracted on the third plugin per #1002).
 *
 * Everything here is a plain function over an injected `fetch`, so the whole
 * recall/prime/capture surface is testable without any harness installed.
 * Each harness binding (`plugins/pi`, `plugins/cline`) is deliberately thin:
 * it wires these functions to lifecycle events and nothing else.
 *
 * What lives here is exactly what the plugins duplicated before extraction —
 * nothing hook-shaped, nothing harness-shaped:
 *   - the fail-open REST client (bearer, timeout, HTTP>=400 → null)
 *   - config resolution from the shared env knobs (+ recall profiles)
 *   - recall → injection TEXT (triggers + strict search, bounded)
 *   - session-start priming digest
 *   - the capture-floor payload for /session-end
 *   - the reversal client (restore / unretract / forget-withdraw), so a
 *     binding that exposes archival can expose its undo on the same path
 *
 * Design contract (shared with the Claude Code hooks — same knobs, same
 * semantics, same env var names):
 *   - Fail-open everywhere. API down, timeout, bad JSON → null, never throw.
 *   - Silence is the common case and must be free: no recall → no message.
 *   - Injected recall is bounded: few results, tight snippets, total cap.
 *
 * THE ONE INVARIANT THIS CORE OWNS (ADR-019 §4, #1002): everything this
 * module produces for injection is a *message body*. There is no function
 * here that yields a system prompt, and no binding may route these strings
 * into one. Model providers cache the prompt as a strict prefix
 * (tools → system → messages); per-turn content in the system prompt
 * invalidates that whole cached prefix every turn and costs more than the
 * recall saves. Bindings append a message after the cached prefix instead,
 * and each binding's test suite pins that.
 */

export interface PalinodeConfig {
  apiUrl: string;
  token?: string;
  /** Named recall profile the channel knobs were derived from. */
  recallProfile: RecallProfileName;
  /** Search hits injected per prompt; 0 disables the search channel. */
  maxResults: number;
  /** Similarity floor for per-turn search. */
  threshold: number;
  /** Trigger channel on/off. */
  triggersOn: boolean;
  /** Prompts shorter than this skip recall entirely. */
  minChars: number;
  /** Total cap on per-turn injected context. */
  maxChars: number;
  /** Per-request timeout in milliseconds. */
  timeoutMs: number;
  /** Route per-turn recall through bounded resolution (`POST /resolve`). */
  resolveOn: boolean;
  /**
   * Per-turn deadline for bounded resolution, in milliseconds. Separate from
   * `timeoutMs` on purpose: this one is spent on EVERY prompt, so it is a
   * latency budget rather than a failure timeout. Past it, the turn falls back
   * to plain search with an explicit marker (see `RESOLUTION_DEADLINE_MARKER`).
   */
  resolveDeadlineMs: number;
  /** Max core memories in the session-start digest; 0 disables priming. */
  coreMaxFiles: number;
  /** Total cap on the session-start digest. */
  coreMaxChars: number;
  /** Minimum user messages before session capture fires. */
  minMessages: number;
}

/**
 * Recall profiles — the same vocabulary the OpenClaw plugin uses, expressed
 * in the hook-shaped knobs above: which channels are on and how much they may
 * inject. A profile is a starting point; explicit env knobs win over it.
 */
export type RecallProfileName =
  | "coding"
  | "monitoring"
  | "investigation"
  | "writing"
  | "conversation"
  | "minimal"
  | "off";

type ProfileKnobs = Pick<PalinodeConfig, "maxResults" | "triggersOn" | "coreMaxFiles">;

export const PROFILES: Record<RecallProfileName, ProfileKnobs> = {
  /** Everything on: priming, triggers, strict search. The default. */
  coding: { maxResults: 3, triggersOn: true, coreMaxFiles: 10 },
  /** Prospective triggers only — for cron/monitor prompts. */
  monitoring: { maxResults: 0, triggersOn: true, coreMaxFiles: 0 },
  /** Search only, wider net — diagnosis sessions. */
  investigation: { maxResults: 8, triggersOn: false, coreMaxFiles: 0 },
  /** Priming only — standing context, no per-turn recall. */
  writing: { maxResults: 0, triggersOn: false, coreMaxFiles: 10 },
  /** Priming + triggers, no search. */
  conversation: { maxResults: 0, triggersOn: true, coreMaxFiles: 10 },
  minimal: { maxResults: 0, triggersOn: false, coreMaxFiles: 0 },
  off: { maxResults: 0, triggersOn: false, coreMaxFiles: 0 },
};

export function isRecallProfileName(value: unknown): value is RecallProfileName {
  return typeof value === "string" && value in PROFILES;
}

/** Per-fired-trigger content cap and max fired triggers injected per prompt.
 *  Mirrors the Claude Code hook's constants. */
const TRIGGER_READ_CHARS = 1200;
const TRIGGER_MAX_FIRED = 2;
/** Per-result snippet cap requested from /search. */
const SNIPPET_MAX_CHARS = 300;

type Env = Record<string, string | undefined>;

function num(env: Env, key: string, fallback: number): number {
  const raw = env[key];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

/** Same env vars as the Claude Code hooks — one set of knobs, every harness.
 *  `overrides` (from a harness's own config surface) win over the env. */
export function configFromEnv(
  env: Env = process.env,
  overrides: Partial<PalinodeConfig> = {},
): PalinodeConfig {
  const profileName = isRecallProfileName(overrides.recallProfile)
    ? overrides.recallProfile
    : isRecallProfileName(env.PALINODE_HOOK_RECALL_PROFILE)
      ? env.PALINODE_HOOK_RECALL_PROFILE
      : "coding";
  const profile = PROFILES[profileName];
  const fromEnv: PalinodeConfig = {
    apiUrl: env.PALINODE_API_URL ?? "http://localhost:6340",
    token: env.PALINODE_API_TOKEN || undefined,
    recallProfile: profileName,
    maxResults: num(env, "PALINODE_HOOK_RECALL_MAX_RESULTS", profile.maxResults),
    // Raw-cosine floor, calibrated in the server's SearchConfig against real
    // bge-m3 (54 pairs): true matches clear 0.5 at 98% but 0.7 at only 28%.
    // An earlier 0.75 default made the search channel silently dead.
    threshold: num(env, "PALINODE_HOOK_RECALL_THRESHOLD", 0.5),
    triggersOn:
      env.PALINODE_HOOK_RECALL_TRIGGERS === undefined || env.PALINODE_HOOK_RECALL_TRIGGERS === ""
        ? profile.triggersOn
        : env.PALINODE_HOOK_RECALL_TRIGGERS !== "0",
    minChars: num(env, "PALINODE_HOOK_RECALL_MIN_CHARS", 12),
    maxChars: num(env, "PALINODE_HOOK_RECALL_MAX_CHARS", 3000),
    timeoutMs: num(env, "PALINODE_HOOK_RECALL_TIMEOUT", 4) * 1000,
    resolveOn:
      env.PALINODE_HOOK_RESOLVE === undefined || env.PALINODE_HOOK_RESOLVE === ""
        ? true
        : env.PALINODE_HOOK_RESOLVE !== "0",
    resolveDeadlineMs: num(env, "PALINODE_HOOK_RESOLVE_DEADLINE", 250),
    coreMaxFiles: num(env, "PALINODE_HOOK_INJECT_MAX_FILES", profile.coreMaxFiles),
    coreMaxChars: num(env, "PALINODE_HOOK_INJECT_MAX_CHARS", 4000),
    minMessages: num(env, "PALINODE_HOOK_MIN_MESSAGES", 3),
  };
  const defined = Object.fromEntries(
    Object.entries(overrides).filter(([, v]) => v !== undefined),
  ) as Partial<PalinodeConfig>;
  return { ...fromEnv, ...defined, recallProfile: profileName };
}

export type FetchFn = typeof fetch;

/** One fail-open request. Any failure — network, HTTP >= 400, bad JSON —
 *  resolves to null. The caller decides what silence means. */
export async function apiJson(
  cfg: PalinodeConfig,
  fetchFn: FetchFn,
  path: string,
  init?: { method?: string; body?: unknown; timeoutMs?: number },
): Promise<unknown | null> {
  try {
    const headers: Record<string, string> = {};
    if (init?.body !== undefined) headers["Content-Type"] = "application/json";
    if (cfg.token) headers["Authorization"] = `Bearer ${cfg.token}`;
    const res = await fetchFn(`${cfg.apiUrl}${path}`, {
      method: init?.method ?? (init?.body !== undefined ? "POST" : "GET"),
      headers,
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
      // `timeoutMs` overrides the config default for one call — what makes a
      // per-turn deadline (250 ms) expressible without shortening the failure
      // timeout every other call depends on.
      signal: AbortSignal.timeout(init?.timeoutMs ?? cfg.timeoutMs),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

interface SearchHit {
  rel_path?: string;
  file_path?: string;
  /** Post-fusion rank value — ~1.0 for any query's top hit. Display fallback only. */
  score?: number;
  /** Raw cosine — the scale the threshold knob filters on. Preferred for display. */
  raw_score?: number | null;
  snippet?: string;
  content?: string;
}

/** Keep aligned with palinode/core/scoring.py; the packages remain independently deployable. */
function describeMatch(result: SearchHit): string {
  const rank = (result.score ?? 0).toFixed(2);
  if (result.raw_score === null) return `keyword match, rank ${rank}`;
  if (typeof result.raw_score === "number") {
    return `${Math.round(result.raw_score * 100)}% match`;
  }
  return `rank ${rank}`;
}

interface FiredTrigger {
  memory_file?: string;
}

/** The qualified bundle `POST /resolve` returns. Fields this adapter reads. */
export interface ResolveBundle {
  /** The rendered bundle — one deterministic template, shared by every surface. */
  text?: string;
  selected?: unknown[];
  conflicts?: unknown[];
  replaced?: unknown[];
  insufficient?: unknown[];
  coverage?: { status?: string; reasons?: string[] };
  omitted_conflicts?: number;
  receipt_ref?: string | null;
}

/**
 * What the turn says when resolution did not come back inside its deadline.
 *
 * The fallback is today's unresolved search hits, and the difference matters:
 * one of them may have been replaced or contradicted by a record nobody
 * checked. Saying so is the whole point — a silent fallback would present a
 * stale assertion with the authority of a resolved one.
 */
/** The fixed frame every per-turn injection carries. Its size is part of the
 *  budget: what is left after it is what the bundle may spend. */
const PREAMBLE = `## Palinode recall (this prompt)

Retrieved from persistent memory; may be stale — verify before relying on
it. More detail: palinode_search / palinode_read.
`;

/** Below this much remaining room there is no honest answer to give: a bundle
 *  cannot fit its frame plus the notice naming a contested group, and falling
 *  back to raw hits would show one side of a conflict as a plain search
 *  result. The channel says nothing instead. */
const RESOLVE_MIN_CHARS = 300;

export const RESOLUTION_DEADLINE_MARKER =
  "_resolution unavailable (deadline) — the memories below are unresolved search " +
  "hits: a replacement or an open conflict may exist that was not checked. " +
  "Call palinode_resolve before relying on one._";

/** The marker a contested row carries in every payload the server renders —
 *  the digest's qualifier and the bundle's contested section both use it. A
 *  trim that would cut through one drops the whole block instead. */
const CONTESTED_MARKERS = ["⚠ contradicts", "Contested (", "Still contested"];

/** What the server says when a conflict did not fit; the same wording the
 *  Python packer emits (`palinode/core/packing.py::_contested_stub`), so a
 *  reader sees one sentence whichever side had to withhold the group. */
function contestedStub(blocks: string[]): string {
  const refs: string[] = [];
  for (const block of blocks) {
    for (const m of block.matchAll(/\[([^\]\s]+?\.md|[^\]\s]+?\/[^\]\s]+?)\]/g)) {
      const key = m[1].endsWith(".md") ? m[1].slice(0, -3) : m[1];
      if (!refs.some((r) => (r.endsWith(".md") ? r.slice(0, -3) : r) === key)) refs.push(m[1]);
    }
  }
  const where = refs.length ? refs.join(", ") : "no source pointers recorded";
  const plural = blocks.length === 1 ? "conflict" : "conflicts";
  return `⚠ ${blocks.length} ${plural} omitted for budget — see ${where}`;
}

/**
 * Trim `text` to `maxChars` **at a unit boundary** — never inside one.
 *
 * THE RULE THIS EXISTS FOR: a final character slice is how a payload the
 * server packed honestly arrives dishonest. `text.slice(0, cap)` can cut a
 * qualifier off a row ("— contradicts insights/b" gone, the claim now reads
 * settled) or cut a conflict block after its first side, and a contested
 * claim that loses its counterpart reads as settled. So the cut lands on a
 * line boundary, a unit that does not fit is dropped whole rather than
 * halved, and if any dropped line was a contested one, the stub the server
 * would have emitted is appended in its place — evicting further kept lines
 * to make room, because "there is a conflict here, here is where" outranks
 * one more ordinary row.
 *
 * A block is a non-indented line plus the indented lines under it (the
 * bundle renders a row's qualifiers and reasons as indented continuations),
 * so a row never loses its own qualification.
 *
 * Unlike the server-side packer this stops at the first block that does not
 * fit rather than continuing past it: the packer is choosing units for a
 * payload it is about to render, while this is trimming a rendered string, and
 * a hole in the middle of one reads worse than a clean tail cut.
 */
export function trimToUnitBoundary(text: string, maxChars: number): string {
  if (maxChars <= 0) return "";
  if (text.length <= maxChars) return text;

  const lines = text.split("\n");
  const blocks: string[][] = [];
  for (const line of lines) {
    if (blocks.length > 0 && /^\s/.test(line) && line.trim() !== "") {
      blocks[blocks.length - 1].push(line);
    } else {
      blocks.push([line]);
    }
  }

  const kept: string[] = [];
  const dropped: string[] = [];
  let used = 0;
  const cost = (block: string[]) => block.join("\n").length + (kept.length ? 1 : 0);
  for (const block of blocks) {
    if (dropped.length === 0 && used + cost(block) <= maxChars) {
      used += cost(block);
      kept.push(block.join("\n"));
    } else {
      dropped.push(block.join("\n"));
    }
  }

  const isContested = (block: string) => CONTESTED_MARKERS.some((m) => block.includes(m));
  const contested = dropped.filter(isContested);
  if (contested.length > 0) {
    for (;;) {
      const stub = contestedStub(contested);
      if (used + stub.length + (kept.length ? 1 : 0) <= maxChars) {
        kept.push(stub);
        break;
      }
      // Below the room for even the notice, nothing is rendered: a fragment
      // that fit by dropping it would read as settled.
      if (kept.length === 0) return "";
      const evicted = kept.pop() as string;
      used -= evicted.length + (kept.length ? 1 : 0);
      // An evicted conflict joins the ones the stub names — the count and the
      // pointers always describe every conflict that is actually missing.
      if (isContested(evicted)) contested.push(evicted);
    }
  }
  return kept.join("\n");
}

/**
 * Bounded resolution for one prompt, under the per-turn deadline.
 *
 * Null on anything that is not a usable bundle — deadline, API down, bad
 * JSON — so the caller can fall back and say that it did.
 */
export async function resolveBundle(
  prompt: string,
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
  maxChars: number = cfg.maxChars - PREAMBLE.length,
): Promise<ResolveBundle | null> {
  const bundle = (await apiJson(cfg, fetchFn, "/resolve", {
    body: {
      query: prompt,
      max_items: Math.max(cfg.maxResults, 1),
      max_chars: maxChars,
    },
    timeoutMs: cfg.resolveDeadlineMs,
  })) as ResolveBundle | null;
  if (!bundle || typeof bundle.text !== "string") return null;
  return bundle;
}

/** Does this bundle say anything? An empty one is silence, not an answer.
 *
 *  `omitted_conflicts` counts: a conflict the server's budget dropped is the
 *  one thing that is emphatically NOT nothing — the bundle carries it by ref
 *  precisely so it cannot be mistaken for a settled question. */
function bundleIsEmpty(bundle: ResolveBundle): boolean {
  const count = (xs?: unknown[]) => (Array.isArray(xs) ? xs.length : 0);
  return (
    count(bundle.selected) +
      count(bundle.conflicts) +
      count(bundle.replaced) +
      count(bundle.insufficient) +
      (typeof bundle.omitted_conflicts === "number" ? bundle.omitted_conflicts : 0) ===
    0
  );
}

/**
 * Per-turn recall: prospective triggers + one memory channel.
 *
 * ROUTING (the contract this adapter owns; see docs/HOW-MEMORY-WORKS.md):
 *
 *   - **Session start** → ordinary priming (`buildCoreDigest`): `/context/prime`
 *     plus the core digest, under `timeoutMs`. No resolution: a startup digest
 *     is orientation, not an answer, and its payload stays separate from this
 *     one.
 *   - **Per turn** (here) → bounded resolution (`POST /resolve`) under
 *     `resolveDeadlineMs` (250 ms). Past the deadline the turn falls back to
 *     today's strict-threshold search, prefixed with
 *     `RESOLUTION_DEADLINE_MARKER` — never silently.
 *   - **Explicit follow-up** → `palinode_search` / `palinode_read` /
 *     `palinode_resolve` as tools. Agent-initiated, no deadline, unbounded by
 *     this budget. Injection is a starting point; these are the way to the
 *     rest.
 *
 * Returns the context block to inject, or null when there is nothing to say.
 */
export async function buildRecallContext(
  prompt: string,
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
): Promise<string | null> {
  if (prompt.length < cfg.minChars) return null;

  // The two channels are independent; run them concurrently (the hosts'
  // hook budgets are tight — Cline's sandbox allows 3 s for the whole hook).
  // Output order stays fixed: triggers first, then search.
  const triggerSection = async (): Promise<string> => {
    if (!cfg.triggersOn) return "";
    const fired = await apiJson(cfg, fetchFn, "/check-triggers", {
      body: { query: prompt },
    });
    if (!Array.isArray(fired)) return "";
    let out = "";
    for (const t of (fired as FiredTrigger[]).slice(0, TRIGGER_MAX_FIRED)) {
      if (!t.memory_file) continue;
      const read = (await apiJson(
        cfg,
        fetchFn,
        `/read?file_path=${encodeURIComponent(t.memory_file)}`,
      )) as { content?: string } | null;
      const body = read?.content;
      if (body) {
        out += `\n### Trigger fired: ${t.memory_file}\n${body.slice(0, TRIGGER_READ_CHARS)}\n`;
      }
    }
    return out;
  };

  const searchSection = async (): Promise<string> => {
    if (cfg.maxResults <= 0) return "";
    const hits = (await apiJson(cfg, fetchFn, "/search", {
      body: {
        query: prompt,
        limit: cfg.maxResults,
        threshold: cfg.threshold,
        max_chars: SNIPPET_MAX_CHARS,
      },
    })) as { results?: SearchHit[] } | SearchHit[] | null;
    const results = Array.isArray(hits) ? hits : hits?.results;
    if (!Array.isArray(results) || results.length === 0) return "";
    const lines = results
      .map((r) => {
        const path = r.rel_path ?? r.file_path ?? "?";
        const body = (r.snippet ?? r.content ?? "").replace(/\n/g, " ");
        return `- [${path}] (${describeMatch(r)}) ${body}`;
      })
      .join("\n");
    return `\n### Related memories\n${lines}\n`;
  };

  const memorySection = async (): Promise<string> => {
    if (cfg.maxResults <= 0) return "";
    if (!cfg.resolveOn) return searchSection();
    const room = cfg.maxChars - PREAMBLE.length;
    if (room < RESOLVE_MIN_CHARS) return "";
    const bundle = await resolveBundle(prompt, cfg, fetchFn, room);
    if (bundle) return bundleIsEmpty(bundle) ? "" : `\n${bundle.text}\n`;
    const fallback = await searchSection();
    // Nothing recalled is nothing to mislead about: silence stays free, and
    // the marker appears only where unresolved hits actually do.
    // `fallback` already opens with its own newline, so the marker slots in
    // ahead of a section that is otherwise byte-identical to today's.
    return fallback ? `\n${RESOLUTION_DEADLINE_MARKER}${fallback}` : "";
  };

  const [triggers, memory] = await Promise.all([triggerSection(), memorySection()]);
  if (!triggers && !memory) return null;

  // The memory section is already sized to fit (the server packed it against
  // the room passed in `max_chars`), so the total is brought under the cap by
  // trimming the trigger section — never by slicing the bundle, which is how
  // a conflict the server kept whole would arrive with one side missing.
  //
  // The trigger trim cuts at a unit boundary and there is deliberately NO
  // final `context.slice(0, cfg.maxChars)`: the only thing such a slice could
  // still reach is the bundle, and when the bundle overruns it is because the
  // server refused to buy room by dropping its own omitted-conflict notice.
  // Slicing that off here would undo exactly the honesty it paid for.
  const room = cfg.maxChars - PREAMBLE.length - memory.length;
  return `${PREAMBLE}${room > 0 ? trimToUnitBoundary(triggers, room) : ""}${memory}`;
}

interface CoreListEntry {
  file?: string;
  name?: string;
  summary?: string;
}

/**
 * Session-start priming: warm server-side session context, then return a
 * bounded digest of `core: true` memories — or null when there are none.
 *
 * Deliberately NOT routed through bounded resolution (see the routing contract
 * on `buildRecallContext`): a startup digest is orientation — "here is what
 * this project keeps" — not an answer to a question, and there is no question
 * yet to resolve. It keeps its own payload and its own budget (`coreMaxChars`,
 * `timeoutMs`), so a per-turn deadline can never shrink the standing context a
 * session opens with, and priming can never spend the per-turn budget.
 */
export async function buildCoreDigest(
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
  cwd: string = process.cwd(),
  sessionId = "",
): Promise<string | null> {
  // Warm /context/prime; result deliberately ignored (older servers 404).
  await apiJson(cfg, fetchFn, "/context/prime", {
    body: { cwd, session_id: sessionId },
  });

  if (cfg.coreMaxFiles <= 0) return null;

  const listing = await apiJson(cfg, fetchFn, "/list?core_only=true");
  if (!Array.isArray(listing) || listing.length === 0) return null;

  const lines = (listing as CoreListEntry[])
    .slice(0, cfg.coreMaxFiles)
    .map((e) => {
      const name = e.name ?? "untitled";
      const summary = e.summary ? ` — ${e.summary}` : "";
      return `- [${e.file ?? "?"}] ${name}${summary}`;
    })
    .join("\n");

  const context = `## Palinode memory (session start)

Persistent memory is connected. Recall details with the palinode_search /
palinode_read tools — they read the live store; session notes are NOT
files in this repo.

Core memories:
${lines}`;

  // One row per line, so the cap lands between rows. A plain slice here could
  // take a row's `⚠ contradicts:` qualifier off the end of the digest and
  // leave a contested memory reading as standing context.
  return trimToUnitBoundary(context, cfg.coreMaxChars);
}

/**
 * The subset of a harness message the capture floor reads. Covers Pi session
 * entries (`{ type, message: { role, content } }`) and Cline agent messages
 * (`{ role, content: [{ type: "text", text }] }`) without importing either.
 */
export interface SessionEntryLike {
  type?: string;
  role?: string;
  content?: unknown;
  message?: { role?: string; content?: unknown };
}

function entryRole(e: SessionEntryLike): string | undefined {
  return e.role ?? e.message?.role ?? e.type;
}

function contentText(c: unknown): string {
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c
      .map((b) => (typeof b === "object" && b && "text" in b ? String((b as { text: unknown }).text) : ""))
      .join(" ");
  }
  return "";
}

/** Text of an entry's user-facing content, or "" when it has none. */
export function entryText(e: SessionEntryLike): string {
  return contentText(e.message?.content ?? e.content);
}

/** Entries a human typed — what the capture floor counts. */
export function userEntries(entries: SessionEntryLike[]): SessionEntryLike[] {
  return entries.filter((e) => entryRole(e) === "user" && entryText(e).trim() !== "");
}

export interface CaptureOrigin {
  project: string;
  /** Recorded as the memory's `source` (e.g. "pi-extension", "cline-plugin"). */
  source: string;
  /** Harness label for the /session-end metadata footer. */
  harness: string;
  /** Which lifecycle event fired the capture (e.g. "session_shutdown", "run_end"). */
  trigger: string;
  sessionId?: string;
  cwd?: string;
}

export interface SessionCapturePayload {
  summary: string;
  project: string;
  source: string;
  harness: string;
  trigger: string;
  decisions: string[];
  blockers: string[];
  session_id?: string;
  cwd?: string;
}

/**
 * Session capture floor: derive the minimal /session-end payload from the
 * session entries. Returns null when the session is too trivial to keep
 * (fewer than `minMessages` user messages) — same gate as the Claude Code
 * floor hook.
 */
export function buildSessionCapture(
  entries: SessionEntryLike[],
  cfg: PalinodeConfig,
  origin: CaptureOrigin,
): SessionCapturePayload | null {
  const users = userEntries(entries);
  if (users.length < cfg.minMessages) return null;

  const squash = (s: string) => s.replace(/\s+/g, " ").trim().slice(0, 200);
  const firstPrompt = squash(entryText(users[0]));
  const lastPrompt = squash(entryText(users[users.length - 1]));
  const latest = users.length > 1 && lastPrompt !== firstPrompt ? ` Latest: ${lastPrompt}` : "";

  return {
    summary: `Auto-captured (${origin.harness} ${origin.trigger}, ${users.length} messages). Topic: ${firstPrompt}${latest}`,
    project: origin.project,
    source: origin.source,
    harness: origin.harness,
    trigger: origin.trigger,
    decisions: [],
    blockers: [],
    ...(origin.sessionId ? { session_id: origin.sessionId } : {}),
    ...(origin.cwd ? { cwd: origin.cwd } : {}),
  };
}

/** POST the capture; fail-open. Returns true when the API accepted it. */
export async function postSessionCapture(
  payload: SessionCapturePayload,
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
): Promise<boolean> {
  const res = await apiJson(cfg, fetchFn, "/session-end", { body: payload });
  return res !== null;
}

// ---------------------------------------------------------------------------
// Reversal client — the inverse of archival and retraction, over the REST
// API. Plain fail-open calls: the result object on 2xx, null otherwise. Each
// maps 1:1 to the registered operation in palinode/core/parity.py so a
// binding exposing them inherits the canonical parameter names.
// ---------------------------------------------------------------------------

export interface RestoreResult {
  file: string;
  /** "active" when restored; "not_archived" when there was nothing to undo. */
  status: string;
  restored_from?: string | null;
  restored_at?: string;
  history_file?: string | null;
  chunks_updated?: number;
  /** Present when the memory still carries a TTL the sweep will act on. */
  expires_at?: string;
}

/** Bring one archived memory back into default recall (inverse of /archive). */
export async function restoreMemory(
  filePath: string,
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
  reason?: string,
): Promise<RestoreResult | null> {
  const body: Record<string, unknown> = { file_path: filePath };
  if (reason !== undefined) body.reason = reason;
  const res = await apiJson(cfg, fetchFn, "/restore", { body });
  return (res as RestoreResult | null) ?? null;
}

export interface UnretractResult {
  file: string;
  /** "unretracted", or "not_retracted" when the pref was not on record. */
  status: string;
  mentions: number;
  retraction_id?: string;
  history_file?: string | null;
  index_error?: string;
}

/** Withdraw one pref's mention-level retraction from one memory. */
export async function unretractMentions(
  filePath: string,
  pref: string,
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
  reason?: string,
): Promise<UnretractResult | null> {
  const body: Record<string, unknown> = { file_path: filePath, pref };
  if (reason !== undefined) body.reason = reason;
  const res = await apiJson(cfg, fetchFn, "/unretract", { body });
  return (res as UnretractResult | null) ?? null;
}

export interface ForgetWithdrawResult {
  file: string;
  status: string;
  pref: string;
  restored: string[];
  unretracted: Array<{ path: string; mentions: number }>;
  requests_archived: string[];
  failed?: Array<{ path: string; op: string }>;
}

/** Take a forget request back: restore + unretract its targets, archive the record. */
export async function withdrawForgetRequest(
  filePath: string,
  cfg: PalinodeConfig,
  fetchFn: FetchFn = fetch,
  reason?: string,
): Promise<ForgetWithdrawResult | null> {
  const body: Record<string, unknown> = { file_path: filePath };
  if (reason !== undefined) body.reason = reason;
  const res = await apiJson(cfg, fetchFn, "/forget-withdraw", { body });
  return (res as ForgetWithdrawResult | null) ?? null;
}
