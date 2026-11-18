/*
 * The Particles curation pane for Claude Code (to §4): a
 * renderer of the shared curation core, drawn with Claude Code's elements.
 *
 * The core decides what each card shows and offers, the forms and requests
 * behind each action, and every string. This module draws a `CardView`,
 * shows each `Form` in place of the card's actions, talks to the configured
 * engine through `$.http.fetch`, and raises a contested belief's card when
 * the agent's Particles `query` uses it. It holds no curation logic.
 *
 * A press starts its action and returns at once: an action may wait on a
 * form for as long as the operator takes, which no hook may.
 */
import type { EngineInterface, Register } from "claude-code";
import {
  type ActionView,
  type Answer,
  type BriefView,
  type CardView,
  type ChoiceOption,
  type CurationCard,
  type Form,
  type RefBlock,
  ApiError,
  CORE_VERSION,
  CurationClient,
  HELP_SUMMARY,
  KIND_FILTER,
  PAGE_NEXT_LABEL,
  PAGE_PREVIOUS_LABEL,
  REFRESH_LABEL,
  SEMANTIC_LABEL,
  SEMANTIC_SKIPPED_TEXT,
  SUBJECT_LABEL,
  SUBJECT_SEARCH_MIN,
  SUBJECT_NO_MATCH,
  SUBJECT_SUGGEST_FAILED,
  cardView,
  emptyText,
  errorMessage,
  failureOf,
  kindFilterLabel,
  afterMemberAction,
  memberCard,
  openCountOf,
  perform,
  queueHeader,
  readOnlyText,
  searchSubjects,
  subjectListing,
  suggestSubjects,
} from "../../curation-core/src/index.ts";
import { type Connection, connectionOf, isConnection, linkBase, startCommand } from "./connection.ts";
import { type ContestedHit, contestedHits } from "./contested.ts";

type $ = EngineInterface;

const PANE = "particles";
const COMMAND = "particles";
/** The engine routes the core calls; an engine that lacks one is older than this pane. */
const ROUTES = [
  "/curation",
  "/curation/cards/{key}",
  "/curation/affirm",
  "/curation/snooze",
  "/review/{particle_id}",
  "/particles/{particle_id}/supersede",
  "/particles/{particle_id}/retract",
  "/particles/{particle_id}/subjects",
  "/subjects/search",
  "/particles/{particle_id}/subject-suggestions",
];
/** The span the per-turn bounds count over: a turn is not visible to a tool hook. */
const TURN_WINDOW_MS = 60_000;
const LOOKUPS_PER_WINDOW = 3;

interface Options {
  engineUrl?: string;
  engineToken?: string;
  mcpServer?: string;
  reviewerId?: string;
}

interface Notice {
  level: "info" | "error";
  text: string;
}

interface PendingForm {
  form: Form;
  values: Record<string, string>;
  /** A `subject` field's matches for what was last typed; absent before a search. */
  matches: Record<string, ChoiceOption[] | undefined>;
  /**
   * A `subject` field's suggestions for its belief: absent
   * until they arrive, null when the read failed (an older engine has no
   * such route).
   */
  suggested: Record<string, ChoiceOption[] | null | undefined>;
  /** The subject picked in a `subject` field, while its text still shows its name. */
  picked: Record<string, { id: string; label: string } | null>;
  /** Counts searches, so an answer to an older one is dropped. */
  searches: number;
  resolve: (answer: Answer) => void;
  /**
   * Where the form is drawn: the key of the card (or session item) it belongs
   * to, or `member:<particle id>` for one belief of a source card, whose own
   * box then holds it, so the operator sees which belief it is about.
   */
  owner: string;
}

interface SessionItem {
  hit: ContestedHit;
  card?: CurationCard;
}

// The pane's state. A module reload or a new session starts it over, which
// costs one fetch.
const fresh = () => ({
  phase: "idle" as "idle" | "loading" | "ready" | "setup" | "error",
  message: "",
  cards: [] as CurationCard[],
  openCount: 0,
  semanticSkipped: false,
  /** The last queue answer's snapshot time; a card lookup needs one (§4). */
  builtAt: null as string | null,
  kind: "",
  semantic: false,
  readOnly: false,
  readOnlyReason: "",
  notices: [] as Notice[],
  form: null as PendingForm | null,
  busy: false,
  helpOpen: new Set<string>(),
  /** Per card key: the beliefs assigned a subject here, kept on the card, marked. */
  assigned: new Map<string, Record<string, string>>(),
  /** Per card key: the first belief of the page drawn (§ paging, below). */
  pages: new Map<string, number>(),
  session: [] as SessionItem[],
  seen: new Set<string>(),
  windowStart: 0,
  lookups: 0,
  noticed: false,
  engineVersion: "",
  missing: [] as string[],
  pendingOpen: false,
});
let S = fresh();

function client($: $, conn: Connection): CurationClient {
  return new CurationClient({
    transport: async (req) => {
      const headers: Record<string, string> = { Accept: "application/json" };
      if (conn.token) headers.Authorization = `Bearer ${conn.token}`;
      if (req.body !== undefined) headers["Content-Type"] = "application/json";
      const init: { method: string; headers: Record<string, string>; body?: string } = {
        method: req.method,
        headers,
      };
      if (req.body !== undefined) init.body = JSON.stringify(req.body);
      const r = await $.http.fetch(conn.base + req.path, init);
      let body: unknown;
      try {
        body = r.text ? JSON.parse(r.text) : undefined;
      } catch {
        body = undefined;
      }
      return { status: r.status, body };
    },
    facts: () => ({ hasToken: !!conn.token, baseUrl: conn.base }),
    depositedBy: "claude-code",
  });
}

function note(level: Notice["level"], text: string): void {
  S.notices = [{ level, text }, ...S.notices].slice(0, 3);
}

function redraw($: $): void {
  $.ui.invalidate("ui.render");
}

async function openPane($: $): Promise<boolean> {
  const opened = await $.ui.open({ id: PANE, title: "Particles" });
  if (opened.isPlaced) {
    S.pendingOpen = false;
    return true;
  }
  if (String(opened.reason).includes("no-surface")) S.pendingOpen = true;
  return false;
}

/** The plugin's options, read once per load (an options change reloads the module). */
const CFG = {
  conn: null as Connection | { reason: string } | null,
  reviewerId: "claude-code",
  server: "particles",
};

// --- Loading -------------------------------------------------------------

/** First load: decide whether a call may be made at all, then fetch. */
async function start($: $): Promise<void> {
  if (!isConnection(CFG.conn)) {
    S.phase = "setup";
    S.message = (CFG.conn as { reason?: string } | null)?.reason ?? "Set the engine URL in the plugin's options.";
    redraw($);
    return;
  }
  const conn = CFG.conn as Connection;
  if (!conn.token) {
    if (!conn.loopback) {
      S.phase = "setup";
      S.message = "Set the engine key in the plugin's options to curate a remote engine.";
      redraw($);
      return;
    }
    // An engine on this machine is asked once, without a key, whether it
    // needs one: a keyless engine serves reads.
    S.phase = "loading";
    redraw($);
    try {
      await client($, CFG.conn as Connection).request("GET", "/events?limit=1");
    } catch (e) {
      S.phase = e instanceof ApiError && e.kind === "unauthorized" ? "setup" : "error";
      S.message =
        S.phase === "setup"
          ? "This engine needs a key. Mint one with `particles engine keys` and set it in the plugin's options."
          : unreachable(e, conn);
      redraw($);
      return;
    }
  }
  await load($);
  void versions($);
}

async function load($: $): Promise<void> {
  if (!isConnection(CFG.conn)) return;
  S.phase = "loading";
  redraw($);
  try {
    const resp = await client($, CFG.conn as Connection).curation({
      ...(S.kind ? { kind: S.kind } : {}),
      ...(S.semantic ? { semantic: true } : {}),
    });
    S.cards = resp.cards ?? [];
    S.openCount = openCountOf(resp.open_count, S.cards.length);
    S.semanticSkipped = resp.semantic_skipped ?? false;
    S.builtAt = resp.built_at ?? null;
    S.phase = "ready";
  } catch (e) {
    S.phase = "error";
    S.message = unreachable(e, CFG.conn as Connection);
  }
  redraw($);
}

function unreachable(e: unknown, c: Connection): string {
  const text = errorMessage(e);
  return e instanceof ApiError && e.kind === "network" ? `${text} Start one with \`${startCommand(c.base)}\`.` : text;
}

/** The engine's version, and the routes it lacks, for the footer. */
async function versions($: $): Promise<void> {
  if (!isConnection(CFG.conn)) return;
  const c = client($, CFG.conn as Connection);
  try {
    S.engineVersion = (await c.health()).version;
    const spec = await c.request<{ paths?: Record<string, unknown> }>("GET", "/openapi.json");
    const paths = spec?.paths ?? {};
    S.missing = ROUTES.filter((r) => !(r in paths));
  } catch {
    /* the footer shows what it has */
  }
  redraw($);
}

// --- Actions -------------------------------------------------------------

function ask($: $, owner: string): (form: Form) => Promise<Answer> {
  return (form) =>
    new Promise((resolve) => {
      const values: Record<string, string> = {};
      for (const f of form.fields ?? []) values[f.name] = f.type === "choice" ? "" : (f.value ?? "");
      const p: PendingForm = { form, values, matches: {}, suggested: {}, picked: {}, searches: 0, resolve, owner };
      S.form = p;
      redraw($);
      // The form draws at once; its suggestions fill in when they arrive.
      for (const f of form.fields ?? []) {
        if (f.type === "subject" && f.particleId) void suggestField($, p, f.name, f.particleId);
      }
    });
}

/** Fetch a `subject` field's suggestions for its belief. */
async function suggestField($: $, p: PendingForm, name: string, particleId: string): Promise<void> {
  if (!isConnection(CFG.conn)) return;
  let found: ChoiceOption[] | null;
  try {
    found = await suggestSubjects(client($, CFG.conn), particleId);
  } catch {
    found = null;
  }
  if (S.form !== p) return;
  p.suggested[name] = found;
  redraw($);
}

/** Search subjects for a `subject` field as the operator types. */
async function searchField($: $, p: PendingForm, name: string, text: string): Promise<void> {
  if (!isConnection(CFG.conn)) return;
  const mine = ++p.searches;
  let found: ChoiceOption[];
  try {
    found = await searchSubjects(client($, CFG.conn), text);
  } catch {
    found = [];
  }
  if (mine !== p.searches || S.form !== p) return;
  // Below the minimum there is no search: the suggestions show instead.
  p.matches[name] = text.trim().length >= SUBJECT_SEARCH_MIN ? found : undefined;
  redraw($);
}

/** A form's answer: a `subject` field submits its picked subject's id. */
function answerOf(p: PendingForm): Record<string, string> {
  const out = { ...p.values };
  for (const [name, pick] of Object.entries(p.picked)) {
    // The text the field showed, so the card can name what was assigned.
    out[SUBJECT_LABEL(name)] = out[name] ?? "";
    if (pick && out[name] === pick.label) out[name] = pick.id;
  }
  return out;
}

/** Run one gesture; `member` is the belief of a source card it acts on. */
async function act($: $, gesture: string, card: CurationCard, member?: BriefView, sessionItem?: SessionItem): Promise<void> {
  if (!isConnection(CFG.conn) || S.busy) return;
  S.busy = true;
  redraw($);
  const brief = member ? (card.particles ?? []).find((p) => p.particle_id === member.particleId) : undefined;
  const target = brief ? memberCard(card, brief) : card;
  try {
    const result = await perform(gesture, target, {
      client: client($, CFG.conn as Connection),
      reviewerId: CFG.reviewerId,
      ask: ask($, brief ? `member:${brief.particle_id}` : card.key),
    });
    if (result.notice) note("info", result.notice);
    if (result.resolved) {
      if (sessionItem) {
        S.session = S.session.filter((s) => s !== sessionItem);
      } else if (brief) {
        // An assigned belief stays on the card, marked; any other resolution
        // removes it, and the card goes once none is left.
        const after = afterMemberAction(card, brief.particle_id, result, S.assigned.get(card.key) ?? {});
        card.particle_ids = after.card.particle_ids;
        card.particles = after.card.particles;
        S.assigned.set(card.key, after.assigned);
        if (after.empty) advance(true);
      } else {
        advance(true);
      }
    }
  } catch (e) {
    const failure = failureOf(e);
    if (failure.kind === "read-only" && !brief) {
      S.readOnly = true;
      S.readOnlyReason = failure.reason;
    } else {
      note("error", errorMessage(e));
    }
  } finally {
    S.busy = false;
    S.form = null;
    redraw($);
  }
}

function advance(resolved: boolean): void {
  const left = S.cards.shift();
  if (left) {
    S.assigned.delete(left.key);
    S.pages.delete(left.key);
  }
  if (resolved) S.openCount = Math.max(S.openCount - 1, S.cards.length);
}

async function showAll($: $, card: CurationCard): Promise<void> {
  if (!isConnection(CFG.conn)) return;
  try {
    const full = await client($, CFG.conn as Connection).curationCard(card.key);
    card.particle_ids = full.particle_ids;
    card.particles = full.particles;
    S.pages.set(card.key, 0);
  } catch (e) {
    note("error", errorMessage(e));
  }
  redraw($);
}

// A source card can name hundreds of beliefs, and the pane refuses a tree past
// 100,000 characters serialized: drawn all at once, Show all drew nothing (the
// owner's live run, 2026-11-08). Twenty beliefs stay well inside it.
const PAGE_SIZE = 20;

function turnPage($: $, card: CurationCard, start: number): void {
  S.pages.set(card.key, Math.max(0, start));
  redraw($);
}

// --- Contested beliefs (§4) -----------------------------------------------

async function raise($: $, hits: ContestedHit[]): Promise<void> {
  if (!isConnection(CFG.conn)) return;
  const now = Date.now();
  if (now - S.windowStart > TURN_WINDOW_MS) {
    S.windowStart = now;
    S.lookups = 0;
    S.noticed = false;
  }
  let added = false;
  for (const hit of hits) {
    if (S.seen.has(hit.inconsistencyId)) continue;
    S.seen.add(hit.inconsistencyId);
    const item: SessionItem = { hit };
    // A lookup with no snapshot can make the engine collect the queue with
    // the model, so it is made only when the last queue answer had one.
    if (S.builtAt && S.lookups < LOOKUPS_PER_WINDOW) {
      S.lookups += 1;
      try {
        item.card = await client($, CFG.conn as Connection).curationCard(`inconsistency:${hit.inconsistencyId}`);
      } catch {
        /* not in the snapshot yet: listed with a link instead */
      }
    }
    S.session.unshift(item);
    added = true;
  }
  if (!added) return;
  S.session = S.session.slice(0, 5);
  redraw($);
  if (!S.noticed) {
    S.noticed = true;
    try {
      const panes = await $.ui.panes();
      const shown = panes.some((p) => p.id === PANE && p.isShown);
      if (!shown && !(await openPane($))) {
        $.ui.toast("Particles: a belief the agent just used is contested. Run /particles to see it.");
      }
    } catch {
      /* a notice that cannot be shown is dropped; the pane lists the hit */
    }
  }
}

export const register: Register = (on, rawOptions) => {
  const options = (rawOptions ?? {}) as Options;
  CFG.conn = connectionOf(options.engineUrl, options.engineToken);
  CFG.reviewerId = (options.reviewerId ?? "").trim() || "claude-code";
  CFG.server = (options.mcpServer ?? "").trim() || "particles";
  const conn = CFG.conn;
  const server = CFG.server;

  // --- Hooks ---------------------------------------------------------------

  on("session.start", async ($, e, next) => {
    S = fresh();
    await $.command.register({ name: COMMAND, description: "Open the Particles curation pane" });
    return next(e);
  });

  on("session.attach", async ($, e, next) => {
    const result = await next(e);
    // An open that found no surface (a reload before the app attached) is
    // made again now that one is here.
    if (S.pendingOpen) await openPane($);
    return result;
  });

  on("command.run", { command: COMMAND }, async ($) => {
    const placed = await openPane($);
    if (S.phase === "idle" || S.phase === "error") void start($);
    return {
      text: placed ? "Particles pane opened." : "The Particles pane will open when the app has a surface to draw it on.",
    };
  }).catch(() => ({ text: "The Particles pane could not open." }));

  // A RegExp, since the server's name is an option: the tool union names only
  // the MCP tools connected when the types were written.
  const queryTool = new RegExp(`^mcp__${server.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}__query$`);
  on("tool.call", { tool: queryTool }, async ($, e, next) => {
    const ran = await next(e);
    try {
      const hits = contestedHits(ran);
      if (hits.length) void raise($, hits);
    } catch {
      /* observation only: never fail the agent's call */
    }
    return ran;
  }).catch(($, e, next) => next(e)); // fail open: the agent's call always goes on

  on("ui.render", { component: "Pane", requestId: PANE }, async ($, e) => {
    const els = $.ui.resolve(e) as Record<string, any>;
    const { Box, Text, Button, Link } = els;
    const webUi = isConnection(conn) ? `${linkBase(conn)}/app#/curate` : "";
    if (!("Input" in els) || !("Select" in els)) {
      return (
        <Box flexDirection="column" gap="sm">
          <Text dimColor>This surface draws no text fields, so curate in the web UI.</Text>
          {webUi ? <Link href={webUi} label="Open the web UI ↗" /> : null}
        </Box>
      );
    }
    const { Input, Select, Markdown } = els;
    if (S.phase === "idle") void start($);

    const link = (route: string, label: string) =>
      isConnection(conn) ? <Link href={`${linkBase(conn)}/app#/${route}`} label={label} /> : <Text>{label}</Text>;

    // --- A form, drawn in place of the actions of the card it belongs to.
    const drawForm = (p: PendingForm) => {
      const fields = p.form.fields ?? [];
      // One text field submits on Enter. The form's own confirm button is its
      // only button: an Input's submit button beside it read as a second one.
      const single = fields.length === 1 && fields[0].type !== "choice" && fields[0].type !== "choices";
      const matchesOf = (name: string) => {
        const text = p.values[name] ?? "";
        const suggested = p.suggested[name];
        const listing = subjectListing(text, suggested ?? undefined, p.matches[name]);
        if (listing === null) {
          // Below the search minimum with no suggestions: say so only when
          // the read failed.
          return suggested === null && text.trim().length < SUBJECT_SEARCH_MIN ? (
            <Text key={`nosuggest:${name}`} size="caption" dimColor>
              {SUBJECT_SUGGEST_FAILED}
            </Text>
          ) : null;
        }
        const found = listing.options;
        if (found.length === 0) {
          return (
            <Text key={`nomatch:${name}`} size="caption" dimColor>
              {SUBJECT_NO_MATCH}
            </Text>
          );
        }
        return (
          <Box key={`matches:${name}`} flexDirection="column" gap="xs">
            {listing.heading ? (
              <Text key={`heading:${name}`} size="caption" weight="semibold">
                {listing.heading}
              </Text>
            ) : null}
            {found.map((o) => (
              <Box key={`match:${name}:${o.value}`} flexDirection="column">
                <Button
                  key={`pick:${name}:${o.value}`}
                  label={`${p.picked[name]?.id === o.value ? "●" : "○"} ${o.label}`}
                  plain
                  onPress={() => {
                    p.picked[name] = { id: o.value, label: o.label };
                    p.values[name] = o.label;
                    // A picked suggestion's name is past the search minimum:
                    // keep its list shown, with the pick marked.
                    if (listing.heading) {
                      p.searches += 1;
                      p.matches[name] = found;
                    }
                    redraw($);
                  }}
                />
                {o.detail ? (
                  <Text size="caption" dimColor>
                    {o.detail}
                  </Text>
                ) : null}
              </Box>
            ))}
          </Box>
        );
      };
      const submit = () => {
        S.form = null;
        p.resolve(answerOf(p));
        redraw($);
      };
      const cancel = () => {
        S.form = null;
        p.resolve(null);
        redraw($);
      };
      return (
        <Box key="form" flexDirection="column" gap="xs" borderStyle="round" borderColor="suggestion" padding="sm">
          <Text weight="semibold">{p.form.title}</Text>
          {p.form.message ? <Text dimColor>{p.form.message}</Text> : null}
          {fields.map((f) =>
            f.type === "choices" ? (
              // A checkbox list: any number checked, submitted
              // as the checked values joined by commas.
              <Box key={`field:${f.name}`} flexDirection="column" gap="xs">
                <Text size="caption" dimColor>
                  {f.label}
                </Text>
                {(f.options ?? []).map((o) => {
                  const checked = (p.values[f.name] ?? "").split(",").includes(o.value);
                  return (
                    <Box key={`opt:${f.name}:${o.value}`} flexDirection="column">
                      <Button
                        key={`check:${f.name}:${o.value}`}
                        label={`${checked ? "☑" : "☐"} ${o.label}`}
                        plain
                        onPress={() => {
                          const now = new Set((p.values[f.name] ?? "").split(",").filter(Boolean));
                          if (checked) now.delete(o.value);
                          else now.add(o.value);
                          p.values[f.name] = (f.options ?? [])
                            .map((x) => x.value)
                            .filter((v) => now.has(v))
                            .join(",");
                          redraw($);
                        }}
                      />
                      {o.detail ? (
                        <Text size="caption" dimColor>
                          {o.detail}
                        </Text>
                      ) : null}
                    </Box>
                  );
                })}
              </Box>
            ) : f.type === "choice" ? (
              <Box key={`field:${f.name}`} flexDirection="column" gap="xs">
                <Text size="caption" dimColor>
                  {f.label}
                </Text>
                {(f.options ?? []).map((o) => (
                  <Box key={`opt:${f.name}:${o.value}`} flexDirection="column">
                    <Button
                      key={`choose:${f.name}:${o.value}`}
                      label={`${p.values[f.name] === o.value ? "●" : "○"} ${o.label}`}
                      plain
                      onPress={() => {
                        p.values[f.name] = o.value;
                        redraw($);
                      }}
                    />
                    {o.detail ? (
                      <Text size="caption" dimColor>
                        {o.detail}
                      </Text>
                    ) : null}
                  </Box>
                ))}
              </Box>
            ) : (
              // The hint goes above the field: an Input's own label shares its
              // row and squeezes a long one's field (test pane).
              <Box key={`field:${f.name}`} flexDirection="column" gap="xs">
                <Text size="caption" dimColor>
                  {f.label}
                </Text>
                <Input
                  key={`input:${f.name}`}
                  value={p.values[f.name]}
                  rows={f.type === "textarea" ? 4 : 1}
                  onInput={(v: string) => {
                    p.values[f.name] = v;
                    if (f.type === "subject") {
                      if (p.picked[f.name] && p.picked[f.name]?.label !== v) p.picked[f.name] = null;
                      // Below the minimum the suggestions show again at once.
                      if (v.trim().length < SUBJECT_SEARCH_MIN) redraw($);
                      void searchField($, p, f.name, v);
                    }
                  }}
                  onSubmit={(v: string) => {
                    p.values[f.name] = v;
                    if (single) submit();
                  }}
                />
                {f.type === "subject" ? matchesOf(f.name) : null}
              </Box>
            ),
          )}
          <Box flexDirection="row" flexWrap="wrap" gap="sm">
            <Button key="form:confirm" label={p.form.confirmLabel ?? "Apply"} variant="primary" onPress={submit} />
            <Button key="form:cancel" label="Cancel" onPress={cancel} />
          </Box>
        </Box>
      );
    };

    // --- One card, from the core's view.
    const drawActions = (view: CardView, card: CurationCard, item?: SessionItem) => {
      if (S.form && S.form.owner === card.key) return drawForm(S.form);
      // One action at a time: while any form is open, no other button shows.
      if (S.form) return null;
      const buttons = view.actions.map((a: ActionView) =>
        a.enabled ? (
          <Button
            key={`act:${card.key}:${a.gesture}`}
            label={a.label}
            {...(a.primary ? { variant: "primary" } : {})}
            onPress={() => void act($, a.gesture, card, undefined, item)}
          />
        ) : (
          <Text key={`act:${card.key}:${a.gesture}`} dimColor>
            {`${a.label}: ${a.note ?? "not available here"}`}
          </Text>
        ),
      );
      return (
        <Box flexDirection="row" flexWrap="wrap" gap="sm">
          {buttons}
        </Box>
      );
    };

    const drawBrief = (b: BriefView, flagged: boolean, card: CurationCard, actions?: ActionView[], assigned?: string) => (
      <Box
        key={`brief:${card.key}:${b.particleId}`}
        flexDirection="column"
        gap="xs"
        borderStyle="round"
        borderColor={assigned ? "success" : flagged ? "suggestion" : "subtle"}
        padding="sm"
      >
        <Text>{b.claim}</Text>
        {b.meta ? (
          <Text size="caption" dimColor>
            {b.meta}
          </Text>
        ) : null}
        {b.reason ? (
          <Text size="caption" dimColor>
            {b.reason}
          </Text>
        ) : null}
        {b.source ? (
          b.source.linkable ? (
            <Link href={b.source.text} label={`source: ${b.source.text}`} />
          ) : (
            <Text size="caption" dimColor>{`source: ${b.source.text}`}</Text>
          )
        ) : null}
        {assigned ? (
          <Text key={`assigned:${b.particleId}`} color="success" weight="semibold">
            {`✓ ${assigned}`}
          </Text>
        ) : null}
        {S.form && S.form.owner === `member:${b.particleId}` ? (
          drawForm(S.form)
        ) : actions && !S.form ? (
          <Box flexDirection="row" flexWrap="wrap" gap="sm">
            {actions.map((a) => (
              <Button
                key={`member:${b.particleId}:${a.gesture}`}
                label={a.label}
                onPress={() => void act($, a.gesture, card, b)}
              />
            ))}
          </Box>
        ) : null}
      </Box>
    );

    const drawRef = (r: RefBlock, i: number, card: CurationCard) => {
      switch (r.type) {
        case "side":
          return (
            <Text key={`ref:${i}`} size="caption" weight="semibold">
              {r.label}
            </Text>
          );
        case "brief":
          return drawBrief(r.brief, r.flagged, card, r.actions, r.assigned);
        // Like every other control, these wait while a form is open: a form is
        // drawn inside its belief's box, so moving that belief off the page
        // would hide the form with nothing left to press.
        case "show-all":
          return S.form ? null : (
            <Button key={`show-all:${card.key}`} label={r.label} onPress={() => void showAll($, card)} />
          );
        case "pager":
          return (
            <Box key={`pager:${card.key}`} flexDirection="row" flexWrap="wrap" gap="sm" alignItems="center">
              <Text dimColor>{r.text}</Text>
              {r.previous && !S.form ? (
                <Button
                  key={`page:${card.key}:previous`}
                  label={PAGE_PREVIOUS_LABEL}
                  onPress={() => turnPage($, card, r.start - r.size)}
                />
              ) : null}
              {r.next && !S.form ? (
                <Button
                  key={`page:${card.key}:next`}
                  label={PAGE_NEXT_LABEL}
                  onPress={() => turnPage($, card, r.start + r.size)}
                />
              ) : null}
            </Box>
          );
        case "url":
          return r.url.linkable ? (
            <Link key={`ref:${i}`} href={r.url.text} label={r.url.text} />
          ) : (
            <Text key={`ref:${i}`}>{r.url.text}</Text>
          );
        case "break":
          return null;
        default:
          return (
            <Text key={`ref:${i}`} dimColor italic={r.type === "verdict" || r.type === "note"}>
              {r.text}
            </Text>
          );
      }
    };

    const drawCard = (card: CurationCard, item?: SessionItem) => {
      const view = cardView(card, {
        readOnly: S.readOnly,
        assigned: S.assigned.get(card.key),
        page: { start: S.pages.get(card.key) ?? 0, size: PAGE_SIZE },
      });
      const helpOpen = S.helpOpen.has(card.key);
      return (
        <Box
          key={`card:${card.key}`}
          flexDirection="column"
          gap="sm"
          borderStyle="round"
          borderColor="subtle"
          padding="md"
        >
          <Box flexDirection="row" flexWrap="wrap" gap="sm" alignItems="center">
            <Text color="suggestion">●</Text>
            <Text size="heading" weight="semibold">
              {view.title}
            </Text>
            {view.leverage ? (
              <Text size="caption" dimColor mono>
                {view.leverage}
              </Text>
            ) : null}
          </Box>
          {view.question ? <Text dimColor>{view.question}</Text> : null}
          {view.diagnostic ? <Text>{view.diagnostic}</Text> : null}
          {view.evidence
            ? link(
                `browse?scope=inconsistency&inconsistency_id=${encodeURIComponent(view.evidence.inconsistencyId)}`,
                view.evidence.label,
              )
            : null}
          {view.refs.map((r, i) => drawRef(r, i, card))}
          {S.busy && !S.form ? <Text dimColor>Working…</Text> : drawActions(view, card, item)}
          {view.help.length ? (
            <Button
              key={`help:${card.key}`}
              label={`${helpOpen ? "▾" : "▸"} ${HELP_SUMMARY}`}
              plain
              onPress={() => {
                if (helpOpen) S.helpOpen.delete(card.key);
                else S.helpOpen.add(card.key);
                redraw($);
              }}
            />
          ) : null}
          {helpOpen ? (
            <Markdown key={`helptext:${card.key}`} dimColor text={view.help.map((h) => `**${h.label}**: ${h.hint}`).join("\n\n")} />
          ) : null}
        </Box>
      );
    };

    // --- The pane.
    const header = (
      <Box flexDirection="row" flexWrap="wrap" gap="sm" alignItems="center">
        <Text color="suggestion">◉</Text>
        <Text weight="semibold">Particles</Text>
        <Button key="refresh" label={REFRESH_LABEL} onPress={() => void (S.phase === "ready" ? load($) : start($))} />
        {link("query", "Query ↗")}
        {link("browse", "Browse ↗")}
      </Box>
    );

    const body: unknown[] = [];
    if (S.phase === "setup" || S.phase === "error") {
      body.push(
        <Text key="problem" color={S.phase === "error" ? "error" : undefined}>
          {S.message}
        </Text>,
      );
    } else if (S.phase === "loading" || S.phase === "idle") {
      body.push(
        <Text key="loading" dimColor>
          Loading the queue…
        </Text>,
      );
    } else {
      body.push(
        <Text key="count" size="caption" dimColor>
          {queueHeader(S.cards.length, S.openCount)}
        </Text>,
        <Box key="filters" flexDirection="row" flexWrap="wrap" gap="sm" alignItems="center">
          <Select
            key="kind"
            options={KIND_FILTER.map((k) => ({ value: k || "all", label: kindFilterLabel(k) }))}
            value={S.kind || "all"}
            onSelect={(v: string) => {
              S.kind = v === "all" ? "" : v;
              void load($);
            }}
          />
          <Button
            key="semantic"
            label={`${S.semantic ? "☑" : "☐"} ${SEMANTIC_LABEL}`}
            plain
            onPress={() => {
              S.semantic = !S.semantic;
              void load($);
            }}
          />
        </Box>,
      );
      if (S.readOnly) body.push(<Text key="readonly" color="warning">{readOnlyText(S.readOnlyReason)}</Text>);
      if (S.semanticSkipped) body.push(<Text key="semantic-skipped" dimColor>{SEMANTIC_SKIPPED_TEXT}</Text>);
      for (const [i, n] of S.notices.entries()) {
        body.push(
          <Text key={`notice:${i}`} color={n.level === "error" ? "error" : undefined} dimColor={n.level === "info"}>
            {n.text}
          </Text>,
        );
      }
      if (S.session.length) {
        body.push(
          <Text key="session-title" weight="semibold">
            In this session
          </Text>,
        );
        for (const item of S.session) {
          body.push(
            item.card ? (
              drawCard(item.card, item)
            ) : (
              <Box key={`hit:${item.hit.inconsistencyId}`} flexDirection="column" gap="xs" borderStyle="round" borderColor="subtle" padding="sm">
                <Text>{item.hit.content ? `A contested belief the agent used: ${item.hit.content}` : "A belief the agent used is contested."}</Text>
                {link(
                  `browse?scope=inconsistency&inconsistency_id=${encodeURIComponent(item.hit.inconsistencyId)}`,
                  "Open the conflict in the web UI ↗",
                )}
              </Box>
            ),
          );
        }
      }
      if (S.cards.length) {
        body.push(drawCard(S.cards[0]));
      } else {
        body.push(
          <Text key="empty" dimColor>
            {`✓ ${emptyText(S.openCount)}`}
          </Text>,
        );
      }
      body.push(<Box key="webui">{link("curate", "Open the full queue in the web UI ↗")}</Box>);
    }

    const footer = [`core ${CORE_VERSION}`, `engine ${S.engineVersion || "?"}`];
    if (S.missing.length) footer.push(`engine lacks ${S.missing.join(", ")}`);

    return (
      <Box flexDirection="column" gap="sm">
        {header}
        {body}
        <Text key="footer" size="caption" dimColor color={S.missing.length ? "warning" : undefined}>
          {footer.join(" · ")}
        </Text>
      </Box>
    );
  });
};
