/*
 * The pane against the curation core's contract (Acceptance 4):
 * every example card draws on the desktop and terminal surfaces with a
 * control for every action, and each action, answered as the contract's
 * scripted operator answers, sends exactly the contract's requests. A hook
 * on `http.fetch` stands for the engine.
 *
 *   claude plugin test clients/claude-code-plugin
 */
import { expect, test } from "claude-code/testing";
import contract from "./contract.gen.ts";

const PANE = {
  plugin: "particles-curation",
  component: "Pane",
  requestId: "particles",
  props: {
    title: "Particles",
    isFocused: false,
    bodyColumns: 38,
    placement: "dock",
    scroll: { offset: 0, bodyRows: 40 },
    view: {},
  },
} as const;

const OPTIONS = { options: { engineToken: "test-key", reviewerId: "reviewer-1" } };

/** The contract's requests as this client sends them: it deposits as itself. */
function asThisClient(requests: Req[]): Req[] {
  return JSON.parse(JSON.stringify(requests).replaceAll('"deposited_by":"web-ui"', '"deposited_by":"claude-code"'));
}
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

const SUBJECTS = [
  { id: "11111111-1111-4111-8111-111111111111", canonical_name: "Particles", aliases: ["particles-engine-py"], subject_class: "software", asserted_by: "t" },
  { id: "22222222-2222-4222-8222-222222222222", canonical_name: "Particles spec", aliases: [], asserted_by: "t" },
];

type Req = { method: string; path: string; body?: unknown };
type Engine = {
  requests: Req[];
  /** Every request, the queue and version reads included. */
  all: Req[];
  answer?: (req: Req) => { status: number; body: unknown } | undefined;
  /** The status the suggestion read answers with. */
  suggestStatus: number;
};

const SUGGESTED = [
  { subject: { id: "33333333-3333-4333-8333-333333333333", canonical_name: "Gate checks", aliases: [], subject_class: "concept", asserted_by: "t" }, named: true, neighbours: 0 },
  { subject: SUBJECTS[0], named: false, neighbours: 4 },
];

/** A stand-in engine: the queue holds `card`; writes are recorded. */
function engine(on: any, card: unknown, builtAt: string | null = "2026-11-05T00:00:00Z"): Engine {
  const state: Engine = { requests: [], all: [], suggestStatus: 200 };
  on("http.fetch", (_$: unknown, e: { url: string; init?: { method?: string; body?: string } }) => {
    const url = new URL(e.url);
    const path = url.pathname + url.search;
    const method = e.init?.method ?? "GET";
    const body = e.init?.body ? JSON.parse(e.init.body) : undefined;
    const req: Req = body === undefined ? { method, path } : { method, path, body };
    state.all.push(req);
    const reply = (status: number, value: unknown) => ({
      value: { status, ok: status >= 200 && status < 300, headers: {}, text: JSON.stringify(value) },
    });
    if (method === "GET" && (path === "/curation" || path.startsWith("/curation?"))) {
      return reply(200, { cards: [structuredClone(card)], count: 1, open_count: 9, built_at: builtAt });
    }
    if (path === "/health") return reply(200, { status: "ok", version: "9.9.9" });
    if (path === "/openapi.json") return reply(200, { paths: Object.fromEntries(ROUTES.map((r) => [r, {}])) });
    // A subject field's search is a read the field makes, not the action's.
    if (method === "GET" && path.startsWith("/subjects/search?")) {
      const q = url.searchParams.get("q") ?? "";
      return reply(200, SUBJECTS.filter((x) => x.canonical_name.toLowerCase().includes(q.toLowerCase())));
    }
    // So are a subject field's suggestions.
    if (method === "GET" && path.endsWith("/subject-suggestions")) {
      return reply(state.suggestStatus, state.suggestStatus === 200 ? SUGGESTED : { detail: "boom" });
    }
    state.requests.push(req);
    const custom = state.answer?.(req);
    if (custom) return reply(custom.status, custom.body);
    if (/^\/corpus\/[^/]+\/retract$/.test(path) && (body as { dry_run?: boolean })?.dry_run) {
      return reply(200, { retracted_ids: ["r-1", "r-2"] });
    }
    if (path === "/subjects/relink-gated") return reply(200, { recoverable: 3 });
    return reply(200, {});
  });
  on("session.start", (_$: unknown, e: { cwd: string }) => ({ cwd: e.cwd }));
  on("ui.open", () => ({ value: { isPlaced: true } }));
  on("ui.panes", () => ({ value: [{ id: "particles", title: "Particles", isShown: true, isFocused: false, isPlaced: true }] }));
  on("command.register", () => ({ value: { command: "particles" } }));
  return state;
}

type Trace = {
  forms: { form: { fields?: { name: string; type: string; value?: string }[] }; answer: Record<string, string> }[];
  requests: Req[];
};

/** Answer each form as the contract's scripted operator did, or cancel the first. */
async function operate(ui: any, trace: Trace, cancel: boolean): Promise<void> {
  for (const { form, answer } of trace.forms) {
    if (cancel) {
      await ui.press({ key: "form:cancel" });
      return;
    }
    for (const f of form.fields ?? []) {
      const value = answer[f.name] ?? "";
      if (f.type === "choice") {
        if (value) await ui.press({ key: `choose:${f.name}:${value}` });
      } else if (f.type === "choices") {
        // Toggle each box whose state differs from what the form checked at first.
        const want = new Set(value.split(",").filter(Boolean));
        const had = new Set((f.value ?? "").split(",").filter(Boolean));
        for (const o of f.options ?? []) {
          if (want.has(o.value) !== had.has(o.value)) await ui.press({ key: `check:${f.name}:${o.value}` });
        }
      } else if (value !== (f.value ?? "")) {
        await ui.input({ key: `input:${f.name}`, text: value, kind: "change" });
      }
    }
    await ui.press({ key: "form:confirm" });
  }
}

const cards = Object.entries(contract.cards as Record<string, any>);

for (const surface of ["desktop", "terminal"] as const) {
  for (const [name, entry] of cards) {
    test(`${name} draws on ${surface} with a control per action`, OPTIONS, async ($: any, on: any) => {
      engine(on, entry.card);
      await $.session.start({ cwd: "/", surface, isInteractive: true });
      const ui = await $.ui.mount({ ...PANE, surface });
      await ui.redraw();
      expect((await ui.find({ type: "Text", text: entry.view.title }))?.text).toBe(entry.view.title);
      for (const a of entry.view.actions) {
        if (a.enabled) {
          const control = await ui.find({ key: `act:${entry.card.key}:${a.gesture}` });
          expect(control?.props.label).toBe(a.label);
        } else {
          // A gesture this client cannot send is named, with why, not offered.
          const text = `${a.label}: ${a.note}`;
          expect((await ui.find({ type: "Text", text }))?.text).toBe(text);
        }
      }
      await ui.unmount();
    });
  }
}

for (const [name, entry] of cards) {
  for (const [gesture, traces] of Object.entries(entry.actions as Record<string, { confirm: Trace; cancel: Trace }>)) {
    for (const mode of ["confirm", "cancel"] as const) {
      const trace = traces[mode];
      if (mode === "cancel" && trace.forms.length === 0) continue;
      test(`${name}: ${gesture} (${mode}) sends the contract's requests`, OPTIONS, async ($: any, on: any) => {
        const eng = engine(on, entry.card);
        await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
        const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
        await ui.redraw();
        await ui.press({ key: `act:${entry.card.key}:${gesture}` });
        await operate(ui, trace, mode === "cancel");
        expect(eng.requests).toEqual(asThisClient(trace.requests));
        await ui.unmount();
      });
    }
  }
  if (entry.member) {
    for (const [gesture, trace] of Object.entries(entry.member.actions as Record<string, Trace>)) {
      test(`${name}: member ${gesture} acts on the belief's own key`, OPTIONS, async ($: any, on: any) => {
        const eng = engine(on, entry.card);
        await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
        const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
        await ui.redraw();
        await ui.press({ key: `member:${entry.member.card.particle_ids[0]}:${gesture}` });
        await operate(ui, trace, false);
        expect(eng.requests).toEqual(asThisClient(trace.requests));
        await ui.unmount();
      });
    }
  }
}

// A belief's form opens in that belief's box, so the operator sees which
// belief it is about, with one confirm button (owner's live run, 2026-11-07).

/** The keys of every element under a found one. */
function keysUnder(el: unknown, out: string[] = []): string[] {
  if (el && typeof el === "object") {
    const node = el as { key?: string; props?: { key?: string }; children?: unknown[] };
    const key = node.key ?? node.props?.key;
    if (typeof key === "string") out.push(key);
    for (const child of node.children ?? []) keysUnder(child, out);
  }
  return out;
}

for (const [name, entry] of cards.filter(([, e]) => e.member)) {
  test(`${name}: a belief's form opens inside that belief's box`, OPTIONS, async ($: any, on: any) => {
    engine(on, entry.card);
    await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
    const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
    await ui.redraw();
    const pid = entry.member.card.particle_ids[0];
    await ui.press({ key: `member:${pid}:assign-subject` });
    const inside = keysUnder(await ui.find({ key: `brief:${entry.card.key}:${pid}` }));
    expect(inside).toContain("input:subject");
    expect(inside).toContain("form:confirm");
    expect(await ui.findAll({ type: "Button", text: "Assign" })).toHaveLength(1);
    // Every other control waits while the form is open.
    const all = keysUnder(await ui.find({ type: "Box" }));
    expect(all.filter((k) => k.startsWith("act:") || k.startsWith("member:"))).toEqual([]);
    await ui.press({ key: "form:cancel" });
    expect(await ui.find({ key: `member:${pid}:assign-subject` })).toBeDefined();
    await ui.unmount();
  });
}

// --- Assign subject's search (the owner's live run, 2026-11-07) -----------

const SOURCE = (contract.cards as Record<string, any>)["10-subjectless_source"];

async function openAssign($: any, on: any): Promise<{ ui: any; eng: Engine; pid: string }> {
  const eng = engine(on, SOURCE.card);
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  const pid = SOURCE.member.card.particle_ids[0];
  await ui.press({ key: `member:${pid}:assign-subject` });
  return { ui, eng, pid };
}

test("typing lists matching subjects, and a pick assigns by id", OPTIONS, async ($: any, on: any) => {
  const { ui, eng, pid } = await openAssign($, on);
  await ui.input({ key: "input:subject", text: "partic", kind: "change" });
  await ui.redraw();
  expect((await ui.find({ key: `pick:subject:${SUBJECTS[1].id}` }))?.props.label).toBe("○ Particles spec");
  expect((await ui.find({ type: "Text", text: "software · also: particles-engine-py" }))?.text).toBe("software · also: particles-engine-py");
  await ui.press({ key: `pick:subject:${SUBJECTS[0].id}` });
  expect((await ui.find({ key: `pick:subject:${SUBJECTS[0].id}` }))?.props.label).toBe("● Particles");
  expect((await ui.find({ key: "input:subject" }))?.props.value).toBe("Particles");
  await ui.press({ key: "form:confirm" });
  expect(eng.requests).toEqual([
    { method: "POST", path: `/particles/${pid}/subjects`, body: { subject_id: SUBJECTS[0].id } },
  ]);
  await ui.unmount();
});

test("text edited after a pick assigns by name", OPTIONS, async ($: any, on: any) => {
  const { ui, eng, pid } = await openAssign($, on);
  await ui.input({ key: "input:subject", text: "partic", kind: "change" });
  await ui.redraw();
  await ui.press({ key: `pick:subject:${SUBJECTS[0].id}` });
  await ui.input({ key: "input:subject", text: "Particles core", kind: "change" });
  await ui.press({ key: "form:confirm" });
  expect(eng.requests).toEqual([
    { method: "POST", path: `/particles/${pid}/subjects`, body: { subject_name: "Particles core" } },
  ]);
  await ui.unmount();
});

test("an assigned belief stays, marked, with Edit to correct it", OPTIONS, async ($: any, on: any) => {
  const { ui, pid } = await openAssign($, on);
  await ui.input({ key: "input:subject", text: "partic", kind: "change" });
  await ui.redraw();
  await ui.press({ key: `pick:subject:${SUBJECTS[0].id}` });
  await ui.press({ key: "form:confirm" });
  const mark = "✓ Subject assigned: Particles";
  expect((await ui.find({ type: "Text", text: mark }))?.text).toBe(mark);
  expect(await ui.find({ key: `member:${pid}:supersede` })).toBeDefined();
  expect(await ui.find({ key: `member:${pid}:assign-subject` })).toBeUndefined();
  expect(await ui.find({ key: `member:${pid}:retract` })).toBeUndefined();
  // The card stays, with its own actions, while a belief is left to handle.
  expect(await ui.find({ key: `act:${SOURCE.card.key}:next-card` })).toBeUndefined();
  await ui.unmount();
});

test("once every belief is assigned the card offers Next card", OPTIONS, async ($: any, on: any) => {
  // Every belief briefed: a card with beliefs still unshown is never all assigned.
  const card = { ...SOURCE.card, particle_ids: SOURCE.card.particles.map((b: { particle_id: string }) => b.particle_id) };
  const eng = engine(on, card);
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  const ids: string[] = card.particle_ids;
  for (const id of ids) {
    await ui.press({ key: `member:${id}:assign-subject` });
    await ui.input({ key: "input:subject", text: `Subject for ${id.slice(-2)}`, kind: "change" });
    await ui.press({ key: "form:confirm" });
  }
  expect(eng.requests).toHaveLength(ids.length);
  const next = await ui.find({ key: `act:${SOURCE.card.key}:next-card` });
  expect(next?.props.label).toBe("Next card");
  await ui.press({ key: `act:${SOURCE.card.key}:next-card` });
  expect(eng.requests).toHaveLength(ids.length);
  expect(await ui.find({ key: `act:${SOURCE.card.key}:next-card` })).toBeUndefined();
  await ui.unmount();
});

test("no match says what Assign will do with the name", OPTIONS, async ($: any, on: any) => {
  const { ui } = await openAssign($, on);
  await ui.input({ key: "input:subject", text: "zzz", kind: "change" });
  await ui.redraw();
  const text = "No subject matches. Assign resolves the name as typed, creating the subject if needed.";
  expect((await ui.find({ type: "Text", text }))?.text).toBe(text);
  await ui.unmount();
});

test("a 403 turns the pane read-only, in the engine's words", OPTIONS, async ($: any, on: any) => {
  const plain = (contract.cards as Record<string, any>)["22-stale"];
  const eng = engine(on, plain.card);
  eng.answer = () => ({ status: 403, body: { detail: "Belief writes are disabled for store 'default'." } });
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  await ui.press({ key: `act:${plain.card.key}:affirm` });
  const text = "Belief writes are disabled for store 'default'. Reviewing without write gestures.";
  expect((await ui.find({ type: "Text", text }))?.text).toBe(text);
  expect(await ui.find({ key: `act:${plain.card.key}:affirm` })).toBeUndefined();
  await ui.unmount();
});

// --- The contested-belief notice --------------------------------

const CONFLICT = (contract.cards as Record<string, any>)["19-inconsistency"].card;
const CONFLICT_ID = CONFLICT.inconsistency_id as string;

function queryResult(contested: string) {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          answer: "…",
          particles: [{ id: "00000000-0000-4000-8000-0000000000aa", content: "The default port is 8000.", contested }],
        }),
      },
    ],
  };
}

/** The agent's Particles query, answered with one hit contested by `contested`. */
function agentQuery(on: any, contested: string): void {
  on("tool.call", { tool: "mcp__particles__query" }, () => ({ result: queryResult(contested) }));
}

async function query($: any): Promise<void> {
  await $.tool.call({ tool: "mcp__particles__query", question: "What is the default port?" });
}

test("a contested hit raises that conflict's card", OPTIONS, async ($: any, on: any) => {
  const eng = engine(on, (contract.cards as Record<string, any>)["00-stale"].card);
  agentQuery(on, CONFLICT_ID);
  eng.answer = (req) =>
    req.path === `/curation/cards/${encodeURIComponent(`inconsistency:${CONFLICT_ID}`)}`
      ? { status: 200, body: CONFLICT }
      : undefined;
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  await query($);
  await ui.redraw();
  expect((await ui.find({ type: "Text", text: "In this session" }))?.text).toBe("In this session");
  expect(await ui.find({ key: `act:${CONFLICT.key}:resolve` })).toBeDefined();
  expect(eng.requests).toEqual([
    { method: "GET", path: `/curation/cards/${encodeURIComponent(`inconsistency:${CONFLICT_ID}`)}` },
  ]);
  await ui.unmount();
});

test("a malformed contested id makes no lookup", OPTIONS, async ($: any, on: any) => {
  const eng = engine(on, (contract.cards as Record<string, any>)["00-stale"].card);
  agentQuery(on, "../../subjects?x=1");
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  await query($);
  await ui.redraw();
  expect(eng.requests).toEqual([]);
  expect(await ui.find({ type: "Text", text: "In this session" })).toBeUndefined();
  await ui.unmount();
});

test("with no snapshot, a contested hit is listed and not looked up", OPTIONS, async ($: any, on: any) => {
  const eng = engine(on, (contract.cards as Record<string, any>)["00-stale"].card, null);
  agentQuery(on, CONFLICT_ID);
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  await query($);
  await ui.redraw();
  expect(eng.requests).toEqual([]);
  expect((await ui.find({ type: "Text", text: "In this session" }))?.text).toBe("In this session");
  await ui.unmount();
});

test("no key and a remote engine: no call at all", { options: { engineUrl: "https://engine.example" } }, async ($: any, on: any) => {
  const eng = engine(on, (contract.cards as Record<string, any>)["00-stale"].card);
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  expect(eng.all).toEqual([]);
  await ui.unmount();
});

test("an http engine that is not on this machine is refused", { options: { engineUrl: "http://engine.example", engineToken: "k" } }, async ($: any, on: any) => {
  const eng = engine(on, (contract.cards as Record<string, any>)["00-stale"].card);
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  expect(eng.all).toEqual([]);
  await ui.unmount();
});

// --- Show all (the owner's live run, 2026-11-08) ---------------------------

/** The pane refuses a tree past this many characters serialized. */
const TREE_BOUND = 100_000;

async function keys(ui: any): Promise<string[]> {
  return keysUnder(await ui.find({ type: "Box" }));
}

async function treeSize(ui: any): Promise<number> {
  return JSON.stringify(await ui.find({ type: "Box" })).length;
}

test("Show all pages a source card's beliefs, inside the pane's size bound", OPTIONS, async ($: any, on: any) => {
  // As many beliefs as the live card held, with claims of ordinary length.
  const extra = Array.from({ length: 220 }, (_, i) => `99999999-0000-4000-8000-${String(i).padStart(12, "0")}`);
  const card = { ...SOURCE.card, particle_ids: [...SOURCE.card.particle_ids, ...extra] };
  const ids: string[] = card.particle_ids;
  const briefs = SOURCE.card.particles as { particle_id: string }[];
  const claim = (i: number) => `Belief ${i}. ${"A claim of ordinary length about the store. ".repeat(4)}`;
  const full = { ...card, particles: ids.map((id, i) => ({ ...briefs[0], particle_id: id, content: claim(i) })) };
  const eng = engine(on, card);
  eng.answer = (req) =>
    req.method === "GET" && req.path === `/curation/cards/${encodeURIComponent(card.key)}`
      ? { status: 200, body: full }
      : undefined;
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  const show = `show-all:${card.key}`;
  expect((await ui.find({ key: show }))?.props.label).toBe(`Show all ${ids.length} (${ids.length - briefs.length} more)`);
  await ui.press({ key: show });
  await ui.redraw();
  expect(await keys(ui)).not.toContain(show);
  const pageText = (a: number, b: number) => `Beliefs ${a}–${b} of ${ids.length}`;
  expect((await ui.find({ type: "Text", text: pageText(1, 20) }))?.text).toBe(pageText(1, 20));
  expect((await ui.find({ type: "Text", text: claim(19) }))?.text).toBe(claim(19));
  expect(await ui.find({ type: "Text", text: claim(20) })).toBeUndefined();
  expect(await keys(ui)).not.toContain(`page:${card.key}:previous`);
  expect(await treeSize(ui)).toBeLessThan(TREE_BOUND);
  await ui.press({ key: `page:${card.key}:next` });
  await ui.redraw();
  expect((await ui.find({ type: "Text", text: pageText(21, 40) }))?.text).toBe(pageText(21, 40));
  expect(await keys(ui)).toContain(`member:${ids[20]}:assign-subject`);
  expect(await keys(ui)).not.toContain(`member:${ids[0]}:assign-subject`);
  await ui.press({ key: `page:${card.key}:previous` });
  await ui.redraw();
  expect((await ui.find({ type: "Text", text: pageText(1, 20) }))?.text).toBe(pageText(1, 20));
  for (let i = 0; i < 20; i++) await ui.press({ key: `page:${card.key}:next` }).catch(() => undefined);
  await ui.redraw();
  const lastStart = Math.floor((ids.length - 1) / 20) * 20;
  expect((await ui.find({ type: "Text", text: pageText(lastStart + 1, ids.length) }))?.text).toBe(
    pageText(lastStart + 1, ids.length),
  );
  expect(await keys(ui)).not.toContain(`page:${card.key}:next`);
  expect(await treeSize(ui)).toBeLessThan(TREE_BOUND);
  await ui.unmount();
});

test("the pager and Show all wait while a belief's form is open", OPTIONS, async ($: any, on: any) => {
  const extra = Array.from({ length: 40 }, (_, i) => `99999999-0000-4000-8000-${String(i).padStart(12, "0")}`);
  const card = { ...SOURCE.card, particle_ids: [...SOURCE.card.particle_ids, ...extra] };
  const ids: string[] = card.particle_ids;
  const briefs = SOURCE.card.particles as { particle_id: string }[];
  const full = { ...card, particles: ids.map((id, i) => ({ ...briefs[0], particle_id: id, content: `Belief ${i}` })) };
  const eng = engine(on, card);
  eng.answer = (req) =>
    req.method === "GET" && req.path === `/curation/cards/${encodeURIComponent(card.key)}`
      ? { status: 200, body: full }
      : undefined;
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  await ui.press({ key: `member:${ids[0]}:assign-subject` });
  expect(await keys(ui)).not.toContain(`show-all:${card.key}`);
  await ui.press({ key: "form:cancel" });
  await ui.press({ key: `show-all:${card.key}` });
  await ui.redraw();
  expect(await keys(ui)).toContain(`page:${card.key}:next`);
  await ui.press({ key: `member:${ids[0]}:assign-subject` });
  const open = await keys(ui);
  expect(open).not.toContain(`page:${card.key}:next`);
  expect(open).toContain("input:subject");
  await ui.press({ key: "form:cancel" });
  expect(await keys(ui)).toContain(`page:${card.key}:next`);
  await ui.unmount();
});

// --- Assign subject's suggestions --------------------------------

test("the form opens with suggestions, each saying why, and a pick assigns by id", OPTIONS, async ($: any, on: any) => {
  const { ui, eng, pid } = await openAssign($, on);
  await ui.redraw();
  expect(eng.all).toContainEqual({ method: "GET", path: `/particles/${pid}/subject-suggestions` });
  expect((await ui.find({ type: "Text", text: "Suggested" }))?.text).toBe("Suggested");
  expect((await ui.find({ key: "pick:subject:33333333-3333-4333-8333-333333333333" }))?.props.label).toBe("○ Gate checks");
  expect((await ui.find({ type: "Text", text: "concept · named in the claim" }))?.text).toBe("concept · named in the claim");
  expect((await ui.find({ type: "Text", text: "software · on 4 similar beliefs" }))?.text).toBe("software · on 4 similar beliefs");
  await ui.press({ key: `pick:subject:${SUBJECTS[0].id}` });
  expect((await ui.find({ key: `pick:subject:${SUBJECTS[0].id}` }))?.props.label).toBe("● Particles");
  await ui.press({ key: "form:confirm" });
  expect(eng.requests).toEqual([
    { method: "POST", path: `/particles/${pid}/subjects`, body: { subject_id: SUBJECTS[0].id } },
  ]);
  await ui.unmount();
});

test("typing replaces the suggestions with matches, and clearing brings them back", OPTIONS, async ($: any, on: any) => {
  const { ui } = await openAssign($, on);
  await ui.redraw();
  await ui.input({ key: "input:subject", text: "spec", kind: "change" });
  await ui.redraw();
  expect(await ui.find({ type: "Text", text: "Suggested" })).toBeUndefined();
  expect((await ui.find({ key: `pick:subject:${SUBJECTS[1].id}` }))?.props.label).toBe("○ Particles spec");
  expect(await ui.find({ key: "pick:subject:33333333-3333-4333-8333-333333333333" })).toBeUndefined();
  await ui.input({ key: "input:subject", text: "", kind: "change" });
  await ui.redraw();
  expect((await ui.find({ type: "Text", text: "Suggested" }))?.text).toBe("Suggested");
  expect(await ui.find({ key: "pick:subject:33333333-3333-4333-8333-333333333333" })).toBeDefined();
  await ui.unmount();
});

test("a failed suggestion read leaves the field usable, with a note", OPTIONS, async ($: any, on: any) => {
  const eng = engine(on, SOURCE.card);
  eng.suggestStatus = 500;
  await $.session.start({ cwd: "/", surface: "desktop", isInteractive: true });
  const ui = await $.ui.mount({ ...PANE, surface: "desktop" });
  await ui.redraw();
  const pid = SOURCE.member.card.particle_ids[0];
  await ui.press({ key: `member:${pid}:assign-subject` });
  await ui.redraw();
  const note = "Suggestions are not available from this engine. Search instead.";
  expect((await ui.find({ type: "Text", text: note }))?.text).toBe(note);
  await ui.input({ key: "input:subject", text: "partic", kind: "change" });
  await ui.redraw();
  expect(await ui.find({ key: `pick:subject:${SUBJECTS[0].id}` })).toBeDefined();
  await ui.press({ key: `pick:subject:${SUBJECTS[0].id}` });
  await ui.press({ key: "form:confirm" });
  expect(eng.requests).toEqual([
    { method: "POST", path: `/particles/${pid}/subjects`, body: { subject_id: SUBJECTS[0].id } },
  ]);
  await ui.unmount();
});

test("the form does not repeat the claim the card shows above it", OPTIONS, async ($: any, on: any) => {
  const { ui } = await openAssign($, on);
  await ui.redraw();
  const claim: string = SOURCE.card.particles[0].content;
  const texts = JSON.stringify(await ui.find({ key: "form" }));
  expect(texts.includes(claim)).toBe(false);
  await ui.unmount();
});
