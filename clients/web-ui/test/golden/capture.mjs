/*
 * Golden capture of the web UI's curation feed (step 1).
 *
 * Renders every example card in ../../../curation-core/fixtures/cards.json
 * through the real feed (src/feed.ts with the real API client), presses each
 * control, answers each sheet from a fixed script, and records what the feed
 * drew, which sheets it opened and the exact requests it sent. The result is
 * the baseline the shared curation core must reproduce (step 2) and the web
 * UI must still produce once it runs on the core (step 3).
 *
 *   node test/golden/capture.mjs           # write fixtures/golden.json
 *   node test/golden/capture.mjs --check   # fail if the feed's output changed
 */
import esbuild from "esbuild";
import { mkdtempSync, readFileSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { dirname, join, resolve } from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { document, dump } from "./dom.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "../..");
const FIXTURES = resolve(WEB, "../curation-core/fixtures");
const GOLDEN = join(FIXTURES, "golden.json");
const ENGINE = "http://engine.test";

globalThis.document = document;

// --- Bundle the real feed, with sheet.ts replaced by the scripted stub ------

const out = join(mkdtempSync(join(tmpdir(), "golden-")), "feed.mjs");
await esbuild.build({
  stdin: {
    contents:
      'export { CurationFeed } from "./src/feed";\n' +
      'export { ParticlesApiClient } from "./src/api";\n',
    resolveDir: WEB,
    loader: "ts",
  },
  bundle: true,
  format: "esm",
  platform: "neutral",
  outfile: out,
  logLevel: "error",
  plugins: [
    {
      name: "golden-sheet",
      setup(build) {
        build.onResolve({ filter: /(^|\/)sheet$/ }, () => ({
          path: join(HERE, "sheet-stub.mjs"),
        }));
      },
    },
  ],
});
const { CurationFeed, ParticlesApiClient } = await import(pathToFileURL(out).href);

// --- The scripted engine and operator ---------------------------------------

/** Fixed answers to a sheet's fields, by field name, so every run is the same. */
const ANSWERS = {
  subject: "Example subject",
  subjects: "Alpha, Beta",
  content: "Scripted revised claim",
  entry_id: "entry-1",
  text: "Scripted source text",
  note: "scripted note",
};

function answer(spec) {
  const out = {};
  for (const f of spec.fields ?? []) {
    if (f.type === "choice") out[f.name] = f.options?.[0]?.value ?? "";
    // A checkbox list keeps what it checked at first, else checks the first option.
    else if (f.type === "choices") out[f.name] = f.value ?? f.options?.[0]?.value ?? "";
    else out[f.name] = ANSWERS[f.name] ?? f.value ?? `scripted ${f.name}`;
  }
  return out;
}

/** Plain data of a sheet spec (labels, messages, fields, options). */
function sheetData(spec) {
  return JSON.parse(JSON.stringify(spec));
}

async function flush() {
  for (let i = 0; i < 30; i++) await new Promise((r) => setImmediate(r));
}

/**
 * Mount the feed on one queue response. `respond(method, path, body)` returns
 * `{status, body}` or undefined for a plain 200 `{}`.
 */
async function mount(queue, respond = () => undefined, sheet = "answer") {
  const trace = { sheets: [], requests: [] };
  globalThis.__goldenSheet = async (spec) => {
    trace.sheets.push(sheetData(spec));
    if (sheet === "cancel") return null;
    if (typeof sheet === "function") return sheet(spec, trace.sheets.length);
    return answer(spec);
  };
  const fetchFn = async (url, init) => {
    const path = url.slice(ENGINE.length);
    const body = init.body === undefined ? undefined : JSON.parse(init.body);
    const method = init.method;
    let res;
    if (method === "GET" && path.startsWith("/curation?") || path === "/curation") {
      // A fresh copy per answer: the feed edits a source card in place when
      // one of its beliefs is resolved, which must not leak between runs.
      res = { status: 200, body: structuredClone(queue) };
    } else {
      trace.requests.push(body === undefined ? { method, path } : { method, path, body });
      res = respond(method, path, body) ?? { status: 200, body: {} };
    }
    return {
      status: res.status,
      ok: res.status >= 200 && res.status < 300,
      json: async () => res.body,
    };
  };
  const client = new ParticlesApiClient(
    { baseUrl: ENGINE, bearerToken: "token", allowUnauthenticated: false },
    fetchFn,
  );
  const root = document.createElement("main");
  let needsSettings = 0;
  const feed = new CurationFeed(root, {
    client,
    reviewerId: "reviewer-1",
    onNeedsSettings: () => needsSettings++,
  });
  await feed.refresh();
  await flush();
  return { feed, root, trace, needsSettings: () => needsSettings };
}

function queueOf(card, openCount = 9) {
  return { cards: [card], count: 1, open_count: openCount, semantic_skipped: false };
}

/** The card's own gesture buttons (not a member belief's). */
function topButtons(root) {
  const card = root.querySelector(".card");
  if (!card) return [];
  const holder = card.children[card.children.length - 1];
  const row = holder.querySelector(".gestures") ?? holder.children[0];
  return row ? row.children.filter((b) => b.tagName === "BUTTON") : [];
}

function label(btn) {
  return btn.childNodes.length ? btn.childNodes[0].textContent : "";
}

function after(root) {
  return {
    banners: root.querySelectorAll(".banner").map((b) => [b.className, b.textContent]),
    card: root.querySelector(".card") ? root.querySelector(".card").querySelector(".kind").textContent : null,
    header: root.querySelector(".session-count")?.textContent ?? null,
  };
}

/** Engine answers that let each multi-step flow run to its end. */
function engine(card) {
  return (method, path, body) => {
    if (path.match(/^\/corpus\/[^/]+\/retract$/) && body?.dry_run) {
      return { status: 200, body: { retracted_ids: ["r-1", "r-2"] } };
    }
    if (path === "/subjects/relink-gated") return { status: 200, body: { recoverable: 3 } };
    if (method === "GET" && path.startsWith("/curation/cards/")) {
      const full = structuredClone(card);
      full.particles = (card.particle_ids ?? []).map((id, i) => ({
        particle_id: id,
        content: `Full brief ${i}`,
        effective_confidence: 0.5,
        status: "ACTIVE",
        subject_labels: [],
      }));
      return { status: 200, body: full };
    }
    return undefined;
  };
}

// --- Scenarios ---------------------------------------------------------------

const { cards } = JSON.parse(readFileSync(join(FIXTURES, "cards.json"), "utf8"));
const golden = { comment: "Captured from clients/web-ui/src/feed.ts by test/golden/capture.mjs.", cards: {}, queue: {} };

for (const card of cards) {
  const entry = { view: null, actions: [], members: [], swipe: {} };
  const base = await mount(queueOf(card));
  entry.view = dump(base.root);

  for (const [i, btn] of topButtons(base.root).entries()) {
    const name = label(btn);
    const disabled = !!btn.disabled;
    if (disabled) {
      entry.actions.push({ label: name, disabled: true });
      continue;
    }
    for (const mode of ["answer", "cancel"]) {
      const m = await mount(queueOf(card), engine(card), mode);
      topButtons(m.root)[i].onclick();
      await flush();
      if (mode === "cancel" && m.trace.sheets.length === 0) continue;
      entry.actions.push({ label: name, mode, sheets: m.trace.sheets, requests: m.trace.requests, after: after(m.root) });
    }
  }
  if (card.kind === "uncited_url" && card.corpus_url?.startsWith("http")) {
    // The deposit fallback: the engine cannot fetch the URL (HTTP 400).
    const m = await mount(queueOf(card), (method, path) =>
      path === "/corpus/deposit/url" ? { status: 400, body: { detail: "fetch failed" } } : undefined,
    );
    topButtons(m.root).find((b) => label(b) === "Deposit").onclick();
    await flush();
    entry.actions.push({ label: "Deposit", mode: "url-fails", sheets: m.trace.sheets, requests: m.trace.requests, after: after(m.root) });
  }

  // A source card's member beliefs carry their own gestures.
  if (card.kind === "subjectless_source") {
    const probe = await mount(queueOf(card));
    const rows = probe.root.querySelector(".refs").querySelectorAll(".gestures");
    for (const [j, btn] of (rows[0]?.children ?? []).entries()) {
      const m = await mount(queueOf(card), engine(card));
      m.root.querySelector(".refs").querySelectorAll(".gestures")[0].children[j].onclick();
      await flush();
      entry.members.push({ label: btn.textContent, sheets: m.trace.sheets, requests: m.trace.requests, view: dump(m.root.querySelector(".card")) });
    }
    const m = await mount(queueOf(card), engine(card));
    const showAll = m.root.querySelector(".refs").children.find((c) => c.tagName === "BUTTON");
    showAll.onclick();
    await flush();
    entry.showAll = { requests: m.trace.requests, view: dump(m.root.querySelector(".card")) };
  }

  for (const [dir, dx] of [["right", 200], ["left", -200]]) {
    const m = await mount(queueOf(card), engine(card));
    const el = m.root.querySelector(".card");
    el.dispatch("pointerdown", { clientX: 0 });
    el.dispatch("pointermove", { clientX: dx, buttons: 1 });
    el.dispatch("pointerup", {});
    await flush();
    entry.swipe[dir] = { sheets: m.trace.sheets.map((s) => s.title), requests: m.trace.requests, after: after(m.root) };
  }
  golden.cards[card.fixture] = entry;
}

// Error answers, on one plain card (outcomes).
const plain = cards.find((c) => c.key === "stale:errors");
golden.errors = {};
for (const [status, body] of [
  [401, {}],
  [403, { detail: "Belief writes are disabled for store 'default'." }],
  [404, { detail: "No open card with key 'stale:errors'." }],
  [422, { detail: [{ loc: ["body", "particle_id"], msg: "field required" }] }],
  [500, {}],
  [503, { detail: "This engine has no PARTICLES_API_KEY, so it serves reads only." }],
]) {
  const m = await mount(queueOf(plain), () => ({ status, body }));
  topButtons(m.root).find((b) => label(b) === "Still true").onclick();
  await flush();
  golden.errors[status] = { requests: m.trace.requests, view: dump(m.root) };
}

// Queue-level drawing: empty batches, the semantic notice, a refused token.
for (const [name, q] of [
  ["empty-clear", { cards: [], count: 0, open_count: 0 }],
  ["empty-batch-done", { cards: [], count: 0, open_count: 5 }],
  ["semantic-skipped", { ...queueOf(plain), semantic_skipped: true }],
  ["batch-equals-open", queueOf(plain, 1)],
]) {
  const m = await mount(q);
  golden.queue[name] = dump(m.root);
}

// --- Write or check ----------------------------------------------------------

const text = `${JSON.stringify(golden, null, 1)}\n`;
if (process.argv.includes("--check")) {
  const prior = readFileSync(GOLDEN, "utf8");
  if (prior !== text) {
    const a = JSON.parse(prior);
    const diffs = [];
    for (const k of new Set([...Object.keys(a.cards), ...Object.keys(golden.cards)])) {
      if (JSON.stringify(a.cards[k]) !== JSON.stringify(golden.cards[k])) diffs.push(`card ${k}`);
    }
    for (const sect of ["errors", "queue"]) {
      if (JSON.stringify(a[sect]) !== JSON.stringify(golden[sect])) diffs.push(sect);
    }
    console.error(`golden: the feed's output changed: ${diffs.join(", ") || "header"}`);
    process.exit(1);
  }
  console.log(`golden: unchanged (${Object.keys(golden.cards).length} cards)`);
} else {
  writeFileSync(GOLDEN, text);
  console.log(`golden: wrote ${GOLDEN} (${Object.keys(golden.cards).length} cards)`);
}
