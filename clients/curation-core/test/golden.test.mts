/*
 * The core reproduces the web UI as it was before the core existed
 * (step 2): every form, every request and every outcome in
 * fixtures/golden.json, and every string each card drew, in order.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import {
  type CardView,
  type CurationCard,
  cardView,
  emptyText,
  failureOf,
  memberCard,
  queueHeader,
  readOnlyText,
  SEMANTIC_SKIPPED_TEXT,
  swipeGesture,
} from "../src/index.ts";
import { FIXTURES, engineFor, loadCards, recordingClient, run, type Respond, type Trace } from "./harness.mts";

type Dump = string | [string, string, Record<string, unknown>, ...Dump[]];

const golden = JSON.parse(readFileSync(join(FIXTURES, "golden.json"), "utf8"));
const cards = loadCards();

/** Every text node under a dump, in document order, without empty strings. */
function texts(d: Dump): string[] {
  if (typeof d === "string") return d ? [d] : [];
  return d.slice(3).flatMap((c) => texts(c as Dump));
}

function find(d: Dump, cls: string): Dump | null {
  if (typeof d === "string") return null;
  if (d[1].split(" ").includes(cls)) return d;
  for (const c of d.slice(3)) {
    const hit = find(c as Dump, cls);
    if (hit) return hit;
  }
  return null;
}

function findAll(d: Dump, cls: string, out: Dump[] = []): Dump[] {
  if (typeof d === "string") return out;
  if (d[1].split(" ").includes(cls)) out.push(d);
  for (const c of d.slice(3)) findAll(c as Dump, cls, out);
  return out;
}

/** The strings a card view carries, in the order the web UI draws them. */
function viewTexts(v: CardView): string[] {
  const out: string[] = [v.title];
  if (v.leverage) out.push(v.leverage);
  if (v.question) out.push(v.question);
  out.push(v.diagnostic);
  if (v.evidence) out.push(v.evidence.label);
  for (const r of v.refs) {
    switch (r.type) {
      case "brief":
        out.push(r.brief.claim);
        if (r.brief.meta) out.push(r.brief.meta);
        if (r.brief.reason) out.push(r.brief.reason);
        if (r.brief.source) out.push("source: ", r.brief.source.text);
        for (const a of r.actions ?? []) out.push(a.label);
        break;
      case "side":
      case "show-all":
        out.push(r.label);
        break;
      case "url":
        out.push(r.url.text);
        break;
      case "break":
        break;
      default:
        out.push(r.text);
    }
  }
  for (const a of v.actions) {
    out.push(a.label);
    if (a.note) out.push(a.note);
  }
  if (v.help.length) {
    out.push("What do these do?");
    for (const h of v.help) out.push(h.label, h.hint);
  }
  return out.filter(Boolean);
}

function goldenTrace(t: Trace) {
  return { sheets: t.forms, requests: t.requests };
}

function outcomeOf(after: { banners: [string, string][]; card: string | null }) {
  const info = after.banners.filter(([c]) => c.includes("info")).map(([, t]) => t);
  return { resolved: after.card === null, notice: info[0] };
}

for (const card of cards) {
  const g = golden.cards[card.fixture];

  test(`${card.fixture}: the view carries the drawn strings in order`, () => {
    const drawn = texts(find(g.view, "card") as Dump);
    assert.deepEqual(viewTexts(cardView(card, { readOnly: false })), drawn);
  });

  test(`${card.fixture}: each action asks, sends and ends as before`, async () => {
    const view = cardView(card, { readOnly: false });
    for (const a of g.actions) {
      const action = view.actions.find((x) => x.label === a.label);
      assert.ok(action, `no action labelled ${a.label}`);
      if (a.disabled) {
        assert.equal(action.enabled, false);
        continue;
      }
      let respond: Respond = engineFor(card);
      if (a.mode === "url-fails") {
        respond = (req) =>
          req.path === "/corpus/deposit/url" ? { status: 400, body: { detail: "fetch failed" } } : undefined;
      }
      const trace = await run(action.gesture, structuredClone(card), respond, a.mode === "cancel" ? "cancel" : "answer");
      assert.equal(trace.error, undefined, `${a.label}: ${String(trace.error)}`);
      assert.deepEqual(goldenTrace(trace), { sheets: a.sheets, requests: a.requests }, `${a.label} ${a.mode}`);
      const want = outcomeOf(a.after);
      assert.equal(trace.resolved, want.resolved, `${a.label} ${a.mode} resolved`);
      assert.equal(trace.notice, want.notice, `${a.label} ${a.mode} notice`);
    }
  });

  test(`${card.fixture}: swipes stand for the same gestures`, async () => {
    for (const dir of ["right", "left"] as const) {
      const gesture = swipeGesture(card, dir, false);
      const trace =
        gesture && gesture !== "advance"
          ? await run(gesture, structuredClone(card), engineFor(card))
          : { forms: [], requests: [] };
      assert.deepEqual(
        { sheets: trace.forms.map((f) => f.title), requests: trace.requests },
        { sheets: g.swipe[dir].sheets, requests: g.swipe[dir].requests },
        dir,
      );
    }
  });

  if (card.kind === "subjectless_source") {
    test(`${card.fixture}: a member belief acts on its own key`, async () => {
      const first = (card.particles ?? [])[0];
      const view = cardView(card, { readOnly: false });
      const brief = view.refs.find((r) => r.type === "brief");
      assert.ok(brief && brief.type === "brief" && brief.actions);
      const actions = brief.actions;
      for (const [i, m] of (g.members as { label: string; sheets: unknown; requests: unknown }[]).entries()) {
        const action = actions[i];
        assert.equal(action.label, m.label);
        const trace = await run(action.gesture, memberCard(card, first), engineFor(card));
        assert.deepEqual(goldenTrace(trace), { sheets: m.sheets, requests: m.requests });
      }
    });

    test(`${card.fixture}: show all fetches the whole card`, async () => {
      const trace: Trace = { forms: [], requests: [] };
      await recordingClient(trace, engineFor(card)).curationCard(card.key);
      assert.deepEqual(trace.requests, g.showAll.requests);
    });
  }
}

test("engine failures are reported as before", async () => {
  const plain = cards.find((c) => c.key === "stale:errors") as CurationCard;
  for (const [status, body] of [
    [401, {}],
    [403, { detail: "Belief writes are disabled for store 'default'." }],
    [404, { detail: "No open card with key 'stale:errors'." }],
    [422, { detail: [{ loc: ["body", "particle_id"], msg: "field required" }] }],
    [500, {}],
    [503, { detail: "This engine has no PARTICLES_API_KEY, so it serves reads only." }],
  ] as const) {
    const trace = await run("affirm", structuredClone(plain), () => ({ status, body }));
    const g = golden.errors[status];
    assert.deepEqual(trace.requests, g.requests);
    const banners = findAll(g.view, "banner").map((b) => texts(b).join(""));
    const failure = failureOf(trace.error);
    const shown = failure.kind === "read-only" ? readOnlyText(failure.reason) : failure.kind === "error" ? failure.text : "";
    assert.deepEqual(banners, [shown], `HTTP ${status}`);
  }
});

test("the queue's own words are as before", () => {
  const header = (d: Dump) => texts(find(d, "session-count") as Dump).join("");
  assert.equal(header(golden.queue["empty-clear"]), queueHeader(0, 0));
  assert.ok(texts(golden.queue["empty-clear"]).includes(emptyText(0)));
  assert.equal(header(golden.queue["empty-batch-done"]), queueHeader(0, 5));
  assert.ok(texts(golden.queue["empty-batch-done"]).includes(emptyText(5)));
  assert.equal(header(golden.queue["batch-equals-open"]), queueHeader(1, 1));
  assert.equal(header(golden.queue["semantic-skipped"]), queueHeader(1, 9));
  assert.ok(texts(golden.queue["semantic-skipped"]).includes(SEMANTIC_SKIPPED_TEXT));
});
