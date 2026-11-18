/*
 * A belief of a source card assigned a subject here stays on the card,
 * marked, with Edit to correct it; the card offers Next card once every
 * belief on it is assigned (the owner's live run, 2026-11-07).
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  type ActionResult,
  type CurationCard,
  NEXT_CARD,
  SUBJECT_LABEL,
  afterMemberAction,
  cardView,
  memberCard,
  perform,
} from "../src/index.ts";
import { loadCards, recordingClient, type Trace } from "./harness.mts";

const SOURCE = loadCards().find((c) => c.kind === "subjectless_source") as CurationCard;
const [FIRST, ...REST] = (SOURCE.particles ?? []).map((p) => p.particle_id);

function briefsOf(view: ReturnType<typeof cardView>) {
  return view.refs.filter((r) => r.type === "brief");
}

test("an assigned belief is marked and offers Edit only", () => {
  const view = cardView(SOURCE, { readOnly: false, assigned: { [FIRST]: "Particles" } });
  const [first, second] = briefsOf(view);
  assert.equal(first.type === "brief" && first.assigned, "Subject assigned: Particles");
  assert.deepEqual(first.type === "brief" && first.actions?.map((a) => a.label), ["Edit"]);
  assert.equal(second.type === "brief" && second.assigned, undefined);
  assert.deepEqual(second.type === "brief" && second.actions?.map((a) => a.gesture), ["assign-subject", "supersede", "retract"]);
  assert.notEqual(view.actions[0]?.gesture, NEXT_CARD);
});

test("read-only still shows the mark, with no action", () => {
  const [first] = briefsOf(cardView(SOURCE, { readOnly: true, assigned: { [FIRST]: "Particles" } }));
  assert.equal(first.type === "brief" && first.assigned, "Subject assigned: Particles");
  assert.equal(first.type === "brief" && first.actions, undefined);
});

test("every belief assigned: the card offers Next card, which sends nothing", async () => {
  const card: CurationCard = { ...SOURCE, particle_ids: [FIRST, REST[0]], particles: SOURCE.particles?.slice(0, 2) };
  const view = cardView(card, { readOnly: false, assigned: { [FIRST]: "A", [REST[0]]: "B" } });
  assert.deepEqual(view.actions.map((a) => [a.gesture, a.label, a.primary]), [[NEXT_CARD, "Next card", true]]);
  const trace: Trace = { forms: [], requests: [] };
  const done = await perform(NEXT_CARD, card, { client: recordingClient(trace, () => undefined), ask: async () => null, reviewerId: "r" });
  assert.deepEqual(done, { resolved: true });
  assert.deepEqual(trace.requests, []);
});

test("assign reports the name it showed, or the typed text", async () => {
  const brief = SOURCE.particles![0];
  for (const [answer, expected] of [
    [{ subject: "11111111-1111-4111-8111-111111111111", [SUBJECT_LABEL("subject")]: "Particles" }, "Particles"],
    [{ subject: "  Particles core " }, "Particles core"],
  ] as const) {
    const trace: Trace = { forms: [], requests: [] };
    const result = await perform("assign-subject", memberCard(SOURCE, brief), {
      client: recordingClient(trace, () => undefined),
      ask: async () => ({ ...answer }),
      reviewerId: "r",
    });
    assert.deepEqual(result, { resolved: true, assigned: expected });
  }
});

test("after an action: assigned stays, anything else leaves, the last one empties the card", () => {
  const assign: ActionResult = { resolved: true, assigned: "Particles" };
  const a = afterMemberAction(SOURCE, FIRST, assign, {});
  assert.deepEqual(a.assigned, { [FIRST]: "Particles" });
  assert.equal(a.card.particle_ids?.length, SOURCE.particle_ids?.length);
  // Edit on the assigned belief retires it: it leaves, and so does its mark.
  const b = afterMemberAction(a.card, FIRST, { resolved: true }, a.assigned);
  assert.deepEqual(b.assigned, {});
  assert.ok(!b.card.particle_ids?.includes(FIRST));
  assert.equal(b.empty, false);
  const kept = afterMemberAction(SOURCE, FIRST, { resolved: false }, { x: "y" });
  assert.deepEqual([kept.card, kept.assigned, kept.empty], [SOURCE, { x: "y" }, false]);
  const lone: CurationCard = { ...SOURCE, particle_ids: [FIRST], particles: SOURCE.particles?.slice(0, 1) };
  assert.equal(afterMemberAction(lone, FIRST, { resolved: true }, {}).empty, true);
});
