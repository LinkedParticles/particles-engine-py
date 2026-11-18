/*
 * A demotion card records a ruling on a replacement: affirm says
 * the replacement was right, dismiss that both claims hold. Its buttons and
 * boxes say so (the owner's live run, 2026-11-07: "Still true" read as the
 * opposite ruling).
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { type CurationCard, cardView } from "../src/index.ts";
import { loadCards } from "./harness.mts";

const DEMOTION = loadCards().find((c) => c.kind === "demotion") as CurationCard;

test("a demotion card's buttons name the ruling each records", () => {
  const view = cardView(DEMOTION, { readOnly: false });
  const byGesture = Object.fromEntries(view.actions.map((a) => [a.gesture, a.label]));
  assert.equal(byGesture.affirm, "Replacement is right");
  assert.equal(byGesture.dismiss, "Both hold");
  const help = Object.fromEntries(view.help.map((h) => [h.label, h.hint]));
  assert.match(help["Replacement is right"], /^The replacement was right\./);
  assert.match(help["Both hold"], /^Both claims hold\..*stays retired/);
});

test("its two boxes say which is retired and which replaced it", () => {
  const view = cardView(DEMOTION, { readOnly: false });
  const order = view.refs.map((r) => (r.type === "side" ? r.label : r.type === "brief" ? r.brief.particleId : r.type));
  assert.deepEqual(order.slice(0, 4), ["Retired", DEMOTION.particle_ids![0], "Replacement", DEMOTION.particle_ids![1]]);
});
