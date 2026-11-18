/*
 * A conflict card quotes each claim once, in its box (the owner's live run,
 * 2026-11-09: the diagnostic, the boxes and the resolve form each repeated
 * both claims). The diagnostic gives way to the engine's conflict note, and
 * the resolve form carries the claims as its `about`, which a client whose
 * form covers the card shows and one that draws it under the card leaves out.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { type CurationCard, cardView } from "../src/index.ts";
import { engineFor, loadCards, run } from "./harness.mts";

const CARDS = loadCards();
const NOTED = CARDS.find((c) => c.fixture === "19-inconsistency") as CurationCard;
const UNNOTED = CARDS.find((c) => c.kind === "inconsistency" && c.conflict_note == null) as CurationCard;

test("a conflict card shows its note in place of a diagnostic that quotes its claims", () => {
  assert.equal(cardView(NOTED, { readOnly: false }).diagnostic, "");
  const noted = { ...NOTED, conflict_note: "Claim A was retired by update." };
  assert.equal(cardView(noted, { readOnly: false }).diagnostic, "Claim A was retired by update.");
});

test("an engine that sends no note keeps its diagnostic", () => {
  assert.equal(cardView(UNNOTED, { readOnly: false }).diagnostic, UNNOTED.diagnostic);
});

test("the resolve form names the claims as what it is about, not in its message", async () => {
  const trace = await run("resolve", NOTED, engineFor(NOTED));
  const [form] = trace.forms;
  assert.equal(form.about, "A: “The default port is 8000.”\nB: “The default port is 8080.”");
  assert.equal(form.message, "To decide later, cancel and snooze the card.");
  assert.doesNotMatch(form.message ?? "", /8000|8080/);
});
