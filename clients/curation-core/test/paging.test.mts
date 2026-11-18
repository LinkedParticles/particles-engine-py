/*
 * A source card's beliefs drawn a page at a time, for a surface whose drawing
 * has a size bound: the Claude Code pane drew nothing when Show all asked it
 * for 215 beliefs at once (the owner's live run, 2026-11-08).
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { type CurationCard, cardView } from "../src/index.ts";
import { loadCards } from "./harness.mts";

const SOURCE = loadCards().find((c) => c.kind === "subjectless_source") as CurationCard;

/** The source card with `n` beliefs, every one briefed (as Show all returns it). */
function full(n: number): CurationCard {
  const base = (SOURCE.particles ?? [])[0];
  const ids = Array.from({ length: n }, (_, i) => `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`);
  return { ...SOURCE, particle_ids: ids, particles: ids.map((id, i) => ({ ...base, particle_id: id, content: `Belief ${i}` })) };
}

const claims = (card: CurationCard, start: number, size: number) =>
  cardView(card, { readOnly: false, page: { start, size } })
    .refs.filter((r) => r.type === "brief")
    .map((r) => (r.type === "brief" ? r.brief.claim : ""));
const pager = (card: CurationCard, start: number, size: number) =>
  cardView(card, { readOnly: false, page: { start, size } }).refs.find((r) => r.type === "pager");

test("a page draws its slice of the beliefs, and says where it stands", () => {
  const card = full(45);
  assert.deepEqual(claims(card, 20, 20), Array.from({ length: 20 }, (_, i) => `Belief ${20 + i}`));
  assert.deepEqual(pager(card, 0, 20), { type: "pager", text: "Beliefs 1–20 of 45", start: 0, size: 20, previous: false, next: true });
  assert.deepEqual(pager(card, 20, 20), { type: "pager", text: "Beliefs 21–40 of 45", start: 20, size: 20, previous: true, next: true });
  assert.deepEqual(pager(card, 40, 20), { type: "pager", text: "Beliefs 41–45 of 45", start: 40, size: 20, previous: true, next: false });
});

test("a start past the end shows the last page", () => {
  assert.deepEqual(claims(full(25), 40, 20), ["Belief 20", "Belief 21", "Belief 22", "Belief 23", "Belief 24"]);
  assert.equal(pager(full(25), 40, 20)?.type === "pager" && pager(full(25), 40, 20)?.start, 20);
});

test("one page's worth needs no pager, and the preview still offers Show all", () => {
  const refs = cardView(SOURCE, { readOnly: false, page: { start: 0, size: 20 } }).refs;
  assert.equal(refs.some((r) => r.type === "pager"), false);
  assert.equal(refs.some((r) => r.type === "show-all"), (SOURCE.particle_ids ?? []).length > (SOURCE.particles ?? []).length);
});

test("without a page every briefed belief is drawn", () => {
  const refs = cardView(full(45), { readOnly: false }).refs;
  assert.equal(refs.filter((r) => r.type === "brief").length, 45);
  assert.equal(refs.some((r) => r.type === "pager"), false);
});

test("a card that is not a source card is never paged", () => {
  const other = loadCards().find((c) => c.kind !== "subjectless_source" && (c.particles ?? []).length > 1) as CurationCard;
  const n = (other.particles ?? []).length;
  const refs = cardView(other, { readOnly: false, page: { start: 0, size: 1 } }).refs;
  assert.equal(refs.filter((r) => r.type === "brief").length, n);
});
