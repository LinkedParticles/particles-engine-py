/*
 * The subject search behind a `subject` field: the request it sends, the
 * picks it lists, and when it sends nothing.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ApiError,
  type CurationCard,
  type Form,
  perform,
  SUBJECT_SEARCH_LIMIT,
  SUBJECT_SUGGESTED,
  searchSubjects,
  subjectListing,
  suggestSubjects,
} from "../src/index.ts";
import { recordingClient, type Trace } from "./harness.mts";

const SUBJECTS = [
  { id: "s-1", canonical_name: "Particles", aliases: ["Particles", "particles-engine-py", "the SDK", "Particles SDK"], subject_class: "software", asserted_by: "x" },
  { id: "s-2", canonical_name: "Particles spec", aliases: [], subject_class: null, asserted_by: "x" },
  { canonical_name: "No id", aliases: [], asserted_by: "x" },
];

function client(trace: Trace, body: unknown = SUBJECTS) {
  return recordingClient(trace, (req) => (req.path.startsWith("/subjects/search") ? { status: 200, body } : undefined));
}

test("a search sends one bounded read and lists each subject as a pick", async () => {
  const trace: Trace = { forms: [], requests: [] };
  const picks = await searchSubjects(client(trace), "  partic ");
  assert.deepEqual(trace.requests, [
    { method: "GET", path: `/subjects/search?q=partic&limit=${SUBJECT_SEARCH_LIMIT}&order=relevance` },
  ]);
  assert.deepEqual(picks, [
    { value: "s-1", label: "Particles", detail: "software · also: particles-engine-py, the SDK, Particles SDK" },
    { value: "s-2", label: "Particles spec" },
  ]);
});

test("fewer than two characters search nothing", async () => {
  const trace: Trace = { forms: [], requests: [] };
  assert.deepEqual(await searchSubjects(client(trace), " p "), []);
  assert.deepEqual(trace.requests, []);
});

test("a query is sent encoded", async () => {
  const trace: Trace = { forms: [], requests: [] };
  await searchSubjects(client(trace, []), "a&b=c");
  assert.equal(trace.requests[0].path, `/subjects/search?q=a%26b%3Dc&limit=${SUBJECT_SEARCH_LIMIT}&order=relevance`);
});

test("never more than the limit", async () => {
  const trace: Trace = { forms: [], requests: [] };
  const many = Array.from({ length: 20 }, (_, i) => ({ id: `s-${i}`, canonical_name: `S ${i}`, aliases: [], asserted_by: "x" }));
  assert.equal((await searchSubjects(client(trace, many), "s ")).length, 0);
  assert.equal((await searchSubjects(client(trace, many), "s 1")).length, SUBJECT_SEARCH_LIMIT);
});

// --- Suggestions ---------------------------------------------------

const BELIEF = "00000000-0000-4000-8000-0000000000aa";
const SUGGESTED = [
  { subject: SUBJECTS[0], named: true, neighbours: 3 },
  { subject: SUBJECTS[1], named: true, neighbours: 0 },
  { subject: { id: "s-3", canonical_name: "Gate", aliases: [], subject_class: "concept", asserted_by: "x" }, named: false, neighbours: 1 },
  { subject: SUBJECTS[2], named: false, neighbours: 2 },
];

function suggesting(trace: Trace, status = 200) {
  return recordingClient(trace, (req) =>
    req.path.includes("/subject-suggestions") ? { status, body: status === 200 ? SUGGESTED : { detail: "boom" } } : undefined,
  );
}

test("suggestions are one read for the belief, each pick saying why", async () => {
  const trace: Trace = { forms: [], requests: [] };
  const picks = await suggestSubjects(suggesting(trace), BELIEF);
  assert.deepEqual(trace.requests, [{ method: "GET", path: `/particles/${BELIEF}/subject-suggestions` }]);
  assert.deepEqual(picks, [
    { value: "s-1", label: "Particles", detail: "software · named in the claim · on 3 similar beliefs" },
    { value: "s-2", label: "Particles spec", detail: "named in the claim" },
    { value: "s-3", label: "Gate", detail: "concept · on 1 similar belief" },
  ]);
});

test("a failed suggestion read throws, so the client can say so", async () => {
  const trace: Trace = { forms: [], requests: [] };
  await assert.rejects(suggestSubjects(suggesting(trace, 500), BELIEF), ApiError);
});

test("the field lists suggestions below the search minimum and matches past it", () => {
  const sugg = [{ value: "s-1", label: "Particles" }];
  const found = [{ value: "s-2", label: "Particles spec" }];
  assert.deepEqual(subjectListing("", sugg, undefined), { heading: SUBJECT_SUGGESTED, options: sugg });
  assert.deepEqual(subjectListing(" p ", sugg, found), { heading: SUBJECT_SUGGESTED, options: sugg });
  assert.deepEqual(subjectListing("pa", sugg, found), { options: found });
  assert.equal(subjectListing("pa", sugg, undefined), null);
  assert.equal(subjectListing("", [], undefined), null);
  assert.equal(subjectListing("", undefined, undefined), null);
});

test("the assign form names its belief on the subject field and leaves the claim to `about`", async () => {
  const trace: Trace = { forms: [], requests: [] };
  const card = {
    key: `no_subject:${BELIEF}`,
    kind: "no_subject",
    particle_ids: [BELIEF],
    particles: [{ particle_id: BELIEF, content: "The gate checks run before a commit." }],
  } as unknown as CurationCard;
  await perform("assign-subject", card, {
    client: suggesting(trace),
    reviewerId: "r",
    ask: async (form) => {
      trace.forms.push(form);
      return null;
    },
  });
  const form = trace.forms[0] as Form;
  assert.equal(form.about, "The gate checks run before a commit.");
  assert.ok(!(form.message ?? "").includes("The gate checks"));
  assert.equal(form.fields?.[0].particleId, BELIEF);
});
