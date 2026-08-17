# Keeping a Particles store healthy

Use this when you are checking the store's condition, or when a read turned up
something that looks wrong.

## The three health surfaces

| Tool | Answers |
|---|---|
| `lint` | What is structurally wrong right now — stale claims, broken provenance, retraction cascades, and (with the semantic pass) contradictions between sources. |
| `quality_report` | What the store looks like in aggregate — counts by status, extractor, and source. Use it to notice drift, not to fix anything. |
| The curation queue | What is worth fixing *first* — a finite, leverage-ranked list of cards, each naming the gestures that resolve it. |

Prefer the curation queue over raw lint output when you are deciding what to
act on. Lint tells you everything that is wrong; the queue tells you which of
those actually matters, and it is bounded so it can be finished.

## A contradiction is not a bug to fix

When two sources disagree, the store records the disagreement rather than
silently picking a winner. That record is the product working.

**Ruling on it is the operator's decision, not yours.** The review step turns a
contradiction into a reusable source-trust policy — a judgment about which
source to believe in general — and that judgment belongs to the person who owns
the store. Your job is to *surface* it clearly: say that the claims conflict,
say what each one asserts, say where each came from. Do not resolve it by
asserting the version you prefer, and do not retract the one you disagree with.

## Stale is not wrong

A belief that has aged out of relevance sinks in ranking; one whose validity
window has expired is flagged. Neither means the claim was false — it means it
may no longer hold. Say "this was recorded in March and may be out of date"
rather than treating it as an error, and check whether a newer belief
supersedes it before you act.

## What you should do versus escalate

**Do:** run the health checks; report what they found in plain language;
surface contradictions and staleness in your answers; deposit a source that
would resolve a gap.

**Escalate:** anything that retires another principal's belief. Retracting a
claim someone else asserted, merging two beliefs, resolving a contradiction,
or changing how much a source is trusted are operator gestures. Name what you
found and what would fix it, then stop.

## Canonical documentation

- Lint findings and the review loop —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/operator-guide/lint-and-review.md>
- The first-run memory audit and the curation queue —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/claude-code.md>
- Concepts (status, contestedness, decay) —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/concepts.md>

If this file and those pages disagree, the pages are right — report the drift.
