# Recording a belief in Particles

Use this when you are about to write something into the Particles store.

## Nothing is ever deleted

The store is append-only. You do not edit a belief and you do not remove one.
A correction **supersedes**; a withdrawal **retracts**; a disagreement stays
**disputed** in the open. All three are status changes that keep the original
readable, with its provenance intact. If you catch yourself wanting to "fix" a
belief in place, the operation you want is `particle_supersede`.

## Which verb

| You have | Use | Why |
|---|---|---|
| Raw material worth keeping, but no claim yet | `deposit_text` | Cheap archival, zero extraction. Returns a `corpus_entry_id` a later assertion can cite. |
| One specific claim you can state in a sentence | `particle_assert` | One belief, one sentence, with its own confidence and provenance. |
| A claim that replaces one already in the store | `particle_supersede` | Retires the predecessor and asserts the successor with a `supersedes` pointer, in one transaction. This is the right verb for "the world changed" *and* for "we were wrong". |
| A belief that should no longer be believed at all | `particle_retract` | Retires it with a reason. The record stays; the belief stops answering. |

One claim per assertion. If your sentence contains "and" joining two
independent facts, it is two particles — a compound assertion cannot be
superseded, contested, or retracted by half.

## Your writes rank below the operator's

Agent-asserted beliefs carry a lower trust weight than operator-asserted ones,
by design. This is not a penalty; it is what lets an operator's correction win
over your assertion without anyone editing anything. Two consequences worth
internalising:

- **Record faithfully, do not arbitrate.** If two sources disagree, assert what
  each says. Do not pick a winner and assert only that — the store has a
  contradiction surface for exactly this, and collapsing it early destroys the
  signal the operator needs.
- **Do not restate an existing belief to raise its standing.** Duplicate claims
  are detected and collapsed as co-evidential; asserting the same thing twice
  does not make it more true, it makes the curation queue longer.

## Capture is mostly not your job

Memory is harvested outside your control: session transcripts and changed
memory files are deposited and extracted automatically. You do **not** need to
remember to record things — that failure mode is exactly what the harvest
exists to avoid.

The explicit deposit verb is an escape hatch for "record this one, now", not a
capture strategy. Reach for it when the operator asks you to remember something
specific, or when a decision was reached that the transcript alone would state
ambiguously. Otherwise, let the harvest do it.

## If the store is not set up yet

If the write tools are absent or every read comes back empty, the integration
probably is not installed. You can install it yourself:

```
particles init claude-code --json
```

It merges the memory hooks into the harness settings, creates and enables a
`memory` store if none exists, and prints a JSON result: what it created, and a
`next_steps` list of what is left for the human. Report those next steps rather
than acting on them — the first-run memory audit in particular is the
operator's call, because it reads their existing memory directory.

This installs the *local* integration only. It mints no credential and grants
itself no new access: the write allowlist is unchanged, and anything you write
still lands at agent trust.

## Canonical documentation

- Concepts, statuses, and the supersession model —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/concepts.md>
- The Claude Code memory loop (what gets harvested, and when) —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/claude-code.md>

If this file and those pages disagree, the pages are right — report the drift.
