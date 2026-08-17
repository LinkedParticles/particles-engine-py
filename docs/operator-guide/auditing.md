# Auditing operator actions

Particles keeps an **append-only operator event log**: every
deliberate operator decision that changes what the knowledge base asserts, how
an entity is identified, or how a source is trusted is recorded as an immutable
event — *who, when, why, and which records it touched*.

This is the durable answer to questions the per-record state can't give you on
its own: *"why was this source retracted? what merged into this subject? what's
the history of trust changes?"* A record's current fields tell you its state
now; the event log tells you the **decisions that produced it**.

## What gets logged

A verb records an event when it is **operator-initiated**, **mutates an
epistemic commitment** (an assertion, an entity's identity, or a trust
judgment), and **lacks a complete durable history of its own**. In practice:

| Verb(s) | Event type |
|---|---|
| `subjects split` | `SUBJECTS_SPLIT` |
| `subjects merge` | `SUBJECTS_MERGED` |
| `subjects alias` | `SUBJECT_ALIASED` |
| `subjects confirm` | `SUBJECT_LINK_CONFIRMED` |
| `subjects unlink` | `SUBJECT_LINK_REMOVED` |
| `subjects set-class` | `SUBJECT_RECLASSIFIED` |
| `trust set`, `trust statement-set`, `trust lens adopt` / `unadopt` | `TRUST_CHANGED` |
| `review` (§9.6 resolution) | `REVIEW_RESOLVED` |
| `links add` | `RELATION_ADDED` |
| `links remove` | `RELATION_REMOVED` |
| `particle tag` | `PARTICLE_TAGGED` |
| `particle untag` | `PARTICLE_UNTAGGED` |
| `corpus retract` | `SOURCE_RETRACTED` |
| `particle retract` | `PARTICLE_RETRACTED` (actor `cli:particle-retract`) |
| `memory useful` | `BELIEF_MARKED_USEFUL` (actor `cli:memory-useful` / `http:/memory/useful`) |
| *(system-emitted)* §6.6 trust resolution drops a candidate | `CONFLICT_CANDIDATE_DROPPED` |

Read-only and pipeline commands (`query`, `lint`, `extract`, `reindex`,
`export`, `deposit`) do **not** log — they have their own provenance. One
deliberate exception: when the §6.6 ladder's
`SUPERSEDED_BY_EXISTING` verdict drops a freshly extracted candidate in favour
of a strictly higher-trust existing claim, the candidate is never persisted, so
the event log is the *only* durable record of it — the
`CONFLICT_CANDIDATE_DROPPED` event carries the candidate excerpt, the verdict,
and the winning particle id. An empty event list for a record otherwise
honestly means *no operator action touched it*.

State-attached rationale is **kept on its record, not relocated**:
`trust.basis` stays on the trust statement (its standing justification) and the
event snapshots it, so the change history survives the next overwrite.

## Reading the log

The log is exposed identically on all three front-ends.

**CLI**

```bash
# Newest first; filter by a record it touched, or by type
particles events list
particles events list --subject <subject-id>
particles events list --particle <particle-id>
particles events list --type TRUST_CHANGED --limit 20

# One event in full (header + refs + payload)
particles events show <event-id>
```

`--particle`, `--subject`, and `--entry` are mutually exclusive (each filters
to events that touched that record).

**HTTP**

```
GET /events?subject=<id>&type=SUBJECTS_MERGED&limit=20
GET /events/{event_id}
```

**MCP**

The read-only `events_list` and `event_show` tools expose the same data to an
agent, so an assistant can introspect the audit trail in its loop.

## A source was retracted — what now?

When a publisher *retracts* an article you've deposited, you want your derived
claims marked `RETRACTED` **without** erasing the source — the question *"what
did we believe before the retraction?"* must stay answerable. Use
`corpus retract`, **not** `corpus delete`:

```bash
# Preview: which particles would flip, what's skipped, snapshots preserved
particles corpus retract <entry-id> --dry-run

# Apply, recording why
particles corpus retract <entry-id> --reason "NYT issued a correction 2026-05-30"

# Cascade staleness to anything that cited those now-retracted claims
particles lint
```

`corpus retract` transitions every live particle (ACTIVE / INCONSISTENCY) from
the source to `RETRACTED` with reason `SOURCE_RETRACTED`, records a
`SOURCE_RETRACTED` event carrying your `--reason` and the affected particle
ids, and **leaves the corpus entry, its snapshots, and the particles intact**.
It is idempotent (a second run finds nothing live) and skips particles that are
already `SUPERSEDED` / `PROVENANCE_STALE` / `RETRACTED`. It does *not* itself
cascade — the follow-up `particles lint` flags downstream particles
`PROVENANCE_STALE`.

Reach for `corpus delete` only when the source should be erased entirely (a
privacy request or a mistaken deposit) — that path destroys the audit trail by
design.

## One belief went stale — retiring just that one

`corpus retract` is the right tool when a *source* is bad. When one claim out of
a source's dozen has gone stale — a memory file whose other notes are still true,
an extracted rule that a later edit reversed — it is far too broad, and flipping
`mcp.write.allow_cross_asserter` to let an agent reach the row is worse: a
standing grant widening what every future agent session may mutate, to fix one
belief. `particles particle retract` is the narrow instrument:

```bash
particles particle retract 20a27e4a --reason "Superseded by the AGENTS.md rule forbidding the PATH prefix" --dry-run
```

```
  20a27e4a-3f76-4f79-af6f-ef38ea693d01
  status      ACTIVE
  asserted by general-extractor
  confidence  1.00
  content     `pre-commit` can be found by prefixing `PATH="$PWD/.venv/bin:$PATH"`.
  reason      Superseded by the AGENTS.md rule forbidding the PATH prefix

--dry-run: nothing written.
```

Drop `--dry-run` to apply it. The target is printed and confirmed first — you
identified it by eight characters, so see what you are about to retire; `--yes`
skips the prompt for scripted use.

```
Retracted 20a27e4a… (EXPLICIT_RETRACTION); reason recorded.
Run `particles lint` to cascade PROVENANCE_STALE to dependents.
```

The belief becomes `RETRACTED` with reason `EXPLICIT_RETRACTION`, carries the `retired_at` stamp, and lands in the log under its own actor:

```
2026-07-25 00:30  dc6e4ea1…  PARTICLE_RETRACTED  by cli:particle-retract
    reason: Superseded by the AGENTS.md rule forbidding the PATH prefix
    refs:   particle:20a27e4a…
```

Three properties worth knowing:

- **It runs under *operator* authority, not agent policy.** It works with no
  store MCP-write-enabled and `allow_cross_asserter` left `false` — the guardrail governs what *agents* may mutate and is untouched. The CLI is your
  own hands on your own store.
- **An operator-asserted belief is still refused.** A particle whose confidence
  came from `HUMAN_REVIEW` is not retractable this way; revising one is
  [Review's](lint-and-review.md) job.
- **Idempotent by the guard, not by a special case.** A second run reports
  `Particle '…' is RETRACTED, not ACTIVE` and exits 1, writing nothing.

There is no un-retract verb: the §6.6 transition table has no
`RETRACTED → ACTIVE` edge, so recovery means re-extraction or a fresh assertion.
The confirmation prompt is the guard.

## Notes

- The log is **append-only** — there is no edit or delete path.
- `actor` records the interface entry-point (the CLI verb today); it becomes
  the authenticated principal when multi-user lands.
- Growth is unbounded by design; a retention/compaction policy is a future
  operator-side concern (§ Deferred).
