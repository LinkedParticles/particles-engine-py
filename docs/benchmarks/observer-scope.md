# Project observer scope: does a belief vanish from its own project?

When one memory store serves several projects and a session reads it through
its project's observer (`claude_code.observer_scope: project`), the
write path still runs across the whole store, so a belief project A states can
be retired by something project B did, and A's session then sees neither its
own claim nor the one that replaced it. The ADR named two such mechanisms and
declined to flip the default until the rate was measured; the dogfood
store, holding one real project, cannot produce that number.
`particles benchmark observer` does.

## What it does

A seeded generator builds two repositories, `alpha` and `beta`, whose
`MEMORY.md` files talk about the same generic subjects (*the repository*, *the
test suite*, *the deploy pipeline*, *the release process*) and evolve
independently over `--days`. Three kinds of line:

| Kind | Example | What it exercises |
|---|---|---|
| `fact` | "The default branch of the repository is `main`." | A value for a slot both projects have. Values are chosen independently, so the projects often disagree. |
| `rule` | "Every commit to the repository needs a sign-off." | Slot-less and byte-identical wherever it appears. Stated by both projects it becomes **one** particle. |
| `own` | "The build of project alpha takes seven minutes." | A subject only this project has. The control. |

Each day, every project whose file changed is harvested exactly as the
SessionEnd hook would harvest it (`LOCAL_MARKDOWN`, `MUTABLE`, a `file://`
URI, the `claude-code` and `project:` tags), then extracted through the
**real** ingest pipeline (rung 2.5, duplicate suppression, the generation
cascade, all of it) with perception scripted: an extractor that emits one
candidate per line, and a contradiction probe that answers from the slot
table. Every other LLM purpose is refused and counted, so a run makes zero
calls. Two global lines are deposited by hand, keyless, as the lines every
observer should see.

At the end of every day, for each project, every line its file currently
states is checked through that project's observer. If it is not in view, the
particle holding it says why:

- **`superseded_by_update`**: rung 2.5 paired it with the
  other project's value for the same subject, and the other project's was the
  later deposit. In `single` store mode two memory files are one lineage:
  `file://` URIs have no authority, both are `LOCAL_MARKDOWN`, neither carries
  an author.
- **`cascade`**: the generation cascade. Both projects stated the
  line, it became one particle, the other project dropped it, and the new
  snapshot of the other project's entry retired the particle for both.
- **`active_elsewhere`**: the store holds the claim ACTIVE, but the surviving
  particle is attested only by the other project's sources: this project's own
  copy was retired in an earlier round of supersession, and its restatement
  has not been re-deposited since. The store believes it, this project's file
  says it, and the lens hides it. This one the ADR did not anticipate; it is
  the only cost the lens adds over the store-wide view.

Beside each line the same check runs without the lens (is the claim ACTIVE
anywhere?), and the lens's own correctness is checked (does a project see
anything it never stated?). Rates carry their denominators.

## Result (2026-09-21)

Twenty seeds, fourteen days, `single` store mode, the real embedding model,
zero LLM calls. Full report: [`observer-scope-2026-09-21.md`](observer-scope-2026-09-21.md).

| Measure | Rate |
|---|---|
| Own lines in view for their project | 79.4% (3,393 / 4,273) |
| … the same lines ACTIVE anywhere (no lens) | 80.2% (3,428 / 4,273) |
| Own lines retired by the other project's activity | 20.6% (880 / 4,273) |
| Winner in view after a cross-project supersession | 0.0% (0 / 810) |
| Lines in view a project never stated (lens leak) | 0.0% (0 / 3,393) |
| In view, `fact` lines | 62.4% (1,397 / 2,240) |
| In view, `rule` lines | 97.5% (1,436 / 1,473) |
| In view, `own` lines | 100.0% (560 / 560) |

Causes: `superseded_by_update` 810 · `cascade` 35 · `active_elsewhere` 35.

## Reading the numbers

**A shared-subject fact is out of view for its own project 37.6% of the
time.** The mechanism is a ping-pong. Day one, alpha says `main` and beta says
`master`; beta's deposit is later, so rung 2.5 retires alpha's line. Whenever
alpha's file next changes for any reason, its re-emitted `main` is a fresh
candidate (the retired copy is not in the duplicate index) and it retires
beta's `master`, whose next re-deposit retires it back. Each project holds
its value for roughly the intervals between the other project's deposits.
The winner is **never** in view for the loser (0 of 810), so under the lens
the loser sees nothing for that slot; without the lens it sees the other
project's value. Neither is what the loser's own file says.

**A shared rule is out of view 2.5% of the time**, for exactly the window
between the other project dropping it and this project's next re-deposit.
This is the cascade acting on a particle two entries attest, which is a
defect on its own terms: the entry that still attests the line was never
asked.

**The control holds and the lens is correct.** Lines about a subject only one
project has never vanish, and no project sees a line it never stated.

**The lens itself costs 0.8 points.** Compare the first two rows: nearly every
own line the lens hides is one the store had already retired. The difference,
35 `active_elsewhere` checkpoints, is the lens's whole marginal cost, and it
is the price of "attested in" being read from the *surviving* particle. A
re-deposit of the stating project folds its attestation onto that particle
and the line comes back.

**What the world does not represent.** Every fact slot here is shared by both
projects and valued independently, a worst case for overlap. Real projects
overlap on generic subjects less, and the rate scales with that overlap; the
ping-pong only starts once they do.

## Result after observer-aware reconciliation (2026-09-23)

The write path now reconciles a pair only when every project that currently
observes the existing claim also observes the update, a project
observes a claim only while one of its sources currently states it, and the
generation cascade retires only what no current source states. The fixture
gained what the first run could not see: a **`chunked` arm**, whose extractor
sends two lines per chunk through the real carry-forward so an unchanged line
is carried rather than re-emitted; a **write census**, which attributes every
retirement as own or cross-project from the retired claim's scope just before
the extraction that retired it; and, after the last day, **one global line
contested by each project**. The generator is unchanged, so every world is the
first run's world. Full reports: [`observer-scope-2026-09-23.md`](observer-scope-2026-09-23.md).

| Measure | First run (2026-09-21) | `lines` arm | `chunked` arm |
|---|---|---|---|
| Own lines in view for their project | 79.4% | **100.0%** (4,273 / 4,273) | **100.0%** (4,273 / 4,273) |
| … the same lines ACTIVE anywhere (no lens) | 80.2% | 100.0% | 100.0% |
| In view, `fact` lines | 62.4% | 100.0% (2,240) | 100.0% (2,240) |
| In view, `rule` lines | 97.5% | 100.0% (1,473) | 100.0% (1,473) |
| In view, `own` lines | 100.0% | 100.0% (560) | 100.0% (560) |
| Lines in view a project never stated (lens leak) | 0 | 0 | 0 |
| Cross-project supersessions | 810 checkpoints | **0** | **0** |
| Cross-project cascade retirements | — | 0 | 0 |
| Own-value supersessions (`SUPERSEDED_BY_UPDATE`) | — | 63 | 63 |
| Declined pairs / with a `CONTRADICTS` relation | — | 936 / 936 | 262 / 262 |
| Distinct `CONTRADICTS` relations | — | 157 | 157 |
| Global line contested by a project → review | — | 40 / 40 | 40 / 40 |
| Chunks carried forward | — | — | 926 of 1,432 |

**Each part, measured alone.** The same twenty seeds after each slice of the
change, own lines in view (`fact` / `rule`), with the cross-project
supersessions and cascades the census attributed:

| After | `lines` arm | `chunked` arm |
|---|---|---|
| §1 + §5 (re-observation recorded; identical files are two entries) | 79.4% (62.4 / 97.5) · 647 + 27 | 71.6% (58.7 / 80.7) · 545 + 57 |
| + §3 (observed means currently stated) | 79.4% (62.4 / 97.5) · 647 + 27 | 71.6% (58.7 / 80.7) · 545 + 57 |
| + §4 (the cascade retires only unattested claims) | 80.3% (62.4 / 100.0) · 647 + 0 | 74.7% (58.7 / 90.0) · 529 + 0 |
| + §2 (the precondition) | **100.0% (100.0 / 100.0) · 0 + 0** | **100.0% (100.0 / 100.0) · 0 + 0** |

The first row reproduces the first run exactly on the `lines` arm (79.4%,
causes 810 / 35 / 35), which is what makes the comparison a like-for-like one.
§3 moves nothing on its own, because until §2 and §4 land a line a project's
file drops is retired by the cascade anyway; it matters once claims survive a
project's edit. On the real dogfood store it is visible at once (below).

**What the chunked arm found.** Before the final slice, the `chunked` arm lost
own lines that only a project's own edits had touched. The carry-forward
lookup matched a chunk through the provenance edge index, whose single row per
(particle, entry) keeps the *first* chunk hash: a claim folded in from another
file, or whose line had moved into a different chunk, was neither carried
forward nor re-emitted, and the cascade then retired a line the file still
stated. A chunk whose text came back after one of its lines had been retired
was a cache hit on the survivors, so the restated line never came back. The
lookup now reads the provenance refs, and a chunk whose claim was retired by
the cascade or an update is re-extracted once. Without the second half the
arm stands at 99.1%.

**Reading it.** The ping-pong is gone: two projects that disagree about a slot
each keep their own value, joined by one recorded contradiction (157 distinct
pairs across the twenty worlds), and a project's own update still retires its
own earlier value (63 times). The lens now costs nothing: the first two rows
are equal, so every line a project states is a line it sees. A project that
contradicts one of the operator's global lines reaches review every time,
where before rung 2.5 retired the operator's line silently.

**Probe spend.** The precondition runs after the probe, so a declined pair is
paid for once per extraction that meets it: 936 declined probes on the `lines`
arm, where every line is re-emitted on every deposit, against 262 on the
`chunked` arm, where unchanged lines are carried forward. The first run paid
the same probes to retire the other project's line instead.

**On the dogfood store** (a copy of 28,245 ACTIVE beliefs, one real project),
the change is small and all of it expected. This repository's observer sees
27,951 beliefs; 294 read as *lapsed*, and they are exactly the five memory
files whose superseded generations the generation backlog sweep has never
retired (`particles corpus refresh --backfill-cascade` would retire 304 there).
The same sweep now retires 304 beliefs instead of 361: 57, in 12 entries, are
still stated by another source's current snapshot. The scope join takes
0.15–0.23 s over every ACTIVE belief (0.17–0.25 s before) and the scoped
digest 2.1–2.2 s (2.05–2.2 s before), well inside the 10 s hook deadline.

## What it decides

The decision record blocks the default flip on a non-trivial rate until the
write path is made observer-aware. 37.6% of shared-subject facts is non-trivial, so
`claude_code.observer_scope` stays `store` by default. The remedy is on the
write path, not the lens: candidacy that does not pair two claims whose
observer scopes are disjoint, which would turn the ping-pong into
two projects each holding their own value, the situation the thesis
describes. The cascade half is a smaller, separate fix.

**2026-09-23.** With the write path observer-aware, every line each project
states is in view on both arms, and no retirement crosses a project. The
blocker the first run found is addressed; flipping `claude_code.observer_scope`
to `project` by default remains the owner's decision, and the
fixture remains a worst case on overlap.

Re-run after any change to rung 2.5, duplicate suppression, carry-forward, the
cascade, the observer precondition or the lens, on both arms:

```bash
uv run particles benchmark observer --days 14 $(for i in $(seq 1 20); do printf -- "--seed %d " $i; done)
uv run particles benchmark observer --days 14 --arm chunked $(for i in $(seq 1 20); do printf -- "--seed %d " $i; done)
```
