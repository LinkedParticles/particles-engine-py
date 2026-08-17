# Tuning

The single number queries actually rank by is **`effective_confidence`**:

```
effective_confidence = confidence.value
                     × extractor_trust_weight
                     × source_trust_rank
                     × recency_factor
```

Four things you tune.

## Extractor calibration

`confidence.value` is stored **as calibrated at extraction time**
: when the extractor carries an active calibration, the
per-extractor temperature is applied to the raw
self-reported value before storage — `sigmoid(logit(raw) / T)`, logit-space
and the particle is stamped `CALIBRATED_BENCHMARK` with a
`calibration_ref` audit trail. The calibration is built from benchmark suites:

```bash
uv run particles extractor calibrate <extractor-id>
```

Three things to know before you run it:

* **It reads its own suite family, not the benchmark suites.** The default is
  `tests/benchmark/calibration/`, whose gold sets deliberately name
  only *part* of what an extractor emits so the fit sees both a correct and an
  incorrect label. A §13.3 suite under `tests/benchmark/suites/` wants the
  opposite — total gold coverage, or it scores its extractor down on precision
  — which is exactly the all-matched label set a temperature cannot be fitted
  from. `extractor benchmark` never reads the calibration directory, and
  `extractor calibrate` never reads the benchmark one.
* **The verb refuses to store a fit it cannot believe**, and prints why. Four
  conditions: degenerate labels (everything matched, or nothing did), a
  temperature on an optimizer bound, fewer than two distinct *movable*
  confidences, and a fit that does not actually reduce calibration error. The
  first is a property of the suite and you can fix it by authoring a partial
  gold set; **the last two are properties of the extractor** and no suite will
  clear them. An extractor that states `1.0` on most of its output has almost
  nothing temperature scaling can act on — those values are exact fixed points,
  so they are dropped from the fit and the run reports how many.
* **Any calibration fitted before v1.115.0 is inert.** Those fits were built
  on a labelling bug that forced every one of them to the bound; they are
  reported as `NOT APPLIED` by `particles extractor calibrations
  <extractor-id>` and the affected pairings mint `EXTRACTOR_DIRECT` particles
  until re-fitted. Particles already stored keep the confidence they were
  minted with — `particles reindex --extractor-id <id>` re-mints
  them.

The scalar is stored in the `extractor` table. If an extractor
systematically overconfidences, calibration pulls it back. If it
underconfidences, calibration pushes it up.

Two timing consequences:

- **Calibrate extractors *before* extraction volume accumulates.**
  Cross-extractor confidence comparability depends on each particle
  being stored under a fitted calibration; particles extracted before
  the fit keep their raw values forever.
- **Recalibration propagates only through re-extraction.** A particle
  keeps the calibration in force at its creation; re-running
  `extractor calibrate` affects future extractions only (and chunk-hash
  carry-forward deliberately skips unchanged chunks on reindex). For a
  live, store-wide correction of an extractor you no longer trust, use
  `trust_weight` below — that is the read-time lever.

## Extractor trust weight

Per-extractor multiplier in `[0.0, 1.0]`, stored in the `extractor`
table. Default 1.0. Drop a misbehaving extractor's weight to
demote its claims without dropping them entirely:

```bash
uv run particles extractor trust-set <extractor-id> 0.5
```

The design forbids *increasing* trust weights above the SDK default
(demotion-only) — the SDK is meant to ratchet trust down based on
observed inconsistencies, not up based on hope.

### Conformance trust cap

Opt-in (default **off**): an extractor whose last `extractor conform`
run showed a *genuinely evaluable* REQUIRED failure has its **effective**
trust weight clamped at query time — closing the loop between the conformance report and the live trust lever, without touching
the stored `trust_weight` and without making conformance a CI gate
(report-only stance is preserved).

```yaml
conformance:
  trust_cap:
    enabled: true       # default false → byte-for-byte current behaviour
    cap_value: 0.5      # failing extractors' effective weight clamped to this
    exempt: []          # extractor ids the cap never applies to
```

Two properties make it safe to enable on a partial fixture corpus:

- **Unknown ≠ failed.** An extractor with no fixture reports "0 particles
  → REQUIRED at 0 %"; that is *unevaluated*, not a failure, and never
  clamps. Only an extractor that actually ran fixtures and missed a
  REQUIRED field is demoted — so enabling the cap does **not** clobber
  the extractors that lack fixtures.
- **Self-healing.** The clamp is recomputed every read from the stored
  verdict. Fix the extractor, re-run `extractor conform`, and the next
  run records a pass → the clamp lifts automatically. Run
  `particles extractor conform <id>` to (re)record an extractor's verdict;
  it writes the status best-effort and still works with no store (CI).

An operator who disagrees with a specific clamp adds the extractor to
`exempt` (or disables the block) — an auditable config act.

## Source trust rank

Per-*source* multiplier in `[0.0, 1.0]`, evaluated at query time from
your trust statements and URL rules. Where extractor trust
demotes a *tool*, source trust demotes a *publisher*:

```bash
# Demote everything from one domain
uv run particles trust set sketchy.example 0.2

# Or a URL-pattern modifier stacked on top
uv run particles trust set --modifier '/sponsored/' -- -0.3
```

A source with **no** applicable statement or rule is strictly neutral
(factor 1.0) — silence is not distrust, and an unconfigured store ranks
exactly as if the factor didn't exist. The same statements feed the
§6.6 conflict ladder at extraction time; since they also bite
at query, export, and federated-query time — in federated queries the
**viewer's** rules apply to every store's candidates.

`SourceTrustStatement`s take precedence over URL rules within their
domain, resolved by a four-tier cascade (first match wins):
CORPUS_ENTRY-scoped → AUTHOR-scoped → SOURCE_TYPE-scoped → the URL
baseline above. Review's PREFER rulings write CORPUS_ENTRY-scoped
statements automatically; the other scopes are written directly with
`particles trust statement-set` — e.g. trusting (or demoting)
everything by one author:

```bash
particles trust statement-set numismatics AUTHOR some-author 0.9 \
    --basis "domain expert, verified track record"
```

### Lenses — adopting someone else's trust policy

Instead of hand-building every rule, you can adopt a published **lens**
 — a community's bundled trust policy, deposited as a
`TrustLensDefinition` JSON:

```bash
particles deposit ./acme-numismatics.json   # publish / materialise
particles trust lens list
particles trust lens adopt acme-numismatics
```

An adopted lens's statements, URL rules, and extractor-weight overrides
compose into the same query-time factor. Two rules govern composition:
your **local rules always win** per key, and across multiple adopted
lenses the **most skeptical value wins**. A lens can only demote sources
it explicitly names — adopting one never penalises sources it is silent
about, and unadopting (`trust lens unadopt`) restores the prior policy
exactly. Both operations are recorded in the operator event log.

### Contestedness — measuring how much a lens changes the picture

Once you have **two or more policies** (your local policy plus at least
one adopted lens), Particles can measure each claim's **contestedness**
: the max−min spread of its effective confidence evaluated
separately under each policy. A claim every policy renders the same way
behaves as a fact; one that swings sharply between policies is visibly
contested, and the surfaces attribute *which* policy renders what.

```bash
particles query "..." --contestedness   # per-result spread, attributed by policy
```

Contestedness is **disclosure, not a discount** — it never changes
ranking, effective confidence, or conflict resolution; it only shows you
how much your lens choices move the rendering. It is absent (not zero)
when you have fewer than two policies. The prose exporters render a
`[!contested]` callout, and `particles lint` emits a store-level
`CONTESTEDNESS_DISTRIBUTION` report, both gated by
`contestedness.callout_threshold` (default `0.2`) — raise it to surface
only sharply-divergent claims, lower it to see faint divergence.

## Recency factor

Content age decay. The factor is
`max(floor, 0.5^(age_days / half_life_days))` — effective confidence
halves every `half_life_days`, never dropping below `floor`.
Configured per source-type under `content_age_decay.sources`:

```yaml
content_age_decay:
  sources:
    REDDIT_POST:
      half_life_days: 60
      floor: 0.10
    GITHUB_REPO:
      half_life_days: 365
      floor: 0.40
```

Source types that move fast (news, forum threads) decay quickly;
static documents decay slowly. A source type with **no entry** does
not decay at all (factor 1.0), and content with no known publication
date never decays — omit a source type to disable decay for it.

### Decay rules in a lens

The `content_age_decay` config above is your store's **local** decay
policy. A trust lens can also carry a fourth `decay_rules` layer, so a
community publishes its temporal judgment ("this is how fast we think
these sources go stale") alongside its trust ranks — and you adopt it the
same way (`particles trust lens adopt <name>`):

```jsonc
"decay_rules": [
  { "scope": "source_type",  "pattern": "REDDIT_POST",
    "half_life_days": 60,   "floor": 0.10 },
  { "scope": "url_pattern",  "pattern": "reddit\\.com/r/wallstreetbets",
    "half_life_days": 7,    "floor": 0.05 },
  { "scope": "url_pattern",  "pattern": "reddit\\.com/r/AskHistorians",
    "half_life_days": 1825, "floor": 0.50 }
]
```

Resolution (most-skeptical-wins, mirroring the trust layers):

- **Your local config wins** for a source type it configures; adopted
  lenses only fill source types you leave unset, and across multiple
  lenses the shortest half-life / lowest floor applies.
- A **URL-pattern rule is more specific than the source-type default** and
  overrides it in *either* direction — so a lens can make one subreddit
  *more* durable (a longer half-life) than the `REDDIT_POST` default, not
  only faster.

With no decay-bearing lens adopted, decay is exactly your `content_age_decay`
config — adopting a lens never changes a store that ignores it.

## Usefulness lens (composition)

The projection (`MEMORY.md`) and session-start digest rank by
**effective confidence**, which measures how likely a belief is *true* — not
how *useful* it is to reload. On agent-memory those pull apart: a precise
dated fact scores high but is rarely load-bearing next session, while a soft
working guideline scores lower yet governs behaviour every session. The
usefulness lens adds a **utility rank-lift** so the beliefs the agent actually
*acts on* rise to the top of what renders.

The signal is mined from the harvest, not asked for: a session transcript
records what the agent *did* (its tool calls), so a belief is credited with
utility when the session's actions demonstrably apply it (ran a documented
command, used a workaround). It is **action-based**, never inferred from your
reactions.

Ranking on the projection path is

```
rank_score = effective_confidence + rank_lift x ln(1 + reinforcement)
```

where `reinforcement` is the recency-weighted count of times the belief was
acted on.

```yaml
utility:
  enabled: true          # apply the utility rank-lift to projection/digest ranking
  default:
    half_life_uses_days: 30   # unreinforced utility fades over ~a month
    rank_lift: 0.015          # lambda — how far usefulness may reorder the head
  mining:
    behavioural_matching: true   # LLM-judge soft guidelines (literal match is always on)
    max_behavioural_calls: 50    # per-run cap on those LLM calls
```

**Upgrading from a pre-ADR-0204 config:** `weight`, `floor` and `cap` are
retired — delete them and set `rank_lift`, or the config fails validation.
There is no automatic translation: the old parameters described a multiplier,
`rank_lift` describes an additive term, and a `weight` of 0.5 reused as a
lambda would be roughly 25x too strong. Adopted lenses carrying the old
vocabulary read as silent about utility until re-deposited.

### Calibrating `rank_lift`

`rank_lift` is the single knob, and it is worth calibrating against your own
store rather than porting a number. There is a verb for it:

```bash
particles memory sweep-rank-lift --target <particle-id> --head 60 --head 200
```

Read-only — no writes, no LLM calls, no embeddings. Name the beliefs you assert
*ought* to reach the projection head with `--target` (repeatable), pass each
head size you actually render with `--head`, and the report shows, at every
lambda on a grid: where those beliefs land, how many head slots hold distinct
content, the admissible band per surface, their intersection, and whether your
configured value sits inside it. `--format json` for machine consumption.

Store particle ids are raw UUIDs; the `p-xxxxxxxx` form you see in the digest
is an 8-character truncation behind a display prefix. `--target` accepts either
— the full UUID, the truncated form with or without `p-`, or any unique id
prefix — and resolves it against the store's ACTIVE beliefs before sweeping.

A target that matches no ACTIVE belief, or that matches more than one, **stops
the sweep with an error** rather than being scored. That is deliberate: a
target the sweep cannot find ranks nowhere, so it fails the "target reaches the
head" criterion at every lambda, and the report would come back with an empty
band on every surface — a typo rendered as "this store cannot be calibrated".
If you get `matches no ACTIVE belief`, check the id with `particles particle
show <id>`; a retracted or superseded belief is reported by name, since only
ACTIVE beliefs are ranked.

**Why you have to name the targets.** `rank_lift` is deliberately *not*
auto-fitted. Unlike the temperature fit, which minimises a
real loss against labelled benchmark answers, nothing here labels which belief
*should* occupy a head slot — so the sweep asks you, rather than inventing a
formula. Measured on a real store, every candidate formula failed: the
`effective_confidence` spread the obvious fit would key on is exactly zero
across the rendered head, because the [uncalibrated-confidence
cap](../ADR/active/0182-uncalibrated-confidence-cap.md) ties the whole head at
one value. A spread-keyed fit therefore returns `0` and switches the feature
off.

**Two things bracket the answer.** Below the band, your load-bearing beliefs
stay buried. Above it, one over-extracted belief's near-duplicates start
crowding the head — which is a data-quality signal, not a tuning signal. If the
sweep's ceiling is what is binding for you, the fix is deduplication (see
`particles links suggest`, `particles links dedup` and `particles curate`), not
a smaller `rank_lift`; tuning it down only hides the duplicates in the surface
where they are most visible. `0` disables the lift entirely.

**But check that your dedup pass can reach the clusters setting the ceiling.**
On the dogfood store it could not. `particles links dedup` collapsed 181
exact-duplicate groups (775 redundant ACTIVE copies), cutting near-duplicate
mass from 16.0% to 3.1% of ACTIVE — and the ceiling *fell*, `0.0190 → 0.0165` at
`N = 200`, rather than rising.

The reason is a finder gap, not a matching-strictness one. The two clusters that
set the new ceiling are **21 byte-identical copies each** — well within an
exact-match merge's reach — but they carry 1/21 and 0/21 subject links, and the
duplicate finder iterates Subjects. So the verb reports *zero remaining groups*
while 211 exact-duplicate groups and 534 redundant ACTIVE copies are still
there, 87% of all redundant copies left in the store. The pass drained the
subject-linked low-reinforcement tail; the projection head is exactly where the
subject-less, highest-reinforcement copies sit.

The operational lesson: a clean `links dedup` dry run is **not** evidence that
duplicates are gone. Check the head directly — the sweep's `distinct/N` columns
are the honest signal — and treat the ceiling as a standing over-extraction
signal rather than something one dedup pass retires.

**The two edges are not symmetric.** The floor degrades gently — your target
belief slides down the head a few slots at a time. The ceiling is a cliff: on
the dogfood store one grid step past it (`0.0165 → 0.0170`) costs 19 of the
digest's 200 slots to duplicate copies, and another step costs 40. Prefer the
lower half of the intersection.

**The band belongs to the surface, not to the store.** It is a function of the
rendered head size `N`, and `N` differs per surface — the `MEMORY.md`
projection's per-section `top_k`, the digest's `mcp.recall.digest_max_beliefs`.
A larger head is stricter, because it has more room to expose duplicate pairs.
If you render several head sizes, pass all of them and use the intersection; on
a store with real over-extraction the intersection can be empty, in which case
decide which surface the value is calibrated for and record it.

For reference, on the dogfood store (27.0k beliefs) the three rendered head
sizes admit `0.011` and up (`top_k: 60`), `0.006` and up (`max_lines: 120`) and
`0.004` and up (`digest_max_beliefs: 200`). Do not port those numbers — they are
a property of that store's confidence spread and event volume.

Note those bands have **no upper edge**, which is new. Every earlier
calibration on this store was squeezed between a floor and a duplicate-cluster
ceiling; subject-agnostic merge drained the clusters that set the
ceiling, and the largest in-head duplicate cluster is now 2 at every `λ` up to
`0.6`. Where the ceiling exists it is a **cliff**, not a slope, so bias away
from it; where it does not, the floor is what you budget margin against.

**If you are upgrading from `0.011`:** the default is now `0.015`. `0.011` was
the log-midpoint of the then-measured intersection `0.0075–0.0165`; with the
ceiling gone that rule no longer yields a finite answer, and `0.011` had become
the *binding floor* — one grid step below it the canonical target drops
out of the top-60 head entirely. `0.015` restores 1.36× margin above that floor
while keeping all 60 head slots distinct. An explicit `rank_lift` in your own
`config.yaml` is untouched; run the sweep against your own head sizes to see
whether it still sits in-band.

**Check `ω` whenever you raise `λ`.** If you run the owner lens, its
floor tracks `λ` — a larger utility term holds the head harder against the
lens's flat-step cohort, so the same `ω` promotes fewer viewer beliefs. On the
dogfood store, `λ` of `0.011 / 0.015 / 0.020 / 0.025 / 0.030` moved the `ω`
floor to `0.018 / 0.024 / 0.032 / 0.040 / 0.048`, so a shipped `ω = 0.04` falls
out of band by `λ = 0.030` and the viewer cohort leaves the `N = 60` head
altogether. Re-run `particles memory sweep-owner-lift` after any `λ` change;
past roughly `0.02` the two are a joint calibration, not independent knobs.

Expect to re-run the sweep occasionally rather than once. The band drifts, and
not monotonically: on an actively growing store it moves *down* as reinforced
near-duplicate clusters accumulate (the opposite direction from the quiet-store
drift measured earlier), and a dedup pass that reaches the head can remove
the ceiling outright, as was done here.

Notes:

- **Cold start is confidence-only.** A fresh store has no utility evidence,
  so it ranks exactly as before; usefulness switches on as sessions
  accumulate. Turning the feature on later? `particles memory rebuild-utility`
  re-mines every harvested transcript so history counts.
- **Projection / digest only.** The lift never touches semantic-search
  (`query`) ranking — usefulness governs *what stays important*, not *what is
  relevant to a question*.
- **It ranks; it does not re-score.** `rank_score` is an ordering key, not a
  confidence: it is not capped at 1.0, and the confidence a surface *displays*
  is always the untouched effective confidence. Utility never edits what a
  belief claims about the world.
- **Promotion-only.** `rank_lift` is non-negative, so an unused belief is
  never pushed *down* — it simply does not receive a lift.
- **A lens can carry it too.** A `TrustLensDefinition` may add a
  `utility_rules` layer beside `decay_rules`; across adopted lenses it
  composes most-skeptical-wins (least promotion), like the other layers.

## Owner-relevance lens

The third read-time axis. Truth asks *is it believable*, usefulness asks *has
it earned its place*, and this asks **is it about me** — so the slice of a
store that is about you is ranked up on the recall surfaces instead of being
drowned by domain claims about the topics you merely discussed.

```yaml
owner_lens:
  enabled: true
  subjects: ["Jeff", "Jeff Gage"]   # who the viewer is
  rank_lift: 0.04                   # omega — calibrate it, see below
```

`subjects` is a **list** because a person's Subject fragments in practice —
list every alias your store actually resolved (`particles subjects search
<name>`). Entries may be canonical names or Subject ids, and resolution is
local-only: an entry that matches nothing is logged and contributes nothing
rather than being guessed at, and if *none* resolve the lens is inert.

It lives in **your** config rather than in the store on purpose. Viewer
identity is reader-local: several contributors sharing one store each need
their own viewer, and one person with a work store and a personal store has
one viewer for both.

**Where it applies:** the `MEMORY.md` projection, the session-start digest,
and the graph view (there it decides which *nodes* survive `--max-nodes`, and
never what the graph renders as confidence). It does **not** apply to
`particles query` — that surface already has `--subject` for asking about
someone specifically; the lens exists to fix the surfaces that have no query
at all.

### Calibrating `rank_lift`

`rank_lift` ships `0.0` — the lens is inert until you set it — and has **no
recommended default**, because a viewer's share of a store varies enormously
by genre (0.2 % on a code-and-ADR store, ~5 % on one built from chat imports).

```bash
particles memory sweep-owner-lift --head 60 --head 120 --target p-4fbcc320
```

Unlike the usefulness lift, this one multiplies a flat 0/1 indicator, so it is
a **threshold over the whole cohort**: below it nothing moves, above it every
belief about you arrives in the head at once. Calibrate against the *share of
the head* your cohort takes, not against one belief's rank — which is what the
sweep reports. Prefer a value in the middle of a wide plateau over the band's
edge: a plateau is what makes the outcome robust as the store grows.

The number to have an opinion about is **amplification** — your share of the
head divided by your share of the store, which the report prints for you. A raw
`rank_lift` means nothing across stores; a ratio means the same thing
everywhere. On the reference store a 0.2 % cohort at `rank_lift: 0.04` takes
8 % of a top-60 — roughly 40× — which is a clear presence without displacing
the domain knowledge that is the rest of the store's value.

If the report says the band is **open at the top**, the upper number is where
the grid stopped, not a ceiling: every value swept passed. That is normal when
your cohort is small enough that it could never take half the head at any
plausible value. Raise `--grid-max` if you want the real ceiling, but the
amplification is the better guide.

`--target` names beliefs that must **stay** in the head, checked as a
non-regression against `rank_lift: 0.0`. A target that was already outside the
head before the lens existed is reported in its own section rather than
silently emptying the band — that is a real regression, but not one this lens
caused.

## Cross-exporter quality threshold

`min_particle_confidence` is the universal drop-floor
honored by every exporter. Particles with `effective_confidence` <
`min_particle_confidence` are dropped from every rendered output:

```yaml
exporter_common:
  min_particle_confidence: 0.65
```

Operator runs export with `--min-particle-confidence` for
per-invocation overrides.

## Benchmark + compare

To measure whether a tuning change actually improved things:

```bash
# single extractor
uv run particles extractor benchmark numista-coin-extractor

# A/B two extractors against the same corpus
uv run particles extractor benchmark-compare \
    --extractor-id numista-coin-extractor \
    --extractor-id numista-coin-extractor-v3
```

The comparison reports precision / recall / calibration_error per
suite × extractor. Use it before rolling a calibration
or trust-weight change to verify the direction of effect.
