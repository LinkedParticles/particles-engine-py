# Co-evidential curation

Two particles are **co-evidential** when they assert the same underlying
claim — paraphrases drawn from different sources. Linking them
(`CO_EVIDENTIAL`, §6.10) collapses the duplicate at query time so a claim
backed by five sources counts as one well-supported claim, not five
competing ones.

Finding those pairs is the job of `particles links suggest`.

!!! info "Migration — candidates moved out of `lint` (0.46.0)"

    Through 0.45.x, `particles lint` emitted one INFO
    `CO_EVIDENTIAL_CANDIDATE` finding per similar pair. On a real corpus
    that was **tens of thousands** of findings — more than 96 % of all
    lint output — drowning the structural errors lint exists to surface.

    As of 0.46.0, lint no longer emits `CO_EVIDENTIAL_CANDIDATE` at all.
    Candidate proposal is now `particles links suggest`. Any script that
    parsed `lint` JSON for `CO_EVIDENTIAL_CANDIDATE` entries must switch
    to `links suggest --output-format json`.

## "I ran `links suggest` and got 22,000 candidates — what now?"

A five-figure candidate count is normal for a large corpus and is **not**
a problem to fix by hand. The point of the verb is that you never run
22,000 `links add` commands. Instead you let an LLM judge the pairs in
batches and auto-link only the confident paraphrases.

The workflow has three gears:

| Command | What it does | Cost |
|---|---|---|
| `links suggest --subject "<name>"` | List candidate pairs for one Subject. No LLM, no mutation. | Free |
| `links suggest --all --llm-judge` | For every Subject, batch the candidate cluster to the LLM and label each pair `PARAPHRASE` / `DISTINCT` / `UNSURE`. No mutation. | ~cents per Subject |
| `links suggest --all --llm-judge --apply --yes` | Same, then auto-link every `PARAPHRASE` pair. `DISTINCT` / `UNSURE` are reported but never linked. | ~cents per Subject |

### Start narrow

Pick one noisy Subject and look before you leap:

```bash
particles links suggest --subject "1 Pfennig (1948-1950) GDR"
```

Each line is a candidate pair with its cosine similarity. If the pairs
look like genuine paraphrases, judge them:

```bash
particles links suggest --subject "1 Pfennig (1948-1950) GDR" --llm-judge
```

Now each pair carries a verdict. When you trust the verdicts, apply:

```bash
particles links suggest --subject "1 Pfennig (1948-1950) GDR" --apply
```

`--apply` implies `--llm-judge` and links only the `PARAPHRASE` pairs.

### Then go wide

Once you trust the workflow on a few Subjects, run the whole corpus:

```bash
particles links suggest --all --llm-judge --apply --yes
```

The `--yes` is required because `--apply` refuses to link more than
`links_suggest.apply_confirm_threshold` pairs (default 10) without
explicit confirmation — a guard so a stray invocation can't link
thousands at once.

## Exact duplicates: `links dedup`

The candidates above are *near*-duplicates and need judgment. A large share of
a real store's duplicate mass needs none: the same sentence, byte for byte,
extracted again from a re-fetched snapshot. `particles links dedup`
 collapses those
and only those.

```bash
particles links dedup                  # read-only census — writes nothing
particles links dedup --apply          # merge (needs the config opt-in below)
```

**The predicate is exact content equality — a SHA-256 comparison.** There is no
similarity threshold and no LLM call, and this is permanent rather than
conservative-for-now. The hash is taken over the *normalized* string — runs of
whitespace collapsed and a sentence-final period trimmed, wording and case
untouched — which is the same key extract-time suppression uses, so what the
store refuses to mint twice is exactly what this verb can still clean up. On a
27,008-belief store the normalization is worth 3 groups out of 206 (all
trailing-period twins); it is here so `--apply` reporting "0 groups" means the
store is clean, not that the mop was keyed one notch tighter than the guard.
Measured across 38,069 candidate pairs on a real store,
byte-identical content sits at cosine exactly `1.000000` while the highest
non-identical pair is `0.998920`; but *below* identity, cosine does not order
duplicate-likelihood at all — the 0.97–0.99 band hand-scores **worse** (56.7 %
true duplicates) than the 0.95–0.97 band beneath it (73.3 %), and the worst
false positive (`claude-opus-4-6` vs `claude-opus-4-5`) sits at **0.9951**,
above where anyone would set a "very high similarity" threshold by intuition.
So there is deliberately no middle tier: everything short of identity stays
advisory under `links suggest`.

**What a merge does — and the three things it never does.** Per group (one
content hash, N ≥ 2 ACTIVE copies), a survivor is elected deterministically
(subject-linked before subject-less, then earliest `asserted_at`, ties broken
by smallest id), linked
`CO_EVIDENTIAL` to each redundant copy with `created_by = EXACT_DUPLICATE`, and
those copies transition `ACTIVE → SUPERSEDED` with `status_reason =
DUPLICATE_MERGED`. One `DUPLICATES_MERGED` operator event per group records the
survivor, every superseded id, and the config in force. It never **deletes** —
supersession is a ledger transition, so every copy stays readable via
`particle show` and recoverable from the event. It never **mutates the
survivor**. And it never touches a non-truth-apt or non-asserted particle.

Because the merge is idempotent (a merged copy is no longer ACTIVE), a second
run is a no-op.

!!! info "`links dedup` is the mop; the leak fix is automatic"

    `links dedup` cleans up duplicates that already exist. It does not stop
    new ones, and on a store fed by the [Claude Code
    harvest](../user-guide/claude-code.md) the mass regrows fast — measured at
    **15.8 % of everything extracted** over an eight-day window, because every
    re-deposit of a memory file or transcript re-extracted claims the store
    already held verbatim.

    Since 1.90.0 that leak is closed:
    extraction no longer mints a particle whose claim is already held by an
    ACTIVE particle with the same subjects and `stance:holder`. The new
    source's provenance ref is appended to the existing particle instead, so
    nothing is lost, and `particles extract` reports the count
    (`Extracted 3 particles (12 duplicate(s) suppressed).`). It is **on by
    default** — unlike `--apply` above, it only declines to *create*, so a
    suppression cannot drop a distinct fact. Turn it off with
    `extraction.duplicate_suppression.enabled: false`.

    So on a current store you should need `links dedup` once, for the backlog.

!!! warning "Read this before turning it on"

    Merging removes N−1 provenance refs from the **ACTIVE-only** read surface.
    The corroboration is not destroyed — it moves onto the `CO_EVIDENTIAL`
    edges and stays walkable from the survivor — but a consumer that counts
    corroborating sources by reading ACTIVE particles *without following
    relations* will see fewer. Audit any such consumer first. On the dogfood
    store every duplicate group had more than one distinct provenance ref, so
    this is the expected case, not an edge case.

**Coverage is partial, by design.** Copies carrying **no Subject** *are*
reached (
the Subject is an election preference, not a membership gate), but copies whose
Subjects **disagree** are never one group; non-`FALSIFIABLE` particles are
excluded by the truth-apt
gate; and identical text held by *different* `stance:holder` principals is
never merged. Note `--subject <id>` necessarily excludes subject-less copies —
an orphan is in no Subject — so a whole-store run is the superset. Expect
`links dedup` to collapse the truth-apt exact duplicates that agree on their
Subject (or have none) — not "all duplicates".

### Config

Auto-merge is the only store-mutating curation path, so it is **off by
default** and stays off across upgrades. `--apply` refuses to write until you
opt in:

```yaml
links_suggest:
  auto_merge:
    enabled: false   # set true to permit `links dedup --apply`
    max_per_run: 500 # cap on groups per run; the remainder is disclosed
```

The dry run needs no opt-in. A capped run reports how many groups and copies
remain rather than reading as a complete cleanup — re-run to continue.

### Reverting: `links unmerge`

Every merge is undoable with one command
. Find the
group's event, then revert it:

```bash
particles events list --type DUPLICATES_MERGED
particles links unmerge <event-id> --dry-run   # plan: what comes back, what doesn't
particles links unmerge <event-id>             # asks before writing
```

The undo is *exact* rather than approximate: the retained copies return to
ACTIVE **keeping their ids** and with their `status_reason` cleared, only the
merge's own `EXACT_DUPLICATE` links are dropped (a co-evidential link you or
the judge made on the same pair survives), and the survivor is never touched.
So `merge` followed by `unmerge` leaves the store exactly as it started — no
new rows, no tombstones.

To undo a whole run rather than one group, use `--run <run-id>` — every event
a single `--apply` writes carries the same run id. Merges recorded before run
ids existed are reachable with `--since <ts>` (optionally `--until <ts>`).

!!! note "Copies that moved on are skipped, not restored"

    If a copy changed since the merge — an earlier partial revert brought it
    back and something later retracted it — the revert names it and moves on
    rather than overturning the later decision. The rest of the group still
    restores; one drifted row never blocks the others. Re-running is safe:
    everything already ACTIVE is skipped.

`links unmerge` is **not** gated on `auto_merge.enabled`. That flag authorizes
merging, so switching it off after a run you regret must not also take away the
cleanup. Note the revert is not durable against the *next* `links dedup
--apply`, which will re-merge the same identical-content copies — if you reverted
because the merge was wrong for this store, turn `enabled` back off.

## Tuning

All knobs live under `links_suggest` in `config.yaml`
(see [Configuration](configuration.md)):

| Key | Default | Effect |
|---|---|---|
| `candidate_threshold` | `0.92` | Cosine-similarity floor for proposing a pair. Higher = fewer, more obvious candidates. |
| `max_cluster_size` | `50` | Per-Subject candidate clusters larger than this fan out across multiple LLM calls; a transitive cluster is never split. Fan-out emits a `WARNING` in the report. |
| `apply_confirm_threshold` | `10` | `--apply` over this many pairs needs `--yes`. |
| `auto_merge.enabled` | `false` | Permits `links dedup --apply` to write. Off by default, permanently — a stock install never auto-mutates a store. |
| `auto_merge.max_per_run` | `500` | Cap on **groups** merged per `links dedup --apply` run. A capped run discloses the remainder. |

!!! note "Renamed config key"

    `candidate_threshold` was `lint.co_evidential_candidate_threshold`
    before 0.46.0. The old key is still read for one minor cycle and
    logs a deprecation warning on load — update your `config.yaml` to
    the new key to silence it.

## Surfaces

- **CLI** — `particles links suggest` (all three modes; `--apply` here) and
  `particles links dedup` (exact-duplicate auto-merge; local store only).
- **HTTP** — `POST /links/suggest` mirrors the CLI, including `--apply`
  (pass `{"mode": "APPLY", "confirmed": true}`). Returns `409` when
  `APPLY` exceeds the confirm threshold without `confirmed`.
- **MCP** — the `links_suggest` tool exposes **report mode only** (the
  MCP surface is read-only);
  agents surface candidates, the operator applies them via CLI/HTTP.

## What's deferred

- **Cross-Subject candidates.** Today's similarity scan is per-Subject;
  a property of one Subject that paraphrases a property of another is not
  surfaced. That needs its own ADR.
- **AUTO_CLUSTER_V1.** Full transitive-closure clustering with persisted
  cluster IDs (§ Deferred)
  layers on top of this verb; `--apply` today is pair-wise, not
  cluster-wise.
- **Auto-merging *near* duplicates.** `links dedup` handles exact identity
  only. Extending it needs the LLM judge's own precision measured against a
  hand-labelled sample, which has not happened yet
  (§ Deferred).
- **Ambiguous-home orphans.** When an identical-content group offers subject-less
  copies more than one candidate Subject, `links dedup` declines to guess and
  merges them only among themselves
  ,
  so the group keeps two survivors of identical text.
