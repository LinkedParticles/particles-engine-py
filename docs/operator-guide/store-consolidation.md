# Store consolidation (merging two divergent local stores)

You run Particles on more than one machine, each with its own local store,
and the two have **diverged** — different corpus entries, different particles,
different subjects. This page is the validated procedure for folding one store
into the other so you end up with a single canonical archive.

The running example is the real one this page was written for:
a **Mac Mini** store (the superset — designate it **canonical**) and a
**MacBook Air** store whose unique source material you want to fold in.

!!! danger "Both procedures back up the canonical store first and are reversible"
    Nothing here mutates a store in place without a copy-out backup. The merge
    is owner-supervised and needs **both machines available**. Run the
    [backup step](#backup-and-rollback) before any import.

## Which mechanism — decide first

There are two ways to move knowledge between stores. They are **not**
equivalent, and the right choice depends on how much you are folding in.

| | **Re-deposit + re-extract** | **Interchange export/import** |
|---|---|---|
| What moves | the **source material** (URLs / files) | the **ACTIVE particles + subjects** |
| Provenance in the canonical store | **valid** — points at real local corpus entries | **orphaned** — points at the *other* store's entry IDs |
| Corpus / blobs | rebuilt locally | **not moved** (provenance dangles) |
| Trust policy, event log, relations, lenses, non-ACTIVE history | rebuilt / re-run | **dropped** (see below) |
| Cost | one LLM extraction per re-deposited entry | one LLM reconcile probe per high-similarity collision |
| Best when | a **small** set of re-fetchable sources (the Air's ~21 entries) | the source is **gone** and the particles are not re-derivable (hand-asserted beliefs) |

**Recommendation for the Mini ⇐ Air merge: re-deposit + re-extract.** The Air
has only ~21 corpus entries, its sources are ordinary URLs/files, and this is
the only path that lands them in the canonical store with **intact provenance
and a real corpus**. Reach for interchange import only for particles you
cannot re-derive from a source (e.g. agent/operator beliefs asserted directly
over the MCP write surface).

## What the interchange bundle carries — and what it drops

A store-export bundle is a directory of exactly three files: `manifest.json`,
`particles.jsonl`, and `subjects.jsonl`. It carries **only the knowledge-graph
core**. Everything else is a deferred bundle member and is **silently dropped** on export.

This was validated end-to-end with a scratch export ⇄ import across two
throwaway SQLite stores (no real data touched). Seeded store **A** held 4
particles (3 ACTIVE + 1 INCONSISTENCY), 2 subjects, 1 corpus entry, 1 trust
statement, 2 operator events, 1 CO_EVIDENTIAL relation. After
`export → import` into empty store **B**:

| State in A | In the bundle | In B after import | Survives? |
|---|---|---|---|
| 3 ACTIVE particles (CLAIM + NARRATIVE) | 3 units | 3 ACTIVE | ✅ substrate only |
| 1 INCONSISTENCY particle | — | 0 | ❌ **only ACTIVE is exported** |
| 2 subjects (1 Wikidata-QID, 1 bare-local) | 2 units | 2 subjects | ✅ by external ref → name |
| 1 corpus entry + snapshot + blob | — | 0 | ❌ provenance **orphaned** |
| 1 trust statement | — | 0 | ❌ dropped |
| 2 operator events (audit log) | — | 0 | ❌ dropped |
| 1 CO_EVIDENTIAL relation | — | 0 | ❌ dropped |

Concretely, dropped on every interchange round-trip:

- **Non-ACTIVE particles.** Export is ACTIVE-only. SUPERSEDED, RETRACTED,
  PROVENANCE_STALE, and **INCONSISTENCY** particles do not travel — you lose
  the supersession history and the entire contradiction/dispute ledger.
- **Corpus entries, snapshots, and raw blobs.** The particle's `provenance`
  still carries the **source store's** `corpus_entry_id` / `snapshot_id`, but
  those rows do not exist in the target. The reference dangles: `particles
  lint` will flag `CORPUS_LINK_GAP`, and "show me the source" cannot resolve.
- **SourceTrustStatements** (your trust policy).
- **The operator event log** (the audit trail of retractions, merges, reviews,
  trust changes).
- **Particle relations** — `CO_EVIDENTIAL` clusters, the `PART_OF` /
  `SEQUENCE_IN` edges that reconstruct **narrative** prose, and the
  `ENDORSES` / `DISPUTES` stance edges. NARRATIVE particles travel as
  substrate, but the edges that make them readable do not.
- **Adopted-lens / viewpoint state** — a store
  export materialises particles but drops the viewpoint that rendered them.

What **does** survive: the ACTIVE particle substrate (content, the calibrated
`confidence.value`, uncertainty, provenance *descriptors*, status, tags,
properties, `context_fingerprint`, `extractor_ref`, contributors,
`assertion_modality`), and subjects resolved by external reference. Derived
quantities (`effective_confidence`) are correctly **never** serialized — they
recompute on import.

## Collision analysis — importing the Air into the populated Mini

The validated run imported into an **empty** store (clean: 3 imported, 0
dropped). Importing into the **populated** Mini is different — every imported
particle runs the §6.6 reconciliation ladder against the Mini's ACTIVE set.

**Particle IDs — no collision.** Import mints a **fresh** UUID for every
particle; the source UUID rides along as `sourceParticleId` (origin metadata
only). The validation confirmed the imported particle's ID differs from the
source. So there is no primary-key clash — but see idempotency below.

**No exact-duplicate dedup → import is not idempotent.** Import identity is
embedding-similarity + §6.6, **not** claim identity. Re-running the same import
does not no-op: an identical claim re-embeds to the same vector, matches its
own twin, is judged *corroborating* (not contradicting), and is **inserted
again as a near-duplicate**. Run an interchange import **once**.

**Importing into a non-empty store invokes the LLM.** When an imported claim is
highly similar to an existing Mini claim, reconciliation runs an LLM
contradiction probe (needs `ANTHROPIC_API_KEY`, and costs one call per
collision). Outcomes per imported particle:

- *No similar claim* → inserted clean as ACTIVE.
- *Similar, not contradictory* → **corroborates**; inserted alongside (a
  near-duplicate you may later link or merge).
- *Similar and contradictory* → resolved by **reconciliation mode** (below).

**Reconciliation mode matters**. The effective
mode for a store is: an explicit `reconciliation.per_store` entry wins;
otherwise an **MCP-write-enabled store defaults to `multi`**; otherwise the
global `reconciliation.store_mode` (default `single`).

- `single` — a confirmed contradiction **auto-supersedes**: either the import
  demotes an existing Mini particle to PROVENANCE_STALE, or the import itself
  is **dropped** (counted in the import summary's `dropped`) and the drop is
  logged as a `CONFLICT_CANDIDATE_DROPPED` event. Silent for the operator.
- `multi` — a confirmed contradiction is **quarantined as an INCONSISTENCY**
  for you to `review`, and nothing is auto-superseded. If the Mini's `default`
  store is MCP-write-enabled, it is already in `multi`, so expect a review
  queue rather than silent supersession after an import.

**Subject IDs — no UUID collision, but two real merge hazards.** Subjects never
travel by local UUID. Import resolves each ref **external ref → canonical name
→ create bare-local**. The hazards:

- *Same entity, no shared external ref.* A bare-local subject (no QID) on the
  Air whose name does not exactly match the Mini's imports as a **new
  duplicate** subject.
- *Different entities, same name.* Two distinct things that share a canonical
  name **merge erroneously** — name resolution cannot tell them apart.
- *Same entity, different authorities.* `wikidata:Q42` vs `numista:N123` for
  the same entity **do not merge** (cross-authority `sameAs` is deferred).

Re-deposit + re-extract sidesteps all of this: subjects are re-resolved live
against the Mini's own subject graph at extraction time.

## Procedure A — re-deposit + re-extract (recommended for the Air's ~21 entries)

Run with **both machines available**. Commands are run on each machine's local
default store.

**0. Back up the canonical (Mini) store.** See [Backup and rollback](#backup-and-rollback).

**1. On the MacBook Air — list its corpus entries.**

```bash
particles corpus list
```

Note the `SOURCE` column (the URI / file path) for each of the ~21 entries.

**2. On the Mac Mini — list its corpus entries and diff.**

```bash
particles corpus list > mini-entries.txt
```

Compare the two `SOURCE` lists. The entries present on the Air but **not** on
the Mini are the unique set to fold in. (There is no built-in cross-store
corpus diff — compare the URI columns by hand or with `comm`/`diff` on the
source columns.)

**3. On the Mac Mini — re-deposit and re-extract each unique source.**

```bash
# For each unique source URL or file from step 2:
particles deposit "https://example.org/the-air-only-source"
particles extract --all-pending
```

`deposit` re-fetches the source into the Mini's corpus (new local entry +
snapshot + blob); `extract --all-pending` runs the extractor over every
freshly deposited snapshot, producing ACTIVE particles with **valid local
provenance**, reconciled against the Mini's existing knowledge through §6.6.
Preserve any deposit flags the Air used (e.g. `--journal`, `--tags`,
`--date`, `--source-type`) so the entry is classified identically.

**4. Verify.**

```bash
particles quality          # counts by status / source — expect the Mini's totals to grow
particles lint             # no new CORPUS_LINK_GAP (provenance is local and valid)
```

**When a source can't be re-fetched.** If an Air entry's URL is dead and you
have no local file, you cannot re-derive its particles — that is the one case
to fall back to Procedure B for those specific particles, accepting orphaned
provenance.

## Procedure B — interchange export/import (when re-derivation is impossible)

Use this only for particles you cannot rebuild from a source. Be aware it
moves the Air's **entire** ACTIVE set (there is no "export only the unique
entries" filter), drops everything in the table above, and orphans provenance.

**0. Back up the canonical (Mini) store** ([below](#backup-and-rollback)).

**1. On the MacBook Air — export a bundle.**

```bash
particles interchange export -o air-bundle/
# add --store <handle> if the Air's archive is a named (non-default) store
```

This writes `air-bundle/{manifest.json,particles.jsonl,subjects.jsonl}`.
Inspect `manifest.json` — its `counts` should match the Air's ACTIVE-particle
and subject totals.

**2. Transfer `air-bundle/` to the Mac Mini** (USB, `scp`, AirDrop — any file
copy).

**3. Dry-run into a *scratch copy* of the Mini, not the real store.** There is
**no `--dry-run` flag**, and import is not idempotent, so preview against a
throwaway copy:

```bash
# On the Mini, with the canonical store idle:
cp particles.db /tmp/mini-scratch.db
DATABASE_URL="sqlite+aiosqlite:////tmp/mini-scratch.db" \
  particles interchange import air-bundle/
DATABASE_URL="sqlite+aiosqlite:////tmp/mini-scratch.db" particles lint
DATABASE_URL="sqlite+aiosqlite:////tmp/mini-scratch.db" particles quality
```

Read the import summary (`imported / dropped / subjects_created`) and the lint
output. A high `dropped` count or many new INCONSISTENCY particles is your
signal to reconsider before touching the real store.

**4. Import into the canonical Mini store** (only once you are satisfied):

```bash
particles interchange import air-bundle/
```

**5. Clean up after the import.**

```bash
particles lint     # expect CORPUS_LINK_GAP findings — orphaned provenance from the Air
particles review   # resolve any INCONSISTENCY particles the import surfaced (multi mode)
```

You will **not** recover the Air's trust policy, event log, relations, adopted
lenses, or non-ACTIVE history — re-establish trust statements and re-link
co-evidential clusters by hand if you need them.

## Backup and rollback

A store is a SQLite database file plus a content-addressed blob directory.
Both must be copied together.

1. **Quiesce the store.** Stop any running engine / CLI / MCP server — SQLite
   is single-writer, and you want a consistent copy.
2. **Copy the database and its WAL sidecars, and the blob directory.** Paths
   come from `config.yaml` (`storage.database_url` and `storage.blob_dir`;
   defaults `./particles.db` and `./corpus_blobs`):

    ```bash
    mkdir -p mini-backup-2026-06-16
    cp particles.db particles.db-wal particles.db-shm mini-backup-2026-06-16/ 2>/dev/null
    cp -R corpus_blobs mini-backup-2026-06-16/
    ```

   (The `-wal` / `-shm` files exist only while the DB is open; copying them
   when present is harmless and safe.)
3. **Rollback** = restore those files over the live ones (store still
   quiesced). Because both procedures only ever *add* to the canonical store,
   restoring the pre-merge backup fully undoes a merge.

This is the same durability model as a [schema migration](schema-migration.md):
the corpus + blobs are the durable record, and particles are rebuildable from
them.

## Findings — tooling gaps surfaced by this validation

These are limitations the merge runs into. Each is noted with where it belongs
in the decision register; **no new PDR rows were added by this validation** —
flagging only.

- **Provenance orphaning on import.** Particles import with provenance pointing
  at the source store's corpus entries, which don't exist in the target →
  `CORPUS_LINK_GAP`. This is a direct consequence of corpus blobs being a
  deferred bundle member — already tracked.
  No new PDR needed.
- **No selective / delta export.** Export is whole-store ACTIVE; there is no
  "only entries not already in the target" or "since timestamp" filter, which
  is exactly what a fold-in-the-unique-entries merge wants. Tracked by the
  delta-export PDR. No new PDR needed.
- **Dropped trust / events / relations / lenses.** Covered by
  pending items (trust, events, relation edges) and
  lenses (deferred). No new PDR needed.
- **Import is not idempotent and has no `--dry-run`.** Re-running an import
  duplicates particles (no claim-identity dedup before §6.6), and the only way
  to preview is to import into a throwaway DB copy. This is **not** cleanly
  covered by an existing row and is the strongest candidate for a **new PDR**
  if a safe, repeatable "merge only what's missing" import workflow is wanted
  — recommend the owner open one rather than relying on the scratch-copy
  workaround. (Not opened here, per the scope of this validation.)

## See also

- [Schema migration](schema-migration.md) — `db init --force`, what survives a
  major bump (the same corpus-is-durable model).
- [Lint and review](lint-and-review.md) — resolving the `CORPUS_LINK_GAP` and
  INCONSISTENCY findings an import can surface.
- [Auditing operator actions](auditing.md) — the event log that interchange
  does **not** carry across stores.
