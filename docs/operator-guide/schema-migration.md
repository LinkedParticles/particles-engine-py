# Schema migration

`SCHEMA_VERSION` (in `particles/core/schema.py`) is the version of the
particle record format, and every particle records the version it was
written under. It is `MAJOR.MINOR.PATCH`. Two versions with the same
major are compatible; a different major is breaking.

Two things version a store, and they move independently:

| | What it versions | How it moves |
|---|---|---|
| `SCHEMA_VERSION` | what a particle record means | an ADR that changes the record format |
| the Alembic revision | the tables and columns that hold it | `particles db init` / `alembic upgrade head` |

## Minor and patch versions

A minor version only adds: new optional fields whose absence means what a
record meant before. An SDK reads and writes a store holding any minor of
its own major, in either direction:

- **A newer SDK on an older store** reads the older particles with the
  new fields at their defaults. `particles lint` reports them as
  `SCHEMA_VERSION_OLDER_MINOR` (INFO). The release that added the fields
  names the command that fills them in; `particles reindex
  --include-schema-stale` re-extracts those particles' sources instead,
  which is a paid run (the dry run reports its size first).
- **An older SDK on a newer store** reads the newer particles without the
  fields it does not know, and writes new particles at its own version.
  `particles lint` reports the newer ones as `SCHEMA_VERSION_NEWER_MINOR`
  (INFO). It can still retract or overrule a newer particle, but it never
  replaces one with its own copy (a reindex re-extraction, a dated
  restatement, a duplicate merge): those are skipped and counted, and wait
  for a current SDK.

A patch version changes nothing about the record and is ignored by every
comparison. Moving between minors needs no rebuild. The storage half is
the ordinary migration chain:

```bash
uv run alembic upgrade head
```

The migration registry centralises every ORM module's metadata via
`particles/_orm_modules.py`. Adding a new table is one Alembic
revision + an entry in `_orm_modules.py`. A migration shipped with a
minor schema version is additive (nullable columns, columns with a
server default, new tables or indexes), so an older SDK keeps writing to
the migrated tables.

SDKs released before same-major compatibility (see the changelog entry
that introduced it) refuse a store holding any version other than
`1.0.0`, so upgrade every installation that shares a store before
running a release that moves `SCHEMA_VERSION`.

## Major bumps

A `1.x.y → 2.0.0` schema bump means the particle on-disk shape
changed in a way that's not backwards-compatible. The operator path:

1. **Read the release notes / ADR** for the major bump. It will
   document what changed and the migration cost.
2. **Backup the corpus.** The corpus blobs + `corpus_entries` /
   `snapshots` tables survive major bumps; only particle-store
   tables get scrapped.
3. **Run `particles db init --force`.** This drops every
   particle-store table (particles, subjects, particle_subjects,
   relations, source_trust_statements, extractor records) and
   resets every snapshot's `extraction_status` to `PENDING`. The
   corpus itself is preserved. The command confirms before dropping.
4. **Re-extract** with `particles extract --all-pending`. For a `MUTABLE`
   source with several snapshots (a memory or rule file edited over time),
   only the newest is re-extracted, so the rebuilt store holds the same
   current beliefs but not the `PROVENANCE_STALE` history of each
   intermediate version. Set `extraction.collapse_superseded_pending: false`
   before re-extracting if you want that history rebuilt, at one LLM call per
   version. See [Mutable local sources](mutable-local-sources.md#only-the-newest-pending-version-is-extracted).

The "scrap-and-re-extract" path is what `--force` exists for. It's
the supported migration mechanism between schema major versions
because the corpus is the durable record and particles are
rebuildable from corpus + extractor.

## Mid-migration safety

If the store holds particles of another major `SCHEMA_VERSION`, the CLI
fails fast with a message that says which way the store differs, rather
than silently running incompatible code. See [troubleshooting](troubleshooting.md)
for the exact error and recovery path.

## What's NOT preserved

- Particles, subjects, the particle_subjects join table, relations
- SourceTrustStatements
- Extractor calibration history
- Particle embeddings (regenerated on first query after reindex)

## What IS preserved

- Corpus entries + every snapshot
- The blob directory contents
- `corpus_follow_edges`
- Conformance baseline JSON files under `docs/conformance/baseline/`
