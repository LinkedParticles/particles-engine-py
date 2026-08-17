# Schema migration

`SCHEMA_VERSION` (in `particles/core/schema.py`) is the on-disk
schema's identity. Major-version bumps are breaking.

## Minor / patch bumps

The Alembic migration chain handles minor and patch bumps
transparently:

```bash
uv run alembic upgrade head
```

The migration registry centralises every ORM module's metadata via
`particles/_orm_modules.py`. Adding a new table is one Alembic
revision + an entry in `_orm_modules.py`; the schema_version stays
the same.

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
4. **Re-extract** with `particles extract --all-pending`.

The "scrap-and-re-extract" path is what `--force` exists for. It's
the supported migration mechanism between schema major versions
because the corpus is the durable record and particles are
rebuildable from corpus + extractor.

## Mid-migration safety

If the SDK and the on-disk database disagree on `SCHEMA_VERSION`,
the CLI fails fast with a friendly message rather than silently
running incompatible code. See [troubleshooting](troubleshooting.md)
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
