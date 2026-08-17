# particles/store/AGENTS.md

Loaded by agents when reading or editing files under `particles/store/`.

This package owns the SQLAlchemy ORM (the `*Row` classes), the read/write
accessors over them, and the in-memory subject-resolution cache. It is
the seam between persistence (rows) and the rest of the codebase
(Pydantic models defined in `core/schema.py`).

## The store ↔ store cycle (particle_store ↔ subject_store)

`particle_store.py` and `subject_store.py` legitimately import each
other's `*Row` classes — particles link to subjects via
`particle_subjects`, and subjects expose helpers that return particle
rows. Hoisting either import to module top breaks the import chain
(`ImportError: cannot import name '…' from partially initialized
module`).

The two cross-imports stay as function-body deferred imports, marked
with `# defer: cycle — particle_store and subject_store import each
other's ORM Row classes. See root AGENTS.md § Code conventions →
Deferred imports.` (case 1 in that doc). The canonical example lives
at `particle_store.py:204`.

If you find yourself adding a new store module that needs to import
from both particle_store and subject_store, the same convention
applies — defer inside the function body, comment with `# defer:
cycle`.

## `subject_cache.py` belongs here, not in `extraction/`

The resolver in `ingest/subject_resolver.py` populates the
subject-resolution cache; the store in `subject_store.py` invalidates
it on mutation. Pre-C1, the cache lived inside the resolver and
`subject_store` reached *up* into `extraction/` to call `clear_cache()`
— a layer inversion (lower → higher).

The fix (commit `7f922ad`, audit-1 C1): move the cache to `store/`
where both layers can depend on it without inversion. Don't move it
back. If the cache grows new responsibilities (e.g. a second consumer
beyond the resolver), keep it under `store/` and add a clear API note
to its module docstring.

## Boundary: store returns Pydantic, not ORM rows

Store accessors meant for consumption outside `store/` must return
Pydantic models (`Particle`, `Subject`, `CorpusEntry`, etc.), not the
`*Row` instances. Use the `row.to_model()` helper that each ORM class
defines.

The exception: store-internal helpers and the few places where
`api/cli/*.py` does a direct ORM query for a CLI-only purpose
(documented per-case). Don't expand the exception list without a
specific reason — once ORM rows escape, callers start writing
session-aware code outside `store/`, which is the layering boundary
we're protecting.

## Files

| File | What it owns |
|---|---|
| `particle_store.py` | `ParticleRow` ORM + accessors (`get_particle`, `list_particles_filtered`, `get_particles_by_status`, `get_active_particles_with_embeddings`, …) |
| `subject_store.py` | `SubjectRow`, `ParticleSubjectRow` ORM + accessors (`get_subject`, `list_all_subjects`, `search_subjects`, `link_particle_to_subjects`, alias management, merge, external-ref management) |
| `subject_cache.py` | Process-level in-memory cache `_cache: dict[str, CacheEntry]`; `cache_get`, `cache_set`, `clear_cache` |
| `relation_store.py` | `RelationEdgeRow` ORM (co-evidential / supersedes / etc. relations between particles) |
| `extractor_store.py` | `ExtractorRecordRow` ORM (Extension A registry — extractor IDs, versions, trust weights) |
| `trust_store.py` | `SourceTrustRow`, `TrustStatementRow` ORM (Extension B source trust policy + per-source-domain statements) |
| `taxonomy_store.py` | `TaxonomyDefinitionRow`, `TagNodeRow` ORM (Extension C folksonomies) |
| `wikidata_cache.py` | Persistent Wikidata-label lookup cache (read-through, rate-limit-aware) |

## Conventions

- **`async def` everywhere.** All accessors take `AsyncSession` as the
  first positional argument.
- **One file per entity.** Don't co-locate accessors for `SubjectRow` in
  `particle_store.py` even when the call sites would be convenient.
- **No business logic.** A store accessor reads or writes rows; it does
  not decide whether a write *should* happen. That decision belongs in
  `operations/` (write paths) or in `ingest/pipeline.py` (the
  §6.6-aware extract dispatch).
- **No `print()` or `logging`** at the accessor surface — store layer
  failures are bugs, surfaced via raised exceptions.
- **Mapped[T] / mapped_column()** — SQLAlchemy 2.x style only; never
  legacy `Column()`.
