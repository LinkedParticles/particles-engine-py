# Writing an exporter

An exporter walks the particle store and writes an external
representation — Obsidian vault, Anki deck, JSON Lines file, …

## The contract (in 30 seconds)

```python
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from particles.exporters.summaries import BaseExporterSummary

class MyExporter:
    FORMAT = "myformat"                    # unique lowercase slug

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> BaseExporterSummary:
        ...
        return MyFormatSummary(...)
```

Plus a Pydantic `MyFormatSummary(BaseExporterSummary)` in
`particles/exporters/summaries.py`. Add one line to
`particles/exporters/registry.py::_make_exporters()`. Done.

## Worked examples in the tree

| Output shape | File | Notes |
|---|---|---|
| Directory of `.md` files (vault) | `particles/exporters/obsidian/` | Template dispatch; pivot vs coin vs generic |
| Single flat file (deck) | `particles/exporters/anki.py` | Cards from `properties` or content |
| Directory of cited articles | `particles/exporters/wiki.py` | LLM synthesis via shared `article_synthesis/` |
| Directory of bullet-outline pages | `particles/exporters/logseq/` | Logseq's native format; particle IDs as block UUIDs for cross-page citation |
| External HTTP API (no file) | `particles/exporters/notion.py` | The first API-target exporter; idempotent upsert into one Notion database. Reference for the credential pattern below. |
| Single self-contained `.html` graph | `particles/exporters/graph/` | Scoped epistemic graph view: mandatory scope + disclosed caps; vendored Cytoscape.js inlined so the artifact works offline. The canonical contract notes live in `particles/exporters/AGENTS.md`. |

## Cross-exporter contract

Every shipped exporter — and yours — must honour two options
:

| Option | Type | Default | Behaviour |
|---|---|---|---|
| `min_particle_confidence` | `float` | `0.0` | Drop particles below this `effective_confidence` BEFORE any per-exporter step. Filter input is *effective* confidence, never raw `confidence.value`. |
| `min_particles` | `int` | per-exporter | Minimum *post-filter* particle count required to render a subject. |

Your summary must include `particles_dropped_below_threshold: int`
when the threshold is non-zero (inherited from `BaseExporterSummary`).

## Output shape

The `output` path is:
- a **directory** for vault-style outputs (Obsidian, Wiki)
- a **file** for single-file outputs (Anki, JSON Lines)
- `None` for API-based outputs (Notion, GitHub Pages push)

Your exporter is responsible for `output.parent.mkdir(...)` / atomic
writes / cleanup of stale entries from prior runs.

## API-target exporters & credentials

If your exporter writes to an external HTTP API (it has `output=None`)
and needs a credential, follow the Notion exporter's pattern:

1. Add a `get_<x>_api_key[_optional]()` getter to
   `particles/secrets.py` (raise when the target has no anonymous mode,
   optional when it does). The token lives **only** there — never in
   `config.yaml`, never on `ParticlesConfig`, never a CLI flag.
2. Set a class attribute `REQUIRES_SECRET = "<ENV_VAR>"` so the CLI
   pre-flight verifies the env var is present before any work, and call
   the getter as the **first statement** of `export()` (the
   authoritative, partial-write-proof check).
3. Put non-secret target parameters (ids, property names) in a config
   sub-model + `config.yaml.sample`.
4. Make `--dry-run` issue zero API writes.

The full contract: [`particles/exporters/AGENTS.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/exporters/AGENTS.md)
§ API-target exporters & the credential pattern.

## Reaching across the seam

Exporters may call `store/` helpers directly for simple list / get
queries (the rule of thumb: the second time you copy a query, lift
it into `store/`). The shared helpers already exposed:

- `get_particles_by_status(session, Status.ACTIVE)` — ACTIVE particles
- `list_all_subjects(session)` — every Subject
- `list_particle_subject_pairs(session)` — the join table
- `get_entry_uri_map(session, entry_ids=None)` — corpus URLs

Don't recreate these in your exporter.

## Synthesis-ready exporters

If your exporter wants per-Subject LLM-synthesised prose, import
from `particles/exporters/article_synthesis/` rather than
reimplementing. The helper provides the cache key, citation
validation, Layer-B judge, and the fallback structured-listing
render. The Obsidian, Wiki, and Logseq exporters all share this
machinery.

The canonical contract: [`particles/exporters/AGENTS.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/exporters/AGENTS.md).
