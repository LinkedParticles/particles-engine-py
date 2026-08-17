# Authoring a conformance fixture

Conformance checks an extractor's *completeness* — does it populate
the REQUIRED / RECOMMENDED fields of the particle schema? Distinct
from benchmark (which checks correctness).

## Phase 1 (current): report-only

conformance Phase 2 (CI-blocking gate) is deferred to
post-1.0. v1.0.0 ships with Phase 1: the validator runs, prints a
report, but does not block merges. Run manually:

```bash
uv run particles extractor conform <extractor-id>
```

Two prerequisites un-defer Phase 2 (open ADRs will handle them):

1. **Diversity rule decision** — `uncertainty_nature` is REQUIRED
   to have ≥2 distinct values, but every shipped extractor correctly
   hardcodes EPISTEMIC. The rule needs to be removed, generalised,
   or downgraded to ADVISORY.
2. **Fixture corpus coverage** — only 6 of the ~14 registered
   extractors have fixtures today. The rest need realistic fixtures
   before Phase 2 can flip.

## Fixture format

Each fixture is a directory under `tests/conformance/fixtures/<fixture-id>/`
with exactly three files:

| File | Contents |
|---|---|
| `manifest.yaml` | `fixture_id`, `source_type`, `expected_acceptors: [<extractor-id>, …]`, optional `notes` |
| `content.bin` | Raw bytes the extractor receives (API response / HTML / etc.) |
| `snapshot.json` | Serialised `Snapshot` (sha256, etag, content_published_at, …) |

Plus an entry in the top-level `tests/conformance/fixtures/MANIFEST.yaml`.

## Worked example layout

```
tests/conformance/fixtures/numista-coin-001/
├── manifest.yaml
├── content.bin        # 18 KB JSON from the Numista API
└── snapshot.json      # Snapshot row mirroring what deposit would create
```

Use `numista-coin-001/` as the reference layout when adding a new
fixture.

## What "completeness" actually measures

The validator walks the contract in `particles/conformance/contract.py`
against the particles your extractor emits, then reports per-field
population rate. Buckets:

- **REQUIRED** — must be populated in 100 % of emitted particles
- **RECOMMENDED** — soft target (default 80 %; `--recommended-threshold`)
- **OPTIONAL** — informational only

The contract is the source of truth. Don't introduce extractor-side
checks against population rate — the validator does that uniformly.

## Adding or modifying a fixture invalidates prior reports

`fixture_corpus_hash` is deterministic over `(fixture_id, content,
snapshot)` tuples. Two reports are only comparable if
their corpus hashes match. This is intentional: comparing reports
across an evolving fixture corpus is meaningless.

The canonical contract: [`particles/conformance/AGENTS.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/conformance/AGENTS.md).
