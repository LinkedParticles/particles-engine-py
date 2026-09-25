# Authoring a conformance fixture

Conformance checks an extractor's *completeness*: does it populate
the REQUIRED / RECOMMENDED fields of the particle schema? Distinct
from benchmark (which checks correctness).

`particles extractor conform` is the command this page is about. It is
**not** the same as `particles conformance check`, which tests whether an
implementation reproduces the deterministic test vectors of the
[Conformance Profile](https://linkedparticles.org/spec/conformance-profile/).
Fixture authors only deal with the first one.

## Phase 1 (current): report-only

Conformance runs in Phase 1: the validator runs and prints a report,
but CI does not block merges or extractor registration on the result.
A CI-blocking Phase 2 is deferred until after 1.0. Run manually:

```bash
uv run particles extractor conform <extractor-id>
```

Useful flags:

| Flag | Effect |
|---|---|
| `--fixtures <dir>` | Load fixtures from another directory. Default: `tests/conformance/fixtures`, resolved **relative to the working directory**, so run from the repository root |
| `--recommended-threshold <0–1>` | The RECOMMENDED floor (default `0.8`) |
| `--format table\|json` | Output format |
| `--fail-on error\|warn` | `error` (default) exits 1 when a REQUIRED field fails; `warn` also exits 1 on RECOMMENDED warnings |
| `--all-accepted` | Widen the fixture set (see [Which fixtures a run scores](#which-fixtures-a-run-scores)) |

An unknown extractor id exits 2.

Report-only does not mean inert: an operator can opt in to the
[conformance trust cap](../operator-guide/tuning.md#conformance-trust-cap),
which clamps the effective trust weight of an extractor whose last report
showed a genuinely evaluable REQUIRED failure. A normal `conform` run
stores that verdict for the extractor when a store is available. A fixture
you author here is what that lever reads.

What still stands between Phase 1 and a blocking Phase 2 is **fixture
corpus coverage**. Every registered extractor except four has at least
one fixture today. The four without one are the three `github-*`
extractors and `docstring-extractor`. A gate run against the current
corpus would fail those four at 0 % coverage. That would be a
missing-data failure, not a real regression. New fixtures for those
extractors are the most useful contribution here.

## Fixture format

Each fixture is a directory under `tests/conformance/fixtures/<fixture-id>/`
with exactly three files:

| File | Contents |
|---|---|
| `manifest.yaml` | `fixture_id`, `source_type`, `expected_acceptors: [<extractor-id>, …]`, optional `notes` describing what the fixture exercises |
| `content.bin` | Raw bytes the extractor receives (API response / HTML / etc.) |
| `snapshot.json` | Serialised `Snapshot` (sha256, etag, content_published_at, …) |

Each fixture also needs an entry in the top-level
`tests/conformance/fixtures/MANIFEST.yaml`.

Discovery does **not** read `MANIFEST.yaml`. The loader walks the
directory and loads every subdirectory that has a `manifest.yaml`,
sorted by name. It skips hidden directories (`.`-prefixed),
`__`-prefixed directories, and subdirectories with no `manifest.yaml`.
A subdirectory that has a `manifest.yaml` but lacks `content.bin` or
`snapshot.json` raises an error and stops the run. Two
things follow from that:

- A fixture you forget to list in `MANIFEST.yaml` is still live in every
  run and still counts toward the corpus hash. Keep the list complete
  anyway, because readers use it to see what the corpus covers.
- A work-in-progress directory without a `manifest.yaml` is ignored.
  Once you add the manifest, the other two files must exist.

## Worked example layout

```
tests/conformance/fixtures/numista-coin-001/
├── manifest.yaml
├── content.bin        # JSON from the Numista API, trimmed to what the extractor reads
└── snapshot.json      # Snapshot row mirroring what deposit would create
```

Use `numista-coin-001/` as the reference layout when adding a new
fixture.

## Adding a fixture

**Fastest path.** If the source is already deposited in a local store, run:

```bash
uv run particles extractor generate-fixture <entry-id>
```

It takes the entry's latest RESPONSE snapshot and its stored blob, writes
the three files, and adds the entry to `MANIFEST.yaml`. It accepts an
entry-id prefix. Options: `--id` (fixture id; the default is derived from
the entry URI and id), `--source-type` (overrides the entry's type),
`--output-dir`, and `--force` (overwrite an existing fixture directory).
It leaves `expected_acceptors: []` and a stub `notes:` on purpose. Fill
both in yourself (step 5 below) once you have checked that the fixture
exercises the extractor's real code paths. The store-free writer behind
the command is `particles.conformance.fixtures.write_fixture`, if you
want to script it.

**By hand:**

1. Pick a `<fixture-id>`: lowercase-kebab, unique across the corpus, and
   ending in a counter (`web-article-002`). The corpus hash and every
   report use the `fixture_id` in `manifest.yaml`, not the directory name,
   so keep the two identical.
2. Create `tests/conformance/fixtures/<fixture-id>/`.
3. Add the three files. `content.bin` should be a realistic blob that
   runs through the extractor's normal code paths. Trim it to the
   fields the extractor actually reads. The trimmed blob then also shows
   readers which part of the source schema the extractor depends on.
4. Add an entry to `tests/conformance/fixtures/MANIFEST.yaml`.
5. Set `expected_acceptors` to every extractor that *should* match the
   source. This field records **intent**. It does not control which
   extractor scores the fixture (the registry decides that; see below).
   Fill it in accurately anyway. The test suite checks that the extractor
   the registry actually routes the fixture to is listed here. It also
   checks that each fixture is selected by exactly one extractor. If
   either check fails, work out which side is wrong. Usually an
   extractor's `accepts()` or its position in the registry has moved.

## Which fixtures a run scores

A run scores the fixtures whose `source_type` the **production
registry routes** to the extractor under test. This is the same
predicate the extract pipeline uses. It is not "every fixture the
extractor's `accepts()` would take". The general extractor is the
fallback, so its `accepts()` is always true. Scoring it on everything
it accepts would measure REQUIRED coverage over inputs the pipeline
never sends it. Because `subject_ids` must reach 100 %, the choice of
fixtures decides the verdict.

`--all-accepted` restores that wider set, for deliberate probes such as
"what would the fallback do with a Wikibase blob?". A widened run is
report-only and **never stores** the trust-cap verdict, because the
stored verdict describes production behaviour. Both modes write the
fixture set they scored into `quality_notes[0]`. Check that note to
see which kind of run produced a saved report.

## Which pairing a run reports

`ConformanceReport.extraction_provider_model` names the
`"<provider>:<model>"` pair that produced the scored particles. The
report reads it from the particles the run just minted, which record it
at the completion call. It does not come from the configuration file.
It is `null` for a deterministic extractor, even one that fetches from
the network, because such an extractor makes no completion call. If a
single run shows more than one pairing, the report records all of them
plus a quality note saying the report is not a valid baseline. That
only happens when the configuration is reloaded during a run.

## What "completeness" actually measures

The validator walks the contract in `particles/conformance/contract.py`
against the particles your extractor emits, then reports per-field
population rate. Buckets:

- **REQUIRED**: must be populated in 100 % of emitted particles
- **RECOMMENDED**: soft target (default 80 %; `--recommended-threshold`)
- **OPTIONAL**: informational only

The contract is the source of truth. Don't introduce extractor-side
checks against population rate; the validator does that uniformly.

Things to know when reading a report:

1. **Candidates are converted before they are measured.** Each candidate
   goes through `candidate_to_particle()`, the same step the real
   pipeline uses. The report therefore describes what would be
   persisted, not the raw candidate. During a fixture run, a
   candidate's subject names stand in for subject ids. Only presence is
   measured, not whether the names resolve.
2. **"Populated" means non-null and non-empty.** An empty string, list,
   or dict counts as unpopulated.
3. **List paths use `[]`.** `provenance[].snapshot_id` resolves to the
   `snapshot_id` of every provenance entry. If *any* of them is missing,
   the whole path counts as unpopulated for that particle.
4. **Enum fields also get a histogram.** For `uncertainty_nature`,
   `particle_type`, `status`, `confidence.calibration_source`, and
   `canonical_form`, the report adds a `value_counts` histogram and a
   distinct-value count.
5. **A diversity rule can apply on top of the rate.** A diversity rule
   has a severity. At `FAIL`, a violation fails the field even at 100 %
   population. At `ADVISORY`, the violation goes into `advisories` and
   never changes `passed`, the `--fail-on` exit code, or the trust cap.
   The only diversity rule today is on `uncertainty_nature` (at least 2
   distinct values), and it is `ADVISORY`. A structured extractor that
   reports one distinct value there is often correct, because many
   source vocabularies carry no signal about stochastic quantities. An
   LLM-backed extractor may pass on one capture and fail on the next.
   **Do not "fix" the advisory by emitting token `ALEATORY` values.** A
   made-up epistemic classification is exactly what the rule is meant
   to catch. On a sampled extractor, look at the `value_counts` margin,
   not the distinct count.
6. **Some rates have a smaller denominator than the run.** `subject_ids`
   is measured only over particles that the specification says should
   carry a subject. These are left out: non-CLAIM particles,
   document-metadata claims, non-asserted (declined / hypothetical)
   claims, and author-scoped claims. They are counted in
   `FieldStat.excluded_count`, not as failures. Check that count before
   comparing two rates. If every particle is excluded, the field is
   reported as **unevaluated** (rate 0.0, not passing, "all exempt"
   reason), never as 100 %. An extractor cannot declare exemptions for
   itself. The exempt classes come from the specification.
7. **A run with no fixtures shows REQUIRED at 0 %.** If the corpus routes
   nothing to an extractor, every REQUIRED field reads 0 % and a quality
   note says no fixture was scored. This means "no data", not
   "regression", and it never triggers the trust cap. Check
   `particle_count` and `quality_notes` before acting on a 0 %.
8. **Unprefixed `properties` keys produce warnings.** Each key in
   `properties` must have the form `prefix:LocalName`. Keys without a
   `:` are listed in `quality_notes`. They do not fail the run.
9. **Completeness, not correctness.** The validator asks "was this field
   populated?", never "was it populated with the *right* value?". Use the
   [benchmark suites](benchmark-suites.md) for correctness.

## Validation layers

The conformance validator does not wrap JSON Schema and SHACL. It is a
separate, fourth layer:

| Layer | Where | What it enforces |
|---|---|---|
| 1. Pydantic | `particles.core.schema.Particle` construction | Types, `min_length`, presence of fields the model requires; raises on violation. Runs whenever a particle is built, including inside a conform run |
| 2. JSON Schema | `particles.conformance.jsonschema` | The spec-level contract in `particle.schema.json` (Draft 7). Called through the library API |
| 3. SHACL | `particles.conformance.shacl` | RDF-shape validation of JSON-LD records. Called through the library API |
| 4. Conformance | `particles.conformance.validator` | Per-extractor field-population *rates* across the fixture corpus |

Layers 1–3 give a pass or fail per record. Layer 4 is coverage
telemetry: it tells you what fraction of emitted particles populated
each contract field. `extractor conform` runs layers 1 and 4. It does
not run layers 2 and 3 over your fixture output, so call them yourself
if your extractor emits JSON-LD that has to interoperate.

## Artifact paths

Layers 2 and 3 read the normative artifacts from `artifacts/schemas/`
in a source checkout. An installed wheel packages them as
`particles/_artifacts/schemas/` and prefers that copy.

- `particle.schema.json`: JSON Schema (Draft 7)
- `context.jsonld`: JSON-LD `@context` for RDF interchange
- `shacl/{ParticleShape,SubjectShape,CorpusSnapshotShape,ProvenanceChainShape,TrustStatementShape}.ttl`:
  the five normative SHACL shapes

If a file or its library (`jsonschema`, `pyshacl`) is missing, that
layer is skipped with a logged warning rather than raising. A skip is
**not** a pass. Confirm the artifacts are present before you read a
clean layer-2/3 result as meaningful.

## Adding or modifying a fixture invalidates prior reports

`fixture_corpus_hash` is a SHA-256 over the `(fixture_id, content,
snapshot)` tuples of every discovered fixture, sorted by `fixture_id`.
The hash covers the **whole corpus**, not only the fixtures a given
extractor scored. Adding, editing, renaming, or removing any fixture
therefore changes the hash for every extractor's report. Two reports
can only be compared if their corpus hashes match. This is intentional:
comparing reports across a changing fixture corpus tells you nothing.

The contract the validator walks is
[`particles/conformance/contract.py`](https://github.com/LinkedParticles/particles-core-py/blob/main/particles/conformance/contract.py).
Conformance is Client-layer, so it ships in `linkedparticles-core` and its
source lives in the `particles-core-py` repository.
