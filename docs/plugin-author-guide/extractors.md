# Writing an extractor

An extractor turns raw source bytes into structured
`CandidateParticle` instances. It's the SDK's pluggable interface
to a new domain.

## The contract (in 30 seconds)

```python
from particles.core.schema import ApplicabilityClause

# Module-level identity. Stored in the database; see "Identity constants" below.
SOURCE_TYPE = "MY_SOURCE_TYPE"
EXTRACTOR_ID = "my-extractor"
EXTRACTOR_VERSION = "0.1.0"
DEFAULT_TRUST_WEIGHT = 0.50

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",                                   # RFC 2119 keyword
        domain_uri="http://www.wikidata.org/entity/Q…",
        domain_label="short human label",
        source_types=[SOURCE_TYPE],
    )
]


class MyExtractor:
    EXTRACTOR_ID = EXTRACTOR_ID                # unique slug
    EXTRACTOR_VERSION = EXTRACTOR_VERSION      # SemVer; bump when behaviour changes
    DEFAULT_TRUST_WEIGHT = DEFAULT_TRUST_WEIGHT  # in [0.0, 1.0]; demotion-only
    APPLICABILITY = APPLICABILITY

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        ...
        return ExtractionResult(candidates=[...], quality_notes=[...])
```

Then add one line to `particles/extraction/registry.py::_make_extractors()`.
That's it.

The protocol itself requires only `EXTRACTOR_ID`, `EXTRACTOR_VERSION`,
`accepts()` and `extract()`. `DEFAULT_TRUST_WEIGHT` and `APPLICABILITY` are
read with `getattr`, so omitting them does not fail at import, but every
shipped extractor sets both and you should too: an extractor without
`DEFAULT_TRUST_WEIGHT` is registered at a silent `0.7`, and one without
`APPLICABILITY` gives its source type no domain (unless the operator maps
one in config), so domain-scoped trust and domain-gated subject authorities
cannot see its particles.

`DEFAULT_TRUST_WEIGHT` is the value you ship; it is written to the store once,
when the extractor is first registered, and an operator can demote it further
from there. It is one of the four factors composing
[effective confidence](../user-guide/concepts.md#confidence); see
[Operator guide → extractor trust weight](../operator-guide/tuning.md#extractor-trust-weight)
for the lever on the other end. Because the stored value is never overwritten
by a later registration, changing the default in a new release affects only
stores that have never seen your extractor.

## Worked examples in the tree

| Style | File | Notes |
|---|---|---|
| HTTP API → JSON parse | `particles/extraction/numista/` | Three extractors (coin, issuer, listing) for one host; structured `properties` particles |
| HTML scraping | `particles/extraction/github/` | repo + gist + pages variants; `_shared.py` helper |
| LLM-extracted prose | `particles/extraction/general.py` | The catch-all; calibrates per-chunk |
| Social link card | `particles/extraction/reddit.py` | The reference implementation of the two-step LLM shape below |
| Hybrid prose + metadata | `particles/extraction/hackernews.py`, `mastodon.py` | The reference implementations of the three-step LLM shape below |
| Deterministic AST parse | `particles/extraction/docstrings.py` | LLM-free; `PYTHON_SOURCE` docstrings → one particle per symbol, fixed `0.95` confidence, no calibration / benchmark. Reads the file's dotted module path from the `entry_uri_r` kwarg |
| Structure-canonical parse | `particles/extraction/rdf.py` | LLM-free **and** network-free; `RDF_GRAPH` → one particle per triple, `canonical_form: STRUCTURED`: the triple is the assertion and `content` is a derived verbalization |
| Migration from another store | `particles/extraction/mcp_memory.py` | LLM-free **and** network-free; reads an `@modelcontextprotocol/server-memory` `memory.jsonl` export and turns each record into a particle. The special case worth reading is not the parsing but the *attribution*; see below |
| Structure-canonical API reading | `particles/extraction/wikidata.py` | LLM-free; `WIKIDATA_API` → one particle per statement, carrying the `wd:` / `wdt:` triple Wikidata published. Labels for `content` are fetched live, the documented exception below |

## Identity constants: what must never change

Each extractor module declares its identity as module-level constants and
mirrors them as class attributes. They are persisted, so they carry
compatibility rules that no type signature states:

| Constant | Stored where | Rule |
|---|---|---|
| `SOURCE_TYPE` | on every corpus entry, and as the key of applicability clauses, trust statements, and several config lists | Unique across the registry and **never renamed**. A rename orphans every stored entry of the old type: nothing will route to your extractor again. |
| `EXTRACTOR_ID` | on every particle's provenance, and as the key of the stored extractor record (its trust weight, its conformance verdict) | Unique and **never renamed**. A rename registers a brand-new extractor at the default trust weight and strands the operator's settings on the old id. |
| `EXTRACTOR_VERSION` | on every particle's provenance | SemVer, bumped whenever output changes; see [Modifying an existing extractor](#modifying-an-existing-extractor). |
| `DEFAULT_TRUST_WEIGHT` | the extractor record, on first registration only | In `[0.0, 1.0]`. |
| `APPLICABILITY` | the extractor record | A list of `ApplicabilityClause`s (`MUST` / `SHOULD` / `MUST_NOT` over a domain and a list of source types). The first `MUST` clause covering a source type names that source type's domain. |

Naming: `SOURCE_TYPE` is `UPPER_SNAKE_CASE`; `EXTRACTOR_ID` is a lower-case,
hyphenated slug, conventionally ending `-extractor` (`numista-coin-extractor`, `general-extractor`).

## Registering: the two-file rule

A new extractor is one new module under `particles/extraction/` plus one line
in
[`registry.py::_make_extractors()`](https://github.com/LinkedParticles/particles-core-py/blob/main/particles/extraction/registry.py).
No pipeline, schema, store, CLI or API change is needed.

```python
def _make_extractors() -> list[ExtractorPlugin]:
    from particles.extraction.myextractor import MyExtractor
    ...
    return [
        ...,
        MyExtractor(),      # before GeneralExtractor
        GeneralExtractor(),  # fallback: must stay last
    ]
```

Import your class **inside** the factory, as the existing entries do, so a
broken import in one plugin cannot break every consumer of the registry.

**Order is routing.** The pipeline hands a snapshot to the first extractor, in
registry order, that has no `MUST_NOT` clause for its source type and whose
`accepts()` returns `True`. `GeneralExtractor` accepts every source type, so an
entry placed after it is never reached. Keep `accepts()` narrow: return `True`
only for the source types you handle, or you will capture another extractor's
traffic.

**Extractors are Client-layer code.** They produce store-free *candidates*;
the Engine reconciles and persists them. Your module must not import
`particles.store`, `particles.corpus`, `particles.db`, or `particles.ingest`;
the project's import-layer check fails the build if it does. Fetching a URL
and writing the corpus blob is a separate role, the **importer**, which lives
on the Engine side in `particles/ingest/importers/`; many sources need none,
because a plain HTTP GET or a file deposit is the default.

### One host, several source types: the package layout

When one host emits more than one `SOURCE_TYPE` and the code outgrows a single
readable file, make the module a package: one extractor per file plus a shared
helper module. `particles/extraction/github/` is the reference:

```
particles/extraction/github/
    __init__.py     # re-exports identity constants + extractor classes
    _shared.py      # auth helpers, HTTP retry policy, URL parsing
    repo.py         # GitHubRepoExtractor
    gist.py         # GitHubGistExtractor
    pages.py        # GitHubPagesExtractor
```

The package `__init__.py` re-exports every `SOURCE_TYPE_*`, `EXTRACTOR_ID_*`,
`EXTRACTOR_VERSION_*`, and extractor class, so the registry and callers import
from `particles.extraction.<host>` whichever layout you choose. Use the single
module until a host genuinely has more than one extractor.

## What the pipeline passes to `extract()`

`extract()` receives the `Snapshot`, the stored `content` bytes, and keyword
arguments. Always accept `**kwargs` and ignore the ones you do not need; new
ones are added over time. Today the pipeline passes:

| Kwarg | Meaning |
|---|---|
| `session` | The database session. Pass it through to helpers that need it; never use it to import or call the store yourself. |
| `corpus_entry_id` | The corpus entry being extracted. |
| `source_type` | The entry's source type. |
| `entry_uri_r` | The entry's URL, for extractors that parse identity out of it (GitHub repo paths, the docstring extractor's module path). |
| `deposited_by` | Who deposited the entry. Only migration extractors use it; see below. |
| `completion_pool` | A shared batch for LLM requests. Only pool-aware LLM extractors use it. |
| `supersede_ids` | Set during reindex. If you use the carry-forward helper, pass it through, or reindex treats the particles it is replacing as cache hits and never re-runs the model. |

## LLM-driven extractors: two shapes

These are conventions, not protocol extensions: `ExtractorPlugin` stays four
members. Pick one shape up front.

**Shape 1: two-step.** For sources whose whole substance is prose. Split
`extract()` into two private methods:

```python
def _normalise(self, content: bytes, snapshot: Snapshot) -> NormalizedDocument:
    """Source-format parsing only: JSON / HTML / etc. → prose chunks.
    No LLM calls. Surface author_id, content_published_at, quality_notes,
    and any domain-injected subjects as fields on NormalizedDocument.
    """
    ...

async def _extract_claims(self, doc: NormalizedDocument, **kwargs) -> ExtractionResult:
    """LLM claim extraction over the NormalizedDocument. Typically
    extract_with_carry_forward(doc.chunks, ...) plus any post-extraction
    stamping (e.g. injecting doc.injected_subjects on every candidate).
    """
    ...

async def extract(self, snapshot: Snapshot, content: bytes, **kwargs) -> ExtractionResult:
    doc = self._normalise(content, snapshot)
    return await self._extract_claims(doc, **kwargs)
```

`NormalizedDocument` is defined in `particles/extraction/general.py`.
`RedditExtractor` is the reference implementation. The carry-forward helper
hashes each chunk's prompt text so an unchanged chunk re-uses its existing
particles instead of calling the model again.

**Shape 2: three-step.** For hybrid sources whose blob also carries structured
metadata (scores, counts, identifiers, instance or host info) that belongs in a
particle's `properties` rather than in the prose the model sees. `_normalise`
returns the prose document *and* the parse context, and `extract()` prepends a
synthesised metadata candidate:

```python
def _normalise(self, content: bytes, snapshot: Snapshot) -> tuple[NormalizedDocument, Ctx]:
    ...

async def extract(self, snapshot, content, **kwargs):
    doc, ctx = self._normalise(content, snapshot)
    result = await self._extract_claims(doc, **kwargs)
    meta = _build_<noun>_meta_candidate(ctx, doc.injected_subjects)
    if meta is not None:
        result.candidates.insert(0, meta)
    return result
```

References: `HackerNewsExtractor` (`_build_story_meta_candidate`) and
`MastodonExtractor` (`_build_status_meta_candidate`). Pick Shape 2 when the
source carries structured metadata that should land in `properties`, or
identifiers a future relation kind will need (see
[Relation kinds](#relation-kinds)). Its rules:

1. The synthesiser is a **module-level function**,
   `_build_<noun>_meta_candidate(ctx, injected_subjects) -> CandidateParticle | None`,
   so it can be unit-tested without instantiating the extractor.
2. It **returns `None`** rather than fabricating a metadata particle from
   defaults when the source is missing the fields that identify it. The
   model-derived claims then stand alone.
3. The metadata candidate goes **first** in `result.candidates`, so output
   order is deterministic.
4. It uses `UncertaintyNature.EPISTEMIC` (engagement numbers are observed, not
   inferred) and a confidence of `0.95`, since it is read directly from
   structured data.

A **structured / deterministic** extractor (Wikidata, Numista, Nomisma, RDF,
the docstring extractor) uses neither shape: it has no prose stage and no LLM
call, overrides `extract()` directly, and returns a fixed candidate list for a
fixed input blob.

### The extraction-model stamp

Every particle records the `"<provider>:<model>"` pairing that produced it, in
`Particle.extraction_provider_model`; operators use it to re-extract exactly
what one model produced (`particles reindex --provider-model`). The stamp is
written **at the completion call**, not by your extractor logic and not by the
pipeline: the call site uses `particles.llm.complete_with_provider_model`, which
returns `(text, provider_model)`, and copies the pairing onto
`CandidateParticle.provider_model` for every candidate it parsed.

- **Route through an existing call site and you have nothing to do.** The
  general extractor's `_call_llm` and the journal extractor's
  `_call_journal_llm` already stamp; the carry-forward helper goes through the
  former.
- **If you call the model directly**, call `complete_with_provider_model`
  (not plain `complete()`) and set `candidate.provider_model`, or your
  particles are silently unstamped.
- **A deterministic extractor stamps nothing.** Its candidates carry `None`,
  which is the correct record. Do not fill it from the configured LLM: that is
  false provenance, and it would make `reindex --provider-model` sweep in
  particles no model touched.
- **Never recompute or backfill the stamp.** Carried-forward particles keep
  the pairing that actually produced them.

## Structure-canonical extractors

Almost every extractor produces `canonical_form: PROSE`: `content` is the
assertion, and any `structured_claim` annotates it. A structure-native source
inverts that: the RDF extractor parses a triple that *is* the assertion and
derives readable prose from it, so the prose is what may be regenerated. If you
write one, these rules follow that do not apply elsewhere:

- **Build the triple from the deposited bytes alone.** It is the assertion, so
  it must be bit-identical on every re-extraction of the same snapshot.
- **Prefer to build `content` from those bytes too, and say so when you can't.**
  A derived `content` should be reproducible, which for the RDF extractor means
  every label comes from the document; it never fetches. `wikidata.py` is the
  one documented exception: a Wikibase entity blob carries labels for itself and
  for nothing it references, so rendering `P19 → Q350` as "place of birth:
  Cambridge" is only possible from the API that published the statement. Its
  module docstring records the trade; do not copy the pattern without the same
  argument, and never let a fetch reach the triple.
- **Improving the verbalization is an `EXTRACTOR_VERSION` bump plus
  `particles reindex --extractor-version <old>`**, never an edit of stored
  rows, and never `particles structure`: the backfill regenerates *derived*
  annotations and skips `STRUCTURED` particles for exactly this reason. (The
  operator's view of both verbs:
  [reindex](../operator-guide/lint-and-review.md#reindex) and
  [structured claims](../operator-guide/structured-claims.md).)
  "Derived" describes what a regeneration pass may produce; it is not a licence
  to mutate `content`, which stays immutable.
- **`content` must never be empty.** `Particle.content` requires at least one
  character. End your verbalization ladder at something that always exists;
  the RDF extractor's last rung is the full IRI.
- **`STRUCTURED` is a property of a candidate, not of an extractor.** A
  candidate is `STRUCTURED` exactly when its `content` is a deterministic
  rendering of its `structured_claim` **and of no other fact**, so one
  `extract()` call may emit both forms (the Numista coin extractor does).
  Three shapes recur:
    - *one parsed field, templated* → `STRUCTURED`;
    - *an entity infobox* (one subject, several properties, one summary
      sentence) → `PROSE`, because `content` states several facts and a
      triple states one. It may still carry a `structured_claim` annotation
      (for example an `rdf:type` triple); Nomisma's candidates are this shape;
    - *the source's own free text, passed through* → `PROSE`, with no
      annotation. That `content` is asserted by the source, and marking it
      `STRUCTURED` would claim the SDK may regenerate someone else's words.

Set `canonical_form` on the `CandidateParticle`; `candidate_to_particle` passes
it through. A candidate that claims `STRUCTURED` without a `structured_claim`
is demoted to `PROSE` with a logged warning rather than raising, so one
malformed candidate cannot lose the whole pass; do not rely on it. Attach
`external_refs` keyed by subject name so the triple's IRI subject can be bound
to the resolved Subject; without them the claim's subject stays unbound for
exactly the particles carrying the best identifiers.

A structured extractor carries no benchmark / ECE gate, because
there is no probabilistic output to calibrate (calibration is
identity; the operator-side picture is
[extractor calibration](../operator-guide/tuning.md#extractor-calibration)).
When a structured extractor mints code-like subject names (snake_case
identifiers, dotted paths), exempt its source type from the non-entity subject
gate via `subject_gate.exempt_source_types` so the names are not stripped; see
the `PYTHON_SOURCE` precedent.

## Relation kinds

Typed edges between particles use the `RelationType` enum in
`particles/core/schema.py`. **Never invent a kind string.** Some members are
active (`CO_EVIDENTIAL`, `PART_OF`, `SEQUENCE_IN`, `ENDORSES`, `DISPUTES`, and
`CONTRADICTS`, which only the Engine's reconciliation writes, for a
contradiction it leaves standing between two projects); others are reserved
names with no consumer yet (`BOOSTS`, `QUOTES`, `REPLIES_TO`, `MENTIONS`). Emitting a reserved kind first needs its
consumer surface (query filter, CLI parsing, lint) built, and a kind not in the
enum needs the enum extended; both are design changes to propose before
building, not something an extractor does alone.

Extractors do not write relations: the Engine writes them after reconciliation,
once both endpoint ids exist. What an extractor *should* do, when its source
carries the identifiers a reserved kind will need, is capture them as
namespaced `properties`, so a later backfill can build the edges without
re-reading the blobs. The Mastodon extractor records
`mastodon:reblogOfStatusId`, `mastodon:reblogOfAccountAcct` and
`mastodon:reblogOfStatusUri` for the future `BOOSTS` kind.

## Modifying an existing extractor

Bump `EXTRACTOR_VERSION` (patch or minor) whenever a change would produce
different particles from the same source bytes: new or changed prompts,
parsing, confidence, subjects, properties, verbalizations. The version is what
makes the change reach existing data: `particles reindex --extractor-version
<old>` finds and re-extracts every particle the prior version produced, and the
carry-forward cache is keyed on it, so without a bump unchanged chunks keep
their old particles. Do not rename `SOURCE_TYPE` or `EXTRACTOR_ID` as part of a
modification (see [Identity constants](#identity-constants-what-must-never-change)).

## Extractor configuration parameters

Tuneable parameters (item limits, score thresholds, timeouts) go in the
configuration model, never in module-level constants or `os.environ.get()`:

1. Add a sub-model in `particles/config.py`:
   ```python
   class MyExtractorConfig(BaseModel):
       max_items: int = 50
       min_score: int = 2
   ```
2. Add it to `ParticlesConfig` as
   `my_extractor: MyExtractorConfig = Field(default_factory=MyExtractorConfig)`.
3. Because extractors are Client-layer, add the section name to
   `CLIENT_SECTIONS` in `particles/config.py`, and add the section to
   `config.yaml.sample` with a comment on each field and the `# [client]` tag
   on its header line. A test fails if the two disagree. (What the tag means to
   an operator: [Configuration](../operator-guide/configuration.md#which-sections-apply-to-your-install).)
4. Read it **inside the function that uses it**, at call time:
   `get_config().my_extractor.max_items`. Reading it at import or in
   `__init__` freezes the value and ignores config reloads.

Secrets (API tokens) never go in config; they are read through
`particles.secrets`.

## Things you must do

- **Define the identity constants** described above.
  The version is what the chunk-hash cache keys on for
  carry-forward; bumping it invalidates prior particles from this
  extractor.
- **Implement `accepts(source_type) -> bool`.** It routes snapshots to you,
  and the benchmark / conformance runners use the same routing to pick
  applicable suites.
- **Return `ExtractionResult(candidates=..., quality_notes=...)`,
  not a list of `Particle`.** The pipeline wraps each
  `CandidateParticle` with extractor-agnostic provenance via
  `candidate_to_particle()`.
- **Set `uncertainty_nature=EPISTEMIC` for knowledge claims.**
  ALEATORY is reserved for claims about a genuinely stochastic quantity. See
  [conformance](conformance.md) for how the diversity rule treats this.

## Things you shouldn't do

- Don't call the database directly from the extractor. The pipeline
  passes you the snapshot + content; it stores the resulting
  particles. Side-effect-free `extract()` is the contract.
- Don't catch exceptions and return zero candidates silently; raise.
  The pipeline catches and records the failure properly.
- Don't read config at extractor `__init__` time; read it inside
  `extract()`. The pipeline may construct your extractor before
  config is loaded.

## Migration extractors: the attribution rules

A migration extractor reads **another memory store's export** and turns each
record into a particle. It is a structured (no-LLM) extractor: the records are
already claim-sized, and running them through the general extractor would
paraphrase atomic claims and add hallucination risk to content that had none.
`mcp_memory.py` is the reference. It has three extra obligations, all of which
exist so that a migrated belief is honest about being second-hand:

1. **Declare your confidence's origin.** Set
   `CandidateParticle.calibration_source = CalibrationSource.IMPORTED` and take
   the value from `config.migration.import_confidence`, one flat floor for the
   whole import. `candidate_to_particle` honours a declared source verbatim and
   skips temperature scaling, which would be meaningless over a number no model
   produced.

   **Never map the source store's own score onto `confidence_value`.** A
   confidence is fixed at creation and multiplies through every ranking, so a
   foreign scalar would silently re-rank the user's whole store. Preserve it as
   a tag instead (`<source>:score=…`), where it stays legible and inert.

2. **Cite the record, not the store you read it from.** Set
   `provenance_location` to the record's position in the deposited blob (a line
   number, a JSON pointer); it lands on the source provenance reference, so the
   claim points at bytes this store holds and hashed. **Never synthesise a
   provenance reference into the source store's own identifiers.** The SDK
   never fetched them and cannot re-verify them, so that would be fabricated
   provenance.

3. **Attribute the act.** Put a `ContributorRef` with role `importer` on
   `contributors`, built from the `deposited_by` kwarg the pipeline passes.
   Whatever the export records about *its* authors travels as `author` /
   `agent` contributors and `content_published_at`, never as provenance.

Give the format its own source-type string: it keys your `APPLICABILITY`, your
`DEFAULT_TRUST_WEIGHT`, the operator's source-type trust statement, and
`subjects.skip_live_authorities_source_types`. Add it to that last list, because
a per-entity live ontology lookup over a bulk export is slow,
network-dependent, and can rewrite canonical names the migrating user never
chose. (What that list switches off is described in
[Subject Authorities](subject-authorities.md).)

Carry the source store's entity identity as an `ExternalRef` in a namespace of
its own, via `CandidateParticle.external_refs`. That is what lets a second
export of the same store re-attach to the same Subjects instead of forking the
graph.

Make the import **rehearsable**. Write the mapping as one synchronous function
from the parsed export to an `ExtractionResult`, call it from `extract()`, and
call it again from a `preview_<format>_export(content)` function that hands the
result to `particles.extraction.migration_preview.build_preview`. That is what
the verb's `--dry-run` prints: the counts, what was dropped, and a sample,
computed without a store. The rule that matters is that the preview **runs**
your mapping and never re-implements it, because a report from a second code
path can agree with itself while disagreeing with the import. Count what your
parser cannot place (unknown fields, skipped records) and surface it as quality
notes, so the dry run and the real run disclose the same losses.

**An entity exists in the store only through a candidate that names it.** There
is no subject-only candidate, so two things follow for a format whose entities
can be empty. Put an entity's metadata (`subject_classes`, `external_refs`) on
*every* candidate that names it, edges included, because for an entity with no
records of its own the edge is the only candidate that will. And an entity no
candidate names does not migrate: do not invent a placeholder claim to carry
it, which would be a belief the source never held. Report it by name in
`quality_notes` instead, as `mcp_memory.py` does.

The **fetch** is not your job. A file-shaped export is deposited by the ordinary
file path and needs no plugin at all; a hosted store behind an authenticated API
needs an importer, which is a separate role, and your extractor runs unchanged
either way.

## Conformance + benchmark

Once your extractor works, write:

1. A [conformance fixture](conformance.md) so `particles extractor
   conform <id>` reports field-population rates against your source.
   Every REQUIRED field should be populated on 100 % of your output and every
   RECOMMENDED field above the `0.8` default threshold. If you only ever emit
   one `uncertainty_nature`, the diversity rule raises an advisory, not a
   failure; for a structured extractor that is the honest result, so do not
   manufacture an ALEATORY emission to clear it.
2. A [benchmark suite case](benchmark-suites.md) so you can pin
   precision / recall / calibration_error against a gold-standard
   expected-particle list. (Not applicable to deterministic extractors; see
   above.)

Conformance is report-only: it never changes the store or blocks
registration. An operator can, however, opt in to capping the effective trust
of an extractor with a genuine REQUIRED failure; see
[Conformance trust cap](../operator-guide/tuning.md#conformance-trust-cap).

## Checklist

- [ ] `SOURCE_TYPE` is unique, `UPPER_SNAKE_CASE`, and will not change
- [ ] `EXTRACTOR_ID` is a unique slug that will not change; `EXTRACTOR_VERSION` is SemVer; both are class attributes
- [ ] `EXTRACTOR_VERSION` is bumped if this changes existing extraction output
- [ ] `APPLICABILITY` and `DEFAULT_TRUST_WEIGHT` are set
- [ ] `accepts()` returns `True` only for the source types you handle
- [ ] `extract()` accepts `**kwargs`, raises on failure, and imports nothing from `store`, `corpus`, `db`, or `ingest`
- [ ] LLM calls stamp `provider_model` (or route through a call site that does); deterministic extractors stamp nothing
- [ ] Tuneable parameters live in a config sub-model, tagged `[client]`, read at call time
- [ ] Tests in `tests/test_<name>.py` cover `accepts()`, parsing, and any structured properties
- [ ] A conformance fixture exists and `particles extractor conform <id>` passes locally
- [ ] LLM-driven only: a benchmark suite exists and `particles extractor benchmark <id>` was run
- [ ] Registered in `_make_extractors()`, before `GeneralExtractor`

The canonical contract is the protocol itself, in
[`particles/extraction/registry.py`](https://github.com/LinkedParticles/particles-core-py/blob/main/particles/extraction/registry.py).
Extraction is Client-layer, so it ships in `linkedparticles-core` and its
source lives in the `particles-core-py` repository.
