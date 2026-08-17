# Writing an extractor

An extractor turns raw source bytes into structured
`CandidateParticle` instances. It's the SDK's pluggable interface
to a new domain.

## The contract (in 30 seconds)

```python
class MyExtractor:
    EXTRACTOR_ID = "my-extractor"          # unique slug
    EXTRACTOR_VERSION = "0.1.0"            # SemVer; bump when behaviour changes
    TRUST_WEIGHT = 1.0                     # in [0.0, 1.0]; demotion-only

    def accepts(self, source_type: str) -> bool:
        return source_type == "MY_SOURCE_TYPE"

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

## Worked examples in the tree

| Style | File | Notes |
|---|---|---|
| HTTP API → JSON parse | `particles/extraction/numista.py` | Single-URL fetch; structured `properties` particles |
| HTML scraping | `particles/extraction/github.py` | gist + repo + pages variants; `_shared.py` helper |
| LLM-extracted prose | `particles/extraction/general.py` | The catch-all; calibrates per-chunk |
| Social link card | `particles/extraction/reddit.py` | Implements `primary_url()` |
| Deterministic AST parse | `particles/extraction/docstrings.py` | LLM-free; `PYTHON_SOURCE` docstrings → one particle per symbol, fixed `0.95` confidence, no calibration / benchmark. Reads the file's dotted module path from the `entry_uri_r` kwarg |
| Structure-canonical parse | `particles/extraction/rdf.py` | LLM-free **and** network-free; `RDF_GRAPH` → one particle per triple, `canonical_form: STRUCTURED` — the triple is the assertion and `content` is a derived verbalization |
| Structure-canonical API reading | `particles/extraction/wikidata.py` | LLM-free; `WIKIDATA_API` → one particle per statement, carrying the `wd:` / `wdt:` triple Wikidata published. Labels for `content` are fetched live — the documented exception below |

**Structure-canonical extractors are a special case worth knowing about.**
Almost every extractor produces `canonical_form: PROSE` — `content` is the
assertion, and any `structured_claim` annotates it. A structure-native source
inverts that: the RDF extractor parses a triple that *is* the assertion and
derives readable prose from it, so the prose is what may be regenerated. If you
write one, three rules follow that do not apply elsewhere:

- **Build the triple from the deposited bytes alone.** It is the assertion, so
  it must be bit-identical on every re-extraction of the same snapshot.
- **Prefer to build `content` from those bytes too, and say so when you can't.**
  A derived `content` should be reproducible, which for the RDF extractor means
  every label comes from the document — it never fetches. `wikidata.py` is the
  one documented exception: a Wikibase entity blob carries labels for itself and
  for nothing it references, so rendering `P19 → Q350` as "place of birth:
  Cambridge" is only possible from the API that published the statement. Its
  module docstring records the trade; do not copy the pattern without the same
  argument, and never let a fetch reach the triple.
- **Improving the verbalization is an `EXTRACTOR_VERSION` bump plus
  `particles reindex --extractor-version <old>`**, never an edit of stored
  rows, and never `particles structure` — the backfill regenerates *derived*
  annotations and skips `STRUCTURED` particles for exactly this reason.
  "Derived" describes what a regeneration pass may produce; it is not a licence
  to mutate `content`, which stays immutable.

A **structured / deterministic** extractor (Wikidata, Numista, Nomisma, the
docstring extractor) has no prose stage and no LLM call: it overrides
`extract()` directly and returns a fixed candidate list for a fixed input blob.
Such extractors carry no benchmark / ECE gate — there is no
probabilistic output to calibrate (calibration is identity). When a
structured extractor mints code-like subject names (snake_case identifiers,
dotted paths), exempt its source type from the non-entity gate via
`subject_gate.exempt_source_types` so the names are not stripped — see the
`PYTHON_SOURCE` precedent.

## Things you must do

- **Define `EXTRACTOR_ID`, `EXTRACTOR_VERSION`, `TRUST_WEIGHT` class attributes.**
  The version is what the chunk-hash cache keys on for
  carry-forward; bumping it invalidates prior particles from this
  extractor.
- **Implement `accepts(source_type) -> bool`.** Used by the
  benchmark / conformance runners to filter applicable suites.
- **Return `ExtractionResult(candidates=..., quality_notes=...)`,
  not a list of `Particle`.** The pipeline wraps each
  `CandidateParticle` with extractor-agnostic provenance via
  `candidate_to_particle()`.
- **Set `uncertainty_nature=EPISTEMIC` for knowledge claims.**
  ALEATORY is reserved for inherent-randomness sources (probabilistic
  models). See [conformance](conformance.md) for the open question
  about diversity rules.

## Things you shouldn't do

- Don't call the database directly from the extractor. The pipeline
  passes you the snapshot + content; it stores the resulting
  particles. Side-effect-free `extract()` is the contract.
- Don't catch exceptions and return zero candidates silently — raise.
  The pipeline catches and records the failure properly.
- Don't read config at extractor `__init__` time; read it inside
  `extract()`. The pipeline may construct your extractor before
  config is loaded.

## Conformance + benchmark

Once your extractor works, write:

1. A [conformance fixture](conformance.md) so `particles extractor
   conform <id>` reports field-population rates against your source.
2. A [benchmark suite case](benchmark-suites.md) so you can pin
   precision / recall / calibration_error against a gold-standard
   expected-particle list.

The canonical contract: [`particles/extraction/AGENTS.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/extraction/AGENTS.md).
