# Authoring a benchmark suite

A benchmark suite measures an extractor's *correctness*: given a
fixed source, did it produce the expected particles at high enough
confidence? Run via `particles extractor benchmark <id>` (single)
or `particles extractor benchmark-compare --extractor-id A
--extractor-id B` (multi-extractor bake-off).

This is distinct from a [conformance fixture](conformance.md), which
checks *completeness* (did the extractor populate the schema's
required and recommended fields?). A benchmark checks whether it
emitted the right facts.

The harness is **report-only**: it never writes to the particle
store, never records a calibration, and never gates extractor
registration. Pass/fail is the caller's decision (`--fail-on
precision --fail-threshold 0.8` in CI, for example).

## File format

Suites live in `tests/benchmark/suites/<suite-id>.yaml`. The schema
is normative; field names match techspec §13.3 verbatim
(`particles/benchmark/schema.py`).

```yaml
suite_id: numismatic-seed-001
name: Numismatic seed benchmark
version: 0.1.0
domain: numismatics
source_types: [NUMISTA_API_COIN]
cases:
  - case_id: numista-coin-001
    fixture: numista-coin-001                  # reuses tests/conformance/fixtures/<id>/
    expected:
      - content: "The 5 Pfennigs coin from the GDR has aluminium composition."
        confidence_min: 0.95
        uncertainty_nature: EPISTEMIC
        required: true
```

The `fixture:` form reuses a conformance fixture. Use inline
`source_snapshot:` + `inline_content:` for ad-hoc cases that don't
warrant a conformance fixture.

### Every field

| Level | Field | Required | Notes |
|---|---|---|---|
| suite | `suite_id` | yes | Lowercase-kebab, unique across the project. `--suite <id>` selects on it. |
| suite | `name`, `version`, `domain` | yes | Free text; `version` is reported beside every result. |
| suite | `source_types` | yes | A list. Decides which extractor owns the suite; see [below](#which-suites-run-against-which-extractor). |
| suite | `cases` | yes | A list of cases. |
| suite | `metrics` | no | A list of `{name, definition}` mappings documenting extra metrics; see [Three normative metrics](#three-normative-metrics). |
| suite | `published_by`, `published_at` | no | Attribution; `published_at` is ISO-8601. |
| case | `case_id` | yes | Unique within the suite. |
| case | `fixture` | one of | A conformance fixture id. |
| case | `source_snapshot` | one of | An inline snapshot; only `content_hash` is mandatory, the rest default. |
| case | `inline_content` | no | The source bytes for an inline snapshot (a YAML block scalar is fine). |
| case | `expected` | yes | A list of expected particles. |
| expected | `content` | yes | The gold claim text. |
| expected | `confidence_min` | yes | A floor; see below. |
| expected | `uncertainty_nature` | yes | A valid `UncertaintyNature` value (e.g. `EPISTEMIC`). |
| expected | `required` | no | Defaults to `true`. |

A case must set exactly one of `fixture` or `source_snapshot`; both, or
neither, is a load error. An inline case looks like this:

```yaml
  - case_id: prose-survey-001
    source_snapshot:
      snapshot_id: "c0000000-0000-0000-0000-000000000001"
      captured_at: "2026-06-03T00:00:00+00:00"
      content_hash: "b9e7c3bd…"            # SHA-256 of inline_content
      content_published_at: "2026-06-03T00:00:00+00:00"
    inline_content: |
      <!DOCTYPE html>
      …
    expected:
      - …
```

The loader is strict where silence would lose gold data: an unknown key
inside `cases[]` or `expected[]` fails the suite. An unknown key at the
suite root only logs a warning, so a suite can carry a field a newer runner
understands. A suite that fails to load is logged and **skipped** (the run
carries on without it), so check the log if a suite you expect is missing
from the output. Suites are discovered in filename order.

Only the suite-level `source_types` routes an inline case; the case itself
carries no source type. A `fixture:` case uses its fixture's source type,
and if the extractor declines that type the case is skipped with a
quality note.

## Which suites run against which extractor

`particles extractor benchmark <id>` with no `--suite` runs only the
suites that extractor is the **production routing choice** for: a suite
matches if the registry would route at least one of its `source_types`
to that extractor (first registered plugin with no MUST_NOT clause for the
type whose `accepts()` returns true, the same selection the extract
pipeline makes; see [Writing an extractor](extractors.md)).

What that means for an author:

- **`source_types` decides the owner.** There is no "this suite is for
  extractor X" field. Declaring `WEB_PAGE` means the general extractor;
  declaring `REDDIT_POST` means the Reddit extractor. A suite whose types
  no extractor is the routing choice for matches nothing, and the verb
  says so.
- **Don't reach for `accepts()` reasoning.** The general extractor is the
  fallback and accepts every source type, so "which extractors could read
  this?" would hand it every domain suite and report the result as its
  score. Routing precedence is what keeps a domain suite measuring the
  domain extractor.
- **A suite in the tree must auto-match exactly one extractor.** The
  registry tests (`tests/test_extractor_registry.py`) enforce this for
  both `tests/benchmark/suites/` and `tests/benchmark/calibration/`, and
  pin which suites the general extractor owns; a new `WEB_PAGE`-style
  suite means updating that expectation deliberately.
- **`--suite <id>` bypasses routing**: use it to deliberately measure
  one extractor against another's suite.
- `extractor calibrate` uses the same routing rule;
  `benchmark-compare` deliberately does not (a named cross-extractor
  bake-off is its whole job).

## Adding a new suite

1. Pick a `suite_id` (lowercase-kebab, unique across the project) and a
   `case_id` for each case (unique within the suite).
2. Set `source_types` to types your extractor is the routing choice for.
3. If the source already has a conformance fixture, reference it by
   `fixture: <fixture-id>`. Otherwise add an inline `source_snapshot` +
   `inline_content` block. Adding a *new* conformance fixture just to
   serve a benchmark has a cost: it changes the fixture-corpus hash every
   stored conformance report carries (see
   [Adding or modifying a fixture invalidates prior reports](conformance.md#adding-or-modifying-a-fixture-invalidates-prior-reports)),
   so prefer the inline form unless the fixture belongs in the conformance
   corpus on its own merits.
4. Author each `expected` particle by inspecting the extractor's
   *current* output: copy the actual content string and confidence
   verbatim, then mark `required: true/false` by whether its absence
   would be a real regression.
5. Drop the file under `tests/benchmark/suites/` and run
   `particles extractor benchmark <your-extractor-id>` to confirm it is
   discovered and routed to your extractor.

## `required: true` vs `required: false`

| `required` | Affects |
|---|---|
| `true` | The case's recall denominator. A miss is a recall failure. |
| `false` | Contributes to precision if matched; absence is not penalised. |

Use `required: true` only for facts whose absence would mean the
extractor has lost its core value (e.g. "the structured-properties
summary line" for a coin extractor, not "the manufacturer's
historical anecdote").

Precision is computed over **every** emitted particle, so an emission the
gold set does not name counts against precision even when it is true. Aim
for gold coverage of everything the extractor legitimately emits from the
fixture; mark the less important ones `required: false` rather than
leaving them out.

## `confidence_min` is a floor, not a target

An emitted particle that matches semantically but reports confidence
below `confidence_min` becomes an `under_confidence` partial match.
It counts for neither precision nor recall, but is separately
reported so you can see *"the extractor got it right but stated it
too timidly."*

## How emitted claims are matched to expected ones

Matching is a greedy, one-to-one assignment in descending order of
similarity: each emitted particle matches at most one expected particle
and vice versa.

- The default **embedding** judge accepts a pair at cosine ≥ 0.80
  (`--threshold`).
- The **LLM** judge (`--judge llm`) accepts pairs ≥ 0.80 on similarity
  alone, rejects pairs below 0.65 without asking, and asks the model only
  about the band in between.

**Subject-aware matching.** A particle's subjects are a separate field, so
a well-behaved extractor may emit "Operating costs for 2025 were
$940,000." and link the subject rather than restate it. The judge
therefore scores each emitted claim twice (as its bare `content`, and
with the subject names it does not already mention prepended) and keeps
the higher similarity. For you as an author this means:

- **Both gold styles work.** Gold copied verbatim from subject-elided
  extractor output, and human-written gold that names the subject inline,
  both match. You do not need to restate or strip subjects.
- **Don't write the subject twice** into gold that the extractor will
  also prefix: the qualified rendering only adds subjects the content
  omits, and doubled naming measures worse than either clean form.
- **Don't lower the threshold to rescue near-misses.** It admits unrelated
  claims rather than reading related ones correctly. A restatement that
  scores below 0.65 is invisible even to the LLM judge; if a correct claim
  keeps missing, check whether its gold wording is a fair paraphrase.
- Numbers from runs made with `benchmark.subject_aware_matching: false`
  (the pre-subject-aware comparison, kept for replaying old results) are
  not comparable with current ones.

## Three normative metrics

`precision`, `recall`, `calibration_error` are mandated by techspec
§13.3 and the runner always reports them. `calibration_error` is not just a
score: it is the input an operator fits a temperature against, so a suite
authored here is what makes
[extractor calibration](../operator-guide/tuning.md#extractor-calibration)
possible at all, and
[benchmark + compare](../operator-guide/tuning.md#benchmark-compare) is how
they check a tuning change moved the needle.

The suite's optional `metrics:` list lets you *document* additional,
domain-specific metrics you expect a runner to report. The reference
runner parses that list but computes only the three normative metrics;
anything else in it is not computed.

## Repeat runs

Extraction is a sampling process, so one run is one sample: the same
fixture can score noticeably different recall on back-to-back runs.
`particles extractor benchmark <id> --runs N` repeats each suite N times
and reports each metric's mean, range and standard deviation. `--fail-on`
is evaluated against the **mean**, not the worst run. A repeat run costs
N× the LLM calls: the projected cost always prints first, `--estimate`
stops there, and a projection above `benchmark.confirm_call_threshold`
needs `--yes` or an interactive confirmation. Nothing about a suite file
changes for repeat runs; the same suite serves both.

Each run's report is saved as JSON under `benchmark.runs_dir` (unless
`--no-save`), and every emitted claim is recorded with its outcome
(`matched` / `under_confidence` / `spurious`), confidence and text, so you
can read *why* precision fell without re-running. Set
`benchmark.record_claim_text: false` if the fixture text must not land in
report files.

## What the frozen schema covers

The suite *input* schema (suite, case, expected particle and metric
declaration) is frozen by the techspec. You cannot add a field to an
expected particle (a modality label, a validity date, a polarity) for your
extractor's purposes; the loader will reject it. Properties the §13.3
shape cannot express are measured by separate harnesses with their own
suite formats (see [Other harnesses](#other-harnesses)).

## What good fixtures look like

- **Realistic content**: a real API response, a real article, not
  a synthetic minimal example, trimmed to the paths the case
  exercises. The fixtures in tree run from a few hundred bytes to ~3 KB.
- **High-signal expected list**: 5–20 particles per case, covering
  the structured / descriptive / catalogue-reference axes the
  extractor is meant to populate.
- **Calibrated `confidence_min`**: what the extractor *currently*
  emits, not a target. A future improvement that emits at higher
  confidence still passes; a regression that emits below the floor
  is caught as under-confidence.

## Calibration suites

`particles extractor calibrate <id>` fits a temperature that maps an
extractor's stated confidence onto observed correctness. It reads suites
from a separate directory, `tests/benchmark/calibration/`, and
`extractor benchmark` never looks there. The file format, loader, runner
and matching are exactly the ones above; only the directory and the
authoring rules differ, because the two purposes want different gold
coverage. A benchmark suite wants near-total coverage (a sparse gold set
reads as imprecision). A temperature fit needs **both** labels (correct
and incorrect emissions), and a gold set that names everything leaves
nothing to fit against.

### Authoring contract

- **`confidence_min: 0.0` on every expectation.** A floor is a claim
  about correctness a calibration suite should not make. (For calibration
  a timid-but-correct match counts as correct anyway.)
- **Deterministic extractors: make gold coverage deliberately partial**:
  roughly two-thirds to three-quarters of what the extractor emits, so
  both labels are present with margin. Say in the file header *which*
  emission is deliberately unnamed, so nobody "completes" the gold set
  and silently makes the suite unusable. This works because a parser
  emits the same set every run, so the omission (and the base rate it
  implies) is a fixed property of the file.
- **LLM extractors: do the opposite and name every claim the fixtures
  support.** For an LLM extractor an omission labels a claim *incorrect*
  that you know is correct, and the fit then learns your chosen coverage
  fraction instead of the extractor's calibration. Accept a refusal as the
  honest verdict rather than engineering one away.
  `prose-calibration-001` is the worked example; read its header.
- **Span several fixtures** where the corpus allows it; a single fixture
  yields few fittable pairs.
- **Inline the source bytes** (`source_snapshot:` + `inline_content:`)
  when a fixture exists only to calibrate, rather than adding a
  conformance fixture: that changes the conformance fixture-corpus hash,
  and a fixture engineered to elicit hedged confidence is not something
  the completeness check wants to measure anyway.
- **`source_types` decides the owner**, exactly as for benchmark suites,
  and each calibration suite must auto-match exactly one extractor.

### What a suite cannot fix

`calibrate` refuses to persist a fit when it cannot mean anything, and
names the reason:

| Refusal | Can the suite fix it? |
|---|---|
| degenerate labels (every fittable emission matched, or none did) | Yes: adjust gold coverage per the contract above. |
| predictor degeneracy (fewer than two distinct movable confidences, including all-saturated 0.0 / 1.0 output) | Not by the gold set. It *can* be moved by **fixture** design: prose in which the author's own certainty varies (a firm count beside a provisional one, two sources that disagree, a printed correction) draws a spread out of the same extractor that flatly stated prose never does. |
| fit landed on the optimizer bound | No: a property of the data, not the file. |
| non-improving fit | No: a property of the extractor. |

If you are editing a suite to make a refusal go away, check which
condition fired first.

Three more things bear on authoring for an LLM extractor:

- `calibrate` defaults to the **LLM judge** (`--judge embedding` to
  override), because a missed paraphrase there is a false *incorrect*
  label, not just a lost precision point. Restatements scoring below 0.65
  are still rejected without asking.
- A judge call that fails is treated as "not aligned", so an unreachable
  judge model manufactures incorrect labels. **Check the log before
  believing a fit.**
- One pass of an LLM extractor is a noisy estimate of the temperature.
  `calibrate --runs N` fits over the pooled pairs from N passes (cost
  N×, behind the same estimate/confirm gate); prefer it over persisting a
  single run.

## Other harnesses

The §13.3 harness above measures claim content. Separate harnesses, each
with its own suite format and directory, measure other properties of
emitted particles: `particles extractor benchmark-modality`
(`tests/benchmark/modality/`), `benchmark-polarity`
(`tests/benchmark/polarity/`), and `benchmark-validity`
(`tests/benchmark/validity/`). They share the routing rule above. Never
put their files in `tests/benchmark/suites/`: each discovery walker
would log-and-skip the other's files. The whole-pipeline memory benchmark
(`particles benchmark memory`) is not an extractor benchmark at all; see
[Agent memory benchmarks](../benchmarks.md).

The suite schema is normative and declared in
[`particles/benchmark/schema.py`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/benchmark/schema.py).
