# Authoring a benchmark suite

A benchmark suite measures an extractor's *correctness* — given a
fixed source, did it produce the expected particles at high enough
confidence? Run via `particles extractor benchmark <id>` (single)
or `particles extractor benchmark-compare --extractor-id A
--extractor-id B` (multi).

## File format

Suites live in `tests/benchmark/suites/<suite-id>.yaml`. The schema
is normative — field names match techspec §13.3 verbatim
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

## `required: true` vs `required: false`

| `required` | Affects |
|---|---|
| `true` | The case's recall denominator. A miss is a recall failure. |
| `false` | Contributes to precision if matched; absence is not penalised. |

Use `required: true` only for facts whose absence would mean the
extractor has lost its core value (e.g. "the structured-properties
summary line" for a coin extractor — not "the manufacturer's
historical anecdote").

## `confidence_min` is a floor, not a target

An emitted particle that matches semantically but reports confidence
below `confidence_min` becomes an `under_confidence` partial match.
It counts for neither precision nor recall, but is separately
reported so you can see *"the extractor got it right but stated it
too timidly."*

## Three normative metrics

`precision`, `recall`, `calibration_error` are mandated by techspec
§13.3 and the runner always reports them. Domain-specific metrics
are opt-in extensions; see [`particles/benchmark/AGENTS.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/benchmark/AGENTS.md)
for the extension hook.

## What good fixtures look like

- **Realistic content** — a real API response, a real article, not
  a synthetic minimal example. ~10–50 KB is typical.
- **High-signal expected list** — 5–20 particles per case, covering
  the structured / descriptive / catalogue-reference axes the
  extractor is meant to populate.
- **Calibrated `confidence_min`** — what the extractor *currently*
  emits, not a target. A future improvement that emits at higher
  confidence still passes; a regression that emits below the floor
  is caught as under-confidence.

The canonical contract: [`particles/benchmark/AGENTS.md`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/benchmark/AGENTS.md).
