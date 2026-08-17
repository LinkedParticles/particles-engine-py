# Concepts

Particles stores knowledge as atomic claims rather than prose: every
record is one statement carrying its own source, confidence, and
lifecycle status. Nothing is edited in place — corrections supersede,
retractions cascade, and disagreements between sources become visible
records you can review. Trust and staleness are applied when you
query, never written into the stored claim.

These are the words the Particles SDK uses for the pieces. Skim once;
the rest of the guide assumes you've seen them.

## Particle

The minimal unit of knowledge: one natural-language claim plus its
structured metadata envelope. Every particle carries:

- **content** — the claim itself, written as a single sentence.
- **subjects** — the canonical entities the claim is about.
- **confidence** — a stated likelihood (0.0–1.0), immutable once
  asserted. See [confidence](#confidence) below.
- **provenance** — which corpus entry the claim was extracted from.
- **uncertainty_nature** — `EPISTEMIC` (incomplete knowledge) vs
  `ALEATORY` (inherent randomness). PSUM terminology.
- **status** — `ACTIVE` / `SUPERSEDED` / `RETRACTED` / `INCONSISTENCY` /
  `PROVENANCE_STALE`. See [status](#status).
- **extractor_ref** — which extractor produced it (used for trust
  weighting).
- **structured_claim** *(optional)* — the same claim rendered as one
  subject-predicate-object triple, derived from `content` by tooling and
  stamped with what produced it. It is an *annotation*, not an assertion:
  it can be regenerated at any time and never affects the claim, its
  confidence, or its provenance. Many particles have none, permanently —
  some prose has no honest triple. See
  [operator guide → structured claims](../operator-guide/structured-claims.md).

For the full schema see [`spec/technical-specification.md`](../spec/technical-specification.md)
§6.

## Subject

A canonical real-world entity. Particles are statements *about*
subjects; subjects are the nodes of the knowledge graph. Examples:
"5 Pfennigs (1948-1950) GDR" (a coin), "Andrej Karpathy" (a person),
"Berlin" (a place).

Subjects are resolved at extraction time against Wikidata and other
ontologies (Numista, Nomisma). A subject may have multiple
`external_ids` (wikidata:Q1234, numista:8562, …) and a list of
aliases.

## Status

| Status | Meaning |
|---|---|
| `ACTIVE` | Currently valid; participates in queries and exports |
| `SUPERSEDED` | Replaced by a newer particle |
| `RETRACTED` | Explicitly invalidated; the source itself was wrong |
| `INCONSISTENCY` | Contradicts another particle; needs operator review |
| `PROVENANCE_STALE` | The corpus entry it was extracted from has new content; needs re-extraction |

Transitions go through `status.validate_transition()` — operators
don't poke `status` directly.

## Confidence

The SDK separates two quantities:

- **`confidence.value`** — the extractor's confidence as calibrated at
  creation time. **Immutable.** When the extractor carries an active
  temperature-scaling calibration, the scaled value is what
  gets stored, stamped `calibration_source: CALIBRATED_BENCHMARK` with
  a `calibration_ref` audit trail; otherwise the raw self-reported
  value is stored as `EXTRACTOR_DIRECT`.
- **`effective_confidence`** — `confidence.value` × extractor trust
  weight × source trust rank × content-age decay. The number queries
  actually rank by. Computed at query time, never stored.

For operator tuning of trust weights and thresholds, see the
[operator guide](../operator-guide/tuning.md).

## Provenance

Every particle has `provenance: list[ProvenanceRef]` linking it to
the corpus entry / snapshot it was extracted from. The corpus is
the immutable, append-only archive of source material. Particles
are *rebuildable* from corpus + extractor; the corpus is the
durable record of "what we saw, when."

For the two-layer architecture details and the
`particles/corpus/AGENTS.md` contributor guide.

## Where to go next

- [Querying](querying.md) — how ranking and tag filters work.
- [Exporting](exporting.md) — Obsidian / Anki / wiki shapes.
- [Operator guide → lint and review](../operator-guide/lint-and-review.md) —
  how to handle INCONSISTENCY and PROVENANCE_STALE.
