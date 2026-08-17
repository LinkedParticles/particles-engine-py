# Extractor conformance baseline

This directory holds the v1.0.0 frozen baseline of each registered
extractor's conformance report, captured by

```
particles extractor conform <extractor-id> --format json > docs/conformance/baseline/<extractor-id>.json
```

Phase 1 (current): report-only — the baseline is informational. The
validator does *not* block PRs even on REQUIRED failures. The intent
is that a future PR that regresses an extractor's conformance numbers
must update its baseline file in the same commit, with a one-line
rationale.

Phase 2 (deferred): once the fixture corpus covers every built-in
extractor adequately, CI will gate on diffs to these files (REQUIRED
regressions block; RECOMMENDED regressions warn). for
the migration plan and the explicit gating criteria.

Baseline files are captured as fixture coverage warrants, not all at
once. Captured so far:

Every report carries the hash of the **whole** fixture corpus, so adding a
fixture invalidates the comparability of every existing baseline. A PR that
grows the corpus therefore re-captures the untouched baselines in the same
commit; only their `fixture_corpus_hash` and `generated_at` move.
(Narrowing that scope to an extractor's routed subset — so growth in one
source type stops invalidating unrelated baselines — is future work.)

**Which fixtures a capture covers.** A run scores the fixtures the
registry *routes* to the extractor, not every fixture its `accepts()` returns
True for. `quality_notes[0]` in each file records which set produced it — the
six deterministic baselines predated that note and acquired it in the re-capture, so every file can now answer the question. Never check in a
baseline captured with `--all-accepted`: that run deliberately includes
fixtures the extractor is not routed, so its rates are not the extractor's
conformance score.

**Deterministic vs LLM-derived baselines — read `extraction_provider_model`
.** Each file names the `"<provider>:<model>"` pairing that produced
its particles, and **`null` is the discriminator**: a deterministic extractor
makes no completion call, so a null means the file is a parser's output and a
re-capture reproduces it byte for byte apart from `generated_at`. Null is not
"no network" — `wikidata-extractor` fetches labels and is still a parse. Do not
read this null the way the schema reads a *particle's* null ("unrecorded,
possibly predating the field"): a report covers only particles minted during
its own run, so that branch is unreachable here.

A non-null pairing marks a different kind of object. The same extractor version
over the same fixture yields different particles run to run, and different
particles again under a different pairing. For those files:

* A numeric diff is **not** by itself a regression. Only a **tier-level verdict
  change** is — a REQUIRED field crossing PASS ↔ FAIL. A DIVERSITY rule
  flipping is *not* one: since the `uncertainty_nature` rule is
  ADVISORY, and on a sampled extractor it is expected to flip.
* Re-capture costs a live API call, so a corpus-growth PR restamping it is not
  free the way restamping a parser baseline is.
* A pairing change is *recorded*, not yet enforced: comparability is still
  `fixture_corpus_hash` equality alone, so a re-capture on a different model is
  nominally comparable with one it should probably not be compared to
  .
* **Never hand-edit a captured file** — not the pairing, not a count. A file
  that is partly a capture and partly an assertion cannot be read as either.

`general-extractor.json` is the first such file. It became capturable at all
: 1 fixture and ~1 call, where the pre-0232 rule meant 13
non-deterministic extractions. `journal-extractor.json` and
`reddit-extractor.json` joined it — the second and third LLM-backed
baselines, both routed one fixture and both costing one call.

**What the three LLM baselines say together.** The diversity rule decided the
`uncertainty_nature` diversity rule on one fixture, on the explicit ground that
a second baseline could not change the outcome. Two more now exist and the
reading holds — with the sampling half of it no longer a prediction:

* Both new extractors satisfy the rule on capture (`EPISTEMIC 38 / ALEATORY 2`
  and `16 / 2`), so all three LLM extractors do and all seven parsers do not.
  Prose carries the stochastic-quantity signal; catalogue records and generic
  RDF do not. That is the axis, and it is now measured three times.
* **The margin is two particles every time** — 22/2, 38/2, 16/2 across three
  extractors at three particle counts. Not a threshold effect: a handful of
  genuinely-aleatory claims per document is simply what prose yields.
* An immediate second `journal-extractor` run over the same fixture at the same
  pairing returned **39 particles, 1 distinct value** — the rule satisfied and
  then not satisfied, nothing changed but the sample. The analysis argued LLM
  extractors fail the rule intermittently; this is that, observed. Do not read
  a diversity flip between captures as a change in the extractor.

**The `uncertainty_nature` diversity rule is ADVISORY.** Every
structured extractor below reports 1 distinct value and every one of them
**passes** — the finding rides the report in `advisories`, not `failures`, and
does not affect `passed`, the `--fail-on` exit code, or the trust cap.
That is deliberate: the rule's outcome tracks whether the *source vocabulary*
carries a distinguishable stochastic-quantity signal, not whether the extractor
is complete. Read the `value_counts` histogram rather than the
pass/fail — the margin on the one extractor that satisfies the rule is 2
particles of 24, against a run-to-run spread of 21–26, which is exactly why it
is not a gate.

| Baseline | Result | Note |
|---|---|---|
| `general-extractor.json` | **PASS** | Re-captured at the prose-suite 0.2.0 corpus growth (extractor 0.14.0; `extraction_provider_model` = `anthropic:claude-sonnet-4-6` — `llm.extraction` unset, so the `llm.default` pairing applied). The routed set is now **4 fixtures** — `web-article-001` plus the three prose-article-seed-001 v0.2.0 fixtures (`web-article-002` numeric-dense, `web-essay-001` argumentative essay, `web-interview-001` interview) — so this file's scope changed, not just its hash: earlier single-fixture captures are not comparable with it even before the corpus-hash rule says so. 103 particles, every REQUIRED and RECOMMENDED field at 100 %, no failures, no warnings, no advisories. Read it under the LLM-derived rules above: counts and rates move on re-capture (the single-fixture era's five captures spread 21–26 particles), and only a tier-level verdict change is a regression — which this growth demonstrated live. The *first* capture at this corpus drew 102 particles with `subject_ids` populated on 101, a REQUIRED FAIL at 99 %, and the immediate same-pairing re-capture passed clean: one sampled subjectless particle in ~100 is the flake class the tier-level rule exists for, and at 4-fixture particle counts an occasional FAILing capture is expected — re-run before reading it as a regression. **Still the only baseline that satisfies the `uncertainty_nature` diversity rule**, now with real margin: `EPISTEMIC 95, ALEATORY 8` against the single-fixture era's 22/2 — the three added fixtures each carry forward-looking claims, which is where prose produces aleatory classifications at all. The bare-key note history (fixed) lives in this row's git history. Report-only throughout — a baseline surfacing a real defect, and then witnessing its fix, is the whole argument for keeping these files. |
| `journal-extractor.json` | **FAIL** (`subject_ids`) — **stale, do not read as current** | ⚠️ **Superseded on both axes and not yet re-captured.** The extractor is now 0.4.0 (narrowed `subjects` instruction), and `subject_ids` is now measured over the techspec §9 population rather than every particle — so this file's 40 % was produced by a prompt and a denominator that no longer exist. It is kept because it is the observation that change was written from, not because it describes today's behaviour. The re-capture needs one live API call and the key's credit balance was exhausted at activation. Original capture notes follow. Extractor 0.3.0, `extraction_provider_model` = `anthropic:claude-sonnet-4-6`. One fixture (`journal-entry-001`), 40 particles. Every REQUIRED and RECOMMENDED field at 100 % **except `subject_ids` at 40 %** — a REQUIRED failure, and unlike the `general-extractor` near-miss this one is not a sampled flake: the immediate re-run measured 39 particles at 49 %. A journal entry names few resolvable entities, so most of its claims reach the store subjectless and, per §6.7, unreachable by subject-filtered query. Whether that is an extractor defect or the honest shape of journal prose is **not decided here** — the capture is the observation that makes the question askable. The capture is checked in failing on purpose: a baseline that only ever records passes cannot witness anything. Satisfies the diversity rule on this capture (`EPISTEMIC 38, ALEATORY 2`); the re-run did not. |
| `reddit-extractor.json` | PASS | Extractor 0.3.1, `extraction_provider_model` = `anthropic:claude-sonnet-4-6`. One fixture (`reddit-thread-001`), 18 particles, every REQUIRED and RECOMMENDED field at 100 %, no failures, no warnings, **no advisories** — the only baseline in the corpus clean on every axis, because it is an LLM extractor that both populates `subject_ids` fully and satisfies the diversity rule (`EPISTEMIC 16, ALEATORY 2`). `structured_claim` at 94 % and `properties` at 17 % are the dual-emission metadata particles, not a gap. Read it under the LLM-derived rules above: counts and rates move on re-capture. |
| `wikidata-extractor.json` | PASS (1 advisory) | Structure-canonical emission (extractor 0.2.0). Every REQUIRED field at 100 %; the `uncertainty_nature` **diversity** rule reports its standing structured-extractor advisory without failing the run — the shape described in the `rdf-extractor` row below. Five particles from `wikidata-entity-001`, one per value type, with the `commonsMedia` statement correctly skipped. **The only conform run in the corpus that touches the network**: the extractor fetches English labels to verbalize `content`, because a Wikibase entity blob carries labels for itself and for nothing it references. The structured claim — the half that is canonical here — is built from the deposited identifiers alone. Captured with network available; an offline run drops the item-valued statements and falls back to P-ids, which is a smaller, honest report rather than a regression. |
| `rdf-extractor.json` | PASS (1 advisory) | Every REQUIRED field at 100 %; the `uncertainty_nature` **diversity** advisory is the standing structured-extractor reading and the reason the rule was made advisory — a parser reports what a source states, so the residual uncertainty is always about the world (EPISTEMIC) and never about sampling. Generic RDF carries no signal that would distinguish a stochastic quantity, and emitting ALEATORY on some arbitrary triple class would be fabricating an epistemic classification — the advisory is the honest reading, not a defect to fix. Identical to `numista-coin-extractor`'s result on the reference fixture, so this is the standing shape for every structured extractor. |
| `numista-coin-extractor.json` | PASS (1 advisory) | Extractor 0.3.0. Seven particles from `numista-coin-001`; the standing diversity advisory above. `structured_claim` 100 %, `canonical_form` 2 distinct values — **six** STRUCTURED (per-field templated particles whose `content` renders exactly their triple) and **one** PROSE (the entity infobox, whose `content` states fifteen properties while a triple states one). The clearest single view of the §2.1 per-candidate rule. |
| `numista-issuer-extractor.json` | PASS (1 advisory) | Extractor 0.3.0. Five particles from `numista-issuer-001` — two infoboxes (PROSE) and three catalogue references (STRUCTURED); `properties` 40 % is that 2-of-5 split. Same diversity advisory. |
| `numista-listing-extractor.json` | PASS (1 advisory) | Extractor 0.2.0. Two particles from `numista-listing-001`, **`canonical_form` 1 distinct value — all PROSE**. This extractor emits nothing but entity infoboxes, so `structured_claim` at 100 % alongside zero STRUCTURED particles is the correct reading, not a defect: coverage and canonicality are independent axes, and `rdf:type` annotation is the whole reason the extractor gains anything from that ADR at all. It is also the counterexample that killed the *observed*-canonicality axis for conditional diversity application: this parser and the LLM `general-extractor` report the identical `canonical_form` distribution. Same diversity advisory. |
| `nomisma-extractor.json` | PASS (1 advisory) | Extractor 0.3.0. One particle from `nomisma-concept-001`, PROSE (its `content` renders the class *and* the definition — two facts, one triple). The family's best-keyed member: the triple's subject term is the real entity IRI, so `bind_subject_id`'s URI rung binds `subject_id`. Same diversity advisory. |
