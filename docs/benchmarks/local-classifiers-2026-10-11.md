# Local classifiers for duplicate claims and subject mentions (2026-10-11)

**A dated snapshot, as of 2026-10-11.** This page continues the
[2026-10-07 measurement](local-classifiers-2026-10-07.md), which asked whether
a small local model can take over a yes/no question the SDK now sends to a
generative model (phase 2). It measures the two candidates that page
left open: the co-evidential judge, which decides whether two stored claims
assert the same fact, and span-based mention spotting with GLiNER. Neither
cleared its bar, so no default changed and no classifier seam is proposed.

The measurement also found a defect in the production judge: on
`claude-haiku-5-5` its output budget is too small for any batch of four or more
pairs, so the batch comes back with no verdicts. That is described under
[a truncated judge](#a-truncated-judge) and is being fixed separately.

## What was measured

| Purpose | Local candidates | Compared against |
|---|---|---|
| Co-evidential judge | two NLI cross-encoders (mutual entailment), a duplicate-question cross-encoder, a semantic-similarity cross-encoder, and the stored embedding cosine | hand labels on 176 real candidate pairs, and the production judge on `claude-sonnet-4-6` and `claude-haiku-5-5` |
| Subject mention spotting | GLiNER v2.1, small, medium and large | the gold names of a committed prose set, and the subjects the extraction model gave 200 stored claims |

Every run used the committed script `scripts/bench_local_classifiers.py` on
an Apple M5 laptop (10 cores, 24 GB, no fan). Bulk scoring ran on the GPU;
latency is reported on the CPU, one item at a time, from runs made with the
load average below 5. The cross-validated thresholds come from a later rerun
of the local models while other jobs loaded the machine; its scores were
identical and its timings were discarded. The only hosted calls were three runs of the production judge over the
176 pairs, about US$0.09 in all.

## The co-evidential judge

### What it decides

`particles links suggest` proposes two ACTIVE claims as candidates when they
share a Subject and their embeddings have a cosine similarity of at least
`links_suggest.candidate_threshold` (0.92). In its judge and apply modes it
sends those pairs, a Subject's cluster at a time, to the model on the
`semantic_lint` purpose, which answers PARAPHRASE, DISTINCT or UNSURE for each.
The apply mode writes a co-evidential link for every PARAPHRASE, so a false
PARAPHRASE merges two different facts into one corroborated belief. Precision
on the PARAPHRASE class is the property that matters most; recall decides how
much of the store's duplication the judge finds.

This is a symmetric question about two short sentences, with no combination
step, which is why the first page named it the best fit for a cross-encoder.

### The labelled set

No judge verdicts are recorded anywhere a store keeps them, so a labelled set
was built and is now versioned:
[`co-evidential-pairs-2026-10-11.json`](co-evidential-pairs-2026-10-11.json).

- **Source.** The candidate pairs the production shortlist proposes on the
  owner's working store, read from a backup copy: 5,291 pairs. Pairs with
  identical text were left out, since an exact duplicate is merged by its
  content hash and never reaches a judge, as were pairs naming an internal
  record id, a home path, a session identifier or an e-mail address.
- **Sample.** 22 pairs from each 0.01 band of cosine from 0.92 to 1.00
  (seeded), 176 in all. Most of the store's text is software and API notes, and
  the set reflects that.
- **Labels.** Written by the session that ran this measurement, before any
  classifier or hosted judge saw the set, against the judge's own instruction:
  120 *paraphrase*, 44 *distinct*, and 12 *subsumes*, where one claim asserts
  the other's fact plus a further one ("costs $15 per million output tokens"
  against "costs $3 per million input and $15 per million output tokens"). The
  instruction does not decide that case, so those 12 are reported apart and left
  out of the scores.

The scores are reported on the natural mix (164 pairs, 73% paraphrase) and on a
balanced subset (all 44 distinct pairs and 44 seeded paraphrases).

### How each local model answers

- **NLI, mutual entailment.** The pair is a paraphrase when the model's most
  likely label is `entailment` in both directions. The score is the weaker
  direction's entailment probability.
- **Similarity cross-encoders** return one score in [0, 1]. The default reading
  is a score of at least 0.5.
- **The stored cosine** is the shortlist's own signal. Every pair already clears
  0.92, so it shows how far cosine alone separates the classes above the floor.

A threshold fitted on the same pairs it is scored on flatters a model, so each
score-based classifier is also reported with a cross-validated threshold: fitted
on one seeded half, scored on the other, both ways, pooled over 50 repeats.

### Results

On the natural mix (164 pairs: 120 paraphrase, 44 distinct). A *false link* is
a distinct pair called a paraphrase.

| Classifier | Size | Reading | Precision | Recall | κ | False links | AUROC | CPU per pair |
|---|---|---|---|---|---|---|---|---|
| `claude-sonnet-4-6`, production judge | hosted | as answered | 0.991 | 0.958 | 0.91 | 1 | | 191 ms (batched) |
| `claude-haiku-5-5`, production judge, output budget tripled | hosted | as answered | 0.991 | 0.942 | 0.88 | 1 | | 161 ms (batched) |
| `claude-haiku-5-5`, production judge as shipped | hosted | as answered | 1.000 | 0.117 | 0.07 | 0 | | 154 ms (batched) |
| `cross-encoder/stsb-roberta-large` | 355M | ≥ 0.5 | 0.779 | 1.000 | 0.30 | 34 | 0.960 | 57 ms |
| | | cross-validated | 0.942 | 0.987 | 0.86 | 7.3 | | |
| `cross-encoder/quora-distilroberta-base` | 82M | ≥ 0.5 | 0.964 | 0.900 | 0.77 | 4 | 0.953 | 9 ms |
| | | cross-validated | 0.923 | 0.917 | 0.70 | 9.2 | | |
| `cross-encoder/nli-deberta-v3-base` | 184M | mutual entailment | 1.000 | 0.708 | 0.57 | 0 | 0.945 | 67 ms |
| | | cross-validated | 0.902 | 0.910 | 0.65 | 11.9 | | |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 435M | mutual entailment | 1.000 | 0.725 | 0.59 | 0 | 0.931 | 882 ms |
| | | cross-validated | 0.948 | 0.877 | 0.70 | 5.8 | | |
| Stored embedding cosine | | ≥ 0.92 (the shortlist) | 0.732 | 1.000 | 0.00 | 44 | 0.700 | |
| | | cross-validated | 0.743 | 0.894 | 0.06 | 37.2 | | |

On the balanced subset (88 pairs) the ordering is the same: the hosted judge
reaches κ 0.93 (Sonnet) and 0.91 (Haiku with its budget tripled); the best local
readings are the NLI large model's mutual entailment (precision 1.000, recall
0.795, κ 0.80, no false links) and the Quora cross-encoder at 0.5 (precision
0.909, recall 0.909, κ 0.82, 4 false links).

On the 12 *subsumes* pairs, the share each called a paraphrase: Sonnet 4 of 12,
Haiku 3 of 12, both NLI models 1 of 12, Quora 8 of 12, STS-B 12 of 12, cosine
12 of 12.

Agreement with the hosted judge rather than the labels tells the same story. On
the 164 pairs, Sonnet and Haiku (budget tripled) agree with each other at
κ 0.88. Against Sonnet, the Quora cross-encoder agrees at κ 0.77, the NLI models
at 0.59 to 0.61, and the STS-B model at its default reading at 0.27.

### What the local models get wrong

The two families fail in opposite directions.

- **NLI is strict.** Mutual entailment made no false link on any distinct pair,
  but it rejects a restatement that adds a date, a scope or a parenthetical
  ("scored 0.73" against "the inaugural run scored 0.73"; "produced 0 errors
  after the fix" against "produced zero errors"). An NLI model reads the extra
  detail as information the other sentence does not entail, which is correct
  for NLI and the wrong question for this judge. That costs it 27% to 29% of
  the real paraphrases.
- **Similarity models are lenient.** They score two sentences that differ in
  one identifier as near-identical: two different decision records colliding on
  one branch (STS-B 0.97), a hook name against a test name (Quora 0.83), "an
  `[Unreleased]` section was added" against "an entry was added under
  `[Unreleased]`" (both above 0.9). These are the false links. No threshold
  separates them, because real paraphrases also score in the same range, which
  is why the cross-validated rows trade a few false links for a few misses
  rather than removing them.
- **The stored cosine adds almost nothing above the floor.** Its cross-validated
  κ is 0.06 on the natural mix: within the shortlist, the embedding cannot tell
  a restatement from a different fact with the same shape, which is what the
  judge exists for.

The hosted judge's errors, and most disagreements between it and the labels,
fall on pairs a single qualifier separates: "only", "(beta)", "the live store",
"the new function". Those are the pairs on which the label itself is closest to
a judgement call. The one false link both hosted models made was the hook name
against the test name.

### A truncated judge

The production judge allows `max(200, 40 × pairs)` output tokens for a batch,
and its reply is a JSON object with one 17-character pair key and a verdict per
pair. `claude-haiku-5-5` spends about 57 tokens on each entry, because its
tokenizer splits the hexadecimal keys finely. Any batch of four or more pairs
therefore stops at the budget, the reply does not parse, and every pair in the
batch is recorded as UNSURE. Over the 176 pairs in batches of 25 that left 151
pairs UNSURE, the third row above. The adapter logs a truncation warning, but
nothing else reports it. With the budget tripled, the same model is as good as
Sonnet. `claude-haiku-5-5` is the `semantic_lint` model the owner's
configuration names, so the judge mode, the apply mode, the curation duplicate
cards and the re-extraction estimate all lose their verdicts on any cluster
larger than three pairs there. The compiled default (`claude-sonnet-4-6`) is
unaffected.

### Against the bar

The bar for a stand-in judge: no more false links than the hosted judge on this
set (one), and recall within five points of it (at least 0.89). Its threshold
must be fixed in advance or cross-validated.

| Classifier | False links | Recall | Clears |
|---|---|---|---|
| NLI large, mutual entailment | 0 | 0.725 | no: recall |
| NLI base, mutual entailment | 0 | 0.708 | no: recall |
| Quora, 0.5 | 4 | 0.900 | no: false links |
| STS-B, cross-validated | 7.3 | 0.987 | no: false links |

None clears. The pattern does suggest a split: mutual entailment as a
high-precision local first pass, with only the pairs it does not accept sent to
the model, would have settled 85 to 87 of the 120 paraphrases here with no false
link. Zero false links in 44 distinct pairs still leaves an upper confidence
bound near 7%, the judge runs at about US$0.03 per thousand pairs on
`claude-haiku-5-5`, and the subsumes pairs would need a rule the instruction
does not have. A first pass is not pursued.

## Subject mention spotting with GLiNER

### What the comparison can and cannot show

The subject names attached to a claim come from the extraction call itself: the
model lists them with each claim, in the same reply. The `subject_resolution`
purpose is a different step, which chooses among Wikidata candidates for a name
already found, and a span tagger does not do that. A GLiNER model could only
replace the listing step, and only by running beside an extraction call that
would still be made, so there is no LLM call for it to save. The measurement asks
the narrower question anyway: does it find the mentions the model names?

Two sets were used:

- **Gold names in ordinary prose.** The committed
  `tests/benchmark/resolution/ordinary-prose-001.yaml`: eight short passages
  in eight genres, and the 49 proper-named entities in them.
- **Subjects the extraction model gave real claims.** A seeded sample of 200
  ACTIVE claims extracted by `claude-sonnet-5` in the owner's store, read from
  the same backup copy, and the 277 subjects linked to them. Only aggregate
  figures are reported. 75 of those subjects (27%) do not appear in the claim's
  text under their name or any alias ("the refetch ladder", "Agents API",
  `particles-core-py`), so no span tagger can produce them. Recall is measured
  over the other 202.

Each GLiNER model was given twelve entity labels (person, organization, company,
location, product, software, programming language, AI model, file, command,
project, event). A name counts as found *exactly* when a predicted span equals it
after case and punctuation are normalised, and *partially* when a span contains
it or covers at least half of it.

### Results

| Model | Size | Threshold | Prose: exact | Prose: partial | Prose: spans that are no gold name | Store: exact | Store: partial | Store: spans that are no named subject | CPU per claim |
|---|---|---|---|---|---|---|---|---|---|
| `urchade/gliner_small-v2.1` | 153M | 0.5 | 0.898 | 0.918 | 5 of 50 | 0.248 | 0.327 | 150 of 216 | 20 ms |
| | | 0.3 | 0.918 | 0.939 | 13 of 59 | 0.396 | 0.619 | 352 of 475 | |
| `urchade/gliner_medium-v2.1` | 195M | 0.5 | 0.878 | 0.898 | 7 of 51 | 0.361 | 0.490 | 281 of 380 | 37 ms |
| | | 0.3 | 0.939 | 0.959 | 11 of 58 | 0.510 | 0.743 | 474 of 624 | |
| `urchade/gliner_large-v2.1` | 445M | 0.5 | 0.918 | 0.980 | 10 of 58 | 0.426 | 0.698 | 393 of 534 | 109 ms |
| | | 0.3 | 0.918 | 0.980 | 15 of 63 | 0.500 | 0.812 | 587 of 750 | |

On ordinary prose GLiNER does what it was trained for. Every model misses
"Mark Ellison" (it takes "Councillor Mark Ellison"), "London Stock Exchange",
"James Webb Space Telescope" and "Ariane 5" (the large model returns "the James
Webb Space Telescope" and "Ariane 5 rocket", which the partial match counts), and its extra
spans are dates and common nouns ("December 2021", "cast-iron skillet").

On the store's own text it finds at most half of the model's subjects exactly,
and it fails on exactly what this store is about. File paths and module names
split at their punctuation (`particles/extraction/general.py` comes back as
"py"), commits, pull requests, configuration keys and environment variables go
unlabelled, and the spans it does return are mostly not subjects ("merge",
"main", "md"). Lowering the threshold raises partial recall to 0.81 only by
tripling the spans that are not subjects.

### Against the bar

Recall of the model's mentions is what matters, since a mention never spotted
is a subject never linked. The best exact recall on the store sample is 0.51,
and a quarter of the model's subjects are not spans at all. GLiNER does not
clear any reasonable bar here.

## Outcome

- **No default changes.** The co-evidential judge stays on the `semantic_lint`
  model, and subject listing stays in the extraction call.
- **No classifier seam is proposed.** Four of the five purposes the record
  names have now been measured, and none has a local model that clears its
  bar. Only `use_judge` is still unmeasured.
- **The production judge's output budget is a defect** on `claude-haiku-5-5`,
  and is fixed separately: shorter pair keys or a budget sized per entry.
- **The co-evidential pair set is now versioned**, so a later model can be
  measured on the same pairs.

## Reproducing

```bash
uv run python scripts/bench_local_classifiers.py dedup
```

```bash
uv run python scripts/bench_local_classifiers.py dedup --nli-models --sts-models --anthropic-models claude-sonnet-4-6 claude-haiku-5-5
```

```bash
uv run --with gliner --with sentencepiece --with protobuf python scripts/bench_local_classifiers.py mentions --db <a copy of a store>
```

The first command runs the local models only. The second runs the production
judge (paid, a few cents) through the real `judge_claim_pairs` path, and
`--budget-scale 3` multiplies its output budget to measure the model apart from
the truncation. GLiNER is a script-local install and never a project
dependency. The `mentions` command reads a store read-only and prints only
aggregate figures.
