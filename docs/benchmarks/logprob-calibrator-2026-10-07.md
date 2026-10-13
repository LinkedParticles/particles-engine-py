# Log-probability calibrator: does a local extractor know which claims are right? (2026-10-07)

**A dated snapshot, measured 2026-10-07 to 2026-10-08.** The
[2026-10 provider survey](provider-survey-2026-10.md) found that cheap models
match Sonnet on recall and lose on calibration, and that a temperature fitted
per model made held-out calibration worse. A stated confidence is never
rewritten once a particle exists, so a badly calibrated local
extraction is permanent. The proposed remedy separates the two
jobs: the extractor emits claims, and a small calibrator sets each claim's
confidence at creation from features that include the extractor's own token
log-probabilities. Hosted Claude does not expose log-probabilities. Ollama,
llama.cpp and vLLM do.

This page tests that bet before anything is built. It changes no default and
builds no cascade, and it made no paid API call.

## Verdict

**The signal is real on one model and absent on the next, so the bet does not
hold as stated.** Two local models were measured under the same harness.

- **On `qwen2.5:14b-instruct`, log-probabilities over the claim's text rank
  claims; the stated confidence does not.** Held out by case, a logistic
  calibrator over the span log-probabilities ranks correct above incorrect
  claims at AUROC 0.655, against 0.505 for the stated confidence, which is
  chance. Pooled held-out ECE falls from 0.243 to 0.052, and the fit transfers
  across genre in both directions, which the temperature in the 2026-10 survey
  failed to do.
- **On `qwen3:14b`, the same features carry no signal at all.** Every span
  log-probability feature ranks claims at AUROC 0.45 to 0.53, and every calibrator built on
  them ranks *worse* than chance held out. The signal is a property of the
  model, not of the method. See [the second model](#second-model-qwen314b).
- **The stronger model is the closer candidate on quality, and still misses
  the gate.** `qwen3:14b` is right on 63 to 72 % of its article-suite claims
  (full match), against 39 to 54 % for qwen2.5 and 79 % for
  `claude-sonnet-5`. Its best held-out per-run ECE is 0.129 against Sonnet's
  0.045 under the calibration label. The calibrators that lower its ECE do it
  by pulling every claim toward the base rate, not by telling claims apart.
- **Agreement across samples ranked claims on the first set of samples, and
  did not replicate.** On the first three `qwen3:14b` runs, the share of other
  samples that produced the same claim ranked claims at AUROC 0.657 (95 %
  interval 0.567 to 0.776). On three fresh runs it ranked them at 0.52 to 0.53,
  an interval spanning chance, and claims no other sample confirmed were right
  *more* often than claims every sample confirmed. Only the ordering within a
  single source held, weakly (0.63 to 0.69 across both sets). See
  [sample agreement](#sample-agreement).
- **Escalating to the hosted extractor came close on flat prose, and only
  there.** With one paid `claude-sonnet-5` pass (US$2.07), sending a chunk to
  the hosted model when more than a third of its local claims were disagreed
  escalated 56 % of the article suite's chunks and matched the hosted route's
  precision and calibration within its own run-to-run spread, at 3 points less
  recall. The same rule escalated a quarter of the hedged suite's chunks and
  lost 24 points of recall there. The cutoff was picked from three on the same
  data. See [escalation](#escalation-to-the-hosted-extractor).
- **No local route reaches the hosted route on calibration across both
  genres.** The proposal's gate, held-out ECE no worse than the hosted route,
  is not met by any calibrator or escalation policy on the pooled suites.

Read together: a log-probability calibrator cannot be assumed for a local
extractor, and sample agreement is too unstable to carry a cascade on its own.
A cascade that escalates chunks, not claims, is the one arrangement that came
near the hosted route, on one genre, with an untested threshold. Nothing here
supports moving extraction off the hosted route, and nothing here yet justifies
building the calibrator.

## What was measured

This section and the four after it are the first model, `qwen2.5:14b-instruct`.
The second model has [its own section](#second-model-qwen314b), run under the
same harness.

| Parameter | Value |
|---|---|
| Model | `qwen2.5:14b-instruct` (Q4_K_M), Ollama 0.31.1, Apple M5, 24 GB |
| Route | `local` OpenAI-compatible entry, `logprobs: true`, `top_logprobs: 5` |
| Context / budget | 12 288-token context, `extraction.max_tokens` 6 144 |
| Extractor | `general-extractor` 0.16.0 |
| Suites | `prose-article-seed-001` v0.3.0 (6 cases, flat prose) and `prose-calibration-001` v0.1.0 (4 cases, hedged prose) |
| Runs | 3 of each suite, each case its own one-case benchmark |
| Judge | embedding cosine, threshold 0.80, on both suites (no LLM judge) |
| Claims | 386, every one located in its reply and joined to its token log-probabilities |
| Labels | the benchmark's gold-match verdicts only; nothing was labelled by a hosted model |
| Cost | US$0. No hosted call; `ANTHROPIC_API_KEY` was removed from the process |

**Two departures from the survey's harness, both deliberate.** The calibration
suite is judged here by embedding cosine alone, where `particles extractor
calibrate` adds an LLM judge in the contested band; that judge is a paid
Claude call, and it also would have made a Claude model the labeller of the
local model's output. Expect slightly stricter labels on that suite than the
calibration verb would assign. Second, the article suite was run on its own
cases one at a time, so that a claim is only ever joined against the reply
that produced it. Each case is one extraction call either way.

**Speed.** With five alternatives per token the model decoded at about 4 to 5
tokens per second, about 8 minutes per case. The full matrix took about four
hours of wall time. Whether the log-probability request itself costs speed was
not isolated.

## Features

Each claim was found verbatim (JSON-escaped) in the raw reply, and its
features were read off the tokens that generated it:

| Feature | What it is | Univariate AUROC |
|---|---|---:|
| `stated` | the extractor's `confidence_value` | 0.505 |
| `hedges` | count of a fixed 36-term hedge and attribution lexicon in the claim text | 0.538 |
| `span_mean_lp` | mean token log-probability over the claim's text | 0.656 |
| `span_min_lp` | lowest token log-probability in the claim's text | 0.617 |
| `span_mean_margin` | mean gap between the chosen token and the runner-up | **0.696** |
| `span_mean_entropy` | mean entropy over the top-5 alternatives (lower is surer) | 0.322 (0.678 inverted) |
| `span_frac_unsure` | share of the claim's tokens below probability 0.5 | 0.374 (0.626 inverted) |
| `conf_lp` | log-probability of the stated confidence's own tokens | 0.540 |
| `conf_expected` | expected confidence under the model's distribution at the first decimal digit | 0.545 |

Label: semantic match (the calibration label; every outcome other than
`spurious`). **The model's hesitation while writing a claim predicts whether
the claim is right. Its hesitation while writing the confidence number does
not**, and neither does the number. Source type and source trust, the other two
features the proposal names, are constant across these suites (every case is
`WEB_PAGE`, with no trust statement), so this page cannot measure them.

## Results: held out by case

Every claim from a case (across all three runs) is scored by a calibrator
fitted without any claim from that case. The intervals are 95 % cluster
bootstrap intervals over cases, 2 000 resamples, for the difference from the
stated-confidence baseline.

**Semantic-match label (the calibration label):**

| Calibrator | ECE | AUROC | Brier | ΔECE vs stated (95 % CI) | ΔAUROC (95 % CI) |
|---|---:|---:|---:|---|---|
| stated (no fit) | 0.243 | 0.505 | 0.306 | | |
| isotonic(stated) | 0.140 | 0.399 | 0.259 | [-0.206, +0.061] | [-0.194, -0.026] |
| logistic(stated) | 0.194 | 0.293 | 0.252 | [-0.176, +0.095] | [-0.343, -0.079] |
| logistic(stated, hedges) | 0.153 | 0.383 | 0.252 | [-0.208, +0.040] | [-0.218, -0.017] |
| logistic(log-probs) | 0.053 | 0.655 | 0.223 | [-0.239, +0.011] | [-0.002, +0.258] |
| **logistic(stated, hedges, log-probs)** | **0.052** | **0.654** | **0.225** | [-0.247, +0.000] | [+0.002, +0.252] |
| isotonic(conf_expected) | 0.151 | 0.427 | 0.260 | [-0.175, +0.045] | [-0.146, -0.004] |

**Full-match label (the harness's `calibration_error`):**

| Calibrator | ECE | AUROC | ΔECE vs stated (95 % CI) |
|---|---:|---:|---|
| stated (no fit) | 0.313 | 0.582 | |
| logistic(stated) | 0.090 | 0.484 | [-0.279, +0.024] |
| logistic(log-probs) | 0.078 | 0.614 | [-0.312, -0.022] |
| logistic(stated, hedges, log-probs) | 0.094 | 0.644 | [-0.302, -0.011] |

The stated-only calibrators rank *worse* than chance held out (AUROC 0.29 to
0.40). That is what a feature with no signal does under leave-one-case-out:
each fold's fitted slope follows that fold's case mix, and the held-out case
mix differs. A calibrator that only remaps the stated value can still lower
ECE, by moving every claim toward the base rate, and isotonic(stated) does
exactly that. It cannot tell a right claim from a wrong one, because the
stated value never could.

With 10 cases the intervals are wide. The log-probability calibrators' ECE
gain clears zero under the full-match label and touches it under the semantic
label. Their AUROC gain clears zero for the full model and touches it for the
log-probability-only model.

## Results: across genre

Fitted on all three runs of one suite, scored on all three runs of the other,
semantic label:

| Fitted on → scored on | stated ECE / AUROC | logistic(log-probs) ECE / AUROC | logistic(stated, hedges, log-probs) ECE / AUROC |
|---|---|---|---|
| flat (article) → hedged (calibration) | 0.148 / 0.521 | **0.071** / 0.655 | 0.082 / 0.664 |
| hedged (calibration) → flat (article) | 0.295 / 0.519 | **0.089** / 0.701 | 0.102 / 0.710 |

The 2026-10 survey's central negative finding was that a temperature fitted on
hedged prose miscorrects flat prose. These features do not show that failure
in either direction. A temperature has one parameter, a scale on the stated
value, and the stated value carries no information about correctness here. The
log-probability features carry information that does not depend on the genre's
confidence vocabulary.

## Against the hosted route

The survey reports ECE per run on `prose-article-seed-001`, then the mean. The
same aggregation over this page's held-out predictions:

| Route | Label | ECE per run, mean (range) |
|---|---|---|
| `local:qwen2.5:14b-instruct`, stated | calibration | 0.287 (0.189–0.411) |
| `local:qwen2.5:14b-instruct`, logistic(stated, hedges, log-probs) | calibration | **0.114** (0.095–0.147) |
| `anthropic:claude-sonnet-5`, stated (2026-10 survey) | calibration | **0.045** (0.042–0.048) |
| `local:qwen2.5:14b-instruct`, stated | harness | 0.404 (0.325–0.452) |
| `local:qwen2.5:14b-instruct`, logistic(stated, hedges, log-probs) | harness | **0.146** (0.096–0.212) |
| `anthropic:claude-sonnet-5`, stated (2026-10 survey) | harness | **0.069** (0.050–0.097) |

The calibrator closes more than half of the gap and leaves about twice the
hosted route's error. Per-run ECE is biased upward by small runs (10 bins over
65 to 97 claims, against 117 for Sonnet), which inflates the local figures
somewhat. The gap is larger than that bias plausibly accounts for.

**The hosted route's calibration is base-rate calibration, not
discrimination.** The survey's raw `claude-sonnet-5` run files from 2026-10-01
are no longer on disk. The ones that are, read at no cost, show Sonnet's
stated confidence barely ranks its own claims:

| Route (recorded runs) | Extractor | Suite | Runs | Claims | ECE, calibration label (pooled) | AUROC |
|---|---|---|---:|---:|---:|---:|
| `anthropic:claude-sonnet-5` | 0.14.0 | article 0.2.0 | 3 | 251 | 0.064 | 0.615 |
| `anthropic:claude-sonnet-4-6` | 0.15.0 | article 0.2.0 | 3 | 307 | 0.021 | 0.465 |
| `anthropic:claude-sonnet-4-6` | 0.16.0 | article 0.2.0 | 3 | 295 | 0.092 | 0.421 |
| `anthropic:claude-haiku-4-5` | 0.14.0 | article 0.2.0 | 3 | 225 | 0.173 | 0.596 |

On flat prose the hosted extractor is well calibrated mainly because it states
about 0.9 and is right about 90 % of the time. It does not know *which* 10 %
are wrong. On `qwen2.5:14b-instruct` a log-probability calibrator can know that,
which is the property a cascade needs: an escalation rule has to find the
uncertain claims, not only price the average one. The second model shows that
property is not something a local route has by default.

## Second model: `qwen3:14b`

**Measured 2026-10-08, same harness.** `qwen3:14b` (Q4_K_M, about 9 GB) was
chosen as the stronger model that fits this machine. `qwen3:30b-a3b` was
considered and skipped for memory: about 19 GB of weights on a 24 GB machine,
where the first model had already caused heavy swap. Three runs of both suites,
401 claims, every one joined.

**Thinking was turned off.** Qwen3 reasons before it answers by default. On
Ollama's OpenAI-compatible route the reasoning arrives in a separate field, but
the requested log-probabilities cover the reasoning tokens as well. A claim's
text then appears first in the model's draft, so its span would be read off the
wrong tokens. The new `reasoning_effort` field on an `llm.providers` entry,
set to `none`, removes the reasoning entirely (`think: false` did not). It is
unset by default, so no existing request changes. Decoding ran at about 3.7
minutes per case, against about 8 for the first model.

**Quality: much closer to Sonnet.**

| Model | Article-suite precision (full match), per run | Claims per run, article suite |
|---|---|---|
| `qwen2.5:14b-instruct` | 0.540, 0.392, 0.415 | 87, 97, 65 |
| `qwen3:14b` | 0.719, 0.630, 0.714 | 89, 92, 77 |
| `claude-sonnet-5` (2026-10 survey) | 0.821, 0.805, 0.746 | 117, 118, 118 |

**Calibration: no log-probability signal.** Held out by case, semantic label:

| Calibrator | ECE | AUROC | ΔECE vs stated (95 % CI) | ΔAUROC (95 % CI) |
|---|---:|---:|---|---|
| stated (no fit) | 0.222 | 0.502 | | |
| isotonic(stated) | 0.132 | 0.300 | [-0.172, +0.006] | [-0.304, -0.072] |
| logistic(log-probs) | 0.154 | 0.329 | [-0.125, -0.005] | [-0.289, -0.048] |
| logistic(stated, hedges, log-probs) | 0.176 | 0.320 | [-0.097, +0.013] | [-0.299, -0.055] |
| isotonic(conf_expected) | 0.129 | 0.400 | [-0.152, -0.017] | [-0.166, -0.019] |

Univariate AUROC of every log-probability feature lies between 0.45 and 0.53; the stated
confidence is 0.502, and it uses only 10 distinct values across 401 claims
(37 for qwen2.5). Every fitted calibrator ranks worse than chance held out,
the signature of features with no signal (see the note under the first model's
table). The ones that lower ECE do it by pulling every claim toward the base
rate.

Per-run ECE on the article suite, for the hosted comparison:

| Route | Label | ECE per run, mean (range) |
|---|---|---|
| `qwen3:14b`, stated | calibration | 0.219 (0.164–0.291) |
| `qwen3:14b`, best held-out calibrator, logistic(log-probs) | calibration | 0.129 (0.103–0.149) |
| `claude-sonnet-5`, stated | calibration | **0.045** (0.042–0.048) |
| `qwen3:14b`, stated | harness | 0.240 (0.208–0.303) |
| `qwen3:14b`, best held-out calibrator, logistic(log-probs) | harness | 0.133 (0.034–0.225) |
| `claude-sonnet-5`, stated | harness | **0.069** (0.050–0.097) |

**Copying versus knowing.** A claim copied verbatim from its source is
generated with near-certain tokens whether or not it matches gold, so a
log-probability signal could be a proxy for copying. `qwen3:14b` copies far
more often, and the split rules the proxy reading out for the first model:

| Model | Copied verbatim | Accuracy, copied / paraphrased | `span_mean_lp` AUROC, copied / paraphrased | `span_mean_margin` AUROC, copied / paraphrased |
|---|---:|---|---|---|
| `qwen2.5:14b-instruct` | 29 % | 0.65 / 0.58 | 0.511 / **0.713** | 0.653 / **0.727** |
| `qwen3:14b` | 48 % | 0.64 / 0.75 | 0.555 / 0.535 | 0.541 / 0.538 |

The first model's signal lives in its paraphrased claims, where it ranks at
0.71 to 0.73, so it is not a copying effect. The second model has no signal in
either group. Its median per-token probability over a claim is 0.99 (0.84 for
the first model), and even its paraphrased claims are generated at a median of
about 0.96. One reading, not tested here, is that the newer model's
post-training leaves its token distributions near-certain whether or not the
claim is right, which removes the margin a calibrator reads.

## Sample agreement

**Measured 2026-10-08, from the same rows, with no new model call.** The three
runs of each case are three independent samples of the same extraction: the
extractor sends no temperature, so each model samples at its own default, and
claim counts differ between runs of a case (89, 92 and 77 on the article suite
for `qwen3:14b`). For each claim, `fit --agreement` embeds every claim of the
other runs of its case with the benchmark judge's encoder and records two
features:

- `agree_frac`: the share of the other runs holding a claim at cosine
  similarity 0.80 or above (the judge's threshold);
- `agree_mean_sim`: the mean, over the other runs, of the claim's best cosine
  similarity to any claim that run produced.

Only emitted text is read, never a gold label, so both are available at
extraction time to a pipeline that samples more than once. One `qwen3:14b`
case-run (`web-essay-001`, run 3) returned a reply with no claims, the silent
dropout the 2026-10 survey recorded on another model. It is treated as a failed
sample, not as one that disagrees with everything, so that case's claims are
checked against one other sample instead of two.

**Raw features need no fit, so their AUROC is already held out.** Semantic
label; the interval is a 95 % cluster bootstrap over cases, and the
within-case figure is the mean AUROC over cases, which is what an escalation
rule faces when choosing which claims of one source to send on.

| Model | Feature | Pooled AUROC (95 % CI) | Mean within-case AUROC |
|---|---|---|---:|
| `qwen3:14b` | stated | 0.502 [0.411, 0.582] | 0.421 |
| `qwen3:14b` | `span_mean_margin` | 0.498 [0.339, 0.646] | 0.525 |
| `qwen3:14b` | **`agree_frac`** | **0.657** [0.567, 0.776] | **0.691** |
| `qwen3:14b` | `agree_mean_sim` | 0.624 [0.511, 0.795] | **0.718** |
| `qwen2.5:14b-instruct` | stated | 0.505 [0.428, 0.598] | 0.541 |
| `qwen2.5:14b-instruct` | `span_mean_margin` | 0.696 [0.611, 0.773] | 0.688 |
| `qwen2.5:14b-instruct` | `agree_frac` | 0.682 [0.612, 0.739] | 0.656 |
| `qwen2.5:14b-instruct` | `agree_mean_sim` | 0.708 [0.608, 0.786] | 0.665 |

Accuracy by agreement level:

| Model | Other samples agreeing | Share of claims | Accuracy |
|---|---|---:|---:|
| `qwen3:14b` | none | 16 % | 0.45 |
| `qwen3:14b` | half | 15 % | 0.54 |
| `qwen3:14b` | all | 68 % | 0.79 |
| `qwen2.5:14b-instruct` | none | 25 % | 0.36 |
| `qwen2.5:14b-instruct` | half | 30 % | 0.54 |
| `qwen2.5:14b-instruct` | all | 46 % | 0.76 |

**As a calibrator it helps one model and not the other.** Fitted held out by
case, agreement calibrators keep less of the raw ranking than the features
alone: on `qwen3:14b` they reach a pooled held-out AUROC of 0.54 to 0.56, with
ECE 0.139 against 0.132 for an isotonic map of the stated value, and a best
per-run article ECE of 0.175 (calibration label). The ranking survives in the
raw feature and is lost in the fit, which with 10 cases has little to learn
the per-case base rate from. On `qwen2.5:14b-instruct`, logistic(agreement) is
the best calibrator measured on this page: held-out ECE 0.039, AUROC 0.671,
and a per-run article ECE of 0.100 (0.077 to 0.138), still about twice
`claude-sonnet-5`'s 0.045.

**Across genre, agreement transfers.** Fitted on one suite and scored on the
other, the agreement calibrators rank at 0.58 to 0.69 on `qwen3:14b` and 0.63
to 0.74 on `qwen2.5:14b-instruct`, in both directions.

**It did not replicate.** The escalation measurement below needed three fresh
`qwen3:14b` runs with each claim's subjects recorded, which gave a second,
independent set of samples. On it, the raw features lose the pooled signal:

| `qwen3:14b`, fresh runs (2026-10-08) | Pooled AUROC (95 % CI) | Mean within-case AUROC |
|---|---|---:|
| stated | 0.588 [0.456, 0.690] | 0.471 |
| `span_mean_margin` | 0.469 [0.311, 0.657] | 0.655 |
| `agree_frac` | 0.529 [0.382, 0.671] | 0.649 |
| `agree_mean_sim` | 0.512 [0.362, 0.666] | 0.643 |

| Other samples agreeing (fresh runs) | Share of claims | Accuracy |
|---|---:|---:|
| none | 20 % | 0.83 |
| half | 13 % | 0.53 |
| all | 67 % | 0.79 |

These figures treat a run that returned fewer than five claims as a failed
sample (`--min-sample-claims 5`) and drop the 18 claims whose case had no other
healthy sample. The fresh runs had more dropouts than the first: one case came
back empty in two of three runs, and two case-runs returned a single claim,
each of which "disagrees" with nearly everything. With empty runs alone treated
as failed (`--min-sample-claims 1`), `agree_frac` ranks at 0.515. Either way the
first set's pooled result does not hold. The within-case ordering (0.63 to 0.65
here, 0.69 before) is the part that survived both samplings.

Four limits:

- **Agreement measures stability, not truth.** A claim every sample
  hallucinates the same way agrees with itself. On these suites the
  all-agree group is still wrong 21 to 24 % of the time.
- **The feature and the label share an encoder.** The judge matches a claim to
  gold at cosine 0.80 with the same model the feature uses, so a canonically
  phrased claim is favoured by both. This inflates the measured link somewhat,
  by an amount this page cannot separate.
- **Two other samples is the floor.** `agree_frac` takes three values; more
  samples would grade it more finely and cost more.
- **It costs a sample per sample.** Three samples is three local extraction
  calls per chunk, about 11 minutes per case for `qwen3:14b` on this machine.
  A cascade could sample a second time only where the first pass is cheap to
  check, which this page does not measure.

## Escalation to the hosted extractor

**Measured 2026-10-08, one paid run.** The question a cascade has to answer is
whether sending part of the work to the hosted model recovers the hosted
route's quality for less than the hosted route costs. One `claude-sonnet-5`
extraction pass covered both suites for three runs: 31 calls, 574 claims,
**US$2.07** at list price against an estimate of about US$2.00 and an agreed
ceiling of US$2.50, no budget stop and no error. Every escalation policy was
then simulated offline from that pass and three fresh `qwen3:14b` runs, pairing
each case's local run *n* with its hosted run *n*.

All arms are scored with the benchmark's own matcher (embedding cosine 0.80,
subject-aware), re-run on each arm's combined output so that recall is
recomputed. Re-matching a single source reproduces the harness's own verdict on
every claim (321 of 321 local, 574 of 574 hosted). A chunk is one case, one
extraction call, so the hosted cost of an arm is its share of hosted calls
times about US$0.069 per chunk. Local cost is not in dollars: three
`qwen3:14b` samples per chunk, about 11 minutes per case on this machine. A
local run with fewer than five claims is a failed sample and always escalates.

The arms:

- **filter**: keep only the local claims every other sample also produced; no
  hosted call.
- **claim-level**: keep the agreed local claims, and for any chunk with a
  disagreed claim add the hosted output (less its repeats of kept claims).
- **chunk-level, cut X**: send the whole chunk to the hosted model when more
  than X of its local claims are disagreed; otherwise keep the local output.

**Article suite (flat prose)**, mean over three runs (range):

| Arm | Hosted calls | Precision | Recall | ECE, calibration label | ECE, harness label |
|---|---:|---|---|---|---|
| `qwen3:14b` alone | 0 % | 0.794 (0.771–0.827) | 0.513 (0.420–0.640) | 0.132 (0.121–0.139) | 0.140 (0.112–0.155) |
| `claude-sonnet-5` alone | 100 % | 0.780 (0.732–0.809) | 0.793 (0.780–0.800) | 0.058 (0.039–0.078) | 0.092 (0.068–0.106) |
| filter | 0 % | 0.839 (0.800–0.871) | 0.367 (0.340–0.400) | 0.108 (0.092–0.124) | 0.117 (0.092–0.136) |
| claim-level | 94 % | 0.726 (0.691–0.752) | **0.853** (0.840–0.860) | 0.115 (0.095–0.139) | 0.140 (0.117–0.169) |
| chunk-level, cut 20 % | 67 % | 0.768 (0.758–0.787) | 0.740 (0.680–0.800) | 0.074 (0.046–0.090) | 0.095 (0.071–0.110) |
| **chunk-level, cut 33 %** | **56 %** | **0.790** (0.784–0.798) | **0.760** (0.720–0.800) | **0.070** (0.046–0.094) | **0.087** (0.071–0.110) |
| chunk-level, cut 50 % | 39 % | 0.794 (0.784–0.810) | 0.747 (0.700–0.820) | 0.091 (0.072–0.108) | 0.096 (0.073–0.110) |

**Both suites pooled:**

| Arm | Hosted calls | Precision | Recall | ECE, calibration label | ECE, harness label |
|---|---:|---|---|---|---|
| `qwen3:14b` alone | 0 % | 0.740 | 0.471 | 0.166 | 0.174 |
| `claude-sonnet-5` alone | 100 % | 0.777 | 0.746 | 0.059 | 0.079 |
| claim-level | 93 % | 0.712 | 0.771 | 0.129 | 0.149 |
| chunk-level, cut 20 % | 67 % | 0.761 | 0.667 | 0.087 | 0.106 |
| chunk-level, cut 33 % | 43 % | 0.761 | 0.602 | 0.090 | 0.111 |
| chunk-level, cut 50 % | 27 % | 0.753 | 0.578 | 0.117 | 0.131 |

What the tables show:

- **`qwen3:14b` alone matches Sonnet on precision on flat prose and loses on
  recall.** Its claims are as often right (0.794 against 0.780); it states
  about two-thirds as many of the required ones. On this fresh set, its gap
  to the hosted route is mostly recall and dropouts, not wrong claims.
  This is not the same figure as the 0.63 to 0.72 in
  [the second model's section](#second-model-qwen314b). Both are full-match
  precision under the same matcher; the difference is the sample. The fresh
  runs emitted far fewer article-suite claims (52, 83 and 51 per run, against
  89, 92 and 77), and their precision rose as their count fell.
  Read "matches Sonnet on precision" as holding on a run set where it said
  less, not as a property of the model.
- **Chunk-level escalation at a 33 % cut is the closest any local arrangement
  came.** On the article suite it sends 56 % of chunks to Sonnet, about 44 %
  less hosted spend, with precision and both ECE figures inside Sonnet's own
  run-to-run range and recall 3 points lower (0.760 against 0.793).
- **The same cut does not transfer to hedged prose.** On the calibration
  suite it escalates 3 of 12 chunks and recall falls to 0.469 against Sonnet's
  0.706. `qwen3:14b`'s samples disagree less on hedged prose while being wrong
  more, so a disagreement threshold tuned on one genre underescalates on the
  other. Pooled across both suites, no cut reaches the hosted route.
- **Claim-level escalation buys recall, not savings.** Adding the hosted
  output to the agreed local claims gives the highest recall measured on this
  page (0.853 on the article suite against Sonnet's 0.793), because the two
  models miss different claims. It calls the hosted model on 93 % of chunks
  and loses precision, so it is an ensemble, not a cascade.
- **Selection caveat.** The 33 % cut was chosen from three tried on this data,
  with 18 article-suite chunks. Treat it as a hypothesis for a held-out run,
  not a measured rule.

## What this means for the proposal

- **The calibrator (step 2) is not justified.** Token log-probabilities rank
  claims on one model and not the stronger one; agreement features did not
  replicate; no calibrator brings either model to the hosted route's ECE.
- **The cascade (step 3) is the live part, at chunk granularity.** Escalating
  whole chunks on a disagreement threshold came within the hosted route's
  spread on flat prose at about 56 % of its hosted calls. It needs a threshold
  that is not tuned on the data it is scored on, and a genre signal, since the
  same threshold underescalates hedged prose.
- **Recall, not precision, is the local model's gap** on this fresh set, and
  dropouts are a share of it. An escalation trigger on a low claim count per
  chunk (the dropout case, generalised) is untested here and cheap to try.
- **What would settle it:** fix the cut in advance, run fresh local and hosted
  passes on a held-out suite, and score the cascade against the hosted route
  with its cost. That is a second paid run of about the same size.

## How to reproduce

```bash
# llm.extraction routed to an OpenAI-compatible endpoint that returns
# log-probabilities, e.g. provider: local, model: qwen2.5:14b-instruct
PARTICLES_CONFIG=local.yaml uv run python scripts/measure_logprob_calibrator.py \
    run --runs 3 --out rows.jsonl
uv run python scripts/measure_logprob_calibrator.py fit rows.jsonl --coef --agreement
uv run python scripts/measure_logprob_calibrator.py hosted
# escalation: a hosted pass of the same cases, then every policy offline
PARTICLES_CONFIG=sonnet.yaml uv run python scripts/measure_logprob_calibrator.py \
    run --hosted --max-usd 2.50 --runs 3 --out hosted.jsonl
uv run python scripts/measure_logprob_calibrator.py escalate \
    --local rows.jsonl --hosted hosted.jsonl --min-sample-claims 5
```

`run` opens `particles.llm.logprobs.capture_logprobs()` around each case, which
is the only thing that makes the adapter request log-probabilities; outside it
the request is unchanged. The feature rows behind every table on this page are
committed beside it as
[`logprob-calibrator-qwen2.5-14b-2026-10-07.jsonl`](logprob-calibrator-qwen2.5-14b-2026-10-07.jsonl)
[`logprob-calibrator-qwen3-14b-2026-10-08.jsonl`](logprob-calibrator-qwen3-14b-2026-10-08.jsonl),
and, for the escalation measurement, the fresh `qwen3:14b` runs with subjects
([`logprob-calibrator-qwen3-14b-subjects-2026-10-08.jsonl`](logprob-calibrator-qwen3-14b-subjects-2026-10-08.jsonl))
and the hosted pass
([`logprob-calibrator-claude-sonnet-5-2026-10-08.jsonl`](logprob-calibrator-claude-sonnet-5-2026-10-08.jsonl)),
so `fit --coef` reproduces them without rerunning either model. For `qwen3:14b`
the provider entry also sets `reasoning_effort: none`.

Three operational notes. Ollama serves one request at a time by default, so a
long extraction call starves any other client of the same server for its whole
duration. The 24 576-token context first tried pushed this machine into heavy
swap. The three run-3 cases on hedged prose were re-run after the server was
stopped partway through, and no row from the failed attempt was kept.

## Cross-references

- The held-out follow-up, with the escalation rule fixed in advance:
  [Escalation on a held-out suite (2026-10-11)](escalation-holdout-2026-10-11.md).
- The survey whose open question this page answers:
  [Provider survey (2026-10)](provider-survey-2026-10.md).
- Immutable stated confidence, per-pairing temperature scaling,
  and the local adapter (generalized).
