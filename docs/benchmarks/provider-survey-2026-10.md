# Provider survey: calibrated extraction (2026-10)

**A dated snapshot, as of 2026-10-01.** The
[2026-09 survey](provider-survey-2026-09.md) ran every extraction model raw. Its
cheapest row with full recall, `deepseek-v4p1-flash`, matched `claude-sonnet-5`
on recall at a fraction of the cost and lost on calibration. Calibration is the
axis that kept the default where it is, because a stated confidence is never
rewritten once a particle is created. Per-provider temperature
scaling exists to correct exactly that kind of loss, and the 2026-09
survey never fitted it. This page fits it for every row and measures whether
the corrected number closes the gap. A fifth row, `claude-sonnet-5-5` (the
current Sonnet, released after the 2026-09 survey), was added on the same day
under the same harness and is read separately below.

**It does not.** The fitted temperature made held-out calibration worse, not
better, for every row that produced a usable measurement. The exception,
`glm-5p3-flash`, returned no claims at all on most cases, and its two usable
runs moved in opposite directions under the two labels this page reports. The cheap route's calibration gap therefore
cannot be closed by this lever, and no cheaper extraction preset ships on the
strength of this page. Two findings matter more than that verdict, and both are
below: most of the 2026-09 gap was produced by how the benchmark labels a
claim, not by the model, and a temperature fitted on the calibration suite does
not transfer to the benchmark's prose for any of the four models it could be
measured on.

**Added later the same day: the production calibration hurts on flat prose.**
The one calibration actually persisted for the general extractor
(`claude-sonnet-4-6`, T = 2.1736, fitted on the same calibration suite) was
scored the same way against recorded benchmark runs, and it raises held-out
ECE at both extractor versions it has been live beside. It has calibrated no
particle in the owner's store, because extraction has run on `claude-sonnet-5`
throughout. `particles extractor calibrate` now refuses any fit that raises
ECE on recorded runs it was not fitted on. See
[the production calibration, held out](#the-production-calibration-held-out).

It measures a single **extractor's** output against gold particles (the
`particles extractor benchmark*` family), not the whole-pipeline
agent-memory benchmark on the [Benchmarks](../benchmarks.md) page.

!!! danger "This is not a model leaderboard, and it must not be read as one"
    **The extraction prompt, the equivalence judge, and the calibration suite
    were all developed against Claude Sonnet.** Every non-Sonnet row here is
    depressed by an amount nobody has quantified, and that includes both
    Fireworks rows. The calibration judge is itself a Claude model.

    These numbers measure **how well each model drives this extractor, on this
    prompt, under this judge**: a procurement question about our own
    pipeline. They are not a statement about model capability, and a lower row
    is not a worse model.

!!! warning "Compare rows only within this page"
    The benchmark suite moved to v0.3.0 (two cases and gold subjects added),
    the extractor moved to 0.16.0, and the completion budget is now the same on
    every route. None of the 2026-09 or 2026-08 figures are comparable with
    the figures here, including the `claude-sonnet-5` row, which was re-run
    rather than carried over.

## What was measured

| Parameter | Value |
|---|---|
| Benchmark suite | `prose-article-seed-001` v0.3.0 |
| Benchmark cases | 6 |
| Required claims | 50 (across the 6 cases) |
| Benchmark judge | embedding cosine, threshold ≥ 0.80 |
| Calibration suite | `prose-calibration-001` v0.1.0 (4 cases, 59 gold claims) |
| Calibration passes | **13 per model**, pooled into one fit |
| Calibration judge | LLM judge in the contested [0.65, 0.80) cosine band, `claude-sonnet-4-6` for every row |
| Extractor | `general-extractor` 0.16.0 |
| Benchmark runs per model | **3** |
| Completion budget | 32 000 on every route, one retry at 64 000 |
| Measured | 2026-10-01 |

**Fitted on one suite, scored on another.** Each row's temperature was fitted
on the calibration suite, exactly as `particles extractor calibrate
general-extractor --runs 13` fits it, and then applied to the benchmark suite's
claims, which the fit never saw. A single pass's temperature
varies widely (see the per-pass range below), while an earlier 13-pass
measurement on this suite found every leave-one-out refit within 2 % of
the pooled value.

**How the fitted figure is computed.** The benchmark harness converts every
candidate without calibration, so each run file records the raw stated
confidence of every emitted claim. The fitted ECE applies the row's fitted
temperature to those recorded values and recomputes expected calibration error
over the same labels. As a check on the method, the raw ECE recomputed this way
matched the harness's own `calibration_error` exactly on all fifteen runs.

## Results: quality

Mean over 3 runs, with the observed range and sample standard deviation. Rows
are in matrix order, **not ranked**; see the warning above.

| Model | Route | Recall (mean) | range | sd | Precision (mean) | sd | Emitted per run |
|---|---|---:|---:|---:|---:|---:|---|
| `claude-sonnet-5` | Anthropic | 0.827 | 0.80–0.86 | 0.031 | 0.790 | 0.039 | 117, 118, 118 |
| `claude-sonnet-5-5` | Anthropic | 0.707 | 0.68–0.72 | 0.023 | 0.755 | 0.031 | 94, 101, 98 |
| `claude-haiku-4-5` | Anthropic | 0.800 | 0.76–0.84 | 0.040 | 0.855 | 0.031 | 107, 111, 107 |
| `deepseek-v4p1-flash` | Fireworks | 0.827 | 0.82–0.84 | 0.012 | 0.693 | 0.037 | 152, 159, 165 |
| `glm-5p3-flash` | Fireworks | 0.193 | 0.00–0.32 | 0.170 | 0.791 ‡ | 0.064 | 59, 0, 43 |

Recall is the fraction of the 50 required claims recovered; precision is the
fraction of emitted claims matching a required claim. The Fireworks model ids
are `accounts/fireworks/models/<name>`. Subject resolution (the suite's new
gold-subject column) is a property of the resolver, not of the extraction
model, so it cannot separate the rows and is not tabulated. It measured 0.886 on
every run of the first four rows and 0.914 on the `claude-sonnet-5-5` row,
which ran after a subject-resolver fix had landed; nothing in that fix touches
extraction, and the extractor version is unchanged.

‡ **`glm-5p3-flash`'s figures are not quality measurements.** It returned no
claims at all on **13 of 18** benchmark case-runs, including every case of
run 2, and on 28 of 52 calibration case-runs, with no error, no truncation, and
an HTTP 200 on every call. The 2026-09 survey recorded the same failure on
5 of 12 case-runs. A run with no claims scores precision 1.000 and ECE 0.000
under the harness's convention, so this row's precision and every ECE figure
for it on this page are computed over the two runs that produced claims (runs 1
and 3), and its recall of 0.193 is a reliability figure, not a quality one.
Read every `glm-5p3-flash` number below with that in mind.

## Results: calibration, raw beside fitted

Expected calibration error (ECE, lower is better) as the harness reports it,
mean over 3 runs with range and sample standard deviation, raw and with the
row's fitted temperature applied.

| Model | Fitted T | Fit pairs | Per-pass T range | ECE raw | range | sd | **ECE fitted** | range | sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `claude-sonnet-5` | 0.775 | 903 | 0.54–0.95 | 0.069 | 0.050–0.097 | 0.024 | **0.109** | 0.080–0.152 | 0.038 |
| `claude-sonnet-5-5` | 0.650 | 720 | 0.41–0.89 | 0.099 | 0.075–0.120 | 0.022 | **0.171** | 0.144–0.188 | 0.023 |
| `claude-haiku-4-5` | 0.874 | 792 | 0.60–0.98 | 0.054 | 0.037–0.066 | 0.015 | **0.055** | 0.039–0.081 | 0.023 |
| `deepseek-v4p1-flash` | 1.944 | 1 131 | 1.43–4.65 | 0.137 | 0.113–0.157 | 0.022 | **0.177** | 0.142–0.210 | 0.034 |
| `glm-5p3-flash` ‡ | 1.885 | 495 | 0.98–3.76 | 0.092 | 0.051–0.134 | 0.059 | **0.084** | 0.083–0.086 | 0.002 |

Every fit cleared all four of the calibration verb's guards: no saturated
confidences, non-degenerate labels, a temperature off the optimizer's bounds,
and an in-sample improvement on the calibration suite. Each would therefore have
been **persisted** by `particles extractor calibrate`. A temperature below 1
raises confidences and one above 1 lowers them: every Claude row fits a raise
and both Fireworks rows fit a cut.

**The decision rule this page was run to apply.** A cheaper extraction route
would ship as a named preset only if its fitted ECE came within
`claude-sonnet-5`'s own run-to-run spread. For the row the rule was written
for, it does not, under either label on this page. Fitted,
`deepseek-v4p1-flash` measures 0.177 against a Sonnet range of 0.050–0.097, and
the fit moved it away from Sonnet, not toward it (raw 0.137).
`glm-5p3-flash`'s fitted 0.084 does fall inside Sonnet's range on this label,
and it is still not a candidate: the figure rests on two runs and about a
hundred claims, it measures 0.105 against Sonnet's 0.042–0.048 under the
calibration label below, and the model returned no claims at all on 13 of 18
case-runs, so its recall is 0.193. The rule was written for a row that already
matched Sonnet on recall, and only `deepseek-v4p1-flash` does.

### Most of the 2026-09 gap is the label, not the model

The harness's ECE counts a claim as correct only when it is a **full match**: a
claim that matches a gold particle but is stated below that particle's
`confidence_min` floor is scored as an under-confidence partial match, and the
ECE counts it as wrong. The floors on this suite run from 0.60 to 0.85. The
calibration verb deliberately does the opposite: its label is *semantic match*
and includes those partial matches, because labelling a correct claim incorrect
for being stated timidly teaches a temperature that low confidence predicts
error. Scoring the same runs under the calibration label:

| Model | ECE raw, calibration label | sd | ECE fitted, calibration label | sd |
|---|---:|---:|---:|---:|
| `claude-sonnet-5` | 0.045 | 0.004 | 0.080 | 0.026 |
| `claude-sonnet-5-5` | 0.062 | 0.014 | 0.104 | 0.015 |
| `claude-haiku-4-5` | 0.049 | 0.004 | 0.056 | 0.012 |
| `deepseek-v4p1-flash` | 0.053 | 0.022 | 0.095 | 0.008 |
| `glm-5p3-flash` ‡ | 0.079 | 0.071 | 0.105 | 0.034 |

Under the label calibration itself uses, `deepseek-v4p1-flash`'s raw mean is
0.053 against Sonnet's 0.045, so most of the difference the harness reports is
the floor, not the model. What survives is **stability**: its sd is five times
Sonnet's, the same instability the 2026-09 page recorded under the other label.
That gap still sits outside Sonnet's spread (0.042–0.048), so the verdict above
holds under both labels, and in neither does the fitted temperature help.

From 1.169.3 the harness reports both figures itself, `calibration_error`
under full match and `calibration_error_semantic` under the calibration label,
and the second reproduces this table's per-run values from the same run files.

### Why the fitted temperature does not transfer

A temperature is one number, so it can only correct an error that has the same
shape everywhere. The two suites disagree about what a stated confidence means.
Accuracy by confidence bin, under the calibration label on both suites
(matched / emitted, pooled over all passes and runs):

| Model | Bin | Calibration suite (13 passes) | Benchmark suite (3 runs) |
|---|---|---:|---:|
| `claude-sonnet-5` | 0.8–0.9 | 95 % (352 / 370) | 85 % (104 / 122) |
| `claude-sonnet-5` | 0.9–1.0 | 91 % (306 / 337) | 90 % (161 / 178) |
| `claude-sonnet-5-5` | 0.8–0.9 | 100 % (206 / 207) | 80 % (44 / 55) |
| `claude-sonnet-5-5` | 0.9–1.0 | 96 % (361 / 376) | 89 % (165 / 186) |
| `claude-haiku-4-5` | 0.8–0.9 | 92 % (197 / 214) | 81 % (66 / 81) |
| `deepseek-v4p1-flash` | 0.8–0.9 | 84 % (380 / 454) | 79 % (120 / 151) |
| `deepseek-v4p1-flash` | 0.9–1.0 | **69 %** (328 / 476) | **90 %** (197 / 220) |

The calibration suite is built from prose in which the author's own certainty
varies (provisional counts, disputed figures, printed corrections), and it is
built that way on purpose, so that an extractor states a spread of
confidences a temperature can be fitted to. The benchmark suite is flatly
stated prose. On the hedged prose `deepseek-v4p1-flash` is right only 69 % of
the time when it states 0.9 or more, so the fit learns a strong cut
(T = 1.94); on the flat prose the same stated confidence is right 90 % of the
time, and the cut drags those claims from about 0.92 to about 0.78. The Claude
rows show the mirror image: their fits learn a raise from the hedged prose and
overshoot on the flat prose, by as much as 20 points in `claude-sonnet-5-5`'s
0.8 bin.

**This is a finding about the calibration suite's reach, not about any model.**
A temperature fitted on `prose-calibration-001` corrects the genre it was fitted
on and miscorrects a different one. The guard that refuses a non-improving fit
cannot catch this, because it evaluates in-sample. Two cautions follow. The
judges differ across the two columns (an LLM judge in the contested band against
cosine alone), which lowers the benchmark column somewhat. That can account for
part of the Claude rows' gaps of 10 to 20 points; it works against the 21-point
gap in the `deepseek-v4p1-flash` 0.9 bin, where the benchmark column is the
higher one, so that gap is if anything understated. Any production calibration
fitted on this suite carries the same exposure on flatly stated sources; this
page measured that only for the rows above.

### `claude-sonnet-5-5`, the current Sonnet

`claude-sonnet-5-5` was released after the 2026-09 survey at the same list price
and on the same tokenizer as `claude-sonnet-5`, so on this harness it is a
quality question with a cost side effect. It ran at the model's default effort,
with thinking on (the adapter sends no thinking setting), on the same suite,
extractor, judge and budget as every other row.

On this harness it is **not a drop-in replacement for the default**. It
recovered fewer of the required claims (recall 0.707 against 0.827), emitted
fewer claims per run (94–101 against 117–118), and was less precise (0.755
against 0.790) and less well calibrated under both labels (raw ECE 0.099 against
0.069, and 0.062 against 0.045 under the calibration label). It cost 1.8× less
per run, almost entirely because it produced about half the output tokens
(59 760 against 111 950 over three runs).

The danger box at the top applies with a twist. The prompt and the judge were
developed against `claude-sonnet-5`, its predecessor, and the judge matches
emitted and gold claims one to one. A model that states the same facts as fewer,
larger claims loses recall under that judge without missing a fact. This page
did not check whether that is what happened, so the recall gap is a finding
about how `claude-sonnet-5-5` drives this extractor as tuned today, not about
what it can extract. Its fitted temperature failed to transfer in the same way
as every other row's (0.099 raw against 0.171 fitted).

## Results: cost

**Measured, not projected.** Every figure below is the token usage the providers
returned, recorded by the SDK's own usage tracker at the completion port and
priced at list price (prompt-cache writes at 1.25× and reads at 0.10× the
input rate).

| Model | Price in/out per MTok | Calls (3 runs) | Input tok (in + cache-w + cache-r) | Output tok | **$ / run** | Relative |
|---|---|---:|---|---:|---:|---:|
| `claude-sonnet-5` | $2 / $10 | 18 | 32 841 + 0 + 51 624 | 111 950 | **0.3985** | 1.0× |
| `claude-sonnet-5-5` | $2 / $10 | 18 | 32 817 + 0 + 51 624 | 59 760 | **0.2245** | 1.8× cheaper |
| `claude-haiku-4-5` | $1 / $5 | 18 | 62 095 + 0 + 0 | 62 767 | **0.1253** | 3.2× cheaper |
| `deepseek-v4p1-flash` | $0.22 / $0.66 | 18 | 43 106 + 0 + 24 642 | 151 036 | **0.0366** | 10.9× cheaper |
| `glm-5p3-flash` | $0.15 / $0.50 | 18 | 57 885 + 0 + 0 | 297 118 | **0.0524** | 7.6× cheaper |

| Model | Recall | $ / run | **$ per run per unit recall** |
|---|---:|---:|---:|
| `claude-sonnet-5` | 0.827 | 0.3985 | **0.482** |
| `claude-sonnet-5-5` | 0.707 | 0.2245 | **0.318** |
| `claude-haiku-4-5` | 0.800 | 0.1253 | **0.157** |
| `deepseek-v4p1-flash` | 0.827 | 0.0366 | **0.044** |
| `glm-5p3-flash` | 0.193 | 0.0524 | 0.271 *(see dropouts)* |

Fitting is a separate, one-off cost per model: 52 extraction calls plus the
calibration judge's calls.

| Model | Extraction calls | Judge calls | **Fit cost** |
|---|---:|---:|---:|
| `claude-sonnet-5` | 53 | 141 | **$3.35** |
| `claude-sonnet-5-5` | 53 | 94 | **$1.81** |
| `claude-haiku-4-5` | 52 | 107 | **$0.99** |
| `deepseek-v4p1-flash` | 52 | 164 | **$0.36** |
| `glm-5p3-flash` | 52 | 63 | **$0.51** |

The calibration judge cost under $0.08 per model; the fit is almost entirely
extraction. **Total spend for this page: US$9.54**: US$7.05 for the first
four rows against a pre-agreed ceiling of $15, and US$2.48 for the
`claude-sonnet-5-5` row against a separately agreed US$4.50.

## What this means for the cheap route

- **No cheaper extraction preset ships.** The measurement it was conditioned on
  came back negative, so the default extraction routing is unchanged and the
  cheap route remains a configuration change an operator makes for their own
  reasons.
- **The calibration lever to pull is not a temperature on this suite.** A
  fitted correction that transfers needs either a calibration suite in the
  genre being extracted (one per source genre, selected at extraction time) or
  a correction with more than one parameter. Until one exists, a cheap
  extractor's confidences are better stored raw, as `EXTRACTOR_DIRECT`, than
  scaled by a temperature fitted elsewhere.
- **The read-side cap is available and off by default.**
  `confidence.uncalibrated_cap` dampens an `EXTRACTOR_DIRECT` confidence at
  query time without rewriting the stored value. It does not change
  anything measured on this page, which scores stored values.
- **`claude-sonnet-5-5` does not displace the default on these numbers.** It
  is cheaper per run and behind on recall, precision and calibration. Whether a
  prompt tuned for it closes that gap is an open question this page cannot
  answer.
- **The case for `claude-sonnet-5` on calibration is narrower than the 2026-09
  page made it look.** Under the calibration label the raw means are close;
  the incumbent's real advantage is stability, and precision (0.790 against
  0.693).

## The production calibration, held out

**Measured 2026-10-01, after the matrix above, at no API cost**.
Every fit above was a `--dry-run` and none was persisted. One general-extractor
calibration *is* persisted: the pooled 13-pass fit for
`anthropic:claude-sonnet-4-6` that the owner ran on 2026-09-17, on the same
calibration suite, under general extractor 0.15.0. It cleared every in-sample
guard with a large margin and had never been scored out of sample.

| Parameter | Value |
|---|---|
| Stored record | `general-extractor` × `anthropic:claude-sonnet-4-6`, logit transform |
| Temperature | 2.1736 (a cut: 0.93 becomes 0.767, 0.85 becomes 0.690) |
| Fitted on | `prose-calibration-001`, 13 passes pooled, 1 123 pairs, 2026-09-17 |
| In-sample ECE | 0.144 → 0.038 |
| Held-out data | 7 recorded `claude-sonnet-4-6` runs of `prose-article-seed-001` under `benchmark.runs_dir` |
| Held-out claims | 307 at extractor 0.15.0, 448 at extractor 0.16.0 |

**The method is this page's.** Each recorded run's raw stated confidences were
scaled by the stored temperature and the ECE recomputed over the same labels;
the recomputed raw ECE matched the harness's own `calibration_error` on all
seven runs. Suite 0.3.0 differs from 0.2.0 only by two added cases and the
subject-resolution block, so the four original cases carry the same gold in
both, and the 0.2.0 runs are held-out data for this question. Two runs dated
2026-09-27 that also report suite 0.3.0 were scored against a reworded gold set
that never reached `main`, and they are excluded. The judge caveat above
applies: these runs used the embedding judge, the fit an LLM judge.

### Verdict: it hurts

| Extractor | Runs | Label | ECE raw, mean (range) | **ECE fitted, mean (range)** | Runs made worse |
|---|---:|---|---|---|---:|
| 0.15.0 (fitted on this version) | 3 | harness | 0.035 (0.025–0.052) | **0.170** (0.151–0.196) | 3 of 3 |
| 0.15.0 | 3 | calibration | 0.030 (0.016–0.052) | **0.165** (0.151–0.193) | 3 of 3 |
| 0.16.0 (current) | 4 | harness | 0.100 (0.090–0.109) | **0.108** (0.096–0.127) | 3 of 4 |
| 0.16.0 | 4 | calibration | 0.100 (0.095–0.106) | **0.111** (0.095–0.126) | 3 of 4 |

On flatly stated prose the production calibration makes calibration worse. On
the extractor version it was fitted under, it multiplies ECE by about five: that
version was already well calibrated on flat prose, and the cut took its 0.93s,
which were right 91.5 % of the time, down to about 0.76. On the current version
the harm is small but in the same direction under both labels, and the pooled
figures agree (0.091 → 0.106 harness, 0.093 → 0.111 calibration label).

Accuracy by confidence bin under the calibration label, pooled over the runs of
each version (correct / claims):

| Extractor | Bin | Raw: mean stated | Raw: accuracy | Fitted: mean stated | Fitted: accuracy |
|---|---|---:|---:|---:|---:|
| 0.15.0 | 0.9–1.0 raw → 0.7–0.8 fitted | 0.928 | 91.5 % (225 / 246) | 0.760 | 90.7 % (243 / 268) |
| 0.15.0 | 0.8–0.9 raw → 0.6–0.7 fitted | 0.853 | 88.9 % (48 / 54) | 0.669 | 96.9 % (31 / 32) |
| 0.16.0 | 0.9–1.0 raw → 0.7–0.8 fitted | 0.932 | 84.2 % (299 / 355) | 0.764 | 85.0 % (328 / 386) |
| 0.16.0 | 0.8–0.9 raw → 0.6–0.7 fitted | 0.856 | 77.8 % (63 / 81) | 0.670 | 83.0 % (39 / 47) |

The fitted bins hold almost the same claims as the raw bins one row up (the cut
moves 0.95 below 0.80, so a few cross over). The raw extractor at 0.16.0 is
overconfident by about nine points in its most populated bin; the fitted one is
underconfident by about nine points in the same claims. The temperature learned
a cut of the right sign for this version and about twice the right size,
because the hedged prose it was fitted on is where the extractor's high
confidences are least often right.

### Exposure: none so far, latent from here

No particle in the owner's store carries this temperature. Every
general-extractor particle extracted since the record was persisted ran under
`anthropic:claude-sonnet-5`, which has no stored calibration, so all of them are
`EXTRACTOR_DIRECT`. The 253 `CALIBRATED_BENCHMARK` particles in the store
predate it (2026-06-25, general extractor 0.10.0). The record stays applied for
its pairing, though: configuring extraction back to `claude-sonnet-4-6` would
start writing these cut confidences, immutably. Retiring it is
one command, and the right one on this evidence:

```bash
uv run particles extractor calibration-forget general-extractor anthropic:claude-sonnet-4-6
```

### What transfers, and what does not

Two further measurements on the same recorded runs, each a pooled fit scored on
runs it did not see, under the calibration label:

| Fitted on | Scored on | T | ECE raw → fitted |
|---|---|---:|---|
| 3 runs at 0.16.0, leaving one out (4 folds) | the fourth run | 1.57–1.64 | improved in **4 of 4** folds (0.095–0.106 → 0.030–0.065) |
| 2 runs at 0.15.0, leaving one out (3 folds) | the third run | 1.02–1.11 | improved in 1 of 3 folds |
| all 3 runs at 0.15.0 | all 4 runs at 0.16.0 | 1.067 | 0.094 → 0.086 |
| all 4 runs at 0.16.0 | all 3 runs at 0.15.0 | 1.609 | 0.021 → **0.096** |

A temperature fitted on flat prose does transfer to other flat prose of the same
extractor version. A temperature is therefore a property of the genre and of the
extractor version as well as of the `provider:model` pairing, and the stored
record is keyed on the pairing alone. Both suites declare the same source type
(`WEB_PAGE`), so nothing available at extraction time can tell hedged prose from
flat prose and select a fit per genre.

### What changed

- **`particles extractor calibrate` now checks every fit out of sample before
  persisting it.** The fitted temperature is scored on every recorded
  `extractor benchmark` run under `benchmark.runs_dir` for the same extractor,
  extractor version and extraction `provider:model`, excluding the suites the
  fit consumed. A fit that raises their ECE is refused, and so is a fit with no
  such run to check against. The check reads files, so it makes no API call.
  Against the owner's recorded runs it refuses the production temperature.
- **A persisted record carries its held-out figures**, and `particles extractor
  calibrations` prints them, or flags a record persisted before the check as
  never checked out of sample.
- **No stored confidence was rewritten.** Particles keep the confidence they were
  created with. The check governs only what may be persisted from
  here on.

Until a genre signal exists at extraction time, the general extractor is best
left uncalibrated, minting `EXTRACTOR_DIRECT` particles with their raw stated
confidence. The held-out check now enforces that for any fit that does not
transfer.

## Method notes

- **The budget is uniform again.** The 2026-09 page ran the two routes at
  different completion budgets because the Anthropic adapter could not accept
  more than 21 333 tokens without streaming. It now streams above that ceiling,
  so every row here ran at 32 000, and that page's comparability caveat no
  longer applies. No call stopped at the budget. One reply in each Sonnet
  row's fitting came back unusable (logged as unparseable for
  `claude-sonnet-5`, as apparently truncated JSON for `claude-sonnet-5-5`, with
  the provider reporting a normal finish both times) and was retried once at
  64 000, which is the extra call in each of those fitting counts.
- **Fireworks prompt caching engaged.** `deepseek-v4p1-flash` reported cached
  input tokens (8 214 per run). They are priced here with the same 0.10×
  multiplier as the Anthropic cache reads. If Fireworks bills them at the full
  input rate, that row costs $0.0382 per run rather than $0.0366, and its
  relative figure is 10.4× rather than 10.9×; no conclusion on this page turns
  on the difference.
- **Abort guards.** The run was set to stop on any failed judge call, because a
  failed judge call is silently labelled "not aligned" and would bias a fit
  rather than fail it, and on any extraction API error. Neither fired.
- **The empty extractions are still invisible to the harness.** The 2026-09
  page recorded that a case returning no claims produces no quality note, so a
  run built partly from silence reports a plausible-looking number. It still
  does: `glm-5p3-flash`'s run 2 reports precision 1.000 and ECE 0.000 from no
  output at all, and that run was billed $0.051, the same as each of the two
  runs that did produce claims.

## How to reproduce

One configuration file per model, routing `llm.extraction` at the model under
test and pinning `llm.benchmark` (the calibration judge). Then, per model:

```bash
PARTICLES_CONFIG=survey-model.yaml uv run particles extractor calibrate \
    general-extractor --runs 13 --dry-run --yes
PARTICLES_CONFIG=survey-model.yaml uv run particles extractor benchmark \
    general-extractor --suite prose-article-seed-001 --runs 3 --yes
```

The first prints the pooled temperature without persisting it; the second
writes one JSON report per run under `benchmark.runs_dir`. The benchmark never
applies a stored calibration, so the fitted ECE is computed from those files:
apply `TemperatureScaler(temperature=T).calibrate_batch()` to each run's
`per_case[].emitted_claims[].confidence` values and pass them, with
`outcome == "matched"` as the label, to
`particles.extraction.calibration.expected_calibration_error`. For the
calibration-label variant, count every outcome other than `spurious` as correct.

Cost comes from wrapping each phase in `particles.llm.usage.track_usage()`;
the CLI verbs do not print it. Fireworks API keys are
`PARTICLES_LLM_API_KEY_<NAME>` in the environment, never in the config file,
and the provider entry is the one the 2026-09 page records.

## Appendix: raw per-run values

Each cell is the three benchmark runs in order.

| Model | Recall | Precision | ECE raw | ECE fitted | ECE raw, calibration label | ECE fitted, calibration label |
|---|---|---|---|---|---|---|
| `claude-sonnet-5` | 0.800, 0.860, 0.820 | 0.821, 0.805, 0.746 | 0.061, 0.050, 0.097 | 0.080, 0.094, 0.152 | 0.043, 0.042, 0.048 | 0.066, 0.064, 0.111 |
| `claude-sonnet-5-5` | 0.720, 0.680, 0.720 | 0.787, 0.752, 0.724 | 0.075, 0.103, 0.120 | 0.144, 0.180, 0.188 | 0.046, 0.069, 0.071 | 0.089, 0.120, 0.103 |
| `claude-haiku-4-5` | 0.800, 0.840, 0.760 | 0.860, 0.883, 0.822 | 0.066, 0.037, 0.058 | 0.044, 0.039, 0.081 | 0.048, 0.046, 0.053 | 0.041, 0.064, 0.062 |
| `deepseek-v4p1-flash` | 0.820, 0.840, 0.820 | 0.678, 0.736, 0.667 | 0.140, 0.113, 0.157 | 0.210, 0.142, 0.179 | 0.028, 0.060, 0.072 | 0.103, 0.094, 0.088 |
| `glm-5p3-flash` | 0.320, 0.000, 0.260 | 0.746, 1.000, 0.837 | 0.134, 0.000, 0.051 | 0.086, 0.000, 0.083 | 0.129, 0.000, 0.029 | 0.082, 0.000, 0.129 |

Emitted claims per case, run 1 / run 2 / run 3:

| Model | web-article-001 | web-article-002 | web-essay-001 | web-interview-001 | web-article-003 | web-article-004 |
|---|---|---|---|---|---|---|
| `claude-sonnet-5` | 20 / 19 / 20 | 25 / 26 / 26 | 15 / 16 / 14 | 23 / 24 / 25 | 16 / 16 / 16 | 18 / 17 / 17 |
| `claude-sonnet-5-5` | 15 / 19 / 16 | 22 / 22 / 22 | 15 / 12 / 14 | 16 / 18 / 17 | 13 / 15 / 16 | 13 / 15 / 13 |
| `claude-haiku-4-5` | 19 / 21 / 19 | 25 / 26 / 27 | 10 / 11 / 12 | 23 / 23 / 22 | 16 / 16 / 13 | 14 / 14 / 14 |
| `deepseek-v4p1-flash` | 24 / 22 / 27 | 28 / 31 / 29 | 31 / 29 / 35 | 26 / 30 / 30 | 21 / 23 / 23 | 22 / 24 / 21 |
| `glm-5p3-flash` | 22 / **0** / **0** | **0** / **0** / **0** | **0** / **0** / **0** | **0** / **0** / **0** | 18 / **0** / 24 | 19 / **0** / 19 |

Per-pass fitted temperature over the 13 calibration passes, in order:

| Model | Per-pass T |
|---|---|
| `claude-sonnet-5` | 0.54, 0.90, 0.75, 0.62, 0.72, 0.81, 0.79, 0.85, 0.68, 0.69, 0.95, 0.77, 0.93 |
| `claude-sonnet-5-5` | 0.60, 0.65, 0.60, 0.60, 0.63, 0.72, 0.41, 0.63, 0.64, 0.89, 0.62, 0.64, 0.73 |
| `claude-haiku-4-5` | 0.67, 0.97, 0.89, 0.93, 0.60, 0.96, 0.90, 0.96, 0.91, 0.84, 0.98, 0.77, 0.92 |
| `deepseek-v4p1-flash` | 4.65, 1.71, 1.55, 1.61, 1.98, 1.92, 1.43, 1.68, 1.76, 2.08, 2.35, 1.71, 2.20 |
| `glm-5p3-flash` | 2.53, 1.99, 2.54, 1.52, 1.08, 3.76, 1.40, 1.17, 1.40, 1.47, 0.98, 3.35, 3.39 |

## Cross-references

- The prior snapshot whose open calibration question this page answers:
  [Provider survey (2026-09)](provider-survey-2026-09.md).
- Temperature-scaling calibration, managed per `provider:model`
  pairing, with the guards that refuse a fit that cannot mean
  anything.
- The read-side cap on uncalibrated confidence.
- Immutable stated confidence, which is why calibration is a load-bearing axis.
- The extraction-quality benchmark harness.
