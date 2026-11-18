# Rank agreement, continued: the hosted judge and the extractor benchmark (2026-11-08)

**A dated snapshot, measured 2026-11-08 and 2026-11-09.** The
[2026-10-12 measurement](local-rank-agreement-2026-10-12.md) found that
`qwen3:14b` on the owner's Mac mini ranked six LongMemEval read-side variants
the way `claude-sonnet-5` did, with the local model both answering and judging.
That result left two questions open, and this page answers both:

- **A.** Does the agreement survive when the hosted judge scores the local
  answers? A sweep that keeps the hosted judge, for comparability with
  published figures, needs that answer.
- **B.** Does the same hold on the write side? The extractor benchmark varies
  how claims are read out of a document, a different purpose routed to the
  local model, and the earlier page said nothing about it.

It changes no default.

## Verdict

**B: the bar was met, at its edge, and only for recall.** On the extractor
benchmark, `qwen3:14b` ranked six write-side variants by recall the way
`claude-sonnet-5-5` did with Kendall's tau-b **0.600**, exactly the bar of 0.6
(12 of 15 pairs in order, 3 out of order, no ties). The hosted best variant
was the local best, and all 5 pairs the hosted model separated by more than
its own run-to-run spread were ordered the same way; the 3 pairs out of order
were all within 0.025 on the hosted model. Precision did not transfer at all
(tau-b 0.067): the local extractor held 0.96 to 0.99 precision on every
variant, while the hosted one ran from 0.56 to 0.81.

**A: the bar was met.** With `claude-sonnet-5` judging the local answers,
Kendall's tau-b was **0.733** against the bar of 0.6, the hosted best variant
was inside the local top 2, and all 11 pairs the hosted model separated by 0.05
or more were ordered the same way locally. The local judge was not what carried
the earlier agreement: on the same 891 local answers the two judges returned
the same verdict 877 times (98.4 %), with the 14 disagreements split 6 to 8
between the two directions.

## A. Local answers, hosted judge

### What was run

The six local reports of 2026-10-12 hold every local answer per question, so
`particles benchmark memory rejudge` re-scored them with `claude-sonnet-5` as
the judge under judge protocol 2, the judge and protocol of the hosted sweep.
No answer call was made and the retrieval stage is unchanged. The bar is the
one fixed for the earlier measurement (`BAR` in
`scripts/measure_rank_agreement.py`): tau-b ≥ 0.6, and the hosted best variant
inside the local top 2.

### Results

| Variant | Hosted answer, hosted judge | Local answer, local judge | Local answer, hosted judge |
|---|---:|---:|---:|
| `particles-k10` | 0.787 | 0.639 | 0.646 |
| `particles-k20` | 0.813 | **0.667** | **0.660** |
| `particles-k40` | **0.820** | 0.646 | 0.653 |
| `chunks-k10` | 0.733 | 0.540 | 0.533 |
| `chunks-k10-b800` | 0.407 | 0.413 | 0.427 |
| `chunks-k40-b2400` | 0.720 | 0.540 | 0.540 |

The first column is the hosted sweep of 2026-09-18 and 2026-09-20; the second
is the earlier page's local column; the third is new. The local accuracies are
over the 147 particle-arm and 150 chunk-arm questions the local runs scored.

| Reading | Local judge (2026-10-12) | Hosted judge (2026-11-08) |
|---|---:|---:|
| Kendall's tau-b against the hosted sweep | 0.828 | **0.733** |
| Hosted best (`particles-k40`) inside the local top 2 | yes | yes |
| Pairs with a hosted gap ≥ 0.05, ordered the same way | 11 of 11 | 11 of 11 |

The tau-b fell because the hosted judge split a tie the local judge had left
between `chunks-k10` and `chunks-k40-b2400` the wrong way round: the hosted
model separates those two by 0.013, inside its own run-to-run spread. Every
well-separated pair stayed in order.

**What follows for a sweep.** The answering model carries both the lower
absolute scores and the ordering, and the judge adds almost nothing either
way. An exploratory read-side sweep can keep the local judge; one that wants
its figures judged by the hosted model can re-judge the local answers at about
US$0.12 a variant of 150 questions.

### Cost

882 judge calls on `claude-sonnet-5`. `rejudge` prints no usage line, so the
cost was not metered: at the rate measured on the
[answer-dates run](answer-dates-2026-11-03.md) (234 judge calls, 63,000 input
and 6,000 output tokens), it was about **US$0.71**.

## B. The extractor benchmark

### What was run

The extractor benchmark over the three prose suites the local-model
kit uses: `prose-article-seed-001`, `prose-calibration-001` and
`prose-holdout-001`, 20 documents of 1,300 to 2,900 characters with 222
required gold claims, scored by the harness's embedding judge at a cosine of
0.80. The judge is local, so the extractor (`general-extractor` 0.22.0) is the
only model call: `claude-sonnet-5-5` on the hosted arm, the compiled default,
and `qwen3:14b` with thinking off on the local arm.

The six variants are write-side knobs a development loop varies: the optional
components of the extraction prompt, and the chunk size of the chunked read.

| Variant | What it changes | Calls a pass |
|---|---|---:|
| `control` | nothing: the production prompt and a whole read | 20 |
| `bare` | the six optional prompt components off (scope, modality, polarity, stance, validity, structured claim) | 20 |
| `chunk-1000` | a chunked read at 1,000 characters | 43 |
| `chunk-600` | a chunked read at 600 characters | 59 to 60 |
| `chunk-300` | a chunked read at 300 characters | 118 |
| `bare-chunk-600` | `bare` and `chunk-600` together | 59 |

### What was fixed in advance

The sweep, the bar and the scoring are in
`scripts/measure_extraction_rank_agreement.py` (`VARIANTS`, `BAR`,
`HOSTED_RUNS`, `BUDGETS`), pinned by
`tests/test_measure_extraction_rank_agreement.py`, and were committed before
either arm ran.

| What | Value |
|---|---|
| Score per variant | pooled recall: required gold claims matched over required gold claims, summed over the 20 documents, then averaged over the passes |
| Passes | 3 on the hosted arm, which samples; 1 on the local arm, plus a repeat of `control` to see its spread |
| Output budget | the production 32,000 with a retry at 64,000 on the hosted arm; the kit's 8,192 with a retry at 12,288 on the local arm, which fits beside the prompt in the mini's 32,768-token context. No reply on either arm reached its budget |
| Agreement metric | Kendall's tau-b between the six hosted and the six local scores |
| Bar | tau-b ≥ 0.6, and the hosted best variant inside the local top 2: the read-side bar unchanged |
| Reported beside the bar | the pairs whose hosted gap is at least the larger of 0.05 and the widest hosted range of any variant (0.054 here, `control`) must be concordant; tau-b on precision |

**One change was made after the pre-registration, to the local arm's
plumbing only.** The first local pass ran the embedding judge on the mini's
GPU beside the 14B model; Metal ran out of GPU memory, returned no usable
vectors, and the judge matched nothing (49 claims emitted and none matched on
one suite). That pass was discarded unread beyond that check, the judge's
encoder was moved to the CPU for the local arm, and a run whose judge matches
nothing on a suite with ten or more claims now fails instead of writing a
score. The hosted arm's judge ran on the laptop's GPU, so the two arms' encoders
round differently; that can move only a claim whose cosine lies within rounding
distance of 0.80. After review, `score` was made to read only the
pre-registered passes (three hosted, one local) and to compare tau-b with a
tolerance of 1e-9, neither of which changes a figure here. The variants, the
bar and the scoring rule did not change.

### Results

| Variant | Hosted recall (mean, range over 3) | Local recall | Hosted precision | Local precision |
|---|---:|---:|---:|---:|
| `control` | 0.721 (0.689–0.743) | 0.590 | 0.807 | 0.963 |
| `bare` | 0.722 (0.707–0.739) | 0.644 | 0.800 | 0.988 |
| `chunk-1000` | 0.728 (0.721–0.734) | 0.622 | 0.721 | 0.959 |
| `chunk-600` | 0.761 (0.748–0.784) | 0.725 | 0.654 | 0.971 |
| `chunk-300` | 0.703 (0.698–0.707) | 0.626 | 0.564 | 0.965 |
| `bare-chunk-600` | **0.782** (0.770–0.788) | **0.748** | 0.649 | 0.962 |

| Reading | Value |
|---|---:|
| Kendall's tau-b on recall | **0.600** (bar ≥ 0.6) |
| Hosted best (`bare-chunk-600`) inside the local top 2 | yes, first |
| Pairs with a hosted gap ≥ 0.054, ordered the same way | 5 of 5 |
| Kendall's tau-b on precision (reported) | 0.067 |

**The local repeat.** A second local pass of `control` scored recall 0.613
against the first pass's 0.590, so the local extractor is not deterministic at
this configuration, as the read-side answerer was; its spread on this variant
(0.023) is under half the hosted model's (0.054). Averaging the two passes
leaves every rank and the recall tau-b unchanged (0.600) and moves the
precision tau-b to 0.333, still far from any agreement.

Read plainly:

- **The two variants worth taking forward were found locally.** Both arms put
  `bare-chunk-600` first and `chunk-600` second, by margins well clear of
  either model's noise. A local sweep would have sent the right pair to the
  hosted model.
- **The middle of the table is noise on both models.** `control`, `bare`,
  `chunk-1000` and `chunk-300` lie within 0.025 of one another on the hosted
  arm, inside the 0.054 spread of `control`'s own three passes. The three
  pairs out of order are all in that cluster. A bar of tau-b over all pairs
  counts them anyway, which is why this sweep met it only at the edge.
- **Precision has to be measured on the hosted model.** Chunking cut hosted
  precision from 0.81 to 0.56 and left the local model's between 0.96 and
  0.99. The local extractor states fewer claims and keeps to the safest ones,
  so it cannot show a precision cost. Any write-side decision that turns on
  precision, which includes every chunk-size choice on this page, stays a
  hosted measurement.
- **The local model is lower on recall by 0.03 to 0.13** and nearer the hosted
  model on the variants that read in smaller pieces. As on the read side, a
  local score is not an estimate of a hosted one.
- **Two findings about the variants themselves,** outside the question this
  page asks: the optional prompt components cost about two and a half times
  the output (US$0.31 a pass for `bare` against US$0.77 for `control`) with
  no recall difference on these documents, and 600-character chunks raised
  recall by about 0.04 while lowering precision by about 0.15. Neither is a
  recommendation; both are leads for a hosted measurement of their own.

### Cost and time

| Arm | Calls | Output tokens | Wall-clock | Cost |
|---|---:|---:|---:|---:|
| Hosted, 3 passes of 6 variants, on the laptop | 958 | 1,395,243 | 2.0 h | **US$15.26** at list price, metered |
| Local, 1 pass of 6 variants, on the Mac mini | 319 | 252,137 | 7.4 h | US$0 |

The local pass ran at about 9.5 output tokens a second, so its time follows the
output: `bare` took 26 minutes and `chunk-600` 108. The model ran 93 % on the
GPU at the server's 32,768-token context (Ollama 0.40.3, `qwen3:14b` digest
`bdbd181c33f2`, Q4_K_M), every request under the benchmark run lock, with the
load average between 0.9 and 2.1 at each variant's start.

**What the workflow saves on a sweep like this one.** The whole sweep on the
hosted model cost US$15.26. Exploring locally and confirming the local top three
(`bare-chunk-600`, `chunk-600`, `bare`) on the hosted model at three passes
each would have cost US$5.69 here, and the top two US$4.76, against 7.4 hours
of the mini: a saving of 63 to 69 %. The larger gain is that the exploratory
runs need no spending approval.


## Limits

- **One sweep per benchmark.** Each verdict is for the variants measured here.
  A sweep over different knobs is a new measurement.
- **Close pairs are not resolved,** on either benchmark. A pair the hosted
  model separates by less than its own run-to-run spread is ordered by
  sampling on both models, and a local sweep says nothing about it.
- **A local accuracy is not an estimate of a hosted one.** On both benchmarks
  the local scores are compressed and shifted; only the order is compared.

## How to reproduce

```bash
# A: re-judge each committed local report with the hosted judge (about US$0.12 each)
uv run particles benchmark memory rejudge \
    docs/benchmarks/longmemeval-s150-qwen3-14b-particles-k10-2026-10-12.json \
    --output particles-k10.json --yes   # with llm.benchmark on claude-sonnet-5, judge_protocol 2
uv run python scripts/measure_rank_agreement.py score --local-dir <the six re-judged reports>
```

```bash
# B: the hosted arm (about US$5 a pass), then the local arm on the benchmark host
uv run python scripts/measure_extraction_rank_agreement.py run --arm hosted \
    --variant control --run 1 --out-dir <hosted-dir>      # each variant, runs 1 to 3
python3 scripts/bench_lock.py --label "<task>" -- \
    uv run python scripts/measure_extraction_rank_agreement.py run --arm local \
    --base-url http://127.0.0.1:11434/v1 --variant control --run 1 --out-dir <local-dir>
uv run python scripts/measure_extraction_rank_agreement.py score \
    --hosted-dir <hosted-dir> --local-dir <local-dir>
```

## Cross-references

- The proposal this measures: explore on a local model, confirm the finalists
  on the hosted one.
- The first measurement: [rank agreement, 2026-10-12](local-rank-agreement-2026-10-12.md).
- Every run of B, both arms, with its counts, usage and quality notes:
  [`extractor-rank-agreement-2026-11-08.json`](extractor-rank-agreement-2026-11-08.json).
- The re-judged reports:
  [`particles-k10`](longmemeval-s150-qwen3-14b-particles-k10-hosted-judge-2026-11-08.json),
  [`particles-k20`](longmemeval-s150-qwen3-14b-particles-k20-hosted-judge-2026-11-08.json),
  [`particles-k40`](longmemeval-s150-qwen3-14b-particles-k40-hosted-judge-2026-11-08.json),
  [`chunks-k10`](longmemeval-s150-qwen3-14b-chunks-k10-hosted-judge-2026-11-08.json),
  [`chunks-k10-b800`](longmemeval-s150-qwen3-14b-chunks-k10-b800-hosted-judge-2026-11-08.json),
  [`chunks-k40-b2400`](longmemeval-s150-qwen3-14b-chunks-k40-b2400-hosted-judge-2026-11-08.json).
