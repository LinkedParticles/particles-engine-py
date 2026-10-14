# The default model: claude-sonnet-5-5 against claude-sonnet-4-6 (2026-10-13)

**A dated snapshot, as of 2026-10-13.** The compiled default model moved from
`claude-sonnet-4-6` to `claude-sonnet-5-5` in 1.189.0, and every purpose that
falls back to `llm.default` moved with it. This page records the measurements
that gated the move. Each gate and its bar were registered before any run,
and the outcome rules were fixed with them: a purpose whose gate
failed would have kept `claude-sonnet-4-6` through a compiled per-purpose
selection. Every gate passed, so nothing stayed behind.

The raw figures, per-run usage included, are in
`default-model-adr0317-2026-10-13.json` beside this page. The labels behind
items 2b and 2c, with the owner's review, are in
`default-model-adr0317-labels-2026-10-13.yaml`.

## Setup

| | |
|---|---|
| Extractor | `general-extractor` 0.17.0 |
| Arms | `claude-sonnet-5-5` and `claude-sonnet-4-6`, every purpose routed to the arm |
| Isolation | a scratch store, blob directory, home, Claude Code state directory and `benchmark.runs_dir` per run; projection off |
| Usage | the SDK's usage tracker at the completion port, priced at list price |
| Spend | US$3.81 in all: US$3.31 for items 1 to 4, US$0.29 for the pre-labels of item 2d, US$0.21 for the conformance capture |

The two extraction arms ran on the same day, on the same code, so their
comparison is paired. Earlier comparisons of these two models on this suite
were not: they paired one `claude-sonnet-4-6` run with three
`claude-sonnet-5-5` runs made eleven days apart, at an older extractor.

## Results

| Item | What it measures | `claude-sonnet-5-5` | `claude-sonnet-4-6` | Bar | Result |
|---|---|---|---|---|---|
| 1 | Input tokens for one run's extraction prompts (6 calls, 90,286 characters) | 33,475 (1.3437×) | 24,913 | none: sets the estimate factor | factor 2.98 |
| 2 | Extraction recall, `prose-article-seed-001`, 3 runs | 0.713 (0.72, 0.74, 0.68) | 0.887 (0.88, 0.90, 0.88) | no lower than 4-6 less 0.05 | **fails**; item 2b decides |
| 2 | Raw calibration error | 0.078 | 0.079 | no higher than 4-6 plus 0.03 | pass |
| 2 | Cases that returned no claim | 0 | 0 | none | pass |
| 2b | Recall counting claims that state the required fact | 0.907 | 0.907 | gap no more than 0.05 | pass (gap 0.0001) |
| 2c | Unsupported claims among 30 unmatched ones | 1 | 0 | 5-5 no more than 4-6 plus 2 | pass |
| 3 | Subject-resolution judge, seed suite (35 names), 2 passes | 35 of 35, both passes | 0.971 (2026-10-02) | at least 0.943 | pass |
| 3 | Subject-resolution judge, ordinary prose (49 names), 2 passes | 46 of 49, both passes | 47 of 49 (2026-10-02) | 4-6 less one name | pass, on the bar |
| 3 | A name `top_hit` links correctly, linked elsewhere | none | | none | pass |
| 4 | Contradiction probe, 30 pairs, 3 runs | 30 of 30 each run | 30 of 30 (2026-10-07) | 30 of 30 | pass |
| 4 | Co-evidential judge, 164 labelled pairs | recall 0.942, 1 false link, 0 cut replies | recall 0.958, 1 false link (2026-10-11) | recall at least 0.908, at most 1 false link | pass |

Item 1's token ratio matches `claude-sonnet-5` (1.3432×), on whose tokenizer
`claude-sonnet-5-5` is. Item 3's ordinary-prose result sits exactly on its
bar: one name fewer than `claude-sonnet-4-6`'s shipped arm, which is what the
bar allows. The 9 answers of 176 the co-evidential judge left unsure were
individual verdicts, not a batch cut at its output budget.

## The recall gap was mostly how the benchmark pairs claims

The extractor benchmark matches each required claim to one emitted claim and
spends that emitted claim. A model that writes fewer, larger claims, one
sentence carrying two required facts, therefore loses recall without missing a
fact. `claude-sonnet-5-5` emitted 93 to 100 claims a run against
`claude-sonnet-4-6`'s 146 to 148. Of the 31 required claims it was scored as
missing, 23 had an emitted claim within the judge's own 0.80 similarity
threshold, already spent on another required claim.

Item 2b measured this directly. For each required claim one arm matched in two
or three runs and the other in at most one, the other arm's three nearest
emitted claims were labelled: stated alone, stated in a compound claim, or not
stated. Counting both kinds of stated as recall closes the gap entirely:

| | Items | Stated alone | Stated in a compound | Not stated | Adjusted recall |
|---|---:|---:|---:|---:|---:|
| `claude-sonnet-5-5` | 31 | 12 | **17** | 2 | 0.907 |
| `claude-sonnet-4-6` | 9 | 2 | 1 | 6 | 0.907 |

**The compound count is the finding to carry forward.** More than half of the
facts `claude-sonnet-5-5` was scored as missing are stated inside a claim that
also states something else. A claim is retired whole: when a later claim
updates one fact in a compound, the claim that also carries the facts that did
not change can retire with it. That is the same shape as a known residual on
`claude-sonnet-4-6`, where a claim that relied on an old value was retired
with it. The move does not create the risk, but on the new default
it has more claims to act on.

### How items 2b and 2c were labelled

`claude-opus-5-5`, neither arm, pre-labelled all 100 items at temperature 0.
For 2b it saw the required fact and the three nearest claims; for 2c, the claim
and the case's full source, with the arms shuffled and unnamed. The owner then
reviewed every pre-label that could move a bar against `claude-sonnet-5-5`
(each 2b "not stated" and each 2c "unsupported", 9 in all) and a seeded random
audit of 10 of the remaining 91. The owner agreed with all 19. Because no
audited label was overturned, the 81 unreviewed pre-labels stand as
registered. This protocol was fixed after the paired runs and before any label
existed (item 2d).

One inconsistency is on record. The required fact "Halcyon Grid Cooperative
members saw their monthly electric bills fall by between $8 and $23 in 2025"
appears once per `claude-sonnet-5-5` run, and each run's nearest claim is the
same sentence without the year. The pre-labeller marked run 2 "not stated" (the
owner agreed: the year is part of the fact) and runs 1 and 3 "stated in a
compound", which were not reviewed. Reading all three the strict way leaves 27
of 31 stated against the 22 the bar needed, so the result is unchanged.

## Cost

Measured on item 2's 18 extraction calls per arm, list price, unbatched:

| | Input per call | Cache read per call | Output per call | US$ per call |
|---|---:|---:|---:|---:|
| `claude-sonnet-5-5` | 2,706 | 2,709 | 4,139 | 0.048 |
| `claude-sonnet-4-6` | 2,046 | 1,988 | 6,203 | 0.100 |

`claude-sonnet-5-5` sends about a third more input tokens for the same text
and thinks adaptively, and still costs about half as much per extraction call:
its price is lower and it writes about two thirds of the output. On short probe
calls, measured on 2026-10-11, the two cost about the same, since there the
extra tokens and the thinking use up the price difference.

## The estimate factors that shipped

| Setting | Entry | Measured from |
|---|---|---|
| `benchmark_memory.chars_per_token_by_model` | `claude-sonnet-5-5: 2.98` | 4.0 ÷ 1.3437 (item 1) |
| `audit.estimate_output_scale_by_model` | `claude-sonnet-5-5: 0.534` | 59,760 ÷ 111,950 output tokens against `claude-sonnet-5` on the same 18 calls (2026-10 provider survey) |
| | `claude-sonnet-4-6: 0.800` | 111,662 ÷ 74,494 against `claude-sonnet-5-5` on item 2's calls, times 0.534 |
| `benchmark_memory.estimate_output_tokens_per_extraction_call_by_model` | `claude-sonnet-5-5: 3578`, `claude-sonnet-4-6: 5363` | 6,700 times each scale |

The `chars_per_token` scalar stays 4.0, so every model without an entry,
local and OpenAI-compatible ones included, keeps the factor it had.

## The conformance baseline

`docs/conformance/baseline/general-extractor.json` was re-captured on
`claude-sonnet-5-5` at 0.17.0 over the same four fixtures: 67 particles against
the earlier capture's 104, and no tier-level verdict change.

## How to reproduce

Each item runs inside a scratch environment that routes every purpose to the
arm and points `benchmark.runs_dir` at a private directory.

```bash
uv run particles extractor benchmark general-extractor --suite prose-article-seed-001 --runs 3 --yes
uv run python scripts/measure_subject_resolution.py tests/benchmark/resolution/ordinary-prose-001.yaml --selection llm_judge --out judge.json
uv run python scripts/bench_local_classifiers.py probe --models --anthropic-model claude-sonnet-5-5
uv run python scripts/bench_local_classifiers.py dedup --nli-models --sts-models --anthropic-models claude-sonnet-5-5
uv run particles extractor conform general-extractor --format json
```

The token count of item 1 sends the exact extraction prompts of one benchmark
run, captured with no model call, to the token-counting endpoint, which bills
nothing.
