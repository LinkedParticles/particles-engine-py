# Chunk-level escalation on a held-out suite (2026-10-11)

**A dated snapshot, measured 2026-10-11.** The
[log-probability calibrator page](logprob-calibrator-2026-10-07.md) ended on one
live idea: run extraction on a local model, and send a whole chunk to the hosted
extractor only when the local samples of that chunk disagree. On flat prose that
arrangement came within the hosted route's run-to-run spread for about 56 % of
its hosted calls. The cut was picked from three tried on the same 18 chunks it
was scored on, so the result was a hypothesis. This page tests it
with the rule fixed in advance, on a suite that played no part in choosing it.

It changes no default and builds no cascade.

## Verdict

**The pre-registered rule does not pay.** The gate was fixed before the run:
against the hosted route, pooled over the held-out suite, the cascade had to
keep precision within 0.03, recall within 0.05 and calibration error within
0.03, while sending at most 70 % of chunks to the hosted model. It met three of
the four. It missed recall by 0.003: 0.696 against the hosted route's 0.749.

- **The rule barely fired.** It escalated 3 of 30 chunks, all of them the same
  case. On these sources the three `qwen3:14b` samples of a chunk mostly agreed
  with one another (in 13 of 30 chunks every claim was produced by every other
  sample), and the new low-yield trigger never fired once.
- **Escalation added almost nothing over the local model alone.** Local alone
  reached recall 0.687; the cascade reached 0.696. The 6-point recall gap to the
  hosted route is spread across chunks whose samples agree, which is exactly
  where a disagreement signal cannot see it.
- **Local alone was much closer to the hosted route than on the development
  suites.** Recall 0.687 against 0.749 (0.471 against 0.746 on the development
  suites), with no dropouts in 30 local calls.
- **Precision and calibration error favour the local model here, and that is
  partly an artefact of the suite.** See [reading precision on this
  suite](#reading-precision-on-this-suite). Recall is the comparison to trust.

Read together: a disagreement-triggered cascade is not justified on this
evidence. The signal that looked useful on the development suites tracked
dropouts and unstable samples, and on a run with neither it had nothing to say
about the claims the local model missed.

## What was fixed in advance

Commit `4e8706f0`, pushed before any model read the held-out suite, holds three
things:

| What | Value |
|---|---|
| Escalation rule | escalate a chunk when its local run has fewer than 5 claims (a failed sample), when more than 0.33 of its claims are not produced by every other healthy sample of the chunk, or when it yields fewer than 1.0 claims per source sentence |
| Gate | against the hosted route, mean over runs, pooled: precision ≥ hosted − 0.03, recall ≥ hosted − 0.05, ECE (calibration label) ≤ hosted + 0.03, hosted calls ≤ 70 % |
| Held-out suite | `prose-holdout-001` v0.1.0, ten new cases, gold frozen before any run |

The disagreement cut is the 2026-10-08 hypothesis, unchanged. The low-yield
trigger is the one new parameter, read off the development rows only: from 0.8
claims per sentence up it brought the development article suite's recall to the
hosted route's, and 1.0 was the low edge of a plateau on both development
suites. `scripts/measure_escalation_holdout.py dev` reprints that derivation
from the committed development rows. The rule's constants and the gate are
pinned by `tests/test_measure_escalation_holdout.py`.

## The held-out suite

`tests/benchmark/escalation/prose-holdout-001.yaml`: ten self-authored
synthetic pages, five in each register the earlier measurement found the rule
behaves differently on.

- **Flat prose**: an engineering post, a civic news item, an essay, an interview
  and a results note.
- **Hedged prose**: a preliminary incident review, a field-trial note, a
  disputed archive attribution, a budget forecast with ranges, and a news item
  with a printed correction and two accounts that disagree.

Gold coverage is total, as in `prose-calibration-001`: every claim a page
supports is named, 157 expectations of which 113 are required. Every
`confidence_min` is 0.0, so the harness label and the calibration label agree.
No expectation was added or reworded after a run, including the paraphrase
twins the earlier suites gained from observed emissions. The suite sits in its
own directory, so neither `particles extractor benchmark` nor `particles
extractor calibrate` reads it by default and nothing will be fitted on it by
accident.

## What was run

| Parameter | Value |
|---|---|
| Local route | `qwen3:14b` (Q4_K_M), Ollama 0.31.1, 12 288-token context, `reasoning_effort: none`, `extraction.max_tokens` 6 144, on a 24 GB M5 MacBook Air |
| Hosted route | `anthropic:claude-sonnet-5`, extraction defaults |
| Extractor | `general-extractor` 0.16.0 |
| Runs | 3 of each route, each case its own one-case benchmark |
| Judge | embedding cosine at 0.80, subject-aware, as the harness matches |
| Cost | local US$0; hosted **US$1.56** at list price (30 calls, 47 990 input and 143 621 output tokens, no budget stop) against an estimate of US$2.00 and an agreed ceiling of US$2.50 |
| Local time | about 2 h 15 min for 29 calls (one case was replayed from the smoke test), 4.5 minutes per call on average with the first calls nearer 10 |

Each run recorded into its own replay directory, so every reply here
can be replayed at no cost. The runs need separate directories because the
replay key is the request, which is identical across runs of the same case.

## Results

Mean over the three runs (range). A chunk is one case, one extraction call.

**All held-out cases:**

| Arm | Hosted calls | Precision | Recall | ECE, calibration label |
|---|---:|---|---|---|
| `qwen3:14b` alone | 0 % | 0.868 (0.848–0.879) | 0.687 (0.673–0.699) | 0.058 (0.055–0.060) |
| `claude-sonnet-5` alone | 100 % | 0.734 (0.728–0.745) | **0.749** (0.743–0.752) | 0.121 (0.108–0.128) |
| **pre-registered rule** | 10 % | 0.855 (0.847–0.870) | 0.696 (0.681–0.708) | 0.067 (0.064–0.069) |
| rule, kept local claims at 0.73 | 10 % | 0.855 (0.847–0.870) | 0.696 (0.681–0.708) | 0.138 (0.121–0.150) |

**Flat prose:**

| Arm | Hosted calls | Precision | Recall | ECE, calibration label |
|---|---:|---|---|---|
| `qwen3:14b` alone | 0 % | 0.928 (0.907–0.945) | 0.747 (0.741–0.759) | 0.045 (0.036–0.060) |
| `claude-sonnet-5` alone | 100 % | 0.737 (0.709–0.770) | 0.765 (0.759–0.778) | 0.161 (0.131–0.188) |
| pre-registered rule | 0 % | 0.928 (0.907–0.945) | 0.747 (0.741–0.759) | 0.045 (0.036–0.060) |

**Hedged prose:**

| Arm | Hosted calls | Precision | Recall | ECE, calibration label |
|---|---:|---|---|---|
| `qwen3:14b` alone | 0 % | 0.804 (0.784–0.820) | 0.633 (0.610–0.644) | 0.100 (0.082–0.124) |
| `claude-sonnet-5` alone | 100 % | 0.730 (0.716–0.750) | 0.734 (0.729–0.746) | 0.101 (0.097–0.108) |
| pre-registered rule | 20 % | 0.780 (0.759–0.792) | 0.650 (0.627–0.661) | 0.133 (0.118–0.144) |
| rule, kept local claims at 0.73 | 20 % | 0.780 (0.759–0.792) | 0.650 (0.627–0.661) | 0.077 (0.062–0.097) |

On flat prose the rule escalated nothing. On hedged prose it escalated the
correction case in all three runs (between 0.44 and 0.45 of its claims were
disagreed), which bought 1.7 points of recall. The 2026-10-08 rule without
the yield trigger made the same decisions on every chunk, and the yield trigger
alone made none.

The secondary arm stores every kept local claim at 0.73, the share of kept
local claims that were right on the development rows, since the local model's
own stated confidence ranks its claims at chance there. On this suite the local
model was right on far more than 73 % of its flat-prose claims, so the fixed
value is too low on flat prose and about right on hedged prose. A cascade would
still have to say what confidence a kept claim receives, and a single
development-fitted constant does not transfer.

## Reading precision on this suite

The hosted route's precision (0.734) is lower here than on the development
suites (0.777), and the local model's is higher (0.868 against 0.740). Most of
that difference comes from how the suite is labelled, not from which claims are
right.

Of the hosted route's 114 claims labelled spurious, 96 lie between cosine 0.65
and 0.80 of some gold claim, the band where a correct claim reworded past the
judge's floor lands, and 14 more clear 0.80 against a gold claim another
emission had already taken (a finer split of one sentence). Only 4 are far from
every gold claim. For the local model the same counts are 29, 1 and 12 of 42.
The development suites carry paraphrase twins added after observed runs, which
absorb exactly these near-misses. This suite has none by design, because adding
them after looking at a held-out run would tune the labels to the outputs.
`claude-sonnet-5` states more claims (14.3 per case against 10.6) in more
varied words, so the missing twins cost it more.

The same mechanism reaches calibration error, since a correct claim labelled
spurious is also a miscalibrated one. Recall is less exposed: a required gold
claim is matched by whichever emission comes closest, and a reworded claim has
several emissions to come close with. The gate's recall bound is the one that
failed, and it is the least confounded of the four.

## Limits

- **One suite, ten cases, three runs.** The recall miss is 0.003, against a
  run-to-run range of 0.681 to 0.708 for the cascade. A second held-out suite could
  land either side of the bound. The verdict stands as pre-registered; the
  margin says the arrangement is not far off, not that it works.
- **The yield trigger's denominator is crude.** `source_sentences` reads the
  page as text, markup included, and splits only after a full stop, question
  mark or exclamation mark followed by whitespace, so a sentence that ends just
  before a closing tag is not counted. It undercounts the development and
  held-out pages alike, to between about a third and three quarters of their
  sentences (half to two-thirds for most pages), so the rule
  was applied consistently. It is still not a measure of claim yield anyone
  should build on.
- **The held-out pages are shorter and cleaner than the development ones.**
  They are 1 330 to 1 727 bytes against 1 500 to 2 800, and the local model
  returned no empty or near-empty reply on any of them. The development result
  leaned heavily on dropouts. A pool of real deposits would test the rule where
  dropouts happen.

## What this means for the proposal

- **Decline the disagreement-triggered cascade.** The pre-registered
  rule missed its gate. More to the point, on a run without dropouts the signal
  barely fired, and the claims the local model missed sat in chunks whose
  samples agreed.
- **The live observation is the local model itself.** `qwen3:14b` alone came
  within 6 points of the hosted route's recall on this suite. If local
  extraction is revisited, the question is no longer which chunks to escalate.
  It is whether the remaining recall gap matters for a given store, measured
  on that store's own deposits, with the confidence a local claim is stored at
  settled first.

## How to reproduce

```bash
# the derivation of the rule, from the committed development rows (no model call)
uv run python scripts/measure_escalation_holdout.py dev
# local samples (no hosted call; ANTHROPIC_API_KEY is removed from the process)
PARTICLES_CONFIG=local.yaml uv run python scripts/measure_escalation_holdout.py \
    run --runs 3 --out local.jsonl
uv run python scripts/measure_escalation_holdout.py estimate --local local.jsonl
# the hosted pass (bills; stops before a case could pass the ceiling)
PARTICLES_CONFIG=sonnet.yaml uv run python scripts/measure_escalation_holdout.py \
    run --hosted --max-usd 2.50 --runs 3 --out hosted.jsonl
uv run python scripts/measure_escalation_holdout.py score \
    --local local.jsonl --hosted hosted.jsonl
```

Replies are recorded under `~/.particles/llm-replay/pdr-0965/<route>/run-<n>`,
so a rerun on the same machine replays every call. The claim rows behind every
table on this page are committed beside it, for
[`qwen3:14b`](escalation-holdout-qwen3-14b-2026-10-11.jsonl) and
[`claude-sonnet-5`](escalation-holdout-claude-sonnet-5-2026-10-11.jsonl), so
`score` reproduces them without rerunning either model.

## Cross-references

- The measurement this one tests:
  [Log-probability calibrator (2026-10-07)](logprob-calibrator-2026-10-07.md),
  § Escalation to the hosted extractor.
- The replay cache that makes reruns free, immutable stated
  confidence, and the local adapter (generalized).
