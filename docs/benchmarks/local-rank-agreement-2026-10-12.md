# Rank agreement between a local and a hosted model (2026-10-12)

**A dated snapshot, measured 2026-10-12.** Benchmark sweeps feed relative
decisions: variant A against variant B. A local model can serve such a decision
when it orders the variants the way the hosted model does, even where its
absolute scores are lower; the few configurations that will be reported are
then rerun on the hosted model. That precondition is a property of
each benchmark and is measured, never assumed. This page measures it for one
benchmark, LongMemEval, on one sweep.

It changes no default.

## Verdict

**The bar was met.** `qwen3:14b` ranked the six variants of this sweep the way
`claude-sonnet-5` did, closely enough for the decision the sweep was run to
make.

- **Kendall's tau-b was 0.828**, against a bar of 0.6. It is the same under both
  scorings this page reports (see [Results](#results)).
- **The hosted best variant, `particles-k40`, ranked second locally**, inside
  the required top 2. The local model put `particles-k20` first, 0.021 ahead;
  the hosted model put it second, 0.007 behind.
- **Every pair the hosted model separated by 0.05 or more was ordered the same
  way locally**, 11 of 11. The two disagreements are a swap and a tie among
  pairs the hosted model itself separated by less than its own run-to-run
  spread.
- **The local model was much worse in absolute terms, and that did not matter
  for the ranking.** Its accuracy ran 0.15 to 0.19 below the hosted model's on
  every variant except the weakest, where both scored about 0.41. The local
  model compresses the scale from the top and keeps the order.

On this benchmark, for read-side knobs, the precondition of the proposal holds:
an exploratory sweep could run on the local model at no cost, and the hosted
model would then be needed only to confirm the two or three configurations
worth reporting. The result is for one sweep, and the [limits](#limits) say
what it does not show.

## The sweep

The LongMemEval `s` read-side grid run on `claude-sonnet-5` on 2026-09-18 and
2026-09-20: 150 questions, seed 13, answer scaffold 2, judge protocol 2,
`claude-sonnet-5` as answerer and judge. The particle memory at three retrieval
depths, and the raw-transcript chunk comparator at three context sizes.

| Variant | Memory | Top-k | Context budget | Hosted accuracy | Hosted report |
|---|---|---:|---:|---:|---|
| `particles-k10` | particles | 10 | | 0.787 | [report](longmemeval-s150-v2-p2-topk10-2026-09-18.json) |
| `particles-k20` | particles | 20 | | 0.813 | [report](longmemeval-s150-v2-p2-topk20-2026-09-18.json) |
| `particles-k40` | particles | 40 | | **0.820** | [report](longmemeval-s150-v2-p2-topk40-2026-09-18.json) |
| `chunks-k10` | chunks | 10 | | 0.733 | [report](longmemeval-s150-chunks-v2-p2-2026-09-20.json) |
| `chunks-k10-b800` | chunks | 10 | 800 | 0.407 | [report](longmemeval-s150-chunks-v2-p2-budget800-2026-09-20.json) |
| `chunks-k40-b2400` | chunks | 40 | 2 400 | 0.720 | [report](longmemeval-s150-chunks-v2-p2-topk40-budget2400-2026-09-20.json) |

It was chosen for three reasons.

- **The hosted side costs nothing.** The particle variants reuse the persisted
  scratch stores, and the chunk comparator needs no store, so no variant makes
  an extraction call. The only calls a variant makes are the answer and the
  judge. The committed reports carry every hosted verdict per question, so the
  hosted side is re-scored from them at no cost. No sweep of four or more
  variants was recorded in the completion replay cache, which was
  added after these sweeps ran; the replay cache holds only the two-arm
  escalation measurement.
- **Its variants are spread.** Hosted accuracy runs from 0.407 to 0.820. In the
  other recent LongMemEval ablations every variant lands within about 0.03 of
  every other, which is inside the run-to-run spread of repeated hosted runs,
  so a rank test on them would measure sampling.
- **It is the kind of sweep the proposal would move to a local model.** Read-side
  knobs (retrieval depth, context budget, the memory under test) are the ones a
  development loop varies most often.

The notes comparator is left out because its session notes are written by the
extraction model and were not kept, so a local notes arm would need several
thousand new extraction calls. The full-context and no-memory baselines are left
out: the first does not fit the local context window, and the second would rank
last on both models and add easy pairs.

## What was fixed in advance

The bar, the sweep and the scoring are in `scripts/measure_rank_agreement.py`
(`BAR`, `VARIANTS`) and pinned by `tests/test_measure_rank_agreement.py`. Both
were committed before any local answer was produced.

| What | Value |
|---|---|
| Score per variant | accuracy as the harness reports it (`qa_particles.accuracy`), on each model |
| Agreement metric | Kendall's tau-b between the six hosted and the six local accuracies |
| Bar, part 1 | tau-b ≥ 0.6, which allows at most 3 discordant pairs of 15 when nothing ties |
| Bar, part 2 | the hosted best variant (`particles-k40`) inside the local top 2, ties at the cutoff counted inside |
| Verdict | the bar is met only when both parts hold |
| Reported, not the bar | concordance on the pairs whose hosted gap is at least 0.05; about one and a half times the widest spread three hosted repeats of one arm have shown on this benchmark (0.034), so a narrower pair may be ordered by sampling alone |
| Reported, not the bar | an inputs check: per question, the context particle ids (particle arms) or the retrieval recall (chunk arms) must match the hosted report on at least 95 % of questions, or the page says the code or store moved between the runs |

Four of the fifteen pairs are inside the 0.05 band (`particles-k10`, `-k20`
and `-k40` among themselves, and `chunks-k10` against `chunks-k40-b2400`). The
hosted best variant leads the second by 0.007, so part 2 of the bar can fail on
sampling alone. It is kept because the decision this proposal serves is which
configuration to confirm on the hosted model, and a local ranking that puts the
hosted best third would send the wrong finalists.

## The local route

The local arm runs the same code the hosted sweep ran, commit `bcd05827`
(1.146.5), in its own checkout. The persisted stores were built before a later
schema migration, and current code fails on every question against them, so
running the hosted sweep's own revision both reads the stores and keeps the
answer prompt, judge prompt and retrieval identical between the two models.
This section was settled after the bar was committed and before any local
answer was scored; the bar, the sweep and the scoring did not change.

| Parameter | Value |
|---|---|
| Code | `bcd05827` (1.146.5), the revision that committed the hosted reports |
| Model | `qwen3:14b` (Q4_K_M), digest `bdbd181c33f2`, as the tag `qwen3:14b-16k-nothink` |
| Context | `num_ctx` 16 384. The server's default is 4 096, which would truncate the chunk arms' contexts |
| Thinking | off. 1.146.5 has no `reasoning_effort` setting, so the tag's chat template always takes the model's own thinking-off branch (the `/no_think` turn suffix and an empty think block), which is the prompt Ollama builds for `reasoning_effort: none`. A tag of its own is against the host's current one-tag-per-model rule, which asks for thinking to be turned off per request; 1.146.5 cannot send that request, so the tag was the only way, and it was removed after the run |
| Server | Ollama 0.40.3 on the owner's Mac mini (Apple M4, 16 GB), `127.0.0.1:11434`, flash attention, `q8_0` KV cache |
| Route | the compiled-in `local` OpenAI-compatible entry, 900 s request timeout |
| Purposes on the local route | `benchmark_answer` (the answerer) and `benchmark` (the judge) |
| Subject rendering | particle ids, as every report of this period rendered them (1.146.5 predates the setting) |
| Isolation | the run waited for one other local-model job on the same server to finish, but a second session sent requests for a different model tag until about 18:10 PDT, and the server swapped models several times between 18:02 and 18:09. `particles-k10`, the first variant, started at 17:52, so its wall-clock time is contaminated and is reported from a separate re-timed run (see [Results](#results)). The host's run lock was introduced while the sweep was running, so the sweep did not hold it; the re-time held it from its fifth minute to its end |
| Cost | US$0. `ANTHROPIC_API_KEY` held a placeholder that is not a key, so a hosted call would have been refused rather than billed; none was made |

## Results

Both models answered and judged with the same code, the same prompts and the
same retrieved context. The inputs check found no question whose context
differed from the hosted run's, in any variant.

### As the harness reports it (the pre-registered score)

| Variant | Hosted accuracy | Hosted rank | Local accuracy | Local rank | Local questions scored |
|---|---:|---:|---:|---:|---:|
| `particles-k10` | 0.787 | 3 | 0.639 | 3 | 147 |
| `particles-k20` | 0.813 | 2 | **0.667** | 1 | 147 |
| `particles-k40` | **0.820** | 1 | 0.646 | 2 | 147 |
| `chunks-k10` | 0.733 | 4 | 0.540 | 4 (tied) | 150 |
| `chunks-k10-b800` | 0.407 | 6 | 0.413 | 6 | 150 |
| `chunks-k40-b2400` | 0.720 | 5 | 0.540 | 4 (tied) | 150 |

Kendall's tau-b 0.828; hosted best inside the local top 2; 11 of 11 separated
pairs concordant. **Bar met.**

### On the 147 questions every report scored

The three particle variants scored 147 questions locally, not 150. For three
questions (`001be529`, `1d4da289`, `gpt4_76048e76`) the persisted store no
longer matched the session it was built from: all three store files were
written on 2026-09-20 between 18:57 and 19:03 PDT, after the hosted particle
runs, by a run this page has not identified. Against such a store the harness
re-extracts, and with no usable key the extraction was refused, so the harness
skipped the question. The chunk variants use no store and scored all 150.

Mixed denominators could bias a ranking, so this reading scores every report,
hosted and local, on the 147 questions all twelve scored. The hosted side is
re-scored from its own per-question verdicts. This scoring was added after the
defect was found, which was after the first local variant's accuracy (0.639)
had been seen and before any other local variant had finished. It does not
replace the pre-registered score; it is reported beside it.

| Variant | Hosted accuracy | Local accuracy |
|---|---:|---:|
| `particles-k10` | 0.782 | 0.639 |
| `particles-k20` | 0.810 | **0.667** |
| `particles-k40` | **0.816** | 0.646 |
| `chunks-k10` | 0.728 | 0.531 |
| `chunks-k10-b800` | 0.401 | 0.401 |
| `chunks-k40-b2400` | 0.714 | 0.531 |

Kendall's tau-b 0.828; hosted best inside the local top 2; 11 of 11 separated
pairs concordant. **Bar met.** The two scorings agree on every rank.

### Wall-clock time

The local variants ran one question at a time, each question making one answer
call and one judge call to `qwen3:14b` on the Mac mini's GPU (all 41 layers
offloaded, the GPU at 99 to 100 % utilisation through the run).

| Variant | Local wall-clock | Per question |
|---|---:|---:|
| `particles-k10` | 2 182 s (36 min) | 15 s |
| `particles-k20` | 3 460 s (58 min) | 23 s |
| `particles-k40` | 5 789 s (96 min) | 39 s |
| `chunks-k10` | 5 370 s (90 min) | 36 s |
| `chunks-k10-b800` | 2 009 s (33 min) | 13 s |
| `chunks-k40-b2400` | 4 731 s (79 min) | 32 s |

The first run of `particles-k10` took 2 498 s, but it shared the server with
another session from its start at 17:52 until about 18:10 PDT, and one of its
questions waited 5.5 minutes through a model swap, so that figure is not
reported as its time. The figure above is from a second run made after the
sweep, alone on the server: the Ollama log records no model load during it, so
no other request reached the server. The second run's accuracy was again
0.639, and its verdict matched the first run's on all 147 questions, so the
local answers are deterministic at this configuration and the contamination
reached the timing only.

The hosted particle variants took 404, 398 and 414 seconds on 2026-09-18,
running four questions at a time against the Anthropic API; the hosted chunk
runs left no log. The two are not a like-for-like comparison of the models'
speed. What the local figures say is how long an exploratory sweep takes on
this host: about 6.5 hours for six variants of 150 questions.

## Limits

- **One sweep, one benchmark, read-side knobs only.** The variants change what
  the answering model is shown. A sweep that changes extraction would route a
  different purpose to the local model, and nothing here says the local
  extractor ranks write-side variants as the hosted one does.
- **The local model answered and judged.** Both purposes ran on `qwen3:14b`, as
  an exploratory sweep would run them. The result therefore cannot say whether
  the local answerer or the local judge carries the agreement; a run with the
  hosted judge held fixed would separate them, at a cost.
- **Close variants are not resolved.** The local model swapped the two best
  particle variants and tied two chunk variants. Those pairs are inside the
  hosted model's own run-to-run spread, so neither model separates them, and a
  local sweep can be trusted to rank variants only where the hosted gap would
  be larger than about 0.05.
- **One run per variant on each model.** No local repeat was made, so the local
  model's run-to-run spread is not known. The `particles-k10` re-time is a
  second sample of one variant only.
- **The scale is compressed.** A local accuracy is not an estimate of a hosted
  one. The local model was 0.15 to 0.19 lower on the strong variants and equal
  on the weakest, so differences between strong variants shrink locally.

## How to reproduce

```bash
# the hosted side, from the committed reports (no model call)
uv run python scripts/measure_rank_agreement.py hosted
# the six local commands (run them in a checkout of bcd05827), for a given
# store set and output directory
uv run python scripts/measure_rank_agreement.py commands \
    --store-dir ~/.particles/benchmark/stores/s150 --out-dir ~/bench/pdr-1006
# score the local reports against the hosted ones and apply the bar
uv run python scripts/measure_rank_agreement.py score --local-dir ~/bench/pdr-1006
# the committed local reports carry a longer name; copy them to <variant>.json
# in a directory of their own to re-score them without rerunning
```

## Cross-references

- The proposal this measures: explore on a local model, confirm the finalists on the hosted one.
- The replay cache that removes repeated calls.
- The sweep's hosted runs: [agent-memory benchmark](../benchmarks.md).
- The local reports: [`particles-k10`](longmemeval-s150-qwen3-14b-particles-k10-2026-10-12.json),
  [`particles-k20`](longmemeval-s150-qwen3-14b-particles-k20-2026-10-12.json),
  [`particles-k40`](longmemeval-s150-qwen3-14b-particles-k40-2026-10-12.json),
  [`chunks-k10`](longmemeval-s150-qwen3-14b-chunks-k10-2026-10-12.json),
  [`chunks-k10-b800`](longmemeval-s150-qwen3-14b-chunks-k10-b800-2026-10-12.json),
  [`chunks-k40-b2400`](longmemeval-s150-qwen3-14b-chunks-k40-b2400-2026-10-12.json).
