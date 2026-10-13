# Local NLI classifiers for the yes/no LLM purposes (2026-10-07)

**A dated snapshot, as of 2026-10-07.** Several LLM purposes ask a generative
model a yes/no question: do these two claims contradict, does this set of
claims support this sentence. A natural-language-inference (NLI)
cross-encoder answers exactly that question, runs on a CPU, and costs nothing
per call. This page measures whether one can stand in for the model on two of
those purposes before any default changes (phase 1). Neither
purpose cleared its bar, so no default changed and no classifier seam is
proposed.

## What was measured

Three DeBERTa-v3 NLI cross-encoders, from the Hugging Face hub:

| Model | Parameters | Trained on |
|---|---|---|
| `cross-encoder/nli-deberta-v3-small` | 142M | SNLI + MultiNLI |
| `cross-encoder/nli-deberta-v3-base` | 184M | SNLI + MultiNLI |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 435M | MultiNLI, FEVER-NLI, ANLI, LingNLI, WANLI |

Each was scored against two purposes:

1. **The `semantic_lint` contradiction probe**, on a 30-pair known-answer set.
2. **The leakage entailment judge**, on the sentences a real leakage run
   judged.

Every run used the committed script `scripts/bench_local_classifiers.py` on
an Apple M5 (10 cores, 24 GB). Bulk scoring ran on the GPU in half
precision. Half precision reproduced the full-precision probe results
exactly. Latency is reported on the CPU, one pair at a time, because that is
the call shape a lint probe would have in production. The only hosted calls
were three runs each of `claude-sonnet-4-6`, `claude-haiku-4-5` and
`claude-haiku-5-5` on the 30 probe pairs, under a dollar in all.

**Rerun on 2026-10-08.** The first runs overlapped heavy use of the same
machine by other local-model jobs (a load average near 190), which slowed
every timing and made the local LLM time out. The probe was run again on an
idle machine (load average below 3, no timeouts), and the local LLM three
times. The NLI verdicts were identical, so only their timings changed, and
the figures below are the idle-machine ones. The entailment run's store
snapshot had been removed by then, so it was repeated on a read-only copy of
the live store against the earlier, first-protocol leakage run's verdicts. That
gives idle-machine timings and a second label set, described below.

## The contradiction probe

### The pair set

The 30 pairs behind the earlier local-model measurement were never
committed and could not be recovered. A replacement set was written to the
same specification and is now versioned:
[`contradiction-probe-pairs-2026-10-07.json`](contradiction-probe-pairs-2026-10-07.json).
It holds 10 clear contradictions, 10 clear compatibles (restatement, narrower
claim, unrelated detail), and 10 subtle pairs. Six of the subtle pairs are
traps that do not contradict: a temporal qualification ("in 2019 the CEO was
X" against "the current CEO is Y") or two subjects that share a name. Four
are real contradictions that need a second step: the same date with a
different number, a scope exception, an exclusive "only", and an overlapping
employment period.

To check that the new set is about as hard as the old one, it was also run
through the production probe path (`_llm_check_contradiction`, with its
fenced prompt, response schema and parser) on the local `qwen2.5:14b-instruct`.
That model scored 29 of 30 on the original set in one run. Here it scores 26
to 28 of 30 across three runs, the variation discussed below. The hosted
`claude-sonnet-4-6`, which scored 30 of 30 on the original set, scores 30 of
30 here in each of three runs, so the new set is no harder than the old one
for the model the bar was set against.

The probe's configured default is `claude-haiku-4-5`, so both Haiku
generations were measured too. Each scored 30 of 30 in every run. One run of
the 30 pairs cost $0.019 on `claude-haiku-4-5` (13,100 input and 1,150
output tokens) and $0.055 on `claude-sonnet-4-6`. `claude-haiku-5-5` used
about 16,900 input and 2,750 output tokens per run, about twice the output of
`claude-haiku-4-5` for the same verdicts. At its list price ($0.10 input,
$0.50 output per million tokens) that is $0.003 a run, a sixth of
`claude-haiku-4-5`'s cost. Its tokenizer counts about 30% more tokens for the
same text, and the longer replies are the rest of the difference; neither
outweighs a tenfold lower price.

### Results

An NLI model reads each pair in both directions, and the pair counts as a
contradiction when either direction's most likely label is `contradiction`.

| Classifier | All | Clear contradictions | Clear compatibles | Subtle | CPU latency per pair |
|---|---|---|---|---|---|
| NLI small | 20/30 | 10/10 | 7/10 | 3/10 | 14 ms |
| NLI base | 22/30 | 10/10 | 7/10 | 5/10 | 26 ms |
| NLI large | **26/30** | 10/10 | 9/10 | 7/10 | 330 ms |
| `qwen2.5:14b-instruct`, production path, three runs | 27, 28, 26 /30 | 9, 8, 9 /10 | 10, 10, 10 /10 | 8, 10, 7 /10 | 3.6 to 4.0 s (median) |
| `claude-sonnet-4-6`, production path, three runs | 30, 30, 30 /30 | 10/10 each | 10/10 each | 10/10 each | 1.5 to 1.7 s (median) |
| `claude-haiku-4-5`, production path, three runs | 30, 30, 30 /30 | 10/10 each | 10/10 each | 10/10 each | 0.8 s (median) |
| `claude-haiku-5-5`, production path, three runs | 30, 30, 30 /30 | 10/10 each | 10/10 each | 10/10 each | 0.8 to 1.0 s (median) |
| For reference, the original set: `claude-sonnet-4-6` | 30/30 | | | | 1.4 s (median) |
| For reference, the original set: `qwen2.5:14b-instruct` | 29/30 | | | | 4.4 s (median) |

What the misses have in common:

- **Two subjects with one name read as a contradiction.** All three NLI
  models flagged all three different-subject traps (Paris, France against
  Paris, Texas; two exporters; two people named John Smith). An NLI model
  assumes both sentences describe the same scene, which is the opposite of
  what the probe instruction asks.
- **Unrelated claims read as a contradiction.** The small and base models
  flagged "Alice speaks French" against "Alice speaks Spanish" and two other
  pairs that are simply about different things. Their training data labels
  an unrelated caption a contradiction. The large model, trained on more
  varied data, made this error once.
- **Temporal qualification.** The small and base models flagged two or three
  of the three temporal traps. The large model flagged none.
- **The misses are confident.** Moving the threshold on the contradiction
  probability from 0.3 to 0.9 changes the base and large scores by nothing,
  so no threshold rescues them.

Every NLI miss bar one is a false positive. The large model caught all 14
real contradictions, including the subtle scope exception that
`qwen2.5:14b` missed in two of its three runs.

**The local LLM's verdicts vary between runs.** The probe asks for
temperature 0, and the adapter sends it, yet three runs of `qwen2.5:14b` on
an idle machine scored 27, 28 and 26 and missed different pairs each time.
Every run missed at least one clear contradiction ("Mount Everest is the
tallest mountain" against "K2 is taller than Mount Everest"; "requires Python
3.11 or newer" against "runs on Python 3.8"). The earlier measurement's 29 of
30 was a single run, and its finding that `qwen2.5:14b` misses only in the
safe direction does not hold up on repetition: here every run missed a real
contradiction. The NLI models gave the same
verdicts on every run.

### Against the bar

The earlier measurement set the bar for a local probe: at most one miss,
none in a clear tier, and a miss only in the safe direction. The large model
misses four, one of them in the clear-compatible tier, so it does not clear
the bar. The small and base models are further off.

A pattern does stand out: the large model never missed a real contradiction.
Used as a screen in front of the model, passing on only the pairs it flags,
it would have skipped 12 of the 16 non-contradicting pairs here. That
measurement does not carry over to the store, though. The lint pass draws its
candidate pairs from claims that share a subject and sit close in embedding
space, which is where same-name and same-topic pairs concentrate, so the
screen would pass on a larger share of them. The probe also already runs
through the batch API at a cost of cents a night. A screen is not pursued.

## The entailment judge

### The labelled set

The leakage benchmark's second-protocol run of 2026-10-03 (on the owner's
store, composer `claude-sonnet-5`, judge `claude-opus-5-5`) left a checkpoint
recording every judged sentence's verdict and the ids of the 40 particles the
composer answered from. The store snapshot that run used was still on disk
on 2026-10-07, so each sentence's premises could be read back. The one input
not recovered is the
constituent claims of any narrative among the 40, which the composer also saw,
so a sentence drawn from one reads as unsupported to the NLI model. That
biases the NLI figures slightly against it.

The set is the 1,463 claim-bearing sentences the judge scored as supported
or unsupported, 25.1% of them unsupported. Sentences the judge found to assert
nothing were left out, because an NLI model has no such class. That makes the
task easier for NLI than the real one. The labels are the judge's, not a
human's, so agreement here measures whether an NLI model can stand in for the
judge, which is the question a replacement asks.

### How an NLI model judges a sentence

An NLI model compares one premise with one hypothesis, so a sentence is
scored against each of its 40 premises in turn and is **supported** when the
best single premise entails it with probability at least *t*. A sentence
that combines two claims cannot be entailed by either alone, so a second
reading also joins the three best premises into one and takes the higher
score. Both readings are reported.

All three models were scored on a seeded sample of 500 sentences (25.4%
unsupported). The small model was also run over all 1,463 and scored within
0.03 AUROC of its sample figure, so the sample stands for the set.

**A second label set.** The same day's first-protocol run (the judge rules
before it reasons; 338 questions, 1,609 claim-bearing sentences, 23.6%
unsupported) was scored on the 2026-10-08 rerun, with premises read from a
read-only copy of the live store. Particle content never changes after
creation, so the copy returns the same premise text; none of the 20,000
premises looked up was missing. It was sampled the same way (500 sentences,
23.6% unsupported).

### Results

AUROC measures how well the score ranks supported above unsupported
sentences (0.5 is chance, 1.0 is perfect). Cohen's κ measures agreement with
the judge beyond chance (0 is chance, 1 is complete agreement). The row at
*t* = 0.5 uses the better of the two readings.

| Model | AUROC, best premise | AUROC, with top-3 joined | Best κ over *t* | At *t* = 0.5: flagged unsupported | Of those, judge agrees | Of judge's unsupported, caught |
|---|---|---|---|---|---|---|
| NLI small | 0.625 | 0.630 | 0.15 | 64% | 31% | 80% |
| NLI base | 0.642 | 0.661 | 0.17 | 70% | 32% | 87% |
| NLI large | 0.586 | 0.611 | 0.14 | 49% | 32% | 61% |

On the second label set:

| Model | AUROC, best premise | AUROC, with top-3 joined | Best κ over *t* | At *t* = 0.5: flagged unsupported | Of those, judge agrees | Of judge's unsupported, caught |
|---|---|---|---|---|---|---|
| NLI small | 0.632 | 0.642 | 0.15 | 65% | 29% | 78% |
| NLI base | 0.617 | 0.624 | 0.10 | 71% | 27% | 81% |
| NLI large | 0.598 | 0.619 | 0.14 | 54% | 29% | 67% |

Timings, from the idle-machine rerun (the first run's, under load, were two
and a half to three and a half times slower):

| Model | GPU, half precision, per pair | CPU per pair | CPU per answer sentence (40 premises) |
|---|---|---|---|
| NLI small | 5 ms | 20 ms | about 0.8 s |
| NLI base | 10 ms | 38 ms | about 1.5 s |
| NLI large | 29 ms | 581 ms | about 23 s |

No model comes close. The judge calls a quarter of the sentences
unsupported, and every model flags between half and three quarters, of
which the judge agrees with about one in three, barely above the 25% base
rate. Agreement beyond chance tops out at κ = 0.17. The larger model ranks
no better than the smaller ones, and joining the top three premises helps by
at most 0.02 AUROC. The second label set gives the same picture, with AUROC
0.60 to 0.64 and κ at most 0.15.

The reason is the rubric. The judge accepts a paraphrase, a less specific
restatement, a combination of several claims, or an inference that follows
directly, and rejects the composer's interpretation layered onto supported
claims. NLI models are trained on single short premise and hypothesis pairs
about one scene, so they reject restatements the judge accepts and cannot
see the line between a summary and an embellishment. A long technical
premise also differs a great deal from that training data.

On a CPU, judging one answer of five sentences takes about 8 s with the base
model and about 2 minutes with the large one. Speed is not what rules these
models out; agreement is.

### Against the bar

A stand-in judge has to reproduce the judge's verdicts closely enough that a
benchmark's headline rate does not move. Here even the best model doubles
the flagged rate and disagrees with the judge on half the sentences. It does
not clear the bar. The leakage benchmark's requirement that the judge be a
different model from the composer stays met by the hosted judge
it already uses.

## Outcome

- **No default changes.** The `semantic_lint` probe and both entailment judge
  callers (abstraction promotion and the leakage benchmark) stay on their
  configured LLM providers.
- **No classifier seam is proposed.** A seam is worth its code only once a
  purpose has a model that clears its bar, and none has.
- **Not measured here:** the dedup and co-evidential judge, the `use_judge`
  purpose, and span-based mention spotting for subject resolution (GLiNER).
  The dedup judge is the closest fit for a cross-encoder, because whether
  two claims restate each other is a symmetric two-sentence question with no
  combination step, and it is the next candidate. Each needs its own labelled
  set first.
- **The probe pair set is now versioned**, so a later model, local or
  hosted, can be measured on the same pairs. A set of real candidate pairs
  from a store would measure the different-subject case better than
  hand-written pairs do. The store's probe-verdict ledger did not yet hold
  contradiction-probe verdicts when this was run.

## Reproducing

```bash
uv run python scripts/bench_local_classifiers.py probe --local-llm qwen2.5:14b-instruct
```

```bash
uv run python scripts/bench_local_classifiers.py entailment --sample 500 --checkpoint <leakage checkpoint JSONL> --db <the store snapshot the run used>
```

The entailment command reads a leakage run's checkpoint and its store
snapshot read-only, and prints only aggregate figures: no sentence, question
or premise leaves the machine. The probe command calls a local Ollama model
only when `--local-llm` is given, and a hosted model only when
`--anthropic-model` is given, reporting that run's list-price spend.
