# Benchmarks

How good is Particles as an agent memory? This page is the publication
surface for the agent-memory benchmark evaluation:
LongMemEval (Wu et al., ICLR 2025) run against the **pipeline the
agent-memory wedge actually ships** (deposit each haystack chat session as
a `CONVERSATION` corpus entry, standard extraction + §6.6 reconciliation,
`top-k` query retrieval) under default configuration and default
thresholds. Not a benchmark-tuned lab build.

## Results

> **SUBSET run: 150 of 500 questions.** The table of record, measured
> 2026-09-18 to 2026-09-20 under **answer scaffold 2** and **judge protocol
> 2**. It replaces the inaugural 2026-08-16 table outright, which is kept
> below under § The inaugural 2026-08 run; the two are never averaged or
> mixed. Selection tuple: LongMemEval `s` (cleaned) · dataset revision
> `98d7416c24c778c2fee6e6f3006e7a073259d48f` · `sample_seed=13` · strata =
> all six question types · `limit=150` · answer / judge / extraction model
> `anthropic:claude-sonnet-5` · embedding model `all-MiniLM-L6-v2` ·
> thresholds `extraction.similarity_threshold=0.8`,
> `confidence.uncalibrated_cap.enabled=false` · **`subject_rendering: names`**
> (the default since 1.148.2; set `uuids` to reproduce any figure this page
> published before it, including the whole 2026-08 section). `top_k` and the
> read budget vary by row and are stated on it. Raw report JSON is linked from
> every row.

**End-to-end QA.** One answering model (`anthropic:claude-sonnet-5`) on every
row, one judge, one question set. What differs between rows is the memory and
the read budget, and nothing else.

| Condition | n | Read budget (measured mean) | Accuracy |
|---|---:|---:|---:|
| **`qa_particles`, `top_k` 40** (the shipped default) | **150** | 5,182 chars / **2,149 tok** | **0.804** ‡ |
| `qa_particles`, `top_k` 10 (tight budget) | 150 | 1,292 chars / 542 tok | 0.800 |
| `qa_full_context` (the ceiling) | 148 † | 496,320 chars / 166,725 tok | **0.878** |
| `qa_no_memory` (the floor) | 150 | none | 0.047 |
| `qa_notes` comparator, budget-matched to the lead row | 150 | 5,381 chars / 2,107 tok | 0.700 |
| `qa_chunks` comparator, budget-matched to the lead row | 150 | 6,140 chars / 2,266 tok | 0.600 |
| `qa_notes` comparator, budget-matched to the `top_k` 10 row | 150 | 2,564 chars / 950 tok | 0.400 |
| `qa_chunks` comparator, budget-matched to the `top_k` 10 row | 150 | 1,531 chars / 562 tok | 0.320 |
| `qa_notes` comparator, unbudgeted (`top_k` 10) | 150 | 25,893 chars / 9,677 tok | 0.880 |
| `qa_chunks` comparator, unbudgeted (`top_k` 10) | 150 | 10,328 chars / 3,767 tok | 0.733 |

‡ **The lead row is the mean of three runs** (0.807 / 0.807 / 0.800), the only
row on this page with repeats. It has them because the subject-rendering
ablation below needed them; every other row is a single run, and § Ablations
measures what that is worth (two runs of one configuration disagree on 4 to 8
questions).

† **`qa_full_context` is scored over 148 questions, not 150.** Two
`temporal-reasoning` answers (`gpt4_7f6b06db`, `gpt4_9a159967`) returned no
text block inside `max_tokens` and are excluded from the denominator rather
than scored incorrect (see § Two measurement families). **0.878 is an n=148
figure wherever it is quoted**, including on the project's front page.

Particles rows: `top_k` 40
[r1](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-2026-09-20.json) ·
[r2](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-r2-2026-09-20.json) ·
[r3](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-r3-2026-09-20.json),
[20](benchmarks/longmemeval-s150-v3-names-topk20-2026-09-20.json),
[10](benchmarks/longmemeval-s150-v3-names-topk10-2026-09-20.json). Ceiling and
floor: [`v2-rejudge-p2`](benchmarks/longmemeval-s150-v2-rejudge-p2-2026-09-19.json).
Comparator rows are linked in § Comparator memories.

**Read budgets here are measured, not assumed.** Every particle line the
answering model saw was rebuilt from the report's recorded
`context_particle_ids` against the kept store set, and every token figure is
a real count from the answering model's own tokenizer rather than a
characters-per-token estimate. The distinction matters more than it sounds:
the particle context runs at **2.41 characters per token**, against 2.8 for
transcript chunks, 2.7 for session notes and 3.0 for the raw haystack, so equal
*characters* is not equal *cost*. It used to run at 1.97, because each line
carried its subjects as raw UUIDs; § The subject rendering is why it no longer
does and what that was worth.

Character figures are over all 150 questions. Token figures are over all 150
for the two particles rows and over a sample for the rest, because the
distributions are tight and counting is the expensive part: 24 questions for
`qa_full_context` (160,939 to 171,417 tokens) and 25 for each comparator row.

At `top_k` 40 the particle context is **1.29%** of the full history's tokens,
and at `top_k` 10 it is **0.33%**.

**Retrieval depth buys recall and does not buy answers.** Each depth was run
three times, so this is the one place on the page where the sweep has error
bars rather than point estimates:

| `top_k` | Read budget | Recall@k | Precision@k | r1 | r2 | r3 | **Mean** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 542 tok | 0.943 | 0.663 | 0.800 | 0.807 | 0.793 | **0.800** |
| 20 | 1,079 tok | 0.959 | 0.508 | 0.800 | 0.813 | 0.813 | **0.809** |
| **40** (shipped) | 2,149 tok | **0.969** | 0.367 | 0.807 | 0.807 | 0.800 | **0.804** |

The retrieval columns are deterministic and improve monotonically with depth.
The answers do not: the three means span 0.9 points, every depth's own range
overlaps every other's, and `top_k` 20 scores highest. Within-depth churn is 2
to 7 questions, which is the whole of the spread. **Quadrupling the read budget
from 542 to 2,149 tokens buys no measurable accuracy.**

This corrects a claim this page made from single runs. Under the previous
`uuids` rendering the same sweep read 0.787 / 0.813 / 0.820 and was described
as buying about three points for the extra depth; those were one run per depth,
and the gain does not survive repetition at the current default. The marginal
particles between rank 10 and rank 40 do cover more evidence sessions (Recall@k
0.943 to 0.969) and they also dilute the context (Precision@k 0.663 to 0.367),
and on this benchmark those two cancel.

`top_k` 40 leads the table because it is what every product surface ships, not
because it measured best. **The operational reading is that `top_k` 10 answers
as well on a quarter of the read budget**, which is a live question about the
product default rather than the benchmark's, and is not settled by one subset
of 150 questions.

**Retrieval stage** (143 questions scored; **7 abstention questions
excluded**: `6aeb4375_abs`, `f685340e_abs`, `80ec1f4f_abs`, `88432d0a_abs`,
`0862e8bf_abs`, `gpt4_70e84552_abs`, `gpt4_93159ced_abs`, see § Two
measurement families):

| `top_k` | Recall@k | Precision@k |
|---:|---:|---:|
| **40** (shipped default) | **0.969** | **0.367** |
| 20 | 0.959 | 0.508 |
| 10 | 0.943 | 0.663 |

Per question type, at the shipped `top_k` 40:

| Question type | n (retr.) | Recall@40 | Precision@40 | n (QA) | `qa_particles` | `qa_full_context` | `qa_no_memory` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **All** | **143** | **0.969** | **0.367** | **150** | **0.804** | **0.878** § | **0.047** |
| knowledge-update | 21 | 0.976 | 0.402 | 23 | 0.884 | 0.957 | 0.087 |
| multi-session | 38 | 1.000 | 0.461 | 40 | 0.892 | 0.825 | 0.050 |
| single-session-assistant | 17 | 0.882 | 0.335 | 17 | 0.647 | 1.000 | 0.000 |
| single-session-preference | 9 | 1.000 | 0.239 | 9 | 0.370 | 0.667 | 0.000 |
| single-session-user | 20 | 1.000 | 0.266 | 21 | 0.968 | 0.952 | 0.048 |
| temporal-reasoning | 38 | 0.947 | 0.350 | 40 | 0.750 | 0.842 | 0.050 |

**The two n columns are different denominators and the table keeps them
apart.** The retrieval columns exclude the 7 abstention questions, which are
unscoreable at the retrieval stage by protocol, so their per-type counts are
smaller; the QA columns score all 150, abstentions included. Never read a
recall figure and an accuracy figure as being over the same questions.

§ The `qa_full_context` column is n=148: the two excluded answers are both
`temporal-reasoning`, so that row is scored over 38. The `qa_particles` column
is the mean of the lead row's three runs, per type.

Read plainly. At the configuration that actually ships, the store's top-40
particles cover the labeled evidence session 97% of the time, and an answering
model given only those forty claims, 1.3% of the history's tokens, answers 80%
of questions correctly, against 88% when the same model is handed the entire
haystack and 4.7% with no memory at all. **The full-context baseline still
wins overall**, by about seven points, and it is published beside our number
because a ceiling that beats us is the only kind worth reporting. It wins
decisively on `single-session-assistant` (1.000 against 0.647) and
`single-session-preference` (0.667 against 0.370), where the answer sits
verbatim in one session the haystack contains; it loses on `multi-session`
(0.825 against 0.892), where the answer has to be assembled across sessions
and a claim store has already done the assembling. The persistent weak row is
`single-session-preference` at n=9: retrieval is perfect (Recall@40 1.000) and
the answer is still wrong nearly two times in three, so this is a reader-side
or extraction-side loss, not a retrieval one. Cost of the store-building run at
Sonnet 5 introductory pricing, with extraction routed through the Message
Batches API: ≈ US$2.70 per question.

**Run notes** (disclosed so the numbers can be read correctly):

- **The particle rows share one store set**, built 2026-09-18 by a fresh paid
  extraction over the same 150 haystacks and kept with `--store-dir`. The
  `top_k` sweep and the re-judge replay that set, so the three depths differ
  in retrieval and nothing else. Its write-side tuple is pinned in the set's
  manifest and re-checked on every replay.
- Sonnet 5 rejects the `temperature` parameter, so the answer and judge calls
  ran at the model's default sampling: a re-run reproduces these numbers only
  within sampling noise. Steps 1 and 4 of § Why the scores changed measure
  that noise directly.
- **Extraction ran pooled**, via `--pooled`, the fan-in to the batch API, at
  `extraction.max_tokens=16384`. A share of sessions still hit that cap and
  lost the tail of their candidate list; the retrieval and QA numbers include
  that loss.
- The judge is Sonnet 5 running the dataset's official per-type autoeval
  templates verbatim (see § Judge deviation): comparable within this page,
  not to the paper's leaderboard, because the judge *model* is still ours and
  not the paper's.
- **`qa_no_memory` is 0.047 and every one of those seven correct answers is an
  abstention question.** With no memory the model gets the "you never told me"
  questions right by declining, and nothing else right at all.

## Why the scores changed

The headline moved from 0.733 to 0.804 between the inaugural run and this
one. A benchmark number that moves without an account of *what* moved it is
worth very little, so here is the whole distance, one change at a time, each
step holding everything else fixed. Every row is a committed report;
`top_k` is 10 until the last step, and the condition is `qa_particles`
throughout.

| # | What changed at this step | Report | `qa_particles` | Δ |
|---:|---|---|---:|---:|
| 0 | the published 2026-08-16 table | [`2026-08-16`](benchmarks/longmemeval-s150-2026-08-16.json) | 0.733 | - |
| 1 | a fresh extraction: new stores, same scaffold, same judge | [`v1-reuse`](benchmarks/longmemeval-s150-v1-reuse-2026-09-18.json) | 0.740 | +0.007 |
| 2 | answer scaffold 1 → 2, still scored by judge protocol 1 | [`v2`](benchmarks/longmemeval-s150-v2-2026-09-18.json) | 0.740 | 0.000 |
| 3 | judge protocol 1 → 2, re-scoring the *same* stored answers | [`v2-rejudge-p2`](benchmarks/longmemeval-s150-v2-rejudge-p2-2026-09-19.json) | 0.780 | +0.040 |
| 4 | fresh answer calls under that same scaffold-2 / protocol-2 tuple | [`v2-p2-topk10`](benchmarks/longmemeval-s150-v2-p2-topk10-2026-09-18.json) | 0.787 | +0.007 |
| 5 | `top_k` 10 → 40, the shipped default | [`v2-p2-topk40`](benchmarks/longmemeval-s150-v2-p2-topk40-2026-09-18.json) | 0.820 | +0.033 |
| 6 | subjects rendered as names, not UUIDs | [`names` ×3](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-2026-09-20.json) | 0.804 | −0.016 |

Steps 1, 4, 5 and 6 are inside the sampling noise once repeated, step 2 is a
change to how the benchmark asks its question and it bought nothing, and step 3
is a change of ruler. **Not one of the six is an improvement to the memory**,
and only step 3 survives repetition as a real movement at all. The only step that measures the memory at all is step 1, and
what it measures is that the extractor is where it was in August.

**1. A fresh extraction changed nothing.** The store set was rebuilt from
scratch: a new paid extraction over the same 150 haystacks under the same
tuple, then answered under the same scaffold and scored by the same judge.
Accuracy went 0.733 to 0.740, which is one question. Fifteen of the 150
verdicts differ, eight of them upward and seven downward, so the underlying
agreement is lower than the aggregate suggests and the aggregate still did
not move. Extraction is a sampling process; this is what its run-to-run
spread looks like. The extractor neither improved nor regressed.

**2. Answer scaffold v2 rewrote the answers and left the score alone.** The
system turn shared by the three QA conditions is versioned, and v2 is the
default since 1.141.2: question-type-blind reader guidance aimed squarely at
the inaugural run's reader-side failure buckets (conditional abstention,
preference grounding, an enumerate-merge-qualify-count protocol, date
arithmetic, latest-dated-wins). Of the inaugural run's 40 `qa_particles`
misses, 22 had the evidence session fully inside the top 10 and 6 partially:
the memory had handed the model the answer and the model had not used it.
The rewrite was real, only 16 of 150 answers came back byte-identical, and
under the old judge it bought exactly nothing: eight verdicts differed, four
up and four down, 0.740 to 0.740. The matching figure is a coincidence, not
a replay.

**3. The judge is the largest single step, and it is a change of ruler, not
of product.** Protocol 1 was the 1.74.0 port of the dataset's
`get_anscheck_prompt` templates, and it was a paraphrase rather than a copy.
Three of its omissions were load-bearing. The abstention prompt had dropped
the gold explanation and the official clause that a reply offering other
information while denying the asked fact still counts, so textbook refusals
("I don't have that information, the context only mentions a cat named Luna,
not a hamster") were scored as failures to abstain. The preference prompt
labelled the rubric a "reference answer" and lost "the model does not need to
reflect all the points". Temporal reasoning lost the official off-by-one
leniency, and knowledge-update was stricter than the official "previous
information alongside the updated answer is correct". Protocol 2, the default
since 1.141.11 and the protocol this table is scored under, is those
templates verbatim, pinned byte-exact by a unit test. It is the more correct
judge for a plain reason: the quantity being measured is *the benchmark's*
definition of a correct answer, and protocol 1 was measuring our restatement
of it. The answering model, the answers, and the memory are identical across
step 3. Nothing about the product changed; only the measurement did.

Under protocol 2 all seven abstention questions are credited in every
condition, and that is most but not all of the step. Protocol 1 had already
credited 3 of the 7 on the particles path, 5 of 7 on full context, and all 7
with no memory, so the same judge change is worth +4 questions here, +2
there, and nothing at all to the floor. The remaining `qa_particles` flips
are two non-abstention questions; no verdict moved downward.

**4. Re-answering under the same tuple moved one question.** Step 3 re-judged
stored answers; step 4 is the `top_k` 10 point of the sweep, which paid for
fresh answer calls under that identical scaffold-2 / protocol-2 tuple, and it
came back 0.787 against 0.780. One question, in the same direction and of the
same size as step 1. The decomposition quotes 0.787 because it is the
`top_k` 10 point of the single-run `uuids` sweep that step 5 continues, and
therefore like-for-like with the step after it; 0.780 and 0.787 are never
averaged. The headline table's `top_k` 10 row is a different measurement: the
mean of three later runs under the `names` rendering, 0.800.

**The rescoring lifted the ceiling more than it lifted Particles, and the
scaffold did it, not the judge.** This is the sentence a sceptical reader
would otherwise have to find alone, so it is stated here. Across steps 2 and 3
together, `qa_full_context` went 0.793 to 0.878 (+0.085) while `qa_particles`
at `top_k` 10 went 0.740 to 0.780 (+0.040). The judge is not what did it: the
judge (step 3) was worth +0.040 to Particles and
+0.020 to the ceiling, and the whole of the difference is step 2, where the
scaffold moved the ceiling 0.793 to 0.858 and Particles not at all. Part of
that step is a denominator change, since step 2 is where the ceiling's two
unscored answers first appear; counted as wrong over all 150 it is 0.847,
still +0.053.

**On the three-run mean the headline table quotes, Particles at `top_k` 10 is
91.1% of the ceiling (0.800 over 150 questions, against 0.878 over 148), down
from 92.4% in the inaugural table (0.733 against 0.793, both over 150); on
the single `uuids` sweep run the decomposition walks (0.787) it is 89.6%.**
How much of that is real: the three current runs (0.793, 0.800, 0.807) are
90.3%, 91.1% and 91.8% of the ceiling, so every one of them sits below the
inaugural share, by 0.6 to 2.1 points. On the mean the gap is 1.4 points,
which is 1.8 questions of 150 on the Particles side. That is smaller than the
movement this page measures between repeats of one configuration (a total that
moves by up to 5 questions, § The subject rendering), and both ceilings it
divides by, 0.793 and 0.878, are single runs, where one question on the
current ceiling moves the share by 0.7 points. The direction is consistent; a
drop of this size is not distinguishable from run-to-run noise at this n.

At `top_k` 40 the share depends on the rendering. Under the `uuids` rendering
of step 5 (0.820, identical on all three runs of the ablation below) it is
93.4%, above the inaugural 92.4%; under the shipped `names` rendering the
headline table quotes (0.804, the mean of three runs) it is 91.6%, below it,
with the runs spanning 91.1% to 91.8%. Neither figure measures an improvement
in the memory: the gap between them is a rendering choice, and both ride four
times the read budget of `top_k` 10, which the depth sweep above found buys no
measurable accuracy.

The floor moved too, downward: 0.080 to 0.047. That is the scaffold, not the
judge. In August the no-memory condition got the 7 abstention questions plus
5 lucky guesses (four `single-session-assistant`, one `temporal-reasoning`);
under scaffold v2 the model declines instead of guessing, and the floor is
exactly the 7 abstention questions and nothing else. A lower floor widens the
band the memory has to earn.

**5. Raising `top_k` from 10 to 40 buys about three points, and it is not
free.** The last step is the only one that changes what a person who installs
this gets, because 40 is what every product surface defaults to (the query
API's `top_k`, the CLI `--top-k`, the MCP `query` tool, and since 1.141.10
`benchmark_memory.top_k`). Accuracy goes 0.787 to 0.813 to 0.820 at `top_k`
10 / 20 / 40, and Recall@k goes 0.943 to 0.959 to 0.969. The price is read
budget and precision: the measured context grows from 1,599 characters (811
tokens) to 6,375 characters (3,246 tokens), and Precision@k falls from 0.663
to 0.367, so at the default roughly two of every three retrieved particles
are not from a labeled evidence session. Twenty is the efficient point on
this curve (+0.026 for 2× the budget, against a further +0.007 for another
2×); 40 is what ships, so 40 leads the table. **This step does not survive
repetition and is retained only as history.** Each depth has since been run
three times under the current default and the sweep is flat (0.800 / 0.809 /
0.804 at 10 / 20 / 40, every range overlapping); the +0.033 above was one run
per depth. See § Results for the repeated sweep. The step is left in the table
because the published figure really did move this way, not because the
mechanism held up.

**6. Rendering subjects as names instead of UUIDs cut a third of the read
budget and moved the headline down by two questions.** The subjects field was
being written as raw `subject_ids`, which cost tokens and told the answering
model nothing; since 1.148.2 the default renders the subject's name. That
takes the shipped context from 3,246 to 2,149 tokens, 1.95% of the history to
1.29%, and the published figure from 0.820 to 0.804. The ablation below ran
each rendering three times and found no accuracy difference any of them can
distinguish from sampling, so **this step buys a third of the budget and its
apparent cost is noise**, which is also why the table quotes a mean of three
runs here rather than a single number. The honest summary of all six steps:
the memory did not change, the ruler did, the configuration did, and the
context got cheaper.

## Ablations: one knob at a time over one store set

The 150 stores the 2026-09-18 run built were kept, so a read-side knob can be
flipped without re-extracting anything. Each arm below replays those stores
(`--reuse-stores`) and deposits and extracts nothing, so only arms over that
one store set are comparable with each other: a re-extraction is a new
particle population. **The control is the table of record at the top of this
page**, which is that same run at `top_k` 10 and 40, so an arm can be read
against it directly. (Until 2026-09-20 the table at the top was the inaugural
2026-08 run, a different extraction, and this paragraph said the opposite;
demoting that table to § The inaugural 2026-08 run is what changed it.) The
retrieval-only arms (`--no-qa`) made zero LLM calls and
cost US$0.00. They were run under a deliberately invalid API key, so a call
the estimate had not projected would have failed instead of billing.

Every arm was run at the control's `top_k` of 10 and at the shipped default of
40. A same-day replay of the control reproduced the 2026-09-18 retrieval
numbers question for question at both depths, so an arm differs from its
control by its knob and nothing else.

**Retrieval stage** (143 questions; the same 7 abstention questions excluded):

| Arm | What differs from the control | Recall@10 | Precision@10 | Recall@40 | Precision@40 |
|---|---|---:|---:|---:|---:|
| control ([`10`](benchmarks/longmemeval-s150-abl-control-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-control-topk40-2026-09-20.json)) | nothing | 0.943 | 0.663 | 0.969 | 0.367 |
| confidence cap on ([`10`](benchmarks/longmemeval-s150-abl-cap-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-cap-topk40-2026-09-20.json)) | `confidence.uncalibrated_cap.enabled: true` (cap 0.7) | 0.951 | 0.687 | 0.970 | 0.403 |
| decay, 30-day half-life ([`10`](benchmarks/longmemeval-s150-abl-decay-h30-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-decay-h30-topk40-2026-09-20.json)) † | a `CONVERSATION` decay rule, floor 0 | 0.944 | 0.690 | 0.973 | 0.410 |
| decay, 90-day half-life ([`10`](benchmarks/longmemeval-s150-abl-decay-h90-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-decay-h90-topk40-2026-09-20.json)) † | as above | 0.944 | 0.690 | 0.973 | 0.410 |
| decay, 365-day half-life ([`10`](benchmarks/longmemeval-s150-abl-decay-h365-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-decay-h365-topk40-2026-09-20.json)) † | as above | 0.948 | 0.690 | 0.970 | 0.409 |

† **These arms do not measure a preference for recent sessions, and this page
does not claim they do.** See the second point below.

Read plainly:

- **The confidence cap buys retrieval precision and costs no recall.**
  Precision rises 2.4 points at `top_k` 10 and 3.7 at 40 (per question: up on
  37 and down on 23 at 10; up on 81 and down on 19 at 40), and recall moves
  by two questions at most. The mechanism is visible in the stores: every
  particle is raw extractor output, 73 % of them state a confidence above
  the 0.7 cap, and the rank score is a weighted *sum* of similarity and
  effective confidence. Clamping flattens the confidence term for three
  quarters of each store, so similarity decides more of the order. An
  uncalibrated stated confidence says how sure the extractor was that the
  claim was made. It says nothing about whether the claim answers the
  question, and as a ranking input here it behaved as noise.
- **The decay arms came out degenerate, and the way they did is the
  finding.** Stock configuration has no decay rule for conversation
  sources, so a decay arm needs one added; with that done the arm does
  differ from the control. Decay, however, is evaluated at the run's wall-clock
  instant, the haystack sessions are dated 2022 to 2023, and the rank score
  is a sum, so it is not scale-invariant. Under any half-life of a year or
  less every recency factor lands between 0.15 and effectively zero, the
  confidence term drops out of the score, and the arm measures *ranking by
  similarity alone*. Half-lives of 30 and 90 days returned identical tables.
  The reports say so in their own notes. Read these rows as a second,
  stronger version of the cap arm (confidence term removed rather than
  flattened: precision up 2.7 points at 10 and 4.4 at 40), not as evidence
  about recency. Measuring recency preference needs decay evaluated at each
  question's own date, which the read path does not offer yet; that arm is
  still open.
- **What this does and does not license.** Retrieval is deterministic given
  a store, so these differences are not ranker noise. They are still one
  extraction of one 150-question subset, read at session granularity, with
  no QA column: a precision gain of this size may or may not move answer
  accuracy, and given the noise floor measured below, one QA run per arm
  would not settle it.

### The arms that write to the store

Two ablations change the particle population before retrieval instead of the
ranking after it: the scheduled **maintenance cycle** (`--consolidation`) and
the **co-evidential duplicate judge** (`--dedup-judge`). Each ran over its own
copy of the kept stores, because both write in place and the original is
every other arm's control. Both carry a `qa_particles` column; the two
baselines do not depend on the store and are not re-run. The controls are
the 2026-09-18 `v2-p2-topk10` and `topk40` reports above: the same stores,
the same answer scaffold and judge protocol.

| Arm | Recall@10 | Precision@10 | `qa_particles`@10 | Recall@40 | Precision@40 | `qa_particles`@40 |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.943 | 0.663 | 0.787 | 0.969 | 0.367 | 0.820 |
| maintenance cycle on ([`10`](benchmarks/longmemeval-s150-abl-consolidation-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-consolidation-topk40-2026-09-20.json)) | 0.929 | 0.658 | 0.747 | 0.965 | 0.365 | 0.767 |
| duplicate judge on ([`10`](benchmarks/longmemeval-s150-abl-dedup-judge-topk10-2026-09-20.json) · [`40`](benchmarks/longmemeval-s150-abl-dedup-judge-topk40-2026-09-20.json)) | 0.941 | 0.664 | 0.780 | 0.969 | 0.366 | 0.819 ‡ |

‡ 149 questions scored; one answer call returned no text inside its token
budget and is excluded, not scored wrong.

Each arm paid for its pass once, at `top_k` 10. The `top_k` 40 column replays
the same modified stores with the pass's caps at zero, so the population is
identical at both depths; each `40` report says so in its first note.

Read plainly, and published as-is:

- **On this benchmark the maintenance cycle made answers worse, and one pass
  did it.** The cycle made 36,627 LLM calls across the 150 stores (read from
  each store's own run record): 30,000 in the census, which is report-only
  and cannot move retrieval, and 6,627 in the same-subject update sweep, the
  only pass that changed a store. The passes for document supersession,
  pending extraction, utility mining and abstraction had nothing to act on
  in a freshly extracted store of conversation transcripts and made no call. At list
  price that is about US$10, or **seven cents per store per cycle**, four
  fifths of it census. The sweep retired 203 particles in 100 stores as
  superseded by a newer value. Reading them shows the failure: it retires
  *coexisting* facts about one subject as if the later replaced the earlier.
  "The user recently returned from a solo trip to New York City that lasted
  five days" was retired in favour of a trip to Hawaii, under a question
  asking for the days spent in both; "used to swim competitively in college"
  in favour of "used to play tennis competitively in high school", under a
  question asking how many sports. That is why the loss sits in
  `multi-session` (0.850 to 0.725 at `top_k` 10, 0.900 to 0.775 at 40), the
  question type that aggregates across sessions, while `knowledge-update`,
  the type the sweep exists for, did not move at either depth (0.870, 0.913).
- **How much of the QA drop is the cycle, and how much is sampling.** The
  answering and judging models sample, so two runs over an *identical*
  context disagree. The per-question context ids make that measurable.
  Where the cycle left a question's context byte-identical, verdicts still
  flipped: 3 right-to-wrong and 2 wrong-to-right at `top_k` 10, 5 and 1 at
  40. Where the cycle changed the context (22 questions at 10, 34 at 40),
  verdicts went right-to-wrong 5 times at each depth and wrong-to-right 0
  and 1 times, 4 of the 5 losses `multi-session` both times. The
  headline drops of 4.0 and 5.3 points therefore overstate the effect. About 3 points
  are attributable, all in one direction, and **a single-run QA difference
  under roughly 3 points on this subset is inside the noise**. The
  retrieval columns are deterministic and carry no such caveat.
- **What this does not show.** LongMemEval scores a settled corpus at one
  instant, so it rewards keeping everything and cannot credit a pass for
  retiring a value that really did change. The
  [memory-rot benchmark](benchmarks/rot.md) measures that side. Read
  together: the update sweep's precision, how often what it retires was
  truly replaced, is what needs work, and this table is the first
  measurement of what its false positives cost.
- **The duplicate judge changed nothing measurable.** It judged 3,571
  candidate pairs and linked 1,083 as the same claim, for about US$1, and
  every metric sits within one question of the control. Part of that is
  structural: linked claims are collapsed *after* the `top_k` cut, so the
  pass shortens the context (39.8 particles instead of 40) without
  refilling the slots it frees, and it cannot bring new evidence into view.

### The subject rendering: three arms, three repeats each

Every `qa_particles` line above ends in the claim's subjects rendered as raw
UUIDs, which is why the particle context tokenizes at 1.97 characters per
token against 3.0 for the raw haystack. A UUID carries nothing the answering
model can read, so the obvious question is whether the benchmark is paying
read budget for noise. `benchmark_memory.subject_rendering` makes that
measurable: `uuids` (the default, and what every published figure on this page
was measured under), `names` (each subject's `canonical_name`), or `none` (no
subject field at all). Every arm replays the same store set at `top_k` 40, so
retrieval, ranking and the retrieved particles are identical and only the
context text differs.

**This is the one measurement on this page that was run three times per arm**,
because the first pass could not tell its own result from sampling. Nine runs,
150 questions each.

| Rendering | Read budget | vs `uuids` | r1 | r2 | r3 | **Mean** | Range |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`uuids`** (published default) | 6,375 chars / 3,238 tok | - | 0.820 | 0.820 | 0.820 | **0.820** | 0.820 |
| `names` | 5,182 chars / 2,149 tok | **−33.6%** | 0.807 | 0.807 | 0.800 | **0.804** | 0.007 |
| `none` | 3,869 chars / 1,548 tok | **−52.2%** | 0.793 | 0.813 | 0.827 | **0.811** | 0.033 |

Reports: `uuids`
[r1](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-uuids-2026-09-20.json) ·
[r2](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-uuids-r2-2026-09-20.json) ·
[r3](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-uuids-r3-2026-09-20.json);
`names`
[r1](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-2026-09-20.json) ·
[r2](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-r2-2026-09-20.json) ·
[r3](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-names-r3-2026-09-20.json);
`none`
[r1](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-none-2026-09-20.json) ·
[r2](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-none-r2-2026-09-20.json) ·
[r3](benchmarks/longmemeval-s150-v2-p2-topk40-subjects-none-r3-2026-09-20.json).
Each `r1` is the first pass; `r2` and `r3` were run under a scratch `HOME` so
the checkpoint directory was empty and every one of the six is a genuinely
fresh sample rather than a free replay. The `uuids` `r1` report is the only
replay in the set: its run tuple is byte-identical to the table of record's
`top_k` 40 row, so it restored that run's outcomes and reproduced 0.820
without making a call.

**Repeating it is what made the result readable, and it overturned the first
reading.** Two samples of the *same* rendering disagree on 4 to 8 questions
(mean 5.3 across the nine available pairs), which is the churn the
maintenance-cycle arm above also measured and is the same size as the 8 and 6
flips the first pass had cited as evidence that the rendering mattered. Those
flip counts established nothing. What the repeats show instead:

- **The `uuids` arm returned 0.820 three times running** while disagreeing
  with itself on 4 to 8 individual questions each time. The total is stable;
  which questions make it up is not.
- **`none` is not distinguishable from `uuids`.** Its mean is 0.9 points
  lower, and its own range (0.793 to 0.827) straddles `uuids` entirely. On a
  per-question score out of 3, `uuids` is better on 7 questions and worse on
  5.
- **`names` sits consistently just below `uuids`**, at every repeat, by 1.6
  points on the mean. Per question it is better on 2 and worse on 9. That is
  the most suggestive signal in the table and it is still not significant at
  this n.
- **Almost nothing is actually decided by the rendering.** Of 150 questions,
  only 7 to 9 are unstable at all under any rendering, and the count that one
  rendering always gets right while another always gets wrong is **one**, in
  each direction, for both comparisons.

**The honest reading is therefore that a third to a half of this context is
free to drop.** Removing the subject field entirely cuts the read budget by 52.2%,
which would take the shipped default from 1.95% of the full history's tokens
to 0.93%, and costs nothing this measurement can detect. That the *larger*
saving (`none`) scores closer to the default than the smaller one (`names`) is
itself a sign there is no real effect here to order.

**The obvious explanation for `names` was checked rather than assumed, and it
is wrong.** A plausible mechanism is that names collapse subjects a UUID keeps
apart. They do not: across 40 of the scratch stores, 0 of 30,076 subjects
share a `canonical_name` with another subject in the same store.

**The default moved to `names` in 1.148.2, and not to `none`.** Taking a
third of the read budget for no measurable accuracy cost is worth doing, and
the table of record above is now measured that way. `none` is the larger
saving and was not taken, for two reasons that are not statistical. First, the
`qa_particles` context is specified as "claim text, subjects, dates", so
rendering the subject readably is a **correction** while removing the field is
a redefinition of what the benchmark measures. Second, `none` scoring above
`names` is exactly the noise these repeats established: choosing between them
on a 0.7-point gap would be selecting on sampling, which is the error this
whole section exists to avoid. `none` stays available for anyone who wants to
re-open the specification question with data.

Changing the default moved every published `qa_particles` figure, so the table
of record was re-measured as a whole rather than patched: `top_k` 10 and 20
re-run, `top_k` 40 taken as the mean of the three repeats above, and all four
budget-matched comparator arms re-pinned to the new, smaller particles context.
The ceiling, the floor and the entire retrieval stage are untouched, because
neither baseline reads the particle context and the rendering happens after
ranking.

Cost of the whole experiment: **US$7.97** (nine arms, of which one replayed
free).

## Comparator memories: the same run with a different memory

The four conditions above anchor Particles against *no memory* and *the whole
haystack*. They say nothing about how it compares with the memory an agent
harness already gives you. To answer that, the same 150 questions were re-run
with the
particle store swapped for two **comparator memories**
(`particles benchmark memory --memory chunks|notes --no-baselines`): the same
selection tuple, the same answer scaffold, the same answering model, the same
judge, the same session-granularity retrieval scoring. Only the memory
differs.

- **`chunks`: raw-transcript RAG.** Every session cut into turn-aligned
  chunks of ≤ 1,500 characters, embedded with the same MiniLM model the store
  uses, top-k chunks by cosine handed to the answerer. No write-time LLM call.
  Asks: *does claim extraction add anything over retrieving the transcript
  itself?*
- **`notes`: LLM-written session notes**, the harness-memory pattern (a
  "summarise each conversation into a notes file" agent). Every session
  summarised once by the same Sonnet 5 that wrote the particles, notes
  embedded, top-k notes handed to the answerer. Asks: *does Particles'
  epistemic layer beat plain distillation?*

**These runs are new.** Scaffold v2 changes the answering prompt, so
re-judging the stored 2026-08 comparator answers would not have been
like-for-like: the comparators needed fresh answer calls under the same
scaffold-2 / protocol-2 tuple as the table of record, and that is what they
got on 2026-09-20. The write side was free (the `chunks` memory makes no
write-time call, and all 7,122 session notes replayed from the on-disk cache
the 2026-08 run wrote, at 0 failures), so the whole six-arm re-run cost
answer and judge calls only. The `qa_full_context` and `qa_no_memory` columns
are the particles run's: under an identical tuple they are the same calls, and
the reports say so in their own quality notes.

**End-to-end QA**, every arm at 150 questions with zero exclusions:

| Memory | `top_k` | Budget | Read budget delivered | Accuracy |
|---|---:|---:|---:|---:|
| **Particles** | 40 | none | 5,182 chars / 2,149 tok | **0.804** |
| `notes` | 40 | 1,700 | 5,381 chars / 2,107 tok | 0.700 |
| `chunks` | 40 | 1,700 | 6,140 chars / 2,266 tok | 0.600 |
| **Particles** | 10 | none | 1,292 chars / 542 tok | **0.800** |
| `notes` | 10 | 500 | 2,564 chars / 950 tok | 0.400 |
| `chunks` | 10 | 500 | 1,531 chars / 562 tok | 0.320 |
| `notes` | 40 | 2,400 | 8,122 chars / 3,296 tok | 0.807 |
| `notes` | 10 | none | 25,893 chars / 9,677 tok | 0.880 |
| `chunks` | 10 | none | 10,328 chars / 3,767 tok | 0.733 |

Reports of record, in table order:
[`notes @1700`](benchmarks/longmemeval-s150-v3-names-notes-topk40-budget1700-2026-09-20.json),
[`chunks @1700`](benchmarks/longmemeval-s150-v3-names-chunks-topk40-budget1700-2026-09-20.json),
[`notes @500`](benchmarks/longmemeval-s150-v3-names-notes-topk10-budget500-2026-09-20.json),
[`chunks @500`](benchmarks/longmemeval-s150-v3-names-chunks-topk10-budget500-2026-09-20.json),
[`notes @2400`](benchmarks/longmemeval-s150-notes-v2-p2-topk40-budget2400-2026-09-20.json),
[`notes`](benchmarks/longmemeval-s150-notes-v2-p2-2026-09-20.json),
[`chunks`](benchmarks/longmemeval-s150-chunks-v2-p2-2026-09-20.json). The
budget-matched arms were re-pinned when the particles context shrank by a
third; the superseded arms at the old budgets (`chunks @2400`, `chunks @800`,
`notes @800`) stay in `docs/benchmarks/` as the artifacts behind the figures
this page published before 1.148.2.

**How the budget was matched, and why it is not one number.** The clamp
(`--context-budget`) counts characters at a flat four-per-token, but the three
memories do not tokenize alike: 1.97 characters per token for particles
(subject UUIDs), 2.8 for transcript chunks, 2.7 for notes. A single character
budget would therefore hand the three memories different amounts of the thing
that actually costs money. To avoid that, the budget was set per arm to land each
comparator's *delivered token count* on the particles context it is being
compared with, and the delivered size is measured and printed on every row
above rather than assumed from the flag. The match is within 2% on both arms
at `top_k` 40 (3,231 and 3,296 tokens against 3,246) and deliberately loose in
the comparators' favour at `top_k` 10 (920 and 950 against 811), because item
granularity makes the clamp overshoot: a single session note is already longer
than the entire top-10 particle context, so no budget can clamp `notes` below
about 950 tokens. Where the arms are not exactly equal, the comparator has the
larger context.

**Retrieval stage** (143 questions; the same 7 abstention questions excluded):

| Memory | `top_k` | Recall@k | Precision@k | Items per session |
|---|---:|---:|---:|---|
| `particles` | 40 | 0.969 | 0.367 | many (one per claim) |
| `chunks` | 40 | 0.987 | 0.357 | several (one per ≤1.5k chars) |
| `notes` | 40 | 1.000 | 0.046 † | exactly one |
| `particles` | 10 | 0.943 | 0.663 | many |
| `chunks` | 10 | 0.912 | 0.652 | several |
| `notes` | 10 | 0.950 | 0.171 † | exactly one |

† Precision@k is **not comparable across memories of different item
granularity**: a `notes` memory has one item per session, so with one or two
labeled evidence sessions at most one or two of the k retrieved items can ever
be hits. 0.046 at `top_k` 40 is near that structural ceiling, not a defect.
Read recall across the three; read precision only within a memory.

Read plainly, and published as-is:

- **At an unlimited read budget, LLM-written session notes are the strongest
  memory on this benchmark, and they beat Particles.** `notes` scores 0.880
  against Particles' 0.804 at the shipped default, and that 0.880 is level
  with the 0.878 full-context ceiling. It buys this with volume: 9,677 tokens
  of context against Particles' 2,149, four and a half times as much. This is
  the memory pattern an agent harness already ships, and on recall-style
  question answering Particles' epistemic layer (claim granularity,
  provenance, confidence, reconciliation) does not show up as answer accuracy.
- **At the shipped default's own read budget, Particles is ten points ahead of
  session notes and twenty ahead of transcript RAG** (0.804 against 0.700 and
  0.600 at about 2,150 tokens). That margin is larger than this page reported
  before 1.148.2, and **the memory did not change**: the budget did. Dropping
  the UUIDs cut the particles context by a third, and the comparators are far
  more budget-sensitive than Particles is, so the matched point moved down the
  curve to where they fall off it.
- **Session notes need about 50% more context to draw level.** Note
  granularity makes an exact match impossible at `top_k` 40: one note either
  fits or does not, so the achievable budgets bracket the target rather than
  hitting it. At 2,107 tokens (2% *under* Particles' 2,149) `notes` scores
  0.700; at 3,296 tokens (53% over) it scores 0.807, which is finally level
  with Particles' 0.804. Both arms are in the table so the bracket is visible
  rather than resolved in our favour by a choice of budget.
- **At a tight read budget the gap is widest.** At about 550 tokens Particles
  answers 0.800 against 0.400 for session notes and 0.320 for transcript
  chunks, **twice and two and a half times** as many questions, while reading
  fewer tokens than either. One session note or one transcript chunk cannot
  carry an answer assembled from several sessions.
- **Transcript RAG never catches up.** `chunks` unbudgeted (0.733, 3,767
  tokens) is *below* Particles at `top_k` 10 (0.800, 542 tokens), so Particles
  answers more questions on a seventh of the context. Extraction earns its
  keep against raw retrieval at every budget measured.
- Cost: the comparator arms across both pinnings cost about **US$13** in
  total, answer and judge calls only, because `chunks` writes nothing and all
  7,122 session notes replayed from the 2026-08 cache. For comparison, writing
  those notes in August cost ≈ US$70, and building the particle stores
  ≈ US$2.70 per question.

The table therefore says one thing precisely: **Particles' advantage is information
density, the most answer per read-time token, and it grows as the budget
tightens.** Its deficit is coverage: given several times the budget,
whole-session distillation recovers what claim extraction paraphrased away and
then wins outright. Which memory is "better" depends on the read-time budget
the agent can afford, and the crossover sits somewhere between the shipped
default's ~2,150 tokens and the ~3,300 at which notes draw level.

**What could not be made like-for-like**, stated rather than smoothed over:

- **The budgets are matched as closely as item granularity allows, not
  exactly.** Three of the four matched arms read slightly *more* than
  Particles (562 against 542, 950 against 542, 2,266 against 2,149); the
  fourth, `notes` at `top_k` 40, reads 2% *less* (2,107 against 2,149) because
  the next achievable budget is a whole extra note, 53% over. Both sides of
  that bracket are published rather than one being chosen.
- **The lead row is a mean of three runs; every other row is one run.** They
  are not equally precise, and the difference is roughly 5 questions of
  sampling churn (§ Ablations).
- **`qa_full_context` is n=148 and every other row is n=150.** The two
  excluded questions are output-budget failures on our side, not memory
  failures, and excluding them is the lesser distortion (see § Two measurement
  families), but the denominators are not identical.
- **The comparator arms run at the `top_k` of the particles row they are
  matched against** (40 for the ~2,150-token arms, 10 for the ~550-token arms
  and the unbudgeted arms), so the retrieval-stage rows above compare memories at
  equal depth but not at equal item size. Precision is not comparable across
  them at all; recall is.
- **The unbudgeted comparator arms are at `top_k` 10, as in August**, so they
  are comparable with the inaugural comparator table. There is no unbudgeted
  `top_k` 40 comparator arm; at that depth the `notes` context would be about
  100,000 characters, which is a fifth of the whole haystack and no longer a
  memory in any useful sense.
- **The judge model is Anthropic's, not the paper's OpenAI autoeval judge.**
  That deviation is constant across every row here, so it cannot explain any
  gap in this table, but it does mean none of these numbers belongs beside a
  published leaderboard figure (see § Judge deviation).

## The inaugural 2026-08 run (historical record)

Everything below this heading is **superseded**. It is the inaugural
2026-08-16 run, the table this page published until 2026-09-20, kept intact
so that every number the project has ever quoted stays traceable to the report
it came from. It is **not comparable** with the table of record above: it was
answered under answer scaffold 1 and scored under judge protocol 1, and both
have since changed (see § Why the scores changed). Do not mix rows between the
two, and do not average them.

To reproduce this section rather than the current table, set
`benchmark_memory.answer_scaffold: 1`, `benchmark_memory.judge_protocol: 1`,
and `benchmark_memory.top_k: 10`.

> **SUBSET run: 150 of 500 questions.** Inaugural run, 2026-08-16.
> Selection tuple: LongMemEval `s` (cleaned) · dataset revision
> `98d7416c24c778c2fee6e6f3006e7a073259d48f` · `sample_seed=13` · strata =
> all six question types · `limit=150` · `top_k=10` ·
> answer / judge / extraction model `anthropic:claude-sonnet-5` ·
> embedding model `all-MiniLM-L6-v2` · thresholds
> `extraction.similarity_threshold=0.8`,
> `confidence.uncalibrated_cap.enabled=false`. Raw report JSON, the
> artifact of record:
> [`benchmarks/longmemeval-s150-2026-08-16.json`](benchmarks/longmemeval-s150-2026-08-16.json).

**Retrieval stage**: 143 questions scored; **7 abstention questions
excluded** (`6aeb4375_abs`, `f685340e_abs`, `80ec1f4f_abs`, `88432d0a_abs`,
`0862e8bf_abs`, `gpt4_70e84552_abs`, `gpt4_93159ced_abs`).

| Question type | n | Recall@10 | Precision@10 |
|---|---:|---:|---:|
| **All** | **143** | **0.940** | **0.668** |
| knowledge-update | 21 | 0.976 | 0.710 |
| multi-session | 38 | 0.984 | 0.771 |
| single-session-assistant | 17 | 0.882 | 0.753 |
| single-session-preference | 9 | 0.889 | 0.311 |
| single-session-user | 20 | 0.950 | 0.620 |
| temporal-reasoning | 38 | 0.908 | 0.613 |

**End-to-end QA**: 150 questions, one answering model
(`anthropic:claude-sonnet-5`) across all three conditions:

| Question type | n | `qa_particles` | `qa_full_context` (baseline) | `qa_no_memory` (floor) |
|---|---:|---:|---:|---:|
| **All** | **150** | **0.733** | **0.793** | **0.080** |
| knowledge-update | 23 | 0.739 | 0.826 | 0.087 |
| multi-session | 40 | 0.800 | 0.750 | 0.050 |
| single-session-assistant | 17 | 0.647 | 0.941 | 0.235 |
| single-session-preference | 9 | 0.333 | 0.667 | 0.000 |
| single-session-user | 21 | 0.762 | 0.905 | 0.048 |
| temporal-reasoning | 40 | 0.775 | 0.725 | 0.075 |

Read plainly, as it was published: the store's top-10 particles cover the
labeled evidence session(s) 94 % of the time, and an answering model given
only those ten particles answers correctly 73 % of the time, against 79 % when
the same model is handed the entire haystack, and 8 % with no memory at all.
**The full-context baseline wins overall**, by 6 points; it wins large on the
three single-session types and loses to the particle path on `multi-session`
and `temporal-reasoning`. The gap between 94 % retrieval recall and 73 %
answer accuracy is the claim-granularity cost: the right *session* is
retrieved but the ten particles do not always carry the specific fact; the
`single-session-preference` row (0.31 precision, 0.33 accuracy, n=9) is the
sharpest instance. Cost of the run at Sonnet 5 introductory pricing, with
extraction routed through the Message Batches API: ≈ US$2.70 per question.

**Run notes for this run** (disclosed so its numbers can be read correctly):

- **The run was executed in five checkpointed segments**, the
  per-question checkpoint, spanning two harness fixes that
  landed on `main` mid-run. The first 12 questions were extracted with live Wikidata subject
  resolution on for every conversational subject name; the remaining 138 with
  it off (`subjects.skip_live_authorities_source_types`, 1.129.8). Retrieval
  and QA scoring never read subject identity, so the effect on the table is
  near zero, but §6.6 reconciliation is subject-gated, so it is not provably
  zero.
- **Five questions were re-run after a budget fix.** Sonnet 5's adaptive
  thinking spends from the same `max_tokens` as the answer, and five calls
  (three `qa_full_context` answers at the old 1024 cap, two abstention-judge
  verdicts at the old 16 cap) returned no text block and were scored
  incorrect. The caps were raised (4096 / 1024) and those five questions
  were dropped from the checkpoint and re-run end to end. A higher cap never
  changes a reply that finished under the lower one, so the other 145
  questions are unaffected. The final report carries **zero** failed
  answer or judge calls.
- **Extraction ran pooled**, via `--pooled`, the fan-in to the batch API, at
  `extraction.max_tokens=16384` (the default 8192 truncated ~45 % of
  sessions in a smoke test). Even so, **≈ 437 of ≈ 6,900 extraction calls
  (~6 %) hit the 16k cap** and lost the tail of that session's candidate
  list; the retrieval and QA numbers include that loss.
- Sonnet 5 rejects the `temperature` parameter, so the answer and judge calls
  ran at the model's default sampling.
- **The run used answer scaffold v1 and judge protocol 1.** Both defaults have
  since moved, which is most of the distance between this table and the
  current one; § Why the scores changed decomposes it step by step.
- **The run used `top_k=10`, below the shipped configuration.** Every product
  surface defaults to 40, and since 1.141.10 so does `benchmark_memory.top_k`.
  This table therefore understated what the default configuration retrieves
  and answers.

### The inaugural comparator tables (2026-08, superseded)

These are the 2026-08 comparator runs, under scaffold 1 and judge protocol 1.
They are superseded by the fresh arms in § Comparator memories, which were
re-answered under the current scaffold and judge. Reports of record:
[`chunks`](benchmarks/longmemeval-s150-chunks-2026-08-16.json),
[`notes`](benchmarks/longmemeval-s150-notes-2026-08-16.json),
[`chunks @500`](benchmarks/longmemeval-s150-chunks-budget500-2026-08-17.json),
[`notes @500`](benchmarks/longmemeval-s150-notes-budget500-2026-08-17.json).

**Retrieval stage** (143 questions; same 7 abstention questions excluded):

| Memory | Recall@10 | Precision@10 | Items per session |
|---|---:|---:|---|
| `particles` | **0.940** | 0.668 | many (one per claim) |
| `chunks` | 0.912 | 0.652 | several (one per ≤1.5k chars) |
| `notes` | **0.950** | 0.171 | exactly one |

**End-to-end QA, as configured** (150 questions, one answering model):

| Question type | n | `qa_particles` | `qa_chunks` | `qa_notes` | `qa_full_context` (reused) | `qa_no_memory` (reused) |
|---|---:|---:|---:|---:|---:|---:|
| **All** | **150** | **0.733** | **0.693** | **0.813** | **0.793** | **0.080** |
| knowledge-update | 23 | 0.739 | 0.783 | 0.870 | 0.826 | 0.087 |
| multi-session | 40 | 0.800 | 0.450 | 0.775 | 0.750 | 0.050 |
| single-session-assistant | 17 | 0.647 | 1.000 | 0.882 | 0.941 | 0.235 |
| single-session-preference | 9 | 0.333 | 0.889 | 0.778 | 0.667 | 0.000 |
| single-session-user | 21 | 0.762 | 0.810 | 0.810 | 0.905 | 0.048 |
| temporal-reasoning | 40 | 0.775 | 0.650 | 0.800 | 0.725 | 0.075 |

**Budget-matched arm** (`--context-budget 500`, ~2,000 characters):

| Question type | n | `qa_particles` (~1.3–2k chars) | `qa_chunks` @ 2k | `qa_notes` @ 2k | `qa_chunks` unclamped | `qa_notes` unclamped |
|---|---:|---:|---:|---:|---:|---:|
| **All** | **150** | **0.733** | **0.293** | **0.393** | 0.693 | 0.813 |
| knowledge-update | 23 | 0.739 | 0.391 | 0.391 | 0.783 | 0.870 |
| multi-session | 40 | 0.800 | 0.200 | 0.125 | 0.450 | 0.775 |
| single-session-assistant | 17 | 0.647 | 0.412 | 0.882 | 1.000 | 0.882 |
| single-session-preference | 9 | 0.333 | 0.333 | 0.556 | 0.889 | 0.778 |
| single-session-user | 21 | 0.762 | 0.429 | 0.714 | 0.810 | 0.810 |
| temporal-reasoning | 40 | 0.775 | 0.200 | 0.250 | 0.650 | 0.800 |

The claim this page carried on those numbers was that at a fixed ~2k-character
read budget Particles answered 1.9× as many questions correctly as session
notes and 2.5× as many as transcript RAG. Both multiples have since been
re-measured under the current scaffold and judge, at a budget matched in
tokens rather than characters, and both came down: see § Comparator memories
for what the data says now.

## Two measurement families, never merged

Every run measures four conditions and reports them in **two separately
labeled families**. Conflating them is the endemic dishonesty mode of the
memory-benchmark space ("our memory retrieves the right session 85% of the
time" quoted as "answers correctly 85% of the time"), so the separation is
structural: the report model has no aggregate score field, and the renderer
has no way to merge the sections.

**Retrieval stage**: a property of the particle store and its ranker,
saying nothing about answer accuracy:

| Condition | What it measures |
|---|---|
| `retrieval` | Evidence-session **Recall@k** and **Precision@k** of the store's top-k query result, scored by mapping each retrieved particle through its provenance chain (particle → corpus entry → URI-R → haystack session) against the dataset's labeled evidence sessions |

**Abstention questions are excluded here, with a disclosed count.** An
abstention variant (`*_abs`) has no evidence session by protocol (the
right answer is "you never told me"), so retrieval is unscoreable for it,
and blending it in would inflate mean recall (a vacuous 1.0 with nothing to
miss) and deflate mean precision (every retrieved particle a "false
positive" against an empty set). The exclusion keys off the protocol flag,
not the label shape: the cleaned dataset labels each abstention question
with its *near-miss* session (the one that mentions a similar-but-different
fact), and that label is deliberately not treated as evidence. They
contribute to no retrieval aggregate; the table discloses the excluded count
and lists each excluded question as `n/a (abstention)`. They remain fully **in** the QA
family below, where the dataset's protocol scores them (credit for declining
to answer).

**End-to-end QA**: an answering LLM on top of (or instead of) the memory,
which can recover from bad retrieval or fumble good retrieval:

| Condition | What it measures |
|---|---|
| `qa_particles` | Accuracy of the answering model given the question + the top-k retrieved particles (claim text, subjects, dates) |
| `qa_full_context` | **The baseline that must not be buried**: the *same* model, same prompt scaffold, with the entire concatenated haystack instead of retrieved particles. Published either way, including if it wins |
| `qa_no_memory` | The same model with the question only: the parametric-guessing / abstention floor |

Conditions ii–iv use **one answering model pinned to one resolved model
id**; the runner refuses to run a QA condition set whose resolved models
differ. If a baseline condition was skipped, its row renders as `not run`.
There is no flag to omit it, so a partial comparison is always visibly
partial.

**A call that produced no verdict is excluded from the accuracy denominator,
with a disclosed count.** If the answer or the judge call yields no usable
reply, that question is not scored wrong for that condition: it is scored
not at all, counted, and named in the table. The reason is asymmetry:
`qa_full_context` sends ~115k tokens per call and `qa_no_memory` sends three
lines, so any shared failure rate would land almost entirely on the baseline,
weakening it with transport noise rather than with anything about the memory.
The exclusion is disclosed per condition and **split by cause**, because the
two mean opposite things to a reader:

- **output-budget**: the reply carried no text within `max_tokens`. An
  extended-thinking model spends its thinking from the same budget, so this is
  a configuration error on our side; the table says so, and the fix is to
  raise the cap and re-run those questions. It is deliberately not retried,
  since an identical call at an identical cap reproduces it.
- **infra**: the call still failed after its retries.

The table also states, before any of this, whether the full-context baseline
*fits* the answering model's context window on the variant being run. A run
whose haystack would overflow is refused rather than reported: an overflowing
baseline is not a weaker baseline, it is a destroyed one, while the
question-only condition sails through untouched. This is the standing
precondition on the larger `m` variant.

Both rules were added in v1.137.1 and bind every run on this page. The
superseded 2026-08 run carries zero failed answer or judge calls, so nothing
in that section was ever restated. The table of record carries exactly two,
both output-budget failures on `qa_full_context`, which is why its ceiling is
an n=148 figure and says so on every row that quotes it.

## Judge deviation: read before comparing

Answers are scored by an LLM judge following the dataset's
per-question-type autoeval protocol, **ported to an Anthropic judge**
(routed through the `llm.benchmark` purpose) rather than the paper's OpenAI
judge. Abstention-variant questions score per the dataset protocol (credit
for declining to answer). Consequence: **numbers on this page are
comparable within the table (same judge, same protocol, same selection),
not across leaderboards.** An OpenAI-judge protocol-fidelity option is
deferred until cross-leaderboard comparability becomes a requirement.

**The judge *prompt* deviation is fixed, and fixing it moved the numbers.**
Until 1.141.11 the judge ran `benchmark_memory.judge_protocol: 1`, the 1.74.0
port, which turned out to be a paraphrase of the dataset's
`get_anscheck_prompt` templates rather than a copy, deviating in ways that
land on specific categories: the abstention prompt omitted the gold
*explanation* and the official clause that a response which offers other
information while denying the asked fact still counts (so "I don't have that
information, the context only mentions a cat named Luna, not a hamster" was
judged *not* an abstention); the preference prompt labelled the rubric a
"reference answer" and dropped "the model does not need to reflect all the
points"; temporal reasoning lost the off-by-one leniency; knowledge-update was
stricter than the official "previous information alongside the updated answer
is correct". **Protocol 2, the default since 1.141.11, is the official
templates verbatim on the same Anthropic judge, and it is what the table of
record is scored under**: the model deviation remains, the prompt deviation is
gone. The protocol rides the run tuple and the checkpoint key, so a protocol-1
run and a protocol-2 run are never mixed. Set `judge_protocol: 1` to reproduce
the superseded 2026-08 section instead. § Why the scores changed isolates what
this one change was worth, on identical answers: +0.040 to `qa_particles`,
+0.020 to the full-context ceiling, and nothing at all to the floor.

## Subset labeling discipline

Any run over fewer than all questions is a **subset run**, and the table
header must say so, including the full selection tuple that makes it
reproducible: dataset revision, variant, sample seed, strata (question
types), limit, resolved answer/judge model ids, the resolved **extraction
model id** and **embedding model id** (the store's contents are a function
of the first and the ranking of the second; two runs that differ on either
are different pipelines, not comparable), `top_k`, and a snapshot of the
pipeline thresholds in effect. The answer, extraction, and judge model
resolutions are each pinned mid-run by refusal: a drift aborts the run
rather than silently mixing pipelines. Two runs with the same recorded
tuple are comparable; anything else is disclosed drift. Subset and full-run
numbers are never mixed in one table.

## Reproducing

```bash
# Cost preview only: no LLM call is made
particles benchmark memory --estimate

# Dev loop (defaults: 10 questions, s variant, seeded stratified selection)
particles benchmark memory

# The publishable runs (operator-invoked, never inside a test suite)
particles benchmark memory --limit 150 --variant s --format json --output report.json
particles benchmark memory --all --variant s --format json --output report.json

# The comparator memories over the same selection (reuse the particles run's
# qa_full_context / qa_no_memory columns; same tuple, same calls)
particles benchmark memory --limit 150 --variant s --memory chunks --no-baselines --top-k 10 --format json --output chunks.json
particles benchmark memory --limit 150 --variant s --memory notes --no-baselines --top-k 10 --format json --output notes.json

# The budget-matched comparator arms, clamped to the particles context at the
# top_k they are compared against (see the comparator section for the values)
particles benchmark memory --limit 150 --variant s --memory notes --no-baselines --top-k 40 --context-budget 2400 --format json --output notes-matched.json
particles benchmark memory --limit 150 --variant s --memory chunks --no-baselines --top-k 10 --context-budget 800 --format json --output chunks-tight.json

# The subject-rendering ablation (config-only, like the scaffold and judge
# knobs): set benchmark_memory.subject_rendering to uuids | names | none and
# replay one store set at a fixed top_k. Retrieval is identical across the
# three arms; only the context text changes.
particles benchmark memory --limit 150 --variant s --top-k 40 --store-dir STORES --reuse-stores --no-baselines --format json --output subjects.json

# Re-score a saved report under the current judge protocol and judge model.
# Only the judge call is re-run over each row's stored answer: no answer call,
# so the full-context baseline is not re-paid and the answers do not change.
particles benchmark memory rejudge report.json --output report-rejudged.json

# Ablations: keep the stores once, then replay them. A replay deposits and
# extracts nothing, so a retrieval-only arm makes no LLM call at all. The knob
# is one config overlay per arm (PARTICLES_CONFIG names it); --fresh discards
# any checkpoint left under the same key.
particles benchmark memory --limit 150 --variant s --store-dir stores/ --format json --output control.json
PARTICLES_CONFIG=cap-on.yaml particles benchmark memory --limit 150 --variant s \
  --store-dir stores/ --reuse-stores --no-qa --top-k 10 --fresh --format json --output cap-on.json
```

An arm that writes (`--consolidation`, `--dedup-judge`) changes the store files
in place, so run it over a copy of the kept set: the original is the control
population for every other arm.

The dataset (LongMemEval v1 cleaned, MIT-licensed, ~3 GB) is downloaded on
demand from HuggingFace at a pinned revision with SHA-256 verification and
cached under `~/.particles/benchmark/longmemeval/`, never vendored into
the repository. Answering routes through the `llm.benchmark_answer`
config purpose; the judge through `llm.benchmark`. Each question runs in an
ephemeral scratch store, so a benchmark run never touches a user store.

`rejudge` exists because the judge stage is separable from the answer stage:
every QA row records the answering model's reply (since 1.141.1) and the
judge prompt is versioned (`benchmark_memory.judge_protocol`, since 1.141.11).
It takes a `--format json` report and writes a complete report of record:
the retrieval stage copied unchanged, every QA condition re-scored with
`accuracy` and `accuracy_by_type` recomputed, `selection.judge_protocol` and
`selection.judge_model_id` set to what was used, and a first quality note
naming the source report path and both judge tuples (the source's and the
new one). Rows with no stored answer (excluded at answer time, or written
with `benchmark.record_claim_text: false`) cannot be re-scored and stay
excluded, disclosed under the `unrecorded` cause; a report with no stored
answers at all is refused. The judge model is pinned across the pass exactly
as in a full run. The dataset is needed (the judge prompt uses the question
text and reference answer, which the report does not carry): the report's
recorded variant and revision are downloaded on demand, or pass
`--dataset-file`. The output file is always JSON; `--format` chooses what is
printed.

## The query gate: the relevance floor on real questions

The query operation refuses, without calling the model, when nothing it
retrieved is close to the question. A wrongly refused question is invisible
by construction, so the gate has its own measurement:
[the relevance floor, measured on real questions](benchmarks/relevance-floor.md).
Replaying 338 real questions against a 27,980-belief store, the default floor
(0.25) refuses 11 of them, and all 11 were judged unanswerable: of the 71
questions a judge found the store could answer, it refused none (the first
judged false negative appears at a floor of 0.40). The floor is a coarse
filter, though: 95.9% of unanswerable questions pass it, and the answer
step's own decline rule does most of the refusing. It is a separate
family from both tables above and is never merged with them.

## Relationship to the extractor benchmarks

The `particles extractor benchmark*` verbs (including the modality and
polarity variants) measure a single **extractor's** output against gold
particles. This page's benchmark measures the **whole pipeline** against
gold answers: a different system under test, reported under its own
`particles benchmark` verb group.

`particles extractor benchmark` additionally persists each run's report as
a JSON file under `benchmark.runs_dir` (default `~/.particles/benchmark/runs/`),
stamped with the resolved extraction provider:model pairing, the durable
raw series behind provider comparisons and calibration-drift analysis.
Pass `--no-save` for a throwaway run.

Extraction is a sampling process, so a single run is a single sample:
`--runs N` repeats each suite N times and reports each metric's mean, range
and standard deviation rather than one point estimate: the error bars a
provider comparison needs before it calls a gap real. Each pass still
persists its own report file. The cost is N× the LLM calls, so the repeat
path prints its projection first and asks before spending above
`benchmark.confirm_call_threshold` (`--estimate` prints and exits; `--yes`
pre-confirms). `--fail-on` is evaluated against the mean across runs.
A single run (the default) is unchanged and never gated.

Each report also carries, per case, an `emitted_claims` record for every
claim the extractor produced: its text, its stated confidence, and whether
the judge scored it matched, under-confidence, or spurious. Without it a
saved report is not auditable: the harness never writes to the particle
store, so an emitted particle's uuid resolves to nothing once the run ends,
and a hallucinated claim reads exactly like a correct claim the gold set
happens not to list. With it, a precision figure can be opened up, and often
should be. A single 2026-09-13 run of `prose-article-seed-001` on
`claude-haiku-4-5` scored 32 spurious claims, of which **6 were the *same
fact* as a claim the same run reported as `MISSED REQUIRED`**, sitting at
0.66–0.78 cosine, just under the 0.80 embedding threshold, mostly because
the emitted claim elided the subject ("The wreck lies at 41 meters depth"
against gold "The Meridian Rose wreck lies at a depth of 41 meters"). Each of
those six is charged twice, once against precision and once against recall,
and nothing in the metrics says so.

**That finding has since been fixed, and the fix moved the numbers.** It was a
mechanism for the bias the
[2026-09 provider survey](benchmarks/provider-survey-2026-09.md) warns about
and calls "not small or quantified": the extraction prompt and the equivalence
judge were both tuned against Sonnet, so a model that frames a claim
differently (here, without restating the subject) was penalised twice for
wording rather than once for content. A particle's subjects are a *field*, so
the judge was scoring the pair on a difference the schema itself mandates.
Since 1.140.0 the judge scores each emitted claim under both renderings, bare
and subject-qualified, and keeps the better one; the 0.80 floor did
not move. On the same fixed emission set that cost precision 0.800 → 0.912 and
recall 0.714 → 0.857.

**Benchmark figures from before 1.140.0 are not comparable with figures after
it.** That includes both provider-survey pages and every archived run file.
Set `benchmark.subject_aware_matching: false` to reproduce an old number.
Reading a run's `emitted_claims` remains how you tell a real error from a
judge artefact; the next known artefact of this kind is compound gold claims,
which one-to-one assignment cannot satisfy at all.

Set `benchmark.record_claim_text: false` to suppress the text when running
against a corpus whose content must not land in a run file; ids, counts and
every metric are unaffected.
