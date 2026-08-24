# Benchmarks

How good is Particles as an agent memory? This page is the publication
surface for the agent-memory benchmark evaluation:
LongMemEval (Wu et al., ICLR 2025) run against the **pipeline the
agent-memory wedge actually ships** — deposit each haystack chat session as
a `CONVERSATION` corpus entry, standard extraction + §6.6 reconciliation,
`top-k` query retrieval — under default configuration and default
thresholds. Not a benchmark-tuned lab build.

## Results

> **SUBSET run — 150 of 500 questions.** Inaugural run, 2026-08-16.
> Selection tuple: LongMemEval `s` (cleaned) · dataset revision
> `98d7416c24c778c2fee6e6f3006e7a073259d48f` · `sample_seed=13` · strata =
> all six question types · `limit=150` · `top_k=10` ·
> answer / judge / extraction model `anthropic:claude-sonnet-5` ·
> embedding model `all-MiniLM-L6-v2` · thresholds
> `extraction.similarity_threshold=0.8`,
> `confidence.uncalibrated_cap.enabled=false`. Raw report JSON, the
> artifact of record:
> [`benchmarks/longmemeval-s150-2026-08-16.json`](benchmarks/longmemeval-s150-2026-08-16.json).
> A later full-500 run replaces this table outright — the two are never
> averaged or mixed.

**Retrieval stage** — 143 questions scored; **7 abstention questions
excluded** (`6aeb4375_abs`, `f685340e_abs`, `80ec1f4f_abs`, `88432d0a_abs`,
`0862e8bf_abs`, `gpt4_70e84552_abs`, `gpt4_93159ced_abs`), see below.

| Question type | n | Recall@10 | Precision@10 |
|---|---:|---:|---:|
| **All** | **143** | **0.940** | **0.668** |
| knowledge-update | 21 | 0.976 | 0.710 |
| multi-session | 38 | 0.984 | 0.771 |
| single-session-assistant | 17 | 0.882 | 0.753 |
| single-session-preference | 9 | 0.889 | 0.311 |
| single-session-user | 20 | 0.950 | 0.620 |
| temporal-reasoning | 38 | 0.908 | 0.613 |

**End-to-end QA** — 150 questions, one answering model
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

Read plainly: the store's top-10 particles cover the labeled evidence
session(s) 94 % of the time, and an answering model given only those ten
particles answers correctly 73 % of the time — against 79 % when the same
model is handed the entire haystack, and 8 % with no memory at all. **The
full-context baseline wins overall**, by 6 points; it wins large on the
three single-session types (where the whole answer sits in one session the
haystack contains verbatim) and loses to the particle path on
`multi-session` and `temporal-reasoning` (where the answer is assembled
across sessions). The gap between 94 % retrieval recall and 73 % answer
accuracy is the claim-granularity cost: the right *session* is retrieved
but the ten particles do not always carry the specific fact — the
`single-session-preference` row (0.31 precision, 0.33 accuracy, n=9) is the
sharpest instance. Cost of the run at Sonnet 5 introductory pricing, with
extraction routed through the Message Batches API: ≈ US$2.70 per question.

**Run notes** (disclosed so the numbers can be read correctly):

- **The run was executed in five checkpointed segments** — the
  per-question checkpoint — spanning two harness fixes that
  landed on `main` mid-run. The first 12 questions were extracted with live Wikidata subject
  resolution on for every conversational subject name; the remaining 138 with
  it off (`subjects.skip_live_authorities_source_types`, 1.129.8). Retrieval
  and QA scoring never read subject identity, so the effect on the table is
  near zero — but §6.6 reconciliation is subject-gated, so it is not provably
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
- **Extraction ran pooled** — `--pooled`, the fan-in to the batch API
   — at `extraction.max_tokens=16384` (the default 8192 truncated ~45 % of
  sessions in a smoke test). Even so, **≈ 437 of ≈ 6,900 extraction calls
  (~6 %) hit the 16k cap** and lost the tail of that session's candidate
  list; the retrieval and QA numbers include that loss.
- Sonnet 5 rejects the `temperature` parameter, so the answer and judge
  calls ran at the model's default sampling: a re-run will not reproduce
  these numbers exactly, only within sampling noise.
- The judge is Sonnet 5 following the dataset's per-type autoeval protocol
  (see § Judge deviation): comparable within this table, not to the
  paper's leaderboard.

## Comparator memories — the same run with a different memory

The four conditions above anchor Particles against *no memory* and *the
whole haystack*. They say nothing about how it compares with the memory an
agent harness already gives you. So the same 150 questions were re-run with
the particle store swapped for two **comparator memories**
(`particles benchmark memory --memory chunks|notes --no-baselines`): the same
selection tuple, the same answer scaffold and answering model, the same
judge, the same session-granularity retrieval scoring; only the memory
differs. The `qa_full_context` and `qa_no_memory` columns are the particles
run's — under an identical tuple they are the same calls — and are marked
*reused*. Reports of record:
[`chunks`](benchmarks/longmemeval-s150-chunks-2026-08-16.json),
[`notes`](benchmarks/longmemeval-s150-notes-2026-08-16.json).

- **`chunks` — raw-transcript RAG.** Every session cut into turn-aligned
  chunks of ≤ 1,500 characters, embedded with the same MiniLM model the
  store uses, top-10 chunks by cosine handed to the answerer. No write-time
  LLM call. Asks: *does claim extraction add anything over retrieving the
  transcript itself?*
- **`notes` — LLM-written session notes**, the harness-memory pattern (a
  "summarise each conversation into a notes file" agent). Every session
  summarised once by the same Sonnet 5 that wrote the particles (median
  note 2.4k characters), notes embedded, top-10 notes handed to the
  answerer. Asks: *does Particles' epistemic layer beat plain distillation?*

**Retrieval stage** (143 questions; same 7 abstention questions excluded):

| Memory | Recall@10 | Precision@10 | Items per session |
|---|---:|---:|---|
| `particles` | **0.940** | 0.668 | many (one per claim) |
| `chunks` | 0.912 | 0.652 | several (one per ≤1.5k chars) |
| `notes` | **0.950** | 0.171 † | exactly one |

† Precision@10 is **not comparable across memories of different item
granularity**: a `notes` memory has one item per session, so with one or
two labeled evidence sessions at most one or two of ten retrieved items can
ever be hits — 0.17 is close to that structural ceiling, not a defect. Read
recall across the three; read precision only within a memory.

**End-to-end QA** (150 questions, one answering model, `claude-sonnet-5`):

| Question type | n | `qa_particles` | `qa_chunks` | `qa_notes` | `qa_full_context` (reused) | `qa_no_memory` (reused) |
|---|---:|---:|---:|---:|---:|---:|
| **All** | **150** | **0.733** | **0.693** | **0.813** | **0.793** | **0.080** |
| knowledge-update | 23 | 0.739 | 0.783 | 0.870 | 0.826 | 0.087 |
| multi-session | 40 | 0.800 | 0.450 | 0.775 | 0.750 | 0.050 |
| single-session-assistant | 17 | 0.647 | 1.000 | 0.882 | 0.941 | 0.235 |
| single-session-preference | 9 | 0.333 | 0.889 | 0.778 | 0.667 | 0.000 |
| single-session-user | 21 | 0.762 | 0.810 | 0.810 | 0.905 | 0.048 |
| temporal-reasoning | 40 | 0.775 | 0.650 | 0.800 | 0.725 | 0.075 |

Read plainly, and published as-is:

- **Particles beats raw-transcript RAG** by 4 points overall — decisively
  where the answer is assembled across sessions (`multi-session` 0.80 vs
  0.45, `temporal-reasoning` 0.78 vs 0.65) and it loses on every
  single-session type, where a verbatim chunk carries the exact wording a
  claim paraphrases away. Claim extraction earns its keep for
  cross-session synthesis, not for lookup.
- **LLM-written session notes beat Particles by 8 points — and beat the
  full-context baseline by 2.** Distillation of the *whole session* into
  notes, retrieved at session granularity, is the strongest memory in this
  table on this benchmark, and it is the memory pattern an agent harness
  already ships. On recall-style question answering, Particles' epistemic
  layer (claim granularity, provenance, confidence, reconciliation) does
  not show up as answer accuracy; only `multi-session` (0.80 vs 0.78) is a
  Particles edge, and it is inside sampling noise at n=40.
- **The read-time context is not equal**, and that is part of the story:
  the ten retrieved particles are ~1.3–2k characters (mean claim ~90
  characters plus date and subjects), ten chunks up to 15k, ten notes ~24k
  (median). The notes memory hands the answerer roughly ten times the
  context Particles does — still ~5 % of the haystack, which is why it
  beats full context — so the table above compares *memories as
  configured*. The budget-matched arm is below.
- Cost: chunks ≈ US$2 (150 answer + judge calls, no write-time calls);
  notes ≈ US$70 (7,123 session notes: 3,088 through the Batches API,
  ~4,000 at full price after the batch queue stalled — every batched note
  is cached beside the checkpoints, so a re-run pays nothing). 84 notes
  (1.2 %) were truncated at the 2,048-token cap and 4 (0.06 %) failed to
  write (those sessions were absent from that question's memory —
  disclosed in the report's quality notes).

### Budget-matched arm — the same comparators at Particles' context size

The two comparators were re-run with the answerer's context clamped to
**500 tokens (~2,000 characters)** — the same clamp the particles path
exposes (`--context-budget`): items are appended in rank order
until the next would exceed the budget, the first always kept, so a `notes`
memory keeps its top note (median 2.4k chars — slightly *more* than the
particles context), a `chunks` memory one or two chunks. Retrieval and
`top_k` are untouched (recall is identical to the unclamped runs);
`selection.context_budget_tokens=500` marks the reports. Reports of record:
[`chunks @500`](benchmarks/longmemeval-s150-chunks-budget500-2026-08-17.json),
[`notes @500`](benchmarks/longmemeval-s150-notes-budget500-2026-08-17.json).
Cost ≈ US$3 (answer + judge only; every note came from the cache).

| Question type | n | `qa_particles` (~1.3–2k chars) | `qa_chunks` @ 2k | `qa_notes` @ 2k | `qa_chunks` unclamped | `qa_notes` unclamped |
|---|---:|---:|---:|---:|---:|---:|
| **All** | **150** | **0.733** | **0.293** | **0.393** | 0.693 | 0.813 |
| knowledge-update | 23 | 0.739 | 0.391 | 0.391 | 0.783 | 0.870 |
| multi-session | 40 | 0.800 | 0.200 | 0.125 | 0.450 | 0.775 |
| single-session-assistant | 17 | 0.647 | 0.412 | 0.882 | 1.000 | 0.882 |
| single-session-preference | 9 | 0.333 | 0.333 | 0.556 | 0.889 | 0.778 |
| single-session-user | 21 | 0.762 | 0.429 | 0.714 | 0.810 | 0.810 |
| temporal-reasoning | 40 | 0.775 | 0.200 | 0.250 | 0.650 | 0.800 |

Read plainly: **at equal read-time context, Particles wins by a wide
margin** — 0.733 against 0.393 for session notes and 0.293 for transcript
chunks, and the clamp is if anything generous to the comparators (a single
note is longer than the whole particles context). The two comparators lose
almost everything on the cross-session types (`multi-session` 0.20 / 0.125,
`temporal-reasoning` 0.20 / 0.25 against 0.80 / 0.78): one session's note
or chunk cannot carry an answer assembled from several sessions, while ten
claims from up to ten sessions can. Notes hold up on the single-session
types (0.88 / 0.71 on assistant / user), where the top note *is* the
answer's session and its ~2.4k characters carry the fact verbatim.

So the two tables together say one thing precisely — and it is the claim
this page supports: **at a fixed read-time context budget of ~2k
characters, Particles answers 1.9× as many LongMemEval questions correctly
as LLM-written session notes (0.733 vs 0.393) and 2.5× as many as RAG
over the raw transcript (0.733 vs 0.293).** Its edge is **information
density** — the most answer per read-time token — and its deficit is
coverage: given a much larger budget, whole-session distillation
recovers what claim extraction paraphrased away and then some. Which memory
is "better" depends on the read-time budget the agent can afford; on this
benchmark the crossover lies somewhere between ~2k and ~24k characters of
context, and pinning it (a sweep of `--context-budget` on the notes path)
is the natural next run.

## Two measurement families — never merged

Every run measures four conditions and reports them in **two separately
labeled families**. Conflating them is the endemic dishonesty mode of the
memory-benchmark space ("our memory retrieves the right session 85% of the
time" quoted as "answers correctly 85% of the time"), so the separation is
structural: the report model has no aggregate score field, and the renderer
has no way to merge the sections.

**Retrieval stage** — a property of the particle store and its ranker,
saying nothing about answer accuracy:

| Condition | What it measures |
|---|---|
| `retrieval` | Evidence-session **Recall@k** and **Precision@k** of the store's top-k query result, scored by mapping each retrieved particle through its provenance chain (particle → corpus entry → URI-R → haystack session) against the dataset's labeled evidence sessions |

**Abstention questions are excluded here, with a disclosed count.** An
abstention variant (`*_abs`) has no evidence session by protocol — the
right answer is "you never told me" — so retrieval is unscoreable for it,
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

**End-to-end QA** — an answering LLM on top of (or instead of) the memory,
which can recover from bad retrieval or fumble good retrieval:

| Condition | What it measures |
|---|---|
| `qa_particles` | Accuracy of the answering model given the question + the top-k retrieved particles (claim text, subjects, dates) |
| `qa_full_context` | **The baseline that must not be buried**: the *same* model, same prompt scaffold, with the entire concatenated haystack instead of retrieved particles. Published either way — including if it wins |
| `qa_no_memory` | The same model with the question only — the parametric-guessing / abstention floor |

Conditions ii–iv use **one answering model pinned to one resolved model
id**; the runner refuses to run a QA condition set whose resolved models
differ. If a baseline condition was skipped, its row renders as `not run` —
there is no flag to omit it, so a partial comparison is always visibly
partial.

**A call that produced no verdict is excluded from the accuracy denominator,
with a disclosed count.** If the answer or the judge call yields no usable
reply, that question is not scored wrong for that condition — it is scored
not at all, counted, and named in the table. The reason is asymmetry:
`qa_full_context` sends ~115k tokens per call and `qa_no_memory` sends three
lines, so any shared failure rate would land almost entirely on the baseline,
weakening it with transport noise rather than with anything about the memory.
The exclusion is disclosed per condition and **split by cause**, because the
two mean opposite things to a reader:

- **output-budget** — the reply carried no text within `max_tokens`. An
  extended-thinking model spends its thinking from the same budget, so this is
  a configuration error on our side; the table says so, and the fix is to
  raise the cap and re-run those questions. It is deliberately not retried,
  since an identical call at an identical cap reproduces it.
- **infra** — the call still failed after its retries.

The table also states, before any of this, whether the full-context baseline
*fits* the answering model's context window on the variant being run. A run
whose haystack would overflow is refused rather than reported: an overflowing
baseline is not a weaker baseline, it is a destroyed one, while the
question-only condition sails through untouched. This is the standing
precondition on the larger `m` variant.

Both rules were added in v1.137.1 and bind future runs. **No number on this
page changed**: the run below carries zero failed answer or judge calls (see
its run notes), so there was nothing to restate.

## Judge deviation — read before comparing

Answers are scored by an LLM judge following the dataset's
per-question-type autoeval protocol, **ported to an Anthropic judge**
(routed through the `llm.benchmark` purpose) rather than the paper's OpenAI
judge. Abstention-variant questions score per the dataset protocol (credit
for declining to answer). Consequence: **numbers on this page are
comparable within the table — same judge, same protocol, same selection —
not across leaderboards.** An OpenAI-judge protocol-fidelity option is
deferred until cross-leaderboard comparability becomes a requirement.

## Subset labeling discipline

Any run over fewer than all questions is a **subset run**, and the table
header must say so — including the full selection tuple that makes it
reproducible: dataset revision, variant, sample seed, strata (question
types), limit, resolved answer/judge model ids, the resolved **extraction
model id** and **embedding model id** (the store's contents are a function
of the first and the ranking of the second — two runs that differ on either
are different pipelines, not comparable), `top_k`, and a snapshot of the
pipeline thresholds in effect. The answer, extraction, and judge model
resolutions are each pinned mid-run by refusal — a drift aborts the run
rather than silently mixing pipelines. Two runs with the same recorded
tuple are comparable; anything else is disclosed drift. Subset and full-run
numbers are never mixed in one table.

## Reproducing

```bash
# Cost preview only — no LLM call is made
particles benchmark memory --estimate

# Dev loop (defaults: 10 questions, s variant, seeded stratified selection)
particles benchmark memory

# The publishable runs (operator-invoked, never inside a test suite)
particles benchmark memory --limit 150 --variant s --format json --output report.json
particles benchmark memory --all --variant s --format json --output report.json

# The comparator memories over the same selection (reuse the particles run's
# qa_full_context / qa_no_memory columns — same tuple, same calls)
particles benchmark memory --limit 150 --variant s --memory chunks --no-baselines --format json --output chunks.json
particles benchmark memory --limit 150 --variant s --memory notes --no-baselines --format json --output notes.json
```

The dataset (LongMemEval v1 cleaned, MIT-licensed, ~3 GB) is downloaded on
demand from HuggingFace at a pinned revision with SHA-256 verification and
cached under `~/.particles/benchmark/longmemeval/` — never vendored into
the repository. Answering routes through the `llm.benchmark_answer`
config purpose; the judge through `llm.benchmark`. Each question runs in an
ephemeral scratch store, so a benchmark run never touches a user store.

## Relationship to the extractor benchmarks

The `particles extractor benchmark*` verbs — including the modality and
polarity variants — measure a single **extractor's** output against gold
particles. This page's benchmark measures the **whole pipeline** against
gold answers — a different system under test, reported under its own
`particles benchmark` verb group.

`particles extractor benchmark` additionally persists each run's report as
a JSON file under `benchmark.runs_dir` (default `~/.particles/benchmark/runs/`),
stamped with the resolved extraction provider:model pairing — the durable
raw series behind provider comparisons and calibration-drift analysis
. Pass `--no-save` for a throwaway run.

Extraction is a sampling process, so a single run is a single sample:
`--runs N` repeats each suite N times and reports each metric's mean, range
and standard deviation rather than one point estimate — the error bars a
provider comparison needs before it calls a gap real. Each pass still
persists its own report file. The cost is N× the LLM calls, so the repeat
path prints its projection first and asks before spending above
`benchmark.confirm_call_threshold` (`--estimate` prints and exits; `--yes`
pre-confirms). `--fail-on` is evaluated against the mean across runs.
A single run (the default) is unchanged and never gated.
