# What a run costs: measured spend, the nightly floor, and the levers

**A dated snapshot, as of 2026-10-18.** Particles spends money in one place:
calls to a hosted language model. Embedding and retrieval run on your machine
and make no API call. This page gives what each kind of run has cost when it
was measured, what a night of scheduled maintenance costs when nothing new has
arrived, and what each setting that changes the bill saves.

Every dollar figure here comes from the token counts the provider returned on
the calls a run actually made, priced at list price. A figure that is an
estimate says so beside it, and § Not yet measured lists what has no
measurement and what run would supply it.

!!! danger "Model rows on this page are evidence about cost, not a ranking of vendors"
    Where this page compares models, it reports what each one cost and
    returned **on this project's extraction prompt and judge, which were both
    developed against Claude Sonnet**. Every other model is measured on ground
    tuned for a competitor, by an amount nobody has quantified. A cheaper row is
    a statement about what a configuration change would cost here, never a
    statement that one model or provider is better than another.

## The short answer

| Run | What it cost | Model | Measured |
|---|---|---|---|
| First audit of an agent's memory files: 96 files, 212,792 characters | **US$7.97** in 96 minutes, about 8 cents a file and 3.7 cents per 1,000 characters | `claude-sonnet-5` | 2026-09-25 |
| First audit of the same directory today: 114 files, 253,858 characters, default configuration | **US$6.11** in about 50 minutes, about 5 cents a file and 2.4 cents per 1,000 characters | `claude-sonnet-5-5` | 2026-10-18 |
| Nightly maintenance on a living store of 48,000 to 56,000 beliefs | **US$2.56 to US$3.81 a night**, five nights | `claude-sonnet-5-5`, probes on `claude-haiku-5-5` | 2026-10-12 to 10-16 |
| The same nights before the model change | US$6.32 to US$9.05 a night | `claude-sonnet-5`, probes on `claude-haiku-4-5` | 2026-10-03 to 10-10 |
| A night on which nothing new arrived | **US$0.00: no model calls**, on a 57,060-belief store, and on each of ten earlier such runs | any | 2026-10-18 on 1.192.24; 2026-08-27 to 09-25 |
| A query: retrieval | Nothing; it runs locally | none | by construction |
| A query: the written answer | **US$0.006 to US$0.019**, median about US$0.012; one question the relevance floor refused cost nothing | `claude-sonnet-5-5` | 2026-10-18, ten queries |
| The trip-versus-move benchmark, five runs | **US$0.21** and US$0.22 | `claude-sonnet-5-5` and `claude-sonnet-4-6` | 2026-10-14 |

The pattern behind the table: **extraction is the bill.** Reading a source
into claims is where nearly all of the money goes, and it is paid once per
version of a source. Maintenance is paid on what changed since the last run,
so a store that receives nothing costs nothing to maintain, and a store that
grows does not get more expensive to keep for that reason alone.

## Prices used

List prices in US$ per million tokens, read from Anthropic's pricing page on
**2026-10-18**. These are also the figures the CLI ships with
(`llm.price_per_mtok`); an operator entry for the same model replaces them.

| Model | Input | Output | Cache read | Used here for |
|---|---:|---:|---:|---|
| `claude-sonnet-5-5` | 2.00 | 10.00 | 0.10 (0.05×) | Extraction, second readings, the use judge, query answers, and every call of the 2026-10-18 audit; the default since 1.189.0 |
| `claude-sonnet-5` | 2.00 | 10.00 | 0.20 (0.10×) | The first audit and the nights before 2026-10-12 |
| `claude-sonnet-4-6` | 3.00 | 15.00 | 0.30 (0.10×) | The default before 1.189.0 |
| `claude-haiku-5-5` | 0.10 | 0.50 | 0.01 (0.10×) | Contradiction probes from 2026-10-09; prompts up to 100,000 tokens |
| `claude-haiku-4-5` | 1.00 | 5.00 | 0.10 (0.10×) | Contradiction probes before 2026-10-09 |

The Message Batches API takes **50 %** off input and output, and a five-minute
cache write costs 1.25× input. The nightly figures below were priced by
releases that charged every cache read at 0.10× input; `claude-sonnet-5-5` bills
its cache reads at 0.05×, so those figures are high by one or two cents a
night. The `claude-sonnet-5` price of
US$2/US$10 is the standard price: the increase to US$3/US$15 once scheduled
for 2026-09-01 did not happen.

## The first audit

`particles audit` reads an agent's memory directory into claims and reports the
contradictions, near-duplicates and stale entries it finds. On 2026-09-25 it
ran over 96 Claude Code memory files into a fresh store:

| | |
|---|---|
| Source | 96 files, 212,792 characters (about 53,000 tokens of source) |
| Extraction | 104 calls on `claude-sonnet-5` (8 of them retries), `extraction.max_tokens` 16,384 with a retry at 20,000, the budgets of that release |
| Contradiction checks | 374 calls on `claude-haiku-4-5` |
| Result | 1,916 beliefs; 92 files extracted in full, 4 in part, none lost |
| Time | 96 minutes, calls one at a time |
| **Billed** | **US$7.97**, the same to the cent as the change in the account's prepaid balance (US$20.82 to US$12.85) and as the audit's own usage line |

That is **US$0.083 a file, US$0.037 per 1,000 characters, and US$0.004 a
belief.** Today's defaults raise the output budgets to 32,000 and 64,000. A
budget is a cap, not a charge, so the larger budgets raise the ceiling of an
estimate and not the bill.

A budget set too small costs more, not less. An earlier run over the same 96
files at an 8,192-token budget cost about US$10 by the account balance and left
17 files empty or cut short, because a reply that hits its budget is billed in
full and its later claims are lost.

### Per file is the wrong unit for long documents

The cost of extraction follows how much a document asserts, which tracks its
length, not its file count. A memory note is short. A design document is not:
on 2026-10-04 the audit ran over fourteen of this project's own design records,
which came out at about **7 claims per 1,000 characters and about 400 output
tokens per claim**. Six of them, 13,000 to 22,000 characters each, took 13
extraction calls and wrote 317,000 output tokens and 776 claims, about US$3.31
of extraction on `claude-sonnet-5`, and the fourteen together cost about
US$8 to US$9. One 20,000-character record costs about what ten memory notes do.

The audit's estimate used to assume a flat 7,000 output tokens per call, which
fits short notes and priced these records about three times low. Since 1.174.2
it prices each call by the characters it reads (about two output tokens per
character plus 1,600 per call on `claude-sonnet-5`, scaled per model) and
leaves out files already extracted. Both measured runs above fall inside the
range it now prints.

### On today's default model

`claude-sonnet-5-5` became the compiled default in 1.189.0. On 2026-10-18 the
audit ran over the same memory directory as it stands today, into a fresh
store, with no configuration at all, so every model call went to the default:

| | |
|---|---|
| Source | 114 files, 253,858 characters |
| Extraction | 114 calls on `claude-sonnet-5-5`, no retries; 433,000 output tokens |
| Other calls | 228 contradiction checks, 137 subject-resolution calls, 3 second readings, all on `claude-sonnet-5-5` |
| Result | 1,512 beliefs; all 114 files extracted in full |
| Time | about 50 minutes |
| **Cost** | **US$6.11** at list price, US$5.06 of it extraction, by the audit's own usage line |

That is **US$0.054 a file, US$0.024 per 1,000 characters, and US$0.004 a
belief.** The estimate printed before it said US$3.46–6.89 expected, with a
US$45 ceiling, and the bill landed near the top of that range.

The two audits are not the same experiment, so the comparison is per
character, not per run. The directory has changed since September (114 files
and 254,000 characters against 96 and 213,000), and the September run sent its
contradiction checks to `claude-haiku-4-5` under the store owner's
configuration, where this one sends everything to the default model. Per 1,000
characters the bill fell from 3.7 cents to 2.4. `claude-sonnet-5-5` wrote about
286 output tokens per belief here; on the extraction benchmark it wrote about
half the output of `claude-sonnet-5` for the same work (59,760 against 111,950
tokens over the same 18 calls).

## The nightly cycle on a living store

`particles memory consolidate` runs the store's maintenance once a night:
extract what is queued, reconcile updates, re-read flagged contradictions,
mine which beliefs sessions used, and once a week a census of the whole store.
Every pass that calls a model works on what changed since the previous run,
under a per-pass call cap, and the extraction and use-judge passes ride the
Message Batches API at half price.

These are the project's own store, the maintainer's daily Claude Code memory,
read from each night's run log ("LLM usage … ≈ $X at list price") and checked
against the models each night's run record names. Every night ran 1.168.8 or
later (its run record carries the budget block that release added); the five
nights on `claude-sonnet-5-5` ran 1.182.2 through 1.191.4.

| Night | Extraction model | Extraction calls | Extraction output tokens | Probe model | Active beliefs | **Night's cost** |
|---|---|---:|---:|---|---:|---:|
| 2026-10-03 | `claude-sonnet-5` | 66 | 954k | `claude-haiku-4-5` | 39,571 | **US$7.28** |
| 2026-10-04 | `claude-sonnet-5` | 76 | 1.3M | `claude-haiku-4-5` | 41,426 | **US$7.33** |
| 2026-10-06 | `claude-sonnet-5` | 53 | 1.3M | `claude-haiku-4-5` | 43,656 | **US$9.05** |
| 2026-10-08 | `claude-sonnet-5` | 58 | 1.2M | `claude-haiku-4-5` | 45,697 | **US$7.96** |
| 2026-10-10 | `claude-sonnet-5` | 53 | 862k | `claude-haiku-5-5` | 47,809 | **US$6.32** |
| 2026-10-12 | `claude-sonnet-5-5` | 49 | 235k | `claude-haiku-5-5` | 48,213 | **US$2.65** |
| 2026-10-13 | `claude-sonnet-5-5` | 57 | 293k | `claude-haiku-5-5` | 48,909 | **US$3.19** |
| 2026-10-14 | `claude-sonnet-5-5` | 65 | 418k | `claude-haiku-5-5` | 49,233 | **US$3.81** |
| 2026-10-15 | `claude-sonnet-5-5` | 52 | 340k | `claude-haiku-5-5` | 55,578 | **US$2.95** |
| 2026-10-16 | `claude-sonnet-5-5` | 48 | 236k | `claude-haiku-5-5` | 55,831 | **US$2.56** |

The census ran on 10-08 and 10-15. The run of 2026-10-09 is left out because
its release had no price for `claude-haiku-5-5` and printed tokens only.

**What moved the cost was the model, not the work.** Every night in the table
extracted the same thing: 20 queued session transcripts (the per-night cap)
plus the three to five local files that had changed, about 50 to 65 extraction
calls. On 2026-10-12 the store's configuration moved extraction and the second
readings from `claude-sonnet-5` to `claude-sonnet-5-5`, and extraction output
fell from 0.86M–1.3M tokens a night to 0.24M–0.42M. Priced from those counts,
extraction went from US$4.80–7.21 a night to US$1.50–2.54, or from 8 to 13
cents per extraction call to 3 to 4 cents. Moving the contradiction probes to
`claude-haiku-5-5` on 2026-10-09 took that pass from up to about US$1.05 a
night to about US$0.10.

The other cost work of those weeks is on both sides of the line: the remembered
verdicts that stop the cycle re-asking a pair it has already cleared, the weekly
census, and extraction of only the appended tail of a transcript were all in
the releases every night in the table ran. Faster retrieval over a large store
shipped on 2026-10-17, after the last night here, and changes query time, not
model calls.

The cheaper model is not free on every axis. On the extraction benchmark
`claude-sonnet-5-5` recovered fewer of the required claims than
`claude-sonnet-5` (recall 0.707 against 0.827; see the
[2026-10 provider survey](provider-survey-2026-10.md)). The default moved to it
after separate measurements of the update and contradiction checks.

**Store size does not drive the nightly cost.** Over these two weeks the store
grew from about 40,000 to 56,000 active beliefs and the cost fell. What a night
costs follows what arrived since the previous one. This store has carried a
queue of 300 to 430 session transcripts waiting for extraction since July,
drained 20 a night, so every night in the table is a working night.

For a larger sample of small stores: across the 150 freshly built stores of the
LongMemEval run, one maintenance cycle made 36,627 model calls for about
US$10, **about seven cents per store per cycle**, four fifths of it the census
([Benchmarks](../benchmarks.md)).

## The scheduled-maintenance floor: a quiet night

The floor is what a night costs when nothing new has arrived. **It measured
zero: US$0.00, no model calls.**

The measurement, on 2026-10-18 with release 1.192.24: a copy of the store
above (57,060 active beliefs), with its own configuration and with the queue of
waiting transcripts switched off, so that the only new material was what had
changed since its last run two days earlier. The cycle ran twice, back to back.

| | First run | Second run (the quiet night) |
|---|---|---|
| What it found | 3,146 beliefs changed since the previous run, 42 sessions to mine | 3 beliefs, the ones the first run itself wrote |
| Model calls | 323 | **0** |
| Cost | US$1.17 (the use judge US$0.95 of it) | **US$0.00** |
| Wall time | 27 minutes | 25 seconds |

The second run did every pass and found nothing to ask a model: each pass that
can call one reads only what changed since the previous run, the census reads
what changed since the previous census (and was not due), and the use judge
mines only sessions that happened since. The first run is what two days of
use cost to maintain when no new source arrives.

The store's own records agree. On **ten scheduled runs between 2026-08-27 and
2026-09-25 in which no new belief entered it, every pass ran and none called a
model**, on releases from before the cycle priced its own usage and before
several of today's passes existed.

## A query

Retrieval embeds the question with a local model (`all-MiniLM-L6-v2`) and
scores it against the store's claims on your machine: no API call, at any
store size. The deterministic query modes (`--group-by` and the other
aggregates) make no model call at all.

A written answer is one call to the answering model (`llm.query_response`),
which reads the retrieved claims and the question. Ten questions were run on
2026-10-18 against a copy of the 57,060-belief store above, at the default of
40 claims, on `claude-sonnet-5-5`, with each call metered. They were the four
questions the store's history shows were actually put to it, then six of its
owner's own questions taken in a fixed order:

| | Measured |
|---|---|
| Model calls | 9 (one question fell below the relevance floor and was answered with no model call) |
| Input per call | 2,427 to 3,410 tokens |
| Output per call | 104 to 1,350 tokens |
| **Cost per answered query** | **US$0.006 to US$0.019, median about US$0.012** |
| Ten queries | US$0.10 |

Output decides the spread: a one-line answer cost under a cent, and a long one
two cents. At 10 claims the retrieved context measured 542 tokens on
LongMemEval, against 2,149 at 40, and answered as well
([Benchmarks](../benchmarks.md)). Latency is measured separately:
[query latency](query-latency.md).

For bulk ingestion, the LongMemEval stores give a rate: building them cost
about **US$2.70 per question** on `claude-sonnet-5` through the batch API, for
a chat history of about 167,000 tokens each, or about US$16 per million tokens
of conversation read into claims.

## The levers, and what each one saves

| Lever | Setting | What it saved, where measured |
|---|---|---|
| **Model per purpose** | `llm.extraction`, `llm.semantic_lint`, `llm.verification`, `llm.query_response`, and the others; each falls back to `llm.default` | The nightly cost above: US$6.32–9.05 to US$2.56–3.81 a night. Contradiction probes on `claude-haiku-5-5`: up to about US$1.05 to about US$0.10 a night |
| **The batch API** | `llm.batch` | Half price on extraction and the use judge. Already applied in every nightly figure above |
| **Per-pass caps** | `consolidation.max_pending_entries`, `consolidation.max_update_probes`, `utility.mining.max_behavioural_calls`, `audit.max_contradiction_probes` | Bound every pass, so a night costs at most its caps. A capped pass says what it left for the next run |
| **The census cadence** | `consolidation.census.interval_hours` (default 168, weekly) | The census is the largest maintenance pass on a fresh store (four fifths of the LongMemEval cycle). Weekly instead of nightly pays it once in seven nights |
| **A dollar budget per run** | `consolidation.budget_usd`; `--max-usd` on the benchmark verbs | A pass whose estimate would cross the budget is skipped and named in the report; a batch chunk that would cross it is not sent |
| **Estimate before spending** | `particles audit --estimate`, `particles memory consolidate --dry-run`, `--estimate` on the benchmark verbs | Free: no deposit, no write, no model call. See the note below on how close they come |
| **No model at all** | `particles memory consolidate --structural-only` | Zero model calls. Runs the structural passes and says which were skipped |
| **A large enough output budget** | `extraction.max_tokens` | A budget too small cost more and lost claims (about US$10 against US$7.97 for the same 96 files) |
| **Retrieval depth** | `--top-k` | 10 claims answered as well as 40 on LongMemEval on a quarter of the context, 542 against 2,149 tokens |
| **A local model** | `provider: local` per purpose (an Ollama endpoint) | No API spend for the purposes routed there. See below for what has and has not been shown |

**How close the estimates come.** The audit's estimate is fitted to the two
measured audits above and contains both. The consolidation dry run plans pass 1
the way the extractor will read each snapshot: a transcript's new tail rather
than the whole of it, no call for a chunk whose claims the store already holds,
and nothing past the per-source call cap. It applies the batch discount where
the run would batch (extraction and the use judge), prices the extraction
prompt's cached prefix at the model's cache-read rate, and expects a transcript
to write about 0.4 times the output per character of a memory note or a design
record, since it states fewer claims. On the store above, on 2026-10-18, it
projected about US$4.88 for a night that would extract 20 queued transcripts,
US$3.06 of it extraction on 71 calls, against US$2.56–3.81 measured on the five
comparable nights before (extraction US$1.49–2.54). Before those changes the
same night read about US$17. Replayed over the snapshots each of those five
nights extracted, the extraction estimate came out 1.4 to 2.3 times what was
billed. Read the dry run as an upper bound within about a factor of two, and the
run's own usage line as the cost. `consolidation.budget_usd` decides with the
same estimate.

**Cheaper extraction models.** The two provider surveys measured what other
extraction models cost and returned on the same benchmark. In the
[2026-09 survey](provider-survey-2026-09.md), the cheapest model with full
recall matched `claude-sonnet-5` on recall (0.800 each) at about a thirteenth
of the cost per run, and was worse on precision (0.719 against 0.797) and on
calibration (expected calibration error 0.160 against 0.096). The
[2026-10 survey](provider-survey-2026-10.md) re-measured every row on a larger
suite and fitted a calibration to each, and the fit did not close that gap. A
stored confidence is never rewritten after a claim is created, so calibration
is the axis that kept extraction on Sonnet by default, and no cheaper
extraction preset ships. Routing extraction elsewhere is a configuration
change for an operator whose bar is different. Read both surveys with the
warning at the top of this page: the prompt and judge are tuned on Sonnet.

**Local models.** On LongMemEval, `qwen3:14b` running locally ranked six
read-side variants the way `claude-sonnet-5` did (Kendall's tau-b 0.828 against
a bar of 0.6), while scoring 0.15 to 0.19 lower in absolute accuracy
([rank agreement](local-rank-agreement-2026-10-12.md)). An exploratory sweep of
read-side settings can therefore run locally at no API cost, with the hosted
model needed only to confirm the few settings worth reporting. That result is
about ranking read-side variants on one benchmark. It says nothing about
extraction quality on a local model, which has not been measured to a standard
that would ship as a default.

## Not yet measured

| What | The run that measures it | Its cost, by the CLI's own estimate |
|---|---|---|
| Dense documents on `claude-sonnet-5-5` | The fourteen design records above, re-run on the new default | Not yet estimated |

## Reproducing

```bash
# Free: print what an audit of a memory directory would cost, deposit nothing
particles audit ~/.claude/projects/<project>/memory --estimate

# Free: price tonight's maintenance pass by pass, write nothing, call nothing
particles memory consolidate --dry-run

# Cap a run in dollars (config.yaml)
#   consolidation:
#     budget_usd: 3.00

# Every metered run ends with its measured cost, for example
#   LLM usage: 48 extraction calls (claude-sonnet-5-5), ... ≈ $2.56 at list price.

# The trip-versus-move benchmark: estimate, then a run capped per store
particles benchmark trip-vs-move --model claude-sonnet-5-5 --estimate
particles benchmark trip-vs-move --model claude-sonnet-5-5 --max-usd 0.50
```

The run of record for each nightly figure is the store's `CONSOLIDATION_RUN`
event, which carries the same token counts; `particles events list` reads it
back.
