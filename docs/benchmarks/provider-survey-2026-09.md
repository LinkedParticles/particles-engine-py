# Provider survey: extraction quality and cost (2026-09)

**A dated snapshot, as of 2026-09-13.** This page resolves the
**2026-09-01 re-evaluation** that the
[2026-08 survey](provider-survey-2026-08.md) scheduled for itself, and adds the
thing that survey had no figures for at all: **measured dollar cost per run.**

Four configurations were run against the general extractor, **three times
each**, on the same suite and judge the 2026-08 table used. Two are the
Anthropic incumbents the re-evaluation is about; two are cheap models that did
not exist when the earlier survey was written.

It measures a single **extractor's** output against gold particles (the
`particles extractor benchmark*` family), not the
whole-pipeline agent-memory benchmark on the [Benchmarks](../benchmarks.md)
page. Different system under test, different verb group.

!!! danger "This is not a model leaderboard, and it must not be read as one"
    **The extraction prompt and the equivalence judge were both tuned against
    Claude Sonnet.** The 2026-08 survey already refused to treat
    `claude-opus-5`'s 0.69 as a capability verdict for exactly this reason: the
    prompt and judge do not credit a model that frames claims differently. That
    bias applies to **every** non-Sonnet row here, including both Fireworks
    rows, and it is not small or quantified.

    So these numbers measure **how well each model drives this extractor, on
    this prompt, under this judge**, a procurement question about our own
    pipeline. They are *not* a comparative statement about model capability,
    and a lower row is not a worse model. Publishing them as a ranking of
    vendors would be the same error this project criticises vendors for: a
    benchmark tuned to the author's incumbent, presented as a neutral
    comparison.

!!! note "A snapshot, not a live leaderboard"
    Per-run reports persist as JSON under `benchmark.runs_dir`
    (default `~/.particles/benchmark/runs/`), stamped with the resolved
    `provider:model` pairing. Those files are gitignored; **this page is the
    committed interpretation of them**, as the 2026-08 survey is of its own.
    Every individual run's numbers are in
    [the raw per-run appendix](#appendix-raw-per-run-values) so nothing here
    rests on a summary statistic alone. (The report JSONs beside this page are
    LongMemEval artifacts of record, a different harness; the provider survey
    deliberately commits interpretation rather than run files.)

## What was measured

Identical to the 2026-08 headline configuration, so the two pages' *method*
is comparable even where their sampling is not:

| Suite parameter | Value |
|---|---|
| Suite | `prose-article-seed-001` (general-extractor extraction quality) |
| Suite version | v0.2.0 |
| Cases | 4 |
| Required claims | 35 (across the 4 cases) |
| Judge | embedding cosine, threshold ≥ 0.80 |
| Extractor | `general-extractor` 0.14.0 |
| Runs per model | **3** (`--runs 3`) |
| Measured | 2026-09-13 |

**Three runs, not one.** The 2026-08 page warned that a single run is a single
sample and recorded a 0.88/0.75 recall swing on an identical fixture. Every
row here is a mean over three independent passes with its range and standard
deviation beside it (`--runs N`; no caching between passes: measuring sampling
variance is the point).

## Results: quality

Mean over 3 runs, with the observed range and sample standard deviation. Rows
are in matrix order, **not ranked**; see the warning above.

| Model | Route | Recall (mean) | range | sd | Precision (mean) | sd | ECE (mean) | sd | Emitted per run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `claude-sonnet-5` | Anthropic | **0.800** | 0.74–0.83 | 0.049 | **0.797** | 0.072 | **0.096** | 0.027 | 84, 84, 83 |
| `claude-haiku-4-5` | Anthropic | 0.676 | 0.66–0.71 | 0.033 | 0.742 | 0.040 | 0.177 | 0.035 | 74, 75, 76 |
| `deepseek-v4p1-flash` | Fireworks | **0.800** | 0.77–0.83 | 0.029 | 0.719 | 0.096 | 0.160 | 0.132 | 113, 99, 100 |
| `glm-5p3-flash` | Fireworks | 0.419 | 0.29–0.54 | 0.129 | 0.701 | 0.063 | 0.172 | 0.017 | 28, 75, 56 |

Recall is the fraction of the 35 required claims recovered; precision is the
fraction of emitted claims matching a required claim; ECE is expected
calibration error (lower is better). The Fireworks model ids are
`accounts/fireworks/models/<name>`.

**`glm-5p3-flash`'s 0.419 is not a quality measurement**; read
[the dropout finding](#glm-5p3-flash-returns-nothing-on-whole-cases-with-no-error)
before using that row for anything.

**Both Anthropic rows re-measured lower than their 2026-08 single-run
numbers** (sonnet-5 0.800 against 0.94, haiku-4-5 0.676 against 0.74) on the
same suite version, extractor version, judge, threshold, and token budget.
The old 0.94 lies **outside** the three-run range observed here (0.74–0.83).
Two causes are available and this run cannot separate them: the 2026-08 figure
was a single sample and single samples on this suite run optimistic, and
`claude-sonnet-5` is not the same served model it was in August (the tokenizer
changed under it). Treat the 2026-08 column as a point estimate of unknown
position in its own distribution, not as a regression baseline.

## Results: cost

**Measured, not projected**: every figure below comes from the token counts the
providers returned on the 12 calls each row actually made. Anthropic cache
tiers are priced as billed (writes 1.25×, reads 0.10× the input rate), which is
why the input column exceeds the uncached prompt for `claude-sonnet-5`.

**Pricing disclosure.** `claude-sonnet-5` is priced at **$2/$10 per MTok**,
the price Anthropic's pricing page now states became the standard price, with
the increase to $3/$15 that had been scheduled for 2026-09-01 cancelled
(verified 2026-09-17). An earlier revision of this page priced the row at
$3/$15, the increase the 2026-08 survey had been told to expect; every
dollar figure and relative multiple below has been recomputed at $2/$10 from
the same measured token counts, which are unchanged.

| Model | Price in/out per MTok | Budget | Calls | Input tok | Output tok | **$ / run** | $ / 3 runs | Relative |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `claude-sonnet-5` | $2 / $10 | 16 384 | 12 | 46 749 | 80 639 | **0.2873** | 0.8619 | 1.0× |
| `claude-haiku-4-5` | $1 / $5 | 16 384 | 12 | 34 140 | 42 723 | **0.0826** | 0.2478 | 3.5× cheaper |
| `deepseek-v4p1-flash` | $0.22 / $0.66 | 32 768 | 12 | 38 379 | 87 065 | **0.0220** | 0.0659 | **13.1× cheaper** |
| `glm-5p3-flash` | $0.15 / $0.50 | 32 768 | 12 | 31 914 | 208 055 | **0.0363** | 0.1088 | 7.9× cheaper |

**Cost per unit of recall**: cost per run divided by mean recall, the figure
that makes the quality/price trade explicit:

| Model | Recall | $ / run | **$ per run per unit recall** |
|---|---:|---:|---:|
| `claude-sonnet-5` | 0.800 | 0.2873 | **0.359** |
| `claude-haiku-4-5` | 0.676 | 0.0826 | **0.122** |
| `deepseek-v4p1-flash` | 0.800 | 0.0220 | **0.028** |
| `glm-5p3-flash` | 0.419 | 0.0363 | 0.087 *(see dropout finding)* |

Two observations, both narrow and both about *this* harness:

- **`deepseek-v4p1-flash` matched `claude-sonnet-5`'s mean recall exactly
  (0.800 vs 0.800) at 1/13.1 of the cost per run**, with a *tighter* recall
  spread (sd 0.029 vs 0.049). It is worse on precision (0.719 vs 0.797) and
  much worse on calibration stability (ECE sd 0.132 vs 0.027; its three runs
  were 0.310, 0.111, 0.060). It also emits far more candidates per run
  (99–113 vs 83–84), which is where the precision goes.
- **The cheap tier is not uniformly cheap in practice.** `glm-5p3-flash` has
  the lowest list price of the four but costs *more* per run than
  `deepseek-v4p1-flash`, because it spends 2.4× the output tokens (208 055 vs
  87 065) on reasoning. List price per MTok did not predict cost per run here;
  only measurement did.

### Projected versus actual

The pre-spend gate (`--estimate`) and what was actually billed:

| | Projected (harness `--estimate`) | Actual (measured) |
|---|---|---|
| Extraction calls per model | ≥ 12 | 12 |
| Input tokens per model (3 runs) | ~8 399 | 31 914 to 46 749 |
| Dollar cost | **not reported** | $0.0659 to $0.8619 per model |
| **Whole matrix** | — | **$1.28** |

Total spend on the matrix was **US$1.28**, against a $20 ceiling, so the
projection never needed escalating. Two gaps in the estimate are worth
recording, because both make it optimistic:

- **It does not report money at all.** It reports calls and input tokens; the
  dollar figures on this page came from instrumenting the provider responses
  for this survey, since the report model carries no token usage (its schema is
  frozen). Converting the projection to currency needs a
  per-model price table the SDK does not have.
- **It undercounts input by roughly 3.8×.** It derives tokens from the source
  bytes and ignores the extraction system prompt, which is the larger half
  (33 556 of 42 124 prompt characters per run). And it cannot project output
  tokens, which are **94 % of the bill** for `claude-sonnet-5` and **96 %** for
  `glm-5p3-flash`. The estimate is a useful floor on call volume and close to
  useless as a cost forecast.

## The 2026-09-01 re-evaluation, resolved

The 2026-08 survey adopted `claude-sonnet-5` for extraction during its
introductory pricing window and scheduled this check for the day list pricing
and a new tokenizer landed. **One landed and one did not.** The tokenizer
changed. The price did not: Anthropic's pricing page states that the $2/$10
introductory price became the standard price and that the increase to $3/$15
scheduled for 2026-09-01 will not occur (verified 2026-09-17; an earlier
revision of this page assumed it had). So the only cost change to account
for is the tokenizer. The arithmetic it asked for:

**The tokenizer change is real and larger than the ~30 % the earlier page
estimated.** Measured exactly and free of sampling noise with the token-counting
endpoint, over the byte-identical extraction prompt for one run of this suite
(42 124 characters):

| Model | Input tokens for the identical prompt | Chars/token | vs `claude-sonnet-4-6` |
|---|---:|---:|---:|
| `claude-sonnet-5` | 15 599 | 2.70 | **1.369×** |
| `claude-sonnet-4-6` | 11 396 | 3.70 | 1.000× |
| `claude-haiku-4-5` | 11 392 | 3.70 | 0.9996× |

So the same text costs **+36.9 % more tokens** on `claude-sonnet-5` than on
`claude-sonnet-4-6`, and `claude-haiku-4-5` is on the older tokenizer (matching
sonnet-4-6 to within 4 tokens in 11 396). Both sides of the bill inflate (a
re-tokenized output is billed the same way), so the multiple applies to the
whole call, not just the prompt.

Two consequences follow, and they are the answer to the question the earlier
page posed:

- **Against its own adoption-era price, `claude-sonnet-5` costs ~1.37× what it
  did when it was chosen**: the tokenizer effect alone, since the price is
  unchanged at $2/$10. The 2026-08 page's "~1.3×" tokenizer estimate was right
  in direction and slightly low in magnitude; its projected ~2.05× compounded
  that with a 1.5× price rise that never happened.
- **Against `claude-sonnet-4-6`, which stays at $3/$15, `claude-sonnet-5`
  costs ~0.91× as much for identical work**: 1.369× the tokens at 2/3 the
  per-token price. Per token it is the dearer tokenizer; per unit of work it
  is slightly the cheaper model.

**Is `claude-sonnet-5` still the right default?** On this harness it is still
the **quality** leader on the axes it led on in August: it has the best
precision (0.797) and by a wide margin the best calibration (ECE 0.096, sd
0.027; the next best mean is 0.160 at sd 0.132), and it is the only row that
never dropped a case. What changed is that it is **no longer the recall
leader** (`deepseek-v4p1-flash` ties it at 0.800 for 1/13.1 the cost), and its
cost per unit recall is **13.1× the cheapest row and 2.9× `claude-haiku-4-5`**.

That is the measurement. **The decision is not made here**, for two reasons
stated plainly:

1. **No quality threshold has been set.** Nothing on this page says any model
   is sufficient or insufficient for production extraction, because the bar it
   would be measured against does not exist yet. Recall is presented beside
   cost so that setting the bar and reading off the consequence are one step.
2. **No configured default was changed by this run.** `llm.extraction` is
   untouched; every row was produced by a throwaway config file. Changing the
   routing is an owner decision this page only informs.

### What the owner now needs to decide

Three questions, in the order that makes them answerable:

1. **What is the minimum acceptable recall for production extraction**, and,
   separately, the minimum acceptable **precision** and **calibration**?
   Calibration is the axis where the incumbent's lead is largest and where a
   cheap substitute degrades most, and a stated confidence is immutable once a
   particle is created, so an ill-calibrated extractor's
   mistakes are permanent in a way a low-recall one's are not.
2. **Given that bar, is the ~13.1× cost premium for `claude-sonnet-5` over
   `deepseek-v4p1-flash` buying anything the bar requires?** On these numbers
   it buys precision (+0.078), calibration stability, and zero case dropouts,
   not recall.
3. **Should a candidate substitute be re-measured on a fairer harness first?**
   Every non-Sonnet number here is depressed by an unknown amount by the
   Sonnet-tuned prompt and judge. Any decision to switch on the strength of a
   cheap row should be preceded by a prompt re-tune for that model, exactly as
   the 2026-08 page concluded for `claude-opus-5`.

A fourth option this run makes visible and cannot price: **`claude-haiku-4-5`
batched** is $0.50/$2.50, halving its already-3.5×-cheaper row, and it is on
the cheaper tokenizer. It was not measured batched here.

## Method notes

The 2026-08 method notes still hold. Four findings are new, and three of them
are the difference between a row that runs and a row that silently misreports.

### The token budget can no longer be uniform across routes

The 2026-08 survey's answer was a single setting, `extraction.max_tokens:
16384`, which "cleared it for every model in the trial." That no longer holds
in **either** direction, so the two families here ran at different budgets,
not by choice:

- **The Fireworks reasoning models need more than 16 384.** At 16 384,
  `glm-5p3-flash` truncated mid-JSON (`finish_reason=length`, the parser
  reporting "Unterminated string") on 2 of 4 cases in the first pass. Both
  Fireworks rows therefore ran at **32 768**, where neither truncated once in
  12 calls. The measured output volume says why: `glm-5p3-flash` averages
  17 338 output tokens per call, comfortably past the old ceiling before it
  begins the answer.
- **The Anthropic native adapter cannot accept 32 768 at all.** It is a
  non-streaming client, and the SDK refuses a non-streaming request whose
  `max_tokens` implies more than ten minutes of work; the guard is
  `3600 × max_tokens / 128000 > 600`, i.e. a hard ceiling of **21 333
  tokens**. At 32 768 every Anthropic call fails with "Streaming is required
  for operations that may take longer than 10 minutes", and the failure is
  *not* loud in the metrics: the run completes, each case records an API-error
  quality note, and the report shows recall 0.000 with 0 emitted. Both
  Anthropic rows were therefore run at **16 384**, where they truncate nothing
  (0 of 12 calls).

So a budget that satisfies the Fireworks reasoning models is rejected outright
by the Anthropic route, and the harness has no per-route budget knob: the
setting is global. **This is a comparability caveat on the tables above**: the
Fireworks rows had twice the completion budget of the Anthropic rows. It
favours the Fireworks rows, and it could not be equalised without either
truncating them or breaking the Anthropic rows.

### `glm-5p3-flash` returns nothing on whole cases, with no error

The single most important finding on this page, and the reason its 0.419 recall
must not be read as a quality score. Per case-run:

| Run | web-article-001 | web-article-002 | web-essay-001 | web-interview-001 |
|---|---:|---:|---:|---:|
| 1 | **0** | 27 | **0** | 1 |
| 2 | 23 | **0** | 26 | 26 |
| 3 | **0** | 31 | **0** | 25 |

**5 of 12 case-runs emitted zero particles** (the other three models: 0 of 12
each). These are not truncations, parse failures, or API errors; there were
none of any kind in the run, and the calls returned HTTP 200 having spent
9 660–22 188 output tokens. The model reasoned at length and returned an empty
claim list, and **nothing in the harness flags it**: a zero-emission case
produces no quality note, so the run reports a plausible-looking recall number
built partly from silence.

On the 7 case-runs where it returned a non-empty result it matched **44 of 63
required claims (0.70)**, a different-looking model entirely. The headline
0.419 is that 0.70 diluted by five empty answers. Neither figure is a quality
verdict; together they say the failure is one of **reliability**, and that a
mean over runs is the wrong summary for it.

This is also a gap in the harness worth knowing independently of any model: an
empty extraction is indistinguishable in the report from a case the extractor
legitimately found nothing in.

### Prompt caching is on, and it moves the token accounting

`llm.prompt_cache` is enabled by default, and it engaged for
`claude-sonnet-5` only: 21 659 cache-read and 1 969 cache-write tokens across
its 12 calls, against 0 for `claude-haiku-4-5`. Cache reads are *excluded* from
the `input_tokens` the API reports, so reading that field alone makes sonnet-5
look like it consumed **fewer** input tokens than haiku-4-5 (23 121 vs 34 140)
on a byte-identical prompt, the exact opposite of the truth. Summing all three
tiers gives 46 749 vs 34 140, restoring the 1.37× the tokenizer measurement
predicts. Any cost analysis that reads `input_tokens` without adding the cache
tiers will understate a cached Anthropic model and mis-rank it against an
uncached one.

### The OpenAI-compatible provider settings, unchanged

Both Fireworks rows used the 2026-08 settings verbatim, and they remain
necessary:

```yaml
llm:
  providers:
    fireworks:
      base_url: https://api.fireworks.ai/inference/v1
      max_tokens_param: max_completion_tokens
      send_temperature: false
      structured_output: strict
      timeout_seconds: 300
extraction:
  max_tokens: 32768        # 16384 truncates glm-5p3-flash; see above
```

One correction to the earlier page's example: **`extraction.timeout_seconds`
is not a field on the config model** and is silently ignored. The timeout that
takes effect for an OpenAI-compatible route is the provider entry's own
`timeout_seconds`.

## Model availability

**Both Fireworks ids were verified live against the serverless model list on
2026-09-13** before any run: `accounts/fireworks/models/deepseek-v4p1-flash`
and `accounts/fireworks/models/glm-5p3-flash`. The 2026-08 page records two
plausible-looking ids that do not exist (`glm-5p2-flash`, `qwen3p7-max`); check
the live list rather than pattern-matching a name.

**Qwen 3.8 was deliberately excluded, not overlooked.** Its Flash variants
(`qwen3p8-flash-next-fp8`, `-nvfp4`, `qwen3p8-27b`) carry **no serverless
pricing** and would require a dedicated deployment, which is a different cost
model that cannot be compared per-MTok against the rows above. The only
serverless-priced 3.8 is `qwen3p8-max` at **$2/$6**: real, but not a cheap
option, and so outside this matrix's question. Revisit if a serverless Flash
tier is priced.

## How to reproduce

One config file per model, one CLI verb, as in the 2026-08 page. Route
`llm.extraction` at the model under test, then:

```bash
PARTICLES_CONFIG=survey-model.yaml uv run particles extractor benchmark \
    general-extractor --suite prose-article-seed-001 --runs 3 --estimate
```

`--estimate` prints the projected call volume and exits without calling; drop
it (and pass `--yes`) to run. Mind the two route-dependent budgets above:
32 768 for a Fireworks reasoning model, at most 21 333 for the Anthropic native
adapter. API keys are `PARTICLES_LLM_API_KEY_<NAME>` in the environment, never
in the config file.

**The dollar figures on this page are not reproducible from the CLI alone.**
The report model carries no token usage, so cost was measured by wrapping the
two provider transports and summing the usage each response returned. Anyone
repeating this should do the same and price the Anthropic cache tiers
separately.

## Appendix: raw per-run values

Every individual pass, so no claim above rests only on a mean. Each row is the
three runs in order.

| Model | Recall | Precision | ECE | Emitted | Input tok (in + cache-w + cache-r) | Output tok |
|---|---|---|---|---|---|---|
| `claude-sonnet-5` | 0.743, 0.829, 0.829 | 0.762, 0.750, 0.880 | 0.120, 0.102, 0.066 | 84, 84, 83 | 23 121 + 1 969 + 21 659 | 80 639 |
| `claude-haiku-4-5` | 0.657, 0.714, 0.657 | 0.730, 0.787, 0.711 | 0.179, 0.141, 0.211 | 74, 75, 76 | 34 140 + 0 + 0 | 42 723 |
| `deepseek-v4p1-flash` | 0.771, 0.800, 0.829 | 0.619, 0.727, 0.810 | 0.310, 0.111, 0.060 | 113, 99, 100 | 38 379 + 0 + 0 | 87 065 |
| `glm-5p3-flash` | 0.286, 0.543, 0.429 | 0.643, 0.693, 0.768 | 0.191, 0.158, 0.166 | 28, 75, 56 | 31 914 + 0 + 0 | 208 055 |

Per-case emitted counts, which is where `glm-5p3-flash`'s spread comes from.
The four cases are `web-article-001`, `web-article-002`, `web-essay-001`,
`web-interview-001` in order, and each cell is run 1 / run 2 / run 3:

| Model | web-article-001 | web-article-002 | web-essay-001 | web-interview-001 |
|---|---|---|---|---|
| `claude-sonnet-5` | 19 / 19 / 20 | 26 / 25 / 26 | 14 / 18 / 15 | 25 / 22 / 22 |
| `claude-haiku-4-5` | 19 / 18 / 19 | 25 / 25 / 26 | 10 / 11 / 10 | 20 / 21 / 21 |
| `deepseek-v4p1-flash` | 24 / 21 / 20 | 31 / 31 / 29 | 25 / 25 / 24 | 33 / 22 / 27 |
| `glm-5p3-flash` | **0** / 23 / **0** | 27 / **0** / 31 | **0** / 26 / **0** | 1 / 26 / 25 |

Token counts are totals over each model's 12 calls; cost follows from them at
the prices in the cost table. All four rows: 12 of 12 calls completed, 0
truncated, 4 of 4 cases run in every pass.

## Cross-references

- The extraction-quality benchmark harness itself, whose report
  schema is frozen and therefore carries no token usage.
- Repeat runs and the mean-with-spread aggregate the tables above report, plus
  the estimate/confirm cost gate that runs before any spend.
- Named OpenAI-compatible providers, where adding a vendor is configuration,
  never code, and the completion-provider port they plug into.
- The per-particle provider stamp: every particle records the `provider:model`
  pairing that produced it.
- Prompt caching, whose tiers the cost table prices as billed.
- Immutable stated confidence, which is why calibration is a load-bearing axis
  in the decision above.
- The prior snapshot this page re-evaluates:
  [Provider survey (2026-08)](provider-survey-2026-08.md).
