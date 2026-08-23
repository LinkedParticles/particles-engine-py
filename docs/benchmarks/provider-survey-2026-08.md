# Provider survey — extraction quality (2026-08)

**A dated snapshot, as of 2026-08-04.** Between 2026-08-01 and 2026-08-04
the operator benchmarked **14 model configurations across 5 vendors** on the
general-extractor's extraction quality, plus a set of Fireworks pricing
checks. This page is the durable record of that survey: the numbers,
the method, and the operational verdict.

It measures a single **extractor's** output against gold particles (the
`particles extractor benchmark*` family) — not the whole-pipeline
agent-memory benchmark on the [Benchmarks](../benchmarks.md) page. Different
system under test, different verb group.

!!! note "This is a snapshot, not a live leaderboard"
    The raw per-run reports live as JSON under `benchmark.runs_dir`
    (default `~/.particles/benchmark/runs/`), stamped with the resolved
    `provider:model` pairing — the durable series behind provider
    comparisons and calibration-drift analysis.
    Those run files are gitignored; **this page is the committed
    interpretation of them.** Re-run the harness (see
    [How to reproduce / extend](#how-to-reproduce-extend)) to refresh the
    numbers against newer models.

## What was measured

All headline numbers are from the **`prose-article-seed-001`** suite at
**suite version v0.2.0**:

| Suite parameter | Value |
|---|---|
| Suite | `prose-article-seed-001` (general-extractor extraction quality) |
| Suite version | v0.2.0 |
| Cases | 4 |
| Required claims | 35 (across the 4 cases) |
| Judge | embedding cosine, threshold ≥ 0.80 |
| Extractor | `general-extractor` 0.14.0 |

Three metrics per configuration: **recall** (fraction of the 35 required
claims recovered), **precision** (fraction of emitted claims that matched a
required claim), and **ECE** (expected calibration error — lower is better).
"Emitted" is the raw count of candidate particles the model produced across
the 4 cases.

## Results

Ranked by recall. Bold marks the two Anthropic incumbents relevant to the
adoption decision.

| Model | Vendor | Recall | Precision | ECE | Emitted | Notes |
|---|---|---:|---:|---:|---:|---|
| **claude-sonnet-5** | Anthropic | **0.94** | **0.87** | **0.03** | 85 | **Champion — ADOPTED for extraction 2026-08-04** |
| **claude-sonnet-4-6** | Anthropic | 0.94 | 0.85 | 0.15 | 104 | Prior incumbent |
| kimi-k3 | Fireworks | 0.89 | 0.77 | 0.13 | 99 | Best outsider; no economic case (see below) |
| gpt-5.6-terra | OpenAI | 0.89 | 0.76 | 0.18 | 102 | |
| claude-haiku-4-5 | Anthropic | 0.74 | 0.73 | 0.19 | 77 | The cheap tier ($1/$5; $0.50/$2.50 batched) |
| glm-5p2 | Fireworks | 0.74 | 0.68 | 0.28 | 100 | |
| gpt-5.6-luna | OpenAI | 0.71 | 0.62 | 0.34 | 106 | High variance — see single-case runs below |
| claude-opus-5 | Anthropic | 0.69 | 0.58 | 0.29 | 105 | **Not a capability verdict** — see below |
| minimax-m3 | Fireworks | 0.60 | 0.59 | 0.32 | 86 | |
| deepseek-v4-flash | Fireworks | 0.57 | 0.48 | 0.44 | 104 | |
| qwen3p7-plus | Fireworks | 0.57 | 0.58 | 0.36 | 83 | Alibaba closed model, not self-hostable |
| gpt-5.6-sol | OpenAI | 0.51 | 0.71 | 0.26 | 70 | Erratic (0.88 recall single-case v0.1.0) |
| deepseek-v4-pro | Fireworks | 0.51 | 0.43 | 0.53 | 108 | |

The Fireworks router ids are `accounts/fireworks/routers/<name>` (e.g.
`accounts/fireworks/routers/kimi-k3`).

### Verdict and caveats

- **claude-sonnet-5 is the champion and was ADOPTED for extraction on
  2026-08-04.** It ties sonnet-4-6 on recall (0.94), edges it on precision
  (0.87 vs 0.85), and is dramatically better calibrated (ECE 0.03 vs 0.15) —
  while emitting fewer, tighter candidates (85 vs 104). See
  [pricing and the Sept 1 re-evaluation](#pricing-and-the-sept-1-re-evaluation).

- **claude-opus-5's 0.69 is not a capability verdict.** Its output was
  fluent with no truncation; the low score is a scope mismatch. The prompt
  and judge are tuned for sonnet, and opus-5 expands scope — emitting claims
  that a sonnet-tuned prompt+judge does not credit. Evaluating opus-5 fairly
  needs a prompt re-tune first; treat this row as "unmeasured against a fair
  harness," not "worse than haiku."

- **kimi-k3 is the best outsider (0.89) but has no economic case.**
  Fireworks prices it at **$3/$15 per MTok — sonnet list price — and offers
  no batch API**, so there is no cost lever to justify the ~5-point recall
  and ~10-point precision gap below sonnet-5.

- **gpt-5.6-luna and gpt-5.6-sol are erratic.** Both scored far higher on
  earlier single-case v0.1.0 runs (luna 0.88 and 0.75 recall; sol 0.88) than
  on the 4-case v0.2.0 suite (0.71 and 0.51). High variance across cases —
  not a stable ranking.

## Pricing and the Sept 1 re-evaluation

claude-sonnet-5 was adopted **during its introductory pricing window**:

| Period | Input / Output (per MTok) | Notes |
|---|---|---|
| through 2026-08-31 | **$2 / $10** | Introductory pricing |
| from 2026-09-01 | **$3 / $15** | List price, **with a new tokenizer (~+30% tokens)** |

At list price the new tokenizer's token inflation makes sonnet-5 cost
**~1.3× sonnet-4-6** for the same work. **A re-evaluation is due 2026-09-01**
to confirm sonnet-5 is still the right default once introductory pricing
ends and the tokenizer change lands.

Other pricing anchors from the survey:

- **claude-haiku-4-5** — the cheap tier: **$1 / $5**, or **$0.50 / $2.50
  batched**.
- **Fireworks kimi-k3** — **$3 / $15**, equal to sonnet list price, no batch
  API.

## Method notes

These are the operational settings that made the non-Anthropic runs
comparable at all. Record them: without them, most of the table would be
empty rows or truncation failures.

### Token budget and timeout (all reasoning models)

Every reasoning model — **DeepSeek-V4, Kimi K3, the Fireworks GLM / MiniMax
/ Qwen models, and the entire GPT-5.6 family** — required:

```yaml
extraction:
  max_tokens: 16384       # up from the 8192 default
  timeout_seconds: 300
```

At the **8192 default**, a reasoning model spends its thinking tokens from
the *same* completion budget and runs out mid-JSON-array: the reply comes
back truncated with `finish_reason=length` (the parser reports
"Unterminated string"). **16384 cleared it for every model in the trial.**
The adapter now logs a WARNING naming the budget when a `length` truncation
occurs, so this failure mode is surfaced rather than silent. Anthropic
models did not need the bump.

### OpenAI-compatible adapter flags

The OpenAI provider entry (`llm.providers.<name>`) needed:

```yaml
llm:
  providers:
    openai:
      base_url: https://api.openai.com/v1
      max_tokens_param: max_completion_tokens   # reasoning models reject max_tokens
      send_temperature: false                    # …and reject non-default temperatures
      structured_output: strict                  # strict-dialect JSON schemas
```

Conversely, **claude-sonnet-5 rejects the `temperature` param** the older
Anthropic models accepted; the native adapter **degrades gracefully**
(drops the param) rather than erroring.

### Calibration

sonnet-5 was calibration-probed on **`prose-calibration-001`**:
**65/65 candidates were judged correct**, so the calibrator **correctly
refused to fit** — the labels were degenerate (all-correct gives the fit no
signal). For that model/suite pairing, **`EXTRACTOR_DIRECT` is the measured
optimum**: an uncalibrated direct-confidence pass-through is the right
behavior, not a fallback. A newly routed model discloses `EXTRACTOR_DIRECT`
until a benchmark-driven calibration exists for its own `provider:model`
pairing.

## Earlier single-case runs (v0.1.0 — indicative only)

Before the v0.2.0 4-case suite existed, models were probed on a single case
under **suite version v0.1.0**: **8 required claims, so each claim is worth
±12.5 points of recall.** These numbers are **coarse and indicative only** —
do not compare them against the v0.2.0 table above, and do not average the
two.

| Model | Recall | Precision | ECE |
|---|---:|---:|---:|
| claude-sonnet-4-6 | 1.00 | 0.86 | 0.14 |
| gpt-5.6-luna | 0.88 | 0.82 | 0.18 |
| gpt-5.6-luna (2nd run) | 0.75 | 0.68 | 0.31 |
| gpt-5.6-terra | 0.75 | 0.72 | 0.25 |
| gpt-5.6-sol | 0.88 | 0.59 | 0.38 |

The two luna rows (0.88 vs 0.75 recall on the *same* single case) and sol's
0.88-here / 0.51-there swing are the origin of the "erratic GPT-5.6" caveat
above.

## Hallucinated model names — do not re-chase

Two Fireworks model ids were probed during the survey and **do not exist**.
They are plausible-looking autocomplete hallucinations; recorded here so a
future session does not re-chase them:

| Probed id | Status |
|---|---|
| `glm-5p2-flash` | Does not exist on Fireworks |
| `qwen3p7-max` | Does not exist on Fireworks (announced "coming soon" only) |

## How to reproduce / extend

The survey is one CLI verb per model, routed by config. To reproduce a row
or add a new model:

**1. Route the extraction purpose at the model under test.** For an
Anthropic model, set `llm.default` (or `llm.extraction`) in a config file;
for a non-Anthropic vendor, add a named provider and point
`llm.extraction` at it. Example for a reasoning model — note the token
budget and timeout the method notes require:

```yaml
# survey-model.yaml
llm:
  extraction:
    provider: fireworks
    model: accounts/fireworks/routers/kimi-k3
  providers:
    fireworks:
      base_url: https://api.fireworks.ai/inference/v1
      max_tokens_param: max_completion_tokens
      send_temperature: false
      structured_output: strict
      timeout_seconds: 300
extraction:
  max_tokens: 16384
  timeout_seconds: 300
```

The API key is the secret `PARTICLES_LLM_API_KEY_<NAME>` in the environment
(here `PARTICLES_LLM_API_KEY_FIREWORKS`), never in the config file.

**2. Run the suite against the general-extractor**, with the config routing
the model:

```bash
PARTICLES_CONFIG=survey-model.yaml uv run particles extractor benchmark general-extractor --suite prose-article-seed-001
```

`--estimate` prints the projected LLM cost and exits without calling. The
run persists a report JSON under `benchmark.runs_dir` stamped with the
resolved `provider:model` pairing (pass `--no-save` for a throwaway run).
Extraction is a sampling process, so a single run is a single sample — pass
`--runs N` to repeat and report each metric's mean ± spread (the error bars
a provider comparison needs before it calls a gap real).

**3. Calibrate the newly-routed model** on `prose-calibration-001`
 before trusting its stored confidences; until then its particles
disclose `EXTRACTOR_DIRECT`. Use `particles extractor calibrate`.

## Cross-references

- Named OpenAI-compatible providers — adding a vendor is configuration,
  never code.
- The per-particle provider stamp: every particle records the
  `provider:model` pairing that produced it.
- The benchmark suite contract and routing, and the
  `prose-calibration-001` calibration suite.
- The `particles extractor calibrate` verb and its degenerate-fit
  refusals.
- The extraction-quality benchmark harness itself.
- The persisted `runs/` JSON convention lives under `benchmark.runs_dir`;
  see [Benchmarks § Relationship to the extractor benchmarks](../benchmarks.md#relationship-to-the-extractor-benchmarks).
