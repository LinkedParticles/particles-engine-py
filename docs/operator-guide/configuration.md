# Configuration

All tuneable parameters live in `particles/config.py` as a Pydantic
model. The canonical sample is `config.yaml.sample` in the repo
root — copy it to `config.yaml` to override defaults. `config.yaml`
is gitignored.

## Which `config.yaml` loads

Discovery walks upward, git-style:

1. `PARTICLES_CONFIG`, if set — absolute authority. A path that does not
   exist resolves to *no config file* rather than falling through, so this
   is also the explicit opt-out from everything below.
2. `./config.yaml` in the working directory.
3. The nearest `config.yaml` in an ancestor directory — first match wins.
   The walk stops after examining the directory that holds a `.git` entry
   (a *file* in a git worktree, a directory in a normal checkout), so a
   `config.yaml` in `$HOME` is never inherited by the projects beneath it.
4. Otherwise: compiled-in defaults plus env-var overrides.

Step 3 is why a verb run from `scripts/`, from a git worktree, or from a
hook spawn now gets the config you wrote at the repo root. Before it, such
a process silently reverted **every** knob to its compiled default — the
cause behind blobs written somewhere nobody looks. `particles config
validate` and `particles hook doctor` both print the file they resolved;
when "which config am I loading?" matters, ask them rather than guess.

## Precedence

```
env var override   →  highest
config.yaml field  →  operator tuning
compiled default   →  fallback
```

Env-var overrides are registered in `_ENV_OVERRIDES` in
`particles/config.py`. The common ones:

| Env var | Config field | Purpose |
|---|---|---|
| `DATABASE_URL` | `database_url` | SQLite path |
| `PARTICLES_BLOB_DIR` | `blob_dir` | Where deposited blobs are stored |
| `PARTICLES_CONFIG` | — (bootstrap) | Path to a non-default `config.yaml` |
| `TRUST_DIFFERENTIAL_THRESHOLD` | `trust.differential_threshold` | When trust differences flag inconsistencies |
| `RECONCILIATION_STORE_MODE` | `reconciliation.store_mode` | `single` (default) or `multi` — consensus-store reconciliation regime |

For the full env-var list and field-by-field tuning options, see
`config.yaml.sample`.

## Secrets

Secrets never live in `config.yaml`. Application code calls helpers
in `particles/secrets.py`:

| Secret | Env var | Used by |
|---|---|---|
| Anthropic API key | `ANTHROPIC_API_KEY` | Extractor, semantic lint, wiki synthesis |
| Numista API key | `NUMISTA_API_KEY` (optional) | Numista extractor |
| GitHub API key | `GITHUB_API_KEY` (optional) | GitHub extractor (gist / repo / pages) |

Migrating to a secrets manager (1Password, AWS, sops, …) only
requires editing `particles/secrets.py`.

## LLM provider selection

Every chat/completion call routes through a `CompletionProvider` port
. The `llm` section picks the `(provider, model)` pairing per
**purpose** — a `default` plus optional overrides for `extraction`,
`semantic_lint`, `query_response`, `synthesis`, and `benchmark`:

```yaml
llm:
  default:
    provider: anthropic
    model: claude-sonnet-4-6
  # Route a single purpose to a different model — e.g. a cheaper model for
  # high-volume extraction while synthesis stays on the default.
  # extraction:
  #   model: claude-haiku-4-5
```

`provider` is `anthropic` (the native adapter) or the name of any entry in
`llm.providers` (see the next section). `max_tokens` is **not**
set here — it is per-call and lives with each call site
(`extraction.max_tokens`, `extraction.query_max_tokens`,
`wiki.max_tokens`).

### Named providers (any OpenAI-compatible endpoint)

Every non-Anthropic vendor — hosted (OpenAI, DeepSeek, Kimi, gateways) or
local (Ollama, llama.cpp's server, vLLM, LM Studio) — speaks the OpenAI
`chat/completions` dialect, so all of them are **named entries** in
`llm.providers`: adding a vendor is a config block, never code.
The endpoint, resilience, and dialect policy are per-entry; only the
per-purpose `model` string lives in the purpose override. The `local` entry
(an Ollama endpoint) is compiled in, so `provider: local` works
with zero configuration.

```yaml
llm:
  default:
    provider: anthropic
    model: claude-sonnet-4-6
  extraction:                # send only extraction to a cheaper vendor
    provider: openai
    model: gpt-5.6-luna
  providers:
    openai:
      base_url: https://api.openai.com/v1  # adapter appends /chat/completions
      max_tokens_param: max_completion_tokens  # reasoning models reject max_tokens
      send_temperature: false                  # …and non-default temperatures
      structured_output: strict                # strict-dialect JSON schemas
    local:                   # override the compiled-in Ollama entry if needed
      base_url: http://localhost:11434/v1
      timeout_seconds: 120
```

The API key is a secret named after the entry:
`PARTICLES_LLM_API_KEY_<NAME>` (`PARTICLES_LLM_API_KEY_OPENAI`, …), set in
the environment, never in `config.yaml`. The `local` entry also honours the
legacy `PARTICLES_LOCAL_LLM_API_KEY`. Endpoints that enforce no auth (bare
Ollama / llama.cpp) need no key and the adapter omits the `Authorization`
header.

The two dialect knobs are declarative statements about a known endpoint,
not runtime negotiation: `max_tokens_param` picks which body member carries
the length cap, and `send_temperature: false` drops `temperature` from
requests entirely. A wrong knob fails loudly with the endpoint's own
HTTP 400. `structured_output: strict` transforms JSON schemas to the
OpenAI-strict dialect (every key required, optionality as union-with-null)
— required for api.openai.com; leave the default `auto` for tolerant
endpoints like Ollama.

> **Deprecation:** the pre-0227 `llm.local` block is honoured as
> `llm.providers.local` with a warning for one release cycle — move it
> under `providers` when convenient.

#### Reasoning models need a bigger token budget

A reasoning model (DeepSeek-V4, Kimi K3, the GPT-5.6 family) spends its
thinking tokens from the **same completion budget as the answer**, so a
prompt that fit comfortably in the default `extraction.max_tokens: 8192`
on a non-reasoning model can exhaust it before the answer is finished — or
before it starts. The endpoint returns HTTP 200 with `finish_reason:
length` and text that stops mid-token, which the extractor's JSON parser
then reports as `Failed to parse extraction response: Unterminated string`.
The adapter logs a WARNING naming the pairing and the budget whenever a
reply comes back truncated, so the two are not confused; a truncated reply
with *no* text at all fails with a budget-shaped error rather than a
generic empty-response one.

Give these models headroom — `extraction.max_tokens: 16384` cleared the
parse failures in the 2026-08 trial (DeepSeek-V4 flash/pro, Kimi K3) — and
consider raising the entry's `timeout_seconds` for large models, since a long
thinking pass takes wall-clock time the default 120 s may not cover:

```yaml
llm:
  providers:
    fireworks:
      base_url: https://api.fireworks.ai/inference/v1
      timeout_seconds: 300     # reasoning passes are slow as well as long
extraction:
  max_tokens: 16384            # thinking + answer share this budget
```

Confidence calibration is **per `(extractor, model)` pairing**:
each `particles extractor calibrate` run stores a record keyed by the
extraction model it ran under, and the pipeline applies the one matching the
configured model. So a *newly* pointed model — including any
`<provider>:<model>` — is uncalibrated until you benchmark it (queries fall
back to the `EXTRACTOR_DIRECT` disclosure meanwhile), but switching **back**
to a model you calibrated before restores its calibration with no re-fit.
List the stored pairings with `particles extractor calibrations
<extractor-id>`.

> **Pick provider names before you benchmark.** The calibration key is
> `<name>:<model>` — the *operator-chosen entry name*, not the vendor —
> so renaming a provider entry orphans every calibration record made under
> the old name. Treat a rename as a recalibration event.

> **Deprecation:** `extraction.model` and `wiki.model` moved into this
> section. The old keys are migrated automatically (to `llm.default.model`
> and `llm.synthesis.model`) with a warning for one release cycle — move
> them to `llm` when convenient.
>
> Note the scope change: `llm.default.model` is the fallback for *every*
> purpose, including semantic lint and the benchmark judge — which were
> previously hard-wired to `claude-sonnet-4-6`. So if you set a non-default
> `extraction.model` (now `llm.default.model`), it will also drive lint and
> benchmark. To keep those on a cheaper model, set `llm.semantic_lint.model`
> / `llm.benchmark.model` explicitly.

## Common knobs

A few config fields you'll likely want to set early:

- `llm.default.model` (and per-purpose `llm.<purpose>.model`) — the
  completion model each purpose uses. See *LLM provider
  selection* above.
- `exporter_common.min_particle_confidence` — the cross-exporter
  quality threshold. Particles below this `effective_confidence` are
  dropped from every export.
- `wiki.min_particles` — minimum particles per subject for the wiki
  exporter to render. Default 3.
- `query.top_k` — top-k truncation for the semantic search.
- `embeddings.progress_bars` — whether the embedding stack prints its
  tqdm progress bars (`Loading weights …` on model load, `Batches …` on
  each encode) to stderr. Default `false` (they are noise for a CLI verb
  like `query`); set `true` — or `PARTICLES_EMBEDDINGS_PROGRESS_BARS=1` —
  to restore them.
- `obsidian.default_output_path` — so `particles export obsidian`
  works without an argument.
- `inbox.file_path` — the iCloud-synced file the `particles inbox`
  commands read URLs from (`inbox.poll_interval_seconds` tunes the
  `inbox watch` cadence). Setup walkthrough: [User Guide → Depositing
  from your phone](../user-guide/inbox.md).
- `reconciliation.store_mode` — `single` (default) for a solo store, or
  `multi` for a multi-contributor / consensus store. In `multi` mode a
  confirmed cross-source contradiction is surfaced as an INCONSISTENCY
  (both claims stay ACTIVE, ranked per-viewer at query time) rather than
  one claim auto-superseding the other on trust — a contributor's claim is
  never dropped by another contributor's trust.

See [Tuning](tuning.md) for the trust / calibration / age-decay
knobs that drive `effective_confidence`.

## Agent-memory projection

The `agent_memory.projection` block governs the `MEMORY.md` projection for
the Claude Code integration — see [User Guide → Claude
Code](../user-guide/claude-code.md#the-memorymd-projection) for the
walkthrough. The load-bearing knobs:

- `agent_memory.projection.enabled` (default `true`) — render + splice the
  `memory-index` region and run the session-start freshness check. `false`
  falls back to a plain digest push with no region writes.
- `agent_memory.projection.fold_authored_lines` (default `true`) — move
  agent-authored lines outside the projected region into the append-only
  archive after each successful harvest (never destroyed).

### Git-versioned projection history

Off by default. When enabled **and** the memory directory is inside a git
repo, each render that changes files under it is committed with a structured
message (run id + ranking-delta summary), giving you a diffable,
rollback-able history of the *view* while the store stays the source of
truth. **Every git failure degrades silently** (logged at debug, never raised)
— the commit is a bonus, the projection is the product.

- `agent_memory.projection.git.enabled` (default `false`) — master switch.
  Committing into your repo is opt-in; turning it on without the memory
  directory being a git repo is harmless (the step is simply inert).
- `agent_memory.projection.git.sign` (default `false`) — `false` passes
  `--no-gpg-sign` so an unattended session-end commit never blocks on a
  signing agent; `true` drops the override and respects your own
  `commit.gpgsign`. This SDK's own GPG requirement is never imposed on your
  memory repo, and a signing failure never fails the projection.
- `agent_memory.projection.git.author_name` / `.author_email` (default
  `null`) — passed per-commit via `-c user.name` / `-c user.email` (never
  written into your git config). `null` uses your repo's own identity; when
  that is absent, the commit degrades silently.
- `agent_memory.projection.git.max_delta_excerpts` (default `6`) — cap on the
  added/removed excerpt lines in the commit message. The count line always
  states the true totals, so a large delta is never silently truncated.

Only files under the memory directory are staged (never `git add -A`), and
the internal backup / snapshot / archive live outside it, so they never enter
your history.
