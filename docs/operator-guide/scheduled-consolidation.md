# Scheduled consolidation — the dream cycle

`particles memory consolidate` runs the cross-session memory
maintenance passes in one verb, in a fixed order, under the existing cost
caps — so "memory that tends itself" becomes a crontab line instead of four
verbs an operator has to remember to run.

The pass list, in order:

1. **Extract catch-up** *(LLM)* — extract PENDING snapshots, oldest first,
   capped at `consolidation.max_pending_entries` per run. A capped run
   discloses the remainder ("12 remain — next run continues").
2. **Reconcile** *(LLM, capped)* — the cross-entry
   document-supersession sweep. Each candidate pair costs one
   replacement-signal probe, spent highest-similarity-first under
   `consolidation.max_reconcile_probes`; a truncated run discloses
   "probed X of Y candidate pairs". Skipped (disclosed) on degraded runs.
3. **Census** *(LLM, capped + scoped)* — the audit's contradiction probe +
   duplicate scan, capped at `audit.max_contradiction_probes` and scoped to
   what changed since the previous run (see [Delta scope](#delta-scope)).
4. **Curation-queue refresh** — the queue, computed from the *same*
   card collection the census already paid for; the report ends with the
   morning's worklist.
5. **Utility mining** *(LLM, bounded)* — the pass over harvested
   session transcripts. The literal tier is LLM-free and always runs; the
   behavioural tier spends **one shared per-run budget**
   (`utility.mining.max_behavioural_calls`) across all sessions, and
   exhaustion is disclosed ("behavioural budget exhausted after N of M
   sessions").
6. **Projection re-render** *(zero-LLM)* — the `MEMORY.md`
   render-splice cycle, via the same harvest-then-render tail the SessionEnd
   hook uses.
7. **Record + report** — one `CONSOLIDATION_RUN` operator event per run.

## Scheduling (launchd / cron)

On a host, cadence comes from the operating system's scheduler, and `--if-due`
makes over-scheduling harmless. There is no
`--install-schedule` verb (deliberately: writing a LaunchAgent plist is a
system-level footprint the installer never touches), so
you install the job by hand, once.

!!! info "Containers schedule themselves"

    An **opt-in** resident daemon mode is added
    (`particles engine serve … --daemon`) that runs this cycle in-process on a
    timer, because a container has neither launchd nor cron. It is a *rider*
    on the external-scheduler contract, not a replacement: everything on this
    page stands for CLI-only use and for hosts that run no daemon, and daemon
    mode is off unless you ask for it. See
    [Running in a container](container-deployment.md).

!!! warning "A scheduled job inherits nothing — bake in absolute paths"

    This is the one way to get a job that *looks* healthy and does nothing.
    A LaunchAgent runs with **no working directory** (effectively `/`), no
    shell profile, and a minimal environment. A bare `particles memory
    consolidate` therefore finds **no `config.yaml`** (so none of your named
    stores or consolidation tuning exists), resolves a relative
    `storage.database_url` against the wrong directory (so it opens an empty
    store rather than yours), and sees **no `ANTHROPIC_API_KEY`** (so every
    semantic pass degrades to structural-only). With no log paths set, the
    disclosures that would tell you all three go nowhere.

    The same trap bit the session hooks, which silently dropped
    harvests until v1.70.2 baked absolute env pins into their command
    strings. Every path and variable below is absolute for that reason —
    substitute your own, and do not shorten them to relative forms.

    **`DATABASE_URL` pins only the `default` store.** It overrides
    `storage.database_url`, and that is the `default` handle's DSN. Any other
    handle resolves from `storage.stores[<handle>]` in `config.yaml` and
    ignores `DATABASE_URL` completely (`particles/db.py`). So:

    - running against **`default`** (the example below) — set both
      `PARTICLES_CONFIG` *and* `DATABASE_URL`;
    - running against a **named store** (`--store memory`, `--store research`)
      — `PARTICLES_CONFIG` is the load-bearing pin, because the DSN comes from
      the config file; `DATABASE_URL` does nothing for that handle, and setting
      it alongside a named store is the misleading combination to avoid.

    **`PARTICLES_BLOB_DIR` matters as much as the database URL.** The corpus
    stores raw source bytes as content-addressed blobs, and
    `storage.blob_dir` defaults to the **relative** `./corpus_blobs` — so a
    process started from a different directory writes blobs somewhere else
    while happily sharing the same database. The rows then reference content
    that is not where this process is looking, and extraction fails with
    `Blob not found for hash …`. Observed in practice: deposits made from
    several git worktrees left one database pointing at blobs scattered
    across five `corpus_blobs/` directories, and blobs written inside a
    worktree that was later deleted were lost outright. Pin it absolutely
    here *and* set `storage.blob_dir` to an absolute path in `config.yaml`.

    To check a store's blobs are all reachable:

    ```bash
    # any blob directories other than the one you intend?
    find "$(dirname /Users/you/src/myproject/particles.db)" -name corpus_blobs -type d
    ```

### macOS (launchd)

Write `~/Library/LaunchAgents/dev.particles.consolidate.plist`. Two things to
substitute:

- **every `/Users/you/src/myproject` path** — with your project's absolute path;
- **the store handle** — the value after `--store` in `ProgramArguments` below.
  A *store handle* names a database: `default` is the implicit store (most
  setups have only this one), and any other handle must be declared under
  `storage.stores` in your `config.yaml` (see `config.yaml.sample`).

  Do not be thrown by `memory` appearing twice in a full command line like
  `particles memory consolidate --if-due --store memory`: the first is the CLI
  *command group*, the second would be a store that happens to be named
  `memory`. The example below uses the handle `default` to keep the two
  distinct — and because of the pin rule in the next paragraph.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.particles.consolidate</string>

  <!-- Absolute path to the venv console script — launchd has no PATH. -->
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/src/myproject/.venv/bin/particles</string>
    <string>memory</string>       <!-- the CLI command group: `particles memory …` -->
    <string>consolidate</string>
    <string>--if-due</string>
    <string>--store</string>
    <string>default</string>      <!-- the STORE HANDLE (see below) -->
  </array>

  <!-- Without these the job runs against compiled-in defaults and an
       empty store. PARTICLES_CONFIG finds your operator config; an
       absolute DATABASE_URL is required whenever the configured DSN is
       not already absolute; PARTICLES_BLOB_DIR pins the content-addressed
       blob store, whose compiled default `./corpus_blobs` is RELATIVE and
       therefore follows the process's working directory. -->
  <key>EnvironmentVariables</key>
  <dict>
    <key>PARTICLES_CONFIG</key>
    <string>/Users/you/src/myproject/config.yaml</string>
    <key>DATABASE_URL</key>
    <string>sqlite+aiosqlite:////Users/you/src/myproject/particles.db</string>
    <key>PARTICLES_BLOB_DIR</key>
    <string>/Users/you/src/myproject/corpus_blobs</string>
  </dict>

  <!-- Belt and braces: also gives relative paths a sane base. -->
  <key>WorkingDirectory</key>
  <string>/Users/you/src/myproject</string>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>30</integer>
  </dict>

  <!-- Cron mails you on failure; launchd does not. These files ARE your
       failure notification — the exit-code table below is only useful if
       someone reads it. -->
  <key>StandardOutPath</key>
  <string>/Users/you/Library/Logs/particles-consolidate.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/you/Library/Logs/particles-consolidate.err</string>
</dict>
</plist>
```

### The API key

The semantic passes read **`ANTHROPIC_API_KEY` from the process environment**
— there is no secrets *file* and no `PARTICLES_SECRETS` variable
(`particles/secrets.py` is an internal module that calls
`os.environ.get("ANTHROPIC_API_KEY")`; it is deliberately absent from
`config.yaml`, because secrets never live in config). And **launchd does not
source shell files** — `EnvironmentVariables` is a literal key→value dict, so
pointing it at `~/.zshenv` or `~/.zprofile` sets a useless string and the job
runs key-less. Three honest options:

**1. Key in the plist** (simplest; the plist becomes a secret — `chmod 600`,
never commit it):

```xml
  <key>EnvironmentVariables</key>
  <dict>
    <key>PARTICLES_CONFIG</key>
    <string>/Users/you/src/myproject/config.yaml</string>
    <key>DATABASE_URL</key>
    <string>sqlite+aiosqlite:////Users/you/src/myproject/particles.db</string>
    <key>ANTHROPIC_API_KEY</key>
    <string>sk-ant-…</string>
  </dict>
```

**2. Wrapper script** (keeps the key out of the plist; the script is what
sources a file). Point `ProgramArguments` at the wrapper instead of at
`particles`:

```sh
#!/bin/sh
# ~/bin/particles-consolidate.sh — chmod 700
. "$HOME/.particles-env"          # a chmod 600 file: export ANTHROPIC_API_KEY=sk-ant-…
export PARTICLES_CONFIG=/Users/you/src/myproject/config.yaml
export DATABASE_URL=sqlite+aiosqlite:////Users/you/src/myproject/particles.db
export PARTICLES_BLOB_DIR=/Users/you/src/myproject/corpus_blobs
exec /Users/you/src/myproject/.venv/bin/particles memory consolidate --if-due --store default
```

Verified to work from a stripped environment (`env -i`), which is what launchd
supplies. Note `.` (POSIX source), not `source` — the script runs under
`/bin/sh`, not zsh.

**3. No key at all.** A key-less run is *honest, not broken*: it completes,
writes its run record, and discloses the downgrade rather than reporting a
clean bill. In the log you will see

```
  semantic passes skipped: no API key (structural-only run)
```

and the structural passes (curation refresh, projection re-render) still do
their work. Choose this if you would rather not put a key on disk; see
[Degradation](#degradation-structural-only-is-disclosed-never-silent).

Install, verify, and remove:

```bash
# install (modern launchd; bootstrap replaces the deprecated `load -w`)
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/dev.particles.consolidate.plist

# confirm it is registered
launchctl print gui/$UID/dev.particles.consolidate | head -20

# force one run now instead of waiting for 03:30 — the real smoke test
launchctl kickstart -p gui/$UID/dev.particles.consolidate

# uninstall
launchctl bootout gui/$UID/dev.particles.consolidate
```

After the kickstart, **verify it hit your store rather than an empty one**:

```bash
# The run must appear in the store the job actually targeted. `events list`
# reads the *default* store and has no --store flag. For the `default` handle
# below, pointing DATABASE_URL at the same file the plist pins is exactly the
# check you want. (If your job targets a NAMED store, set PARTICLES_CONFIG here
# instead and give that store's DSN — DATABASE_URL cannot reach it.)
DATABASE_URL=sqlite+aiosqlite:////Users/you/src/myproject/particles.db \
  particles events list --type CONSOLIDATION_RUN --limit 1

# ...and the log should not be reporting an empty or missing store
tail -20 ~/Library/Logs/particles-consolidate.err
```

If `events list` shows no new record, the job ran against the wrong store —
re-check `PARTICLES_CONFIG` and `DATABASE_URL` in the plist before trusting
the schedule.

### Linux / BSD (cron)

```cron
30 3 * * *  cd /path/to/project && PARTICLES_CONFIG=/path/to/project/config.yaml .venv/bin/particles memory consolidate --if-due --store default
```

`cd` covers the working-directory half of the trap, and cron mails you on a
non-zero exit (see the exit-code table below), so cron needs less scaffolding
than launchd — but set `PARTICLES_CONFIG` explicitly anyway, since cron's
environment is also minimal.

`--if-due` reads the verb's own last successful `CONSOLIDATION_RUN` event
(an interactive `particles audit` writes the same event type but does not
count; a disclosed structural-only run does — so a key-less setup retries
next interval instead of hot-looping) and exits 0 without running (one log
line, no run record) when that run is younger than
`consolidation.min_interval_hours` (default 20 — daily scheduling with
headroom for clock drift). A laptop that was asleep at 03:30 can therefore
safely retry hourly; two overlapping schedules collapse to one run per
interval.

### Exit codes (cron observability)

| Code | Meaning |
|---|---|
| `0` | Success — including disclosed structural-only runs and `--if-due` / lock skips |
| `1` | One or more passes failed (the run record is still written; stderr names them) |
| `2` | The cycle could not start (unusable store, config error) |

Cron's mail-on-nonzero fires exactly on the two conditions an operator must
see.

### Reading the logs, and knowing when a run finished

**The log files append; nothing rotates them.** launchd opens
`StandardOutPath` / `StandardErrorPath` in append mode, so every run stacks
below the last one — `cat` will show you the *oldest* run first and get less
useful over time. Use `tail`:

```bash
tail -50 ~/Library/Logs/particles-consolidate.log   # the most recent run
tail -f  ~/Library/Logs/particles-consolidate.log   # watch one finish live
```

**Is it still running?** launchd gives no completion notification, and a first
run on a populated store can take a long while (extraction backlog + the
census). Three checks, most direct first:

```bash
# alive? (silent = finished)
pgrep -f "memory consolidate"

# state, and the exit code of the last COMPLETED run
launchctl print gui/$UID/dev.particles.consolidate | grep -E "state|last exit code"

# the authoritative record — written even on partial failure
DATABASE_URL=sqlite+aiosqlite:////Users/you/src/myproject/particles.db \
  particles events list --type CONSOLIDATION_RUN --limit 2
```

`last exit code` reports the *previous* run while a new one is in flight, so
read it together with `state` (`running` vs `not running`); map the value with
the table above.

### Rotating the logs

Nothing rotates these files for you. `newsyslog.d` is the system mechanism but
needs root to install and validate, so the self-contained option — and the one
that composes with the wrapper script above — is to rotate at the **start** of
the wrapper, when no run is in flight. That ordering matters: the job is
periodic rather than long-lived, so between runs no process holds the file
open and a rename-based rotation is safe.

```sh
#!/bin/sh
# ~/bin/particles-consolidate.sh — rotate first, then run.
rotate_log() {                      # rotate_log <path> <max_bytes> <generations>
  f=$1; max_bytes=$2; keep=$3
  [ -f "$f" ] || return 0
  [ "$(stat -f%z "$f")" -gt "$max_bytes" ] || return 0
  i=$keep
  while [ "$i" -gt 1 ]; do
    prev=$((i - 1))
    [ -f "$f.$prev.gz" ] && mv "$f.$prev.gz" "$f.$i.gz"
    i=$prev
  done
  gzip -c "$f" > "$f.1.gz" && : > "$f"   # truncate in place, keep the inode
}

LOGDIR="$HOME/Library/Logs"
rotate_log "$LOGDIR/particles-consolidate.log" 524288 7   # 512 KB, 7 generations
rotate_log "$LOGDIR/particles-consolidate.err" 524288 7

. "$HOME/.particles-env"
export PARTICLES_CONFIG=/Users/you/src/myproject/config.yaml
export DATABASE_URL=sqlite+aiosqlite:////Users/you/src/myproject/particles.db
export PARTICLES_BLOB_DIR=/Users/you/src/myproject/corpus_blobs
exec /Users/you/src/myproject/.venv/bin/particles memory consolidate --if-due --store default
```

Verified behaviour: generations shift `.1.gz` → `.2.gz` → …, the count is
capped at `keep`, `.1.gz` holds the most recent content, and the live file is
truncated to zero rather than unlinked (`: >`), so anything still holding the
descriptor keeps writing to the same inode.

## Flags

```
particles memory consolidate
  --store HANDLE          # default: the default store
  --if-due                # cadence guard (see above)
  --structural-only       # skip all LLM passes (disclosed, never silent)
  --scope delta|store     # semantic-pass scope (default delta)
  --output FILE           # also write the run report as Markdown
  --format markdown|json  # terminal format (default markdown)
  --verbose / --debug
```

There is no confirmation prompt and no `--yes`: an autonomous verb cannot
prompt, so cost is bounded *by construction* — the existing caps plus delta
scope — and confirmation is replaced with disclosure.

## Delta scope

By default the semantic passes probe only the particles created or modified
since the watermark, plus particles from corpus entries deposited since then.
The watermark is the previous eligible run's **`started_at`** (not its
completion — so nothing written mid-cycle ever falls between two runs'
windows; overlap re-probes are idempotent), and the scope is computed *after*
pass 1, so the particles extraction just minted are censused in the same run.
Watermark-eligible means: a successful, non-degraded run by the consolidation
verb itself — a structural-only night or an interactive `particles audit`
never advances the watermark. Nightly cost therefore scales with the day's
delta, not the store. The first run (no prior eligible record) and
`--scope store` run store-wide — still capped, with the "probed X of Y
candidate pairs" disclosure. The below-cap tail of *pre-existing* pairs is
never reached by scheduled runs; a deliberate `--scope store` run or
`particles lint` remains the exhaustive instrument.

## Degradation — structural-only is disclosed, never silent

With no API key, an open circuit breaker, `--structural-only`, or
`consolidation.semantic: false`, the LLM-free passes (curation refresh,
projection render) still run in full; extraction and the probe-bearing
reconcile sweep are skipped (each disclosed), the census runs structural
finders + REPORT-mode duplicates only, and utility mining runs the literal
tier only. Every skip is disclosed in the report and recorded on the run
record (`semantic_degraded` + reason). A degraded run's contradiction line
reads **"not probed this run"**, never "0" — and a degraded run never
advances the delta watermark, so everything it did not probe stays in scope
for the next full run.

## The run record

Each run writes one `CONSOLIDATION_RUN` operator event with a
versioned payload (`format: 1`): per-pass status/durations, per-pass LLM call
counts, the machine-readable census (probe counts, duplicate totals, pending
backlog, utility events), degradation disclosures, provider/model per
purpose, and `started_at` / `completed_at` (`started_at` is the next run's
delta watermark — see [Delta scope](#delta-scope)). "Is the cycle actually
running?" is one command:

```bash
particles events list --type CONSOLIDATION_RUN
```

`particles audit` records the same event shape (`actor: audit`) and the
report's headline lines carry "+2 since last run" deltas against the most
recent prior run of either kind — but an audit event neither advances the
consolidation watermark nor satisfies `--if-due` (the audit runs none of the
cross-session passes, so it cannot stand in for a consolidation run).

## Concurrency and failure

- **One cycle at a time.** A `consolidate.lock` file in the integration state
  directory (`claude_code.state_dir`) serializes cycles; a held lock exits 0
  with "already running — skipped". A lock whose pid is dead or older than
  `consolidation.lock_timeout_minutes` (default 120) is stale and reclaimed.
- **Interactive sessions interleave safely.** The lockfile serializes
  *cycles*, not writes: each pass takes the cross-process write lock
  per transaction exactly as its verb always has, and every pass is
  idempotent.
- **Failure mid-pass: continue and report.** A failing pass is caught,
  recorded on the run record (`failed(<error>)`), and the cycle continues —
  the zero-LLM projection render still runs, so a flaky network night never
  leaves `MEMORY.md` staler than it had to be.

## Configuration

```yaml
consolidation:
  min_interval_hours: 20    # --if-due threshold
  extract_pending: true     # pass 1 on/off
  max_pending_entries: 20   # pass 1 per-run cap
  extract_batching: true    # pass 1 pooled half-price batching
  semantic: true            # LLM passes on scheduled runs
  max_reconcile_probes: 50  # pass 2 per-run probe cap (highest-similarity-first)
  lock_timeout_minutes: 120 # stale cycle-lock reclaim
```

Detection thresholds deliberately stay where they live (`audit.*`, `lint.*`,
`utility.mining.*`, `links_suggest.*`): consolidation composes the finders,
it does not re-tune them.

### Half-price probes — batch completion

Nobody is waiting for a 03:30 run, so its two largest probe populations — the
contradiction probe (capped at `audit.max_contradiction_probes`) and the
behavioural utility matcher (`utility.mining.max_behavioural_calls`) — are
submitted to the Anthropic **Message Batches API** as one job each instead of
one call each. All token usage in a batch is billed at **50%**. At today's caps
that is roughly 1200 of the cycle's ~1250 nightly probe calls.

The trade is latency for price: a batch usually completes within an hour and
may take up to 24, which is why only a caller that has declared itself
latency-tolerant takes it. `particles lint` and the first-run `particles audit`
always run the sequential per-probe loop and keep their answer-in-seconds, no
matter what is configured here.

```yaml
llm:
  batch:
    enabled: true                # false ⇒ everything sequential, as before 0218
    min_requests: 4              # below this, a batch is not worth the round trip
    max_requests_per_batch: 1000 # a larger set is chunked into successive batches
    poll_interval_seconds: 30    # how often processing_status is checked
    max_wait_seconds: 3600       # per-batch wall-clock ceiling
```

Failure behaviour is the existing probe-unavailable degradation, at three
levels. A batch the provider refuses outright falls back to sequential calls
(you lose the discount, not the pass). A batch still processing after
`max_wait_seconds` is cancelled and its probes report unavailable, so a stuck
job cannot hold the cycle open until the API's own 24-hour expiry. An individual
request that errored or expired inside an otherwise healthy batch comes back as
one unanswered probe — counted in the run's disclosure, never silently dropped.

One pass is **not** batched: the reconcile sweep (50 probes/night, whose loop
skips candidates already demoted earlier in the same loop). It is still billed
per call.

To turn the whole thing off and get a cycle that finishes fast at full price,
set `llm.batch.enabled: false`.

### Half-price extraction — the pooled extract pass

The extract pass is the cycle's dominant **token** consumer (the Claude Code
session harvests are large transcripts), and since it rides the same
Message Batches discount: the capped pending set runs as concurrent
per-snapshot tasks whose chunk requests merge into **one nightly batch job**
through a completion pool, so the whole set — single-chunk documents
included — clears `llm.batch.min_requests` together. Wall clock for the pass
becomes roughly one batch turnaround instead of 10–20 sequential multi-minute
calls.

What does *not* change: interactive `particles extract` and `reindex` stay
sequential; per-snapshot failure isolation is the existing machinery (an
unanswered request resets its snapshot to PENDING for the next night; an
account-level failure stops the pass with one disclosed error); and at most
one snapshot per corpus entry runs per night, so carry-forward keeps seeing
the previous snapshot's persisted chunk hashes. Vision-flagged PDF pages and
image sources are multimodal and still run sequentially at full price.

`consolidation.extract_batching: false` restores the serial per-snapshot loop
exactly; `llm.batch.enabled: false` keeps the pooling but degrades dispatch to
sequential full-price calls.

### Local-model structured output

The cycle's JSON-shaped LLM calls (extraction, the contradiction probe, the
behavioural utility matcher) now pass their reply schemas through the
completion port. When a purpose is routed to a named OpenAI-compatible
provider (e.g. the compiled-in `local` entry), the entry's
`structured_output: auto` (the default) sends OpenAI-style
`response_format: {type: json_schema, strict: true}` so a local model cannot
silently drop particles behind unparseable JSON; an endpoint that rejects the
parameter gets one retry without it (a logged downgrade back to
tolerant-parser reliability). Set `structured_output: off` to disable. The
Anthropic provider ignores the schema in v1. Note: v1 ships the semantic
passes on the configured (Anthropic) provider — routing `semantic_lint` to a
local model is one config edit, pending the probe-quality measurement named
in the consolidation design.
