# Claude Code memory

One command wires a Particles store into [Claude Code](https://code.claude.com)
as managed agent memory: standing context is **pushed** into every
session's context window at session start, and everything a session produced is
**harvested** into the corpus at session end — no agent cooperation required.
The agent is stateless compute; the store is managed storage; the integration
moves data between them on lifecycle events the agent does not control.

## Install

```bash
particles init claude-code
```

That one command:

1. **Merges two hook entries** into `~/.claude/settings.json` — a
   `SessionStart` hook running `particles hook session-start --store <handle>`
   and a `SessionEnd` hook running `particles hook session-end --store
   <handle>`, both with the absolute path of your `particles` executable so
   they work regardless of shell PATH. The merge is **marker-owned**: your
   existing settings and hooks are preserved byte-for-byte semantically;
   re-running `init` replaces exactly the Particles-owned entries
   (repair/upgrade); an unparseable settings file is an error, never a
   rewrite.
2. **Selects the memory store.** If `mcp.write.enabled_stores` names exactly
   one store, that's the one. Several → pass `--store <handle>`. **None (fresh
   install) → `init` creates and enables a `memory` store**: it initialises
   the store database (default `~/.particles/memory.db`) and appends
   `storage.stores.memory` + `mcp.write.enabled_stores` to your `config.yaml`
   under the same never-clobber discipline (parse, preserve everything else —
   comments included — append, verify).
3. **Provisions the state directory** `~/.particles/claude-code/` (the hook
   log, the projection manifest `memory.yaml`, its render snapshot, the
   one-deep `MEMORY.md.pre-render` backup, and the fold archive live there),
   writes the default `memory.yaml` when absent, and **seeds the projected
   region** into any existing `~/.claude/projects/*/memory/MEMORY.md` (an
   empty sentinel pair at the top of the file; all your content is preserved
   below it — see § The MEMORY.md projection).
4. **Installs the agent-onboarding skill files** into
   `~/.claude/skills/particles/` — three short Markdown files telling the agent
   which write verb to reach for, how to read effective confidence and the
   contested marker, and that ruling on a contradiction is *your* call, not
   its. Skip with `--no-skills`; manage them separately with
   `particles skills install` / `particles skills list`.
5. **Runs the first-run memory audit** over your existing
   `~/.claude/projects/*/memory/` directories: harvest → extract → one census
   report of what your agent's memory already contains — potential
   contradictions, likely-duplicate beliefs, probably-stale facts — with the
   cost estimate printed and confirmed first (see § The memory audit). Skip
   with `--no-audit`; a declined estimate or missing `ANTHROPIC_API_KEY`
   never fails the install.

Options: `--project` installs into the current repo's
`.claude/settings.local.json` (the *gitignored* local file — never the
committed `.claude/settings.json`, because the entries embed your store choice
and executable path) · `--dry-run` prints every file it would write ·
`--command <path>` overrides the hook executable for non-standard installs ·
`--json` prints a machine-readable result instead of prose.

### Letting the agent install it

`--json` exists so an agent can run the installer itself and *report* the
outcome rather than you reading over its shoulder:

```bash
particles init claude-code --json
```

stdout carries only the result object — the store, the scope, what was created,
the installed hook commands, and a `next_steps` list of what is left for you.
Because it is non-interactive it implies `--no-audit`, and names the standalone
`particles audit` under `next_steps` rather than skipping it silently.

This is deliberately the **local** half of self-onboarding. The agent mints no
credential and grants itself nothing: the write allowlist
(`mcp.write.enabled_stores`) is untouched, and everything it writes still lands
at agent trust, below yours. A network-exposed MCP write transport would need
its own auth model, which is a separate decision that has not been made.

## What the hooks do

**Session start — the digest push.** `hook session-start` renders the store's
memory digest (one line per
ACTIVE belief, ranked by effective confidence, contested beliefs flagged) and
injects it as `additionalContext`, so the agent's standing knowledge is *in
the context window* before the first prompt — not behind an MCP tool it may
forget to call. A `resume` session already replays its prior context, so the
push is skipped there; `startup`, `clear`, and `compact` get a fresh render.
Two budgets bound the injection: `mcp.recall.digest_max_beliefs` (default 200)
and `claude_code.digest_max_bytes` (default 24 000, truncated on a line
boundary with a disclosed footer).

**Session end — the harvest.** `hook session-end` deposits two kinds of
material — *harvest, don't ask*; the agent took no action to be remembered:

- **The session transcript, distilled — never raw.** A deterministic, LLM-free
  pass keeps user/assistant turns verbatim, elides each tool call to one line
  (`[tool: Bash — git status]`), and **drops tool results** (where payloads
  and secrets concentrate). A redaction pass then masks common credential
  shapes (`sk-…` keys, AWS access key IDs, `Bearer` headers, PEM blocks). The
  result lands as **one corpus entry per session**
  (`claude-code://session/<id>`, `CONVERSATION`, `APPEND_ONLY`) — a grown
  transcript appends a snapshot; an unchanged one is a content-hash no-op.
- **Changed memory files.** Each `*.md` under the project's auto-memory
  directory deposits as `LOCAL_MARKDOWN` / `MUTABLE`, so an edited `MEMORY.md`
  is re-extracted with the right staleness semantics. Claude Code's own
  auto-memory stays enabled — the integration harvests it rather than
  fighting it.
- **Catch-up sweep.** SessionEnd doesn't fire on a crash or SIGKILL, but the
  transcript persists on disk — so after handling the current session the hook
  re-checks up to `claude_code.harvest.catchup_limit` (default 5) recent
  transcripts and harvests any whose content moved. The corpus itself is the
  harvest state; a session missed because the store was unreachable is simply
  retried at the next session end.

**Extraction is deferred by default.** Deposits are LLM-free; beliefs
materialise when extraction runs — `particles extract --all-pending`, the
first-run audit, or (opt-in) `claude_code.harvest.extract_inline: true`
extracts inside the hook, bounded by
`claude_code.harvest.max_extract_entries_per_session` (default 3). Until then,
a belief learned in session *N* appears in the session-*N+1* digest only if
extraction ran in between.

## The MEMORY.md projection

With the hooks installed, `MEMORY.md` stops being an append-only scratch file:
a **sentinel-delimited region at the top of the file** is regenerated from the
store at the tail of every harvest cycle, so what the agent recalls
at session start is the *reconciled, ranked, decayed, contradiction-flagged*
store — not the raw accumulation:

```markdown
<!-- BEGIN PROJECTED: memory-index (manifest: ~/.particles/claude-code/memory.yaml) -->
- Owner prefers general mechanisms over per-genre extractor defaults `p-3f9a2c1d`
- DCO is enforced; every commit needs `git commit -s` `p-71b0de00`
- ⚠ contested — CI floors at Python 3.11 (vs. p-9c447100) `p-08d3e100`

<!-- sources: p-08d3e100, p-3f9a2c1d, p-71b0de00 -->
<!-- END PROJECTED: memory-index -->
```

**What the region is.** A deterministic ranked-bullet view — one line per
ACTIVE belief, ordered by effective confidence, contested beliefs flagged
rather than hidden, each line carrying its `p-<shortid>` drill-down handle
(resolve it with the MCP `particle_show` tool for full provenance). No LLM is
involved in the render: it is free, offline-capable, and byte-stable for a
given store, which is what lets the harvest recognise its own output and never
re-ingest it. Everything **outside** the region stays yours and Claude Code's:
the agent keeps appending memories below; the harvest picks them up. If you
edit *inside* the region, nothing is lost — the edited region is deposited as
authored input on the next cycle and reconciled into the store before the
region is re-rendered.

**Fold-and-archive (default-on).** Once agent-authored lines outside the
region have been harvested, the next cycle *moves* them — never deletes — to
the append-only archive `~/.particles/claude-code/MEMORY.archive.md` (itself
harvested as corpus input), leaving one pointer line behind. The file thus
converges to the projected region + not-yet-harvested lines + the pointer, and
duplication between an authored line and its projected consolidation is
bounded to one cycle. Opt out with
`agent_memory.projection.fold_authored_lines: false` in `config.yaml`; a
harvest that did not succeed never triggers a fold, and every folded line is
recoverable from the archive or the corpus.

**Editing the manifest.** The region is driven by a standard projection
manifest at `~/.particles/claude-code/memory.yaml` — yours to edit:

```yaml
name: memory-index
sections:
  - title: "Memory index"
    query: null            # no semantic refinement — rank purely by eff. conf.
    top_k: 60
    min_confidence: 0.30   # the noise floor
    render: bullets        # deterministic ranked bullets — never LLM prose
max_lines: 120             # document budget — headroom under the 200-line load cap
max_bytes: 16384
```

Add per-topic sections (`tags:` / `subjects:` per section), pin claims the
ranking misses with `select.allow`, exclude noise with `select.deny`
, or tighten the floor and budgets. `max_lines` / `max_bytes`
truncate in rank order — lowest effective confidence dropped first,
`select.allow` pins exempt. You can re-render on demand with
`particles project ~/.particles/claude-code/memory.yaml
~/.claude/projects/<project>/memory/MEMORY.md --splice memory-index
--without-synthesis`.

**Safety posture.** The splice runs only after the same cycle's harvest of
`MEMORY.md` succeeded; the write is atomic (temp file + rename); the pre-splice
file is backed up one-deep to `~/.particles/claude-code/MEMORY.md.pre-render`;
and damaged sentinels (a deleted `END` line, a duplicated pair) make the cycle
**refuse and skip** rather than regenerate your file. Deleting the region
opts that file out — re-run `particles init claude-code` to re-seed it. With
the projection active, the session-start digest push checks the region's
`<!-- sources: … -->` trailer first: if the loaded file already *is* the
current view it injects nothing, and if the store moved since the last render
it injects only the difference. Disable the whole feature with
`agent_memory.projection.enabled: false`.

**Git-versioned history (optional, off by default).** If you keep your memory
directory under git — `git init ~/.claude/projects/<project>/memory` — and set
`agent_memory.projection.git.enabled: true`, each render that changes the file
is committed for you with a structured message: a run id plus a ranking-delta
summary (which beliefs entered or left the index, and whether the top belief
changed). The result is a diffable, rollback-able history of the *view* — the
Letta-MemFS ergonomic — while the store stays the source of truth: a
`git revert` only rewinds the file, and the next render re-projects from the
store. The commit is a **bonus, never a requirement**: any git problem — not a
repo, nothing changed, no configured identity, a signing failure — is logged at
debug and skipped, and never affects the projection itself. Signing is **off by
default** (`--no-gpg-sign`) so an unattended session-end commit can't block on a
signing agent; set `git.sign: true` to respect your own `commit.gpgsign`, and
`git.author_name` / `git.author_email` to stamp a specific identity. Only files
under the memory directory are staged (never `git add -A`), and the internal
backup / snapshot / archive live outside it, so they never end up in your
history. See `config.yaml.sample` (or the operator guide's [Agent-memory
projection](../operator-guide/configuration.md#agent-memory-projection)
section) for the full knob set.

## The memory audit

Rot *prevention* (the hooks, the projection) pays off over weeks; rot
*detection* on the memories you already have is immediate:

```bash
particles audit ~/.claude/projects/<project>/memory     # harvest + extract + report
particles audit                                          # re-audit the store (no harvest)
```

```
Audited 23 memory files → 212 beliefs about 58 subjects.

  4 potential contradictions        (2 cross-file, 2 contested at extract time)
  11 likely-duplicate belief pairs  (unjudged similarity candidates; --judge to verify)
  7 probably-stale facts            (5 aged past their source's decay horizon, 2 expired)

  Also: 3 cited sources never captured · 6 beliefs have no resolvable subject
```

The audit **composes the existing finders** — lint, the store-wide
contradiction probe, `links suggest` duplicate candidates, the quality
dashboard — into complete per-class counts with a few leverage-ranked
exemplars each (claim text included), and every class ends with its next verb
(`particles review`, `particles curate --kind …`, `particles links suggest
--judge`, `particles deposit <url>`). The counts are hedged on purpose:
duplicates are unjudged cosine candidates until `--judge` runs the LLM
verdict pass, contradiction counts are LLM judgments over similarity-gated
candidates, and extractions from memory files carry self-reported (capped,
not benchmark-calibrated) confidence — the report says all of this rather
than overstate.

What to know before running it:

- **The deposits become your real store** — the audited corpus is the same
  store the hooks append to and `MEMORY.md` projects from. Re-running (or
  running `init` after `audit`, in either order) re-processes nothing:
  corpus dedup skips unchanged content and extraction skips COMPLETE
  snapshots.
- **The cost estimate always prints first.** Above
  `audit.confirm_call_threshold` estimated extraction calls (default 50) the
  CLI asks before spending; `--yes` pre-confirms, `--estimate` prints and
  exits without depositing anything, and a non-interactive run without
  `--yes` aborts with the estimate shown.
- **Transcripts are opt-in.** `--transcripts <dir>` harvests session
  `*.jsonl` transcripts newest-first, capped at
  `audit.transcript_max_entries` (default 20; `--max-entries` overrides) —
  they are large, LLM-priced, and lower-signal than the distilled memory
  files, so they never ride the first run silently.
- **No key, no silent clean bill.** With no `ANTHROPIC_API_KEY`, a harvest
  audit refuses before touching the store (extraction is the audit's
  substance); a re-audit of a populated store still runs the structural
  finders and duplicate candidates but says
  `contradiction check skipped: no API key` in the report.
- **The projection renders at the end.** When the MEMORY.md projection is
  enabled, a successful harvest+extract pass finishes by re-rendering the
  `memory-index` region — the activation moment leaves your `MEMORY.md`
  already consolidated.

`--output report.md` also writes the report to a file; `--format json` dumps
the full model; `--store <handle>` audits a named store. Presentation knobs
live under `audit:` in `config.yaml` (`exemplars_per_class`,
`transcript_max_entries`, `confirm_call_threshold`); detection thresholds
stay with their finders.

## Degradation and debugging

The hook verbs **never break a session**: on any failure — database missing,
engine unreachable, write-lock contention, or the internal
`claude_code.hook_deadline_seconds` deadline (default 10 s) — they log and
exit 0 with no output. A memory outage costs you an empty digest, not a hung
session start.

Every hook invocation appends one JSONL line (timestamp, event, session id,
outcome counts, duration, error) to the hook log at
`~/.particles/claude-code/hooks.jsonl` (`claude_code.hook_log_path`
overrides). Transcript *content* is never logged.

```bash
particles hook log --tail 20                                  # is this thing on?
particles hook session-start --store memory < sample.json     # debug loop
```

## Privacy posture

- **Local-only by default.** The hooks read local files and write the local
  store. With a remote engine configured (`engine.base_url`), `session-start`
  uses the remote digest freely (read-only), but `session-end` **refuses to
  ship transcripts off-machine** unless you set
  `claude_code.harvest.allow_remote: true`; refusals are logged and the
  catch-up sweep back-fills once enabled.
- **Distill-then-redact.** Only the distilled rendering is deposited — tool
  results never are. The pattern redaction is best-effort defence in depth,
  **not a guarantee**: review what scrolls through your sessions.
- **Stored, not ephemeral.** Deposits use normal archived mutability classes so
  excerpt-level provenance ("where did I learn this?") works. Prefer
  transcript-free beliefs? Set `claude_code.harvest.transcripts: false`
  (memory-file harvest only).

All hook knobs live under the `claude_code:` section of `config.yaml`, and the
projection's under `agent_memory.projection:` — see `config.yaml.sample`.

## Uninstall

```bash
particles init claude-code --remove
```

Removes exactly the Particles-owned hook entries (everything else in your
settings survives) and reverts the store auto-create **only while the store is
still empty** — a store holding data is never deleted. It also deletes the
`~/.claude/skills/particles/` subdirectory and nothing beside it, so any skill
files of your own in that directory are untouched. The state directory
(hook log history) is kept; delete it manually if you want it gone.
