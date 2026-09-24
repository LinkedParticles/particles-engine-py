# Claude Code memory

One command wires a Particles store into [Claude Code](https://code.claude.com)
as managed agent memory: a **small, ranked digest** of standing
context is pushed into every session's context window at session start (the
top beliefs by effective confidence, bounded to a few thousand tokens, never
the whole store), and everything a session produced is **harvested** into the
corpus at session end. No agent cooperation required.
The agent is stateless compute; the store is managed storage; the integration
moves data between them on lifecycle events the agent does not control.

If you want the store exposed as tools an agent calls instead, the two other
routes are the native MCP server
([Querying → MCP server](querying.md#mcp-server)) and the
[reference memory-server swap](memory-server-swap.md).

**Not using Claude Code?** The MCP server is harness-agnostic: Codex, Cursor,
Windsurf, Zed, OpenCode and Claude Desktop all read a store today, and the
digest can be projected into the instructions file any of them already loads.
[Connecting your coding agent](coding-agents.md) has a config block per harness
and an honest table of which capabilities each one gets.

## What this buys you

A memory file accumulates lines; the store accumulates *claims*, and the
difference shows up exactly where file memory hurts:

- **Rules born from incidents keep their provenance.** "Commit, push, and
  deploy are three separate go-aheads" is a typical agent memory: a rule
  decreed after something went wrong once. As a particle it carries *when* it
  was asserted and *which session* it came from, so a future session asking
  "does this still apply, and what was it protecting against?" is one hop
  from the answer instead of archaeology. And if a later session writes
  "push automatically after each commit," that isn't a silent second line in
  a file; it's a detected contradiction, queued for your ruling.
- **Flaky-infrastructure facts age honestly.** "Sometimes the tailscale link
  between the two servers gets stuck in a slow mode" is true the day it's
  written and misleading the month after it's fixed. In a file it lives
  forever; here its effective confidence decays with age, `lint` flags it
  stale, and the fix *supersedes* it: the old claim retired with a pointer
  to what replaced it, not erased.
- **The store can audit what your files can't.** The first run reads the
  memory you already have and reports what's lurking in it (contradictions,
  likely duplicates, probably-stale facts, cited-but-never-captured
  sources); see [The memory audit](#the-memory-audit). Those are questions a
  directory of markdown cannot answer about itself.

## Install

```bash
particles init claude-code
```

**Before you run it:** by default this gives *every* Claude Code project on
the machine one shared memory, and the first-run audit reads all of them. See
[One store serves every project](#one-store-serves-every-project) for what
that means and how to narrow it.

That one command:

1. **Merges two hook entries** into `~/.claude/settings.json`: a
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
   under the same never-clobber discipline (parse, preserve everything else,
   comments included, append, verify).
3. **Provisions the state directory** `~/.particles/claude-code/` (the hook
   log, the projection manifest `memory.yaml`, and under
   `projects/<project>/` each project's render snapshot, one-deep
   `MEMORY.md.pre-render` backup, and fold archive live there),
   writes the default `memory.yaml` when absent, and **seeds the projected
   region** into any existing `~/.claude/projects/*/memory/MEMORY.md` (an
   empty sentinel pair at the top of the file; all your content is preserved
   below it; see § The MEMORY.md projection).
4. **Installs the agent-onboarding skill files** into
   `~/.claude/skills/particles/`: three short Markdown files telling the agent
   which write verb to reach for, how to read effective confidence and the
   contested marker, and that ruling on a contradiction is *your* call, not
   its. Skip with `--no-skills`; manage them separately with
   `particles skills install` / `particles skills list`.
5. **Runs the first-run memory audit** over your existing
   `~/.claude/projects/*/memory/` directories: harvest → extract → one census
   report of what your agent's memory already contains (potential
   contradictions, likely-duplicate beliefs, probably-stale facts), with the
   cost estimate printed and confirmed first (see § The memory audit). Skip
   with `--no-audit`; a declined estimate or missing `ANTHROPIC_API_KEY`
   never fails the install.

Options: `--project` installs into the current repo's
`.claude/settings.local.json` (the *gitignored* local file, never the
committed `.claude/settings.json`, because the entries embed your store choice
and executable path); it changes only *where the hooks are installed*, not
what steps 3 and 5 read (see
[One store serves every project](#one-store-serves-every-project)) ·
`--dry-run` prints every file it would write ·
`--command <path>` overrides the hook executable for non-standard installs ·
`--json` prints a machine-readable result instead of prose.

### Letting the agent install it

`--json` exists so an agent can run the installer itself and *report* the
outcome rather than you reading over its shoulder:

```bash
particles init claude-code --json
```

stdout carries only the result object: the store, the scope, what was created,
the installed hook commands, and a `next_steps` list of what is left for you.
Because it is non-interactive it implies `--no-audit`, and names the standalone
`particles audit` under `next_steps` rather than skipping it silently.

This is deliberately the **local** half of self-onboarding. The agent mints no
credential and grants itself nothing: the write allowlist
(`mcp.write.enabled_stores`) is untouched, and everything it writes still lands
at agent trust, below yours. A network-exposed MCP write transport would need
its own auth model, which is a separate decision that has not been made.

## One store serves every project

`particles init claude-code` gives every Claude Code project on this machine
the same store. The hooks go into your user-level settings, so they fire in
every project, and they all name one store:

- **The first-run audit reads every project.** It harvests every
  `~/.claude/projects/*/memory/` directory into that one store, not only the
  project you ran `init` from.
- **The harvest follows you.** Each session end deposits that session's
  transcript and that project's memory files into the same store, whichever
  project the session was in.

What a session is **shown** from that store is a separate choice, and you make
it with one line of config.

### Choose what a session sees

```yaml
claude_code:
  observer_scope: store     # the default: every session sees the whole store
  # observer_scope: project # a session sees global beliefs plus its own project's
```

With `store`, the session-start digest ranks *all* of the store's ACTIVE
beliefs, so a belief learned in project A appears in project B's sessions, and
the `MEMORY.md` projection writes that same view into each project's file. For
a rule like "every commit needs a sign-off" that is what you want.

With `project`, a session reads the same store **through its project**. It
sees:

- **global beliefs**: anything you deposited by hand (a web page, a document),
  your user-level rule files (`~/.claude/CLAUDE.md`), and anything you have
  widened (below); plus
- **beliefs observed in this project**: those with at least one source that was
  harvested here. A belief seen in two projects is in view for both.

It does not see another project's beliefs, and it does not see harvested
material that could not be attributed to any project. Nothing is stored on the
belief to make this work: a belief's projects are read from the `project:` tag
on the sources it came from, each time it is read. So this is a reading lens,
not a partition. Storage is shared, maintenance (`lint`, `review`, `curate`,
the nightly consolidation) still sees everything, and switching back is the
same one line. The digest header says which view it is and how many beliefs
are in scope, and the `MEMORY.md` region ends with a comment naming its
observer. **It is a relevance scope, not access control:** one owner, one
store, and the whole store is one flag away.

A git worktree is the same project as its repository (§ What the hooks do), so
a session in a worktree sees its repository's view.

**Before it takes effect, the store needs its project keys brought up to
date**, because older versions stamped a per-worktree name, or nothing:

```bash
particles memory rescope --dry-run     # what would change; writes nothing
particles memory rescope               # adds each source's project key; never removes a tag
```

`particles init claude-code` runs it for you on every install or re-run. Until
it has run once on a store, `observer_scope: project` stays store-wide and the
digest says so, so an upgrade never silently empties your digest;
`particles hook doctor` reports that state. `rescope` lists two things worth
reading. *Unattributed* sources were harvested but cannot be traced to a
project (typically transcripts audited before this existed, whose session files
are gone); they are in view for no project until you attribute them:

```bash
particles memory rescope --default-key .                 # all of them: this directory's project
particles memory rescope --assign <entry-id> <project>   # or one at a time
```

And sources whose only project no longer exists on this machine are listed so
you can give them a live one.

**Make a belief global.** "This rule I learned in one project is how I work
everywhere" is your call, so it is yours to record:

```bash
particles memory widen p-4fbcc320            # one belief
particles memory widen --entry <entry-id>    # every belief from one source
particles memory widen p-4fbcc320 --revoke   # take it back
```

The belief and its sources are untouched; the widening is a standing note the
lens consults. There is deliberately no way for the agent to do this itself.

**The MCP server has its own switch**, because it is launched per session and
is not specific to Claude Code. Register it with `--project-observer cwd` and
it is bound to the project of the directory it was started in:

```bash
claude mcp add --scope user particles -- particles mcp serve --project-observer cwd
```

Its `query`, `particles_list`, `particle_search` and `graph_view` then return
that project's view, and say so in an `observer` field. Passing
`all_projects: true` to any of them reads the whole store for that one call,
and says that too. Looking a belief up by id (`particle_show`) is never
filtered. What the agent writes through a bound server is attributed to that
project: a deposit carries the project's key whatever tags the agent passes,
and an assertion may cite an existing source only if it is one of that
project's. Leave the flag off and the server behaves exactly as before.

**Two projects that disagree both keep their answer.** Once the store has
been rescoped, a newer claim retires an older one only when every project that
currently states the older claim also states the newer one. If project A's
memory says the default branch is `main` and project B's says `master`, both
beliefs stay: each session sees its own, and neither project's edit can retire
the other's. A project still updates itself: when A's memory changes `main` to
`trunk`, A's old value is retired as before. A line both projects state stays
while either still states it, and a project's view follows what its memory
file says *now*: a line the file has dropped leaves that project's view even
while the other project keeps it.

Each such disagreement is recorded as a contradiction between the two beliefs,
so nothing is decided silently. `particles lint` lists it as a `CONTRADICTION`
finding (it needs no model call to do so), and
`particles links list <id> --kind contradicts` shows the pair. You are the
one who sees both projects, so resolving it is yours: widen one belief, retract
it, or leave both. A global belief of yours (a hand-written note, a user-level
rule) is never retired by a project's memory: when a project contradicts it,
the disagreement goes to `particles review` instead.

Measured on the two-project fixture (`docs/benchmarks/observer-scope.md`),
every line each project states is in view for that project, where before
this a shared-subject fact was out of view about a third of the time. `store`
remains the default until you choose otherwise.

### Other ways to narrow it

- **Choose what the first run reads.** `particles init claude-code --no-audit`
  installs the hooks and audits nothing; then run
  `particles audit ~/.claude/projects/<project>/memory` on only the
  directories you pick. This limits what arrives up front. The hooks still
  harvest each project as you work in it.
- **Give one repository its own store**, when what you need is a real boundary
  (a client's code, something you will share) rather than relevance. From that
  repository:

    ```bash
    particles init claude-code --project --store <handle>
    ```

    The hooks land in that repo's `.claude/settings.local.json` and name
    `<handle>` (created if it does not exist), so sessions in that repo read
    from and harvest into that store alone. A `--project` install is about
    this project only: it seeds the projected region into **this project's**
    `MEMORY.md`, audits **this project's** memory directory, and registers
    **this project's** rule documents (not your user-level ones). Two things
    to know:

    - It does not switch off a user-level install. Claude Code runs the hooks
      from every settings file that applies, so with both installed that
      repo's sessions get both digests and are harvested into both stores.
      Run `particles init claude-code --remove` first if you want per-repo
      stores only.
    - Lines folded out of a project's `MEMORY.md` are archived per project
      (§ The MEMORY.md projection) and harvested into whichever store that
      project's hooks name.

## What the hooks do

**Session start: the digest push.** `hook session-start` renders the store's
memory digest (one line per
ACTIVE belief, ranked by effective confidence, contested beliefs flagged) and
injects it as `additionalContext`, so the agent's standing knowledge is *in
the context window* before the first prompt, not behind an MCP tool it may
forget to call. A `resume` session already replays its prior context, so the
push is skipped there; `startup`, `clear`, and `compact` get a fresh render.
Two budgets bound the injection: `mcp.recall.digest_max_beliefs` (default 200)
and `claude_code.digest_max_bytes` (default 24 000, truncated on a line
boundary with a disclosed footer).

The push is deliberately small because rules and facts want different
treatment. A *rule* ("commit, push, and deploy are separate go-aheads") wants
to be in the prompt every session; that reliable presence is the digest's
job, and why it ranks standing, high-confidence beliefs first. A *fact* (the
port a service ran on in May) wants to be **retrievable, not resident**;
that's the MCP query path, and it's why the digest never tries to carry the
store. Ranking composes effective confidence with usage (beliefs that keep
earning recall rise), so the budget is spent on what sessions
actually use, not on whatever was written most recently.

**Session end: the harvest.** `hook session-end` deposits two kinds of
material. This is *harvest, don't ask*; the agent took no action to be remembered:

- **The session transcript, distilled, never raw.** A deterministic, LLM-free
  pass keeps user/assistant turns verbatim, elides each tool call to one line
  (`[tool: Bash — git status]`), and **drops tool results** (where payloads
  and secrets concentrate). A redaction pass then masks common credential
  shapes (`sk-…` keys, AWS access key IDs, `Bearer` headers, PEM blocks). The
  result lands as **one corpus entry per session**
  (`claude-code://session/<id>`, `CONVERSATION`, `APPEND_ONLY`); a grown
  transcript appends a snapshot; an unchanged one is a content-hash no-op.
- **Changed memory files.** Each `*.md` under the project's auto-memory
  directory deposits as `LOCAL_MARKDOWN` / `MUTABLE`, so an edited `MEMORY.md`
  is re-extracted with the right staleness semantics. Claude Code's own
  auto-memory stays enabled; the integration harvests it rather than
  fighting it. **A git worktree shares its repository's memory.** Claude Code
  keeps transcripts per working directory but auto-memory per repository, so
  a session in a linked worktree harvests, and re-renders, the *repository's*
  memory directory (`~/.claude/projects/<repository>/memory/`), and its
  deposits carry the repository's `project:` tag. The hook works this out
  from the session's launch directory by reading the worktree's `.git` file;
  it never runs `git`.
- **Catch-up sweep.** SessionEnd doesn't fire on a crash or SIGKILL, but the
  transcript persists on disk, so after handling the current session the hook
  re-checks up to `claude_code.harvest.catchup_limit` (default 5) recent
  transcripts and harvests any whose content moved. The corpus itself is the
  harvest state; a session missed because the store was unreachable is simply
  retried at the next session end.

**Extraction is deferred by default.** Deposits are LLM-free; beliefs
materialise when extraction runs: `particles extract --all-pending`, the
first-run audit, or (opt-in) `claude_code.harvest.extract_inline: true`
extracts inside the hook, bounded by
`claude_code.harvest.max_extract_entries_per_session` (default 3). Until then,
a belief learned in session *N* appears in the session-*N+1* digest only if
extraction ran in between.

**What extraction costs.** An unchanged memory file costs nothing: it is not
even re-deposited. A memory file edited in several sessions between two
extraction passes costs one call, not one per edit, because only its newest
pending version is extracted (see [Mutable local
sources](../operator-guide/mutable-local-sources.md#only-the-newest-pending-version-is-extracted)).
Session transcripts are the larger share of the bill, and a long transcript
that grew is re-extracted only from where it changed.

## The MEMORY.md projection

With the hooks installed, `MEMORY.md` stops being an append-only scratch file:
a **sentinel-delimited region at the top of the file** is regenerated from the
store at the tail of every harvest cycle, so what the agent recalls
at session start is the *reconciled, ranked, decayed, contradiction-flagged*
store, not the raw accumulation:

```markdown
<!-- BEGIN PROJECTED: memory-index (manifest: ~/.particles/claude-code/memory.yaml) -->
- Owner prefers general mechanisms over per-genre extractor defaults `p-3f9a2c1d`
- DCO is enforced; every commit needs `git commit -s` `p-71b0de00`
- ⚠ contested — CI floors at Python 3.11 (vs. p-9c447100) `p-08d3e100`

<!-- sources: p-08d3e100, p-3f9a2c1d, p-71b0de00 -->
<!-- END PROJECTED: memory-index -->
```

**What the region is.** A deterministic ranked-bullet view: one line per
ACTIVE belief, ordered by effective confidence, contested beliefs flagged
rather than hidden, each line carrying its `p-<shortid>` drill-down handle
(resolve it with the MCP `particle_show` tool for full provenance). No LLM is
involved in the render: it is free, offline-capable, and byte-stable for a
given store, which is what lets the harvest recognise its own output and never
re-ingest it. Everything **outside** the region stays yours and Claude Code's:
the agent keeps appending memories below; the harvest picks them up. If you
edit *inside* the region, nothing is lost; the edited region is deposited as
authored input on the next cycle and reconciled into the store before the
region is re-rendered.

**Fold-and-archive (default-on).** Once agent-authored lines outside the
region have been harvested, the next cycle *moves* them (never deletes) to
that project's append-only archive,
`~/.particles/claude-code/projects/<project>/MEMORY.archive.md` (itself
harvested as corpus input, under that project's key), leaving one pointer line behind. The file thus
converges to the projected region + not-yet-harvested lines + the pointer, and
duplication between an authored line and its projected consolidation is
bounded to one cycle. Opt out with
`agent_memory.projection.fold_authored_lines: false` in `config.yaml`; a
harvest that did not succeed never triggers a fold, and every folded line is
recoverable from the archive or the corpus.

**Editing the manifest.** The region is driven by a standard projection
manifest at `~/.particles/claude-code/memory.yaml`, yours to edit:

```yaml
name: memory-index
sections:
  - title: "Memory index"
    query: null            # no semantic refinement; rank purely by eff. conf.
    top_k: 60
    min_confidence: 0.30   # the noise floor
    render: bullets        # deterministic ranked bullets, never LLM prose
max_lines: 120             # document budget: headroom under the 200-line load cap
max_bytes: 16384
```

Add per-topic sections (`tags:` / `subjects:` per section), pin claims the
ranking misses with `select.allow`, exclude noise with `select.deny`,
or tighten the floor and budgets. `max_lines` / `max_bytes`
truncate in rank order, lowest effective confidence dropped first,
`select.allow` pins exempt. You can re-render on demand with
`particles project ~/.particles/claude-code/memory.yaml
~/.claude/projects/<project>/memory/MEMORY.md --splice memory-index
--without-synthesis`.

**Safety posture.** The splice runs only after the same cycle's harvest of
`MEMORY.md` succeeded; the write is atomic (temp file + rename); the pre-splice
file is backed up one-deep to
`~/.particles/claude-code/projects/<project>/MEMORY.md.pre-render` (one backup
per project, so another project's session never overwrites yours);
and damaged sentinels (a deleted `END` line, a duplicated pair) make the cycle
**refuse and skip** rather than regenerate your file. Deleting the region
opts that file out; re-run `particles init claude-code` to re-seed it. The
region is compared with **that project's own last render** to decide whether
anyone edited it: an unedited region is stripped before harvest, so the
store's rendered bullets never come back in as if you had written them, and an
edit you make inside the region is still harvested as yours. With
the projection active, the session-start digest push checks the region's
`<!-- sources: … -->` trailer first: if the loaded file already *is* the
current view it injects nothing, and if the store moved since the last render
it injects only the difference. Disable the whole feature with
`agent_memory.projection.enabled: false`.

**Git-versioned history (optional, off by default).** If you keep your memory
directory under git (`git init ~/.claude/projects/<project>/memory`) and set
`agent_memory.projection.git.enabled: true`, each render that changes the file
is committed for you with a structured message: a run id plus a ranking-delta
summary (which beliefs entered or left the index, and whether the top belief
changed). The result is a diffable, rollback-able history of the *view* (the
Letta-MemFS ergonomic) while the store stays the source of truth: a
`git revert` only rewinds the file, and the next render re-projects from the
store. The commit is a **bonus, never a requirement**: any git problem (not a
repo, nothing changed, no configured identity, a signing failure) is logged at
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

The audit is a first-run census. To keep it running unattended, and to run
the reconcile, curation and projection passes alongside it, see
[Operator guide → scheduled consolidation](../operator-guide/scheduled-consolidation.md).
Contradictions and duplicates it names are resolved with
[lint and review](../operator-guide/lint-and-review.md) and
[co-evidential curation](../operator-guide/co-evidential.md).

```
Audited 23 memory files → 212 beliefs about 58 subjects.

  4 potential contradictions        (2 cross-file, 2 contested at extract time)
  11 likely-duplicate belief pairs  (unjudged similarity candidates; --judge to verify)
  7 probably-stale facts            (5 aged past their source's decay horizon, 2 expired)

  Also: 3 cited sources never captured · 6 beliefs have no resolvable subject
```

The audit **composes the existing finders** (lint, the store-wide
contradiction probe, `links suggest` duplicate candidates, the quality
dashboard) into complete per-class counts with a few leverage-ranked
exemplars each (claim text included), and every class ends with its next verb
(`particles review`, `particles curate --kind …`, `particles links suggest
--judge`, `particles deposit <url>`). The counts are hedged on purpose:
duplicates are unjudged cosine candidates until `--judge` runs the LLM
verdict pass, contradiction counts are LLM judgments over similarity-gated
candidates, and extractions from memory files carry self-reported (capped,
not benchmark-calibrated) confidence. The report says all of this rather
than overstate.

What to know before running it:

- **The deposits become your real store.** The audited corpus is the same
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
  `audit.transcript_max_entries` (default 20; `--max-entries` overrides).
  They are large, LLM-priced, and lower-signal than the distilled memory
  files, so they never ride the first run silently.
- **No key, no silent clean bill.** With no `ANTHROPIC_API_KEY`, a harvest
  audit refuses before touching the store (extraction is the audit's
  substance); a re-audit of a populated store still runs the structural
  finders and duplicate candidates but says
  `contradiction check skipped: no API key` in the report.
- **The projection renders at the end.** When the MEMORY.md projection is
  enabled, a successful harvest+extract pass finishes by re-rendering the
  `memory-index` region, so the activation moment leaves your `MEMORY.md`
  already consolidated.

`--output report.md` also writes the report to a file; `--format json` dumps
the full model; `--store <handle>` audits a named store. Presentation knobs
live under `audit:` in `config.yaml` (`exemplars_per_class`,
`transcript_max_entries`, `confirm_call_threshold`); detection thresholds
stay with their finders.

## Degradation and debugging

The hook verbs **never break a session**: on any failure (database missing,
engine unreachable, write-lock contention, or the internal
`claude_code.hook_deadline_seconds` deadline, default 10 s) they log and
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

Each `session-end` line also records the `project` it resolved and how
(`project_resolved_from`: `repository`, or `transcript-dir` when the launch
directory could not be identified and the hook fell back to the directory
holding the transcript).

`particles hook doctor --store <handle>` prints the memory directory a session
started from the current directory writes to. It also lists **stray memory
directories**: before 1.147.1 a session in a linked worktree had the
projection create a `memory/MEMORY.md` beside its transcripts, a file Claude
Code never reads. They are harmless and are no longer created, harvested, or
re-rendered; `doctor` names them so you can delete them, and never deletes
anything itself.

## Privacy posture

- **Local-only by default.** The hooks read local files and write the local
  store. With a remote engine configured (`engine.base_url`), `session-start`
  uses the remote digest freely (read-only), but `session-end` **refuses to
  ship transcripts off-machine** unless you set
  `claude_code.harvest.allow_remote: true`; refusals are logged and the
  catch-up sweep back-fills once enabled.
- **Distill-then-redact.** Only the distilled rendering is deposited; tool
  results never are. The pattern redaction is best-effort defence in depth,
  **not a guarantee**: review what scrolls through your sessions.
- **Stored, not ephemeral.** Deposits use normal archived mutability classes so
  excerpt-level provenance ("where did I learn this?") works. Prefer
  transcript-free beliefs? Set `claude_code.harvest.transcripts: false`
  (memory-file harvest only).

All hook knobs live under the `claude_code:` section of `config.yaml`, and the
projection's under `agent_memory.projection:`; see `config.yaml.sample` and
[Operator guide → configuration](../operator-guide/configuration.md#agent-memory-projection).

One gap this integration does not close on its own: your project's *rule*
files (`AGENTS.md`, `CLAUDE.md`) are frozen at deposit unless you opt them in,
so the store can keep asserting a rule you have since changed. See
[Refreshing mutable local sources](../operator-guide/mutable-local-sources.md).

## Uninstall

```bash
particles init claude-code --remove
```

Removes exactly the Particles-owned hook entries (everything else in your
settings survives) and reverts the store auto-create **only while the store is
still empty**; a store holding data is never deleted. It also deletes the
`~/.claude/skills/particles/` subdirectory and nothing beside it, so any skill
files of your own in that directory are untouched. The state directory
(hook log history) is kept; delete it manually if you want it gone.
