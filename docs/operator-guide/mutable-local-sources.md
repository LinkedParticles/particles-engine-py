# Refreshing mutable local sources

A file you deposit is frozen at deposit time. That is the right default for an
archived PDF, and the wrong one for a file you keep editing — a project's
`AGENTS.md`, a `CLAUDE.md`, an agent-memory note. Those get revised, and without
a refresh loop the store keeps asserting whatever the file said the day it was
deposited, at full confidence, long after the rule changed.

This page covers opting such a file in, tracking the whole set of
operating documents at once, what happens when one changes, and the
one-time cleanup for stores that predate the feature.

Why bother: a store fed only by conversation harvest ends up holding claims
*about* your rules rather than the rules themselves. Measured on this project's
own store before — 34 ACTIVE particles mentioned the
never-prepend-`export PATH` rule, and not one of them stated it.

## What "refreshable" means

An entry is re-checked when **both** are true:

| Property | Value | Why |
|---|---|---|
| `fetch_policy` | `LAZY` | The gate. Local deposits default to `NEVER`, so nothing starts refreshing on upgrade — you opt in per source. |
| `mutability` | `MUTABLE` | The promise that content may change anywhere, so a new version *replaces* the previous one rather than adding to it. |

Both are promises you make about the source; the SDK does not guess them.

## Opting a rule file in

```bash
particles deposit ./AGENTS.md --mutability MUTABLE --fetch-policy LAZY
```

```
entry_id:    bb6910e9-856c-4bc5-9de0-1e9f7ad19d48
snapshot_id: cdcf456c-21d3-46a5-a897-b32b9d664fd4
```

Both flags are **local-file only**. A URL deposit takes its class from the
importer that handles it, and a deposit against a remote engine uploads the
bytes rather than the path — the engine has no file to re-read later — so both
cases refuse the flags rather than silently ignoring them.

## Tracking the whole set at once

Depositing rule files one flag-pair at a time gets tedious, and the point of
having them in the store is that *all* of them are there. `particles rules`
 treats the operating documents as a set:

```bash
particles rules
```

```
rule_sources.paths is empty — discovered roots: /path/to/proj, /Users/you/.claude
  —         not tracked                       /path/to/proj/AGENTS.md
  —         not tracked                       /path/to/proj/docs/AGENTS.md

2 rule source(s): 0 tracked, 0 enrolled in the refresh loop (fetch_policy=LAZY),
0 exempt from the document-meta exclusion.
Run `particles rules sync` to track the rest.
```

With `rule_sources.paths` empty the set is **discovered**: the nearest ancestor
of the working directory containing a `.git` entry, plus `~/.claude`. Setting
`paths` turns discovery off and the list is taken literally. Either way the
resolved set is printed before anything is written.

```bash
particles rules sync
```

```
  registered  /path/to/proj/AGENTS.md
  registered  /path/to/proj/docs/AGENTS.md

2 rule source(s): 2 new/changed, 0 unchanged, 0 failed.
Run `particles extract --all-pending` to turn the new snapshots into beliefs.
```

### Why the tag matters beyond addressability

The `rule-file` tag each entry carries is also what lifts the document-meta exclusion for that source. Without it, a rules
document's *prescriptions* — "commits must be made with `uv run git commit -s`"
— are liable to be classified `DOCUMENT_META` and kept off the default query
and projection surfaces, which delivers the descriptive half of a rules
document and silently drops the imperative half.

`rules sync` re-applies that exemption to particles a previous extraction
already wrote, because the stamp is a deterministic function of the entry's
tags rather than a re-classification:

```bash
particles rules sync --restamp-only
```

```
  scope exemption applied to 178 particle(s) across 23 entry(ies)

--restamp-only: 23 rule source(s) checked, 178 particle(s) restamped.
```

It is idempotent — a second run reports 0 — and it never touches
`confidence`. See
[Troubleshooting → My `AGENTS.md` rules were extracted but never show
up](troubleshooting.md#my-agentsmd-rules-were-extracted-but-never-show-up).


Each file lands `LOCAL_MARKDOWN` / `MUTABLE` / `LAZY` and tagged `rule-file`, so
it is in the refresh loop from the first sweep:

```
  4022d333  lazy  · 1 snap · 2026-07-24       /path/to/proj/AGENTS.md
  e337ac1b  lazy  · 1 snap · 2026-07-24       /path/to/proj/docs/AGENTS.md

2 rule source(s): 2 tracked, 2 enrolled in the refresh loop (fetch_policy=LAZY).
```

Re-running is a no-op — identity is the `file://` path, so unchanged content
writes nothing. `--dry-run` resolves and prints without touching the store, and
`particles rules sync <path>…` overrides the configured set for one run.

`particles init claude-code` runs this step for you on install; a failure there
is reported and never fails the install.

Knobs, all under `rule_sources` in `config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Rule-source tracking on/off |
| `paths` | `[]` | Registered files/directories; empty ⇒ discover |
| `filenames` | `AGENTS.md`, `CLAUDE.md` | Basenames collected when walking a directory |
| `max_depth` | `4` | Walk depth relative to each root |
| `max_files` | `200` | Cap per resolution; truncation is always reported |
| `exclude_dirs` | `.git`, `.venv`, `node_modules`, `worktrees`, … | Directory names skipped at any level |

!!! note "`worktrees` is excluded on purpose"
    An agent worktree is a full checkout carrying its own copy of every rule
    document. Registering those would track transient files that vanish with the
    worktree, and whose content is either identical to the canonical file or an
    uncommitted branch draft. The canonical checkout is the source.

### A rule file that carries a projected region

If a rule file self-hosts a projected region (the pattern — sentinels
that the store re-renders into), it is deposited with that region stripped: the
corpus must never archive the store's own output. Its snapshot bytes therefore
differ from the file's bytes, and the refresh ladder above compares exactly
those. Such a file is tracked but held out of the loop, and the report says so:

```
  4022d333  lazy  · 3 snap · 2026-07-24       /path/to/proj/AGENTS.md
  75b602d0  never · 1 snap · 2026-07-24       /path/to/proj/CLAUDE.md

3 rule source(s): 3 tracked, 2 enrolled in the refresh loop (fetch_policy=LAZY).
1 file(s) carry a projected region, so their deposited body differs from their
bytes; re-run `particles rules sync` to refresh those.
```

Keep those current with `particles rules sync`, which applies the same strip.
The rule generalises: a source whose deposit transforms the bytes cannot be
refreshed by a mechanism that compares them.

## Checking for changes

```bash
particles corpus refresh
```

With nothing edited:

```
Checked 1 local source(s): 0 changed, 1 unchanged, 0 missing.
```

After editing the file:

```
  bb6910e9  changed → snapshot 967909e0 PENDING

Checked 1 local source(s): 1 changed, 0 unchanged, 0 missing.
Run `particles extract --all-pending` to turn the new snapshots into beliefs.
```

The check is a two-step ladder, so a sweep over a large vault stays cheap:

1. **`stat` the file.** If its mtime equals the one recorded on the latest
   snapshot, nothing is read and no row is written. This is the common case and
   costs about a microsecond per entry.
2. **Hash the bytes.** Only reached when the mtime moved. A matching SHA-256
   writes a zero-byte `REVISIT` snapshot ("as of now, still the same content");
   a differing one writes a `RESPONSE` snapshot marked `PENDING`.

There is no network involved — no DNS, no redirects, no SSRF surface.

!!! note "An unchanged mtime is a heuristic, not proof"
    Every ordinary editor, `git` operation, and formatter moves a file's mtime,
    and the comparison treats *any* difference — including one that moves
    backwards, as a backup restore can — as "re-read". But a tool that
    deliberately preserves mtime while changing content (`touch -r`) will be
    missed. `particles corpus refresh --force` skips both the mtime check and
    the per-source-type re-fetch floor, and is the escape hatch for that case.

## What happens to the old beliefs

Extraction is what closes the loop:

```bash
particles extract --all-pending
```

```
Extracting 2 pending snapshot(s)…
  bb6910e9…/cdcf456c…  2 particles
  bb6910e9…/967909e0…  4 particles
```

Once the new snapshot is extracted, the beliefs still anchored to the snapshot
it replaced are demoted `ACTIVE → PROVENANCE_STALE`. For a rule file whose
guidance was corrected, that looks like this:

```
1dd8fa39  ACTIVE            snap 967909e0  Using `uv run git commit` puts `.venv/bin` on PATH…
4f19fff3  ACTIVE            snap 967909e0  One should never prepend `PATH="$PWD/.venv/bin:$PATH"`…
b1869fab  ACTIVE            snap 967909e0  The prescribed method for committing is `uv run git commit`.
fede4139  ACTIVE            snap 967909e0  Prepending `PATH="$PWD/.venv/bin:$PATH"` trips a shell-safety check.
57bfccb5  PROVENANCE_STALE  snap cdcf456c  Prefixing commands with `PATH=…` ensures that pre-commit is found.
d8fa22ef  PROVENANCE_STALE  snap cdcf456c  The pre-commit tool is located within the `.venv/bin` directory…
```

Two properties of that demotion are worth knowing:

- **It keys on document generation, not on meaning.** The lint's contradiction
  probe cannot do this job: a rule that was simply *deleted* produces nothing to
  contradict it, and "prefixing PATH works" and "prefixing PATH is forbidden"
  are both true and not logically inconsistent — only the document version says
  which one is operative. Because it needs no semantic judgement, the demotion
  costs no LLM calls.
- **It runs after extraction, not before.** A paragraph you did not touch keeps
  its particle via chunk-hash carry-forward, so an edit to one section does not
  re-pay for the rest of the file.

`PROVENANCE_STALE` is a demotion, not a delete: content, provenance, and
confidence are intact, the particles drop out of query results and the
projection, and they surface in `particles curate` if you want to look at them.

## Running it unattended

The scheduled consolidation cycle runs the same sweep as its first pass, ahead
of extraction — so an edit made today is re-snapshotted, extracted, reconciled,
and reflected in `MEMORY.md` in a single nightly run. See
[Scheduled consolidation](scheduled-consolidation.md) for the launchd/cron
recipe; nothing extra is needed to enable the refresh.

It appears in the run report as:

```
  local sources    12 checked, 1 changed → re-extracting
```

Unlike the LLM-priced passes, this one makes no model calls, so it still runs on
a `--structural-only` night: a run with no API key will not extract, but it will
still notice that the rules moved.

Knobs, all under `local_refresh` in `config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | The consolidation pass on/off |
| `max_entries` | `200` | Per-run cap, oldest-entry-first |
| `follow_symlinks` | `false` | Whether a path that has since become a symlink is followed |

## One-time cleanup for existing stores

The demotion above only fires on snapshots extracted from now on. A store that
has been running for a while — in particular one using the Claude Code session
hooks, which have always re-deposited memory files as `MUTABLE` — carries a
backlog of beliefs from document versions that were superseded long ago and were
never retired.

Preview and apply it with:

```bash
particles corpus refresh --backfill-cascade
```

```
MUTABLE entries with more than one snapshot: 1
2 ACTIVE particle(s) across 1 entry are anchored to a superseded snapshot:
       2  …/project/AGENTS.md

They will be demoted ACTIVE → PROVENANCE_STALE. This is a demotion, not a
delete: content, provenance and confidence are kept, the particles surface in
`particles curate`, and the change is reversible.
Demote 2 particle(s)? [y/N]:
```

Answering `n` changes nothing. `--yes` skips the prompt for scripted use. The
command is idempotent — a second run reports `No particles are anchored to a
superseded snapshot. Nothing to do.`

Two things to expect before you run it:

- **The count can be large.** On the project's own dogfood store this was 3,127
  particles, about one ACTIVE belief in six. That is the accumulated backlog,
  not a bug in the count.
- **Your projection will change.** Those beliefs were ranking and being
  rendered; retiring them is the point, but re-render and skim the diff rather
  than running it unattended the first time.

An entry whose newest snapshot has not been extracted yet is skipped: retiring
the old generation before the replacement exists would leave the store with
neither. Extract first, then backfill.

## Deleted files

If a deposited file has moved or been deleted, the refresh reports it as
`missing` and does nothing else — the beliefs stay ACTIVE. This is deliberate:
"the file is gone" is not "the claims are false", and a `stat` taken during a
`git` operation is not evidence of a deletion. Retire those deliberately with
`particles corpus retract`, or re-deposit the file at its new path.
