# Particles CLI

The `particles` CLI implements the v0.3 Core loop: deposit sources, extract particles (with automatic subject resolution), query them, and maintain the knowledge base over time.

## Prerequisites

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...   # required for extract, query --semantic, lint --semantic
uv run particles db init
```

## Commands

| Command | What it does |
|---|---|
| `config validate` | Load `config.yaml` (+ env-var overrides) through the same path the SDK uses and report a human-readable PASS/FAIL — each invalid field as `section.field: message`, or a readable YAML parse error. Exits non-zero on the first problem, so a deploy can gate on it. With no `config.yaml`, reports valid against compiled-in defaults. |
| `db init` | Create database tables. `--force` drops every particle-store table (preserving the corpus) and resets snapshots to PENDING — the scrap-and-re-extract path for crossing a SCHEMA_VERSION major bump. Confirms before dropping. |
| `init claude-code` | One-command Claude Code memory integration: merges a SessionStart digest-push hook + a SessionEnd harvest hook into `~/.claude/settings.json` (`--project` → the repo's gitignored `.claude/settings.local.json`), provisions `~/.particles/claude-code/`, and — on a fresh install with no write-enabled store — creates and enables a `memory` store via a parse-preserve-append `config.yaml` edit. The installed commands carry absolute env pins (`PARTICLES_CONFIG`, and `DATABASE_URL` for the `default` store) so harvest resolves the same store even though Claude Code runs the hook from the session's working directory — routinely a git worktree, not the store's directory. Marker-owned merge: re-running repairs/upgrades exactly the Particles-owned entries (re-run to add the pins to an install predating this fix); `--remove` deletes exactly those (and reverts the store auto-create only while the store is still empty); `--dry-run` previews; `--command` overrides the hook executable path; `--no-audit` skips the first-run audit hand-off. Also installs the agent-onboarding skill files into `~/.claude/skills/particles/` (`--no-skills` opts out; `--remove` deletes exactly that subdirectory). `--json` emits a machine-readable result on stdout — store, scope, what was created, the installed hook commands, and a `next_steps` list of what is left for the human — so an **agent can run the installer itself** and report the outcome instead of scraping human output. It implies `--no-audit` (the audit hand-off is interactive) and names the standalone `particles audit` under `next_steps`. This is the *local* half of agent-first onboarding: it mints no credential and never widens `mcp.write.enabled_stores` — a network MCP auth model is decision, not this verb's. See [User Guide → Claude Code memory](user-guide/claude-code.md). |
| `skills install` / `skills list` | **Agent-onboarding skill files** — three short, tool-agnostic Markdown files shipped inside the SDK (`particles/skills/`) that tell an agent which write verb to reach for (`recording-a-belief.md`), how to read effective confidence and the contested marker (`asking-the-store.md`), and that ruling on a contradiction is the operator's call, not the agent's (`keeping-it-clean.md`). `skills install` copies them into `~/.claude/skills/particles/` (`--dir` overrides, `--project` uses `./.claude/skills`); the `particles` subdirectory is SDK-owned, so a re-run repairs/upgrades and `--remove` deletes exactly it and nothing beside it. `--dry-run` previews. Tool-agnostic Markdown is deliberate — a harness-specific skill schema would bind the SDK to one vendor's format. The files are documentation shipped as data: each links its canonical doc page and states no contract the docs do not. `particles init claude-code` installs them by default. |
| `hook session-start` / `hook session-end` / `hook log` / `hook doctor` | The machine-facing lifecycle verbs `init claude-code` installs. `session-start` reads the hook JSON on stdin and pushes the store's memory digest into the new session's context (skipped on `resume`; capped by `claude_code.digest_max_bytes`); `session-end` distills the session transcript (speaker turns verbatim, tool calls one-line, tool results dropped), redacts common credential shapes, and deposits it as one APPEND_ONLY `CONVERSATION` corpus entry per session — plus the project's changed memory-file `*.md`s and a catch-up sweep over recent unharvested transcripts. Both degrade to exit 0 with no output on **any** failure and append one JSONL line per invocation to the hook log; `hook log --tail N` prints recent entries. Because Claude Code runs the hook from the session's working directory, `init claude-code` bakes absolute `PARTICLES_CONFIG` / `DATABASE_URL` env pins into the installed commands so the store resolves CWD-independently; `hook doctor --store <handle>` diagnoses that resolution from the current directory (config found, DSN, DB file, corpus tables) and exits non-zero on an unusable store, and a table-missing harvest failure now logs an actionable `hint`. Debug loop: `particles hook session-start --store memory < sample.json`. |
| `memory consolidate` | **The dream cycle** — one scheduled verb that runs the cross-session maintenance passes in order: extract catch-up (PENDING snapshots, oldest first, capped at `consolidation.max_pending_entries`), the reconcile sweep, the capped + delta-scoped contradiction/duplicate census (one `collect_cards` pass), the curation-queue refresh over that same collection, utility mining, and the `MEMORY.md` projection re-render — then writes one `CONSOLIDATION_RUN` operator event (per-pass status/durations/LLM call counts, machine-readable census, `completed_at` = the next run's delta watermark) and renders a delta report ("+2 contradictions since last run"). Cadence is launchd/cron, not a daemon: `--if-due` exits 0 unless the last successful run is older than `consolidation.min_interval_hours`, so over-scheduling is harmless. `--structural-only` (or no API key / open breaker / `consolidation.semantic: false`) still runs the LLM-free passes with every skip disclosed — a degraded contradiction line reads "not probed this run", never "0". `--scope delta\|store`, `--output FILE`, `--format markdown\|json`. Exit codes: 0 success/skip, 1 pass(es) failed, 2 could not start. One cycle at a time via a stale-reclaimed `consolidate.lock`. `particles audit` records the same run event (`actor: audit`), joining the delta chain. See [Operator Guide → Scheduled consolidation](operator-guide/scheduled-consolidation.md). |
| `memory rebuild-utility` | Re-mine every harvested session transcript into fresh utility evidence for the usefulness lens — the backfill that credits history when the feature is turned on or the matchers change. Honours `utility.mining.behavioural_matching` for a cheap literal-only rebuild. |
| `memory useful` | **Mark a belief useful** — the explicit second utility channel, for the belief class the transcript miner cannot observe at all. A prohibition ("never prepend `export PATH`") or a design stance is complied with by *not* acting, and the miner reads tool-call lines only, so those beliefs leave no action trace and stay out of the projection head however the matchers are tuned. `particles memory useful <id>` credits one directly: `PARTICLE_ID` accepts the raw UUID, a unique id prefix, or the `p-xxxxxxxx` display form, resolved against ACTIVE beliefs, and an id matching none or several is a hard error rather than a silent credit to the wrong belief. `--reason` records an operator note on the `BELIEF_MARKED_USEFUL` event. One press is worth `utility.explicit_weight` mined events (default `25.0`) because the miner fires once per session while you fire once, and it is capped at **one credit per belief per principal per day** — press again and the gesture is recorded but not double-counted. Promotion-only and projection/digest-only: it never touches the stored confidence and never claims the belief is *true* (for "still true", the gesture is `curate apply affirm`). Operator-only — deliberately no MCP tool, since an agent crediting its own beliefs would close a self-reinforcement loop the containment cannot reach. |
| `memory sweep-rank-lift` | **Calibrate the usefulness rank-lift** — a read-only sweep of `utility.default.rank_lift` (lambda) over your store: no writes, no LLM calls, no embeddings. Name the beliefs that *ought* to reach the projection head with `--target` (repeatable; accepts the raw UUID, any unique id prefix, or the truncated `p-xxxxxxxx` display form — resolved against ACTIVE beliefs, and an id matching none or several is a hard error rather than a silent rank-0 that would report an empty band) and each rendered head size with `--head` (repeatable; defaults to `mcp.recall.digest_max_beliefs`). Reports, per lambda on the grid, where those beliefs rank and how many head slots hold distinct content, then the admissible band per surface, their intersection, and whether the configured value sits inside it. `--grid-max` / `--grid-steps` set the grid, `--distinct-ratio` the diversity bar (default `0.95`; `1.0` is unsatisfiable at large `N` on any store with over-extraction). `--format json` for machine consumption. Lambda is deliberately **not** auto-fitted — nothing labels which belief *should* hold a head slot, and the confidence spread a fit would key on is flattened to zero by the cap — so the operator supplies the targets and this makes that one command. See [Operator Guide → Calibrating `rank_lift`](operator-guide/tuning.md#calibrating-rank_lift). |
| `memory sweep-owner-lift` | **Calibrate the owner-relevance lift** — the sibling of `sweep-rank-lift` for `owner_lens.rank_lift` (omega): a read-only sweep with no writes, no LLM calls, no embeddings. Omega is *store-specific and ships `0.0`* (the lens is inert until you set it), and it has no compiled-in default because the viewer cohort's size varies by two orders of magnitude between store genres. Unlike the usefulness lift it multiplies a flat 0/1 indicator, so it behaves as a **threshold over the whole cohort** — below it nothing moves, above it every belief about the viewer arrives in the head at once — which is why the report is keyed on the cohort's **share of each head** rather than one belief's rank. `--max-owner-share` bounds that share (default `0.5`), `--min-owner` sets how many viewer beliefs the head must hold, and `--target` (repeatable, same id forms) names beliefs that must *stay* in the head: that is a **non-regression check against omega = 0**, so a target already outside the head for unrelated reasons is disclosed in its own section instead of quietly emptying the band. The usefulness lambda in force is held fixed throughout, so what is measured is what aboutness does to the head usefulness already shaped. `--head` per rendered surface, `--grid-max` / `--grid-steps` for the grid, `--format json` for machine consumption. |
| `deposit` | Add a file or URL to the corpus. A JSON file whose top-level keys are `name` / `version` / `tags` is auto-detected as a `TaxonomyDefinition` (`source_type = TAXONOMY_DEFINITION`); run `extract` on it to materialise the tag tree — see [Querying → Creating a taxonomy](user-guide/querying.md#creating-a-taxonomy). Reddit `/s/` share links (the mobile share-sheet form, e.g. `reddit.com/r/<sub>/s/<id>`) are accepted and resolved to the canonical `/comments/` permalink before fetching, so they deposit as the underlying post + comments rather than falling through to the web extractor. For link-shaped sources (Reddit / HN / Mastodon link cards), `--follow-post-links` deposits the post's primary URL as a separate corpus entry and records the relationship in `corpus_follow_edges`. Defaults to True for Reddit / HN / Mastodon; explicit `--no-follow-post-links` disables. `--follow-comment-links` is reserved but deferred — passing True today logs a warning and proceeds without comment-link follow. For a personal journal, `--journal` marks the deposit as `JOURNAL` so the journal-aware extractor handles it — reifying feelings/opinions and emitting the entry-level `NARRATIVE` graph. For an archival local file, `content_published_at` is captured at deposit so its particles aren't all stamped with the import date: precedence is `--date YYYY-MM-DD` (explicit override) › a leading date line in the content › the file mtime, tuned under the `deposit_date` config section. `--date` applies to local files only and is ignored (with a warning) for URLs. For a single local file that concatenates many dated entries (a journal, changelog, or daily-log where each entry begins with a date on its own line), `--split-by-date` segments it at those date-line boundaries and deposits **N corpus entries**, one per dated section, each carrying its own `content_published_at`. It is opt-in (default off leaves one-file-one-entry behaviour unchanged), local-files-only, mutually exclusive with `--date`, and composes with `--journal` / `--source-type` / `--tags`; a file that is not actually multi-entry deposits as a single entry. For a literal string — a note you want recorded now, without writing a temp file first — `--text '…'` deposits it directly, and `particles deposit -` reads the same content from stdin (the CLI half of the MCP `deposit_text` tool). It defaults to `CONVERSATION` and attributes the entry to `--deposited-by` on both the `deposited_by` and `author_id` axes, so an operator's note is not filed at agent trust. Mutually exclusive with a source argument; the file-only flags (`--date`, `--split-by-date`, `--mutability`, `--fetch-policy`) are refused rather than silently ignored, since a pasted string has no file on disk to revisit. |
| `corpus list` | List deposited corpus entries (entry id, source type, latest-snapshot extraction status, ACTIVE-claim count, source URI). `--source-type` filters. `--json` emits machine-readable records with the **full, untruncated** fields (`entry_id`, `source_type`, `uri_r`, `extraction_status`, `particle_count`, `tags`, `created_at`) — the human table left-truncates `uri_r`, so use `--json` to reconstruct `deposit` commands or diff corpus sets across stores. Local-only (no engine endpoint; refuses in remote mode). |
| `corpus cat <snapshot-or-entry-id>` | Dump a snapshot's stored content — the exact bytes the extractor saw. Accepts a snapshot ID or a corpus-entry ID (prefix OK; an entry resolves to its most-recent snapshot). Default output is the same text the extractor derives from the blob (html2text for HTML, pypdf for PDF), so an **empty preview directly explains an `Empty content` / zero-particle extraction** (a bot-wall, login page, or JS-only shell that fetched but carries no article text). `--raw` streams the original bytes (pipe to a file or `less`); identifying metadata (snapshot id, content hash, byte size) goes to stderr so stdout stays pipeable. Works against a remote engine via `GET /corpus/blob/{selector}`. |
| `corpus links list [entry-id]` | Audit the `corpus_follow_edges` graph written by deposit-time URL following. Without an entry-id, lists every recorded edge; with one, shows outgoing/incoming follows per `--direction {out,in,both}`. The downstream consumer that makes the deposit-follow pipeline operator-visible. |
| `corpus links suggest` | Rank URLs the corpus *cites but has not deposited*, by trust-weighted distinct-source diversity × recency — surfacing primary sources the corpus only knows by hearsay. Suggestion-only: nothing is fetched or crawled. `--limit` / `--min-sources` override the config defaults; `-o json` for machine output. Deposit one with `deposit <url>`. |
| `corpus links dismiss <url>` | Stop a URL from resurfacing in `corpus links suggest`. Permanent by default; `--snooze N` suppresses for N days. Audited in the operator event log. |
| `corpus delete <entry-id>` | Delete a corpus entry, its snapshots, and every particle sourced from it. Also sweeps the index rows keyed on those particles (`particle_subjects`, `particle_tag_edges`, `particle_relations`), removes subjects left with no remaining link, and drops `synthesis_cache` rows keyed on those now-orphaned subjects. `--yes` skips the confirmation prompt. |
| `corpus retract <entry-id>` | **Non-destructive sibling of `delete`**. Bulk-transitions every *live* particle (ACTIVE / INCONSISTENCY) from a source to `RETRACTED` (reason `SOURCE_RETRACTED`) while **preserving** the corpus entry, its snapshots, and the particles — so the audit trail of "we believed X until source Y was retracted" survives. Use this (not `delete`) when a publisher retracts an article. Live-only and idempotent. `--reason` is recorded in the operator event log; `--dry-run` previews the plan; `--yes` skips the prompt. Run `lint` afterwards to cascade `PROVENANCE_STALE` downstream. See [Operator Guide → Auditing operator actions](operator-guide/auditing.md). |
| `corpus prune-orphans` | One-off maintenance sweep for DBs that accumulated dangling rows from older deletes: index rows pointing at deleted particles, subjects with no remaining link, and `synthesis_cache` rows keyed on vanished subjects. Prints a per-table count, then deletes after confirmation (`--yes` to skip). A clean DB reports "Nothing to prune." |
| `corpus refresh [entry-id]` | Re-check deposited **local** sources against the files on disk — the on-demand form of consolidation pass 0.5. Walks every entry with a `file://` URI-R and `fetch_policy=LAZY`, comparing the file's mtime and then its SHA-256 against the latest snapshot; a changed file gets a new PENDING snapshot, which `extract --all-pending` turns into current beliefs (and, for a `MUTABLE` entry, retires the generation it replaced). No network. `--force` skips the mtime check and the re-fetch floor. `--backfill-cascade` instead applies the generation demotion to entries whose snapshots moved *before* previewed, then `--yes`-gated. Opt a file in with `particles deposit <path> --mutability MUTABLE --fetch-policy LAZY`. See [Operator Guide → Refreshing mutable local sources](operator-guide/mutable-local-sources.md). |
| `corpus fsck` | **Exhaustive blob audit + re-home** — the operator-invoked sibling of the sampled reachability probe `config validate` runs. Resolves every blob-bearing snapshot, stats each against the resolved `blob_dir`, and reports three disjoint counts: **present**, **found elsewhere**, **missing**. Search scope is explicit, never inferred: the blob dir plus each repeatable `--search <dir>` (a blob root — the directory holding the two-character shards). It does not hunt for `corpus_blobs/` trees, guess at sibling worktrees, or read git metadata, because the content-addressed layout makes a false positive silent. `--re-home` **copies** (never moves) the strays home, rejecting any candidate whose recomputed SHA-256 does not match the name it was found under; re-running is a no-op. `--dry-run` reports what `--re-home` would copy. **The database is never written** — blobs that are genuinely gone are reported with their entry IDs and URIs, and the choice between re-depositing from source and `corpus retract` stays yours. Exits non-zero while any referenced blob is unreachable. Local-only. |
| `particle retract <id>` | Retire **one** belief under operator authority — the narrow escape hatch beside the cross-asserter guardrail, for when `corpus retract` (every particle from a source) is too broad and widening `mcp.write.allow_cross_asserter` would grant every future agent session more than the job needs. `ACTIVE → RETRACTED` with reason `EXPLICIT_RETRACTION`, routed through `update_particle_status` so the `retired_at` stamp and a `PARTICLE_RETRACTED` event carrying `--reason` are both written, under actor `cli:particle-retract`. `--reason` is required; the target is printed and confirmed first (`--yes` skips, `--dry-run` writes nothing). Deliberately **not** gated on `mcp.write.enabled_stores` — that knob is agent policy, and this is the operator's own store. An operator-asserted (`HUMAN_REVIEW`) belief is still refused; revising one is Review's job. Id prefixes accepted. Local engine only. Run `particles lint` afterwards to cascade `PROVENANCE_STALE`. See [Operator Guide → Auditing operator actions](operator-guide/auditing.md). |
| `rules` | Show the **rule-source set** — the operating documents (`AGENTS.md`, `CLAUDE.md`, …) this store tracks — with each file's entry id, `fetch_policy`, snapshot count, and last-checked date. Registered in `rule_sources.paths`; when that is empty the set is *discovered* (the nearest ancestor of the working directory holding a `.git` entry, plus `~/.claude`), and the resolution is always printed before anything is written. Exists because conversation harvest yields claims *about* your rules, never the rules themselves. |
| `rules sync [path…]` | Deposit / re-register the rule-source set as `LOCAL_MARKDOWN` + `MUTABLE` + `fetch_policy=LAZY`, tagged `rule-file`, so the refresh loop keeps it current from the first sweep. Idempotent (identity is the `file://` path, so unchanged content writes nothing); `--dry-run` resolves and prints without writing; explicit `path` arguments override the configured set for one run. A file carrying a projected region is deposited stripped and therefore held out of the byte-level loop — refresh those by re-running this verb. `particles init claude-code` runs it on install. Also re-applies the scope exemption to the tracked set's already-extracted particles, so a rules document's prescriptions reach the default query + projection surfaces instead of being hidden as `DOCUMENT_META`; `--restamp-only` does just that half (deterministic — no LLM call, no re-extraction). See [Operator Guide → Refreshing mutable local sources](operator-guide/mutable-local-sources.md). |
| `extract` | Run LLM extraction on a corpus entry |
| `query` | Ask a natural language question; supports `--tag <path>` (repeatable) for taxonomy-aware subtree-expanded filtering. `--include-ancestors` additionally walks UP each `--tag`'s parent chain, so a query for a specific node also matches particles tagged only with a broader ancestor term. `DOCUMENT_META` particles (claims about a source's own structure) are excluded by default; pass `--include-document-meta` to include them. Non-asserted particles — a document's rejected / superseded / deferred / counterfactual prose (polarity `DECLINED` / `HYPOTHETICAL`) — are likewise excluded by default; pass `--include-non-asserted` to include them. `--assertion-modality <FALSIFIABLE\|EVALUATIVE\|EXPERIENTIAL\|CONSTITUTIVE>` narrows to one truth-aptness modality (omit for all). `--store <handle>` (repeatable) federates the query across several stores; the first handle is the viewer whose trust policy ranks the merged results. By default each result prints its composed **contested badge** — one `⚠ contested (…)` line naming whichever bases fired: a `DISPUTES` stance on record (with the unverified-holder caveat), a lens-divergence spread ≥ `contestedness.callout_threshold`, or an open INCONSISTENCY referencing the claim (with a `(vs. p-xxx)` drill-down); disable with `contestedness.badge_enabled: false` — see [User guide → Querying](user-guide/querying.md#the-contested-badge). `--contestedness` shows each result's per-claim contestedness — the max−min spread of `effective_confidence` across your policy set (local policy + each adopted lens), attributed per policy; absent unless you have at least two policies (adopt a lens). Disclosure only — neither the badge nor the readings ever change ranking or confidence. `--as-of <ISO-8601>` answers from the beliefs held at that past instant (a bare date = start of that day, UTC): retired hits carry their supersession crossing (current status, retirement instant + its basis, and the replacing belief), decay and the recency window are evaluated at T, and retirements the store cannot date are excluded fail-closed with a disclosure line. A future instant is rejected. See [User guide → As-of time travel](user-guide/as-of.md). |
| `query` (structural claim filters) | The same verb also reads the structured-claim annotation. `--predicate <term>` (exact string, case-insensitive — a CURIE and its expanded IRI are different strings; discover terms with `--predicates`), `--object-eq/--object-gt/--object-lt <v>` (typed comparison: numeric xsd → number, `xsd:date`/`dateTime` → date; claims whose object would not normalize are excluded **and disclosed**, never silently dropped), `--object-contains <s>` (case-insensitive substring). With a question the flags **prefilter** the semantic candidate set (ranking untouched); **without a question the query is deterministic** — no embedding, no LLM call — listing matching claims by effective confidence. `--count` and `--group-by subject\|predicate\|object` are deterministic aggregates (claim counts + effective-confidence min/median/max; a simultaneous question is rejected); `--min-effective-confidence <t>` is the *explicit* aggregate floor (there is no default floor). `--predicates` lists the distinct predicate vocabulary. Every structural result carries the coverage footer ("matched against the N of M ACTIVE particles carrying a structured claim") — absence of a hit is not absence of a belief. Composes with `--store`, `--as-of`, `--tag`, `--subject`, `--min-confidence` unchanged. See [User guide → Querying → Structural claim filters](user-guide/querying.md#structural-claim-filters). |
| `lint` | Check the store for staleness and contradictions |
| `review` | List or resolve INCONSISTENCY particles (`--action PREFER_A \| PREFER_B \| BOTH_VALID \| DEFER`; `--bulk` applies one action to all pending). Every non-DEFER resolution retracts the wrapper, so resolved conflicts leave the queue; `BOTH_VALID` / `PREFER_B` recover a quarantined losing claim as a new ACTIVE particle. See [Operator Guide → Lint and review](operator-guide/lint-and-review.md). |
| `curate` | **Bus-stop editing** — the unified, finite, leverage-ranked curation queue. The bare verb prints "today's N" cards: each *composes* an existing finder (lint staleness / retraction-cascade / broken-provenance / confidence-decay / **recency-decay** (carded) / contradiction / **no-subject**, the contested digest, `links suggest` duplicate pairs, `corpus links suggest` URLs, and failed snapshots) into one card with a leverage score, a diagnostic, and the gestures that resolve it. A belief that feeds a configured documentation-projection manifest (`curation.projection_manifests`) gains **projection-blocking** leverage so the queue surfaces it first — fixing a claim a generated doc depends on is high-value (inert until a manifest is listed). `--limit N` overrides the session size, `--kind K` restricts to one card kind, `--semantic` runs the LLM-assisted finders. **The card collection is served from a persisted snapshot** rather than re-running every finder per invocation — on a ~32k-particle store that was 172 s a call and is now 60–86 ms. The collection is built by the nightly `memory consolidate` cycle or by `--refresh` (which forces a store-wide rebuild — slow, it runs every finder); a store with no collection yet is served live at the old speed and says so in the header, because a *read* never writes the cache. `--no-snapshot` bypasses the cache for one invocation. Suppression, belief status, post-build gesture resolutions, ranking and briefs are all re-evaluated live on every read, so a card you resolved at 09:00 never comes back from an 03:30 snapshot; what *is* up to a day old is the detection itself, and the header line says when the collection was built. `curate apply <gesture> <key>` dispatches a card's gesture onto the existing write surface — `affirm` / `snooze` / `dismiss` / `retract` / `merge` / `deposit` / `assign-subject` (the last with `--subject <id-or-name>`) execute directly (with `--reason` / `--days`), while `supersede` / `edit` / `comment` / `reindex` surface the resolving command. The queue is also exposed over HTTP as `GET /curation` on the remote engine — same `limit` / `kind` / `semantic` / `no_snapshot` parameters, returning `{cards, count}` plus the staleness stamp (`built_at`, `age_seconds`, `stale`, `scope`, `per_kind_scope`, `collection_size`); `POST /curation/rebuild` is the HTTP mirror of `--refresh`; the operator-scoped curation writes (affirm / snooze / per-particle supersede / retract / subject-assign) are exposed over HTTP for the bus-stop PWA. |
| `audit` | **First-run memory audit** — the agent-memory rot census: `particles audit <memory-dir>` harvests the directory (or a single file) into your real store, extracts it, and renders complete per-class counts with leverage-ranked exemplars — *potential contradictions* (probe + extract-time contested), *likely-duplicate belief pairs* (`links suggest` REPORT candidates; `--judge` upgrades to LLM-verified), *probably-stale facts* (expired `valid_until` + age decay + confidence decay) — each class ending with its next verb (`review`, `curate --kind …`, `links suggest --judge`, `deposit <url>`). Composition only: no new detection, thresholds stay with their finders. The cost estimate always prints before extraction; above `audit.confirm_call_threshold` calls it asks (`--yes` pre-confirms; non-interactive aborts); `--estimate` prints and exits without depositing. `--transcripts <dir>` opts in session transcripts (capped at `audit.transcript_max_entries`, newest first; `--max-entries` overrides). The contradiction probe is bounded: capped at `audit.max_contradiction_probes` LLM probes, and a harvest run probes only pairs touching this harvest's beliefs by default — `--scope store` opts into the store-wide set. Under the cap, candidates are consumed in two tiers: intra-harvest pairs (both sides from this harvest) first, mixed harvest↔store pairs second, highest similarity within each tier — so a populated store's coincidental cross-pairs can never starve the harvest's own pairs out of the budget; a binding cap is disclosed as "probed X of Y candidate pairs" with the tier split, and the `--judge` duplicate pass consumes its in-scope pairs in the same two-tier order. Bare `particles audit` re-audits the existing store without harvesting — always store-wide, still capped (no key → structural finders still run, `contradiction check skipped: no API key` disclosed). `--output FILE` writes the Markdown report; `--format json` dumps the model; `--store <handle>` targets a named store. Idempotent against the hook harvest through corpus dedup; with the MEMORY.md projection enabled, a successful run finishes by re-rendering the `memory-index` region. Also runs as the closing step of `init claude-code`. See [User Guide → Claude Code memory](user-guide/claude-code.md#the-memory-audit). |
| `reindex` | Re-extract stale or failed entries. Scopes: `--entry-ids`, `--extractor-version` / `--extractor-id` (after an extractor upgrade), and `--provider-model <provider:model>` — re-extract everything one LLM pairing produced, the handle for undoing an uncalibrated provider swap. The pairing is matched exactly, so `openai:gpt-5.6` does not select `openai:gpt-5.6-luna`; particles with no recorded pairing (deterministic extractors, direct assertions, anything minted before the stamp existed) never match. The scope unit is the snapshot, so a snapshot whose particles came from more than one model is re-extracted in full. The scopes **combine by intersection**: `--entry-ids` with any of the particle-matching flags reindexes only the named entries that also match, and reports any that don't — naming entries can only narrow, never widen (before this, every other flag was silently discarded). `--no-failed` applies to auto-discovery only, since a named entry always resolves to its latest COMPLETE snapshot. Every run prints an upfront **work plan** — entries / snapshots / particles in scope, plus any snapshot whose archived blob is missing (known to fail, listed up to 5 with the rest summarised as a count) — before the first LLM call; `--dry-run` prints the plan and exits without extracting (zero LLM calls, zero writes). Output is a short human summary by default (the plan, the capped blob list, final counts); `--format json` emits the full result envelope including the per-snapshot `snapshot_plans` detail. On a terminal the liveness line shows per-item position — `snapshot 12/89 (entry 0a8fb1a9…) — 3 failed` — as each snapshot completes, so remaining work is estimable from the plan's total. |
| `structure` | Backfill the **derived structured-claim annotation** — one subject-predicate-object triple beside a particle's prose, for exact relational lookup and deterministic contradiction checks. Particles extracted since landed get theirs for free at extraction time (one more field in a reply already being paid for); this verb annotates the ones that predate it, at one LLM call each — hence `--rate-limit-per-minute` and a resumable `--limit`. `--dry-run` reports the whole backlog (never the batch cap), the runs it implies, and current coverage, and writes nothing; `--structurizer-version <old>` regenerates annotations stamped with a superseded version (mirrors `reindex --extractor-version`). It can only write the annotation: a claim's `content`, confidence and provenance are untouched, and a particle whose prose has no honest triple is skipped permanently and without complaint — absence is a legal state, and coverage is reported by `particles quality`, never enforced. Local store only. |
| `reconcile` | Cross-entry document-supersession sweep: demote superseded claims the intra-entry `extract` never reconciles. `--dry-run` reports without mutating; `--verbose` prints per-demotion progress. Demotion-only (the loser stays auditable as `PROVENANCE_STALE` / `DOCUMENT_SUPERSEDED`); single-trust-order stores only. Run after depositing an ADR/spec corpus (incl. its superseded documents) to select current truth before a projection. |
| `interchange` | Export/import/restore a portable store bundle: `interchange export -o <dir> [--store H] [--format jsonl\|yaml]` writes `manifest.json` + `particles.<ext>` + `subjects.<ext>` — `--format jsonl` (default, one unit per line) or `--format yaml` (human-editable YAML-LD, same data model, MUST round-trip); `interchange import <dir> [--store H]` imports it (each particle reconciled through §6.6, fresh ids); `interchange restore <dir-or-file> [--store H]` faithfully reconstructs the bundle's own store into an **empty** target with origin ids preserved and no reconcile — refuses a non-empty store. Import/restore auto-detect the container from the member extension, so either format round-trips with no flag. Subjects travel by external reference; units carry the canonical substrate only. |
| `subjects` | List, search, inspect, alias, confirm, unlink, merge, split, delete, gc, set-class, fix-labels subjects (canonical entities). `unlink SUBJECT_ID NAMESPACE:ID` drops a wrong external-reference binding while leaving the subject + particles intact. `split SOURCE_ID --particle PID [--particle PID …] (--new-name NAME \| --new-external-id NS:ID) [--dry-run]` re-binds some particles off a misjoined source Subject onto a new Subject the resolver canonicalises against the available external KBs. `find-duplicates` reports candidate-duplicate subject pairs by name/alias embedding similarity (review, then `merge`). `list --phantoms-only` restricts the listing to phantom subjects — zero ACTIVE particles. `delete SUBJECT_ID [--force]` removes a phantom subject (refuses a non-phantom one unless `--force`, then detaches it from any ACTIVE particles). `gc [--dry-run]` (alias `prune-empty`) sweeps *every* phantom subject in one pass — the bulk cleanup after split/merge churn; `--dry-run` lists what it would remove without committing. `set-class SUBJECT_ID CLASS` overrides the resolver's Nomisma class (e.g. `nmo:NumismaticObject`) when it mis-classed an entity; records a `SUBJECT_RECLASSIFIED` operator event and is a no-op when the class is unchanged. |
| `particle show` | Inspect a single particle (content, status, confidence, subjects, source, context fingerprint). Also lists the narratives a particle is `PART_OF` and, for a `NARRATIVE`, its constituents in sequence |
| `particle search --fingerprint <hash>` | List particles sharing a context fingerprint |
| `particle narrative <id>` | Show a `NARRATIVE` particle's constituents in `SEQUENCE_IN` order. Add `--synthesize` to render it as one cited prose article instead — falls back to a deterministic cited listing with no API key |
| `particle tag <id> --tag <path>` | Apply one or more taxonomy tags to a particle in place; `--force` allows tags outside any active taxonomy |
| `particle untag <id> --tag <path>` | Remove taxonomy tags from a particle |
| `links add p:abc p:def` | Create a typed link between two particles: `--type co-evidential` (default), `part-of`, or `sequence-in`. `part-of` / `sequence-in` are directional (A → B) and build narratives |
| `links remove p:abc p:def` | Remove a typed relation between two particles |
| `links list p:abc` | Show direct relations and the transitive co-evidential group for a particle. `--kind <kind>` filters the listed relations to one relation kind (e.g. `part-of`, `endorses`; case-insensitive) and skips the co-evidential group section unless the kind is `co-evidential` |
| `links suggest --subject <id\|name>` | Propose co-evidential candidate pairs within a Subject (or `--all`); the curation workflow that replaced the L-IDX-01 lint check. `--llm-judge` adds per-pair PARAPHRASE/DISTINCT/UNSURE verdicts; `--apply` (needs `--yes` past the confirm threshold) auto-links the PARAPHRASE pairs. See [Operator Guide → Co-evidential curation](operator-guide/co-evidential.md). |
| `links dedup` | Merge **identical-content** duplicate beliefs into one survivor — exact content equality under the same normalized key extract-time suppression uses (whitespace runs and a trailing period absorbed; wording and case preserved), no similarity threshold and no LLM call, so every near-duplicate stays advisory under `links suggest`. Read-only by default: it reports the groups and redundant-copy counts and writes nothing. `--apply` merges (links each redundant copy `CO_EVIDENTIAL` to the survivor and supersedes it with `DUPLICATE_MERGED`; never deletes, never mutates the survivor) and refuses unless `links_suggest.auto_merge.enabled: true` is set in `config.yaml`. `--subject <id\|name>` scopes to one Subject; `--output-format json` for scripting. See [Operator Guide → Co-evidential curation](operator-guide/co-evidential.md). |
| `links unmerge <event-id>` | Revert an exact-duplicate auto-merge — the one-command undo for `links dedup --apply`. The retained copies return to ACTIVE **keeping their ids** (so `merge ∘ unmerge` is the identity), only the merge's own `EXACT_DUPLICATE` links are dropped, and the survivor is never touched. Copies that moved on since the merge are skipped and named rather than restored, so one drifted row never blocks recovery of the rest. Prints the full blast radius and asks before writing (`--yes` pre-confirms, `--dry-run` plans only). `--run <run-id>` reverts a whole merge run; `--since <ts> [--until <ts>]` is the migration path for merges recorded before run ids existed. Local-only, and deliberately **not** gated on `links_suggest.auto_merge.enabled` — that flag authorizes merging, so the undo must outlive turning it off. See [Operator Guide → Co-evidential curation](operator-guide/co-evidential.md). |
| `trust` | Source trust policy: `trust set <pattern> <score>` (domain baseline or `--modifier` URL-pattern delta), `trust list`, `trust show <uri>` (resolved score), `trust statement-set <domain> CORPUS_ENTRY\|SOURCE_TYPE\|AUTHOR <ref> <rank>` (scoped SourceTrustStatement, four-tier first-match-wins cascade), `trust set-entry <entry_id> <rank>` (per-entry override convenience that validates the entry and infers its domain; `--domain` to override), `trust cascade` (re-run trust resolution). Since the resolved rank bites at query, export, and federated-query time, not just at §6.6 conflict time. See [Operator Guide → Tuning](operator-guide/tuning.md). |
| `trust lens` | Shareable trust-policy lenses: publish by depositing a `TrustLensDefinition` JSON (`particles deposit lens.json`), then `trust lens list` / `show <name>` / `adopt <name>` / `unadopt <name>`. An adopted lens composes into query/export-time effective confidence under local-wins / most-skeptical-across-lenses. |
| `events list` | Read the operator event log — the append-only audit of operator decisions (retract / split / merge / alias / confirm / unlink / trust / review / links / tags). Filter with `--particle` / `--subject` / `--entry` / `--type` / `--limit`. Mirrored by `GET /events` and the read-only MCP `events_list` tool. See [Operator Guide → Auditing operator actions](operator-guide/auditing.md). |
| `events show <id>` | Show one operator event in full — header, refs, and payload. Mirrored by `GET /events/{id}` and the MCP `event_show` tool. |
| `extractor conform <extractor-id>` | Run an extractor against the conformance fixture corpus and report per-field PASS/WARN/FAIL. Scores the fixtures the registry **routes** to that extractor; `--all-accepted` widens the run to every fixture it `accepts()` — a deliberate probe that never updates the stored verdict |
| `extractor generate-fixture <entry-id>` | Turn a deposited corpus entry's latest snapshot + blob into a conformance fixture skeleton (`manifest.yaml` + `content.bin` + `snapshot.json`) and register it in `MANIFEST.yaml`. `expected_acceptors` is left empty to fill after verifying coverage. Supports `--id`, `--source-type`, `--output-dir`, `--force`. Adding a fixture invalidates prior conformance reports (corpus-hash change) |
| `extractor benchmark <extractor-id>` | Run an extractor against bundled benchmark suites and report precision / recall / calibration_error (supports `--suite`, `--judge embedding\|llm`, `--fail-on`). `--runs N` repeats each suite N times and reports mean ± spread per metric instead of a single noisy point estimate (`--fail-on` then reads the mean); it costs N× the LLM calls, so it prints a cost projection first and confirms above `benchmark.confirm_call_threshold` — `--estimate` prints and exits, `--yes` pre-confirms. Each run's report is also persisted as a JSON file under `benchmark.runs_dir` (default `~/.particles/benchmark/runs/`), stamped with the resolved extraction provider:model pairing, so provider comparisons and calibration series survive the terminal session; `--no-save` skips it |
| `extractor benchmark-compare --extractor-id A --extractor-id B` | Run two or more extractors against the same benchmark corpus and emit a suite × extractor matrix per metric (supports `--suite`, `--judge`, `--threshold`, `--format table\|json`) |
| `extractor benchmark-modality <extractor-id>` | Measure `assertion_modality` classification quality for a genre extractor against modality suites under `tests/benchmark/modality/` — per-modality precision/recall, the dangerous false-non-`FALSIFIABLE` rate the journal default raises, and the whole-entry narrative-emission rate (under the framework; integration-tier — drives the extractor's LLM call). Supports `--suite`, `--judge embedding\|llm`, `--threshold`, `--format table\|json` |
| `extractor benchmark-polarity <extractor-id>` | Measure claim-polarity classification quality (ASSERTED / DECLINED / HYPOTHETICAL) for the general extractor against polarity suites under `tests/benchmark/polarity/` — the dangerous **wrong-`DECLINED` rate** (a real decision wrongly hidden from the default surface), its wrong-hidden superset, and per-polarity precision/recall (under the framework; cap. 1; integration-tier — drives the extractor's LLM call). Supports `--suite`, `--judge embedding\|llm`, `--threshold`, `--format table\|json` |
| `extractor benchmark-validity <extractor-id>` | Measure **event-anchored-validity quality** for the general extractor against validity suites under `tests/benchmark/validity/` — the dangerous **wrong-expiry rate** (a durable fact wrongly assigned a `valid_until` and thereby set up for silent retirement by the §9.3 staleness lint — the over-eager-expiry failure mode), plus existence precision/recall of correct date-bounded extraction and date accuracy (under the framework; integration-tier — drives the extractor's LLM call). Supports `--suite`, `--judge embedding\|llm`, `--threshold`, `--format table\|json` |
| `extractor calibrate <extractor-id>` | Fit a temperature-scaling calibration for an extractor over the **calibration** suites it is the production routing choice for — `tests/benchmark/calibration/`, whose gold sets are deliberately partial so the fit sees both labels (for the suite filter). Stored per `(extractor, provider:model)` pairing, so a provider swap keeps each model's own calibration. Supports `--suite`, `--suites-dir`, `--fixtures`, `--judge`, `--dry-run`, `--regenerate`. Unlike `extractor benchmark`, `--judge` defaults to **`llm`**: the calibration label *is* the equivalence judge's verdict, and a well-behaved extractor's only unmatched emissions are the judge's paraphrase misses — so with the embedding judge those misses are not noise around the signal, they are substantially all of it (measured over `prose-calibration-001`, the fitted `T` moved 0.2486–0.7418 across runs of identical inputs and the persist/refuse verdict flipped with it). The LLM judge is consulted only in the contested `[0.65, 0.80)` band, so the extra cost is bounded; `--judge embedding` restores the cost-free, noisier labelling. The judge is printed beside the label split, because two fits are only comparable when it matches. Warns when another stored pairing for the same extractor is suite-stale — the guard is per pairing, so this run neither re-fits nor retires it. Prints the fit's label split and fittable population, and refuses to persist a fit on degenerate labels, one that lands on an optimizer bound, one with fewer than two distinct movable confidences, or one that does not reduce calibration error |
| `extractor calibrations <extractor-id>` | List the stored calibrations for an extractor, one per `(provider, model)` pairing, and flag any whose **suite set is stale** — fitted over suites that differ from the ones the extractor auto-matches today, so the fit answers a question it is no longer asked. Derived from the record's own `benchmark_suite_id`; needs the benchmark suites, so an installed SDK (where `tests/` is not shipped) prints the listing unannotated. `--suites-dir` overrides the suite directory |
| `extractor calibration-forget <extractor-id> <provider-model>` | Retire one stored calibration record — the counterpart to `extractor calibrate`, for a record that cannot be re-fitted because its model is no longer reachable. Prints the record and confirms; `--yes` skips the prompt. That pairing returns to `calibration_source=EXTRACTOR_DIRECT` for particles minted afterwards; particles already stored keep the confidence they were minted with — `particles reindex --extractor-id <id>` re-mints them |
| `benchmark memory` | Run the **agent-memory system benchmark** — LongMemEval against the whole pipeline (deposit → extract → reconcile → query), not a single extractor (its own `benchmark` group, deliberately not under `extractor`). Reports four conditions in two labeled families: retrieval-stage Recall@k / Precision@k scored through provenance chains, and end-to-end QA accuracy for `qa_particles` vs the `qa_full_context` / `qa_no_memory` baselines under one pinned answer model (`llm.benchmark_answer`; judge on `llm.benchmark`). Estimate/confirm cost gate (`--estimate`, `--yes`); dataset downloaded on demand at a pinned revision with SHA-256 verification. Supports `--limit N`/`--all`, `--variant oracle\|s\|m`, `--types`, `--format table\|json`, `--output`, `--store-dir`, `--dataset-file`. See [Benchmarks](benchmarks.md). |
| `conformance check` | Self-certify this implementation against the **Conformance Profile** (`artifacts/conformance/profile.yaml`) — the behavioural/quantitative ground truth. `--level L2` recomputes the deterministic vectors (effective confidence, recency, calibration, §6.9 noisy-OR) via the SDK's own functions; `--level L3` checks the similarity bands + top-k under the live embedding profile (SKIPPED without the model); `--level all` (default) runs both. Exits non-zero on any FAIL; `--json` for machine output. Distinct from `extractor conform` (per-extractor field completeness). |
| `conformance show` | Print the loaded Conformance Profile — version, float tolerance, reference embedding profile, constants, decay table, and formulas. |
| `export obsidian <dir>` | Export the knowledge base as an Obsidian vault. Supports `--with-synthesis` to splice LLM-synthesised prose articles into per-subject notes; shares the input-hash cache with `export wiki` so running both pays LLM cost once per subject. With `--with-synthesis` it also emits one cited-prose note per NARRATIVE under `Narratives/` (disable via `obsidian.emit_narrative_notes: false`). `--invalidate-stale-links` drops the article-cache hash from any note whose `[[X]]` wikilinks reference a renamed subject. |
| `export anki <file>` | Export per-particle flashcards as an Anki deck |
| `export wiki <dir>` | Synthesise per-subject wiki articles with cited footnotes; supports `--dry-run`, `--regenerate-all`, `--invalidate-stale-links`, `--subjects "A,B"`, and `--without-synthesis` to render every article as the deterministic structured listing — no LLM call, no `ANTHROPIC_API_KEY`, reproducible output. Also emits one cited article per NARRATIVE under `Narratives/` via the mechanism (disable via `wiki.emit_narrative_notes: false`; suppressed by `--subjects`, since narratives are subject-less). Cross-exporter `--min-particle-confidence` filter applies, as does `--include-non-asserted` (non-asserted prose is excluded from the rendered surface by default). |
| `export jsonl <file>` | Dump every ACTIVE particle as JSON Lines — one full particle object per line — for analysis tools (jq / pandas / DuckDB). Cross-exporter `--min-particle-confidence` filter and `--include-non-asserted` apply. Distinct from `interchange export`: this is a flat data dump that excludes non-asserted prose by default, not the round-trippable JSON-LD bundle (which keeps everything). |
| `synthesis-cache` | Inspect + prune the shared article-synthesis cache the prose exporters share: `synthesis-cache list` (index of cached articles), `show <subject-id>` (one subject's cached body + metadata), `vacuum [--dry-run]` (delete provably-unreachable rows — stale prompt versions + orphaned subjects), `evict <subject-id> [--yes]` (drop one subject's entries). |
| `export logseq <dir>` | Export the knowledge base as a Logseq graph. Writes `pages/<subject>.md` in Logseq's native bullet-outline format, with each particle emitted as a block whose `id::` is the particle ID — enabling cross-page citation via `((<particle_id>))` syntax. Supports `--with-synthesis`, `--invalidate-stale-links`, `--min-particles`, `--min-links`, and `--min-particle-confidence`. With `--with-synthesis` it also emits one cited-prose page per NARRATIVE in Logseq's `Narratives/` page namespace — on disk `pages/Narratives___<slug>.md` — plus a `## Narratives` backlink block on each participating subject page (mechanism; disable via `logseq.emit_narrative_notes: false`). Shares the synthesis cache with `export wiki` and `export obsidian`. |
| `export notion` | **Notion exporter** — the first **API-target** exporter: syncs the store into one operator-provided Notion database instead of the filesystem (no output path). Each subject becomes a database row (titled with its display name); particles become page blocks carrying effective confidence + status + a link to corpus provenance. Requires `NOTION_API_KEY` (mint an internal integration at https://www.notion.so/my-integrations and share the target database with it) — the token is read from the environment, **never** a flag. `--database-id` (or `notion.database_id` in config) names the target; `--dry-run` reports the plan with **zero** API writes; `--no-update-blocks` is the create-only opt-out (otherwise re-sync owns the managed block range under a sentinel heading and is idempotent — keyed on the subject id stored in a page property). Cross-exporter `--min-particle-confidence` and `--include-non-asserted` apply. |
| `export graph <file.html>` | **Scoped epistemic graph view** — one self-contained HTML file rendering a *scoped* subgraph: `--subject <id>` (one Subject's neighbourhood, `--hops`, clamped to `graph.max_hops`) `--query "<q>"` (one query's retrieval set — the picture of the knowledge a query consults), `--inconsistency <id>` (a contradiction's evidence: the INCONSISTENCY anchor + its disputants with their true statuses; full id or unique prefix), or `--manifest <path> --section <name>` (a projection section's deterministic selection). Scope is mandatory; a whole-store render does not exist, and the `graph.max_nodes` / per-subject panel caps disclose when they bind. Encodings: effective confidence as opacity (decay renders as literal fading), status as form (`--history` adds SUPERSEDED ghosts + successor chains, RETRACTED tombstones), contested badges with bases on hover, utility as node size. `--as-of <T>` renders the graph as believed at T — two exports at two instants make the static belief-history demo. Cytoscape.js is vendored + inlined: no CDN, opens over `file://`, air-gap-safe. `--min-particle-confidence` and `--include-non-asserted` apply. See [User Guide → Graph view](user-guide/graph-view.md). |
| `project <manifest> [output]` | **Documentation projection** — render a checked-in manifest (`docs/projection/<name>.yaml`) into a *cited view* of the store: each derived section's current-truth particles (ASSERTED-only, `DOCUMENT_META` / superseded excluded — §3) synthesised as clean prose via the `article_synthesis` engine, mechanical blocks spliced verbatim, in manifest order. `--without-synthesis` renders the deterministic, key-free listing (semantics); `--check` is the **drift gate** — regenerates the deterministic `<name>.snapshot.md` and exits non-zero on selection / structure drift (prose drift is advisory, fork #3), needing no API key. `--splice REGION` writes the rendered prose *between* named `<!-- BEGIN/END PROJECTED: REGION -->` sentinels in an existing hand-authored file — self-hosting one section of an otherwise hand-written doc — instead of overwriting the whole file; on a manifest with per-section `region:` bindings it renders only that region's section (the single-region re-roll path). `--splice-all` splices *every* declared region in one pass — the N-region self-hosted document (the README's `what-is` / `design-rationale` / `architecture` regions); every derived section must declare a `region:`. `--export-corpus` writes the manifest's sibling `<name>.corpus.jsonl` drift-gate bundle — exactly the deterministic selection, in the interchange form the gate's ephemeral restore consumes. Spliced bodies render `[[wiki-links]]` as plain text (the host is plain Markdown). The drift gate additionally hard-fails when a declared region's `<!-- sources -->` trailer in the shipped doc no longer matches the store selection — a pin/store change forces a visible re-splice. Refused in remote mode (reads the whole local store). |
| `import vault <dir>` | Walk an existing Obsidian vault (or any Markdown directory) and deposit every `.md` file as `LOCAL_MARKDOWN` for retrospective extract + lint (see § Onboarding an existing vault below). With a remote engine configured (`engine.base_url`), the tree-walk **routes each file to the engine** via `deposit_file` — no new endpoint, idempotent, continue-on-per-file-error — so a thin client can seed the canonical store in one command. |
| `import project <dir>` | Walk a software-project tree and deposit every source file (`.py` by default; `import_project.extensions`) as `PYTHON_SOURCE`, one corpus entry per file. Skips dot-prefixed components and the configured build/cache dirs (`import_project.ignore_dirs`) but keeps underscore module files (`__init__.py` / `_shared.py`). `--ext .py,.pyi` overrides the glob per run; `--tags`, `--deposited-by`, `--verbose` mirror `import vault`. Re-running is idempotent (content-hash dedup). With a remote engine configured, routes each file to the engine via `deposit_file`, like `import vault`. Feeds the symbol-aware docstring extractor — see § Onboarding a source tree below. |
| `import web-clipper <dir>` | Walk an Obsidian Web Clipper captures folder and deposit each `.md` capture as a `WEB_PAGE` entry with the provenance its frontmatter carries restored: `source`/`url` → `uri_r` (fragment-stripped, **not** fetched), `published` → `content_published_at` (below `--date`), frontmatter `tags` ∪ `--tags` → entry tags; the frontmatter-stripped body is the deposited content. A header-less / malformed capture falls back to a plain `LOCAL_MARKDOWN` body deposit. Re-running is idempotent (body-hash dedup). One-shot scan (the watch daemon is deferred) — see § Onboarding a Web Clipper captures folder below. |
| `inbox process` | Deposit URLs queued from an iOS Share Sheet via iCloud Drive — see [User Guide → Depositing URLs from your phone](user-guide/inbox.md) for the iPhone Shortcut setup |
| `inbox watch` | Continuously poll the inbox file and deposit pending URLs as they arrive |
| `inbox status` | Show pending / processed counts in the inbox file |
| `mcp serve` | Run the read-only MCP server over stdio for an AI client (Claude Code, etc.) — see § MCP server |
| `mcp tools` | Print the registered MCP tool surface as JSON (debugging aid; no client needed) |
| `mcp resources` | Print the registered MCP resource surface — the `particles://digest/<store>` session-start memory digest |
| `engine serve <host:port>` | Run the FastAPI engine other machines' thin clients talk to. Derives `api.bind_host` from the bind and enforces the fail-closed gate **before** the socket opens, so `engine serve 0.0.0.0:8000` without a real `PARTICLES_API_KEY` is refused up front (`localhost:8000` is loopback-OK). The client side is config, not a command: set `engine.base_url` (or `PARTICLES_ENGINE_BASE_URL`) and the daily verbs target that engine. `--daemon` adds **resident mode**: the process also runs its own consolidation tick (`--if-due` semantics, so over-ticking is harmless) and the intake watchers in the FastAPI lifespan, so a container needs no launchd/cron beside it — off by default, and `GET /health` discloses each background task's state. Configure the rest under `daemon` in `config.yaml`. See [Operator Guide → Remote engine](operator-guide/remote-engine.md). |

See [Command Reference](cli-reference.md) for the full auto-generated option listing.

## Typical workflow

```bash
# 1. Deposit sources (no API calls, instant)
uv run particles deposit paper.pdf
uv run particles deposit https://en.wikipedia.org/wiki/East_Germany
uv run particles deposit --journal 2025-08-05.txt   # personal-journal genre
uv run particles deposit --date 2012-01-02 concept-doc.md  # stamp an old doc's authorship date
uv run particles deposit --journal --split-by-date 2026.md  # one multi-entry journal → N dated entries

# 2. Extract particles from each entry
uv run particles extract ENTRY_ID_1
uv run particles extract ENTRY_ID_2

# 3. Query
uv run particles query "when was East Germany founded?"

# 4. Maintain
uv run particles lint              # check for staleness
uv run particles review            # resolve any conflicts
uv run particles reindex           # retry any failed extractions
uv run particles reindex --dry-run # print the work plan only — no LLM calls, no writes
uv run particles reindex --verbose # print scope size and per-entry progress
# undo a provider trial: re-extract everything one model produced
uv run particles reindex --provider-model openai:gpt-5.6-luna
# scopes AND together — just this entry, and only if that model produced it
uv run particles reindex --entry-ids 1a2b3c4d --provider-model openai:gpt-5.6-luna
```

### Behavioural change — `lint` is read-only by default (0.45.0)

As of 0.45.0,
`particles lint` no longer applies status transitions by default. It
reports what it finds and leaves the store untouched. To apply the
auto-fixable structural transitions (STALENESS, RETRACTION_CASCADE,
CORPUS_LINK_INTEGRITY), pass `--fix`:

```bash
uv run particles lint          # read-only: reports findings, mutates nothing
uv run particles lint --fix    # apply structural status transitions
```

Scripts or cron jobs that relied on the old implicit `--fix` must add
`--fix` explicitly to retain the prior behaviour. The same flip applies
to `POST /lint` (`fix` defaults to `false`).

## Onboarding an existing vault

The `import vault` verb honors the whitepaper §4.2 / spec §9.4 promise that
`particles lint` runs day-one against an existing Markdown knowledge base
without rebuilding it. One command points at an Obsidian vault (or
any directory of Markdown notes); afterwards the standard
`extract` + `lint` loop is meaningful against the contents.

```bash
# Onboard a hand-written or LLM-prose vault.
uv run particles import vault ~/Documents/MyVault

# Run extraction over everything just deposited.
uv run particles extract --all-pending

# Now lint reports findings about the vault's claims.
uv run particles lint
```

To try the workflow without a vault of your own, the repo bundles a small
sample vault at `tests/fixtures/llm_wiki_vault/` — eleven numismatics notes
with three planted contradictions and one stale claim, documented in the
vault's `_GROUND_TRUTH.md` manifest so a run against it is a measurement
against known ground truth. Note the manifest's caveat: the planted
*cross-note* conflicts are not yet detected by any surface in the core
loop; the fixture doubles as the acceptance test for that
capability.

What the importer does:

- **Recursive `.md` / `.markdown` walk.** Files under any path component
  starting with `.` or `_` are skipped (so Obsidian's `.obsidian/` settings,
  `.trash/`, `_attachments/` scaffolds, and similar metadata-only paths
  do not pollute the corpus).
- **Stamps `SOURCE_TYPE = LOCAL_MARKDOWN`** on each entry. The
  `GeneralExtractor` routes that source type through a frontmatter
  stripper before sending the body to the LLM, so YAML metadata (`tags:`,
  `aliases:`, `created:`, …) at the top of notes does not surface as
  particle content.
- **Idempotent.** Existing `content_hash` deduplication applies — re-running
  on the same vault returns the same entry IDs and adds no new rows.
- **`--verbose`** emits `[i/N] depositing path/to/file.md` progress lines so
  large vaults don't sit silent during onboarding.

Known limitations (all out of scope — flagged as follow-up):

- **Wikilinks (`[[Other Note]]`) are not resolved to subject IDs.** They
  pass through the LLM as plain text. A follow-up ADR is required before
  wiring this in to avoid drift.
- **Frontmatter is dropped, not promoted.** A note with `subject: France`
  in its frontmatter does NOT seed a canonical-name Subject row. Same
  follow-up.
- **Particles-exported callouts round-trip is not implemented.** The
  importer assumes vaults are hand-written or LLM-prose, not the output of
  `particles export obsidian`. Re-ingesting a Particles-exported vault is
  a separate concern.

## Onboarding a source tree

`import project` is the source-code analog of `import vault`: it
walks a software-project tree and deposits **one `PYTHON_SOURCE` corpus entry
per source file**, so the deterministic docstring extractor can turn
documented symbols into queryable, cross-checkable particles.

```bash
# Deposit a project's source tree (one entry per .py file).
uv run particles import project ~/src/myproject

# Extract docstring particles from everything just deposited.
uv run particles extract --all-pending

# Lint surfaces code/design drift: a docstring that contradicts a decided
# ADR/spec particle in the same store (no new lint code).
uv run particles lint
```

What the importer does:

- **Recursive source-glob walk** (`.py` by default; set
  `import_project.extensions`). The ignore policy differs from the vault
  walker by design: dot-prefixed components (`.git`, `.venv`,
  `.mypy_cache`) and the configured build/cache directories
  (`import_project.ignore_dirs` — `__pycache__`, `node_modules`, `build`, `dist`,
  …) are pruned, but **underscore-prefixed module files are kept** —
  `__init__.py`, `_shared.py`, `_logging.py` carry real docstrings, so the vault
  walker's blanket `_`-component skip would silently lose them.
- **Stamps `SOURCE_TYPE = PYTHON_SOURCE`** on each entry, routing it to the
  symbol-aware docstring extractor. A lone `.py` deposited directly
  (`particles deposit foo.py`) routes the same way.
- **`--ext .py,.pyi`** overrides the glob for one run (which files to deposit is
  a per-invocation choice); `--tags`, `--deposited-by`, and `--verbose` mirror
  `import vault`.
- **Idempotent.** Existing `content_hash` deduplication applies — editing one
  file and re-running appends a snapshot only to that file's entry.

Out of scope for now (§ Deferred): depositing a whole *remote* GitHub
repo as a tree (the GitHub importer is single-file), non-Python language source
trees, `.gitignore`-aware walking, and non-source project files (READMEs are
covered by `import vault`).

## Onboarding a Web Clipper captures folder

`import web-clipper` is a **general frontmatter-Markdown intake** with
the [Obsidian Web Clipper](https://obsidian.md/clipper) as its first configured
profile. The Clipper saves each web page to a vault as a Markdown file with a YAML
frontmatter header (`source:`, `published:`, `tags:`, …). Running that folder
through `import vault` would throw away the structured provenance the Clipper
already captured — the entry's `uri_r` becomes the local `file://` path, the
publication date is lost, and the per-file tags are dropped. `import web-clipper`
reads the header and restores all of it.

```bash
# Onboard a folder of Web Clipper captures.
uv run particles import web-clipper ~/Obsidian/Clippings

# Extract over everything just deposited.
uv run particles extract --all-pending

# Lint now reasons over the clippings as the web pages they are.
uv run particles lint
```

What the importer does (per the `web_clipper` config profile):

- **Recursive `.md` / `.markdown` walk** with the vault ignore policy (skips
  `_` / `.` components — a Clipper folder is an Obsidian vault folder).
- **Maps the frontmatter onto deposit fields.** `source` (fallback `url`) becomes
  the entry's `uri_r` — fragment-stripped and **not fetched** (the archived clip
  is the record of what was seen; `mutability=STABLE`, `fetch_policy=NEVER`). So
  corpus-link-gap lint, mention reconciliation, and refetch identity
  all see the real web page, not a local file. `published` becomes
  `content_published_at`, below an explicit `--date`. The frontmatter
  `tags` merge with any run-wide `--tags` (per-file ∪ run-wide).
- **Stamps `SOURCE_TYPE = WEB_PAGE`** — the entry is now trustable, decayable
  (content-age decay), and queryable as the web page it clipped.
- **Deposits the frontmatter-stripped body** as the entry's bytes, so the
  extractor sees the article, not the YAML header.
- **Idempotent (body-hash dedup).** Re-clips that differ only in clip-time
  metadata collapse to one entry; a page clipped twice (same `source:`) collapses
  to one entry with two snapshots, not two `file://`-keyed entries.
- **Graceful degradation.** A capture whose header is absent or malformed falls
  back to a plain `LOCAL_MARKDOWN` body deposit rather than aborting the scan.

The profile is config (`web_clipper.url_keys` / `date_keys` / `tag_keys` /
`source_type`), so a second frontmatter producer is an operator edit, not a code
change. Out of scope for now (§ Deferred): a watching daemon
(`import web-clipper watch`), first-class `title` / `author` capture
, and a second profile + verb generalisation.

## Cross-exporter quality threshold

`particles export <format>` accepts `--min-particle-confidence` for every
shipped exporter (`wiki`, `obsidian`, `anki`, `logseq`, `jsonl`, `notion`).
Particles whose `effective_confidence` (trust-weighted) falls
below the threshold are dropped *before* any per-exporter downstream step —
prompt input, cache hash, count-based `min_particles` check, rendered
output, references. Default `0.0` is a no-op; operators opt in.

```bash
uv run particles export wiki ./out --min-particle-confidence 0.5
uv run particles export obsidian ./vault --min-particle-confidence 0.5
uv run particles export anki ./deck.txt --min-particle-confidence 0.5
uv run particles export logseq ./graph --min-particle-confidence 0.5
uv run particles export notion --dry-run --min-particle-confidence 0.5
```

The per-invocation flag overrides `config.exporter_common.min_particle_confidence`.
See the *Migration* section for the cross-exporter rationale.

## Upgrading across an SDK schema bump

Particles uses a separate `SCHEMA_VERSION` (the on-disk particle shape)
distinct from the SDK package version. Within the v1.x line, schema
bumps are additive-compat per strict SemVer — your existing store keeps
working. Across a major schema bump (the only one to date is v0.3.x → v1.0.0
at the freeze),
the SDK refuses to operate on a store with mismatched-schema particles
and surfaces this error:

```
Particle store contains N ACTIVE particles whose schema_version does not
match the current SDK (SDK is at v1.0.0; store has N at v0.3.0).
the v0.3.x → v1.0.0 upgrade path is scrap-and-re-extract.
Run:
  particles db init --force    # confirms before dropping
  particles extract --all-pending
The corpus is preserved; only the particle store is rebuilt.
```

The corpus (`CorpusEntry` rows + the content-addressed blob store) is
**preserved across the rebuild**. Only the particle store is dropped
and re-extracted from the existing corpus content. Re-extraction cost
is local LLM time; no re-fetch from the internet.

```bash
# 1. Snapshot the existing store metadata in case you want to diff later.
cp particles.db particles.db.0.3.x-backup

# 2. Re-initialise the store fresh (confirms before dropping).
uv run particles db init --force

# 3. Re-extract from every corpus entry already deposited.
uv run particles extract --all-pending
```

`lint` is the one operation that does NOT refuse — it reports
mismatched particles as `SCHEMA_VERSION_MISMATCH` findings so the
operator can diagnose first. `query` / `extract` / `review` /
`reindex` all refuse.

## MCP server

`particles mcp serve` runs a read-only Model Context Protocol server
over stdio, so AI clients (Claude Code, Claude Desktop, other
MCP-aware tools) can consult the particle store with typed JSON tool
calls rather than parsing CLI prose.

### Install

```bash
claude mcp add particles -- uv run --project /path/to/particles-engine-py particles mcp serve
```

The client spawns the server as a child process the first time it
needs one of the registered tools.

### Tool surface

Read-only in v1. Every tool delegates to an existing operation:

| Tool | What it returns |
|---|---|
| `query` | Tag/subject-aware semantic query: NL answer + ranked particles + coverage gaps |
| `particle_show` | One particle with provenance URLs, source types, and linked subjects |
| `particle_search` | All particles sharing a context fingerprint |
| `particles_list` | Particles filtered by status and/or subject (paginated, no embeddings) |
| `subjects_list` | Every subject in the knowledge graph |
| `subjects_search` | Substring search over subject canonical names |
| `subjects_show` | One subject plus the particle IDs that mention it |
| `list_taxonomies` | Every taxonomy + its full tag tree |
| `list_corpus_entries` | Recent corpus entries (paginated, source-type-filterable) |
| `lint` | Structural lint findings (no `--fix`, no LLM semantic checks) |
| `quality_report` | Extraction-quality dashboard snapshot |
| `links_suggest` | Co-evidential candidate pairs within a Subject (report mode only — `--apply` is CLI/HTTP-only) |
| `events_list` | Operator event log, newest first |
| `event_show` | One operator event in full — header, refs, payload |

Mutating tools (deposit, extract, tag/untag, status transitions) are
not shipped — a read-write MCP surface is designed
(proposed, not yet implemented); until it lands the MCP surface
cannot change the store.

### Debugging without a client

```bash
uv run particles mcp tools                    # JSON dump of the registered surface
uv run particles mcp tools --format text      # one-line-per-tool summary
```

The JSON output is what `tests/mcp/tool-schema.json` should match; if
those drift apart, the unit test `TestToolSchemaGolden` fails in PR
review.

## Prefix

All commands must be prefixed with `uv run` unless you activate the virtualenv first:

```bash
# With uv run prefix (recommended)
uv run particles query "..."

# Or activate once per session
source .venv/bin/activate
particles query "..."
```
