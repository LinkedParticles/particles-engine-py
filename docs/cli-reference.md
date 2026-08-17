# `particles`

Particles SDK — epistemic knowledge management for AI agents (v0.3 Core).

**Usage**:

```console
$ particles [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-V, --version`: Show the client version (and, in remote mode, the engine version) and exit.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `db`: Manage the database.
* `deposit`: Deposit a file, URL, or literal text into...
* `extract`: Extract particles from a corpus entry, or...
* `query`: Query the particle store with a natural...
* `lint`: Run lint checks over the particle store.
* `review`: List or resolve INCONSISTENCY particles.
* `reindex`: Re-extract particles for stale or failed...
* `reconcile`: Demote superseded claims across corpus...
* `quality`: Show the extraction quality dashboard.
* `export`: Export the knowledge base to an external...
* `project`: Render (or drift-check) a documentation...
* `audit`: Audit an agent-memory directory: harvest,...
* `structure`: Annotate particles with a structured...
* `subjects`: Manage subjects (canonical real-world...
* `benchmark`: Whole-pipeline system benchmarks (ADR...
* `config`: Inspect and validate Particles...
* `conformance`: Conformance Profile checks (...
* `corpus`: Inspect deposited corpus entries.
* `curate`: Bus-stop editing — the finite,...
* `engine`: Remote engine server.
* `events`: Inspect the operator event log.
* `extractor`: Manage extractor registry (Extension A).
* `hook`: Machine-facing Claude Code lifecycle hooks...
* `import`: Bulk-onboard existing knowledge bases...
* `inbox`: Process URLs queued from an iOS Shortcut...
* `init`: Install (or remove) an agent-harness...
* `interchange`: Export / import portable store bundles...
* `links`: Manage typed relations between particles...
* `mcp`: Model Context Protocol server (read-only,...
* `memory`: Agent-memory maintenance (ADR...
* `particle`: Inspect individual extracted particles.
* `rules`: Operating-rule source documents tracked by...
* `skills`: Install the agent-onboarding skill files...
* `synthesis-cache`: Inspect and prune the shared...
* `trust`: Manage source trust rules.

## `particles db`

Manage the database.

**Usage**:

```console
$ particles db [OPTIONS] ACTION
```

**Arguments**:

* `ACTION`: Action: init  [required]

**Options**:

* `--force`: init: drop every particle-store table (preserving the corpus + blob store) and rebuild from scratch. this is the upgrade path across a SCHEMA_VERSION major bump. Confirms before dropping.
* `--help`: Show this message and exit.

## `particles deposit`

Deposit a file, URL, or literal text into the corpus.

**Usage**:

```console
$ particles deposit [OPTIONS] [SOURCE]
```

**Arguments**:

* `[SOURCE]`: File path or URL to deposit. Pass '-' to read the content from stdin. Omit it when using --text.

**Options**:

* `--text TEXT`: Deposit a literal string instead of a file or URL — the CLI half of the MCP `deposit_text` tool, so you no longer have to write a temp file to record one note. Mutually exclusive with a source argument. `particles deposit -` reads the same content from stdin. Defaults to source-type CONVERSATION; attributed to --deposited-by on both the deposited_by and author_id axes.
* `--deposited-by TEXT`: Agent or operator ID  [default: operator]
* `--source-type TEXT`: Override the source_type (normally auto-detected from the extension / URL / content — you rarely need this). Core values (particles.core.schema.SourceType): WEB_PAGE, PDF, CSV, CONVERSATION, DATA_EXPORT, LOCAL_FILE, LOCAL_MARKDOWN, ACADEMIC_PAPER, FORUM, BLOG, TAXONOMY_DEFINITION, TRUST_LENS_DEFINITION. Domain extractors register their own, e.g. JOURNAL, REDDIT_POST, HACKERNEWS_THREAD, MASTODON_THREAD, GITHUB_REPO / GITHUB_GIST / GITHUB_PAGES, WIKIDATA_API, NUMISTA_API_COIN / NUMISTA_API_ISSUER, NOMISMA_API. Use --journal for the JOURNAL shortcut.
* `--journal`: Mark this deposit as a personal JOURNAL so the journal-aware extractor handles it (reifies feelings/opinions and emits the NARRATIVE graph). Shorthand for --source-type JOURNAL; an explicit --source-type wins.
* `--tags TEXT`: Comma-separated tags
* `--date TEXT`: Record this content's authorship date as content_published_at (ISO YYYY-MM-DD). Overrides the leading-date and file-mtime auto-detection. Local-file deposits only — ignored with a warning for URLs.
* `--split-by-date`: Split a multi-entry local file at standalone date-line boundaries into N corpus entries, each with its own content_published_at (a journal / changelog / daily-log that concatenates many dated entries). Opt-in; default off leaves today's one-file-one-entry behaviour unchanged. Local files only; mutually exclusive with --date. A file that is not actually multi-entry deposits as a single entry.
* `--follow-post-links / --no-follow-post-links`: Follow the post's primary URL for link-shaped sources (Reddit / HN / Mastodon link cards). When unspecified, the extractor's default applies — Reddit / HN / Mastodon default to True, everything else to False.
* `--follow-comment-links / --no-follow-comment-links`: Reserved-but-deferred. Passing --follow-comment-links emits a warning and proceeds as if False — comment-link following is captured § Deferred and will land in a follow-up release.
* `--mutability TEXT`: Mutability class: STABLE | MUTABLE | APPEND_ONLY | EPHEMERAL. Local files default to STABLE. MUTABLE means a new snapshot retires the generation of beliefs it replaces — the right class for a rule file like AGENTS.md that is edited in place. Local deposits only.
* `--fetch-policy TEXT`: Re-fetch policy: LAZY | NEVER. Local files default to NEVER (frozen at deposit). LAZY opts the file into the refresh ladder, so `particles corpus refresh` and the nightly consolidation pass re-check it against disk. Pair with --mutability MUTABLE. Local deposits only.
* `-v, --verbose`: Show importer + fetch INFO logs
* `--debug`: Show URL parsing, request URLs, auth state, and DEBUG logs
* `--help`: Show this message and exit.

## `particles extract`

Extract particles from a corpus entry, or all PENDING entries at once.

**Usage**:

```console
$ particles extract [OPTIONS] [ENTRY_ID]
```

**Arguments**:

* `[ENTRY_ID]`: Corpus entry ID (omit with --all-pending)

**Options**:

* `--snapshot-id TEXT`: Snapshot ID; defaults to latest PENDING
* `--agent-id TEXT`: Asserted-by agent ID  [default: cli-user]
* `--all-pending`: Extract all PENDING snapshots in deposit order
* `-v, --verbose`: Show quality notes and INFO logs
* `--debug`: Show raw LLM prompt/response and DEBUG logs
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

## `particles query`

Query the particle store with a natural language question.

**Usage**:

```console
$ particles query [OPTIONS] [QUESTION]
```

**Arguments**:

* `[QUESTION]`: Natural language question. Omit it with structural claim flags for the deterministic (no-LLM) modes.

**Options**:

* `--min-confidence FLOAT`: Minimum confidence threshold  [default: 0.0]
* `--audience TEXT`: GENERAL, EXPERT, or REGULATORY  [default: GENERAL]
* `--top-k INTEGER`: Number of particles to retrieve  [default: 40]
* `--subject TEXT`: Filter to particles about this subject ID
* `--tag TEXT`: Filter by taxonomy tag (subtree-expanded; repeatable)
* `--include-ancestors`: Also match particles tagged with a broader ancestor of each --tag (up-expansion over taxonomy parent links)
* `--show-particles`: Print retrieved particles with scores before the answer
* `--contestedness`: Show per-result contestedness — the max−min spread of effective confidence across your policy set (local + adopted lenses). Absent when fewer than two policies are configured.
* `--include-document-meta`: Include DOCUMENT_META particles (claims about a source's own structure)
* `--include-non-asserted`: Include non-asserted particles — a document's rejected / superseded / deferred / counterfactual prose (polarity DECLINED / HYPOTHETICAL)
* `--assertion-modality TEXT`: Filter to one modality: FALSIFIABLE, EVALUATIVE, EXPERIENTIAL, or CONSTITUTIVE. Omit to return every modality.
* `--store TEXT`: Federate the query across these store handles (repeatable). Omit to query the default store. The first handle is the viewer whose trust policy ranks the merged results.
* `--predicate TEXT`: Filter to claims whose predicate term equals this string (case-insensitive, exact — a CURIE and its expanded IRI are different strings; discover terms with --predicates).
* `--object-eq TEXT`: Filter to claims whose object equals this value (typed when both sides normalize — numbers and ISO dates — else case-insensitive text).
* `--object-gt TEXT`: Filter to claims whose object is greater than this number or ISO date. Claims whose object would not normalize are excluded and the exclusion count disclosed.
* `--object-lt TEXT`: Filter to claims whose object is less than this number or ISO date (same normalization and disclosure as --object-gt).
* `--object-contains TEXT`: Filter to claims whose object contains this substring (case-insensitive; works on every term kind).
* `--count`: Deterministic aggregate: the number of matching claims with their effective-confidence distribution. No question, no LLM call.
* `--group-by TEXT`: Deterministic aggregate: bucket matching claims by 'subject', 'predicate', or 'object' with per-bucket counts and confidence distribution. No question, no LLM call.
* `--min-effective-confidence FLOAT`: Explicit confidence floor for the aggregate modes; excluded rows are disclosed. There is no default floor.
* `--predicates`: List the distinct predicate terms with kind and claim count — the vocabulary the exact-string --predicate filter matches against.
* `--as-of TEXT`: Answer as of this past instant (ISO-8601; a bare date means the start of that day, UTC): what did the store believe at T, and why did it stop believing it? Retired hits carry their supersession crossing; retirements the store cannot date are excluded with a disclosure line. A future instant is rejected.
* `--help`: Show this message and exit.

## `particles lint`

Run lint checks over the particle store.

**Usage**:

```console
$ particles lint [OPTIONS]
```

**Options**:

* `--fix / --no-fix`: Apply auto-fixable status transitions (STALENESS, RETRACTION_CASCADE, CORPUS_LINK_INTEGRITY).  [default: no-fix]
* `--semantic / --no-semantic`: Run LLM-assisted semantic checks (slower)  [default: no-semantic]
* `--output-format TEXT`: Output format: markdown or json  [default: markdown]
* `-v, --verbose`: Show all findings in full
* `--category TEXT`: Restrict --verbose output to one finding_type (e.g. STALENESS, GRANULARITY_VIOLATION_CANDIDATE)
* `--limit-per-category INTEGER`: Cap verbose findings per category to keep output manageable; remainder is summarised. 0 disables the cap.  [default: 50]
* `--low-coverage-threshold INTEGER`: Subjects with fewer ACTIVE CLAIM particles are flagged  [default: 3]
* `--help`: Show this message and exit.

## `particles review`

List or resolve INCONSISTENCY particles.

    # List all pending conflicts
    particles review

    # Resolve a specific conflict
    particles review PARTICLE_ID --action PREFER_A

    # Resolve all pending conflicts with one action
    particles review --bulk BOTH_VALID
    particles review --bulk PREFER_B          # prefer newer/structured source
    particles review --bulk BOTH_VALID --dry-run  # preview without committing

**Usage**:

```console
$ particles review [OPTIONS] [PARTICLE_ID]
```

**Arguments**:

* `[PARTICLE_ID]`: INCONSISTENCY particle ID; omit to list

**Options**:

* `--action TEXT`: PREFER_A, PREFER_B, BOTH_VALID, DEFER
* `--bulk TEXT`: Apply action to ALL pending conflicts
* `--dry-run`: Preview bulk action without committing
* `--reviewer-id TEXT`: Reviewer identity  [default: cli-user]
* `--domain TEXT`: Domain for trust statement  [default: general]
* `--note TEXT`: Optional reviewer note
* `--help`: Show this message and exit.

## `particles reindex`

Re-extract particles for stale or failed corpus entries.

**Usage**:

```console
$ particles reindex [OPTIONS]
```

**Options**:

* `--entry-ids TEXT`: Comma-separated entry IDs (full or unambiguous prefix); omit for auto. Combines with --extractor-version / --extractor-id / --provider-model by intersection: only the named entries that also match the filter are reindexed, and any that don't are reported.
* `--extractor-version TEXT`: Old extractor version to replace
* `--extractor-id TEXT`: Extractor name (e.g. github-repo-extractor) — re-extract all of its particles regardless of version. Useful when a shared upstream change (e.g. a prompt revision in general.py) affects delegating extractors.
* `--provider-model TEXT`: "<provider>:<model>" pairing (e.g. openai:gpt-5.6-luna) — re-extract every particle that pairing produced. The handle for undoing an uncalibrated provider swap. Matched exactly, and the scope unit is the snapshot, so a snapshot with a model-mixed population is re-extracted whole. Particles with no recorded pairing never match.
* `--no-failed / --no-no-failed`: Skip FAILED snapshot entries. Applies to auto-discovery only — --entry-ids resolves each entry to its latest COMPLETE snapshot.  [default: no-no-failed]
* `--dry-run`: Print the work plan — entries / snapshots / particles in scope, with per-snapshot counts and any known-missing blobs — and exit without extracting: zero LLM calls, zero writes.
* `--format [human|json]`: Output format: a short human summary (default), or the full JSON result envelope including the per-snapshot plan.  [default: human]
* `-v, --verbose`: Print scope size and per-entry progress while reindexing.
* `--help`: Show this message and exit.

## `particles reconcile`

Demote superseded claims across corpus entries (document-supersession sweep).

**Usage**:

```console
$ particles reconcile [OPTIONS]
```

**Options**:

* `--dry-run`: Report what would be demoted without mutating the store.
* `-v, --verbose`: Print scope size and per-demotion progress.
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

## `particles quality`

Show the extraction quality dashboard.

Displays calibration source distribution, corpus snapshot status,
and subject coverage metrics. No LLM calls — instant read from the DB.
For full structural and semantic diagnostics use: particles lint

**Usage**:

```console
$ particles quality [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `particles export`

Export the knowledge base to an external format.

Available formats: obsidian, anki, wiki, logseq, jsonl, notion, graph.

    particles export obsidian ./my-vault
    particles export obsidian ./my-vault --min-particles=1 --min-links=2
    particles export anki ./deck.txt --deck-name="Numismatics" --min-particle-confidence=0.7
    particles export wiki ./my-wiki                    # incremental
    particles export wiki ./my-wiki --dry-run          # cost estimate
    particles export wiki ./my-wiki --regenerate-all   # bypass cache
    particles export wiki ./my-wiki --without-synthesis # deterministic, no LLM
    particles export wiki ./my-wiki --subjects "Pfennig,GDR"
    particles export logseq ./my-graph                 # pages/ + bullet outline
    particles export logseq ./my-graph --with-synthesis
    particles export notion --dry-run                  # plan, zero API writes
    particles export notion --database-id=abc123…      # sync into a Notion DB
    particles export notion --no-update-blocks         # create-only (keep hand-edits)
    particles export graph out.html --subject <id>     # one Subject's neighbourhood
    particles export graph out.html --query "why X?"   # one query's retrieval set
    particles export graph out.html --subject <id> --history --as-of 2006-08-24

The notion exporter is an API target: it takes NO output path and
requires the NOTION_API_KEY environment variable (mint an internal
integration at https://www.notion.so/my-integrations and share the target
database with it). Run with --dry-run first to see the plan without writing.

**Usage**:

```console
$ particles export [OPTIONS] FORMAT [OUTPUT]
```

**Arguments**:

* `FORMAT`: Export format: obsidian | anki | wiki | logseq | jsonl | notion | graph  [required]
* `[OUTPUT]`: Output path (directory for obsidian / wiki / logseq, file for anki / jsonl / graph). Optional for obsidian when ``obsidian.default_output_path`` is set in config.yaml; required for the filesystem exporters. The notion exporter is an API target and takes NO output path.

**Options**:

* `--min-particles INTEGER`: Obsidian/Wiki/Logseq: minimum particle count per subject (0 = all; wiki default 3)
* `--min-links INTEGER`: Obsidian/Logseq: minimum graph link count per subject (0 = all)
* `--deck-name TEXT`: Anki: root deck name prefix  [default: Particles]
* `--min-particle-confidence FLOAT`: Cross-exporter: drop particles with effective_confidence below this threshold before any per-exporter downstream step. Overrides config.exporter_common.min_particle_confidence.
* `--regenerate-all`: Wiki: bypass the per-subject input-hash cache and rewrite every article
* `--invalidate-stale-links`: Wiki/Obsidian/Logseq: scan cached article bodies for [[X]] wikilinks; invalidate any article whose wikilinked subjects' canonical names have drifted since render. Cheaper than --regenerate-all when only a few subjects renamed.
* `--subjects TEXT`: Wiki: comma-separated canonical subject names to limit the export to
* `--dry-run`: Wiki: report cache hits + regen count + token estimate without writing or calling the LLM
* `--with-synthesis`: Obsidian/Logseq: splice LLM-synthesised prose articles into per-subject notes. Requires ANTHROPIC_API_KEY. Shares the synthesis cache with the wiki exporter, so running multiple synthesising exporters pays LLM cost once per subject.
* `--without-synthesis`: Wiki: render every article as the deterministic structured listing — no LLM call, no ANTHROPIC_API_KEY, reproducible output. Bypasses the synthesis cache so existing LLM articles are replaced.
* `--include-non-asserted`: Include non-asserted particles — a document's rejected / superseded / deferred / counterfactual prose (polarity DECLINED / HYPOTHETICAL). Excluded from the rendered surface by default; the round-trippable `interchange` export always keeps them.
* `--subject TEXT`: Graph: render one Subject's neighbourhood — a subject id or an exact (case-insensitive) canonical name / alias. Scope is mandatory for the graph exporter — pass exactly one of --subject or --query; a whole-store render does not exist.
* `--query TEXT`: Graph: render one query's retrieval set — the picture of the knowledge a query consults (top graph.query_top_k hits + their subjects). Mutually exclusive with --subject.
* `--inconsistency TEXT`: Graph: render one contradiction's evidence — the INCONSISTENCY particle (full id or unique prefix) as the anchor, its two disputant beliefs with their true statuses, their subjects and sources. Mutually exclusive with the other scopes.
* `--manifest TEXT`: Graph: with --section, render a projection manifest section's deterministic selection.
* `--section TEXT`: Graph: the manifest section's region id or exact title (with --manifest).
* `--hops INTEGER`: Graph: neighbourhood radius for --subject scope (clamped to graph.max_hops)  [default: 1]
* `--history`: Graph: include retired supersession-chain ancestors as ghosts (dashed, with the successor chain in the panel); the page gets a client-side history toggle
* `--as-of TEXT`: Graph: render the graph as believed at this ISO-8601 instant (single-instant lens; undatable retirements are excluded fail-closed and disclosed). Two exports at two instants make the static belief-history demo.
* `--max-nodes INTEGER`: Graph: per-run node cap (clamped to graph.max_nodes; truncation is disclosed)
* `--database-id TEXT`: Notion: the target database id to sync subjects into for this run (overrides config.notion.database_id). Share that database with your integration first. The NOTION_API_KEY token is read from the environment — never passed as a flag.
* `--no-update-blocks`: Notion: create-only — write a page's particle blocks once and never rewrite the managed block range on re-sync, preserving hand-edits. Default behaviour owns the managed range and overwrites it so re-sync is idempotent.
* `--help`: Show this message and exit.

## `particles project`

Render (or drift-check) a documentation projection.

    particles project docs/projection/readme.yaml README.md
    particles project docs/projection/readme.yaml --without-synthesis
    particles project docs/projection/readme.yaml --check   # CI drift gate
    # splice every declared region of the README in one pass:
    particles project docs/projection/readme.yaml --splice-all
    # re-roll a single sentinel region:
    particles project docs/projection/readme.yaml README.md --splice what-is
    # refresh the committed drift-gate bundle:
    particles project docs/projection/readme.yaml --export-corpus

**Usage**:

```console
$ particles project [OPTIONS] MANIFEST [OUTPUT]
```

**Arguments**:

* `MANIFEST`: Path to a projection manifest (e.g. docs/projection/readme.yaml).  [required]
* `[OUTPUT]`: Output Markdown path. Optional when the manifest sets `output:`.

**Options**:

* `--without-synthesis`: Render the deterministic structured listing — no LLM call, no ANTHROPIC_API_KEY, reproducible output. The drift gate uses this mode.
* `--check`: Drift gate: regenerate the deterministic snapshot and exit non-zero if it differs from the committed `<name>.snapshot.md`. Selection + structure are gated; LLM prose drift is advisory. No API key required.
* `--splice REGION`: Block-splice mode: write the rendered body *between* the `<!-- BEGIN/END PROJECTED: REGION -->` sentinels in the existing output file, preserving everything outside them, instead of overwriting the whole file. The output file must already carry the sentinel pair for REGION. On a manifest with per-section `region:` bindings, renders only that region's section — the single-region re-roll path.
* `--splice-all`: Multi-region block-splice: render every section that declares a `region:` and splice each body into its own sentinel pair in the output file, in one pass. Every derived section must declare a region.
* `--export-corpus`: Write the manifest's sibling `<name>.corpus.jsonl` gate bundle: exactly the particles the manifest's deterministic selection requires, encoded as interchange units the drift gate's ephemeral restore consumes. No render is performed.
* `--verbose`: Per-section progress logging.
* `--help`: Show this message and exit.

## `particles audit`

Audit an agent-memory directory: harvest, extract, and report the rot census.

**Usage**:

```console
$ particles audit [OPTIONS] [PATH]
```

**Arguments**:

* `[PATH]`: Memory directory (or single file) to harvest + audit. Omit to re-audit the existing store without harvesting.

**Options**:

* `--transcripts PATH`: Opt-in: also harvest session transcripts (*.jsonl) from DIR, newest first, capped at audit.transcript_max_entries (--max-entries overrides).
* `--max-entries INTEGER`: Cap harvested entries (default: audit.transcript_max_entries for transcripts; unlimited for memory files).
* `--estimate`: Print the extraction cost estimate and exit — no deposit, no LLM call.
* `--yes`: Skip the cost-confirmation prompt.
* `--judge`: LLM-judge duplicate pairs (verified duplicates) instead of REPORT-mode candidates.
* `--scope TEXT`: Semantic-finding scope (contradiction probe + duplicate scan): 'harvested' (default with PATH — headline counts only pairs touching this harvest's beliefs; the store-wide duplicate total is still disclosed) or 'store' (the whole store; the re-audit default).
* `--output PATH`: Also write the Markdown report to FILE.
* `--format TEXT`: Terminal format: markdown (default) or json.  [default: markdown]
* `--store TEXT`: Audit a named store (default: the default store).  [default: default]
* `-v, --verbose`
* `--debug`
* `--help`: Show this message and exit.

## `particles structure`

Annotate particles with a structured (subject-predicate-object) claim.

The annotation is derived from `content` and is never an assertion: this
verb cannot change a claim, its confidence, or its provenance. Particles
extracted since landed are annotated at extraction time for free;
this pass is for the ones that predate it, and it pays one LLM call each —
hence the rate limit and the resumable batch cap.

Particles whose prose has no honest triple are *skipped*, permanently and
without complaint. Absence of an annotation is a legal state.

**Usage**:

```console
$ particles structure [OPTIONS]
```

**Options**:

* `--limit INTEGER`: Max particles to annotate this run (default: structured_claim.backfill_batch_limit). Use 0 for the whole backlog in one run — safe, because the pass commits as it goes.
* `--rate-limit-per-minute INTEGER`: Max structurizer calls per minute (default: structured_claim.backfill_rate_limit_per_minute); 0 disables the delay.
* `--structurizer-version TEXT`: Regenerate annotations stamped with a version OTHER than this one, instead of annotating unannotated particles (mirrors `reindex --extractor-version`).
* `--dry-run`: Report the whole backlog (not the batch cap), the runs it implies, and current coverage; write nothing.
* `-v, --verbose`: Print per-particle progress.
* `--debug`: Debug logging.
* `--help`: Show this message and exit.

## `particles subjects`

Manage subjects (canonical real-world entities).

    particles subjects list [--order name|degree] [--phantoms-only]
    particles subjects search QUERY
    particles subjects show ID
    particles subjects alias ID NAME [NAME ...]
    particles subjects confirm SUBJECT_ID NAMESPACE:ID
    particles subjects unlink SUBJECT_ID NAMESPACE:ID
    particles subjects merge SOURCE_ID TARGET_ID [--dry-run]
    particles subjects delete SUBJECT_ID [--force]
    particles subjects gc [--dry-run]          (alias: prune-empty)
    particles subjects set-class SUBJECT_ID CLASS   (e.g. nmo:NumismaticObject)
    particles subjects split SOURCE_ID --particle PID [--particle PID ...] \
        (--new-name "Applied Optoelectronics" | --new-external-id wikidata:Q30297735) \
        [--dry-run]

**Usage**:

```console
$ particles subjects [OPTIONS] [ACTION] [REST]...
```

**Arguments**:

* `[ACTION]`: Action: list, search, show, alias, confirm, unlink, merge, split, delete, gc, set-class, fix-labels, find-duplicates  [default: list]
* `[REST]...`: Arguments for the chosen action

**Options**:

* `--dry-run`: Preview merge/split/gc without committing
* `--force`: delete only: remove a non-phantom subject (one with ACTIVE particles)
* `--phantoms-only`: list only: restrict to phantom subjects (zero ACTIVE particles)
* `--order TEXT`: list only: 'name' (alphabetical) or 'degree' (most ACTIVE linked particles first)  [default: name]
* `-p, --particle TEXT`: Particle ID to split off from the source (split only; repeat for multiple)
* `--new-name TEXT`: Approximate name for the new Subject (split only). Canonicalised via the resolver.
* `--new-external-id TEXT`: Authoritative external identifier for the new Subject (split only), e.g. wikidata:Q30297735. Skips resolver search; pulls metadata directly.
* `--help`: Show this message and exit.

## `particles benchmark`

Whole-pipeline system benchmarks — distinct from the per-extractor `particles extractor benchmark*` verbs.

**Usage**:

```console
$ particles benchmark [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `memory`: Run the LongMemEval agent-memory benchmark...

### `particles benchmark memory`

Run the LongMemEval agent-memory benchmark.

Reports four conditions in two labeled families: retrieval-stage
Recall@k / Precision@k (provenance-scored), and end-to-end QA accuracy
for qa_particles, qa_full_context (baseline), and qa_no_memory
(baseline) under one pinned answer model.

**Usage**:

```console
$ particles benchmark memory [OPTIONS]
```

**Options**:

* `--limit INTEGER RANGE`: Questions to run (stratified by type under the pinned seed; default: benchmark_memory.default_question_limit)  [x>=1]
* `--all`: Run every question in the variant (mutually exclusive with --limit)
* `--variant [oracle|s|m]`: LongMemEval variant: oracle | s | m (default: benchmark_memory.variant)
* `--types TEXT`: Comma-separated question-type filter (e.g. 'multi-session,knowledge-update')
* `--estimate`: Print the projected LLM call count + token volume and exit — no LLM call.
* `--yes`: Skip the cost-confirmation prompt.
* `--output PATH`: Write the rendered report to FILE as well as stdout.
* `--format [table|json]`: Output format  [default: table]
* `--store-dir PATH`: Directory for the per-question scratch stores (kept after the run; default: a deleted temp dir)
* `--dataset-file PATH`: Local LongMemEval-format JSON file (skips the pinned download — used by the checked-in fixture and pre-verified copies)
* `--context-budget INTEGER RANGE`: QA-at-budget clamp: cap condition ii's particle context at ~N tokens (rank order; baselines unclamped). Recorded on the run tuple — compare only against a matching run.  [x>=1]
* `--abstraction`: Ablation: run the abstraction-promotion pass (auto mode, age gate 0) on each scratch store between extract and retrieve. Recorded on the run tuple.
* `--concurrency INTEGER RANGE`: Run up to N questions at once (each owns its scratch store; the report is identical to a sequential run's). Practical ceiling is your API rate tier — past ~4-8 the extra parallelism becomes 429 retries, not speed.  [default: 1; x>=1]
* `--fresh`: Discard this experiment's checkpoint and start over. Runs are checkpointed per completed question by default, so an interrupted run resumes (and a completed run replays free) when re-invoked with identical knobs.
* `--pooled`: Dispatch each question's haystack extractions as one pooled Message Batches job — roughly halves the bill on a batch-eligible provider at the cost of latency (a batch's floor is one poll interval). Same model, prompt, and budget, so the report is comparable to an unpooled run's; degrades to sequential calls when llm.batch is off.
* `--batch-qa`: Submit the QA answer + judge calls (conditions ii-iv) as Message Batches jobs — one answer batch and one judge batch per condition, all at 50% price — instead of one sequential call per question. The sibling of --pooled for the answerer/judge (the two compose); same model/prompt/budget, so the report is comparable. Off by default (a batch's floor is one poll interval — the right trade for a paid run, the wrong one for a small/interactive run); degrades to sequential calls when llm.batch is off.
* `--memory [particles|chunks|notes]`: The memory under test: particles (the store — default), or a COMPARATOR memory over the same questions, answer scaffold, judge and retrieval scoring: chunks (raw-transcript RAG, no write-time LLM call) or notes (LLM-written session notes by the extraction model). The report's selection.memory names which ran.  [default: particles]
* `--baselines / --no-baselines`: Run the qa_full_context / qa_no_memory baseline conditions. --no-baselines is for a comparator run reusing the particles run's baseline columns (same tuple ⇒ same calls); they render `not run`.  [default: baselines]
* `--help`: Show this message and exit.

## `particles config`

Inspect and validate Particles configuration.

**Usage**:

```console
$ particles config [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `validate`: Load and validate config.yaml (+ env...

### `particles config validate`

Load and validate config.yaml (+ env overrides); report errors readably.

Resolves the same config the rest of the SDK would load, runs the Pydantic
validation, and prints a human-readable summary. Exits non-zero on the first
invalid field or an unparseable file, so it is safe to gate a deploy on
``particles config validate``.

**Usage**:

```console
$ particles config validate [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `particles conformance`

Conformance Profile checks (behavioural ground truth).

**Usage**:

```console
$ particles conformance [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `check`: Self-certify against the Conformance...
* `show`: Print the loaded Conformance Profile's...

### `particles conformance check`

Self-certify against the Conformance Profile; exit 1 on any FAIL.

**Usage**:

```console
$ particles conformance check [OPTIONS]
```

**Options**:

* `--level TEXT`: Which level to check: L2, L3, or all (default).  [default: all]
* `--json`: Emit the report as JSON instead of text.
* `--help`: Show this message and exit.

### `particles conformance show`

Print the loaded Conformance Profile's version, constants, and formulas.

**Usage**:

```console
$ particles conformance show [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `particles corpus`

Inspect deposited corpus entries.

**Usage**:

```console
$ particles corpus [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List all deposited corpus entries.
* `show`: Show details, extracted particles, and...
* `cat`: Dump a snapshot's stored content — the...
* `delete`: Delete a corpus entry, its snapshots, and...
* `retract`: Retract every live particle from a source,...
* `prune-orphans`: Sweep dangling rows left by older deletes...
* `refresh`: Re-check deposited local sources against...
* `fsck`: Audit every blob the store references;...
* `links`: Inspect cross-entry follow edges.

### `particles corpus list`

List all deposited corpus entries.

**Usage**:

```console
$ particles corpus list [OPTIONS]
```

**Options**:

* `--source-type TEXT`: Filter by source type (PDF, WEB_PAGE, WIKIDATA_API, …)
* `--json`: Emit machine-readable JSON with the full, untruncated fields (entry_id, source_type, uri_r, extraction_status, particle_count, tags, created_at) instead of the human table.
* `--help`: Show this message and exit.

### `particles corpus show`

Show details, extracted particles, and follow edges for a corpus entry.

**Usage**:

```console
$ particles corpus show [OPTIONS] ENTRY_ID
```

**Arguments**:

* `ENTRY_ID`: Entry ID (prefix OK)  [required]

**Options**:

* `--limit INTEGER`: Max particles to show  [default: 10]
* `--help`: Show this message and exit.

### `particles corpus cat`

Dump a snapshot's stored content — the exact bytes the extractor saw.

Accepts a snapshot ID or a corpus-entry ID (prefix OK); an entry resolves to
its most-recent snapshot. By default renders the same text the extractor
derives from the blob (html2text for HTML, pypdf for PDF), so an empty
preview explains an "Empty content" / zero-particle extraction. ``--raw``
streams the original bytes (pipe to a file or ``less``). Identifying
metadata (snapshot id, hash, byte size) is written to stderr, so stdout
stays pipeable.

**Usage**:

```console
$ particles corpus cat [OPTIONS] SELECTOR
```

**Arguments**:

* `SELECTOR`: Snapshot ID or corpus-entry ID (prefix OK; entry → latest snapshot)  [required]

**Options**:

* `--raw`: Write the raw stored bytes to stdout instead of the text preview.
* `--help`: Show this message and exit.

### `particles corpus delete`

Delete a corpus entry, its snapshots, and all particles sourced from it.

**Usage**:

```console
$ particles corpus delete [OPTIONS] ENTRY_ID
```

**Arguments**:

* `ENTRY_ID`: Entry ID (prefix OK)  [required]

**Options**:

* `-y, --yes`: Skip confirmation prompt
* `--help`: Show this message and exit.

### `particles corpus retract`

Retract every live particle from a source, preserving the corpus + snapshots.

The non-destructive sibling of ``corpus delete``: live particles
(ACTIVE / INCONSISTENCY) become RETRACTED with reason SOURCE_RETRACTED; the
entry, its snapshots, and the particles themselves survive so the audit
trail is intact. Idempotent. Run ``particles lint`` afterwards to cascade
PROVENANCE_STALE to downstream particles.

**Usage**:

```console
$ particles corpus retract [OPTIONS] ENTRY_ID
```

**Arguments**:

* `ENTRY_ID`: Entry ID (prefix OK)  [required]

**Options**:

* `--reason TEXT`: Operator rationale, recorded in the event log
* `--dry-run`: Show the plan without writing
* `-y, --yes`: Skip confirmation prompt
* `--help`: Show this message and exit.

### `particles corpus prune-orphans`

Sweep dangling rows left by older deletes (orphan subjects, stale index rows).

Pre-fix ``corpus delete`` removed particles but not the index rows keyed
on them, so long-lived DBs accumulate ``particle_subjects`` /
``particle_tag_edges`` / ``particle_relations`` rows pointing at deleted
particles, subjects with no remaining link, and ``synthesis_cache`` rows
keyed on vanished subjects. This one-off verb cleans all of them.

**Usage**:

```console
$ particles corpus prune-orphans [OPTIONS]
```

**Options**:

* `-y, --yes`: Skip confirmation prompt
* `--help`: Show this message and exit.

### `particles corpus refresh`

Re-check deposited local sources against the files on disk.

Walks every ``LAZY`` entry with a ``file://`` URI-R, comparing the file's
mtime and then its SHA-256 against the latest snapshot. A changed file gets
a new PENDING snapshot; ``particles extract --all-pending`` (or tonight's
consolidation run) turns that into current beliefs.

This is the on-demand form of consolidation pass 0.5 — the scheduled cycle
runs the same sweep nightly.

**Usage**:

```console
$ particles corpus refresh [OPTIONS] [ENTRY_ID]
```

**Arguments**:

* `[ENTRY_ID]`: Refresh one entry (full id or unambiguous prefix). Omit to sweep all.

**Options**:

* `--force`: Tier 3: re-check regardless of fetch_policy and the per-source-type re-fetch floor. The escape hatch for a content change that preserved the file's mtime.
* `--backfill-cascade`: Instead of re-checking sources, apply the generation cascade to MUTABLE entries whose snapshots already moved — demoting ACTIVE particles anchored to a superseded snapshot. Stores that predate the change carry a backlog of these; the forward-looking cascade only fires on newly-extracted snapshots.
* `-y, --yes`: Skip the confirmation prompt
* `-v, --verbose`: Verbose logging
* `--help`: Show this message and exit.

### `particles corpus fsck`

Audit every blob the store references; optionally re-home the strays.

Read-only by default and exhaustive — the operator-invoked sibling of the
sampled probe ``config validate`` runs. Reports three disjoint counts:
**present** in the resolved blob dir, **found elsewhere** under a
``--search`` root, and **missing**.

``--re-home`` copies (never moves) the strays home, rejecting any candidate
whose recomputed SHA-256 does not match the name it was found under. The
database is never written: blobs that are genuinely gone are reported with
their entry IDs and URIs, and the choice between re-depositing from source
and retracting stays yours.

Exits non-zero while any referenced blob is still unreachable.

**Usage**:

```console
$ particles corpus fsck [OPTIONS]
```

**Options**:

* `--search PATH`: Also look for strays under this blob root — the directory holding the two-character shards (repeatable). Nothing is inferred: the audit tells you what is missing so you can point --search at where you think it went.
* `--re-home`: Copy digest-verified strays found under --search into the blob dir.
* `--dry-run`: Report what --re-home would copy, without copying.
* `--help`: Show this message and exit.

### `particles corpus links`

Inspect cross-entry follow edges.

**Usage**:

```console
$ particles corpus links [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List follow edges written by deposit-time...
* `suggest`: Suggest undeposited URLs the corpus...
* `dismiss`: Stop a URL from resurfacing in ``corpus...

#### `particles corpus links list`

List follow edges written by deposit-time URL following.

Without an entry-id, lists every edge in deposit-time order. With an
entry-id, shows the edges touching that entry (outgoing / incoming
per ``--direction``). Operators use this to audit which Reddit / HN
/ Mastodon posts amplified which external sources.

**Usage**:

```console
$ particles corpus links list [OPTIONS] [ENTRY_ID]
```

**Arguments**:

* `[ENTRY_ID]`: Entry ID (prefix OK). Omit to list every follow edge.

**Options**:

* `--direction TEXT`: Filter when entry-id set: out (this→linked), in (others→this), both.  [default: both]
* `--help`: Show this message and exit.

#### `particles corpus links suggest`

Suggest undeposited URLs the corpus frequently cites.

Ranks URLs mentioned across the corpus but not yet deposited, by
trust-weighted distinct-source diversity × recency. Suggestion-only —
nothing is fetched or crawled. Deposit one with ``particles deposit <url>``;
silence one with ``particles corpus links dismiss <url>``.

**Usage**:

```console
$ particles corpus links suggest [OPTIONS]
```

**Options**:

* `--limit INTEGER`: Max suggestions to show (default: config rank_cap).
* `--min-sources INTEGER`: Min distinct citing sources to surface (default: config).
* `-o, --output-format TEXT`: table | json  [default: table]
* `--help`: Show this message and exit.

#### `particles corpus links dismiss`

Stop a URL from resurfacing in ``corpus links suggest``.

A permanent dismiss (the default) suppresses the URL indefinitely; ``--snooze
N`` suppresses it for N days. The action is audited in the operator event
log.

**Usage**:

```console
$ particles corpus links dismiss [OPTIONS] URL
```

**Arguments**:

* `URL`: URL to dismiss (canonicalized before matching).  [required]

**Options**:

* `--snooze INTEGER`: Snooze for N days instead of a permanent dismiss.
* `--help`: Show this message and exit.

## `particles curate`

Bus-stop editing — the finite, leverage-ranked curation queue.

**Usage**:

```console
$ particles curate [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-n, --limit INTEGER`: Cap the cards shown (default: curation.session_size).
* `-k, --kind TEXT`: Restrict to one card kind (e.g. stale, contested).
* `--semantic`: Run the LLM-assisted finders (semantic contradiction).
* `--refresh`: Rebuild the card collection before showing it. Slow — the finders re-run store-wide. Run this once on a store with no collection yet; the nightly `memory consolidate` does it for you after that.
* `--no-snapshot`: Bypass the persisted collection entirely and run the finders for this invocation without caching the result.
* `--verbose`
* `--debug`
* `--help`: Show this message and exit.

**Commands**:

* `apply`: Apply a gesture to a card, dispatching...

### `particles curate apply`

Apply a gesture to a card, dispatching onto the existing write op.

**Usage**:

```console
$ particles curate apply [OPTIONS] GESTURE CARD_KEY
```

**Arguments**:

* `GESTURE`: affirm | snooze | dismiss | retract | merge | deposit | assign-subject | accept | reject  [required]
* `CARD_KEY`: The card key shown in the queue listing.  [required]

**Options**:

* `--reason TEXT`: Rationale (recorded on retract).
* `--days INTEGER`: Snooze window in days.
* `--subject TEXT`: Subject id or name for the assign-subject gesture.
* `--help`: Show this message and exit.

## `particles engine`

Remote engine server.

**Usage**:

```console
$ particles engine [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `serve`: Run the FastAPI engine, binding HOST:PORT.

### `particles engine serve`

Run the FastAPI engine, binding HOST:PORT.

Unifies the bind with the fail-closed gate: HOST sets
``api.bind_host`` so a non-loopback bind without a real ``PARTICLES_API_KEY``
is refused before the socket opens.

With ``--daemon`` (or ``daemon.enabled``) the process also schedules its own
background work in the FastAPI lifespan — the rider on the
external-scheduler contract. Without it, this command behaves exactly as it
always has.

**Usage**:

```console
$ particles engine serve [OPTIONS] HOST:PORT
```

**Arguments**:

* `HOST:PORT`: Interface and port to bind, e.g. 0.0.0.0:8000 (LAN/Tailscale) or localhost:8000 (loopback-only dev).  [required]

**Options**:

* `--daemon`: Resident mode: also run the in-process consolidation tick and intake watchers, so no launchd/cron is needed alongside the engine. Overrides daemon.enabled; configure the rest under `daemon`.
* `--help`: Show this message and exit.

## `particles events`

Inspect the operator event log.

**Usage**:

```console
$ particles events [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List operator events, newest first.
* `show`: Show one operator event in full (header +...

### `particles events list`

List operator events, newest first.

**Usage**:

```console
$ particles events list [OPTIONS]
```

**Options**:

* `--particle TEXT`: Only events touching this particle id
* `--subject TEXT`: Only events touching this subject id
* `--entry TEXT`: Only events touching this corpus entry id
* `--type TEXT`: Only events of this type (e.g. SOURCE_RETRACTED)
* `--limit INTEGER`: Maximum events to show  [default: 50]
* `--help`: Show this message and exit.

### `particles events show`

Show one operator event in full (header + refs + payload).

**Usage**:

```console
$ particles events show [OPTIONS] EVENT_ID
```

**Arguments**:

* `EVENT_ID`: Event ID  [required]

**Options**:

* `--help`: Show this message and exit.

## `particles extractor`

Manage extractor registry (Extension A).

**Usage**:

```console
$ particles extractor [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `conform`: Run an extractor against the conformance...
* `generate-fixture`: Turn a deposited corpus entry into a...
* `list`: List all registered extractors with...
* `trust-set`: Override the trust weight for an extractor.
* `benchmark`: Run extraction-quality benchmarks against...
* `benchmark-modality`: Measure assertion_modality classification...
* `benchmark-polarity`: Measure claim-polarity classification...
* `benchmark-validity`: Measure event-anchored-validity quality...
* `benchmark-compare`: Compare two or more extractors against the...
* `calibrate`: Fit a temperature-scaling calibration for...
* `calibrations`: List stored calibrations per (provider,...
* `calibration-forget`: Retire one stored calibration record (ADR...

### `particles extractor conform`

Run an extractor against the conformance fixture corpus.

Scores the fixtures the production registry routes to this extractor
. ``--all-accepted`` widens the run to every fixture the
extractor would take if handed it — for the fallback that is the
whole corpus, so the result is a deliberate probe, not the extractor's
conformance score, and it never updates the stored conformance verdict.

Phase 1 (current): report-only. Exit code 0 unless --fail-on is given.
Exit code 1 indicates the contract failed (a REQUIRED field missing, or a
FAIL-severity diversity rule violated); --fail-on warn additionally treats
RECOMMENDED warnings as fatal. An ADVISORY diversity finding (
`uncertainty_nature` is the one shipped today) is reported and never
affects the exit code.

**Usage**:

```console
$ particles extractor conform [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID of a registered extractor  [required]

**Options**:

* `--fixtures PATH`: Override fixture directory (default: tests/conformance/fixtures)
* `--recommended-threshold FLOAT RANGE`: Minimum populate-rate for RECOMMENDED fields (0.0–1.0)  [default: 0.8; 0.0<=x<=1.0]
* `--format [table|json]`: Output format  [default: table]
* `--fail-on [error|warn]`: Exit non-zero on errors only (default) or on warnings as well  [default: error]
* `--all-accepted`: Score every fixture the extractor accepts(), not just the ones the registry routes to it. Report-only: never stores the verdict
* `--help`: Show this message and exit.

### `particles extractor generate-fixture`

Turn a deposited corpus entry into a conformance fixture skeleton.

Reads the entry's latest stored snapshot + raw blob and writes
``manifest.yaml`` + ``content.bin`` + ``snapshot.json`` under
``<output-dir>/<fixture-id>``, registering it in ``MANIFEST.yaml``.
``expected_acceptors`` is left empty for you to fill after verifying which
extractors the fixture should exercise.

**Usage**:

```console
$ particles extractor generate-fixture [OPTIONS] ENTRY_ID
```

**Arguments**:

* `ENTRY_ID`: Corpus entry ID (prefix OK)  [required]

**Options**:

* `--id TEXT`: Fixture id (default: a slug of the entry URI + id prefix)
* `--source-type TEXT`: Override the entry's source type
* `--output-dir PATH`: Fixture corpus directory (default: tests/conformance/fixtures)  [default: tests/conformance/fixtures]
* `--force`: Overwrite an existing fixture directory
* `--help`: Show this message and exit.

### `particles extractor list`

List all registered extractors with version, trust weight, and domain coverage.

**Usage**:

```console
$ particles extractor list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `particles extractor trust-set`

Override the trust weight for an extractor.

**Usage**:

```console
$ particles extractor trust-set [OPTIONS] EXTRACTOR_ID WEIGHT
```

**Arguments**:

* `EXTRACTOR_ID`: Extractor ID  [required]
* `WEIGHT`: New trust weight [0.0–1.0]  [required]

**Options**:

* `--help`: Show this message and exit.

### `particles extractor benchmark`

Run extraction-quality benchmarks against an extractor.

Discovers every suite under --suites-dir that this extractor is the
production routing choice for — the registry ladder, read back
through ``select_extractor``, so the fallback extractor no
longer inherits every domain suite. ``--suite`` runs a named suite
regardless of routing. Emits one report per suite, and persists each
report as a JSON file under ``benchmark.runs_dir`` (stamped with the
resolved extraction provider:model pairing) unless --no-save is set.
With --fail-on set, exits non-zero when any suite's named metric
crosses the threshold.

``--runs N`` repeats each suite N times and reports each metric's mean,
range and standard deviation instead of a single point estimate — the
error bars a provider comparison needs. Every pass persists
its own report file, so the series is still one run per JSON envelope.
``--fail-on`` is evaluated against the **mean** across runs.

**Usage**:

```console
$ particles extractor benchmark [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID of a registered extractor  [required]

**Options**:

* `--suite TEXT`: Run only the suite with this suite_id (default: every suite the extractor is the routing choice for)
* `--suites-dir PATH`: Override suite directory (default: tests/benchmark/suites)
* `--fixtures PATH`: Override fixture directory used to resolve `fixture:` references (default: tests/conformance/fixtures)
* `--judge [embedding|llm]`: Equivalence judge: embedding cosine (default) or LLM-judge  [default: embedding]
* `--threshold FLOAT RANGE`: Cosine-similarity threshold for the embedding judge  [default: 0.8; 0.0<=x<=1.0]
* `--format [table|json]`: Output format  [default: table]
* `--fail-on [none|precision|recall|calibration]`: Exit non-zero when the named metric falls below --fail-threshold  [default: none]
* `--fail-threshold FLOAT RANGE`: Threshold for --fail-on (precision/recall: minimum; calibration: maximum)  [default: 0.9; 0.0<=x<=1.0]
* `--no-save`: Skip persisting the run report JSON under benchmark.runs_dir
* `--runs INTEGER RANGE`: Repeat each suite N times and report mean ± spread per metric. Costs N× the LLM calls; N=1 (default) is unchanged  [default: 1; x>=1]
* `--estimate`: Print the projected LLM cost of the run and exit without running
* `-y, --yes`: Pre-confirm the cost gate for a repeat run (--runs N)
* `--help`: Show this message and exit.

### `particles extractor benchmark-modality`

Measure assertion_modality classification quality.

Reports per-modality precision/recall, the dangerous **false-non-FALSIFIABLE
rate** the journal extractor's inverted default raises, and the
whole-entry **narrative-emission rate**. Discovers every modality
suite under --suites-dir the extractor is the production routing choice
for (or runs only --suite). Report-only and **integration-tier** — it drives the
extractor's LLM call, so it needs ANTHROPIC_API_KEY.

**Usage**:

```console
$ particles extractor benchmark-modality [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID of a registered extractor  [required]

**Options**:

* `--suite TEXT`: Run only the modality suite with this suite_id (default: every suite the extractor is the routing choice for)
* `--suites-dir PATH`: Override modality-suite directory (default: tests/benchmark/modality)
* `--judge [embedding|llm]`: Claim-alignment judge: embedding cosine (default) or LLM-judge  [default: embedding]
* `--threshold FLOAT RANGE`: Cosine floor for aligning an emitted claim to a gold label (looser than the content harness's 0.80 — journal claims are reified paraphrases of their gold labels)  [default: 0.65; 0.0<=x<=1.0]
* `--format [table|json]`: Output format  [default: table]
* `--help`: Show this message and exit.

### `particles extractor benchmark-polarity`

Measure claim-polarity classification quality.

Reports the dangerous **wrong-`DECLINED` rate** — a real current decision
(ASSERTED) wrongly classified DECLINED and thereby silently hidden from the
default surface (the headline, the README-projection-trust risk;
cap. 1) — plus its superset the wrong-hidden rate and per-polarity
precision/recall. Discovers every polarity suite under --suites-dir the
extractor is the production routing choice for (or runs only
--suite). Report-only and
**integration-tier** — it drives the extractor's LLM call, so it needs
ANTHROPIC_API_KEY.

**Usage**:

```console
$ particles extractor benchmark-polarity [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID of a registered extractor  [required]

**Options**:

* `--suite TEXT`: Run only the polarity suite with this suite_id (default: every suite the extractor is the routing choice for)
* `--suites-dir PATH`: Override polarity-suite directory (default: tests/benchmark/polarity)
* `--judge [embedding|llm]`: Claim-alignment judge: embedding cosine (default) or LLM-judge  [default: embedding]
* `--threshold FLOAT RANGE`: Cosine floor for aligning an emitted claim to a gold label (looser than the content harness's 0.80 — the general extractor emits near-paraphrases of its gold labels)  [default: 0.65; 0.0<=x<=1.0]
* `--format [table|json]`: Output format  [default: table]
* `--help`: Show this message and exit.

### `particles extractor benchmark-validity`

Measure event-anchored-validity quality.

Reports the dangerous **wrong-expiry rate** — of the aligned claims whose
gold is durable (no boundary), the fraction the extractor wrongly assigned a
``valid_until`` and thereby set up for silent retirement by the §9.3
staleness lint (the headline, the over-eager-expiry risk) — plus existence
precision/recall of correct date-bounded extraction and date accuracy.
Discovers every validity suite under --suites-dir the extractor is the
production routing choice for (or runs only --suite). Report-only
and **integration-tier** — it drives the extractor's LLM call, so it needs
ANTHROPIC_API_KEY.

**Usage**:

```console
$ particles extractor benchmark-validity [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID of a registered extractor  [required]

**Options**:

* `--suite TEXT`: Run only the validity suite with this suite_id (default: every suite the extractor is the routing choice for)
* `--suites-dir PATH`: Override validity-suite directory (default: tests/benchmark/validity)
* `--judge [embedding|llm]`: Claim-alignment judge: embedding cosine (default) or LLM-judge  [default: embedding]
* `--threshold FLOAT RANGE`: Cosine floor for aligning an emitted claim to a gold label (looser than the content harness's 0.80 — the general extractor emits near-paraphrases of its gold labels)  [default: 0.65; 0.0<=x<=1.0]
* `--format [table|json]`: Output format  [default: table]
* `--help`: Show this message and exit.

### `particles extractor benchmark-compare`

Compare two or more extractors against the same benchmark corpus.

    particles extractor benchmark-compare \
        --extractor-id numista-coin-extractor \
        --extractor-id numista-coin-extractor-v3

Cells where an extractor declined a suite's source_type render as
`—` in the table view and `null` in JSON output.

**Usage**:

```console
$ particles extractor benchmark-compare [OPTIONS]
```

**Options**:

* `--extractor-id TEXT`: EXTRACTOR_ID to include in the comparison (repeat ≥2 times)  [required]
* `--suite TEXT`: Restrict to a single suite_id (default: every suite whose source_types intersect ANY supplied extractor's accepts())
* `--suites-dir PATH`: Override suite directory (default: tests/benchmark/suites)
* `--fixtures PATH`: Override fixture directory (default: tests/conformance/fixtures)
* `--judge [embedding|llm]`: Equivalence judge: embedding cosine (default) or LLM-judge  [default: embedding]
* `--threshold FLOAT RANGE`: Embedding-judge cosine threshold  [default: 0.8; 0.0<=x<=1.0]
* `--format [table|json]`: Output format  [default: table]
* `--help`: Show this message and exit.

### `particles extractor calibrate`

Fit a temperature-scaling calibration for an extractor.

Runs every applicable **calibration** suite — `tests/benchmark/calibration/`,
a sibling of the §13.3 `suites/` directory whose gold sets are deliberately
partial — collects (raw_confidence, correct) pairs from
emitted-vs-matched, fits a single T via NLL minimisation, and persists the
result on the extractor record. Subsequent particles produced by this
extractor carry calibration_source=CALIBRATED_BENCHMARK and a
temperature-scaled confidence value. Pre-existing particles are
unaffected — operators who want retroactive application should run
`particles reindex --extractor-id <id>`.

The fit is refused rather than persisted when it cannot mean anything
: degenerate labels, a temperature on an
optimizer bound, fewer than two distinct movable confidences, or a
calibration that does not reduce calibration error.

Unlike `extractor benchmark`, this verb defaults to the **LLM** equivalence
judge — see `_extractor_calibrate` for why the calibration label
cannot afford the embedding judge's paraphrase misses.

**Usage**:

```console
$ particles extractor calibrate [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID of a registered extractor  [required]

**Options**:

* `--suite TEXT`: Restrict calibration to a single suite_id (default: every suite the extractor is the routing choice for)
* `--suites-dir PATH`: Override suite directory (default: tests/benchmark/calibration)
* `--fixtures PATH`: Override fixture directory used to resolve `fixture:` references (default: tests/conformance/fixtures)
* `--judge [embedding|llm]`: Equivalence judge for the calibration label: LLM-judge (default) or embedding cosine  [default: llm]
* `--dry-run / --no-dry-run`: Fit and print but do not persist the calibration record  [default: no-dry-run]
* `--regenerate / --no-regenerate`: Overwrite an existing calibration; without it, an extractor that already has one exits 1  [default: no-regenerate]
* `--help`: Show this message and exit.

### `particles extractor calibrations`

List stored calibrations per (provider, model) for an extractor.

Each `extractor calibrate` run stores one record keyed by the extraction
model it ran under, so several models' calibrations coexist; the one matching
the configured extraction model is applied at extraction time.

Each record is checked for **suite-set staleness**: a fit whose
contributing suites differ from the ones the extractor auto-matches today
answers a question it is no longer asked. The check needs the benchmark
suites, which ship in neither the wheel nor the sdist, so on an installed
SDK the listing prints unannotated.

**Usage**:

```console
$ particles extractor calibrations [OPTIONS] EXTRACTOR_ID
```

**Arguments**:

* `EXTRACTOR_ID`: Extractor id to list calibrations for  [required]

**Options**:

* `--suites-dir PATH`: Override suite directory used to report suite-set staleness (default: tests/benchmark/calibration)
* `--help`: Show this message and exit.

### `particles extractor calibration-forget`

Retire one stored calibration record.

The counterpart to `extractor calibrate`. Before this verb a stored
calibration could only be *replaced* — by re-fitting under the same
pairing — so a record fitted against a model no longer reachable (a local
endpoint since torn down) could not be retired at all without standing
that model back up.

Removing a record returns that pairing to `calibration_source=
EXTRACTOR_DIRECT`, the documented fallback for an uncalibrated pairing.
Particles already in the store keep the confidence they were minted with
; run `particles reindex --extractor-id <id>` to re-mint them.

**Usage**:

```console
$ particles extractor calibration-forget [OPTIONS] EXTRACTOR_ID PROVIDER_MODEL
```

**Arguments**:

* `EXTRACTOR_ID`: EXTRACTOR_ID the calibration belongs to  [required]
* `PROVIDER_MODEL`: The "<provider>:<model>" pairing to retire, as printed by `extractor calibrations`  [required]

**Options**:

* `-y, --yes`: Skip the confirmation prompt (for scripted use)
* `--help`: Show this message and exit.

## `particles hook`

Machine-facing Claude Code lifecycle hooks. Reads the hook JSON from stdin; degrades to exit 0 on any failure.

**Usage**:

```console
$ particles hook [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `session-start`: SessionStart hook: push the memory digest...
* `session-end`: SessionEnd hook: harvest the session...
* `log`: Print recent hook-log entries (one JSONL...
* `doctor`: Diagnose whether ``particles hook``...

### `particles hook session-start`

SessionStart hook: push the memory digest into the session's context.

**Usage**:

```console
$ particles hook session-start [OPTIONS]
```

**Options**:

* `--store TEXT`: Memory store whose digest is pushed.  [required]
* `--help`: Show this message and exit.

### `particles hook session-end`

SessionEnd hook: harvest the session transcript + memory files.

**Usage**:

```console
$ particles hook session-end [OPTIONS]
```

**Options**:

* `--store TEXT`: Memory store the harvest deposits into.  [required]
* `--help`: Show this message and exit.

### `particles hook log`

Print recent hook-log entries (one JSONL line per hook invocation).

**Usage**:

```console
$ particles hook log [OPTIONS]
```

**Options**:

* `-n, --tail INTEGER`: How many recent entries to print.  [default: 20]
* `--help`: Show this message and exit.

### `particles hook doctor`

Diagnose whether ``particles hook`` resolves ``store`` from the current directory.

The lifecycle hooks degrade to exit 0 on any failure, so a mis-resolved
store fails silently. This verb makes that resolution visible:
which config.yaml is found, which DSN the handle resolves to, whether the DB
file exists and carries the corpus tables. Exits non-zero when the store is
unusable, so it can gate an operator's "is this thing on?" check.

**Usage**:

```console
$ particles hook doctor [OPTIONS]
```

**Options**:

* `--store TEXT`: Store handle to check.  [default: default]
* `--help`: Show this message and exit.

## `particles import`

Bulk-onboard existing knowledge bases.

**Usage**:

```console
$ particles import [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `vault`: Walk a Markdown vault and deposit every...
* `project`: Walk a project tree and deposit every...
* `web-clipper`: Walk a frontmatter-Markdown captures...

### `particles import vault`

Walk a Markdown vault and deposit every ``.md`` file as ``LOCAL_MARKDOWN``.

Recursively walks ``vault_dir`` (skipping any path under a ``_`` or ``.``
component — Obsidian's ``.obsidian/`` settings, ``_attachments/``, etc.)
and registers each Markdown file in the corpus. Re-running on the same
vault is idempotent — existing ``content_hash`` deduplication means
unchanged files are not re-deposited.

Typical onboarding workflow:

    particles import vault ~/Documents/MyVault
    particles extract --all-pending
    particles lint

**Usage**:

```console
$ particles import vault [OPTIONS] VAULT_DIR
```

**Arguments**:

* `VAULT_DIR`: Path to an Obsidian vault (or any directory of Markdown notes).  [required]

**Options**:

* `--deposited-by TEXT`: Agent or operator ID.  [default: operator]
* `--tags TEXT`: Comma-separated tags applied to every deposited entry.
* `-v, --verbose`: Print per-file progress while depositing.
* `--debug`: Show DEBUG-level logs from deposit/fetch.
* `--help`: Show this message and exit.

### `particles import project`

Walk a project tree and deposit every source file as ``PYTHON_SOURCE``.

Recursively walks ``project_dir`` for source files (``.py`` by default; see
``import_project.extensions``), skipping dot-prefixed components and the
configured build/cache directories (``import_project.ignore_dirs``) — but
keeping underscore-prefixed module files (``__init__.py`` / ``_shared.py``).
Re-running on the same tree is idempotent: ``content_hash`` deduplication
means only changed files get a new snapshot.

Typical onboarding workflow:

    particles import project ~/src/myproject
    particles extract --all-pending   # docstring extractor runs
    particles lint                    # code/design drift surfaces

**Usage**:

```console
$ particles import project [OPTIONS] PROJECT_DIR
```

**Arguments**:

* `PROJECT_DIR`: Path to a software-project tree (walked recursively).  [required]

**Options**:

* `--deposited-by TEXT`: Agent or operator ID.  [default: operator]
* `--tags TEXT`: Comma-separated tags applied to every deposited entry.
* `--ext TEXT`: Comma-separated file extensions to deposit this run (e.g. '.py'), overriding the configured import_project.extensions set.
* `-v, --verbose`: Print per-file progress while depositing.
* `--debug`: Show DEBUG-level logs from deposit/fetch.
* `--help`: Show this message and exit.

### `particles import web-clipper`

Walk a frontmatter-Markdown captures folder and deposit each as ``WEB_PAGE``.

Recursively walks ``captures_dir`` (skipping any path under a ``_`` or ``.``
component, the vault ignore policy) and deposits each ``.md`` capture with the
provenance its frontmatter carries restored: the ``source:`` / ``url:`` URL
becomes the entry's ``uri_r`` (fragment-stripped, **not** fetched), the
``published:`` date becomes ``content_published_at`` (below an
explicit operator date), the frontmatter ``tags:`` merge with ``--tags``, and
the source type is ``WEB_PAGE`` — so a clipping is trustable, decayable, and
queryable as the web page it is, unlike the same folder run through
``import vault``. The frontmatter-stripped **body** is the deposited content.
A capture whose header is absent / malformed falls back to a plain
``LOCAL_MARKDOWN`` body deposit. Re-running is idempotent (body-hash dedup).

Typical onboarding workflow:

    particles import web-clipper ~/Obsidian/Clippings
    particles extract --all-pending
    particles lint

**Usage**:

```console
$ particles import web-clipper [OPTIONS] CAPTURES_DIR
```

**Arguments**:

* `CAPTURES_DIR`: Path to an Obsidian Web Clipper captures folder (walked recursively).  [required]

**Options**:

* `--deposited-by TEXT`: Agent or operator ID.  [default: web-clipper]
* `--tags TEXT`: Comma-separated tags merged with each capture's frontmatter tags.
* `-v, --verbose`: Print per-file progress while depositing.
* `--debug`: Show DEBUG-level logs from deposit/fetch.
* `--help`: Show this message and exit.

## `particles inbox`

Process URLs queued from an iOS Shortcut via iCloud Drive.

**Usage**:

```console
$ particles inbox [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `process`: Process all pending URLs in the inbox...
* `watch`: Continuously poll the inbox file.
* `status`: Show pending vs processed counts in the...

### `particles inbox process`

Process all pending URLs in the inbox file, then exit.

Suitable for cron / launchd / a desktop keyboard shortcut. Each
pending URL is deposited and the inbox line is rewritten in place
with the resulting ``entry_id``.

**Usage**:

```console
$ particles inbox process [OPTIONS]
```

**Options**:

* `--deposited-by TEXT`: Recorded as the depositor on each entry.  [default: inbox]
* `--help`: Show this message and exit.

### `particles inbox watch`

Continuously poll the inbox file. Ctrl-C to stop.

Uses mtime to skip the file read when nothing has changed since
the last poll — cheap enough to leave running in a terminal tab.

**Usage**:

```console
$ particles inbox watch [OPTIONS]
```

**Options**:

* `--interval INTEGER`: Seconds between polls. Defaults to inbox.poll_interval_seconds (30).
* `--deposited-by TEXT`: Recorded as the depositor on each entry.  [default: inbox]
* `--help`: Show this message and exit.

### `particles inbox status`

Show pending vs processed counts in the inbox file.

**Usage**:

```console
$ particles inbox status [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `particles init`

Install (or remove) an agent-harness memory integration.

**Usage**:

```console
$ particles init [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `claude-code`: Install the Claude Code hook integration...

### `particles init claude-code`

Install the Claude Code hook integration (SessionStart digest push + SessionEnd harvest).

**Usage**:

```console
$ particles init claude-code [OPTIONS]
```

**Options**:

* `--store TEXT`: Memory store handle baked into the hook commands. Default: the single mcp.write.enabled_stores entry; a fresh install auto-creates 'memory'.
* `--project`: Install into the current repo's .claude/settings.local.json (gitignored) instead of the user-level ~/.claude/settings.json.
* `--remove`: Remove exactly the Particles-owned hook entries (and revert the store auto-create while the store is still empty).
* `--dry-run`: Print the resulting files without writing anything.
* `--command TEXT`: Override the hook command base (default: the absolute path of the running `particles` console script).
* `--no-audit`: Skip the first-run memory-audit hand-off.
* `--skills / --no-skills`: Also install the shipped agent-onboarding skill files into the harness's skills directory (a Particles-owned subdirectory; --remove deletes exactly that). Default on — an agent that has the tools but not the guidance is the gap these close.  [default: skills]
* `--json`: Emit a machine-readable result on stdout — what was created, what was merged, and what is left for the human — so an agent can run the installer and report the outcome instead of scraping human-formatted output. Implies --no-audit: the audit hand-off is interactive, and the result names it under next_steps.
* `--help`: Show this message and exit.

## `particles interchange`

Export / import portable store bundles.

**Usage**:

```console
$ particles interchange [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `export`: Write a store-export bundle (manifest +...
* `import`: Import a store-export bundle into a store...
* `restore`: Faithfully reconstruct a bundle into an...

### `particles interchange export`

Write a store-export bundle (manifest + particles/subjects members).

**Usage**:

```console
$ particles interchange export [OPTIONS]
```

**Options**:

* `-o, --output PATH`: Bundle directory to write  [required]
* `--store TEXT`: Store handle to export  [default: default]
* `--format TEXT`: Bundle container: jsonl (canonical, one unit per line) or yaml (human-editable YAML-LD, same data model). Both round-trip through `interchange import` unchanged.  [default: jsonl]
* `--help`: Show this message and exit.

### `particles interchange import`

Import a store-export bundle into a store (single-store writes, §6.6).

**Usage**:

```console
$ particles interchange import [OPTIONS] BUNDLE
```

**Arguments**:

* `BUNDLE`: Bundle directory to import  [required]

**Options**:

* `--store TEXT`: Target store handle  [default: default]
* `--help`: Show this message and exit.

### `particles interchange restore`

Faithfully reconstruct a bundle into an EMPTY store, origin ids preserved.

Unlike ``import`` (claim-fingerprint merge, fresh ids), ``restore`` reconstructs
the bundle's own store: ids are preserved verbatim and no §6.6 reconcile runs.
The target must be empty; a populated target is refused.

**Usage**:

```console
$ particles interchange restore [OPTIONS] BUNDLE
```

**Arguments**:

* `BUNDLE`: Bundle directory or a single particles JSONL file to restore  [required]

**Options**:

* `--store TEXT`: Target store handle (must be empty)  [default: default]
* `--help`: Show this message and exit.

## `particles links`

Manage typed relations between particles (e.g. co-evidential links).

**Usage**:

```console
$ particles links [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `add`: Create a typed relation between two...
* `remove`: Remove a typed relation between two...
* `list`: List all relations incident to a particle,...
* `suggest`: Propose (and optionally resolve)...
* `dedup`: Merge identical-content duplicate beliefs...
* `unmerge`: Revert an exact-duplicate auto-merge,...

### `particles links add`

Create a typed relation between two particles.

**Usage**:

```console
$ particles links add [OPTIONS] PARTICLE_A PARTICLE_B
```

**Arguments**:

* `PARTICLE_A`: Particle A ID (prefix OK — ≥ 8 chars)  [required]
* `PARTICLE_B`: Particle B ID (prefix OK — ≥ 8 chars)  [required]

**Options**:

* `--type TEXT`: Relation type: co-evidential, part-of, or sequence-in. part-of / sequence-in are directional (A → B).  [default: co-evidential]
* `--confidence FLOAT RANGE`: Link confidence in [0, 1]. Defaults to 1.0 for manual operator links.  [default: 1.0; 0.0<=x<=1.0]
* `--help`: Show this message and exit.

### `particles links remove`

Remove a typed relation between two particles.

**Usage**:

```console
$ particles links remove [OPTIONS] PARTICLE_A PARTICLE_B
```

**Arguments**:

* `PARTICLE_A`: Particle A ID (prefix OK — ≥ 8 chars)  [required]
* `PARTICLE_B`: Particle B ID (prefix OK — ≥ 8 chars)  [required]

**Options**:

* `--type TEXT`: Relation type to remove.  [default: co-evidential]
* `--help`: Show this message and exit.

### `particles links list`

List all relations incident to a particle, and its full co-evidential group.

**Usage**:

```console
$ particles links list [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: Particle ID (prefix OK — ≥ 8 chars)  [required]

**Options**:

* `--kind TEXT`: Filter to one relation kind (e.g. co-evidential, part-of, endorses). Case-insensitive; hyphens and underscores both accepted.
* `--help`: Show this message and exit.

### `particles links suggest`

Propose (and optionally resolve) co-evidential candidate links.

**Usage**:

```console
$ particles links suggest [OPTIONS]
```

**Options**:

* `--subject TEXT`: Restrict to one Subject (ID or canonical name).
* `--all`: Scan every Subject. Mutually exclusive with --subject.
* `--threshold FLOAT RANGE`: Cosine-similarity floor (default: links_suggest.candidate_threshold).  [0.0<=x<=1.0]
* `--llm-judge`: Send each Subject's candidate cluster to the LLM for per-pair verdicts.
* `--apply`: Implies --llm-judge; auto-link PARAPHRASE pairs (needs --yes past the cap).
* `--yes`: Confirm --apply when it would link more than apply_confirm_threshold pairs.
* `--output-format TEXT`: Output format: markdown or json  [default: markdown]
* `--help`: Show this message and exit.

### `particles links dedup`

Merge identical-content duplicate beliefs into one survivor.

Exact content equality only — the same normalized key extract-time
suppression uses (whitespace and trailing punctuation absorbed, wording and
case preserved) — so no similarity threshold and no LLM call.
Redundant copies are linked CO_EVIDENTIAL to the survivor and superseded;
nothing is ever deleted and the survivor is never mutated.

**Usage**:

```console
$ particles links dedup [OPTIONS]
```

**Options**:

* `--subject TEXT`: Restrict to one Subject (ID or canonical name). Default: whole store.
* `--apply`: Merge the groups. Requires links_suggest.auto_merge.enabled in config.yaml. Without this flag the run is a read-only census.
* `--output-format TEXT`: Output format: markdown or json  [default: markdown]
* `--limit INTEGER`: Groups listed in markdown output (counts are always complete).  [default: 20]
* `--help`: Show this message and exit.

### `particles links unmerge`

Revert an exact-duplicate auto-merge, restoring the superseded copies.

The exact inverse of `links dedup --apply`: the retained copies return to
ACTIVE keeping their ids, the merge's own CO_EVIDENTIAL links are dropped,
and the survivor is never touched. Copies that moved on since the merge are
skipped and named rather than restored.

**Usage**:

```console
$ particles links unmerge [OPTIONS] [EVENT_ID]
```

**Arguments**:

* `[EVENT_ID]`: The DUPLICATES_MERGED event to revert (from `particles events list`).

**Options**:

* `--run TEXT`: Revert every merge stamped with this run id instead of one event.
* `--since [%Y-%m-%d|%Y-%m-%dT%H:%M:%S]`: Revert every merge at or after this instant. For merges written before run ids existed.
* `--until [%Y-%m-%d|%Y-%m-%dT%H:%M:%S]`: Exclusive upper bound for --since.
* `--dry-run`: Show the plan without writing
* `-y, --yes`: Skip the confirmation prompt
* `--output-format TEXT`: Output format: markdown or json  [default: markdown]
* `--limit INTEGER`: Groups listed in markdown output (counts are always complete).  [default: 20]
* `--help`: Show this message and exit.

## `particles mcp`

Model Context Protocol server (read-only).

**Usage**:

```console
$ particles mcp [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `serve`: Run the read-only Particles MCP server...
* `tools`: Print the registered MCP tool surface...
* `resources`: Print the registered MCP resource surface...

### `particles mcp serve`

Run the read-only Particles MCP server over stdio.

Typical install path on the operator's machine::

    claude mcp add particles -- uv run particles mcp serve

**Usage**:

```console
$ particles mcp serve [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `particles mcp tools`

Print the registered MCP tool surface (name, description, input schema).

Used to verify the contract without spawning an MCP client. The JSON
output is what ``tests/mcp/tool-schema.json`` should match — drift
here means an ``operations/`` signature changed and the MCP surface
needs review.

**Usage**:

```console
$ particles mcp tools [OPTIONS]
```

**Options**:

* `--format TEXT`: Output format: "json" (default) or "text".  [default: json]
* `--help`: Show this message and exit.

### `particles mcp resources`

Print the registered MCP resource surface (digest).

The sibling of ``particles mcp tools`` for the *resources* primitive: the
``particles://digest/{store}`` template plus any concrete per-store digests
listed for the write-enabled / opted-in memory stores. The JSON output is
what ``tests/mcp/resource-schema.json`` should match — drift here means the
resource contract MCP clients see has changed.

**Usage**:

```console
$ particles mcp resources [OPTIONS]
```

**Options**:

* `--format TEXT`: Output format: "json" (default) or "text".  [default: json]
* `--help`: Show this message and exit.

## `particles memory`

Agent-memory maintenance.

**Usage**:

```console
$ particles memory [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `rebuild-utility`: Re-mine harvested session transcripts into...
* `useful`: Mark a belief useful — the explicit...
* `sweep-rank-lift`: Sweep the usefulness rank-lift and report...
* `consolidate`: Run the scheduled consolidation cycle (ADR...
* `serve`: Serve the reference memory-server...
* `tools`: Print the façade's tool surface — name,...
* `sweep-owner-lift`: Sweep the owner-relevance rank-lift and...

### `particles memory rebuild-utility`

Re-mine harvested session transcripts into fresh utility evidence.

**Usage**:

```console
$ particles memory rebuild-utility [OPTIONS]
```

**Options**:

* `--store TEXT`: Store handle to rebuild utility evidence for.  [default: default]
* `-v, --verbose`: Raise diagnostics to INFO and un-aggregate per-item detail.
* `--debug`: Full DEBUG diagnostics and tracebacks (implies --verbose).
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

### `particles memory useful`

Mark a belief useful — the explicit utility gesture.

Use this for the beliefs the transcript miner cannot see: prohibitions
("never do X") and design stances, which you comply with by *not* acting and
which therefore leave no tool-call trace. One press is worth
`utility.explicit_weight` mined events, because the miner fires once per
session while you fire once — and it is capped at one credit per belief per
day, so pressing twice is recorded but not double-counted.

This lifts the belief in the projection and digest **ranking only**. It never
touches the stored confidence, never claims the belief is *true*, and can
only promote — for "still true", the gesture is
`particles curate apply affirm`.

**Usage**:

```console
$ particles memory useful [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: The belief that earned its place. Full UUID, a unique id prefix, or the `p-xxxxxxxx` display form the digest and `particle show` print.  [required]

**Options**:

* `--reason TEXT`: Optional note recorded on the operator event.
* `--store TEXT`: Store handle.  [default: default]
* `-v, --verbose`: Raise diagnostics to INFO and un-aggregate per-item detail.
* `--debug`: Full DEBUG diagnostics and tracebacks (implies --verbose).
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

### `particles memory sweep-rank-lift`

Sweep the usefulness rank-lift and report its admissible band.

Read-only: no writes, no LLM calls, no embeddings. `λ`
(`utility.default.rank_lift`) is deliberately **not** auto-fitted
measured every candidate closed form and found none defensible, because no
label says which belief *should* occupy a head slot. This is the harness
that makes setting it by hand a single command instead of a research
project: name the beliefs that ought to reach the head with `--target`, and
the sweep reports where they land, how many head slots hold distinct
content, the resulting band per surface, and whether the configured value
is inside it.

**Usage**:

```console
$ particles memory sweep-rank-lift [OPTIONS]
```

**Options**:

* `--store TEXT`: Store handle to sweep.  [default: default]
* `--target TEXT`: Particle id of a belief you assert ought to reach the head; repeatable. Full UUID, a unique id prefix, or the `p-xxxxxxxx` digest display form; resolved against ACTIVE beliefs, and an id that matches none (or more than one) is an error rather than a silent rank-0. This is the judgment a fit cannot supply — without any, only head diversity constrains the band.
* `--head INTEGER`: A rendered head size N to evaluate; repeatable. Defaults to the digest's mcp.recall.digest_max_beliefs. Pass every N you actually render — the band is a property of the surface, not the store.
* `--grid-max FLOAT`: Largest lambda to evaluate.  [default: 0.12]
* `--grid-steps INTEGER`: Non-zero grid points; band edges resolve to one step.  [default: 120]
* `--distinct-ratio FLOAT`: Fraction of head slots that must hold distinct content. Not 1.0 — that is unsatisfiable at large N on any store with over-extraction.  [default: 0.95]
* `--format TEXT`: Output format: markdown (default) or json.  [default: markdown]
* `-v, --verbose`: Raise diagnostics to INFO and un-aggregate per-item detail.
* `--debug`: Full DEBUG diagnostics and tracebacks (implies --verbose).
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

### `particles memory consolidate`

Run the scheduled consolidation cycle: the memory dream cycle.

Exit codes (cron observability): 0 — success, including disclosed
structural-only runs and --if-due / lock skips; 1 — one or more passes
failed (run record written); 2 — the cycle could not start.

**Usage**:

```console
$ particles memory consolidate [OPTIONS]
```

**Options**:

* `--store TEXT`: Store handle to consolidate (default: the default store).  [default: default]
* `--if-due`: Exit 0 without running unless the last successful run is older than consolidation.min_interval_hours — makes over-scheduling harmless.
* `--structural-only`: Skip all LLM passes (disclosed in the report and the run record).
* `--scope TEXT`: Semantic-pass scope: 'delta' (default — particles changed since the previous run's watermark) or 'store' (the whole store, still capped).  [default: delta]
* `--output PATH`: Also write the run report as Markdown to FILE.
* `--format TEXT`: Terminal format: markdown (default) or json.  [default: markdown]
* `-v, --verbose`
* `--debug`
* `--help`: Show this message and exit.

### `particles memory serve`

Serve the reference memory-server compatibility façade over stdio.

The drop-in swap for ``@modelcontextprotocol/server-memory``: same nine
tools, same schemas, same responses, backed by a Particles store. In your
MCP client config, replace the reference server's command with::

    "memory": {"command": "uv",
               "args": ["run", "--project", "/path/to/particles",
                        "particles", "memory", "serve"]}

Args:
    store: Store handle to bind. Defaults to the default store. Writes
        additionally require the store to be listed in
        ``mcp.write.enabled_stores`` (default-deny); the write
        tools stay visible either way and refuse with an actionable
        message when it is not.

**Usage**:

```console
$ particles memory serve [OPTIONS]
```

**Options**:

* `--store TEXT`: Store handle to bind (default: the default store).
* `--help`: Show this message and exit.

### `particles memory tools`

Print the façade's tool surface — name, title, schemas, annotations.

The debugging sibling of ``particles mcp tools``, and the generator for
``tests/mcp/memory-tool-schema.json``. That golden is what turns a parity
regression into a failed build instead of a broken agent.

Args:
    output_format: ``json`` (default) or ``text`` for a one-line summary.

**Usage**:

```console
$ particles memory tools [OPTIONS]
```

**Options**:

* `--format TEXT`: Output format: "json" (default) or "text".  [default: json]
* `--help`: Show this message and exit.

### `particles memory sweep-owner-lift`

Sweep the owner-relevance rank-lift and report its band.

Read-only: no writes, no LLM calls, no embeddings. `ω`
(`owner_lens.rank_lift`) is store-specific and deliberately ships `0.0`
(inert) — this is the harness for choosing it. Unlike the utility lift, `ω`
multiplies a flat 0/1 indicator, so it acts as a *threshold* over the whole
viewer cohort: below it nothing moves, above it every belief about the
viewer arrives in the head at once. The report is therefore keyed on the
cohort's **share of the head**, and the utility `λ` in force is held fixed
so the non-regression criterion is measured against the head utility has
already shaped.

**Usage**:

```console
$ particles memory sweep-owner-lift [OPTIONS]
```

**Options**:

* `--store TEXT`: Store handle to sweep.  [default: default]
* `--target TEXT`: Particle id of a belief that must STAY in the head; repeatable. Pass the beliefs your utility lift was calibrated to surface — the third criterion is that adding aboutness does not push them out. Same id forms as `sweep-rank-lift`.
* `--head INTEGER`: A rendered head size N to evaluate; repeatable. Defaults to the digest's mcp.recall.digest_max_beliefs.
* `--grid-max FLOAT`: Largest omega to evaluate.  [default: 0.12]
* `--grid-steps INTEGER`: Non-zero grid points; band edges resolve to one step.  [default: 120]
* `--min-owner INTEGER`: Viewer beliefs the head must hold for an omega to pass (criterion 1).  [default: 1]
* `--max-owner-share FLOAT`: Largest fraction of the head the viewer cohort may occupy (criterion 2). This is the quantity to calibrate against: A(p) is a flat step, so omega behaves as a threshold over the whole cohort rather than a graded lift.  [default: 0.5]
* `--format TEXT`: Output format: markdown (default) or json.  [default: markdown]
* `-v, --verbose`: Raise diagnostics to INFO and un-aggregate per-item detail.
* `--debug`: Full DEBUG diagnostics and tracebacks (implies --verbose).
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

## `particles particle`

Inspect individual extracted particles.

**Usage**:

```console
$ particles particle [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `show`: Show one particle's content, status,...
* `narrative`: Show a NARRATIVE particle's constituents...
* `tag`: Add taxonomy tags to a particle.
* `untag`: Remove taxonomy tags from a particle (ADR...
* `retract`: Retract one belief under operator...
* `search`: List particles sharing a context...

### `particles particle show`

Show one particle's content, status, confidence, subjects, and source URL.

**Usage**:

```console
$ particles particle show [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: Particle ID (prefix OK — first 8 chars)  [required]

**Options**:

* `--help`: Show this message and exit.

### `particles particle narrative`

Show a NARRATIVE particle's constituents in SEQUENCE_IN order.

With ``--synthesize``, render the narrative as one cited prose article by
traversing its SEQUENCE_IN chain — the same synthesis engine the
wiki/Obsidian exporters use, here scoped to a single narrative.

**Usage**:

```console
$ particles particle narrative [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: NARRATIVE particle ID (prefix OK — ≥ 8 chars)  [required]

**Options**:

* `--synthesize`: Render the narrative as one cited prose article instead of listing its constituents.
* `--help`: Show this message and exit.

### `particles particle tag`

Add taxonomy tags to a particle.

**Usage**:

```console
$ particles particle tag [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: Particle ID (prefix OK)  [required]

**Options**:

* `--tag TEXT`: Tag path to add (repeatable, e.g. coins/germany)  [required]
* `--force`: Allow tags that aren't in any active taxonomy
* `--supersede`: Reserved for the immutable-revision audit trail (not yet implemented)
* `--help`: Show this message and exit.

### `particles particle untag`

Remove taxonomy tags from a particle.

**Usage**:

```console
$ particles particle untag [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: Particle ID (prefix OK)  [required]

**Options**:

* `--tag TEXT`: Tag path to remove (repeatable)  [required]
* `--supersede`: Reserved for the immutable-revision audit trail (not yet implemented)
* `--help`: Show this message and exit.

### `particles particle retract`

Retract one belief under operator authority.

The narrow escape hatch beside the cross-asserter guardrail:
``corpus retract`` retires *every* live particle from a source, and flipping
``mcp.write.allow_cross_asserter`` would widen what every future agent
session may mutate in order to fix one row. This retires exactly one.

ACTIVE → RETRACTED with reason ``EXPLICIT_RETRACTION``, routed through
``update_particle_status`` so the ``retired_at`` stamp and the ``PARTICLE_RETRACTED`` event (carrying ``--reason``) are both
written. An operator-asserted (HUMAN_REVIEW) belief is still not retractable
this way — revising one is Review's job. Run ``particles lint`` afterwards to
cascade ``PROVENANCE_STALE`` to anything that depended on it.

**Usage**:

```console
$ particles particle retract [OPTIONS] PARTICLE_ID
```

**Arguments**:

* `PARTICLE_ID`: Particle ID (prefix OK — first 8 chars)  [required]

**Options**:

* `--reason TEXT`: Why this belief is being retired; recorded in the event log  [required]
* `--dry-run`: Show the plan without writing
* `-y, --yes`: Skip the confirmation prompt
* `--help`: Show this message and exit.

### `particles particle search`

List particles sharing a context fingerprint.

**Usage**:

```console
$ particles particle search [OPTIONS]
```

**Options**:

* `--fingerprint TEXT`: Context fingerprint (full SHA-256 hex or prefix ≥ 8 chars)  [required]
* `--limit INTEGER`: Maximum particles to list  [default: 50]
* `--help`: Show this message and exit.

## `particles rules`

Operating-rule source documents tracked by this store.

**Usage**:

```console
$ particles rules [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--store TEXT`: Store handle to report on / write to.  [default: default]
* `-v, --verbose`: Raise diagnostics to INFO and un-aggregate per-item detail.
* `--debug`: Full DEBUG diagnostics and tracebacks (implies --verbose).
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

**Commands**:

* `sync`: Deposit the rule-source set as MUTABLE +...

### `particles rules sync`

Deposit the rule-source set as MUTABLE + LAZY corpus entries.

**Usage**:

```console
$ particles rules sync [OPTIONS] [PATHS]...
```

**Arguments**:

* `[PATHS]...`: Files or directories to register, overriding rule_sources.paths for this run. Omit to use the configured (or discovered) set.

**Options**:

* `--store TEXT`: Store handle to report on / write to.  [default: default]
* `--dry-run`: Resolve and print the set without depositing anything.
* `--restamp-only`: Skip the deposit half; only re-apply the scope exemption to particles already extracted from the tracked set.
* `-v, --verbose`: Raise diagnostics to INFO and un-aggregate per-item detail.
* `--debug`: Full DEBUG diagnostics and tracebacks (implies --verbose).
* `-q, --quiet`: Narration off: suppress progress and non-error diagnostics.
* `--progress / --no-progress`: Liveness on stderr. Default: auto (on when stderr is a terminal).
* `--help`: Show this message and exit.

## `particles skills`

Install the agent-onboarding skill files shipped with the SDK.

**Usage**:

```console
$ particles skills [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `install`: Install (or remove) the shipped...
* `list`: List the skill files this SDK ships, with...

### `particles skills install`

Install (or remove) the shipped agent-onboarding skill files.

**Usage**:

```console
$ particles skills install [OPTIONS]
```

**Options**:

* `--dir PATH`: Skills directory to install into. Default: ~/.claude/skills (or ./.claude/skills with --project). Files land in a 'particles' subdirectory of it.
* `--project`: Install into ./.claude/skills instead of the user-level dir.
* `--remove`: Delete exactly the Particles-owned skills subdirectory.
* `--dry-run`: Print what would be written without writing it.
* `--help`: Show this message and exit.

### `particles skills list`

List the skill files this SDK ships, with their first heading.

**Usage**:

```console
$ particles skills list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `particles synthesis-cache`

Inspect and prune the shared article-synthesis cache.

**Usage**:

```console
$ particles synthesis-cache [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List every cached article (subject, hash,...
* `show`: Print the cached article body(ies) +...
* `vacuum`: Delete unreachable rows: stale prompt...
* `evict`: Evict every cached article for one subject.

### `particles synthesis-cache list`

List every cached article (subject, hash, prompt version, age, size).

**Usage**:

```console
$ particles synthesis-cache list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `particles synthesis-cache show`

Print the cached article body(ies) + metadata for one subject.

**Usage**:

```console
$ particles synthesis-cache show [OPTIONS] SUBJECT_ID
```

**Arguments**:

* `SUBJECT_ID`: Subject ID (prefix OK)  [required]

**Options**:

* `--help`: Show this message and exit.

### `particles synthesis-cache vacuum`

Delete unreachable rows: stale prompt versions + orphaned subjects.

**Usage**:

```console
$ particles synthesis-cache vacuum [OPTIONS]
```

**Options**:

* `--dry-run`: Report what would be removed without deleting
* `--help`: Show this message and exit.

### `particles synthesis-cache evict`

Evict every cached article for one subject.

**Usage**:

```console
$ particles synthesis-cache evict [OPTIONS] SUBJECT_ID
```

**Arguments**:

* `SUBJECT_ID`: Subject ID (prefix OK)  [required]

**Options**:

* `-y, --yes`: Skip the confirmation prompt
* `--help`: Show this message and exit.

## `particles trust`

Manage source trust rules.

**Usage**:

```console
$ particles trust [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List all trust rules (domain baselines and...
* `set`: Add or update a trust rule.
* `show`: Show the resolved trust score for a URI.
* `statement-set`: Write an OPERATOR_DIRECT...
* `set-entry`: Set a per-entry trust override...
* `cascade`: Re-run cascade for all OPERATOR_DIRECT...
* `lens`: Shareable trust-policy lenses.

### `particles trust list`

List all trust rules (domain baselines and URL-pattern modifiers).

**Usage**:

```console
$ particles trust list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `particles trust set`

Add or update a trust rule.

**Usage**:

```console
$ particles trust set [OPTIONS] PATTERN SCORE
```

**Arguments**:

* `PATTERN`: Domain (e.g. en.wikipedia.org) or URL regex pattern  [required]
* `SCORE`: Score [0.0-1.0] for domain rows; modifier delta for --modifier  [required]

**Options**:

* `--modifier`: Treat score as a modifier delta, not a base score
* `--rationale TEXT`: Human-readable rationale
* `--help`: Show this message and exit.

### `particles trust show`

Show the resolved trust score for a URI.

**Usage**:

```console
$ particles trust show [OPTIONS] URI
```

**Arguments**:

* `URI`: URI to resolve trust score for  [required]

**Options**:

* `--help`: Show this message and exit.

### `particles trust statement-set`

Write an OPERATOR_DIRECT SourceTrustStatement and trigger cascade.

**Usage**:

```console
$ particles trust statement-set [OPTIONS] DOMAIN SOURCE_REF_TYPE SOURCE_REF_VALUE TRUST_RANK
```

**Arguments**:

* `DOMAIN`: Domain label (e.g. numismatics)  [required]
* `SOURCE_REF_TYPE`: Reference type: CORPUS_ENTRY | SOURCE_TYPE | AUTHOR  [required]
* `SOURCE_REF_VALUE`: Reference value (entry_id, source_type string, or author identifier)  [required]
* `TRUST_RANK`: Trust rank [0.0–1.0]  [required]

**Options**:

* `--basis TEXT`: Human-readable rationale
* `--help`: Show this message and exit.

### `particles trust set-entry`

Set a per-entry trust override (CORPUS_ENTRY scope).

Convenience over ``trust statement-set CORPUS_ENTRY`` that validates the
entry exists and infers the domain from its ``source_type`` so the override
is consulted by the §6.6 conflict cascade for that domain. Pass ``--domain``
to override the inferred value (required when the source_type has no MUST
applicability clause, e.g. WEB_PAGE / PDF).

**Usage**:

```console
$ particles trust set-entry [OPTIONS] ENTRY_ID TRUST_RANK
```

**Arguments**:

* `ENTRY_ID`: Corpus entry_id to override trust for  [required]
* `TRUST_RANK`: Trust rank [0.0–1.0]  [required]

**Options**:

* `--domain TEXT`: Domain label the override applies to (default: inferred from source_type)
* `--basis TEXT`: Human-readable rationale
* `--help`: Show this message and exit.

### `particles trust cascade`

Re-run cascade for all OPERATOR_DIRECT SourceTrustStatements.

**Usage**:

```console
$ particles trust cascade [OPTIONS]
```

**Options**:

* `--domain TEXT`: Scope cascade to this domain label
* `--dry-run`: Show what would be resolved without writing
* `--help`: Show this message and exit.

### `particles trust lens`

Shareable trust-policy lenses. Publish via `particles deposit <lens>.json`.

**Usage**:

```console
$ particles trust lens [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List materialised lenses and their...
* `show`: Show a lens's full policy entries.
* `adopt`: Adopt a lens: its policy composes into...
* `unadopt`: Remove a lens adoption.

#### `particles trust lens list`

List materialised lenses and their adoption state.

**Usage**:

```console
$ particles trust lens list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

#### `particles trust lens show`

Show a lens's full policy entries.

**Usage**:

```console
$ particles trust lens show [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Lens name (see `particles trust lens list`)  [required]

**Options**:

* `--help`: Show this message and exit.

#### `particles trust lens adopt`

Adopt a lens: its policy composes into this store's trust at query time.

**Usage**:

```console
$ particles trust lens adopt [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Lens name to adopt  [required]

**Options**:

* `--help`: Show this message and exit.

#### `particles trust lens unadopt`

Remove a lens adoption.

**Usage**:

```console
$ particles trust lens unadopt [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Lens name to unadopt  [required]

**Options**:

* `--help`: Show this message and exit.
