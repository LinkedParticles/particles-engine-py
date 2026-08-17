# Lint and review

The lint operation is the SDK's hygiene tool. Run it routinely; it
surfaces problems you'd otherwise discover in queries.

!!! warning "Migration — `lint` is read-only by default (0.45.0)"

    As of 0.45.0, `particles lint` no longer applies status transitions
    by default — it reports and leaves the store untouched. The
    "Status transition (with `--fix`)" column below fires **only** when
    you pass `--fix`. Any cron job or monitoring script that relied on
    the old implicit auto-fix must now add `--fix` explicitly to keep
    transitioning STALENESS / RETRACTION_CASCADE / CORPUS_LINK_INTEGRITY
    particles. The same flip applies to `POST /lint` (`fix` now defaults
    to `false`).

## What lint catches

The headline findings:

| Finding | Cause | Status transition (with `--fix`) |
|---|---|---|
| `STALENESS` | A particle's `valid_until` has passed | `PROVENANCE_STALE` (reason `VALIDITY_EXPIRED`); next reindex re-extracts |
| `RETRACTION_CASCADE` | A particle's provenance chain includes a `RETRACTED` / `SUPERSEDED` particle | `PROVENANCE_STALE` (reason `RETRACTED_DEPENDENCY`) |
| `CORPUS_LINK_INTEGRITY` | A particle references a snapshot that no longer exists | `PROVENANCE_STALE` (reason `CORPUS_ENTRY_MISSING`) |
| `CONTRADICTION` (with `--semantic`) | Two ACTIVE truth-apt particles semantically contradict (LLM-judged), including claims from **different** sources. Candidate pairs are gated by embedding similarity (`lint.contradiction_candidate_threshold`, default 0.6) so the store-wide check does not pay an O(n²) LLM cost. | Report-only — resolve via `particles review`. Lint never creates the `INCONSISTENCY` wrapper itself; the §6.6 extraction-time ladder does. |
| `NO_SUBJECT` | An ACTIVE CLAIM particle has zero subjects (§6.7 says it SHOULD have ≥ 1) — e.g. an import or extraction that could not resolve any subject. Zero-subject claims land in the store rather than being rejected, but are unreachable by subject-filtered query. The §9 populations that legitimately have none are excluded: DOCUMENT_META claims, non-asserted (DECLINED / HYPOTHETICAL) claims, and claims marked `extraction:subject_scope = SELF` — journal claims about the author, whose subject the privacy gate withholds. Same predicate as the conformance `subject_ids` floor | Surfaced for manual decision (re-extract, link a subject, or retract) |
| `RECENCY_DECAY` | An ACTIVE particle whose `effective_confidence` is materially discounted by **content age** alone — its source's `recency_factor` has fallen so that `1 - recency_factor ≥ lint.recency_decay_threshold` (default 0.5). Sources with no decay config or no known publication date never fire. | **Report-only WARNING** — never flips status (age decay is a recoverable discount, not a provenance break). Re-fetch / reindex if a fresher source version exists. |

Beyond these, lint reports coverage and quality diagnostics
(`ORPHAN`, `PHANTOM_SUBJECT`, `LOW_COVERAGE_SUBJECT`,
`CONFIDENCE_DECAY`, `GRANULARITY_VIOLATION_CANDIDATE`,
`PENDING_EXTRACTION`, `SCHEMA_VERSION_MISMATCH`,
`WIKIDATA_LINK_MISMATCH`, `BARE_PROPERTIES_KEY`, …) — all surfaced for manual decision; use
`--verbose --category <type>` to inspect one category in full.

```bash
uv run particles lint                # read-only: structural report, mutates nothing
uv run particles lint --semantic     # adds LLM contradiction check (costs tokens)
uv run particles lint --fix          # apply auto-fixable status transitions
```

## The review workflow

An `INCONSISTENCY` particle is created when the §6.6 extraction-time
ladder finds a new candidate conflicting with an existing claim it
cannot out-rank on trust. The losing candidate is persisted alongside
it, **quarantined** — status `PROVENANCE_STALE` with
`status_reason = CONFLICT_PENDING`, invisible to query —
so review can recover it in full rather than from an excerpt.

```bash
uv run particles review                              # list pending conflicts
uv run particles review <particle-id> --action PREFER_A
uv run particles review --bulk BOTH_VALID --dry-run  # preview a bulk action
```

Four resolution actions:

| Action | Effect |
|---|---|
| `PREFER_A` | The existing claim wins. The challenger is demoted (quarantined claims flip their reason to `CONFLICT_RESOLVED` in place); a reviewer-derived `SourceTrustStatement` for the preferred source is written. |
| `PREFER_B` | The challenger wins. The existing claim is demoted to `PROVENANCE_STALE`; a quarantined challenger is promoted to a **new ACTIVE particle** (fresh ID, provenance preserved); the trust statement is written. |
| `BOTH_VALID` | The contradiction is apparent, not real. Both claims stay queryable with `uncertainty_nature = ALEATORY`; a quarantined challenger is recovered as a new ACTIVE particle. |
| `DEFER` | Record a reviewer note and re-queue — the only action that leaves the conflict open. |

Every non-DEFER resolution **retracts the `INCONSISTENCY` wrapper
itself** (reason `CONFLICT_RESOLVED`), so resolved conflicts leave the
queue — `particles review` lists only what is still pending. Each
resolution also writes a REVIEW audit particle and a `REVIEW_RESOLVED`
event.

The `SourceTrustStatement`s accumulated from PREFER rulings feed the
trust cascade and the query-time source-trust factor — see
[Tuning → source trust rank](tuning.md#source-trust-rank).

One §6.6 verdict never reaches review: `SUPERSEDED_BY_EXISTING` (the
candidate duplicates a strictly higher-trust existing claim) drops the
candidate at extraction time. The drop is audited — a
`CONFLICT_CANDIDATE_DROPPED` event records the candidate excerpt, the
verdict, and the winning particle ID
(see [Auditing](auditing.md)).

## Reindex

When `PROVENANCE_STALE` particles accumulate, or when an extractor
upgrade lands:

```bash
uv run particles reindex                          # re-extract stale snapshots
uv run particles reindex --extractor-id <id>      # re-extract particles from one extractor version
```

Reindex is rate-limited (default 100 extractions per minute). It
respects the chunk-hash carry-forward — particles whose
source chunks didn't change skip the LLM call.

## Quality reports

```bash
uv run particles quality
```

Prints the extraction-quality dashboard — calibration-source
distribution, corpus snapshot status, subject coverage — useful for
"is the store growing healthy or accumulating staleness?" No LLM
calls; instant read from the DB.

## How often to run lint

- After every batch deposit + extract — `lint` catches structural
  errors immediately.
- Weekly for `--semantic` — it's the LLM-expensive variant.
- Before every export — exports already run an implicit `lint
  --semantic=False` pre-pass and splice findings into the exported
  output (per-article callouts in wiki / Obsidian).

## Fixing a misjoined Subject

The Subject resolver is fast and usually right, but when two real-world
entities share a name or acronym it can silently land them on one
Subject. The classic failure: *"AAOI"* — both the Accounting and
Auditing Organization for Islamic Financial Institutions
(Wikidata `Q5326167`) and Applied Optoelectronics (Wikidata
`Q30297735`). Every particle from either gets bound to the first
Subject the resolver picked, and the wiki article ends up confidently
claiming the audit organisation has a Texas manufacturing facility.

`particles subjects split` re-binds the wrongly-attributed particles
onto a new Subject the resolver canonicalises against the available
external KBs. The verb is metadata-only — particle confidence and
content are unchanged.

```bash
# Inspect the misjoined Subject to find the particles that belong elsewhere.
uv run particles subjects show 1a2b3c4d

# Dry-run the split — confirms the new Subject the resolver would create.
uv run particles subjects split 1a2b3c4d \
    --particle pid-aaaa1111 \
    --particle pid-bbbb2222 \
    --new-name "Applied Optoelectronics" \
    --dry-run

# Apply.
uv run particles subjects split 1a2b3c4d \
    --particle pid-aaaa1111 \
    --particle pid-bbbb2222 \
    --new-name "Applied Optoelectronics"
```

If you already know the correct external identifier and want to
sidestep the resolver's search (avoiding the same wrong-match that
created the problem), pass `--new-external-id` instead:

```bash
uv run particles subjects split 1a2b3c4d \
    --particle pid-aaaa1111 --particle pid-bbbb2222 \
    --new-external-id wikidata:Q30297735
```

**What to run next:**

- `particles export <format>` — re-render Obsidian / wiki / Logseq
  with the corrected attribution. The synthesis cache
  hits on every other Subject and re-synthesises only the source +
  new Subjects.
- `particles lint` (structural) — surfaces any now-stale
  `CO_EVIDENTIAL` or `CONTRADICTS` relations that the split may
  have invalidated.
- `particles query <topic>` — sanity-check that the corrected subject
  filter returns what you expected.

`particles extract --all-pending` is **not** the next step. The
split is metadata-only; extraction had nothing to do with the
mis-binding the operator just corrected.

The source Subject is preserved with its remaining particles even
if every particle was moved off it — empty Subjects survive the
split for audit-trail reasons.
