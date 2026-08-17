# Troubleshooting

The CLI translates the three most common operational failures into
actionable messages. The remediation paths:

## "Database tables not found, or a migration is pending."

The schema's not there or the SDK and the DB disagree on
`SCHEMA_VERSION`.

```bash
uv run particles db init                  # first install
uv run alembic upgrade head               # pending migration
uv run particles db init --force          # major schema bump (preserves corpus)
```

See [Schema migration](schema-migration.md) for the `--force` path.

## "Database is locked."

SQLite serialises writers. Particles ships with WAL mode + a 30-second
busy_timeout (per ADR / `particles/db.py`), so most contention
resolves itself. If you hit this:

- A long `extract --all-pending` may have held the write lock for an
  LLM call. The 30-second timeout normally absorbs this; if it
  doesn't, retry the command.
- Concurrent CLI runs against the same DB are not supported.
  Particles is single-writer.
- A wedged process (Ctrl-C mid-extract, crashed test) can leave a
  stale `.db-wal` / `.db-shm`. Remove them with the DB unmounted.

## "SCHEMA_VERSION mismatch."

You're running an SDK binary against a DB written by a different
schema-major. The error message echoes the exception's own
`operator_message` — read it before doing anything else.
Usually the fix is `db init --force` + re-extract.

## Fetch failures

URL deposit can fail for many reasons:

- **SSRF guard** (loopback / RFC 1918 / link-local) — by design;
  `validate_fetch_url()` is called on every fetch including refetch
  (see [particles/corpus/AGENTS.md](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/corpus/AGENTS.md)).
- **403 / 401** — paywalled or auth-walled source. Particles logs
  and continues; no particle is extracted.
- **DNS / network** — the corpus entry is still created with the
  failed fetch recorded in the snapshot's `quality_notes`. A future
  `particles reindex` can retry.

For Reddit / Hacker News / Mastodon URLs, the
[follow-edges behaviour](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/corpus/AGENTS.md#follow-edges-adr-0078)
deposits the post's primary URL as a separate entry. If the follow
fails (paywall on the linked article), the primary deposit succeeds
and a warning is logged.

## Pending snapshots stuck on IN_PROGRESS

If a previous `extract` was killed mid-LLM-call, the snapshot's
`extraction_status` may be `IN_PROGRESS` rather than `PENDING` or
`FAILED`. To recover:

```bash
uv run particles db init --force
```

The `--force` flag resets every snapshot to `PENDING` (without
touching the corpus). A re-extract will pick them up.

Or, for targeted recovery, scripts in `scripts/` can flip an
individual snapshot status — but those are operator-pull, not part
of the public CLI.

## "LLM unavailable (account-level)" — extraction stopped early

```
LLM unavailable (account-level): Your credit balance is too low to access the Anthropic API.
Stopped after 0 of 68 snapshot(s); 68 still PENDING. Fix the API key or credit
balance, then re-run `particles extract --all-pending`.
```

An **account-level** failure fails every call until you fix the account: a bad
or missing key (401), no permission (403), or an exhausted credit balance (a 400
naming the credit balance or billing). Extraction stops at the first one rather
than reissuing a doomed request for every remaining snapshot — and, on a PDF,
for every page of every snapshot.

Nothing is lost. The snapshot in flight is reset `IN_PROGRESS → PENDING` and the
untried ones were never claimed, so the whole queue survives:

```bash
particles quality        # confirm the PENDING count
particles extract --all-pending   # resume once the account is fixed
```

The scheduled cycle behaves the same way — `memory consolidate`'s extract pass
stops instead of burning its per-run cap, and the run report discloses it.

This is distinct from a **per-call** failure (one malformed prompt, a transient
`429` / `5xx`), which is retried per snapshot as before and leaves that snapshot
`PENDING` without stopping the batch.

## I deposited a spec / ADR / contract and got junk "meta" particles

Normative documents (technical specs, ADRs, legal contracts, manuals)
describe their *own* structure as well as the world. An assertoric
extractor pointed at one happily produces claims like *"Section 10.4
defines the exporter"* with a subject of `§10.4` — meta-commentary about
the document, not facts about the world.

Since the general extractor `0.6.0` (a later roadmap milestone), the
extractor classifies each candidate's **scope** — `WORLD` vs
`DOCUMENT_META` — and, in the default `label` mode, keeps `DOCUMENT_META`
particles out of contradiction-checking and out of the default query
surface. They stay in the store; query with `--include-document-meta` to
see them.

```bash
# See what the classifier flagged (still present, just hidden by default):
uv run particles query "what does this document say about itself?" --include-document-meta
```

Knobs live under `extraction_scope` in `config.yaml`
(see [Tuning](tuning.md) and `config.yaml.sample`):

- `mode: label` (default) — tag and hide from the factual surface.
- `mode: suppress` — drop meta-claims at extraction (lossy).
- `mode: passthrough` — tag for inspection but apply no exclusion (use to
  evaluate the classifier before trusting it).
- `enabled: false` — turn the feature off entirely (output reverts to the
  pre-`0.6.0` behaviour).

The classification is semantic, not pattern-based, so it improves with
better extractor models. To relabel documents deposited before `0.6.0`,
run `particles reindex` — the extractor-version bump makes them eligible.

## My `AGENTS.md` rules were extracted but never show up

This is the mirror image of the section above, and it is why the exclusion
is lifted per source. On a rules document — `AGENTS.md`,
`CLAUDE.md`, a runbook — the WORLD / `DOCUMENT_META` question has no clean
answer: such a document's *subject matter* is the project's own apparatus, so
"store accessors must return Pydantic models" reads to the classifier as a
claim about the document. Measured on a 20k-particle store, 178 of one rule
file's 1,149 claims were hidden that way, most of them ordinary facts about
the codebase rather than about the document.

A corpus entry whose tags match `extraction_scope.exempt_source_tags`
(`["rule-file"]` by default — the tag `particles rules sync` writes) therefore
stamps `extraction:scope_action: source_exempt` on its flagged claims, and every surface
lets them through. The scope label is still recorded; only the hiding stops.

```bash
# Which tracked rule files are exempt:
uv run particles rules

# Re-apply the exemption to particles extracted before this shipped
# (deterministic — no LLM call, no re-extraction):
uv run particles rules sync --restamp-only
```

Two consequences worth knowing:

- **A rules document's self-referential sentences become beliefs too**
  ("This file is loaded by agents when editing …"). That is the accepted
  cost: gating the exemption on modality would have recovered 27 of those
  178 claims and left the rest hidden.
- **The exemption follows the tag, not the file.** The same document
  deposited with `particles deposit` is not exempt. Register it with
  `particles rules sync <path>` to make it one.

To switch the behaviour off, empty `extraction_scope.exempt_source_tags`
and re-run `particles rules sync --restamp-only`. Under `mode: suppress` the
exemption never applies — a `DOCUMENT_META` candidate is dropped inside the
extractor, before any source policy is consulted.

## "Blob not found for hash …" — the store's content is somewhere else

Every deposit writes the source bytes to a content-addressed blob under
`storage.blob_dir`, and the snapshot row records the hash. When that
directory does not hold the blob, extraction fails one snapshot at a time
with `Blob not found for hash …` — often long after the deposit, and often
for hundreds of entries at once.

The usual cause is a **relative `blob_dir` combined with an absolute
`DATABASE_URL`**: one database, but blobs written under whichever directory
each process happened to run in. A fix corrected the default so a relative
`blob_dir` now anchors to the store's own directory, but blobs written
*before* that fix stay where they landed.

To check whether a store is affected:

```bash
uv run particles config validate
```

The command warns when the store's rows point at content this process cannot
see, naming the resolved directory and how many of the sampled blobs were
missing. `particles hook doctor --store <handle>` reports the same thing
alongside its store-resolution checks — use that one when the symptom is a
Claude Code hook, since it answers "from *this* directory" explicitly.

Both are detection only: they never fail the command, and a first-run store
with no deposits stays silent. `storage.blob_health_sample` (default 50)
bounds how many hashes are stat'd.

If blobs are missing, the content is usually still on disk in another
`corpus_blobs/` tree — look in the directories the depositing processes ran
from (sibling git worktrees are the classic case). Moving those shards back
under the resolved `blob_dir` restores extraction; the layout is
`<blob_dir>/<first-two-hex-chars>/<full-hash>`, so shard directories merge
cleanly. Blobs written inside a worktree that has since been deleted are
**unrecoverable** — the rows survive pointing at content that no longer
exists, and the entries must be re-deposited from source or retracted.

Set `storage.blob_dir` to an absolute path to stop the problem recurring.

## Where to look when things fail

| Symptom | First check |
|---|---|
| Spec / ADR deposit produced junk "meta" particles | Expected — they're tagged `DOCUMENT_META` and hidden by default; see the section above |
| `Blob not found for hash …` during extraction | `particles config validate` — it warns when the resolved `blob_dir` does not hold the store's content; see the section above |
| Particle missing from query | `lint` for staleness findings; `particles quality` for status distribution |
| Extractor regressed | `extractor benchmark-compare` against the previous extractor version |
| Wiki articles cite the wrong subjects | `--invalidate-stale-links` after `subjects fix-labels` |
| Database performance dropping | Check `particles quality` for stale-particle accumulation; reindex |
| LLM cost spiking | `wiki export --dry-run` reports estimated tokens before paying |

When in doubt, the implementation lives in the per-package
`AGENTS.md` files (`particles/corpus/AGENTS.md`,
`particles/extraction/AGENTS.md`, `particles/exporters/AGENTS.md`)
and the relevant ADRs. The roadmap's `Status` column is the
canonical "is this shipped?" reference.
