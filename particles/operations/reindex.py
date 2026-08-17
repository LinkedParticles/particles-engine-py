"""§9.5 Reindex operation.

Re-extracts particles for a scoped set of corpus entries:
  - Entries whose last extraction used a superseded extractor version
  - Entries with extraction_status = FAILED

Default rate limit: 100 extractions per minute (operator-configurable).
After completion: run a Lint pass over reindexed entries.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    SCHEMA_VERSION,
    ExtractionStatus,
    Particle,
    ProvenanceRefType,
)
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import blob_exists
from particles.corpus.store import (
    find_entry_ids_by_prefix,
    get_entry_uri_map,
    get_latest_completed_snapshot_id,
    list_entry_snapshot_pairs_with_extraction_status,
    list_snapshots_for_entry,
)
from particles.observability import traced
from particles.operations.extract import extract_snapshot
from particles.operations.lint import run_lint
from particles.store.particle_store import (
    get_active_particles_for_entry,
    get_active_particles_with_extractor_id,
    get_active_particles_with_extractor_version,
    get_active_particles_with_provider_model,
    get_active_particles_with_stale_schema_version,
    update_particle_status,
)

log = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_PER_MINUTE = 100


class SnapshotPlan(BaseModel):
    """One planned re-extraction: a snapshot, its entry, and what it costs."""

    entry_id: str
    snapshot_id: str
    uri: str = ""
    #: ACTIVE particles anchored to this snapshot — what a live run would
    #: supersede (the same provenance filter ``_reindex_snapshot`` applies).
    particles: int = 0
    #: The snapshot's blob is absent from the blob store, so extraction is
    #: known to fail with ``FileNotFoundError`` before any LLM call is made.
    blob_missing: bool = False


#: How many per-snapshot "blob missing" lines the human rendering shows before
#: collapsing the rest into a count. A store with hundreds of missing blobs
#: flooded the terminal on the first real run; the full list stays in the JSON
#: envelope (``plan.snapshot_plans``).
BLOB_MISSING_DISPLAY_CAP = 5


class ReindexPlan(BaseModel):
    """Upfront work plan for a resolved reindex scope (computed pre-extraction)."""

    entries: int
    snapshots: int
    particles: int
    missing_blobs: int
    scope_description: str
    snapshot_plans: list[SnapshotPlan] = []

    def format_line(self) -> str:
        """The one-line human summary printed before the first LLM call."""
        line = (
            f"Reindex plan: {self.entries} entries, {self.snapshots} snapshots, "
            f"{self.particles} particles (scope: {self.scope_description})"
        )
        if self.missing_blobs:
            line += f"; {self.missing_blobs} snapshot(s) missing their blob (extraction will fail)"
        return line

    def format_missing_blob_lines(self, cap: int = BLOB_MISSING_DISPLAY_CAP) -> list[str]:
        """Human warning lines for missing blobs, capped at ``cap`` + a remainder."""
        missing = [sp for sp in self.snapshot_plans if sp.blob_missing]
        lines = [
            f"  blob missing: entry {sp.entry_id[:8]}… snapshot "
            f"{sp.snapshot_id[:8]}… — extraction will fail"
            for sp in missing[:cap]
        ]
        if len(missing) > cap:
            lines.append(f"  … and {len(missing) - cap} more (see --format json)")
        return lines


def _describe_scope(
    entry_ids: list[str] | None,
    extractor_version: str | None,
    extractor_id: str | None,
    include_failed: bool,
    provider_model: str | None,
) -> str:
    """Human-readable rendering of the requested scope, auto-discovery included.

    The auto-discovery unions (FAILED/PENDING snapshots, stale schema) apply
    whenever no entries are named — even alongside a particle-matching flag —
    and that widening is exactly what the plan line exists to surface, so it
    is spelled out rather than implied.
    """
    parts: list[str] = []
    if entry_ids:
        parts.append(f"entry-ids {','.join(entry_ids)}")
    if extractor_version:
        parts.append(f"extractor-version {extractor_version}")
    if extractor_id:
        parts.append(f"extractor-id {extractor_id}")
    if provider_model:
        parts.append(f"provider-model {provider_model}")
    if not entry_ids:
        auto = "auto: stale schema"
        if include_failed:
            auto += " + failed/pending"
        parts.append(auto)
    return "; ".join(parts)


async def _build_plan(
    session: AsyncSession,
    scope: list[tuple[str, str]],
    scope_description: str,
) -> ReindexPlan:
    """Per-snapshot counts + blob presence for a resolved scope (pure reads)."""
    by_entry: dict[str, list[str]] = {}
    for entry_id, snapshot_id in sorted(scope):
        by_entry.setdefault(entry_id, []).append(snapshot_id)

    uri_map = await get_entry_uri_map(session, set(by_entry)) if by_entry else {}

    snapshot_plans: list[SnapshotPlan] = []
    for entry_id, snapshot_ids in by_entry.items():
        active = await get_active_particles_for_entry(session, entry_id)
        content_hashes = {
            s.snapshot_id: s.content_hash for s in await list_snapshots_for_entry(session, entry_id)
        }
        for snapshot_id in snapshot_ids:
            count = sum(
                1 for p in active if any(ref.snapshot_id == snapshot_id for ref in p.provenance)
            )
            content_hash = content_hashes.get(snapshot_id)
            snapshot_plans.append(
                SnapshotPlan(
                    entry_id=entry_id,
                    snapshot_id=snapshot_id,
                    uri=uri_map.get(entry_id) or "",
                    particles=count,
                    blob_missing=content_hash is None or not blob_exists(content_hash),
                )
            )

    return ReindexPlan(
        entries=len(by_entry),
        snapshots=len(snapshot_plans),
        particles=sum(sp.particles for sp in snapshot_plans),
        missing_blobs=sum(1 for sp in snapshot_plans if sp.blob_missing),
        scope_description=scope_description,
        snapshot_plans=snapshot_plans,
    )


@traced("reindex")
async def reindex(
    session: AsyncSession,
    entry_ids: list[str] | None = None,
    extractor_version: str | None = None,
    extractor_id: str | None = None,
    include_failed: bool = True,
    provider_model: str | None = None,
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    run_post_lint: bool = True,
    progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
    on_plan: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Reindex corpus entries.

    The particle-selecting scopes — ``extractor_version``, ``extractor_id``,
    ``provider_model`` — union with each other and **intersect** with
    ``entry_ids`` when both are supplied. See ``_identify_scope``
    for why the combination narrows rather than erroring.

    Args:
        entry_ids: explicit list of entries to reindex; if None, auto-discover scope.
        extractor_version: superseded extractor version to replace (filters by extractor_ref).
        extractor_id: extractor name to re-extract regardless of version. Useful when
            a shared upstream (e.g. a prompt change) affects multiple extractors that
            delegate to it but didn't bump their own version.
        include_failed: also reindex entries with FAILED snapshots. Applies to
            auto-discovery only — named entries resolve to their latest
            *COMPLETE* snapshot, so an explicit scope never contains a FAILED
            one to include or exclude.
        provider_model: re-extract particles stamped with this
            ``"<provider>:<model>"`` pairing — the handle for
            undoing an uncalibrated provider swap. Matched exactly; particles
            with no stamp (deterministic extractors, direct assertions, or
            anything minted before the stamp existed) never match.
        rate_limit_per_minute: max extraction jobs per minute.
        run_post_lint: run a Lint pass after reindex completes.
        progress: optional callback for human-readable progress lines. The CLI
            wires this to ``typer.echo`` when ``--verbose`` is set so the
            operator can see that a long-running reindex isn't stuck.
        dry_run: compute and report the work plan, then return without
            extracting — zero LLM calls, zero writes (mirrors the Notion
            exporter's dry-run discipline). The returned summary
            carries the full plan including per-snapshot counts.
        on_plan: optional callback for the upfront work-plan lines (the scope
            summary + any missing-blob warnings), emitted before the first
            extraction. Separate from ``progress`` because the plan is meant
            to print unconditionally while per-entry progress stays opt-in.
        on_status: optional callback fired after **each** snapshot completes
            with a compact position line — ``snapshot 12/89 (entry
            0a8fb1a9…) — 3 failed`` — so a long run's liveness display can
            show how far along it is, not just elapsed time. Distinct from
            ``progress`` (opt-in, one full line per item, appended): the
            status is a single replaceable line the CLI feeds to the
            heartbeat.

    Returns a summary dict with counts and any errors.
    """
    # refuse to reindex into a store with mismatched-schema
    # particles. Reindex writes new ACTIVE particles and supersedes old
    # ones — both operations assume the surrounding store is current.
    from particles.operations.version_guard import assert_store_schema_current

    await assert_store_schema_current(session)

    scope = await _identify_scope(
        session,
        entry_ids,
        extractor_version,
        extractor_id,
        include_failed,
        provider_model,
        progress=progress,
    )
    # The upfront work plan (2026-08-02 incident): report what the resolved
    # scope will cost — entries, snapshots, supersede-able particles, known
    # missing blobs — BEFORE the first LLM call is spent.
    plan = await _build_plan(
        session,
        scope,
        _describe_scope(entry_ids, extractor_version, extractor_id, include_failed, provider_model),
    )
    plan_line = plan.format_line()
    log.info("%s", plan_line)
    emit = on_plan or progress
    if emit is not None:
        emit(plan_line)
        for line in plan.format_missing_blob_lines():
            emit(line)

    if dry_run:
        return {
            "dry_run": True,
            "scope": len(scope),
            "succeeded": 0,
            "failed": 0,
            "failed_entries": [],
            "lint_summary": {},
            "plan": plan.model_dump(),
        }

    delay = 60.0 / rate_limit_per_minute if rate_limit_per_minute > 0 else 0.0
    succeeded: list[str] = []
    failed: list[str] = []
    total = len(scope)

    for i, (entry_id, snapshot_id) in enumerate(scope, start=1):
        if progress is not None:
            uri = await _lookup_entry_uri(session, entry_id)
            progress(f"[{i}/{total}] reindexing {entry_id[:8]}… snap {snapshot_id[:8]}… {uri}")
        try:
            await _reindex_snapshot(session, entry_id, snapshot_id, extractor_version)
            succeeded.append(entry_id)
        except Exception as exc:
            log.error("Reindex failed for entry %s snapshot %s: %s", entry_id, snapshot_id, exc)
            if progress is not None:
                progress(f"[{i}/{total}] FAILED: {exc}")
            failed.append(entry_id)
        if on_status is not None:
            status = f"snapshot {i}/{total} (entry {entry_id[:8]}…)"
            if failed:
                status += f" — {len(failed)} failed"
            on_status(status)
        if delay > 0:
            await asyncio.sleep(delay)

    lint_summary: dict[str, int] = {}
    if run_post_lint and succeeded:
        lint_report = await run_lint(session, fix=True, semantic=False)
        lint_summary = lint_report.summary
        log.info("Post-reindex lint: %s", lint_summary)

    return {
        "dry_run": False,
        "scope": len(scope),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "failed_entries": failed,
        "lint_summary": lint_summary,
        # Full plan, per-snapshot detail included: the human rendering caps the
        # missing-blob list and points at `--format json`, so the envelope must
        # actually carry the complete list. (The old counts-only exclusion
        # existed because the CLI dumped this envelope raw on every run.)
        "plan": plan.model_dump(),
    }


async def _lookup_entry_uri(session: AsyncSession, entry_id: str) -> str:
    """Return the corpus entry's uri_r for progress display, or ``""`` on failure."""
    from particles.corpus.store import CorpusEntryRow

    row = await session.get(CorpusEntryRow, entry_id)
    if row is None or not row.uri_r:
        return ""
    return str(row.uri_r)


def _source_pairs(particles: list[Particle]) -> list[tuple[str, str]]:
    """``(corpus_entry_id, snapshot_id)`` for every SOURCE ref on these particles.

    The scope unit is the *snapshot*, so every selector reduces its matching
    particles to the snapshots that produced them before scopes are combined.
    """
    return [
        (ref.corpus_entry_id, ref.snapshot_id)
        for p in particles
        for ref in p.provenance
        if ref.type == ProvenanceRefType.SOURCE and ref.snapshot_id
    ]


async def _particle_selector_pairs(
    session: AsyncSession,
    extractor_version: str | None,
    extractor_id: str | None,
    provider_model: str | None,
) -> set[tuple[str, str]] | None:
    """Union of the snapshot pairs selected by the particle-matching flags.

    Returns ``None`` — distinct from an empty set — when the caller passed no
    particle-matching flag at all, so callers can tell "no filter requested"
    from "filter requested, matched nothing".
    """
    if not (extractor_version or extractor_id or provider_model):
        return None

    pairs: set[tuple[str, str]] = set()
    if extractor_version:
        pairs.update(
            _source_pairs(
                await get_active_particles_with_extractor_version(session, extractor_version)
            )
        )
    if extractor_id:
        pairs.update(
            _source_pairs(await get_active_particles_with_extractor_id(session, extractor_id))
        )
    # Particles produced by a specific "<provider>:<model>" pairing.
    # Exact equality on the stamped column, never a substring match — pairings
    # nest, so a substring scope would sweep in the sibling model this exists
    # to separate. Note the scope unit is the *snapshot*: a snapshot whose
    # particles are model-mixed is re-extracted whole, which is the intended
    # behaviour for undoing a provider trial but is not a particle-level
    # surgical tool.
    if provider_model:
        pairs.update(
            _source_pairs(await get_active_particles_with_provider_model(session, provider_model))
        )
    return pairs


async def _explicit_entry_scope(
    session: AsyncSession,
    explicit_entry_ids: list[str],
    extractor_version: str | None,
    extractor_id: str | None,
    provider_model: str | None,
    progress: Callable[[str], None] | None,
) -> list[tuple[str, str]]:
    """Scope for named entries, intersected with any particle-matching flags.

    Named entries resolve to their latest COMPLETE snapshot. When a
    particle-matching flag is *also* supplied the two scopes are **intersected**
    : before this, the explicit branch returned early and every other
    flag was discarded silently, so an operator narrowing by both entry and
    model got the whole entry — wider than asked for, and reindex supersedes.

    Intersecting rather than raising is deliberate. The fix has to hold for the
    HTTP route and the Python API too, not just the CLI, and AND-ing
    independent filters is the least-surprising reading on every one of them;
    it also errs strictly narrower, which is the safe direction here. Any
    entry dropped by the intersection is reported — narrowing is safe, but it
    should never be silent either.

    The store-wide auto-discovery unions (FAILED/PENDING snapshots, stale
    schema versions) stay bypassed on this path: an operator who names entries
    must not be handed the rest of the store, and folding the stale-schema
    union in *after* the intersection would re-open the same widening in a new
    place.
    """
    named: list[tuple[str, str]] = []
    for raw_id in explicit_entry_ids:
        entry_id = raw_id
        if len(raw_id) < 36:
            matches = await find_entry_ids_by_prefix(session, raw_id)
            if len(matches) == 1:
                entry_id = matches[0]
            elif len(matches) > 1:
                log.warning(
                    "Ambiguous entry prefix %r matches %d entries; skipping",
                    raw_id,
                    len(matches),
                )
                continue
            else:
                log.warning("Entry prefix %r not found; skipping", raw_id)
                continue
        snap_id = await get_latest_completed_snapshot_id(session, entry_id)
        if snap_id and (entry_id, snap_id) not in named:
            named.append((entry_id, snap_id))

    selected = await _particle_selector_pairs(
        session, extractor_version, extractor_id, provider_model
    )
    if selected is None:
        return named

    scope = [pair for pair in named if pair in selected]
    if len(scope) != len(named):
        flags = ", ".join(
            f"{name}={value!r}"
            for name, value in (
                ("extractor_version", extractor_version),
                ("extractor_id", extractor_id),
                ("provider_model", provider_model),
            )
            if value
        )
        message = (
            f"Reindex scope narrowed: {len(scope)} of {len(named)} named "
            f"entries matched {flags}; the rest are skipped."
        )
        log.warning(message)
        if progress is not None:
            progress(message)
    return scope


async def _identify_scope(
    session: AsyncSession,
    explicit_entry_ids: list[str] | None,
    extractor_version: str | None,
    extractor_id: str | None,
    include_failed: bool,
    provider_model: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Return list of (entry_id, snapshot_id) pairs to reindex."""
    if explicit_entry_ids:
        return await _explicit_entry_scope(
            session,
            explicit_entry_ids,
            extractor_version,
            extractor_id,
            provider_model,
            progress,
        )

    scope: list[tuple[str, str]] = []

    # Auto-discover: PENDING and FAILED snapshots
    if include_failed:
        scope.extend(
            await list_entry_snapshot_pairs_with_extraction_status(
                session, [ExtractionStatus.FAILED, ExtractionStatus.PENDING]
            )
        )

    # Auto-discover: particles matching the extractor / provider-model flags
    selected = await _particle_selector_pairs(
        session, extractor_version, extractor_id, provider_model
    )
    if selected is not None:
        scope.extend(selected)

    # Auto-discover: particles whose schema_version is older than current
    scope.extend(
        _source_pairs(await get_active_particles_with_stale_schema_version(session, SCHEMA_VERSION))
    )

    return list(set(scope))


async def _reindex_snapshot(
    session: AsyncSession,
    entry_id: str,
    snapshot_id: str,
    old_extractor_version: str | None,
) -> None:
    """Re-extract first, then supersede old particles on success.

    Supersession happens AFTER extraction succeeds so that a failed API call
    cannot leave the entry with no ACTIVE particles.
    """
    existing = await get_active_particles_for_entry(session, entry_id)
    to_supersede = [
        p for p in existing if any(ref.snapshot_id == snapshot_id for ref in p.provenance)
    ]

    # Pass the to-be-superseded IDs so conflict detection ignores them;
    # without this, within-entry re-extraction would spuriously create
    # INCONSISTENCY particles against the old versions of the same claims.
    carry_forward_ids: list[str] = []
    suppressed_ids: list[str] = []
    await extract_snapshot(
        session,
        entry_id,
        snapshot_id,
        supersede_ids=frozenset(p.id for p in to_supersede),
        carry_forward_ids_out=carry_forward_ids,
        suppressed_ids_out=suppressed_ids,
    )

    # Carry-forward particles stay ACTIVE under the new snapshot
    # because their chunk's text hashed identically. Exclude them from
    # supersession so the re-extraction is a no-op for unchanged chunks.
    #
    # suppression targets get the same protection, and here it is a
    # correctness requirement rather than an optimisation: the rung already
    # refuses to suppress into a ``supersede_ids`` particle, so a suppression
    # target is by construction some *other* ACTIVE particle — retiring it
    # would leave the re-observed claim with no ACTIVE copy at all.
    carry_forward = set(carry_forward_ids) | set(suppressed_ids)
    for p in to_supersede:
        if p.id in carry_forward:
            continue
        await update_particle_status(
            session, p.id, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REINDEX
        )

    await session.commit()
    if carry_forward:
        log.info(
            "Reindexed entry %s snapshot %s (%d particle(s) carried forward)",
            entry_id,
            snapshot_id,
            len(carry_forward),
        )
    else:
        log.info("Reindexed entry %s snapshot %s", entry_id, snapshot_id)
