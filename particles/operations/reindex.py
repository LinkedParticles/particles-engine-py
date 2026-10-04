# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    SCHEMA_VERSION,
    ExtractionStatus,
    Particle,
    ProvenanceRefType,
)
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import blob_size
from particles.corpus.store import (
    clear_partial_reads,
    find_entry_ids_by_prefix,
    get_entry,
    get_entry_uri_map,
    get_extraction_component_records,
    get_latest_completed_snapshot_id,
    get_partial_snapshot_id,
    list_entry_snapshot_pairs_with_extraction_status,
    list_extraction_bases,
    list_snapshots_for_entry,
)
from particles.extraction.components import ComponentTable
from particles.extraction.general import EXTRACTOR_ID as GENERAL_EXTRACTOR_ID
from particles.extraction.general import general_component_table
from particles.extraction.journal import EXTRACTOR_ID as JOURNAL_EXTRACTOR_ID
from particles.extraction.journal import journal_component_table
from particles.extraction.registry import select_extractor
from particles.extraction.subject_gate import GATE_COMPONENT, gate_digest
from particles.ingest.append_base import in_scope
from particles.ingest.pipeline import SnapshotOutcome
from particles.observability import traced
from particles.operations.extract import collapse_superseded_pending, extract_snapshot
from particles.operations.lint import run_lint
from particles.operations.reindex_scope import (
    ONLY_CHANGED_COMPONENTS_ENABLED,
    ONLY_CHANGED_COMPONENTS_REFUSAL,
    decide_reindex_scope,
    is_prefix,
    resolve_prefix,
    select_changed_components,
    union_selectors,
)
from particles.store.particle_store import (
    get_active_particles_for_entry,
    get_active_particles_with_extractor_id,
    get_active_particles_with_extractor_version,
    get_active_particles_with_provider_model,
    get_active_particles_with_stale_schema_version,
    get_particles_by_ids,
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
    #: Size of the snapshot's raw content in bytes (0 when the blob is
    #: missing): what an estimate prices the re-extraction from.
    source_bytes: int = 0
    #: Whether the entry is replayed whole as append-only.
    replayed: bool = False


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
    replay_entries: AbstractSet[str] = frozenset(),
) -> ReindexPlan:
    """Per-snapshot counts + blob presence for a resolved scope (pure reads).

    An entry in ``replay_entries`` is replayed whole, so its
    particle count is the entry's retirement set, counted once, on its first
    snapshot, rather than per snapshot, where one claim would count as many
    times as it has refs.
    """
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
        replayed = entry_id in replay_entries
        for position, snapshot_id in enumerate(snapshot_ids):
            if replayed:
                count = len(_entry_claims(active, entry_id)) if position == 0 else 0
            else:
                count = sum(
                    1 for p in active if any(ref.snapshot_id == snapshot_id for ref in p.provenance)
                )
            content_hash = content_hashes.get(snapshot_id)
            size = blob_size(content_hash) if content_hash is not None else None
            snapshot_plans.append(
                SnapshotPlan(
                    entry_id=entry_id,
                    snapshot_id=snapshot_id,
                    uri=uri_map.get(entry_id) or "",
                    particles=count,
                    blob_missing=size is None,
                    source_bytes=size or 0,
                    replayed=replayed,
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
    only_changed_components: bool = False,
) -> dict[str, object]:
    """Reindex corpus entries.

    The particle-selecting scopes (``extractor_version``, ``extractor_id``,
    ``provider_model``) union with each other and **intersect** with
    ``entry_ids`` when both are supplied. See ``_identify_scope``
    for why the combination narrows rather than erroring, and
    ``decide_reindex_scope`` for the rules themselves.

    Args:
        entry_ids: explicit list of entries to reindex; if None, auto-discover scope.
        extractor_version: superseded extractor version to replace (filters by extractor_ref).
        extractor_id: extractor name to re-extract regardless of version. Useful when
            a shared upstream (e.g. a prompt change) affects multiple extractors that
            delegate to it but didn't bump their own version.
        include_failed: also reindex entries with FAILED snapshots. Applies to
            auto-discovery only; named entries resolve to their latest
            *COMPLETE* snapshot, so an explicit scope never contains a FAILED
            one to include or exclude.
        provider_model: re-extract particles stamped with this
            ``"<provider>:<model>"`` pairing, the handle for
            undoing an uncalibrated provider swap. Matched exactly; particles
            with no stamp (deterministic extractors, direct assertions, or
            anything minted before the stamp existed) never match.
        rate_limit_per_minute: max extraction jobs per minute.
        run_post_lint: run a Lint pass after reindex completes.
        progress: optional callback for human-readable progress lines. The CLI
            wires this to ``typer.echo`` when ``--verbose`` is set so the
            operator can see that a long-running reindex isn't stuck.
        dry_run: compute and report the work plan, then return without
            extracting: zero LLM calls, zero writes (mirrors the Notion
            exporter's dry-run discipline). The returned summary
            carries the full plan including per-snapshot counts.
        on_plan: optional callback for the upfront work-plan lines (the scope
            summary + any missing-blob warnings), emitted before the first
            extraction. Separate from ``progress`` because the plan is meant
            to print unconditionally while per-entry progress stays opt-in.
        on_status: optional callback fired after **each** snapshot completes
            with a compact position line such as ``snapshot 12/89 (entry
            0a8fb1a9…) — 3 failed``, so a long run's liveness display can
            show how far along it is, not just elapsed time. Distinct from
            ``progress`` (opt-in, one full line per item, appended): the
            status is a single replaceable line the CLI feeds to the
            heartbeat.
        only_changed_components: narrow an ``extractor_version`` scope to the
            snapshots whose recorded extraction components changed since they
            were extracted (``select_changed_components``). Refused
            with ``ValueError`` while ``ONLY_CHANGED_COMPONENTS_ENABLED`` is
            off. A skipped snapshot keeps its old version stamp, so the next
            scope over that version selects it again.

    Returns a summary dict with counts and any errors.
    """
    if only_changed_components and not ONLY_CHANGED_COMPONENTS_ENABLED:
        raise ValueError(ONLY_CHANGED_COMPONENTS_REFUSAL)
    resolved = await resolve_reindex_work(
        session,
        entry_ids=entry_ids,
        extractor_version=extractor_version,
        extractor_id=extractor_id,
        include_failed=include_failed,
        provider_model=provider_model,
        only_changed_components=only_changed_components,
        progress=progress,
        dry_run=dry_run,
    )
    work, replays, plan = resolved.work, resolved.replays, resolved.plan
    scope = [(entry_id, sid) for entry_id, snaps in work for sid in snaps]
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

    done = 0
    for entry_id, snapshot_ids in work:
        i = done + 1
        done += len(snapshot_ids)
        if progress is not None:
            uri = await _lookup_entry_uri(session, entry_id)
            if entry_id in replays:
                progress(
                    f"[{i}-{done}/{total}] replaying {entry_id[:8]}… "
                    f"({len(snapshot_ids)} snapshot(s), append-only) {uri}"
                )
            else:
                progress(
                    f"[{i}/{total}] reindexing {entry_id[:8]}… snap {snapshot_ids[0][:8]}… {uri}"
                )
        try:
            if entry_id in replays:
                await _reindex_append_entry(session, entry_id, snapshot_ids)
            else:
                await _reindex_snapshot(session, entry_id, snapshot_ids[0], extractor_version)
            succeeded.extend([entry_id] * len(snapshot_ids))
        except Exception as exc:
            log.error("Reindex failed for entry %s snapshot(s) %s: %s", entry_id, snapshot_ids, exc)
            if progress is not None:
                progress(f"[{i}/{total}] FAILED: {exc}")
            failed.extend([entry_id] * len(snapshot_ids))
        if on_status is not None:
            status = f"snapshot {done}/{total} (entry {entry_id[:8]}…)"
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


@dataclass(frozen=True)
class ResolvedWork:
    """A reindex scope resolved into work, with its upfront plan."""

    #: ``(entry_id, snapshot_ids)`` in run order: whole-entry replays first.
    work: list[tuple[str, list[str]]]
    #: Append-only entries replayed whole, to their snapshots.
    replays: dict[str, list[str]]
    plan: ReindexPlan
    #: Pairs ``only_changed_components`` dropped from the scope.
    component_skipped: list[tuple[str, str]] = field(default_factory=list)


async def resolve_reindex_work(
    session: AsyncSession,
    *,
    entry_ids: list[str] | None = None,
    extractor_version: str | None = None,
    extractor_id: str | None = None,
    include_failed: bool = True,
    provider_model: str | None = None,
    only_changed_components: bool = False,
    progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> ResolvedWork:
    """Resolve the requested scope into the work a reindex would do, and plan it.

    Shared by :func:`reindex` and the ``--estimate`` sample
    (``operations.reindex_estimate``), so the estimate samples exactly the
    scope a live run would sweep. Writes only on a live run, and only the
    collapse (see :func:`_collapse_for_auto_discovery`); with
    ``dry_run`` it reads.
    """
    # refuse to reindex into a store with mismatched-schema
    # particles. Reindex writes new ACTIVE particles and supersedes old
    # ones — both operations assume the surrounding store is current.
    from particles.operations.version_guard import assert_store_schema_current

    await assert_store_schema_current(session)

    # Apply step, ahead of the gather (D2): the collapse
    # commits under the writer lock, so it runs to completion here and scope
    # identification below only reads.
    collapsed = await _collapse_for_auto_discovery(
        session, entry_ids, include_failed, progress=progress, dry_run=dry_run
    )
    scope = await _identify_scope(
        session,
        entry_ids,
        extractor_version,
        extractor_id,
        include_failed,
        provider_model,
        progress=progress,
        collapsed=collapsed,
    )
    component_skipped: list[tuple[str, str]] = []
    if only_changed_components:
        scope, component_skipped = await _narrow_to_changed_components(
            session, scope, extractor_version
        )
    # an APPEND_ONLY entry the scope reaches through a COMPLETE
    # snapshot is replayed whole, grouped by entry so the order is its capture
    # order however the scope enumerated its snapshots.
    replays, scope = await _group_append_only(session, scope)
    work: list[tuple[str, list[str]]] = [(e, snaps) for e, snaps in replays.items()]
    work += [(entry_id, [snapshot_id]) for entry_id, snapshot_id in scope]
    # The upfront work plan (2026-08-02 incident): report what the resolved
    # scope will cost — entries, snapshots, supersede-able particles, known
    # missing blobs — BEFORE the first LLM call is spent.
    description = _describe_scope(
        entry_ids, extractor_version, extractor_id, include_failed, provider_model
    )
    if component_skipped:
        description += (
            f"; {len(component_skipped)} snapshot(s) skipped, components unchanged "
            "(their version stamp is kept)"
        )
    plan = await _build_plan(
        session,
        [(entry_id, sid) for entry_id, snaps in work for sid in snaps],
        description,
        replay_entries=frozenset(replays),
    )
    return ResolvedWork(work=work, replays=replays, plan=plan, component_skipped=component_skipped)


def current_component_tables() -> dict[str, ComponentTable]:
    """Each component-recording extractor's current table, the subject gate merged in.

    What :func:`select_changed_components` compares a snapshot's record
    against. The gate is source-dependent (exempt source types), so
    it joins the table but never its ``always`` set.
    """
    gate_cfg = get_config().subject_gate
    gate = ComponentTable(
        digests={
            GATE_COMPONENT: gate_digest(
                cli_binaries=gate_cfg.cli_binaries,
                allowlist=gate_cfg.allowlist,
                dispositions=gate_cfg.dispositions,
            )
        }
        if gate_cfg.enabled
        else {}
    )
    return {
        GENERAL_EXTRACTOR_ID: general_component_table().merged(gate),
        JOURNAL_EXTRACTOR_ID: journal_component_table().merged(gate),
    }


async def _narrow_to_changed_components(
    session: AsyncSession,
    scope: list[tuple[str, str]],
    extractor_version: str | None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Keep the pairs whose recorded components changed; return ``(kept, skipped)``.

    Only meaningful over an ``extractor_version`` scope, the one a bump
    creates, and refused without one. Skipping writes nothing (see
    :func:`select_changed_components`).
    """
    if not extractor_version:
        raise ValueError("--only-changed-components narrows an --extractor-version scope only")
    records = await get_extraction_component_records(session, {sid for _, sid in scope})
    selection = select_changed_components(scope, records, current_component_tables())
    return list(selection.kept), selection.skipped


async def _collapse_for_auto_discovery(
    session: AsyncSession,
    entry_ids: list[str] | None,
    include_failed: bool,
    *,
    progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> frozenset[str]:
    """Collapse superseded FAILED/PENDING generations; return the collapsed ids.

    Runs only for the auto-discovery FAILED/PENDING union (no named entries,
    ``include_failed``), the one scope that would otherwise retry them. A
    FAILED or PENDING generation of a MUTABLE entry that a newer snapshot has
    replaced is not worth a retry, and retrying it after the newer one is
    COMPLETE would retire the current generation.

    **Commits** on a live run (the collapse's writer-lock transaction), which
    is why it is its own step before scope gathering rather than a call
    inside it. ``--dry-run`` promises zero writes, so there it only plans the
    collapse, and the returned ids drop the same snapshots from the reported
    scope that a live run would have marked.
    """
    if entry_ids or not include_failed:
        return frozenset()
    collapse = await collapse_superseded_pending(session, dry_run=dry_run)
    if progress is not None:
        line = collapse.summary()
        if line:
            progress(line)
    return frozenset(snapshot_id for _, snapshot_id, _ in collapse.collapsed)


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
    """Gather the snapshot pairs each particle-matching flag selects, unioned.

    Only the flags actually passed are queried; ``union_selectors`` decides
    the ``None``-versus-empty distinction.
    """
    # Particles produced by a specific "<provider>:<model>" pairing.
    # Exact equality on the stamped column, never a substring match — pairings
    # nest, so a substring scope would sweep in the sibling model this exists
    # to separate. Note the scope unit is the *snapshot*: a snapshot whose
    # particles are model-mixed is re-extracted whole, which is the intended
    # behaviour for undoing a provider trial but is not a particle-level
    # surgical tool.
    return union_selectors(
        _source_pairs(await get_active_particles_with_extractor_version(session, extractor_version))
        if extractor_version
        else None,
        _source_pairs(await get_active_particles_with_extractor_id(session, extractor_id))
        if extractor_id
        else None,
        _source_pairs(await get_active_particles_with_provider_model(session, provider_model))
        if provider_model
        else None,
    )


async def _gather_named(
    session: AsyncSession, explicit_entry_ids: list[str]
) -> list[tuple[str, str]]:
    """Resolve each named id (prefix or full) to its latest COMPLETE snapshot.

    An ambiguous or unknown prefix is logged and skipped (``resolve_prefix``
    decides which). An entry with no COMPLETE snapshot contributes nothing,
    unless it is an in-scope APPEND_ONLY entry whose snapshot holds a partial
    read: that snapshot carries ACTIVE claims, so it is named.
    """
    named: list[tuple[str, str]] = []
    for raw_id in explicit_entry_ids:
        matches = await find_entry_ids_by_prefix(session, raw_id) if is_prefix(raw_id) else []
        resolution = resolve_prefix(raw_id, matches)
        if resolution.problem == "ambiguous":
            log.warning(
                "Ambiguous entry prefix %r matches %d entries; skipping",
                raw_id,
                len(matches),
            )
            continue
        if resolution.entry_id is None:
            log.warning("Entry prefix %r not found; skipping", raw_id)
            continue
        snap_id = await get_latest_completed_snapshot_id(session, resolution.entry_id)
        if snap_id is None and await _replay_snapshots(session, resolution.entry_id) is not None:
            snap_id = await get_partial_snapshot_id(session, resolution.entry_id)
        if snap_id:
            named.append((resolution.entry_id, snap_id))
    return named


async def _identify_scope(
    session: AsyncSession,
    explicit_entry_ids: list[str] | None,
    extractor_version: str | None,
    extractor_id: str | None,
    include_failed: bool,
    provider_model: str | None = None,
    progress: Callable[[str], None] | None = None,
    collapsed: AbstractSet[str] = frozenset(),
) -> list[tuple[str, str]]:
    """Return list of (entry_id, snapshot_id) pairs to reindex.

    Gathers what the scope decision needs, then hands it to the pure
    ``decide_reindex_scope`` (D2). Reads only: the collapse
    is applied (or, under ``--dry-run``, planned) by ``reindex`` before this
    runs, and arrives as ``collapsed``.

    Named entries resolve to their latest COMPLETE snapshot and are
    **intersected** with any particle-matching flags. Intersecting
    rather than raising is deliberate: the fix has to hold for the HTTP route
    and the Python API too, not just the CLI, and AND-ing independent filters
    is the least-surprising reading on every one of them; it also errs
    strictly narrower, which is the safe direction for a superseding verb.
    Any entry dropped by the intersection is reported, since narrowing is
    safe but should never be silent. The store-wide auto-discovery unions
    (FAILED/PENDING, stale schema) are not gathered on that path: folding them
    in would re-open the same widening in a new place.
    """
    selected = await _particle_selector_pairs(
        session, extractor_version, extractor_id, provider_model
    )
    if explicit_entry_ids:
        decided = decide_reindex_scope(
            named=await _gather_named(session, explicit_entry_ids),
            selected=selected,
            failed_or_pending=[],
            collapsed=collapsed,
            stale_schema=[],
        )
    else:
        decided = decide_reindex_scope(
            named=None,
            selected=selected,
            failed_or_pending=await list_entry_snapshot_pairs_with_extraction_status(
                session, [ExtractionStatus.FAILED, ExtractionStatus.PENDING]
            )
            if include_failed
            else [],
            collapsed=collapsed,
            stale_schema=_source_pairs(
                await get_active_particles_with_stale_schema_version(session, SCHEMA_VERSION)
            ),
        )

    if decided.narrowed is not None:
        kept, named_count = decided.narrowed
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
            f"Reindex scope narrowed: {kept} of {named_count} named "
            f"entries matched {flags}; the rest are skipped."
        )
        log.warning(message)
        if progress is not None:
            progress(message)
    return decided.pairs


async def _group_append_only(
    session: AsyncSession, scope: list[tuple[str, str]]
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Split the scope into whole-entry replays and ordinary snapshot pairs.

    An entry is replayed when it is APPEND_ONLY, read by the general extractor
    with ``extraction.append_only_delta`` on (the delta's scope, §6), and the
    scope names one of its COMPLETE snapshots: re-extracting that snapshot
    alone would either replace the entry's claims with its tail's or duplicate
    them. The replay covers every COMPLETE snapshot of the entry, in capture
    order. A PENDING or FAILED snapshot the scope names stays an ordinary pair,
    extracted after the replay as the delta it is.
    """
    if not get_config().extraction.append_only_delta:
        return {}, scope
    replays: dict[str, list[str]] = {}
    ordinary: list[tuple[str, str]] = []
    checked: dict[str, list[str] | None] = {}
    for entry_id, snapshot_id in scope:
        if entry_id not in checked:
            checked[entry_id] = await _replay_snapshots(session, entry_id)
        replay = checked[entry_id]
        if replay is not None and snapshot_id in replay:
            replays.setdefault(entry_id, replay)
        else:
            ordinary.append((entry_id, snapshot_id))
    return replays, ordinary


async def _replay_snapshots(session: AsyncSession, entry_id: str) -> list[str] | None:
    """An in-scope append-only entry's COMPLETE snapshots in capture order, else ``None``."""
    entry = await get_entry(session, entry_id)
    if entry is None or not in_scope(entry, select_extractor(entry.source_type)):
        return None
    return [
        row.snapshot_id
        for row in await list_extraction_bases(session, entry_id)
        if row.extraction_status is ExtractionStatus.COMPLETE
    ]


def _entry_claims(particles: list[Particle], entry_id: str) -> list[Particle]:
    """The extractor claims among ``particles`` with a SOURCE ref to ``entry_id``."""
    return [
        p
        for p in particles
        if p.extractor_ref is not None
        and any(
            ref.type is ProvenanceRefType.SOURCE and ref.corpus_entry_id == entry_id
            for ref in p.provenance
        )
    ]


async def _reindex_append_entry(
    session: AsyncSession, entry_id: str, snapshot_ids: list[str]
) -> None:
    """Retire an append-only entry's claims and replay its snapshots in order.

    **Retire** is every ACTIVE extractor claim with a source ref to the entry,
    less the re-anchored restatements the per-snapshot path already exempts,
    and it happens last, as there: the claims are threaded as
    ``supersede_ids`` through every step, so no step pairs against them or
    carries them forward, and they are retired only once every step succeeded.

    **Replay** extracts the snapshots in capture order: the first whole, with
    no base, and each later one as a delta from the one before, each stamping
    its ``extracted_through``. Each claim so cites the snapshot, and the time,
    that first contained its passage.

    A step that fails stops the replay and retires nothing. The steps already
    done have written their claims beside the old ones; running the reindex
    again retires both and replays from the start.
    """
    existing = await get_active_particles_for_entry(session, entry_id)
    to_retire = _entry_claims(existing, entry_id)
    reanchored = await _reanchored_ids(session, to_retire)
    to_retire = [p for p in to_retire if p.id not in reanchored]
    supersede_ids = frozenset(p.id for p in to_retire)

    carry_forward_ids: list[str] = []
    suppressed_ids: list[str] = []
    previous: str | None = None
    for snapshot_id in snapshot_ids:
        outcome = SnapshotOutcome()
        await extract_snapshot(
            session,
            entry_id,
            snapshot_id,
            supersede_ids=supersede_ids,
            carry_forward_ids_out=carry_forward_ids,
            suppressed_ids_out=suppressed_ids,
            skip_if_superseded=True,
            append_base=previous,
            outcome_out=outcome,
            ignore_partial=True,
        )
        if outcome.failed_calls or outcome.skipped is not None:
            raise RuntimeError(
                f"replay stopped at snapshot {snapshot_id[:8]}… "
                f"({outcome.skipped or f'{outcome.failed_calls} failed call(s)'}); "
                "nothing was retired, and a rerun replays the entry from the start"
            )
        previous = snapshot_id

    kept = set(carry_forward_ids) | set(suppressed_ids)
    for p in to_retire:
        if p.id in kept:
            continue
        await update_particle_status(
            session, p.id, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REINDEX
        )
    # a snapshot that is not COMPLETE no longer vouches for text
    # whose claims were just retired. Its offset, marker and component record
    # go in the same transaction, so it is read next as the delta it is from
    # the replayed base.
    cleared = await clear_partial_reads(session, entry_id)
    await session.commit()
    if cleared:
        log.info(
            "Append-only entry %s: cleared the partial read of %d unfinished snapshot(s)",
            entry_id,
            len(cleared),
        )
    log.info(
        "Replayed append-only entry %s over %d snapshot(s); %d claim(s) retired",
        entry_id,
        len(snapshot_ids),
        len([p for p in to_retire if p.id not in kept]),
    )


async def _reanchored_ids(session: AsyncSession, particles: list[Particle]) -> set[str]:
    """The ids among ``particles`` that are re-anchor restatements."""
    predecessors = {p.supersedes for p in particles if p.supersedes is not None}
    if not predecessors:
        return set()
    loaded = await get_particles_by_ids(session, sorted(predecessors))
    replaced = {
        pid
        for pid, prior in loaded.items()
        if prior.status_reason is StatusReason.SUPERSEDED_BY_REANCHOR
    }
    return {p.id for p in particles if p.supersedes in replaced}


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
    # a restatement the re-anchor pass wrote keeps its original's
    # SOURCE refs, but it is the product of a judgement over the passage, not of
    # the extractor being upgraded. Retiring it would bring back the present-tense
    # claim it replaced; a re-emitted copy of that claim is matched to it by the
    # pass instead.
    reanchored = await _reanchored_ids(session, to_supersede)
    to_supersede = [p for p in to_supersede if p.id not in reanchored]

    # Pass the to-be-superseded IDs so conflict detection ignores them;
    # without this, within-entry re-extraction would spuriously create
    # INCONSISTENCY particles against the old versions of the same claims.
    carry_forward_ids: list[str] = []
    suppressed_ids: list[str] = []
    outcome = SnapshotOutcome()
    await extract_snapshot(
        session,
        entry_id,
        snapshot_id,
        supersede_ids=frozenset(p.id for p in to_supersede),
        carry_forward_ids_out=carry_forward_ids,
        suppressed_ids_out=suppressed_ids,
        # no reindex scope legitimately names a collapsed snapshot
        # (named entries resolve to an uncollapsed generation; provenance
        # scopes name snapshots that have particles), so this only ever skips
        # one collapsed by another runner after the scope was built.
        skip_if_superseded=True,
        outcome_out=outcome,
        # the claims a partial read of this snapshot wrote are among
        # those being replaced, so the read starts as if they were not there.
        ignore_partial=True,
    )
    # A call that failed left the snapshot PENDING with nothing written.
    # Retiring the old claims now would leave its text with no claims at all,
    # so the snapshot fails here and keeps them, as a replay step does.
    if outcome.failed_calls:
        raise RuntimeError(
            f"re-extraction of snapshot {snapshot_id[:8]}… left it PENDING "
            f"({outcome.failed_calls} failed call(s)); nothing was retired"
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
