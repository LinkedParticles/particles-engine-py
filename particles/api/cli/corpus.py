"""corpus sub-Typer — inspect deposited corpus entries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from sqlalchemy.ext.asyncio import AsyncSession

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.db import session_scope
from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

if TYPE_CHECKING:
    from particles.corpus.blob_fsck import BlobRef, RehomeOutcome, RejectedBlob

corpus_app = typer.Typer(help="Inspect deposited corpus entries.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")


@corpus_app.command("list")
def corpus_list_cmd(
    source_type: str | None = typer.Option(
        None, help="Filter by source type (PDF, WEB_PAGE, WIKIDATA_API, …)"
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help=(
            "Emit machine-readable JSON with the full, untruncated fields "
            "(entry_id, source_type, uri_r, extraction_status, particle_count, "
            "tags, created_at) instead of the human table."
        ),
    ),
) -> None:
    """List all deposited corpus entries."""
    run(_corpus_list(source_type, json_out))


async def _corpus_list(source_type_filter: str | None, json_out: bool) -> None:
    import json

    from sqlalchemy import func, select

    from particles.api.cli._remote import ensure_local
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.store.particle_store import ParticleRow

    ensure_local("corpus list")

    async with session_scope() as session:
        q = select(CorpusEntryRow).order_by(CorpusEntryRow.created_at)
        rows = (await session.execute(q)).scalars().all()

        # Gather one record per entry first, then render. The JSON mode carries
        # the *full* fields — notably the untruncated ``uri_r`` the human table
        # left-truncates to its last 60 chars — so a script can reconstruct
        # ``deposit`` commands or diff corpus sets across stores.
        records: list[dict[str, Any]] = []
        for row in rows:
            if source_type_filter and row.source_type != source_type_filter:
                continue
            snap_q = (
                select(SnapshotRow)
                .where(SnapshotRow.entry_id == row.entry_id)
                .order_by(SnapshotRow.captured_at.desc())
                .limit(1)
            )
            snap = (await session.execute(snap_q)).scalar_one_or_none()
            status = snap.extraction_status if snap else "none"
            cnt = (
                await session.execute(
                    select(func.count())
                    .select_from(ParticleRow)
                    .where(
                        ParticleRow.status == "ACTIVE",
                        ParticleRow.particle_type == "CLAIM",
                        ParticleRow.provenance_json.like(
                            f"%{escape_like_pattern(row.entry_id)}%", escape=LIKE_ESCAPE
                        ),
                    )
                )
            ).scalar_one()
            records.append(
                {
                    "entry_id": row.entry_id,
                    "source_type": row.source_type,
                    "uri_r": row.uri_r,
                    "extraction_status": status,
                    "particle_count": cnt,
                    "tags": json.loads(row.tags_json or "[]"),
                    "created_at": row.created_at.isoformat(),
                }
            )

        if json_out:
            typer.echo(json.dumps(records, indent=2, sort_keys=True))
            return

        typer.echo(f"{'ENTRY':8}  {'TYPE':14}  {'STATUS':12}  {'PARTS':5}  SOURCE")
        typer.echo("-" * 80)
        for rec in records:
            label = (rec["uri_r"] or "").replace("file://", "")
            label = label[-60:] if len(label) > 60 else label
            typer.echo(
                f"{rec['entry_id'][:8]}  {rec['source_type']:14s}  "
                f"{rec['extraction_status']:12s}  {rec['particle_count']:5d}  {label}"
            )


@corpus_app.command("show")
def corpus_show_cmd(
    entry_id: str = typer.Argument(..., help="Entry ID (prefix OK)"),
    limit: int = typer.Option(10, help="Max particles to show"),
) -> None:
    """Show details, extracted particles, and follow edges for a corpus entry."""
    run(_corpus_show(entry_id, limit))


async def _corpus_show(entry_id_prefix: str, limit: int) -> None:
    from particles.api.client import get_backend

    backend = get_backend()
    if backend.remote:
        # Prefix resolution + particle/subject counts read the local store, so
        # they are local-only; remote shows the entry header the
        # GET /corpus/{id} endpoint serves and degrades the rich block.
        entry = await backend.corpus_show(entry_id_prefix)
        if entry is None:
            typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
            raise typer.Exit(1)
        _render_corpus_entry_header(entry)
        return

    async with session_scope() as session:
        resolved_id = await _resolve_show_entry_id(session, entry_id_prefix)
        entry = await backend.corpus_show(resolved_id)
        if entry is None:
            typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
            raise typer.Exit(1)
        _render_corpus_entry_header(entry)
        await _render_corpus_entry_detail(session, entry, limit)


@corpus_app.command("cat")
def corpus_cat_cmd(
    selector: str = typer.Argument(
        ..., help="Snapshot ID or corpus-entry ID (prefix OK; entry → latest snapshot)"
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Write the raw stored bytes to stdout instead of the text preview.",
    ),
) -> None:
    """Dump a snapshot's stored content — the exact bytes the extractor saw.

    Accepts a snapshot ID or a corpus-entry ID (prefix OK); an entry resolves to
    its most-recent snapshot. By default renders the same text the extractor
    derives from the blob (html2text for HTML, pypdf for PDF), so an empty
    preview explains an "Empty content" / zero-particle extraction. ``--raw``
    streams the original bytes (pipe to a file or ``less``). Identifying
    metadata (snapshot id, hash, byte size) is written to stderr, so stdout
    stays pipeable.
    """
    run(_corpus_cat(selector, raw))


async def _corpus_cat(selector: str, raw: bool) -> None:
    import sys

    from particles.api.client import get_backend

    backend = get_backend()
    try:
        result = await backend.corpus_blob(selector)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if result is None:
        typer.echo(
            f"No blob found for {selector!r} — unknown ID, or the blob is missing "
            "on the engine host.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"snapshot {result.snapshot_id}  sha256:{result.content_hash[:12]}…  "
        f"{len(result.content)} bytes",
        err=True,
    )
    if raw:
        sys.stdout.buffer.write(result.content)
        return

    from particles.extraction.general import content_to_text

    text = content_to_text(result.content)
    if not text.strip():
        typer.echo(
            "(empty text preview — the source rendered to no extractable text, "
            "which is why extraction produced 0 particles. Use --raw to see the "
            "stored bytes.)",
            err=True,
        )
        return
    typer.echo(text)


async def _resolve_show_entry_id(session: AsyncSession, entry_id_prefix: str) -> str:
    """Resolve an entry-id prefix for ``corpus show`` (falls back to the prefix)."""
    from sqlalchemy import select

    from particles.corpus.store import CorpusEntryRow

    if len(entry_id_prefix) < 36:
        result = await session.execute(
            select(CorpusEntryRow).where(
                CorpusEntryRow.entry_id.like(
                    f"{escape_like_pattern(entry_id_prefix)}%", escape=LIKE_ESCAPE
                )
            )
        )
        row = result.scalar_one_or_none()
        return row.entry_id if row else entry_id_prefix
    return entry_id_prefix


def _render_corpus_entry_header(entry: Any) -> None:
    """Render the entry metadata + snapshot list (transport-agnostic)."""
    label = (entry.uri_r or "(no uri)").replace("file://", "")
    typer.echo(f"Entry:       {entry.entry_id}")
    typer.echo(f"Source:      {entry.source_type}")
    typer.echo(f"Location:    {label}")
    typer.echo(
        f"Deposited:   {entry.created_at.strftime('%Y-%m-%d %H:%M')}  by {entry.deposited_by}"
    )
    typer.echo("Snapshots:")
    for snap in sorted(entry.snapshots, key=lambda s: s.captured_at):
        typer.echo(
            f"  {snap.snapshot_id[:8]}…  {snap.extraction_status:12s}"
            f"  {snap.captured_at.strftime('%Y-%m-%d %H:%M')}"
        )


async def _render_corpus_entry_detail(session: AsyncSession, entry: Any, limit: int) -> None:
    """Render the particle counts / subjects / samples (local-only enrichment)."""
    import json as _json

    from sqlalchemy import select

    from particles.store.particle_store import ParticleRow
    from particles.store.subject_store import SubjectRow

    prows = (
        (
            await session.execute(
                select(ParticleRow).where(
                    ParticleRow.status == "ACTIVE",
                    ParticleRow.particle_type == "CLAIM",
                    ParticleRow.provenance_json.like(
                        f"%{escape_like_pattern(entry.entry_id)}%", escape=LIKE_ESCAPE
                    ),
                )
            )
        )
        .scalars()
        .all()
    )

    # Collect distinct subjects
    subject_ids: set[str] = set()
    for p in prows:
        subject_ids.update(_json.loads(p.subject_ids_json))

    typer.echo(
        f"\nParticles:   {len(prows)} active CLAIM particles across {len(subject_ids)} subjects"
    )

    if subject_ids:
        typer.echo("\nSubjects:")
        for sid in sorted(subject_ids):
            srow = await session.get(SubjectRow, sid)
            name = srow.canonical_name if srow else sid
            typer.echo(f"  {sid[:8]}…  {name}")

    typer.echo(f"\nParticles (first {limit}):")
    for p in prows[:limit]:
        typer.echo(f"  [{p.confidence_value:.2f}] {p.id[:8]}…  {p.content[:90]}")

    await _render_follow_edges(session, entry.entry_id)


async def _render_follow_edges(session: AsyncSession, entry_id: str) -> None:
    """Render the cross-entry follow edges touching this entry.

    Surfaces the same ``corpus_follow_edges`` rows the ``corpus links list``
    verb audits, inline in ``corpus show`` so an operator inspecting a single
    entry sees which sources amplified it (incoming) and which external URLs it
    followed (outgoing) without a second command. Skipped entirely when the
    entry has no follow edges — the common case stays uncluttered.
    """
    from sqlalchemy import select

    from particles.corpus.follow_edges import get_follow_sources, get_follow_targets
    from particles.corpus.store import CorpusEntryRow

    out_edges = await get_follow_targets(session, entry_id)
    in_edges = await get_follow_sources(session, entry_id)
    if not out_edges and not in_edges:
        return

    related_ids = {e.target_entry_id for e in out_edges} | {e.via_entry_id for e in in_edges}
    label_map: dict[str, str] = {}
    if related_ids:
        rows = await session.execute(
            select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r).where(
                CorpusEntryRow.entry_id.in_(related_ids)
            )
        )
        for eid, uri in rows.all():
            label_map[eid] = (uri or "").replace("file://", "")

    def _label(eid: str) -> str:
        uri = label_map.get(eid, "")
        short = uri[-60:] if len(uri) > 60 else uri
        return f"{eid[:8]}…  {short}".rstrip()

    if out_edges:
        typer.echo(f"\nOutgoing follows ({len(out_edges)}):")
        for e in out_edges:
            typer.echo(
                f"  → {_label(e.target_entry_id)}"
                f"  [{e.link_type}, {e.discovered_at.date().isoformat()}]"
            )
    if in_edges:
        typer.echo(f"\nIncoming follows ({len(in_edges)}):")
        for e in in_edges:
            typer.echo(
                f"  ← {_label(e.via_entry_id)}"
                f"  [{e.link_type}, {e.discovered_at.date().isoformat()}]"
            )


@corpus_app.command("delete")
def corpus_delete_cmd(
    entry_id: str = typer.Argument(..., help="Entry ID (prefix OK)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete a corpus entry, its snapshots, and all particles sourced from it."""
    run(_corpus_delete(entry_id, yes))


async def _corpus_delete(entry_id_prefix: str, yes: bool) -> None:
    from sqlalchemy import delete, select

    from particles.api.cli._remote import ensure_local
    from particles.corpus.store import CorpusEntryRow, SnapshotRow, get_entry
    from particles.store.particle_store import ParticleRow, ProvenanceEdgeRow
    from particles.store.subject_store import ParticleSubjectRow

    ensure_local("corpus delete")

    async with session_scope() as session:
        if len(entry_id_prefix) < 36:
            result = await session.execute(
                select(CorpusEntryRow).where(
                    CorpusEntryRow.entry_id.like(
                        f"{escape_like_pattern(entry_id_prefix)}%", escape=LIKE_ESCAPE
                    )
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
                raise typer.Exit(1)
            entry_id = row.entry_id
        else:
            entry_id = entry_id_prefix

        entry = await get_entry(session, entry_id)
        if entry is None:
            typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
            raise typer.Exit(1)

        # Count particles to be deleted
        particle_ids_result = await session.execute(
            select(ProvenanceEdgeRow.particle_id).where(
                ProvenanceEdgeRow.corpus_entry_id == entry_id
            )
        )
        particle_ids = list(particle_ids_result.scalars())

        label = (entry.uri_r or entry_id).replace("file://", "")
        typer.echo(f"Entry:     {entry_id[:8]}…  {label}")
        typer.echo(f"Snapshots: {len(entry.snapshots)}")
        typer.echo(f"Particles: {len(particle_ids)} will be deleted")

        if not yes:
            typer.confirm("Delete this entry and all its data?", abort=True)

        # Subjects linked to the doomed particles — captured *before* we drop
        # the join rows, so afterwards we can re-check each for orphanhood.
        candidate_subject_ids: set[str] = set()
        if particle_ids:
            candidate_subject_ids = set(
                (
                    await session.execute(
                        select(ParticleSubjectRow.subject_id).where(
                            ParticleSubjectRow.particle_id.in_(particle_ids)
                        )
                    )
                ).scalars()
            )

        # Delete particles + every index row keyed on them. These tables carry
        # no FK to ``particles`` (SQLite FK enforcement is off), so they would
        # otherwise be left dangling.
        if particle_ids:
            await session.execute(delete(ParticleRow).where(ParticleRow.id.in_(particle_ids)))
            await _purge_particle_index_rows(session, particle_ids)

        # Subjects that lost their last link become orphans; drop them and any
        # synthesis-cache rows keyed on them. A subject still linked to a
        # surviving particle is left untouched.
        subj_removed, synth_removed = await _purge_orphan_subjects(session, candidate_subject_ids)

        # Delete provenance edges
        await session.execute(
            delete(ProvenanceEdgeRow).where(ProvenanceEdgeRow.corpus_entry_id == entry_id)
        )
        # Delete snapshots
        await session.execute(delete(SnapshotRow).where(SnapshotRow.entry_id == entry_id))
        # Delete entry
        await session.execute(delete(CorpusEntryRow).where(CorpusEntryRow.entry_id == entry_id))
        await session.commit()

    parts = [f"{len(particle_ids)} particles removed"]
    if subj_removed:
        parts.append(f"{subj_removed} orphaned subjects")
    if synth_removed:
        parts.append(f"{synth_removed} synthesis-cache rows")
    typer.echo(f"Deleted entry {entry_id[:8]}… ({', '.join(parts)}).")


@corpus_app.command("retract")
def corpus_retract_cmd(
    entry_id: str = typer.Argument(..., help="Entry ID (prefix OK)"),
    reason: str | None = typer.Option(
        None, "--reason", help="Operator rationale, recorded in the event log"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan without writing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Retract every live particle from a source, preserving the corpus + snapshots.

    The non-destructive sibling of ``corpus delete``: live particles
    (ACTIVE / INCONSISTENCY) become RETRACTED with reason SOURCE_RETRACTED; the
    entry, its snapshots, and the particles themselves survive so the audit
    trail is intact. Idempotent. Run ``particles lint`` afterwards to cascade
    PROVENANCE_STALE to downstream particles.
    """
    run(_corpus_retract(entry_id, reason, dry_run, yes))


async def _resolve_entry_id(session: AsyncSession, entry_id_prefix: str) -> str:
    """Resolve an entry-id prefix to a full id, or Exit(1) if absent/ambiguous."""
    from sqlalchemy import select

    from particles.corpus.store import CorpusEntryRow

    if len(entry_id_prefix) >= 36:
        return entry_id_prefix
    rows = (
        (
            await session.execute(
                select(CorpusEntryRow.entry_id).where(
                    CorpusEntryRow.entry_id.like(
                        f"{escape_like_pattern(entry_id_prefix)}%", escape=LIKE_ESCAPE
                    )
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
        raise typer.Exit(1)
    if len(rows) > 1:
        typer.echo(
            f"Entry prefix {entry_id_prefix!r} is ambiguous ({len(rows)} matches).",
            err=True,
        )
        raise typer.Exit(1)
    return rows[0]


async def _corpus_retract(
    entry_id_prefix: str, reason: str | None, dry_run: bool, yes: bool
) -> None:
    from particles.api.client import get_backend

    backend = get_backend()
    # Prefix resolution reads the local store ⇒ local-only; the
    # engine /corpus/{id}/retract endpoint takes a full entry id.
    if backend.remote:
        entry_id = entry_id_prefix
    else:
        async with session_scope() as session:
            entry_id = await _resolve_entry_id(session, entry_id_prefix)

    entry = await backend.corpus_show(entry_id)
    if entry is None:
        typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
        raise typer.Exit(1)
    entry_id = entry.entry_id

    plan = await backend.corpus_retract(entry_id, reason=reason, dry_run=True)
    label = (entry.uri_r or entry_id).replace("file://", "")
    typer.echo(f"Entry:     {entry_id[:8]}…  {label}")
    typer.echo(f"Snapshots: {len(entry.snapshots)} (preserved)")
    typer.echo(f"Retract:   {len(plan.retracted_ids)} live particle(s)")
    for pid in plan.retracted_ids:
        typer.echo(f"  {pid[:8]}…  → RETRACTED")
    if plan.skipped:
        skip_desc = ", ".join(f"{n} {s}" for s, n in sorted(plan.skipped.items()))
        typer.echo(f"Skipped:   {skip_desc} (not live)")

    if dry_run:
        typer.echo("Dry run — no changes written.")
        return
    if not plan.retracted_ids:
        typer.echo("Nothing to retract.")
        return
    if not yes:
        typer.confirm(
            f"Retract {len(plan.retracted_ids)} particle(s) from this source?",
            abort=True,
        )

    result = await backend.corpus_retract(entry_id, reason=reason, dry_run=False)

    typer.echo(
        f"Retracted {len(result.retracted_ids)} particle(s) from {entry_id[:8]}… "
        f"(corpus + snapshots preserved). Run `particles lint` to cascade."
    )


# ---------------------------------------------------------------------------
# Orphan cleanup — shared by ``corpus delete`` and ``corpus prune-orphans``.
# None of the index tables (particle_subjects, particle_tag_edges,
# particle_relations, synthesis_cache) declare a foreign key, and SQLite FK
# enforcement is off, so deleting particles or subjects elsewhere leaves
# dangling rows unless they are swept explicitly here.
# ---------------------------------------------------------------------------


async def _purge_particle_index_rows(session: AsyncSession, particle_ids: list[str]) -> None:
    """Delete every index row keyed on one of ``particle_ids``.

    Sweeps the ``particle_subjects`` join, ``particle_tag_edges``, and
    ``particle_relations`` (either endpoint). Caller deletes the
    ``ParticleRow`` rows themselves.
    """
    from sqlalchemy import delete, or_

    from particles.store.relation_store import ParticleRelationRow
    from particles.store.subject_store import ParticleSubjectRow
    from particles.store.taxonomy_store import ParticleTagEdgeRow

    if not particle_ids:
        return
    await session.execute(
        delete(ParticleSubjectRow).where(ParticleSubjectRow.particle_id.in_(particle_ids))
    )
    await session.execute(
        delete(ParticleTagEdgeRow).where(ParticleTagEdgeRow.particle_id.in_(particle_ids))
    )
    await session.execute(
        delete(ParticleRelationRow).where(
            or_(
                ParticleRelationRow.particle_a.in_(particle_ids),
                ParticleRelationRow.particle_b.in_(particle_ids),
            )
        )
    )


async def _purge_orphan_subjects(
    session: AsyncSession, candidate_ids: set[str] | None
) -> tuple[int, int]:
    """Delete subjects with no remaining ``particle_subjects`` link.

    Also drops any ``synthesis_cache`` rows keyed on the removed subjects.
    Returns ``(subjects_removed, synthesis_rows_removed)``.

    ``candidate_ids`` restricts the orphan check to a known set — the subjects
    that were linked to a just-deleted entry's particles — so a subject still
    linked to surviving particles is never touched. Pass ``None`` to scan
    every subject (the whole-DB ``prune-orphans`` sweep).
    """
    from sqlalchemy import delete, select
    from sqlalchemy.engine import CursorResult

    from particles.store.subject_store import ParticleSubjectRow, SubjectRow
    from particles.store.synthesis_cache_store import SynthesisCacheRow

    if candidate_ids is not None and not candidate_ids:
        return (0, 0)

    linked = select(ParticleSubjectRow.subject_id).distinct()
    stmt = select(SubjectRow.id).where(SubjectRow.id.not_in(linked))
    if candidate_ids is not None:
        stmt = stmt.where(SubjectRow.id.in_(candidate_ids))
    orphan_ids = list((await session.execute(stmt)).scalars())
    if not orphan_ids:
        return (0, 0)

    synth_result: CursorResult[None] = await session.execute(  # type: ignore[assignment]
        delete(SynthesisCacheRow).where(SynthesisCacheRow.subject_id.in_(orphan_ids))
    )
    await session.execute(delete(SubjectRow).where(SubjectRow.id.in_(orphan_ids)))
    return (len(orphan_ids), synth_result.rowcount or 0)


@corpus_app.command("prune-orphans")
def corpus_prune_orphans_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Sweep dangling rows left by older deletes (orphan subjects, stale index rows).

    Pre-fix ``corpus delete`` removed particles but not the index rows keyed
    on them, so long-lived DBs accumulate ``particle_subjects`` /
    ``particle_tag_edges`` / ``particle_relations`` rows pointing at deleted
    particles, subjects with no remaining link, and ``synthesis_cache`` rows
    keyed on vanished subjects. This one-off verb cleans all of them.
    """
    run(_corpus_prune_orphans(yes))


async def _corpus_prune_orphans(yes: bool) -> None:
    from sqlalchemy import delete, func, or_, select
    from sqlalchemy.engine import CursorResult

    from particles.api.cli._remote import ensure_local
    from particles.store.particle_store import ParticleRow
    from particles.store.relation_store import ParticleRelationRow
    from particles.store.subject_store import ParticleSubjectRow, SubjectRow
    from particles.store.synthesis_cache_store import SynthesisCacheRow
    from particles.store.taxonomy_store import ParticleTagEdgeRow

    ensure_local("corpus prune-orphans")

    async with session_scope() as session:
        live_particles = select(ParticleRow.id)
        live_subjects = select(SubjectRow.id)
        linked_subjects = select(ParticleSubjectRow.subject_id).distinct()

        async def _count(stmt: Any) -> int:
            return int((await session.execute(stmt)).scalar_one())

        dangling_join_q = ParticleSubjectRow.particle_id.not_in(live_particles)
        dangling_tag_q = ParticleTagEdgeRow.particle_id.not_in(live_particles)
        dangling_rel_q = or_(
            ParticleRelationRow.particle_a.not_in(live_particles),
            ParticleRelationRow.particle_b.not_in(live_particles),
        )
        orphan_subj_q = SubjectRow.id.not_in(linked_subjects)
        dangling_synth_q = SynthesisCacheRow.subject_id.not_in(live_subjects)

        n_join = await _count(
            select(func.count()).select_from(ParticleSubjectRow).where(dangling_join_q)
        )
        n_tag = await _count(
            select(func.count()).select_from(ParticleTagEdgeRow).where(dangling_tag_q)
        )
        n_rel = await _count(
            select(func.count()).select_from(ParticleRelationRow).where(dangling_rel_q)
        )
        n_subj = await _count(select(func.count()).select_from(SubjectRow).where(orphan_subj_q))
        n_synth = await _count(
            select(func.count()).select_from(SynthesisCacheRow).where(dangling_synth_q)
        )

        total = n_join + n_tag + n_rel + n_subj + n_synth
        typer.echo("Orphaned rows found:")
        typer.echo(f"  particle_subjects (dangling): {n_join}")
        typer.echo(f"  particle_tag_edges (dangling): {n_tag}")
        typer.echo(f"  particle_relations (dangling): {n_rel}")
        typer.echo(f"  subjects (no link):            {n_subj}")
        typer.echo(f"  synthesis_cache (dangling):    {n_synth}")

        if total == 0:
            typer.echo("Nothing to prune.")
            return

        if not yes:
            typer.confirm(f"Delete {total} orphaned rows?", abort=True)

        await session.execute(delete(ParticleSubjectRow).where(dangling_join_q))
        await session.execute(delete(ParticleTagEdgeRow).where(dangling_tag_q))
        await session.execute(delete(ParticleRelationRow).where(dangling_rel_q))
        # Orphan-subject removal must run *after* the dangling join rows are
        # gone, so a subject linked only via a now-deleted row is detected.
        subj_removed, synth_from_subj = await _purge_orphan_subjects(session, None)
        synth_res: CursorResult[None] = await session.execute(  # type: ignore[assignment]
            delete(SynthesisCacheRow).where(dangling_synth_q)
        )
        await session.commit()

    synth_total = synth_from_subj + (synth_res.rowcount or 0)
    typer.echo(
        f"Pruned: {n_join} join, {n_tag} tag, {n_rel} relation rows; "
        f"{subj_removed} subjects; {synth_total} synthesis-cache rows."
    )


@corpus_app.command("refresh")
def corpus_refresh_cmd(
    entry_id: str | None = typer.Argument(
        None, help="Refresh one entry (full id or unambiguous prefix). Omit to sweep all."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Tier 3: re-check regardless of fetch_policy and the per-source-type "
            "re-fetch floor. The escape hatch for a content change that preserved "
            "the file's mtime."
        ),
    ),
    backfill_cascade: bool = typer.Option(
        False,
        "--backfill-cascade",
        help=(
            "Instead of re-checking sources, apply the generation "
            "cascade to MUTABLE entries whose snapshots already moved — demoting "
            "ACTIVE particles anchored to a superseded snapshot. Stores that "
            "predate the change carry a backlog of these; the forward-looking "
            "cascade only fires on newly-extracted snapshots."
        ),
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Re-check deposited local sources against the files on disk.

    Walks every ``LAZY`` entry with a ``file://`` URI-R, comparing the file's
    mtime and then its SHA-256 against the latest snapshot. A changed file gets
    a new PENDING snapshot; ``particles extract --all-pending`` (or tonight's
    consolidation run) turns that into current beliefs.

    This is the on-demand form of consolidation pass 0.5 — the scheduled cycle
    runs the same sweep nightly.
    """
    configure_logging(verbose, False)
    run(_corpus_refresh(entry_id, force, backfill_cascade, yes))


async def _corpus_refresh_backfill(yes: bool) -> None:
    """The §2 backlog sweep: dry-run, confirm, then apply."""
    from particles.api.cli._remote import ensure_local
    from particles.ingest.generation import backfill_superseded_generations

    ensure_local("corpus refresh --backfill-cascade")

    async with session_scope() as session:
        preview = await backfill_superseded_generations(session, dry_run=True)

    typer.echo(
        f"MUTABLE entries with more than one snapshot: {preview.entries_scanned}"
        + (
            f" ({preview.entries_unextracted} have no extracted snapshot yet — skipped)"
            if preview.entries_unextracted
            else ""
        )
    )
    if preview.demoted == 0:
        typer.echo("No particles are anchored to a superseded snapshot. Nothing to do.")
        return

    typer.echo(
        f"{preview.demoted} ACTIVE particle(s) across {preview.entries_affected} "
        f"entr{'y' if preview.entries_affected == 1 else 'ies'} are anchored to a "
        f"superseded snapshot:"
    )
    for entry_id, uri_r, count in preview.per_entry[:10]:
        label = uri_r or entry_id
        typer.echo(f"  {count:>6}  {label[-72:]}")
    if len(preview.per_entry) > 10:
        typer.echo(f"  … and {len(preview.per_entry) - 10} more entries")
    typer.echo(
        "\nThey will be demoted ACTIVE → PROVENANCE_STALE. This is a demotion, not a "
        "delete: content, provenance and confidence are kept, the particles surface "
        "in `particles curate`, and the change is reversible."
    )

    if not yes:
        typer.confirm(f"Demote {preview.demoted} particle(s)?", abort=True)

    async with session_scope(write=True) as session:
        applied = await backfill_superseded_generations(session, dry_run=False)
        await session.commit()
    typer.echo(
        f"Demoted {applied.demoted} particle(s) across {applied.entries_affected} "
        f"entr{'y' if applied.entries_affected == 1 else 'ies'}."
    )


async def _corpus_refresh(
    entry_id: str | None, force: bool, backfill_cascade: bool, yes: bool
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.core.schema import WarcRecordType
    from particles.corpus.fetch import maybe_refetch
    from particles.corpus.store import (
        list_refreshable_local_entries,
        list_snapshots_for_entry,
        resolve_entry_id,
    )

    if backfill_cascade:
        await _corpus_refresh_backfill(yes)
        return

    ensure_local("corpus refresh")

    async with session_scope(write=True) as session:
        if entry_id is not None:
            resolved = await resolve_entry_id(session, entry_id)
            if resolved is None:
                typer.echo(f"Error: no corpus entry matches {entry_id!r}", err=True)
                raise typer.Exit(1)
            targets = [(resolved, "")]
        else:
            targets = await list_refreshable_local_entries(session)

        if not targets:
            typer.echo(
                "No refreshable local sources. An entry is refreshable when it has a "
                "file:// URI-R and fetch_policy=LAZY — deposit with "
                "`--fetch-policy LAZY --mutability MUTABLE` to opt a file in."
            )
            return

        changed = unchanged = missing = 0
        for target_id, _uri in targets:
            snapshots = await list_snapshots_for_entry(session, target_id)
            before = max(snapshots, key=lambda s: s.captured_at).snapshot_id if snapshots else None
            try:
                snap = await maybe_refetch(session, target_id, force=force)
            except Exception as exc:  # noqa: BLE001 — one bad source must not stop the sweep
                typer.echo(f"  {target_id[:8]}  error: {exc}", err=True)
                await session.rollback()
                continue
            if snap is None:
                missing += 1
                typer.echo(f"  {target_id[:8]}  missing (source unavailable)")
            elif snap.snapshot_id == before or snap.warc_record_type is WarcRecordType.REVISIT:
                unchanged += 1
            else:
                changed += 1
                typer.echo(f"  {target_id[:8]}  changed → snapshot {snap.snapshot_id[:8]} PENDING")
            await session.commit()

    typer.echo(
        f"\nChecked {len(targets)} local source(s): {changed} changed, "
        f"{unchanged} unchanged, {missing} missing."
    )
    if changed:
        typer.echo("Run `particles extract --all-pending` to turn the new snapshots into beliefs.")


@corpus_app.command("fsck")
def corpus_fsck_cmd(
    search: list[Path] = typer.Option(
        [],
        "--search",
        help=(
            "Also look for strays under this blob root — the directory holding the "
            "two-character shards (repeatable). Nothing is inferred: the audit tells "
            "you what is missing so you can point --search at where you think it went."
        ),
    ),
    re_home: bool = typer.Option(
        False,
        "--re-home",
        help="Copy digest-verified strays found under --search into the blob dir.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what --re-home would copy, without copying."
    ),
) -> None:
    """Audit every blob the store references; optionally re-home the strays.

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
    """
    run(_corpus_fsck(search, re_home, dry_run))


async def _corpus_fsck(search: list[Path], re_home: bool, dry_run: bool) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.config import get_config
    from particles.corpus.blob_fsck import audit_blobs, rehome_strays
    from particles.corpus.blob_health import store_file_missing

    ensure_local("corpus fsck")

    # Check before opening: `session_scope()` on an absent SQLite path *creates*
    # an empty database, so a diagnostic run from the wrong directory would both
    # litter a stray store and then report "0 blobs, healthy" — a false all-clear
    # on the exact question being asked. Same guard `config validate` uses.
    database_url = get_config().storage.database_url
    if store_file_missing(database_url):
        typer.echo(
            f"No store at {database_url} — nothing to audit, and opening it would "
            "create an empty database here. If you expected a store, you are "
            "probably running from the wrong directory or with the wrong "
            "DATABASE_URL; `particles config validate` prints what resolves.",
            err=True,
        )
        raise typer.Exit(1)

    async with session_scope() as session:
        report = await audit_blobs(session, search_dirs=search)

    typer.echo(f"Blob dir:  {report.blob_dir}")
    for directory in report.search_dirs:
        typer.echo(f"Search:    {directory}")
    typer.echo(f"\nBlobs referenced by the store: {report.total}")
    typer.echo(f"  present:         {len(report.present)}")
    typer.echo(f"  found elsewhere: {len(report.elsewhere)}")
    typer.echo(f"  missing:         {len(report.missing)}")

    _echo_fsck_rejections(report.rejected)

    if report.healthy:
        typer.echo("\nEvery referenced blob is where extraction will look for it.")
        return

    if report.elsewhere and not (re_home or dry_run):
        typer.echo(
            f"\n{len(report.elsewhere)} blob(s) are recoverable — re-run with --re-home "
            "to copy them into the blob dir (the source tree is left untouched)."
        )
    elif re_home or dry_run:
        outcome = rehome_strays(report, dry_run=dry_run)
        _echo_rehome_outcome(outcome)

    _echo_fsck_missing(report.missing)
    raise typer.Exit(1)


def _echo_fsck_rejections(rejected: Sequence[RejectedBlob]) -> None:
    """Report candidates found at the right path whose bytes hash to something else."""
    if not rejected:
        return
    typer.echo(f"\nRejected {len(rejected)} candidate(s) — digest does not match the filename:")
    for bad in rejected:
        typer.echo(f"  {bad.ref.content_hash[:12]}…  {bad.source}")
        typer.echo(f"    hashes to {bad.actual_digest[:12]}… — not copied.")


def _echo_rehome_outcome(outcome: RehomeOutcome) -> None:
    """Report what the re-home copied, or (dry run) what it would copy."""
    if outcome.dry_run:
        typer.echo(f"\nDry run — would copy {len(outcome.copied)} blob(s) into the blob dir:")
        for stray in outcome.copied:
            typer.echo(f"  {stray.ref.content_hash[:12]}…  ← {stray.source}")
        typer.echo("Dry run — no files were written.")
        return

    typer.echo(f"\nCopied {len(outcome.copied)} blob(s) into the blob dir.")
    for stray, error in outcome.failed:
        typer.echo(f"  FAILED {stray.ref.content_hash[:12]}…  {stray.source}: {error}", err=True)


def _echo_fsck_missing(missing: Sequence[BlobRef]) -> None:
    """Report the unrecoverable remainder with enough identity to act on it."""
    if not missing:
        return
    typer.echo(
        f"\n{len(missing)} blob(s) could not be found. The rows survive as pointers to "
        "content that is not on disk; extraction of those snapshots will fail with "
        "`Blob not found for hash …`."
    )
    for ref in missing:
        typer.echo(f"  {ref.content_hash[:12]}…  {ref.label[-64:]}")
        typer.echo(f"    entries: {', '.join(e[:8] for e in ref.entry_ids)}")
    typer.echo(
        "\nNothing was written to the database. Re-deposit each source to recover it, "
        "or `particles corpus retract <entry>` to retire the beliefs it carries."
    )


# ---------------------------------------------------------------------------
# corpus links — audit follow-edges written by deposit-time URL following
#. The edges are write-only today; this verb is the first
# downstream consumer, surfacing them so operators can see the curation
# graph their corpus actually carries.
# ---------------------------------------------------------------------------

corpus_links_app = typer.Typer(
    help="Inspect cross-entry follow edges.", no_args_is_help=True
)
corpus_app.add_typer(corpus_links_app, name="links")


@corpus_links_app.command("list")
def corpus_links_list_cmd(
    entry_id: str | None = typer.Argument(
        None, help="Entry ID (prefix OK). Omit to list every follow edge."
    ),
    direction: str = typer.Option(
        "both",
        "--direction",
        help="Filter when entry-id set: out (this→linked), in (others→this), both.",
    ),
) -> None:
    """List follow edges written by deposit-time URL following.

    Without an entry-id, lists every edge in deposit-time order. With an
    entry-id, shows the edges touching that entry (outgoing / incoming
    per ``--direction``). Operators use this to audit which Reddit / HN
    / Mastodon posts amplified which external sources.
    """
    if direction not in ("out", "in", "both"):
        typer.echo(f"Invalid --direction {direction!r}; use out/in/both.", err=True)
        raise typer.Exit(1)
    run(_corpus_links_list(entry_id, direction))


async def _corpus_links_list(entry_id_prefix: str | None, direction: str) -> None:
    from sqlalchemy import select

    from particles.api.cli._remote import ensure_local
    from particles.corpus.follow_edges import (
        CorpusFollowEdgeRow,
        get_follow_sources,
        get_follow_targets,
    )
    from particles.corpus.store import CorpusEntryRow

    ensure_local("corpus links list")

    async with session_scope() as session:
        # Build entry-id → uri label for nicer output. Cheap on small DBs;
        # one query, one dict.
        uri_rows = await session.execute(select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r))
        label_map: dict[str, str] = {}
        for eid, uri in uri_rows.all():
            label_map[eid] = (uri or "").replace("file://", "")

        def _label(entry_id: str) -> str:
            uri = label_map.get(entry_id, "")
            short = uri[-60:] if len(uri) > 60 else uri
            return f"{entry_id[:8]}…  {short}".rstrip()

        if entry_id_prefix is None:
            result = await session.execute(
                select(CorpusFollowEdgeRow).order_by(CorpusFollowEdgeRow.discovered_at)
            )
            edges = list(result.scalars())
            if not edges:
                typer.echo("No follow edges recorded.")
                return
            typer.echo(f"{'VIA':10}  {'TARGET':10}  {'TYPE':12}  DISCOVERED")
            typer.echo("-" * 80)
            for e in edges:
                typer.echo(
                    f"{e.via_entry_id[:8]}…  {e.target_entry_id[:8]}…  "
                    f"{e.link_type:12s}  {e.discovered_at.date().isoformat()}"
                )
            return

        # Resolve prefix → full entry id.
        if len(entry_id_prefix) < 36:
            entry_result = await session.execute(
                select(CorpusEntryRow).where(
                    CorpusEntryRow.entry_id.like(
                        f"{escape_like_pattern(entry_id_prefix)}%", escape=LIKE_ESCAPE
                    )
                )
            )
            entry_row = entry_result.scalar_one_or_none()
            if entry_row is None:
                typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
                raise typer.Exit(1)
            entry_id = entry_row.entry_id
        else:
            entry_id = entry_id_prefix

        typer.echo(f"Entry:  {_label(entry_id)}")

        if direction in ("out", "both"):
            out_edges = await get_follow_targets(session, entry_id)
            typer.echo(f"\nOutgoing follows ({len(out_edges)}):")
            if not out_edges:
                typer.echo("  (none)")
            for e in out_edges:
                typer.echo(
                    f"  → {_label(e.target_entry_id)}"
                    f"  [{e.link_type}, {e.discovered_at.date().isoformat()}]"
                )

        if direction in ("in", "both"):
            in_edges = await get_follow_sources(session, entry_id)
            typer.echo(f"\nIncoming follows ({len(in_edges)}):")
            if not in_edges:
                typer.echo("  (none)")
            for e in in_edges:
                typer.echo(
                    f"  ← {_label(e.via_entry_id)}"
                    f"  [{e.link_type}, {e.discovered_at.date().isoformat()}]"
                )


# ---------------------------------------------------------------------------
# corpus links suggest / dismiss — citation-signal deposit suggestions
#. Rank URLs the corpus cites but has not deposited, so the
# operator can ground hearsay in the primary source. Suggestion-only: nothing
# is fetched or deposited here.
# ---------------------------------------------------------------------------


@corpus_links_app.command("suggest")
def corpus_links_suggest_cmd(
    limit: int | None = typer.Option(
        None, "--limit", help="Max suggestions to show (default: config rank_cap)."
    ),
    min_sources: int | None = typer.Option(
        None, "--min-sources", help="Min distinct citing sources to surface (default: config)."
    ),
    output_format: str = typer.Option("table", "--output-format", "-o", help="table | json"),
) -> None:
    """Suggest undeposited URLs the corpus frequently cites.

    Ranks URLs mentioned across the corpus but not yet deposited, by
    trust-weighted distinct-source diversity × recency. Suggestion-only —
    nothing is fetched or crawled. Deposit one with ``particles deposit <url>``;
    silence one with ``particles corpus links dismiss <url>``.
    """
    if output_format not in ("table", "json"):
        typer.echo(f"Invalid --output-format {output_format!r}; use table/json.", err=True)
        raise typer.Exit(1)
    run(_corpus_links_suggest(limit, min_sources, output_format))


async def _corpus_links_suggest(
    limit: int | None, min_sources: int | None, output_format: str
) -> None:
    from particles.api.client import get_backend

    report = await get_backend().corpus_links_suggest(limit=limit, min_sources=min_sources)

    if output_format == "json":
        typer.echo(report.model_dump_json(indent=2))
        return

    if not report.suggestions:
        typer.echo(
            "No deposit suggestions — nothing is cited by enough distinct sources yet, "
            "or every candidate is already deposited or dismissed."
        )
        return

    capped = " (rank-capped)" if report.capped else ""
    typer.echo(
        f"Showing {len(report.suggestions)} of {report.total_candidates} suggestion(s){capped}"
    )
    typer.echo(f"{'SCORE':>7}  {'SRCS':>4}  {'LAST SEEN':10}  URL")
    typer.echo("-" * 80)
    for s in report.suggestions:
        typer.echo(
            f"{s.score:7.2f}  {s.distinct_sources:>4}  "
            f"{s.most_recent.date().isoformat():10}  {s.canonical_url}"
        )
    typer.echo("\nDeposit one with:  particles deposit <url>")
    typer.echo("Dismiss one with:  particles corpus links dismiss <url> [--snooze DAYS]")


@corpus_links_app.command("dismiss")
def corpus_links_dismiss_cmd(
    url: str = typer.Argument(..., help="URL to dismiss (canonicalized before matching)."),
    snooze: int | None = typer.Option(
        None, "--snooze", help="Snooze for N days instead of a permanent dismiss."
    ),
) -> None:
    """Stop a URL from resurfacing in ``corpus links suggest``.

    A permanent dismiss (the default) suppresses the URL indefinitely; ``--snooze
    N`` suppresses it for N days. The action is audited in the operator event
    log.
    """
    run(_corpus_links_dismiss(url, snooze))


async def _corpus_links_dismiss(url: str, snooze: int | None) -> None:
    from particles.api.client import get_backend
    from particles.url_canonical import canonicalize_url

    # Canonicalisation is pure (no store), so validate up front in both modes
    # for the same error message; the dismiss itself routes.
    canon = canonicalize_url(url)
    if canon is None:
        typer.echo(f"Not a usable http(s) URL: {url!r}", err=True)
        raise typer.Exit(1)
    outcome = await get_backend().corpus_links_dismiss(url=canon, snooze_days=snooze)
    if snooze:
        typer.echo(
            f"Snoozed {outcome.canonical_url} for {snooze} day(s) "
            f"(until {outcome.suppressed_until.date().isoformat()})."
        )
    else:
        typer.echo(
            f"Dismissed {outcome.canonical_url}; it will no longer appear in "
            f"`corpus links suggest`."
        )
