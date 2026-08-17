"""particle sub-Typer — inspect individual extracted particles by ID.

Read/tag-only except for ``retract``, the one narrow *operator*
mutation: it retires a single belief under operator authority, which is the
escape hatch the cross-asserter guardrail deliberately withholds
from agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from particles.api.cli import app, run
from particles.db import DEFAULT_STORE, session_scope
from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.core.schema import Particle

particle_app = typer.Typer(help="Inspect individual extracted particles.", no_args_is_help=True)
app.add_typer(particle_app, name="particle")


@particle_app.command("show")
def particle_show_cmd(
    particle_id: str = typer.Argument(..., help="Particle ID (prefix OK — first 8 chars)"),
) -> None:
    """Show one particle's content, status, confidence, subjects, and source URL."""
    run(_particle_show(particle_id))


async def _particle_show(id_prefix: str) -> None:
    from particles.api.cli._id_norm import normalise_particle_id
    from particles.api.client import get_backend

    backend = get_backend()
    norm = normalise_particle_id(id_prefix)

    # Prefix resolution reads the local store, so it is local-only;
    # the engine takes full UUIDs. The single-particle fetch routes through the
    # backend in both modes.
    if not backend.remote:
        async with session_scope() as session:
            norm = await _resolve_show_prefix(session, norm, id_prefix)

    particle = await backend.particle_show(norm)
    if particle is None:
        typer.echo(f"Particle {id_prefix!r} not found.", err=True)
        raise typer.Exit(1)

    _render_particle_header(particle)

    # Subject names, provenance URIs, and narrative connective tissue
    # require extra store lookups the single-particle endpoint does not serve,
    # so they render local-only and degrade gracefully in remote mode.
    if not backend.remote:
        async with session_scope() as session:
            await _render_particle_detail(session, particle)


async def _resolve_show_prefix(session: AsyncSession, norm: str, original: str) -> str:
    """Resolve a (possibly truncated) particle id to a full id for ``particle show``."""
    from sqlalchemy import select

    from particles.store.particle_store import ParticleRow

    if len(norm) >= 36:
        return norm
    result = await session.execute(
        select(ParticleRow.id).where(
            ParticleRow.id.like(f"{escape_like_pattern(norm)}%", escape=LIKE_ESCAPE)
        )
    )
    rows = result.scalars().all()
    if not rows:
        typer.echo(f"No particle matches prefix {original!r}.", err=True)
        raise typer.Exit(1)
    if len(rows) > 1:
        typer.echo(f"Ambiguous prefix {original!r} matches {len(rows)} particles:", err=True)
        for r in rows[:10]:
            typer.echo(f"  {r}", err=True)
        raise typer.Exit(1)
    return str(rows[0])


def _render_particle_header(particle: Particle) -> None:
    """Render the transport-agnostic header block (works local + remote)."""
    typer.echo(f"ID:           {particle.id}")
    typer.echo(f"Status:       {particle.status}")
    typer.echo(f"Type:         {particle.particle_type}")
    typer.echo(f"Modality:     {particle.assertion_modality}")
    calib = particle.confidence.calibration_source.value
    typer.echo(f"Confidence:   {particle.confidence.value:.2f} ({calib})")
    typer.echo(f"Uncertainty:  {particle.uncertainty_nature}")
    typer.echo(f"Asserted by:  {particle.asserted_by}")
    typer.echo(f"Asserted at:  {particle.asserted_at.isoformat()}")
    if particle.extractor_ref:
        ext_name = particle.extractor_ref.name
        ext_version = particle.extractor_ref.version
        typer.echo(f"Extractor:    {ext_name} v{ext_version}")
    # Sibling of extractor_ref, not a key inside it — and
    # independently absent: a superseding agent write carries the pairing over
    # with no extractor_ref of its own, while a pre-0229 particle legally has
    # neither. Its own line keeps all four combinations readable.
    if particle.extraction_provider_model:
        typer.echo(f"Model:        {particle.extraction_provider_model}")
    if particle.context_fingerprint:
        typer.echo(f"Fingerprint:  {particle.context_fingerprint[:16]}…")
    typer.echo("")
    typer.echo("Content:")
    typer.echo(f"  {particle.content}")
    typer.echo("")


async def _render_particle_detail(session: AsyncSession, particle: Particle) -> None:
    """Render subjects / provenance / narrative blocks (local-only enrichment)."""
    from particles.core.schema import ParticleType
    from particles.corpus.store import CorpusEntryRow
    from particles.operations.narrative import (
        get_narrative_sequence,
        get_narratives_containing,
    )
    from particles.store.subject_store import SubjectRow

    if particle.subject_ids:
        typer.echo("Subjects:")
        for sid in particle.subject_ids:
            srow = await session.get(SubjectRow, sid)
            name = srow.canonical_name if srow else "(unknown)"
            typer.echo(f"  {sid[:8]}…  {name}")
        typer.echo("")

    if particle.provenance:
        typer.echo("Provenance:")
        for ref in particle.provenance:
            eid = ref.corpus_entry_id
            if not eid:
                typer.echo(f"  {ref.type}  (no corpus entry)")
                continue
            entry = await session.get(CorpusEntryRow, eid)
            uri = entry.uri_r if entry else None
            source_type = entry.source_type if entry else "?"
            typer.echo(f"  {ref.type}  {eid[:8]}…  ({source_type})")
            if uri:
                typer.echo(f"            URL: {uri}")
            if ref.snapshot_id:
                typer.echo(f"            Snapshot: {ref.snapshot_id[:8]}…")

    containing = await get_narratives_containing(session, particle.id)
    if containing:
        typer.echo("")
        typer.echo("Part of narratives:")
        for nar in containing:
            typer.echo(f"  {nar.id[:8]}…  {nar.content}")

    if particle.particle_type == ParticleType.NARRATIVE:
        sequence = await get_narrative_sequence(session, particle.id)
        typer.echo("")
        typer.echo(f"Narrative constituents ({len(sequence)}, in sequence):")
        for i, member in enumerate(sequence, start=1):
            preview = member.content if len(member.content) <= 70 else member.content[:67] + "…"
            typer.echo(f"  {i}. {member.id[:8]}…  {preview}")


@particle_app.command("narrative")
def particle_narrative_cmd(
    particle_id: str = typer.Argument(..., help="NARRATIVE particle ID (prefix OK — ≥ 8 chars)"),
    synthesize: bool = typer.Option(
        False,
        "--synthesize",
        help="Render the narrative as one cited prose article instead "
        "of listing its constituents.",
    ),
) -> None:
    """Show a NARRATIVE particle's constituents in SEQUENCE_IN order.

    With ``--synthesize``, render the narrative as one cited prose article by
    traversing its SEQUENCE_IN chain — the same synthesis engine the
    wiki/Obsidian exporters use, here scoped to a single narrative.
    """
    run(_particle_narrative(particle_id, synthesize=synthesize))


async def _particle_narrative(id_prefix: str, *, synthesize: bool = False) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.core.schema import ParticleType
    from particles.operations.narrative import get_narrative_sequence
    from particles.store.particle_store import get_particle

    ensure_local("particle narrative")

    async with session_scope() as session:
        narrative_id = await _resolve_particle_id(session, id_prefix)
        narrative = await get_particle(session, narrative_id)
        if narrative is None:
            typer.echo(f"Particle {narrative_id!r} not found.", err=True)
            raise typer.Exit(1)
        if narrative.particle_type != ParticleType.NARRATIVE:
            typer.echo(
                f"Particle {narrative_id[:8]}… is {narrative.particle_type}, not NARRATIVE. "
                f"`particle narrative` only applies to NARRATIVE particles.",
                err=True,
            )
            raise typer.Exit(1)

        if synthesize:
            from particles.operations.narrative_synthesis import synthesize_narrative

            article = await synthesize_narrative(session, narrative_id)
            if article is None or not article.constituents:
                typer.echo(
                    "No constituents yet — nothing to synthesize. Link claims with:\n"
                    f"  particles links add <claim-id> {narrative_id[:8]} --type part-of"
                )
                return
            typer.echo(article.body)
            if not article.used_synthesis:
                typer.echo(
                    "\n(structured-listing fallback — no LLM synthesis; "
                    "set ANTHROPIC_API_KEY for prose)",
                    err=True,
                )
            return

        typer.echo(f"Narrative:  {narrative.content}")
        typer.echo(f"ID:         {narrative.id}")
        typer.echo("")

        sequence = await get_narrative_sequence(session, narrative_id)
        if not sequence:
            typer.echo(
                "No constituents yet. Link claims into this narrative with:\n"
                f"  particles links add <claim-id> {narrative_id[:8]} --type part-of\n"
                "and order them with `--type sequence-in` between adjacent claims."
            )
            return
        typer.echo(f"Constituents ({len(sequence)}, in sequence):")
        for i, member in enumerate(sequence, start=1):
            preview = member.content if len(member.content) <= 80 else member.content[:77] + "…"
            typer.echo(f"  {i}. {member.id[:8]}…  {preview}")


@particle_app.command("tag")
def particle_tag_cmd(
    particle_id: str = typer.Argument(..., help="Particle ID (prefix OK)"),
    tag: list[str] = typer.Option(
        ..., "--tag", help="Tag path to add (repeatable, e.g. coins/germany)"
    ),
    force: bool = typer.Option(
        False, "--force", help="Allow tags that aren't in any active taxonomy"
    ),
    supersede: bool = typer.Option(
        False,
        "--supersede",
        help="Reserved for the immutable-revision audit trail (not yet implemented)",
    ),
) -> None:
    """Add taxonomy tags to a particle."""
    if not tag:
        typer.echo("Provide at least one --tag value.", err=True)
        raise typer.Exit(1)
    if supersede:
        typer.echo(
            "--supersede is reserved for Phase B; Phase A edits in place only.",
            err=True,
        )
        raise typer.Exit(2)
    run(_particle_tag(particle_id, list(tag), force=force))


@particle_app.command("untag")
def particle_untag_cmd(
    particle_id: str = typer.Argument(..., help="Particle ID (prefix OK)"),
    tag: list[str] = typer.Option(..., "--tag", help="Tag path to remove (repeatable)"),
    supersede: bool = typer.Option(
        False,
        "--supersede",
        help="Reserved for the immutable-revision audit trail (not yet implemented)",
    ),
) -> None:
    """Remove taxonomy tags from a particle."""
    if not tag:
        typer.echo("Provide at least one --tag value.", err=True)
        raise typer.Exit(1)
    if supersede:
        typer.echo(
            "--supersede is reserved for Phase B; Phase A edits in place only.",
            err=True,
        )
        raise typer.Exit(2)
    run(_particle_untag(particle_id, list(tag)))


async def _resolve_particle_id(session: AsyncSession, id_prefix: str) -> str:
    """Resolve a (possibly truncated) particle ID to a full UUID, or exit."""
    from sqlalchemy import select

    from particles.api.cli._id_norm import normalise_particle_id
    from particles.store.particle_store import ParticleRow

    id_prefix = normalise_particle_id(id_prefix)
    if len(id_prefix) == 36:
        return id_prefix
    result = await session.execute(
        select(ParticleRow.id).where(
            ParticleRow.id.like(f"{escape_like_pattern(id_prefix)}%", escape=LIKE_ESCAPE)
        )
    )
    ids = list(result.scalars())
    if not ids:
        typer.echo(f"No particle matches prefix {id_prefix!r}.", err=True)
        raise typer.Exit(1)
    if len(ids) > 1:
        typer.echo(
            f"Ambiguous prefix {id_prefix!r} matches {len(ids)} particles.",
            err=True,
        )
        raise typer.Exit(1)
    return ids[0]


async def _particle_tag(id_prefix: str, tags: list[str], *, force: bool) -> None:
    from particles.api.cli._id_norm import normalise_particle_id
    from particles.api.client import get_backend

    backend = get_backend()
    if backend.remote:
        # Prefix resolution + taxonomy validation read the local store, so they
        # are local-only; the engine takes full UUIDs and the
        # /particles/{id}/tags endpoint adds without taxonomy validation.
        particle_id = normalise_particle_id(id_prefix)
    else:
        from particles.store.taxonomy_store import tag_exists

        async with session_scope() as session:
            particle_id = await _resolve_particle_id(session, id_prefix)
            if not force:
                for t in tags:
                    if not await tag_exists(session, t):
                        typer.echo(
                            f"Tag {t!r} is not defined in any active taxonomy. "
                            f"Pass --force to apply ad-hoc tags.",
                            err=True,
                        )
                        raise typer.Exit(1)
    added = await backend.particle_tag(particle_id, tags)
    if added:
        typer.echo(f"Tagged {particle_id[:8]}… with {', '.join(added)}")
    else:
        typer.echo(f"Particle {particle_id[:8]}… already had all requested tags.")


async def _particle_untag(id_prefix: str, tags: list[str]) -> None:
    from particles.api.cli._id_norm import normalise_particle_id
    from particles.api.client import get_backend

    backend = get_backend()
    if backend.remote:
        particle_id = normalise_particle_id(id_prefix)
    else:
        async with session_scope() as session:
            particle_id = await _resolve_particle_id(session, id_prefix)
    removed = await backend.particle_untag(particle_id, tags)
    if removed:
        typer.echo(f"Untagged {particle_id[:8]}… from {', '.join(removed)}")
    else:
        typer.echo(f"Particle {particle_id[:8]}… had none of the requested tags.")


@particle_app.command("retract")
def particle_retract_cmd(
    particle_id: str = typer.Argument(..., help="Particle ID (prefix OK — first 8 chars)"),
    reason: str = typer.Option(
        ..., "--reason", help="Why this belief is being retired; recorded in the event log"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan without writing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
) -> None:
    """Retract one belief under operator authority.

    The narrow escape hatch beside the cross-asserter guardrail:
    ``corpus retract`` retires *every* live particle from a source, and flipping
    ``mcp.write.allow_cross_asserter`` would widen what every future agent
    session may mutate in order to fix one row. This retires exactly one.

    ACTIVE → RETRACTED with reason ``EXPLICIT_RETRACTION``, routed through
    ``update_particle_status`` so the ``retired_at`` stamp and the ``PARTICLE_RETRACTED`` event (carrying ``--reason``) are both
    written. An operator-asserted (HUMAN_REVIEW) belief is still not retractable
    this way — revising one is Review's job. Run ``particles lint`` afterwards to
    cascade ``PROVENANCE_STALE`` to anything that depended on it.
    """
    if not reason.strip():
        typer.echo("--reason must be non-empty.", err=True)
        raise typer.Exit(1)
    run(_particle_retract(particle_id, reason, dry_run=dry_run, yes=yes))


#: Event actor for a shell retraction, so ``particles events`` distinguishes it
#: from the HTTP route and from an agent's own writes.
RETRACT_ACTOR = "cli:particle-retract"


async def _particle_retract(id_prefix: str, reason: str, *, dry_run: bool, yes: bool) -> None:
    """Resolve, show, confirm, then retract.

    Deliberately **not** gated on ``mcp.write.enabled_stores``:
    that knob is policy about what agents may do over MCP, and this is the
    operator's own hands on their own store. The guardrail is about
    the *asserter* and is untouched — ``allow_cross_asserter`` stays false and
    the MCP surface is unchanged.
    """
    from particles.api.cli._remote import ensure_local
    from particles.operations.agent_write import retract_belief
    from particles.store.particle_store import get_particle

    ensure_local("particle retract")

    async with session_scope() as session:
        particle_id = await _resolve_particle_id(session, id_prefix)
        target = await get_particle(session, particle_id)
    if target is None:  # pragma: no cover — the resolver already exits on a miss
        typer.echo(f"Particle {id_prefix!r} not found.", err=True)
        raise typer.Exit(1)

    # You identified it by eight characters; see what you are about to retire.
    typer.echo(f"  {particle_id}")
    typer.echo(f"  status      {target.status.value}")
    typer.echo(f"  asserted by {target.asserted_by}")
    typer.echo(f"  confidence  {target.confidence.value:.2f}")
    typer.echo(f"  content     {target.content}")
    typer.echo(f"  reason      {reason}")

    if dry_run:
        typer.echo("\n--dry-run: nothing written.")
        return
    if not yes:
        typer.confirm("\nRetract this belief?", abort=True)

    async with session_scope(write=True) as session:  # writer lock
        try:
            await retract_belief(
                session,
                store=DEFAULT_STORE,
                particle_id=particle_id,
                reason=reason,
                operator=True,
                actor=RETRACT_ACTOR,
            )
        except ValueError as exc:
            # The §6 guards (HUMAN_REVIEW, ACTIVE-only) and the §6.6 transition
            # table surface here. Idempotence is theirs, not a special case: a
            # second run reports "is RETRACTED, not ACTIVE".
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        await session.commit()

    typer.echo(f"\nRetracted {particle_id[:8]}… (EXPLICIT_RETRACTION); reason recorded.")
    typer.echo("Run `particles lint` to cascade PROVENANCE_STALE to dependents.")


@particle_app.command("search")
def particle_search_cmd(
    fingerprint: str = typer.Option(
        ...,
        "--fingerprint",
        help="Context fingerprint (full SHA-256 hex or prefix ≥ 8 chars)",
    ),
    limit: int = typer.Option(50, "--limit", help="Maximum particles to list"),
) -> None:
    """List particles sharing a context fingerprint."""
    run(_particle_search(fingerprint, limit))


async def _particle_search(fingerprint: str, limit: int) -> None:
    from sqlalchemy import select

    from particles.api.cli._remote import ensure_local
    from particles.store.particle_store import ParticleRow

    ensure_local("particle search")

    fp = fingerprint.lower().strip()
    if len(fp) < 8:
        typer.echo("Fingerprint prefix must be at least 8 hex characters.", err=True)
        raise typer.Exit(1)
    if any(ch not in "0123456789abcdef" for ch in fp):
        typer.echo("Fingerprint must be hexadecimal.", err=True)
        raise typer.Exit(1)

    async with session_scope() as session:
        if len(fp) == 64:
            stmt = select(ParticleRow).where(ParticleRow.context_fingerprint == fp)
        else:
            stmt = select(ParticleRow).where(
                ParticleRow.context_fingerprint.like(
                    f"{escape_like_pattern(fp)}%", escape=LIKE_ESCAPE
                )
            )
        result = await session.execute(stmt.limit(limit))
        rows = result.scalars().all()

        if not rows:
            typer.echo(f"No particles found with fingerprint {fp[:16]}….", err=True)
            raise typer.Exit(1)

        typer.echo(f"{len(rows)} particle(s) sharing fingerprint {fp[:16]}…:")
        typer.echo("")
        for r in rows:
            content_preview = r.content if len(r.content) <= 80 else r.content[:77] + "…"
            typer.echo(f"  {r.id[:8]}…  [{r.status}]  {content_preview}")
