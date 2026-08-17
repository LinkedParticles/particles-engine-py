"""review verb — list and resolve INCONSISTENCY particles (§9.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer

from particles.api.cli import app, run
from particles.api.client import get_backend
from particles.core.schema import Particle, ResolutionAction
from particles.db import session_scope


@app.command("review")
def review_cmd(
    particle_id: str | None = typer.Argument(None, help="INCONSISTENCY particle ID; omit to list"),
    action: str | None = typer.Option(None, help="PREFER_A, PREFER_B, BOTH_VALID, DEFER"),
    bulk: str | None = typer.Option(None, "--bulk", help="Apply action to ALL pending conflicts"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview bulk action without committing"),
    reviewer_id: str = typer.Option("cli-user", help="Reviewer identity"),
    domain: str = typer.Option("general", help="Domain for trust statement"),
    note: str | None = typer.Option(None, help="Optional reviewer note"),
) -> None:
    """List or resolve INCONSISTENCY particles.

    \b
        # List all pending conflicts
        particles review

        # Resolve a specific conflict
        particles review PARTICLE_ID --action PREFER_A

        # Resolve all pending conflicts with one action
        particles review --bulk BOTH_VALID
        particles review --bulk PREFER_B          # prefer newer/structured source
        particles review --bulk BOTH_VALID --dry-run  # preview without committing
    """
    # Bulk resolution
    if bulk is not None:
        try:
            bulk_action = ResolutionAction(bulk)
        except ValueError:
            typer.echo(
                f"Unknown action: {bulk!r}. Use PREFER_A, PREFER_B, BOTH_VALID, or DEFER.", err=True
            )
            raise typer.Exit(1)
        particles_list = run(get_backend().review_list())
        if not particles_list:
            typer.echo("No INCONSISTENCY particles pending review.")
            return
        if dry_run:
            typer.echo(f"Dry run: would apply {bulk_action} to {len(particles_list)} conflicts.")
            return
        typer.echo(f"Applying {bulk_action} to {len(particles_list)} conflicts…")
        succeeded = failed = 0
        for p in particles_list:
            try:
                run(get_backend().review_resolve(p.id, bulk_action, reviewer_id, domain, note))
                succeeded += 1
            except Exception as exc:
                typer.echo(f"  Failed {p.id[:8]}…: {exc}", err=True)
                failed += 1
        typer.echo(f"Done: {succeeded} resolved, {failed} failed.")
        return

    # Single-particle listing
    if particle_id is None:
        items = run(_list_review_detail())
        if not items:
            typer.echo("No INCONSISTENCY particles pending review.")
            return
        sep = "─" * 72
        for i, item in enumerate(items, 1):
            inc = item.inconsistency
            pa = item.particle_a
            pb = item.particle_b
            typer.echo(sep)
            typer.echo(f"[{i}/{len(items)}]  {inc.id[:8]}…")
            if pa:
                typer.echo(f"  A: {pa.content}")
                typer.echo(f"     asserted_by: {pa.asserted_by}")
                author_a = _format_author(item.author_a_id, item.author_a_role)
                if author_a:
                    typer.echo(f"     author:      {author_a}")
            b_text = _parse_particle_b(inc.content)
            if b_text:
                b_asserted = pb.asserted_by if pb else "—"
                typer.echo(f"  B: {b_text}")
                typer.echo(f"     asserted_by: {b_asserted}")
                author_b = _format_author(item.author_b_id, item.author_b_role)
                if author_b:
                    typer.echo(f"     author:      {author_b}")
            typer.echo(
                f"  → particles review {inc.id} --action [PREFER_A|PREFER_B|BOTH_VALID|DEFER]"
            )
        typer.echo(sep)
        typer.echo(f"{len(items)} conflicts pending review.")
        return

    # Single-particle resolution
    if action is None:
        typer.echo("--action required when providing a particle_id", err=True)
        raise typer.Exit(1)

    from particles.api.cli._id_norm import normalise_particle_id

    review = run(
        get_backend().review_resolve(
            normalise_particle_id(particle_id),
            ResolutionAction(action),
            reviewer_id,
            domain,
            note,
        )
    )
    typer.echo(f"Review {review.review_id} recorded: {action}")


def _parse_particle_b(inc_content: str) -> str:
    """Extract the Particle B content preview from an INCONSISTENCY particle's text."""
    for line in inc_content.splitlines():
        if line.startswith("Particle B (new): "):
            return line[len("Particle B (new): ") :]
    return ""


@dataclass
class ReviewDetailItem:
    """One row in the Review-detail view (particles + per-particle author info).

    Author info is the ``author_id`` + ``author_role`` from each side's
    SOURCE snapshot (spec §6 v0.2 Core checklist: "Surface author_id and
    author_role in Review UI for UGC corpus entries"). Either author field
    is None for non-UGC sources or when provenance is partial.
    """

    inconsistency: Particle
    particle_a: Particle | None
    particle_b: Particle | None
    author_a_id: str | None
    author_a_role: str | None
    author_b_id: str | None
    author_b_role: str | None


def _format_author(author_id: str | None, author_role: str | None) -> str:
    """Render the author line shown under each side of a review item.

    Returns "" when no author info is recorded (non-UGC source). With an
    ID but no role: just the ID. With both: "ID (role: ROLE)".
    """
    if not author_id:
        return ""
    if not author_role:
        return author_id
    return f"{author_id} (role: {author_role})"


async def _author_for_particle(
    session: Any, particle: Particle | None
) -> tuple[str | None, str | None]:
    """Look up ``(author_id, author_role)`` from the SOURCE snapshot."""
    if particle is None:
        return (None, None)
    from particles.core.schema import ProvenanceRefType
    from particles.corpus.store import get_snapshot

    src = next(
        (r for r in particle.provenance if r.type == ProvenanceRefType.SOURCE),
        None,
    )
    if src is None or src.snapshot_id is None:
        return (None, None)
    snap = await get_snapshot(session, src.snapshot_id)
    if snap is None:
        return (None, None)
    return (snap.author_id, snap.author_role)


async def _list_review_detail() -> list[ReviewDetailItem]:
    """Return enriched (inconsistency, particle_a, particle_b, author info) rows.

    The INCONSISTENCY list comes from the backend (local or remote). Author /
    two-sides enrichment reads SOURCE snapshots, which only the local store can
    serve, so in remote mode the rows carry the inconsistency alone with ``None``
    sides — the renderer skips the A/B blocks and still shows each conflict's
    own ``Particle B`` preview parsed from its content.
    """
    from particles.core.schema import ProvenanceRefType
    from particles.store.particle_store import get_particle

    backend = get_backend()
    inconsistencies = await backend.review_list()
    if backend.remote:
        return [
            ReviewDetailItem(inc, None, None, None, None, None, None) for inc in inconsistencies
        ]

    async with session_scope() as session:
        result: list[ReviewDetailItem] = []
        for inc in inconsistencies:
            pa: Particle | None = None
            pb: Particle | None = None
            particle_refs = [r for r in inc.provenance if r.type == ProvenanceRefType.PARTICLE]
            if len(particle_refs) >= 1:
                pa = await get_particle(session, particle_refs[0].corpus_entry_id)
            if len(particle_refs) >= 2:
                pb = await get_particle(session, particle_refs[1].corpus_entry_id)
            a_id, a_role = await _author_for_particle(session, pa)
            b_id, b_role = await _author_for_particle(session, pb)
            result.append(
                ReviewDetailItem(
                    inconsistency=inc,
                    particle_a=pa,
                    particle_b=pb,
                    author_a_id=a_id,
                    author_a_role=a_role,
                    author_b_id=b_id,
                    author_b_role=b_role,
                )
            )
        return result
