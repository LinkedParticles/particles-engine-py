# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""review verb — list and resolve INCONSISTENCY particles (§9.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import typer

from particles.api.cli import app, run
from particles.api.client import get_backend
from particles.core.schema import Particle, ResolutionAction
from particles.db import session_scope
from particles.operations.review import prior_reviews


@app.command("review")
def review_cmd(
    particle_id: str | None = typer.Argument(None, help="INCONSISTENCY particle ID; omit to list"),
    action: str | None = typer.Option(None, help="PREFER_A, PREFER_B, BOTH_VALID, DEFER, DISCARD"),
    bulk: str | None = typer.Option(None, "--bulk", help="Apply action to ALL pending conflicts"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview bulk action without committing"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation a --bulk DISCARD asks for"
    ),
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

        # Neither side is worth keeping: retract both, no trust verdict
        particles review PARTICLE_ID --action DISCARD --note "session state"

        # Resolve all pending conflicts with one action
        particles review --bulk BOTH_VALID
        particles review --bulk PREFER_B          # prefer newer/structured source
        particles review --bulk BOTH_VALID --dry-run  # preview without committing
        particles review --bulk DISCARD --dry-run     # list what would be retracted

    A --bulk DISCARD lists every conflict with both sides and asks before it
    retracts anything (skip the prompt with --yes). Retraction has no undo.
    """
    # Bulk resolution
    if bulk is not None:
        try:
            bulk_action = ResolutionAction(bulk)
        except ValueError:
            typer.echo(
                f"Unknown action: {bulk!r}. Use PREFER_A, PREFER_B, BOTH_VALID, DEFER, or DISCARD.",
                err=True,
            )
            raise typer.Exit(1)
        if bulk_action is ResolutionAction.DISCARD:
            # The one bulk action that retracts beliefs, with no undo: show
            # every conflict it would close, then ask.
            items = run(_list_review_detail())
            if not items:
                typer.echo("No INCONSISTENCY particles pending review.")
                return
            _echo_items(items, hint=False)
            verb = "would retract" if dry_run else "will retract"
            typer.echo(f"DISCARD {verb} both sides of {len(items)} conflicts.")
            if dry_run:
                return
            if not yes and not typer.confirm("Proceed?", default=False):
                typer.echo("Aborted; nothing written.")
                raise typer.Exit(1)
            particles_list = [item.inconsistency for item in items]
        else:
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
        _echo_items(items, hint=True)
        typer.echo(f"{len(items)} conflicts pending review.")
        return

    # Single-particle resolution
    if action is None:
        typer.echo("--action required when providing a particle_id", err=True)
        raise typer.Exit(1)

    from particles.api.cli._id_norm import normalise_particle_id

    backend = get_backend()
    target = normalise_particle_id(particle_id)
    # Accept the short id the listing, `curate` and `particle show` print. A
    # remote backend resolves ids server-side, so only a local one expands here.
    if not backend.remote:
        target = run(_resolve_local(target))
    try:
        review = run(
            backend.review_resolve(target, ResolutionAction(action), reviewer_id, domain, note)
        )
    except ValueError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Review {review.review_id} recorded: {action}")


async def _resolve_local(id_prefix: str) -> str:
    """Expand a particle id prefix against the local store, or exit."""
    from particles.api.cli.particle import _resolve_particle_id

    async with session_scope() as session:
        return await _resolve_particle_id(session, id_prefix)


def _echo_items(items: list[ReviewDetailItem], *, hint: bool) -> None:
    """Print each conflict's two sides, as the listing and a bulk DISCARD show them."""
    sep = "─" * 72
    for i, item in enumerate(items, 1):
        inc = item.inconsistency
        pa = item.particle_a
        pb = item.particle_b
        typer.echo(sep)
        typer.echo(f"[{i}/{len(items)}]  {inc.id[:8]}…")
        census = _census_lines(inc.content)
        if census:
            # A census record names its claims by side; review
            # resolves every member of a side at once.
            typer.echo("  Two sources disagree (the nightly check; a second reading confirmed it).")
            for line in census:
                typer.echo(f"  {line}")
            for prior in item.history:
                typer.echo(f"  earlier: {prior}")
            if hint:
                typer.echo(
                    f"  → particles review {inc.id}"
                    " --action [PREFER_A|PREFER_B|BOTH_VALID|DEFER|DISCARD]"
                    " (PREFER and DISCARD apply to every claim on a side)"
                )
            continue
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
        if hint:
            typer.echo(
                f"  → particles review {inc.id}"
                " --action [PREFER_A|PREFER_B|BOTH_VALID|DEFER|DISCARD]"
            )
    typer.echo(sep)


def _census_lines(inc_content: str) -> list[str]:
    """The side and reason lines of a census record's content, or ``[]`` for any other record."""
    lines = inc_content.splitlines()
    if not lines or "second reading confirmed it" not in lines[0]:
        return []
    return [
        line for line in lines[1:] if line.startswith(("Side A:", "Side B:", "Second reading:"))
    ]


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
    #: Reviews left on the records this census record replaced,
    #: one rendered line each.
    history: list[str] = field(default_factory=list)


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
            history = [
                f"{prior.action} on {prior.record_id[:8]}… ({prior.reviewed_at:%Y-%m-%d})"
                + (f": {prior.note}" if prior.note else "")
                for prior in await prior_reviews(session, inc)
            ]
            result.append(
                ReviewDetailItem(
                    inconsistency=inc,
                    particle_a=pa,
                    particle_b=pb,
                    author_a_id=a_id,
                    author_a_role=a_role,
                    author_b_id=b_id,
                    author_b_role=b_role,
                    history=history,
                )
            )
        return result
