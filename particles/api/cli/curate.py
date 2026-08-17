"""curate verb — the bus-stop-editing curation queue.

``particles curate`` prints the unified, finite, leverage-ranked worklist that
unions the existing read diagnostics; ``particles curate apply <gesture> <key>``
dispatches a card's gesture onto the existing write op. Local-engine surface
(the HTTP exposure for a thin client is deferred).
"""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.db import session_scope
from particles.operations.curation.cards import CardKind
from particles.operations.curation.snapshot import CurationQueueResult

curate_app = typer.Typer(
    help="Bus-stop editing — the finite, leverage-ranked curation queue.",
    no_args_is_help=False,
)
app.add_typer(curate_app, name="curate")


def _freshness_line(result: CurationQueueResult, *, bypassed: bool = False) -> str:
    """One-line staleness stamp under the store header.

    Honest staleness beats a fast lie: an operator who can see when the
    collection was built can tell a genuinely quiet queue from a stale one, and
    knows whether `--refresh` is what they want.

    ``bypassed`` distinguishes the two ways a run can be live — the operator
    asked for it with ``--no-snapshot``, or the store simply has no collection
    yet — because only the second is something to act on.
    """
    if result.source == "live" or result.built_at is None:
        live = f"Collection: {result.collection_size:,} cards, collected live for this run"
        if bypassed:
            return live + " (--no-snapshot)."
        return live + " — no stored collection yet; `particles curate --refresh` builds one."
    age = result.age_seconds or 0.0
    if age < 3600:
        ago = f"{age / 60:.0f} min ago"
    elif age < 86_400:
        ago = f"{age / 3600:.1f} h ago"
    else:
        ago = f"{age / 86_400:.1f} days ago"
    stamp = result.built_at.strftime("%Y-%m-%d %H:%M UTC")
    line = f"Collection: {result.collection_size:,} cards, built {stamp} ({ago})"
    if result.stale:
        line += " — STALE, run `particles curate --refresh`"
    delta = sorted(k for k, v in result.per_kind_scope.items() if v == "delta")
    if delta:
        line += f"; delta-scoped: {', '.join(delta)}"
    return line + "."


def _resolve_kind(kind: str | None) -> CardKind | None:
    if kind is None:
        return None
    try:
        return CardKind(kind.lower())
    except ValueError as exc:
        valid = ", ".join(k.value for k in CardKind)
        raise typer.BadParameter(f"Unknown kind {kind!r}. Valid: {valid}") from exc


@curate_app.callback(invoke_without_command=True)
def curate_main(
    ctx: typer.Context,
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Cap the cards shown (default: curation.session_size)."
    ),
    kind: str | None = typer.Option(
        None, "--kind", "-k", help="Restrict to one card kind (e.g. stale, contested)."
    ),
    semantic: bool = typer.Option(
        False, "--semantic", help="Run the LLM-assisted finders (semantic contradiction)."
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Rebuild the card collection before showing it. Slow — "
        "the finders re-run store-wide. Run this once on a store with no "
        "collection yet; the nightly `memory consolidate` does it for you "
        "after that.",
    ),
    no_snapshot: bool = typer.Option(
        False,
        "--no-snapshot",
        help="Bypass the persisted collection entirely and run the finders for "
        "this invocation without caching the result.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Show today's curation queue (the default action when no subcommand is given)."""
    if ctx.invoked_subcommand is not None:
        return
    configure_logging(verbose, debug)
    kind_enum = _resolve_kind(kind)
    run(
        _show(
            limit=limit,
            kind=kind_enum,
            semantic=semantic,
            refresh=refresh,
            no_snapshot=no_snapshot,
        )
    )


@curate_app.command("apply")
def curate_apply_cmd(
    gesture: str = typer.Argument(
        ...,
        help="affirm | snooze | dismiss | retract | merge | deposit | assign-subject "
        "| accept | reject",
    ),
    card_key: str = typer.Argument(..., help="The card key shown in the queue listing."),
    reason: str | None = typer.Option(None, "--reason", help="Rationale (recorded on retract)."),
    days: int | None = typer.Option(None, "--days", help="Snooze window in days."),
    subject: str | None = typer.Option(
        None,
        "--subject",
        help="Subject id or name for the assign-subject gesture.",
    ),
) -> None:
    """Apply a gesture to a card, dispatching onto the existing write op."""
    run(_apply(gesture=gesture, card_key=card_key, reason=reason, days=days, subject=subject))


async def _show(
    *,
    limit: int | None,
    kind: CardKind | None,
    semantic: bool,
    refresh: bool = False,
    no_snapshot: bool = False,
) -> None:
    from particles.operations.curation import (
        QueueSource,
        build_curation_queue,
        rebuild_curation_snapshot,
    )
    from particles.operations.quality import get_quality_report

    async with session_scope() as session:
        report = await get_quality_report(session)
        if refresh:
            typer.echo("Rebuilding the curation collection (running every finder)…")
            await rebuild_curation_snapshot(session, semantic=semantic)
        result = await build_curation_queue(
            session,
            limit=limit,
            kind=kind,
            semantic=semantic,
            source=QueueSource.LIVE if no_snapshot else QueueSource.SNAPSHOT,
        )

    typer.echo(
        f"Store: {report.active_particles:,} active · "
        f"{report.inconsistency_particles:,} inconsistency · "
        f"{report.snapshots_failed:,} failed snapshots · "
        f"{report.subjects_without_particles:,} subjects w/o particles"
    )
    typer.echo(_freshness_line(result, bypassed=no_snapshot) + "\n")

    cards = result.cards
    if not cards:
        typer.echo("Curation queue empty — nothing flagged. ✨")
        return

    typer.echo(f"Curation queue — today's {len(cards)} (highest leverage first):\n")
    for i, c in enumerate(cards, 1):
        typer.echo(f"{i}. [{c.kind.value}]  leverage {c.leverage:.2f}")
        typer.echo(f"   {c.diagnostic}")
        typer.echo(f"   gestures: {', '.join(c.suggested_gestures)}")
        typer.echo(f"   key: {c.key}")
        typer.echo("")
    typer.echo("Apply with:  particles curate apply <gesture> <key>")


async def _apply(
    *,
    gesture: str,
    card_key: str,
    reason: str | None,
    days: int | None,
    subject: str | None = None,
) -> None:
    from particles.operations.curation import CurationCard, apply_gesture

    try:
        card = CurationCard.from_key(card_key)
    except ValueError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(1) from exc

    async with session_scope() as session:
        try:
            message = await apply_gesture(
                session, card, gesture, reason=reason, days=days, subject=subject
            )
        except ValueError as exc:
            typer.echo(f"✗ {exc}", err=True)
            raise typer.Exit(1) from exc
        await session.commit()
    typer.echo(f"✓ {message}")
