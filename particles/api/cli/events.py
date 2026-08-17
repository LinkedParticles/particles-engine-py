"""`events` sub-Typer — read the operator event log.

Read-only inspection of operator decisions (retract / split / merge / alias /
confirm / unlink / trust / review / links / tags). Mirrors the `GET /events`
HTTP endpoints and the MCP `events` tool one-to-one — the three front-ends
share the `list_events` / `get_event` read model.
"""

from __future__ import annotations

import typer

from particles.api.cli import app, run

events_app = typer.Typer(help="Inspect the operator event log.", no_args_is_help=True)
app.add_typer(events_app, name="events")


@events_app.command("list")
def events_list_cmd(
    particle: str | None = typer.Option(
        None, "--particle", help="Only events touching this particle id"
    ),
    subject: str | None = typer.Option(
        None, "--subject", help="Only events touching this subject id"
    ),
    entry: str | None = typer.Option(
        None, "--entry", help="Only events touching this corpus entry id"
    ),
    event_type: str | None = typer.Option(
        None, "--type", help="Only events of this type (e.g. SOURCE_RETRACTED)"
    ),
    limit: int = typer.Option(50, "--limit", help="Maximum events to show"),
) -> None:
    """List operator events, newest first."""
    run(_events_list(particle, subject, entry, event_type, limit))


async def _events_list(
    particle: str | None,
    subject: str | None,
    entry: str | None,
    event_type: str | None,
    limit: int,
) -> None:
    from particles.api.client import get_backend
    from particles.store.event_store import OperatorEventType, ref_filter

    # Validate the filter arguments up front (pure — no store) so the operator
    # gets the same actionable message in both transports; the query itself
    # routes through the backend.
    try:
        ref_filter(particle=particle, subject=subject, corpus_entry=entry)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if event_type is not None:
        try:
            OperatorEventType(event_type)
        except ValueError as exc:
            valid = ", ".join(t.value for t in OperatorEventType)
            typer.echo(f"Unknown --type {event_type!r}. Valid: {valid}", err=True)
            raise typer.Exit(1) from exc

    events = await get_backend().events_list(
        particle=particle, subject=subject, entry=entry, event_type=event_type, limit=limit
    )

    if not events:
        typer.echo("No operator events.")
        return
    for e in events:
        when = e.occurred_at.strftime("%Y-%m-%d %H:%M")
        line = f"{when}  {e.event_id[:8]}…  {e.event_type.value}  by {e.actor}"
        typer.echo(line)
        if e.reason:
            typer.echo(f"    reason: {e.reason}")
        if e.refs:
            refs = ", ".join(f"{r.ref_kind.value}:{r.ref_id[:8]}…" for r in e.refs)
            typer.echo(f"    refs:   {refs}")


@events_app.command("show")
def events_show_cmd(
    event_id: str = typer.Argument(..., help="Event ID"),
) -> None:
    """Show one operator event in full (header + refs + payload)."""
    run(_events_show(event_id))


async def _events_show(event_id: str) -> None:
    import json

    from particles.api.client import get_backend

    event = await get_backend().event_show(event_id)
    if event is None:
        typer.echo(f"Event {event_id!r} not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Event:    {event.event_id}")
    typer.echo(f"When:     {event.occurred_at.isoformat()}")
    typer.echo(f"Type:     {event.event_type.value}")
    typer.echo(f"Actor:    {event.actor}")
    if event.reason:
        typer.echo(f"Reason:   {event.reason}")
    if event.refs:
        typer.echo("Refs:")
        for r in event.refs:
            typer.echo(f"  - {r.ref_kind.value}: {r.ref_id}")
    if event.payload:
        typer.echo("Payload:")
        typer.echo(json.dumps(event.payload, indent=2, sort_keys=True))
