"""synthesis-cache sub-Typer — inspect and prune the shared article cache.

The cross-exporter synthesis cache (`synthesis_cache`) never auto-evicts; this
group is the operator's read + cleanup surface over it:
``list`` the index, ``show`` one subject's cached body, ``vacuum`` the
provably-dead rows, ``evict`` a single subject.
"""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.db import session_scope

cache_app = typer.Typer(
    help="Inspect and prune the shared article-synthesis cache.",
    no_args_is_help=True,
)
app.add_typer(cache_app, name="synthesis-cache")


@cache_app.command("list")
def synthesis_cache_list_cmd() -> None:
    """List every cached article (subject, hash, prompt version, age, size)."""
    run(_list_entries())


@cache_app.command("show")
def synthesis_cache_show_cmd(
    subject_id: str = typer.Argument(..., help="Subject ID (prefix OK)"),
) -> None:
    """Print the cached article body(ies) + metadata for one subject."""
    run(_show_entry(subject_id))


@cache_app.command("vacuum")
def synthesis_cache_vacuum_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be removed without deleting"
    ),
) -> None:
    """Delete unreachable rows: stale prompt versions + orphaned subjects."""
    run(_vacuum(dry_run))


@cache_app.command("evict")
def synthesis_cache_evict_cmd(
    subject_id: str = typer.Argument(..., help="Subject ID (prefix OK)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
) -> None:
    """Evict every cached article for one subject."""
    run(_evict(subject_id, yes))


# ---------------------------------------------------------------------------
# async impls
# ---------------------------------------------------------------------------


async def _subject_name_map() -> dict[str, str]:
    from particles.store.subject_store import list_all_subjects

    async with session_scope() as session:
        return {s.id: s.canonical_name for s in await list_all_subjects(session)}


async def _list_entries() -> None:
    from particles.store.synthesis_cache_store import list_cache_entries

    async with session_scope() as session:
        rows = await list_cache_entries(session)
    if not rows:
        typer.echo("Synthesis cache is empty.")
        return
    names = await _subject_name_map()
    typer.echo(f"{len(rows)} cached article(s):")
    for r in rows:
        name = names.get(r.subject_id, "(subject deleted)")
        when = r.generated_at.strftime("%Y-%m-%d")
        verdict = f"  [{r.layer_b_verdict}]" if r.layer_b_verdict else ""
        typer.echo(
            f"  {r.subject_id[:8]}…  {name}  hash={r.input_hash[:8]}…  "
            f"pv={r.prompt_version}  {when}  {len(r.article_body)} chars{verdict}"
        )


async def _resolve_subject(prefix: str) -> str | list[str]:
    """Resolve a subject-id prefix against the cache. Returns the id, or the
    list of matches when zero / ambiguous (caller renders the error)."""
    from particles.store.synthesis_cache_store import list_cache_entries

    async with session_scope() as session:
        rows = await list_cache_entries(session)
    matches = sorted({r.subject_id for r in rows if r.subject_id.startswith(prefix)})
    if len(matches) == 1:
        return matches[0]
    return matches


async def _show_entry(subject_prefix: str) -> None:
    from particles.store.synthesis_cache_store import list_cache_entries

    resolved = await _resolve_subject(subject_prefix)
    if isinstance(resolved, list):
        if not resolved:
            typer.echo(f"No cached articles for subject {subject_prefix!r}.", err=True)
        else:
            typer.echo(
                f"Subject prefix {subject_prefix!r} is ambiguous "
                f"({len(resolved)} matches). Use a longer prefix.",
                err=True,
            )
        raise typer.Exit(1)

    names = await _subject_name_map()
    async with session_scope() as session:
        rows = [r for r in await list_cache_entries(session) if r.subject_id == resolved]
    name = names.get(resolved, "(subject deleted)")
    typer.echo(f"Subject: {resolved}  ({name})")
    for i, r in enumerate(rows, start=1):
        if len(rows) > 1:
            typer.echo(f"\n--- entry {i}/{len(rows)} ---")
        typer.echo(f"input_hash:     {r.input_hash}")
        typer.echo(f"prompt_version: {r.prompt_version}")
        typer.echo(f"generated_at:   {r.generated_at.isoformat()}")
        if r.layer_b_verdict:
            typer.echo(f"layer_b:        {r.layer_b_verdict}")
        typer.echo("")
        typer.echo(r.article_body)


async def _vacuum(dry_run: bool) -> None:
    from particles.render.article_synthesis import _PROMPT_VERSION
    from particles.store.synthesis_cache_store import vacuum_cache

    async with session_scope() as session:
        counts = await vacuum_cache(session, current_prompt_version=_PROMPT_VERSION)
        total = sum(counts.values())
        if dry_run:
            await session.rollback()
        else:
            await session.commit()

    if total == 0:
        typer.echo("Nothing to vacuum — every cache row is reachable.")
        return
    verb = "Would remove" if dry_run else "Removed"
    typer.echo(f"{verb} {total} unreachable cache row(s):")
    typer.echo(f"  stale prompt version: {counts['stale_prompt_version']}")
    typer.echo(f"  orphaned subject:     {counts['orphaned_subject']}")
    if dry_run:
        typer.echo("  (no changes made)")


async def _evict(subject_prefix: str, yes: bool) -> None:
    from particles.store.synthesis_cache_store import evict_subject

    resolved = await _resolve_subject(subject_prefix)
    if isinstance(resolved, list):
        if not resolved:
            typer.echo(f"No cached articles for subject {subject_prefix!r}.", err=True)
        else:
            typer.echo(
                f"Subject prefix {subject_prefix!r} is ambiguous "
                f"({len(resolved)} matches). Use a longer prefix.",
                err=True,
            )
        raise typer.Exit(1)

    if not yes:
        typer.confirm(f"Evict every cached article for {resolved[:8]}…?", abort=True)
    async with session_scope() as session:
        removed = await evict_subject(session, resolved)
        await session.commit()
    typer.echo(f"Evicted {removed} cached article(s) for {resolved[:8]}….")
