"""deposit verb — push a file or URL into the corpus."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.client import get_backend
from particles.db import DEFAULT_STORE
from particles.http import SourceFetchError


def _parse_cli_date(value: str) -> datetime:
    """Parse an operator ``--date`` (ISO ``YYYY-MM-DD`` or full ISO 8601) to UTC.

    Raises ``ValueError`` for anything ``datetime.fromisoformat`` rejects; the
    caller turns that into a CLI error. A date-only value becomes midnight UTC.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@app.command("deposit")
def deposit_cmd(
    source: str | None = typer.Argument(
        None,
        help=(
            "File path or URL to deposit. Pass '-' to read the content from "
            "stdin. Omit it when using --text."
        ),
    ),
    text: str | None = typer.Option(
        None,
        "--text",
        help=(
            "Deposit a literal string instead of a file or URL — "
            "the CLI half of the MCP `deposit_text` tool, so you no longer have "
            "to write a temp file to record one note. Mutually exclusive with a "
            "source argument. `particles deposit -` reads the same content from "
            "stdin. Defaults to source-type CONVERSATION; attributed to "
            "--deposited-by on both the deposited_by and author_id axes."
        ),
    ),
    deposited_by: str = typer.Option("operator", help="Agent or operator ID"),
    source_type: str | None = typer.Option(
        None,
        help=(
            "Override the source_type (normally auto-detected from the "
            "extension / URL / content — you rarely need this). Core values "
            "(particles.core.schema.SourceType): WEB_PAGE, PDF, CSV, "
            "CONVERSATION, DATA_EXPORT, LOCAL_FILE, LOCAL_MARKDOWN, "
            "ACADEMIC_PAPER, FORUM, BLOG, TAXONOMY_DEFINITION, "
            "TRUST_LENS_DEFINITION. Domain extractors register their own, e.g. "
            "JOURNAL, REDDIT_POST, HACKERNEWS_THREAD, MASTODON_THREAD, "
            "GITHUB_REPO / GITHUB_GIST / GITHUB_PAGES, WIKIDATA_API, "
            "NUMISTA_API_COIN / NUMISTA_API_ISSUER, NOMISMA_API. Use --journal "
            "for the JOURNAL shortcut."
        ),
    ),
    journal: bool = typer.Option(
        False,
        "--journal",
        help=(
            "Mark this deposit as a personal JOURNAL so the "
            "journal-aware extractor handles it (reifies feelings/opinions and "
            "emits the NARRATIVE graph). Shorthand for --source-type JOURNAL; "
            "an explicit --source-type wins."
        ),
    ),
    tags: str | None = typer.Option(None, help="Comma-separated tags"),
    date: str | None = typer.Option(
        None,
        "--date",
        help=(
            "Record this content's authorship date as "
            "content_published_at (ISO YYYY-MM-DD). Overrides the leading-date "
            "and file-mtime auto-detection. Local-file deposits only — ignored "
            "with a warning for URLs."
        ),
    ),
    split_by_date: bool = typer.Option(
        False,
        "--split-by-date",
        help=(
            "Split a multi-entry local file at standalone date-line "
            "boundaries into N corpus entries, each with its own "
            "content_published_at (a journal / changelog / daily-log that "
            "concatenates many dated entries). Opt-in; default off leaves "
            "today's one-file-one-entry behaviour unchanged. Local files only; "
            "mutually exclusive with --date. A file that is not actually "
            "multi-entry deposits as a single entry."
        ),
    ),
    follow_post_links: bool | None = typer.Option(
        None,
        "--follow-post-links/--no-follow-post-links",
        help=(
            "Follow the post's primary URL for link-shaped sources "
            "(Reddit / HN / Mastodon link cards). When unspecified, the "
            "extractor's default applies — Reddit / HN / Mastodon default to "
            "True, everything else to False."
        ),
    ),
    follow_comment_links: bool | None = typer.Option(
        None,
        "--follow-comment-links/--no-follow-comment-links",
        help=(
            "Reserved-but-deferred. Passing --follow-comment-links "
            "emits a warning and proceeds as if False — comment-link "
            "following is captured § Deferred and will land in a "
            "follow-up release."
        ),
    ),
    mutability: str | None = typer.Option(
        None,
        "--mutability",
        help=(
            "Mutability class: STABLE | MUTABLE | APPEND_ONLY | EPHEMERAL. "
            "Local files default to STABLE. MUTABLE means a new snapshot retires the "
            "generation of beliefs it replaces — the right class for a "
            "rule file like AGENTS.md that is edited in place. Local deposits only."
        ),
    ),
    fetch_policy: str | None = typer.Option(
        None,
        "--fetch-policy",
        help=(
            "Re-fetch policy: LAZY | NEVER. Local files default to NEVER "
            "(frozen at deposit). LAZY opts the file into the refresh "
            "ladder, so `particles corpus refresh` and the nightly consolidation "
            "pass re-check it against disk. Pair with --mutability MUTABLE. Local "
            "deposits only."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show importer + fetch INFO logs"),
    debug: bool = typer.Option(
        False, "--debug", help="Show URL parsing, request URLs, auth state, and DEBUG logs"
    ),
) -> None:
    """Deposit a file, URL, or literal text into the corpus."""
    configure_logging(verbose, debug)
    # the literal-text path. Resolve it before anything else so the
    # file/URL flags below never see a source that is not one.
    if source == "-":
        if text is not None:
            typer.echo("Error: pass either --text or '-' (stdin), not both.", err=True)
            raise typer.Exit(1)
        text = sys.stdin.read()
        source = None
    if text is not None:
        if source is not None:
            typer.echo(
                "Error: --text deposits a literal string; drop the source argument "
                "(or drop --text to deposit that file/URL).",
                err=True,
            )
            raise typer.Exit(1)
        if not text.strip():
            typer.echo("Error: refusing to deposit empty text.", err=True)
            raise typer.Exit(1)
        for label, unsupported in (
            ("--date", date is not None),
            ("--split-by-date", split_by_date),
            ("--mutability", mutability is not None),
            ("--fetch-policy", fetch_policy is not None),
        ):
            if unsupported:
                # These are properties of a *file on disk* (authorship date,
                # re-read policy). A pasted string has no such source to revisit,
                # so refuse rather than accept and silently ignore.
                typer.echo(
                    f"Error: {label} applies to local-file deposits only, not --text.",
                    err=True,
                )
                raise typer.Exit(1)
        st_text = source_type or ("JOURNAL" if journal else None)
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        entry_id, snapshot_id = run(_deposit_text(text, deposited_by, st_text, tag_list))
        typer.echo(f"entry_id:    {entry_id}")
        typer.echo(f"snapshot_id: {snapshot_id}")
        return
    if source is None:
        typer.echo("Error: give a file path, a URL, '-' for stdin, or --text.", err=True)
        raise typer.Exit(1)
    for label, value, allowed in (
        ("--mutability", mutability, ("APPEND_ONLY", "EPHEMERAL", "MUTABLE", "STABLE")),
        ("--fetch-policy", fetch_policy, ("LAZY", "NEVER")),
    ):
        if value is not None and value.upper() not in allowed:
            typer.echo(
                f"Error: {label} must be one of {', '.join(allowed)}; got {value!r}",
                err=True,
            )
            raise typer.Exit(1)
    mutability = mutability.upper() if mutability else None
    fetch_policy = fetch_policy.upper() if fetch_policy else None
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    st = source_type or ("JOURNAL" if journal else None)
    if split_by_date:
        # --date asserts one authorship date for the whole file, which
        # contradicts "many dated sections." Refuse both rather than guess.
        if date is not None:
            typer.echo(
                "Error: --date and --split-by-date are mutually exclusive "
                "(--date sets one date for the whole file; --split-by-date "
                "derives a date per dated section).",
                err=True,
            )
            raise typer.Exit(1)
        if source.startswith("http://") or source.startswith("https://"):
            typer.echo(
                "Error: --split-by-date is for local files only (URL deposits "
                "derive content_published_at from source metadata).",
                err=True,
            )
            raise typer.Exit(1)
        try:
            rows = run(_deposit_split(source, deposited_by, st, tag_list))
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1)
        typer.echo(f"split into {len(rows)} corpus {'entry' if len(rows) == 1 else 'entries'}:")
        for s_entry_id, s_snap_id in rows:
            typer.echo(f"  entry_id:    {s_entry_id}")
            typer.echo(f"  snapshot_id: {s_snap_id}")
        return
    content_date: datetime | None = None
    if date is not None:
        try:
            content_date = _parse_cli_date(date)
        except ValueError:
            typer.echo(
                f"Error: --date must be an ISO date (e.g. 2026-03-15), got {date!r}",
                err=True,
            )
            raise typer.Exit(1)
    try:
        entry_id, snapshot_id, follow_targets = run(
            _deposit(
                source,
                deposited_by,
                st,
                tag_list,
                follow_post_links=follow_post_links,
                follow_comment_links=follow_comment_links,
                content_date=content_date,
                mutability=mutability,
                fetch_policy=fetch_policy,
            )
        )
    except (ValueError, SourceFetchError) as exc:
        # SourceFetchError = an expected upstream fetch failure (e.g. Reddit's
        # 403 bot-wall) in local mode; surface it cleanly, not as a traceback.
        # In remote mode the engine maps it to a 502 the backend already
        # surfaces as an EngineHttpError.
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"entry_id:    {entry_id}")
    typer.echo(f"snapshot_id: {snapshot_id}")
    if follow_targets:
        # surface every secondary entry the follow machinery
        # deposited so the operator sees what got added beyond the
        # primary. With depth-1 cap, this is at most one row today.
        typer.echo(f"followed {len(follow_targets)} primary URL(s):")
        for f_entry_id, f_snap_id, f_uri in follow_targets:
            typer.echo(f"  → entry_id:    {f_entry_id}")
            typer.echo(f"    snapshot_id: {f_snap_id}")
            typer.echo(f"    uri_r:       {f_uri}")


async def _deposit(
    source: str,
    deposited_by: str,
    source_type: str | None,
    tags: list[str],
    *,
    follow_post_links: bool | None = None,
    follow_comment_links: bool | None = None,
    content_date: datetime | None = None,
    mutability: str | None = None,
    fetch_policy: str | None = None,
) -> tuple[str, str, list[tuple[str, str, str]]]:
    backend = get_backend()
    if source.startswith("http://") or source.startswith("https://"):
        # a URL deposit's mutability and fetch policy come
        # from the importer that handled it, which knows the source's shape.
        if mutability is not None or fetch_policy is not None:
            raise ValueError(
                "--mutability / --fetch-policy apply to local-file deposits only; "
                "a URL deposit takes its class from the importer that handles it."
            )
        # --date is a local-file / archival convenience. URL deposits
        # derive content_published_at from source metadata via the importers, so
        # an explicit --date here is ignored (with a warning).
        if content_date is not None:
            typer.echo(
                "Warning: --date is ignored for URL deposits "
                "(content_published_at comes from the source).",
                err=True,
            )
        outcome = await backend.deposit_url(
            source,
            deposited_by=deposited_by,
            source_type=source_type,
            tags=tags,
            follow_post_links=follow_post_links,
            follow_comment_links=follow_comment_links,
        )
    else:
        outcome = await backend.deposit_file(
            Path(source),
            deposited_by=deposited_by,
            source_type=source_type,
            tags=tags,
            content_date=content_date,
            mutability=mutability,
            fetch_policy=fetch_policy,
        )
    follow_targets = [(f.entry_id, f.snapshot_id, f.uri) for f in outcome.follow_targets]
    return outcome.entry_id, outcome.snapshot_id, follow_targets


async def _deposit_text(
    text: str,
    deposited_by: str,
    source_type: str | None,
    tags: list[str],
) -> tuple[str, str]:
    """Deposit a literal string as one corpus entry.

    Sugar over the same backend seam the MCP ``deposit_text`` tool uses; the
    only difference is attribution, which names the operator rather than the
    agent asserter identity.
    """
    backend = get_backend()
    return await backend.deposit_text(
        text=text,
        tags=tags,
        store=DEFAULT_STORE,
        deposited_by=deposited_by,
        source_type=source_type,
    )


async def _deposit_split(
    source: str,
    deposited_by: str,
    source_type: str | None,
    tags: list[str],
) -> list[tuple[str, str]]:
    """Split a multi-entry local file into N corpus entries."""
    backend = get_backend()
    outcomes = await backend.deposit_file_split(
        Path(source),
        deposited_by=deposited_by,
        source_type=source_type,
        tags=tags,
    )
    return [(o.entry_id, o.snapshot_id) for o in outcomes]
