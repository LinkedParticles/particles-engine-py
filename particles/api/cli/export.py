"""export verb — emit the knowledge base to a registered exporter format.

The canonical list of formats is whatever ``get_exporters()`` returns
(see ``particles/exporters/registry.py::_make_exporters``). The help
text below mirrors that list — keep it in sync when adding a new
exporter (Typer evaluates help text at decoration time, so the format
list can't be derived from the registry at runtime)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from particles.api.cli import app, run
from particles.db import session_scope


@app.command("export")
def export_cmd(
    format: str = typer.Argument(
        ...,
        help="Export format: obsidian | anki | wiki | logseq | jsonl | notion | graph",
    ),
    output: str | None = typer.Argument(
        None,
        help=(
            "Output path (directory for obsidian / wiki / logseq, file for anki / "
            "jsonl / graph). Optional for obsidian when ``obsidian.default_output_path`` "
            "is set in config.yaml; required for the filesystem exporters. The "
            "notion exporter is an API target and takes NO output path."
        ),
    ),
    min_particles: int | None = typer.Option(
        None,
        "--min-particles",
        help="Obsidian/Wiki/Logseq: minimum particle count per subject (0 = all; wiki default 3)",
    ),
    min_links: int | None = typer.Option(
        None,
        "--min-links",
        help="Obsidian/Logseq: minimum graph link count per subject (0 = all)",
    ),
    deck_name: str = typer.Option("Particles", "--deck-name", help="Anki: root deck name prefix"),
    min_particle_confidence: float | None = typer.Option(
        None,
        "--min-particle-confidence",
        help=(
            "Cross-exporter: drop particles with effective_confidence "
            "below this threshold before any per-exporter downstream step. "
            "Overrides config.exporter_common.min_particle_confidence."
        ),
    ),
    regenerate_all: bool = typer.Option(
        False,
        "--regenerate-all",
        help="Wiki: bypass the per-subject input-hash cache and rewrite every article",
    ),
    invalidate_stale_links: bool = typer.Option(
        False,
        "--invalidate-stale-links",
        help=(
            "Wiki/Obsidian/Logseq: scan cached article bodies for [[X]] wikilinks; "
            "invalidate any article whose wikilinked subjects' canonical names "
            "have drifted since render. Cheaper than --regenerate-all "
            "when only a few subjects renamed."
        ),
    ),
    subjects: str | None = typer.Option(
        None,
        "--subjects",
        help="Wiki: comma-separated canonical subject names to limit the export to",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Wiki: report cache hits + regen count + token estimate "
            "without writing or calling the LLM"
        ),
    ),
    with_synthesis: bool = typer.Option(
        False,
        "--with-synthesis",
        help=(
            "Obsidian/Logseq: splice LLM-synthesised prose articles into per-subject "
            "notes. Requires ANTHROPIC_API_KEY. Shares the synthesis "
            "cache with the wiki exporter, so running multiple synthesising "
            "exporters pays LLM cost once per subject."
        ),
    ),
    without_synthesis: bool = typer.Option(
        False,
        "--without-synthesis",
        help=(
            "Wiki: render every article as the deterministic structured listing — "
            "no LLM call, no ANTHROPIC_API_KEY, reproducible output. "
            "Bypasses the synthesis cache so existing LLM articles are replaced."
        ),
    ),
    include_non_asserted: bool = typer.Option(
        False,
        "--include-non-asserted",
        help=(
            "Include non-asserted particles — a document's rejected / superseded / "
            "deferred / counterfactual prose (polarity DECLINED / HYPOTHETICAL). "
            "Excluded from the rendered surface by default; the "
            "round-trippable `interchange` export always keeps them."
        ),
    ),
    subject: str | None = typer.Option(
        None,
        "--subject",
        help=(
            "Graph: render one Subject's neighbourhood — a subject "
            "id or an exact (case-insensitive) canonical name / alias. Scope "
            "is mandatory for the graph exporter — pass exactly one of "
            "--subject or --query; a whole-store render does not exist."
        ),
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        help=(
            "Graph: render one query's retrieval set — the picture of the "
            "knowledge a query consults (top graph.query_top_k hits + their "
            "subjects). Mutually exclusive with --subject."
        ),
    ),
    inconsistency: str | None = typer.Option(
        None,
        "--inconsistency",
        help=(
            "Graph: render one contradiction's evidence — the "
            "INCONSISTENCY particle (full id or unique prefix) as the anchor, "
            "its two disputant beliefs with their true statuses, their "
            "subjects and sources. Mutually exclusive with the other scopes."
        ),
    ),
    graph_manifest: str | None = typer.Option(
        None,
        "--manifest",
        help=(
            "Graph: with --section, render a projection manifest "
            "section's deterministic selection."
        ),
    ),
    graph_section: str | None = typer.Option(
        None,
        "--section",
        help="Graph: the manifest section's region id or exact title (with --manifest).",
    ),
    hops: int = typer.Option(
        1,
        "--hops",
        help="Graph: neighbourhood radius for --subject scope (clamped to graph.max_hops)",
    ),
    history: bool = typer.Option(
        False,
        "--history",
        help=(
            "Graph: include retired supersession-chain ancestors as ghosts "
            "(dashed, with the successor chain in the panel); the page gets a "
            "client-side history toggle"
        ),
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help=(
            "Graph: render the graph as believed at this ISO-8601 instant "
            "(single-instant lens; undatable retirements are excluded "
            "fail-closed and disclosed). Two exports at two instants make the "
            "static belief-history demo."
        ),
    ),
    max_nodes: int | None = typer.Option(
        None,
        "--max-nodes",
        help="Graph: per-run node cap (clamped to graph.max_nodes; truncation is disclosed)",
    ),
    database_id: str | None = typer.Option(
        None,
        "--database-id",
        help=(
            "Notion: the target database id to sync subjects into for this run "
            "(overrides config.notion.database_id). Share that database "
            "with your integration first. The NOTION_API_KEY token is read from "
            "the environment — never passed as a flag."
        ),
    ),
    no_update_blocks: bool = typer.Option(
        False,
        "--no-update-blocks",
        help=(
            "Notion: create-only — write a page's particle blocks once and never "
            "rewrite the managed block range on re-sync, preserving hand-edits "
            ". Default behaviour owns the managed range and "
            "overwrites it so re-sync is idempotent."
        ),
    ),
) -> None:
    """Export the knowledge base to an external format.

    Available formats: obsidian, anki, wiki, logseq, jsonl, notion, graph.

    \b
        particles export obsidian ./my-vault
        particles export obsidian ./my-vault --min-particles=1 --min-links=2
        particles export anki ./deck.txt --deck-name="Numismatics" --min-particle-confidence=0.7
        particles export wiki ./my-wiki                    # incremental
        particles export wiki ./my-wiki --dry-run          # cost estimate
        particles export wiki ./my-wiki --regenerate-all   # bypass cache
        particles export wiki ./my-wiki --without-synthesis # deterministic, no LLM
        particles export wiki ./my-wiki --subjects "Pfennig,GDR"
        particles export logseq ./my-graph                 # pages/ + bullet outline
        particles export logseq ./my-graph --with-synthesis
        particles export notion --dry-run                  # plan, zero API writes
        particles export notion --database-id=abc123…      # sync into a Notion DB
        particles export notion --no-update-blocks         # create-only (keep hand-edits)
        particles export graph out.html --subject <id>     # one Subject's neighbourhood
        particles export graph out.html --query "why X?"   # one query's retrieval set
        particles export graph out.html --subject <id> --history --as-of 2006-08-24

    The notion exporter is an API target: it takes NO output path and
    requires the NOTION_API_KEY environment variable (mint an internal
    integration at https://www.notion.so/my-integrations and share the target
    database with it). Run with --dry-run first to see the plan without writing.
    """
    # (b): export renders the whole store to the local filesystem and
    # has no remote bulk-read route, so refuse in remote mode rather than render
    # the laptop's near-empty store while the daily loop runs on the engine.
    from particles.api.cli._remote import refuse_remote_sync

    refuse_remote_sync("export")

    import os

    from particles.exporters.registry import get_exporters, required_secret

    exporters = get_exporters()
    if format not in exporters:
        typer.echo(f"Unknown format: {format!r}. Available: {list(exporters)}", err=True)
        raise typer.Exit(1)

    # Fail-closed pre-flight (generic over REQUIRES_SECRET): if the
    # chosen exporter declares a required secret, verify the env var is present
    # BEFORE any store read or network call, so the operator learns early. The
    # exporter's first-statement getter call is the authoritative check (a
    # programmatic caller bypasses this); this only fails faster. We check
    # presence here — never echo the value.
    secret_var = required_secret(exporters[format])
    if secret_var is not None and not os.environ.get(secret_var):
        typer.echo(
            f"{secret_var} is required for `export {format}` but is not set. "
            f"Set it in the environment before exporting (the token is never a "
            f"CLI flag). For notion: mint an internal integration at "
            f"https://www.notion.so/my-integrations and share the target "
            f"database with it.",
            err=True,
        )
        raise typer.Exit(2)

    from particles.config import get_config

    obs_cfg = get_config().obsidian
    wiki_cfg = get_config().wiki
    exporter_common_cfg = get_config().exporter_common

    # API-target exporters (notion) take no filesystem path — output is
    # None. Filesystem exporters resolve the output path: explicit CLI argument
    # wins; otherwise fall back to the per-format default in config.yaml
    # (currently only Obsidian exposes one — wiki/anki still require an explicit
    # path). ``~`` is expanded so operators can put paths like
    # ``~/Library/Mobile Documents/iCloud~md~obsidian/Documents/MyVault``
    # in config.yaml without resorting to absolute home directories.
    output_path: Path | None
    if format == "notion":
        output_path = None
    else:
        effective_output: str | None = output
        if effective_output is None and format == "obsidian":
            effective_output = obs_cfg.default_output_path
        if effective_output is None:
            typer.echo(
                f"No output path supplied. Pass it as an argument, or set "
                f"`{format}.default_output_path` in config.yaml.",
                err=True,
            )
            raise typer.Exit(2)
        output_path = Path(effective_output).expanduser()
    # Wiki picks up its own default (3) when --min-particles is unset; Obsidian
    # picks up its own (0). Caller-supplied --min-particles wins for whichever
    # format is selected.
    effective_min_particles: int
    if min_particles is not None:
        effective_min_particles = min_particles
    elif format == "wiki":
        effective_min_particles = wiki_cfg.min_particles
    else:
        effective_min_particles = obs_cfg.min_particles
    subject_list: list[str] | None = (
        [s.strip() for s in subjects.split(",") if s.strip()] if subjects else None
    )

    # Graph: parse --as-of like the query verb does.
    as_of_dt: datetime | None = None
    if as_of is not None:
        try:
            as_of_dt = datetime.fromisoformat(as_of)
        except ValueError:
            typer.echo(
                f"Invalid --as-of value {as_of!r}: expected an ISO-8601 date or "
                f"datetime (e.g. 2006-08-24 or 2006-08-24T12:00:00Z).",
                err=True,
            )
            raise typer.Exit(2) from None

    # Resolve the cross-exporter threshold. Precedence:
    # explicit --min-particle-confidence > config.exporter_common >
    # default 0.0.
    effective_min_particle_confidence: float
    if min_particle_confidence is not None:
        effective_min_particle_confidence = min_particle_confidence
    else:
        effective_min_particle_confidence = exporter_common_cfg.min_particle_confidence

    options: dict[str, object] = {
        "min_particles": effective_min_particles,
        "min_links": min_links if min_links is not None else obs_cfg.min_links,
        "deck_name": deck_name,
        "min_particle_confidence": effective_min_particle_confidence,
        "regenerate_all": regenerate_all,
        "invalidate_stale_links": invalidate_stale_links,
        "subjects": subject_list,
        "dry_run": dry_run,
        "with_synthesis": with_synthesis,
        "without_synthesis": without_synthesis,
        "include_non_asserted": include_non_asserted,
        # Notion. None database_id falls back to config.notion at
        # call time; no_update_blocks opts into create-only re-sync.
        "database_id": database_id,
        "no_update_blocks": no_update_blocks,
        # Graph. Scope is mandatory (exactly one of subject / query /
        # inconsistency / manifest+section); the operation enforces it
        # so a programmatic caller gets the same error.
        "subject": subject,
        "query": query,
        "inconsistency": inconsistency,
        "manifest": graph_manifest,
        "section": graph_section,
        "hops": hops,
        "history": history,
        "as_of": as_of_dt,
        "max_nodes": max_nodes,
    }

    # Per-subject progress lines emitted by the exporters go through the
    # module logger. Synthesis is slow (per-subject LLM calls) and a long
    # export with no output looks frozen, so surface INFO to stdout for
    # the duration of the run. Both wiki and obsidian-with-synthesis
    # emit progress; for obsidian without --with-synthesis the existing
    # summary suffices, so we don't install the handler there.
    progress_handler: logging.StreamHandler[Any] | None = None
    progress_loggers: list[tuple[logging.Logger, int]] = []
    # Notion does one API round-trip per subject (slow), so surface its
    # per-page progress lines like wiki/obsidian-with-synthesis do.
    if format in ("wiki", "notion") or (format == "obsidian" and with_synthesis):
        progress_handler = logging.StreamHandler()
        progress_handler.setLevel(logging.INFO)
        progress_handler.setFormatter(logging.Formatter("%(message)s"))
        for logger_name in (
            "particles.exporters.wiki",
            "particles.exporters.obsidian",
            "particles.exporters.notion",
        ):
            lg = logging.getLogger(logger_name)
            progress_loggers.append((lg, lg.level))
            lg.setLevel(logging.INFO)
            lg.addHandler(progress_handler)

    from particles.exporters.summaries import BaseExporterSummary

    async def _run_export() -> BaseExporterSummary:
        async with session_scope() as session:
            return await exporters[format].export(session, output_path, **options)

    try:
        summary = run(_run_export())
    except ValueError as exc:
        # Exporters raise ValueError for usage errors (missing/duplicate graph
        # scope, an unknown --subject id, a missing output path). One clean
        # line on stderr, never a traceback.
        typer.echo(f"export {format} failed: {exc}", err=True)
        raise typer.Exit(2) from None
    finally:
        # Always tear the progress handler back down; nested exports in
        # one process otherwise stack handlers and double-print.
        if progress_handler is not None:
            for lg, prev in progress_loggers:
                lg.removeHandler(progress_handler)
                lg.setLevel(prev)

    summary_dict = summary.model_dump(exclude_none=True)
    # API-target exporters (notion) have no output path; name the target by its
    # database id from the summary instead.
    target = (
        str(output_path)
        if output_path is not None
        else (f"Notion database {summary_dict.get('database_id', '?')}")
    )
    if summary_dict.get("dry_run"):
        typer.echo(f"Dry-run for {target}")
    else:
        typer.echo(f"Exported to {target}")
    for key, val in summary_dict.items():
        typer.echo(f"  {key}: {val}")
