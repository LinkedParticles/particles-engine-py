# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""import sub-Typer — bulk-onboard existing knowledge bases.

``particles import vault <dir>`` walks an existing Obsidian vault (or any
directory of Markdown notes) and registers each ``.md`` file as a
``LOCAL_MARKDOWN`` corpus entry. ``particles import project <dir>``
walks a software-project tree and registers each source file (``.py`` by
default) as a ``PYTHON_SOURCE`` corpus entry. ``particles import
web-clipper <dir>`` walks a frontmatter-Markdown captures folder and deposits
each capture as a ``WEB_PAGE`` entry with the provenance its frontmatter carries
(URL, publication date, tags) restored. After any of these, an
operator can run ``particles extract --all-pending`` and ``particles lint``
against the deposited corpus without rebuilding anything.

The CLI module is named ``import_vault.py`` (with underscore) because
``import`` is a Python keyword and would conflict with normal import
statements. It also hosts ``import project`` and ``import web-clipper`` — all
three commands share the one ``import`` sub-Typer. The user-facing command name
registered on the Typer app is the plain word ``import``.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from particles.api.cli import app, run
from particles.api.cli._estimate_text import duration_text, range_text
from particles.api.cli._logging import configure_logging
from particles.api.cli._progress import progress_line
from particles.api.client import get_backend
from particles.core.schema import Mutability, SourceType
from particles.db import session_scope

if TYPE_CHECKING:
    from particles.extraction.migration_preview import MigrationPreview
    from particles.operations.vault_import import VaultImportReport

log = logging.getLogger(__name__)


class VaultMutability(StrEnum):
    """The two promises ``import vault --mutability`` accepts."""

    MUTABLE = "MUTABLE"
    STABLE = "STABLE"


# Named with a trailing underscore because ``import`` is a Python keyword
# and we cannot bind the module-level Typer instance to that identifier.
import_app = typer.Typer(
    help="Bulk-onboard existing knowledge bases.",
    no_args_is_help=True,
)
app.add_typer(import_app, name="import")


def _refuse_in_remote_mode(verb: str) -> None:
    """Refuse a local-only ``import`` verb when configured for a remote engine.

    ``import vault`` / ``import project`` walk the **client's** filesystem and
    open a local ``session_scope()`` directly — they have no engine endpoint.
    In remote mode (``engine.base_url`` set ⇒ ``HttpBackend``) that
    local DB is *not* the canonical engine store, so the verb would write to the
    wrong place — or, on a thin client that never ran ``db init``, fail with the
    confusing generic "Database tables not found …" instead of being told the
    verb is local-only. Fail fast with an actionable message.

    Mirrors :func:`particles.api.cli._remote.refuse_remote_sync` (same
    ``get_backend().remote`` detection seam, same echo + ``Exit(1)`` shape) but
    names the per-file ``deposit`` escape hatch the bulk verbs lack — the reason
    a verb-specific message is worth more here than the generic one.
    """
    if get_backend().remote:
        typer.echo(
            f"Error: `import {verb}` is a local-only operation (it "
            f"walks the client's filesystem and has no engine endpoint). In "
            f"remote mode it cannot reach your engine store. Either run this "
            f"command on the engine host, or use `particles deposit <file>` per "
            f"file (which routes to the engine).",
            err=True,
        )
        raise typer.Exit(1)


def _deposit_summary(total: int, unchanged: int, root: Path) -> str:
    """The closing line of ``import vault`` / ``import project``.

    A file whose content matches its entry's latest snapshot writes nothing and
    queues no extraction, so the line says how many of the walked files were
    skipped that way; ``Deposited N file(s)`` still counts every file walked.
    """
    line = f"Deposited {total} file(s) from {root}."
    if unchanged:
        line += f" {unchanged} unchanged (skipped); {total - unchanged} new or changed."
    return line


async def _import_tree_remote(
    paths: list[Path],
    *,
    source_type: str,
    deposited_by: str,
    tags: list[str],
    verbose: bool,
    out_unchanged: list[bool] | None = None,
) -> list[tuple[str, str]]:
    """Route each walked file to the engine via ``backend.deposit_file``.

    The **remote arm** of ``import vault`` / ``import project``: the verb has
    already walked the client's tree with the shared ignore policy
    (:func:`particles.corpus.deposit.iter_vault_files` /
    :func:`~particles.corpus.deposit.iter_project_files`); here each matched file
    is uploaded to the engine's existing ``POST /corpus/deposit/file`` (engine-side
    source-type handling + content-hash dedup), reusing the proven per-file
    ``deposit`` upload path — **no new endpoint**.

    Per-file failures are logged and **skipped** so one bad file does not abort
    the tree; re-running is idempotent (content-hash dedup), so a retry picks up
    only the failures. A lost connection to the engine itself is **not** swallowed
    — it propagates so ``run()`` reports it once instead of logging a skip per
    remaining file.
    """
    import httpx

    from particles.api.client.http import EngineUnreachableError

    backend = get_backend()
    results: list[tuple[str, str]] = []
    total = len(paths)
    for i, path in enumerate(paths, start=1):
        if verbose:
            progress_line(f"[{i}/{total}] depositing {path}")
        try:
            outcome = await backend.deposit_file(
                path,
                deposited_by=deposited_by,
                source_type=source_type,
                tags=tags,
                content_date=None,
            )
            results.append((outcome.entry_id, outcome.snapshot_id))
            if out_unchanged is not None:
                out_unchanged.append(outcome.unchanged)
        except (EngineUnreachableError, httpx.TransportError):
            raise  # the engine is gone — stop rather than log a skip per file
        except Exception as exc:  # noqa: BLE001 — per-file isolation is intentional
            typer.echo(f"  skipped {path}: {exc}", err=True)
            log.warning("import: skipped %s: %s", path, exc)
    return results


@import_app.command("vault")
def import_vault_cmd(
    vault_dir: Path = typer.Argument(
        ...,
        help="Path to an Obsidian vault (or any directory of Markdown notes).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    deposited_by: str = typer.Option("operator", help="Agent or operator ID."),
    tags: str | None = typer.Option(
        None, help="Comma-separated tags applied to every deposited entry."
    ),
    mutability: VaultMutability | None = typer.Option(
        None,
        "--mutability",
        case_sensitive=False,
        help=(
            "Deposit this run's notes MUTABLE (the default: an edit retires what the note "
            "no longer says) or STABLE (an archive folder that will not change)."
        ),
    ),
    confirm_absent: bool = typer.Option(
        False,
        "--confirm-absent",
        help=(
            "Record the notes this walk no longer finds even when a guard asks first: the "
            "first deletion under this folder, or a mass deletion."
        ),
    ),
    keep_missing: bool = typer.Option(
        False,
        "--keep-missing",
        help="Record no deletion this run, for a partial checkout made on purpose.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print per-file progress while depositing."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs from deposit/fetch."),
) -> None:
    """Walk a Markdown vault and deposit every ``.md`` file as ``LOCAL_MARKDOWN``.

    Recursively walks ``vault_dir`` (skipping any path under a ``_`` or ``.``
    component: Obsidian's ``.obsidian/`` settings, ``_attachments/``, etc., and
    any folder holding a Particles export) and registers each Markdown file in
    the corpus, identified by its path. Notes are living documents: each is
    deposited ``MUTABLE``, so after an edit the next ``extract`` retires the
    claims the note no longer states, with no model call. A note whose content
    matches its latest snapshot is reported unchanged and skipped, so only new
    and edited notes are queued for extraction.

    A previously imported note this walk no longer finds is recorded as no
    longer found, and the next ``extract`` retires its claims. Nothing is
    erased: the note's text stays in the corpus and ``query --as-of`` still
    shows what it said. The first deletion under a folder, and any walk that
    would record more than a quarter of its notes as gone, stop and list the
    notes (exit 1) until you pass ``--confirm-absent``. Run the import after
    your sync client reports it is idle.

    Typical onboarding workflow:

    \b
        particles import vault ~/Documents/MyVault
        particles extract --all-pending
        particles lint
    """
    configure_logging(verbose, debug)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    unchanged: list[bool] = []
    if confirm_absent and keep_missing:
        typer.echo("Error: --confirm-absent and --keep-missing cannot be combined.", err=True)
        raise typer.Exit(2)
    if get_backend().remote:
        if mutability is not None or confirm_absent or keep_missing:
            typer.echo(
                "Note: against a remote engine the notes are uploaded one file at a time, "
                "so --mutability, --confirm-absent and --keep-missing do not apply and "
                "deleted notes are not recorded.",
                err=True,
            )
        # walk the client's tree with the shared ignore policy, then
        # route each file to the engine via backend.deposit_file (no new endpoint).
        from particles.corpus.deposit import iter_vault_files

        results = run(
            _import_tree_remote(
                iter_vault_files(vault_dir),
                source_type=SourceType.LOCAL_MARKDOWN,
                deposited_by=deposited_by,
                tags=tag_list,
                verbose=verbose,
                out_unchanged=unchanged,
            )
        )
        typer.echo(_deposit_summary(len(results), sum(unchanged), vault_dir))
        return
    report = run(
        _import_vault(
            vault_dir,
            deposited_by,
            tag_list,
            verbose,
            mutability=Mutability(mutability.value) if mutability else Mutability.MUTABLE,
            confirm_absent=confirm_absent,
            keep_missing=keep_missing,
        )
    )
    for line in _vault_report_lines(report, vault_dir):
        typer.echo(line)
    if report.refusal is not None:
        raise typer.Exit(1)


async def _import_vault(
    vault_dir: Path,
    deposited_by: str,
    tags: list[str],
    verbose: bool,
    *,
    mutability: Mutability,
    confirm_absent: bool,
    keep_missing: bool,
) -> VaultImportReport:
    from particles.operations.vault_import import import_vault

    progress = progress_line if verbose else None
    async with session_scope() as session:
        report = await import_vault(
            session,
            vault_dir,
            deposited_by=deposited_by,
            tags=tags,
            mutability=mutability,
            confirm_absent=confirm_absent,
            keep_missing=keep_missing,
            progress=progress,
        )
        await session.commit()
    return report


#: The budget the import suggests for a first extraction: what a
#: note-taker trying the tool would spend, not a default anything applies.
_START_BUDGET_USD = 3.0

#: How many missing notes a refusal lists before collapsing the rest to a count.
_MISSING_LIST_CAP = 20


def _vault_report_lines(report: VaultImportReport, root: Path) -> list[str]:
    """The summary ``import vault`` prints (§4, §5)."""
    missing = len(report.missing)
    parts = [f"{report.changed} new or changed"]
    if report.unchanged_count:
        parts.append(f"{report.unchanged_count} unchanged")
    if missing:
        parts.append(f"{missing} no longer found")
    owed = report.changed or report.absences
    lines = [
        f"Deposited {len(report.results)} note(s) from {root}: {', '.join(parts)}"
        + ("; changes take effect at the next extract." if owed else ".")
    ]
    lines += _queued_lines(report)
    if report.export_dirs:
        lines.append(
            f"Skipped {len(report.export_dirs)} folder(s) holding a Particles export "
            "(.particles-export.json): the store's own rendering is not your notes."
        )
    if report.unreadable:
        lines.append(
            f"Could not read {len(report.unreadable)} note(s); they were skipped and are "
            "not treated as deleted:"
        )
        lines += [f"  {path}" for path in report.unreadable]
    if report.reclassified:
        lines.append(
            f"Reclassified {len(report.reclassified)} earlier note entr(ies) from STABLE to "
            "MUTABLE; nothing was retired."
        )
        lines.append(
            f"  {report.backfill_entries} of them hold more than one version; "
            f"`particles corpus refresh --backfill-cascade` would retire "
            f"{report.backfill_claims} claim(s) their older versions stated "
            "(it previews first and asks for --yes)."
        )
    if report.walk_errors:
        lines.append(
            f"The walk met {len(report.walk_errors)} error(s) reading {root}, so no note "
            "was recorded as no longer found:"
        )
        lines += [f"  {error}" for error in report.walk_errors]
    if not missing:
        return lines
    share = f"{100 * report.absent_fraction:.0f}% of {report.previously_imported} imported before"
    if report.absences:
        lines.append(
            f"Recorded {missing} note(s) as no longer found ({share}); the next extract "
            "retires their claims."
        )
        lines.append(
            "  This is not an erasure: the text stays in the corpus and `query --as-of` "
            "shows it. `particles corpus delete <entry>` erases a note."
        )
    elif report.kept_missing:
        lines.append(f"{missing} note(s) no longer found ({share}); --keep-missing recorded none.")
    elif report.refusal is not None:
        reason = {
            "first": "This is the first deletion under this folder, so it is shown before "
            "anything is retired.",
            "always": "import_vault.confirm_absent is `always`, so every deletion is shown "
            "before anything is retired.",
            "mass": "That is more notes than a deletion usually removes; an unmounted "
            "volume, an unfinished sync or a folder one level too deep looks the same.",
        }[report.refusal]
        lines.append(f"{missing} note(s) no longer found ({share}), and none were recorded.")
        lines.append(f"  {reason}")
        for note in report.missing[:_MISSING_LIST_CAP]:
            lines.append(f"  {_display_path(note.path, root)}")
        if missing > _MISSING_LIST_CAP:
            lines.append(f"  … and {missing - _MISSING_LIST_CAP} more")
        lines.append(
            "  Re-run with --confirm-absent to record them, or --keep-missing to keep them. "
            "Everything else was deposited."
        )
    return lines


def _queued_lines(report: VaultImportReport) -> list[str]:
    """The price of the extraction the import queued, and how to start small."""
    plan = report.queued
    if plan is None or plan.total is None or not plan.total.calls:
        return []
    lines = [
        f"  The next extract reads {len(plan.snapshots)} note(s): {range_text(plan.total)}, "
        f"{duration_text(plan.total.wall_seconds)}."
    ]
    alt = plan.alternative
    if (
        alt is not None
        and alt.high_usd is not None
        and plan.total.high_usd is not None
        and alt.high_usd < plan.total.high_usd
    ):
        lines.append(f"  On {alt.model}: {range_text(alt, name_model=False)}.")
    if report.flat_mtimes:
        lines.append(
            "  Most notes share one modification time (a copy or a sync rewrote them), so "
            "the newest-first order is path order. Importing one folder first starts smaller."
        )
    lines.append("  To start small, newest notes first:")
    lines.append(f"    particles extract --all-pending --budget-usd {_START_BUDGET_USD:g}")
    return lines


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


@import_app.command("project")
def import_project_cmd(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to a software-project tree (walked recursively).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    deposited_by: str = typer.Option("operator", help="Agent or operator ID."),
    tags: str | None = typer.Option(
        None, help="Comma-separated tags applied to every deposited entry."
    ),
    ext: str | None = typer.Option(
        None,
        "--ext",
        help=(
            "Comma-separated file extensions to deposit this run (e.g. '.py'), "
            "overriding the configured import_project.extensions set."
        ),
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print per-file progress while depositing."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs from deposit/fetch."),
) -> None:
    """Walk a project tree and deposit every source file as ``PYTHON_SOURCE``.

    Recursively walks ``project_dir`` for source files (``.py`` by default; see
    ``import_project.extensions``), skipping dot-prefixed components and the
    configured build/cache directories (``import_project.ignore_dirs``), but
    keeping underscore-prefixed module files (``__init__.py`` / ``_shared.py``).
    Re-running on the same tree is idempotent: ``content_hash`` deduplication
    means only changed files get a new snapshot, and the rest are reported
    unchanged.

    Typical onboarding workflow:

    \b
        particles import project ~/src/myproject
        particles extract --all-pending   # docstring extractor runs
        particles lint                    # code/design drift surfaces
    """
    configure_logging(verbose, debug)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    ext_set = {e.strip() for e in ext.split(",") if e.strip()} if ext else None
    unchanged: list[bool] = []
    if get_backend().remote:
        # walk the client's tree with the shared ignore policy, then
        # route each file to the engine via backend.deposit_file (no new endpoint).
        from particles.corpus.deposit import iter_project_files

        results = run(
            _import_tree_remote(
                iter_project_files(project_dir, ext_set),
                source_type=SourceType.PYTHON_SOURCE,
                deposited_by=deposited_by,
                tags=tag_list,
                verbose=verbose,
                out_unchanged=unchanged,
            )
        )
    else:
        results = run(
            _import_project(project_dir, deposited_by, tag_list, ext_set, verbose, unchanged)
        )
    typer.echo(_deposit_summary(len(results), sum(unchanged), project_dir))


async def _import_project(
    project_dir: Path,
    deposited_by: str,
    tags: list[str],
    extensions: set[str] | None,
    verbose: bool,
    out_unchanged: list[bool] | None = None,
) -> list[tuple[str, str]]:
    from particles.corpus.deposit import deposit_project

    progress = progress_line if verbose else None
    async with session_scope() as session:
        results = await deposit_project(
            session,
            project_dir,
            deposited_by=deposited_by,
            tags=tags,
            extensions=extensions,
            progress=progress,
            out_unchanged=out_unchanged,
        )
        await session.commit()
    return results


@import_app.command("web-clipper")
def import_web_clipper_cmd(
    captures_dir: Path = typer.Argument(
        ...,
        help="Path to an Obsidian Web Clipper captures folder (walked recursively).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    deposited_by: str = typer.Option("web-clipper", help="Agent or operator ID."),
    tags: str | None = typer.Option(
        None, help="Comma-separated tags merged with each capture's frontmatter tags."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print per-file progress while depositing."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs from deposit/fetch."),
) -> None:
    """Walk a frontmatter-Markdown captures folder and deposit each as ``WEB_PAGE``.

    Recursively walks ``captures_dir`` (skipping any path under a ``_`` or ``.``
    component, the vault ignore policy) and deposits each ``.md`` capture with the
    provenance its frontmatter carries restored: the ``source:`` / ``url:`` URL
    becomes the entry's ``uri_r`` (fragment-stripped, **not** fetched), the
    ``published:`` date becomes ``content_published_at`` (below an
    explicit operator date), the frontmatter ``tags:`` merge with ``--tags``, and
    the source type is ``WEB_PAGE``, so a clipping is trustable, decayable, and
    queryable as the web page it is, unlike the same folder run through
    ``import vault``. The frontmatter-stripped **body** is the deposited content.
    A capture whose header is absent / malformed falls back to a plain
    ``LOCAL_MARKDOWN`` body deposit. Re-running is idempotent (body-hash dedup).

    Typical onboarding workflow:

    \b
        particles import web-clipper ~/Obsidian/Clippings
        particles extract --all-pending
        particles lint
    """
    _refuse_in_remote_mode("web-clipper")
    configure_logging(verbose, debug)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    results = run(_import_web_clipper(captures_dir, deposited_by, tag_list, verbose))
    typer.echo(f"Deposited {len(results)} capture(s) from {captures_dir}.")


async def _import_web_clipper(
    captures_dir: Path,
    deposited_by: str,
    tags: list[str],
    verbose: bool,
) -> list[tuple[str, str]]:
    from particles.corpus.deposit import deposit_web_clipper

    progress = progress_line if verbose else None
    async with session_scope() as session:
        results = await deposit_web_clipper(
            session,
            captures_dir,
            deposited_by=deposited_by,
            tags=tags,
            progress=progress,
        )
        await session.commit()
    return results


@import_app.command("mcp-memory")
def import_mcp_memory_cmd(
    export_path: Path = typer.Argument(
        ...,
        help="Path to a reference memory-server memory.jsonl export.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    deposited_by: str = typer.Option(
        "operator", help="Agent or operator ID performing the import."
    ),
    tags: str | None = typer.Option(None, help="Comma-separated tags added to the corpus entry."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Report what the import would produce (counts, what is dropped, a sample) "
            "without depositing or writing anything."
        ),
    ),
    sample: int = typer.Option(
        5, "--sample", min=0, help="With --dry-run: how many mapped records to show."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="With --dry-run: emit the report as JSON instead of text."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs from deposit."),
) -> None:
    """Deposit a reference memory-server ``memory.jsonl`` for migration.

    Brings an existing ``@modelcontextprotocol/server-memory`` graph across:
    entities become Subjects, observations become single-subject particles, and
    relations become two-subject particles, the same encoding
    ``particles memory serve`` reads, so a migrated graph is visible
    through the façade immediately.

    The export is deposited **verbatim** as an ``MCP_MEMORY_EXPORT`` entry
    (``STABLE`` / ``NEVER``: a dump is a record of what was seen, not a live
    handle), and every particle points back at it by line number. Nothing is
    attributed to the incumbent store itself: the SDK never fetched it and
    cannot re-verify it, so provenance names the artifact it actually holds.
    Re-running is idempotent (content-hash dedup).

    One thing does not come across: an entity with no observations that no
    relation names. A store holds a name only through something believed about
    it, so there is nothing to attach it to. The verb lists those entities when
    it runs; an observation-less entity that *is* a relation endpoint migrates,
    with its type.

    Migrated beliefs are deliberately low-confidence: they are second-hand, and
    the incumbent's own scores are preserved as tags rather than becoming
    confidence values. Raise them with ``particles trust set`` once you vouch
    for the source, not by editing the import floor.

    Run it with ``--dry-run`` first. That parses the export and runs the same
    mapping the import runs, then prints what it would produce: entities to
    Subjects, records to particles, everything the mapping drops (including
    the entities that will not migrate, by name), and a sample. It
    opens no store and deposits nothing, so it is safe on a store you care
    about and needs no ``particles db init``. Its counts are what the export
    contributes: on a store that already holds part of it, Subjects re-attach
    and identical claims dedup, so the real import writes no more than this.

    \b
        particles import mcp-memory ~/.mcp/memory.jsonl --dry-run
        particles import mcp-memory ~/.mcp/memory.jsonl
        particles extract --all-pending
        particles lint
    """
    if json_out and not dry_run:
        typer.echo("Error: --json applies to the --dry-run report.", err=True)
        raise typer.Exit(2)
    configure_logging(False, debug)
    if dry_run:
        # Before the remote-mode refusal on purpose: the report is computed from
        # the client's file by Client-layer code and reaches no store, local or
        # remote, so there is nothing for remote mode to get wrong.
        from particles.extraction.mcp_memory import preview_memory_export

        preview = preview_memory_export(
            export_path.read_bytes(), deposited_by=deposited_by, sample_size=sample
        )
        if json_out:
            typer.echo(json.dumps(preview.to_dict(), indent=2, ensure_ascii=False))
        else:
            for line in _render_migration_preview(preview, export_path):
                typer.echo(line)
        if not preview.particles:
            raise typer.Exit(1)
        return
    _refuse_in_remote_mode("mcp-memory")
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    entry_id, _snapshot_id = run(_import_mcp_memory(export_path, deposited_by, tag_list))
    typer.echo(f"Deposited {export_path} as corpus entry {entry_id}.")
    # Said here, at the door, rather than only as a quality note on a later
    # `extract`: an entity nothing is believed about is the one thing this verb
    # leaves behind, and the user should learn that before relying on the
    # migrated graph. The parser is Client-side and store-free, so
    # reading the export twice costs nothing.
    for line in _unmigrated_entity_lines(export_path):
        typer.echo(line)
    typer.echo(
        "Migrated beliefs are second-hand: they carry calibration source IMPORTED and a low "
        "import-floor confidence, and the source store's own scores do not carry over. "
        "Use `particles trust set` to raise the whole export once you vouch for it."
    )
    typer.echo("Next: `particles extract --all-pending` to turn the export into particles.")


def _render_migration_preview(preview: MigrationPreview, export_path: Path) -> list[str]:
    """Render a migration dry-run report as text.

    Written against the format-independent report shape, not against one
    incumbent, so the next ``import <incumbent>`` verb reuses it unchanged.
    """
    floor = ", ".join(f"{value:g}" for value in preview.confidence_values) or "n/a"
    calibration = ", ".join(preview.calibration_sources) or "n/a"
    lines = [
        f"Dry run: nothing was deposited or written. Report for {export_path}",
        "",
        "In the export:",
        *(f"  {count:>7}  {kind}" for kind, count in preview.records.items()),
        "",
        "The import would produce (at most; an existing store re-attaches and dedups):",
        f"  {preview.subjects:>7}  Subjects",
        f"  {preview.particles:>7}  particles  "
        f"({preview.single_subject_particles} about one Subject, "
        f"{preview.multi_subject_particles} linking several)",
        f"           confidence {floor}, calibration source {calibration}",
    ]
    if preview.entities_without_records:
        kept = len(preview.entities_without_records) - len(preview.entities_lost)
        lines += [
            "",
            f"Entities with no observations: {len(preview.entities_without_records)}. "
            "They produce no particle.",
            f"  {len(preview.entities_lost):>7}  would NOT survive the import"
            + (f": {_name_list(preview.entities_lost)}" if preview.entities_lost else ""),
            f"  {kept:>7}  survive as a relation endpoint, with their entity type",
        ]
    lines += ["", "Dropped from the particles, preserved in the export:"]
    lines += [f"  - {note}" for note in preview.dropped] or ["  (nothing)"]
    if preview.sample:
        lines += ["", f"Sample ({len(preview.sample)} of {preview.particles}):"]
        lines += [
            f"  [{item.location or '?'}] {' + '.join(item.subjects)}: {item.content}"
            for item in preview.sample
        ]
    if preview.particles:
        lines += ["", "Next: run again without --dry-run, then `particles extract --all-pending`."]
    else:
        lines += ["", "Nothing in this export would become a particle."]
    return lines


def _name_list(names: list[str], limit: int = 10) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, and {len(names) - limit} more"


def _unmigrated_entity_lines(export_path: Path) -> list[str]:
    """The export's observation-less entities, as the lines the verb prints."""
    from particles.extraction.mcp_memory import (
        empty_entity_notes,
        empty_entity_report,
        parse_memory_jsonl,
    )

    report = empty_entity_report(parse_memory_jsonl(export_path.read_bytes()))
    lines = [f"Note: {note}" for note in empty_entity_notes(report)]
    if report.dropped:
        lines.append(
            "      To keep one, recreate it after the swap with the memory server's "
            "`create_entities` tool (it accepts an empty `observations` list)."
        )
    return lines


async def _import_mcp_memory(
    export_path: Path,
    deposited_by: str,
    tags: list[str],
) -> tuple[str, str]:
    from particles.core.schema import FetchPolicy, Mutability
    from particles.corpus.deposit import deposit_file
    from particles.extraction.mcp_memory import SOURCE_TYPE

    async with session_scope() as session:
        result = await deposit_file(
            session,
            export_path,
            deposited_by=deposited_by,
            # An export is a frozen dump of a store we do not own: there is
            # nothing to re-fetch, and the incumbent may be gone.
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.NEVER,
            source_type=SOURCE_TYPE,
            tags=tags,
        )
        await session.commit()
    return result
