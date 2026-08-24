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

import logging
from pathlib import Path

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.cli._progress import progress_line
from particles.api.client import get_backend
from particles.core.schema import SourceType
from particles.db import session_scope

log = logging.getLogger(__name__)

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
    open a local ``session_scope()`` directly — they have no engine endpoint
    . In remote mode (``engine.base_url`` set ⇒ ``HttpBackend``) that
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


async def _import_tree_remote(
    paths: list[Path],
    *,
    source_type: str,
    deposited_by: str,
    tags: list[str],
    verbose: bool,
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print per-file progress while depositing."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs from deposit/fetch."),
) -> None:
    """Walk a Markdown vault and deposit every ``.md`` file as ``LOCAL_MARKDOWN``.

    Recursively walks ``vault_dir`` (skipping any path under a ``_`` or ``.``
    component — Obsidian's ``.obsidian/`` settings, ``_attachments/``, etc.)
    and registers each Markdown file in the corpus. Re-running on the same
    vault is idempotent — existing ``content_hash`` deduplication means
    unchanged files are not re-deposited.

    Typical onboarding workflow:

    \b
        particles import vault ~/Documents/MyVault
        particles extract --all-pending
        particles lint
    """
    configure_logging(verbose, debug)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    if get_backend().remote:
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
            )
        )
    else:
        results = run(_import_vault(vault_dir, deposited_by, tag_list, verbose))
    typer.echo(f"Deposited {len(results)} file(s) from {vault_dir}.")


async def _import_vault(
    vault_dir: Path,
    deposited_by: str,
    tags: list[str],
    verbose: bool,
) -> list[tuple[str, str]]:
    from particles.corpus.deposit import deposit_vault

    progress = progress_line if verbose else None
    async with session_scope() as session:
        results = await deposit_vault(
            session,
            vault_dir,
            deposited_by=deposited_by,
            tags=tags,
            progress=progress,
        )
        await session.commit()
    return results


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
    configured build/cache directories (``import_project.ignore_dirs``) — but
    keeping underscore-prefixed module files (``__init__.py`` / ``_shared.py``).
    Re-running on the same tree is idempotent: ``content_hash`` deduplication
    means only changed files get a new snapshot.

    Typical onboarding workflow:

    \b
        particles import project ~/src/myproject
        particles extract --all-pending   # docstring extractor runs
        particles lint                    # code/design drift surfaces
    """
    configure_logging(verbose, debug)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    ext_set = {e.strip() for e in ext.split(",") if e.strip()} if ext else None
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
            )
        )
    else:
        results = run(_import_project(project_dir, deposited_by, tag_list, ext_set, verbose))
    typer.echo(f"Deposited {len(results)} file(s) from {project_dir}.")


async def _import_project(
    project_dir: Path,
    deposited_by: str,
    tags: list[str],
    extensions: set[str] | None,
    verbose: bool,
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
    the source type is ``WEB_PAGE`` — so a clipping is trustable, decayable, and
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
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs from deposit."),
) -> None:
    """Deposit a reference memory-server ``memory.jsonl`` for migration.

    Brings an existing ``@modelcontextprotocol/server-memory`` graph across:
    entities become Subjects, observations become single-subject particles, and
    relations become two-subject particles — the same encoding
    ``particles memory serve`` reads, so a migrated graph is visible
    through the façade immediately.

    The export is deposited **verbatim** as an ``MCP_MEMORY_EXPORT`` entry
    (``STABLE`` / ``NEVER``: a dump is a record of what was seen, not a live
    handle), and every particle points back at it by line number. Nothing is
    attributed to the incumbent store itself — the SDK never fetched it and
    cannot re-verify it, so provenance names the artifact it actually holds.
    Re-running is idempotent (content-hash dedup).

    Migrated beliefs are deliberately low-confidence: they are second-hand, and
    the incumbent's own scores are preserved as tags rather than becoming
    confidence values. Raise them with ``particles trust set`` once you vouch
    for the source, not by editing the import floor.

    \b
        particles import mcp-memory ~/.mcp/memory.jsonl
        particles extract --all-pending
        particles lint
    """
    _refuse_in_remote_mode("mcp-memory")
    configure_logging(False, debug)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    entry_id, _snapshot_id = run(_import_mcp_memory(export_path, deposited_by, tag_list))
    typer.echo(f"Deposited {export_path} as corpus entry {entry_id}.")
    typer.echo(
        "Migrated beliefs are second-hand: they carry calibration source IMPORTED and a low "
        "import-floor confidence, and the source store's own scores do not carry over. "
        "Use `particles trust set` to raise the whole export once you vouch for it."
    )
    typer.echo("Next: `particles extract --all-pending` to turn the export into particles.")


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
