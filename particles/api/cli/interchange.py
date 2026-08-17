"""interchange sub-Typer — export / import / restore portable store bundles.

A bundle is a directory of a ``manifest.json`` envelope plus particles / subjects
members, serialized either as canonical JSONL (``particles.jsonl``) or the
human-editable YAML-LD sibling (``particles.yaml``; ``export --format yaml``). Import / restore auto-detect the container from the member
extension, so either form round-trips with no flag. Import runs each particle
through the §6.6 ladder (import is a single-store write that
reconciles).

Restore is the faithful, id-preserving sibling of import: it
reconstructs the bundle's own store into an **empty** target with origin ids
preserved verbatim and no reconcile. It accepts either a bundle directory or a
single particles JSONL file (the projection-gate ``*.corpus.jsonl``), and refuses
a non-empty target so it can never become a UUID-smuggling cross-store merge.
"""

from __future__ import annotations

from pathlib import Path

import typer

from particles.api.cli import app, run
from particles.db import session_scope


class BundleEscapeError(Exception):
    """A bundle member resolves outside its bundle directory (security review F28)."""


def _read_bundle_members(bundle: Path) -> dict[str, str]:
    """Read a bundle directory's regular-file members, refusing escapes (F28).

    Each member is resolved and asserted to be contained within the bundle root
    before it is read: a member that is a symlink pointing outside the bundle
    directory could otherwise redirect the read to an arbitrary local file
    (security review F28). Mirrors the F1 containment idiom in
    ``particles/render/markdown.py`` (``is_within_directory``). Raises
    :class:`BundleEscapeError` on a member that escapes; the caller translates it
    to a clean CLI error.
    """
    root = bundle.resolve()
    files: dict[str, str] = {}
    for member in bundle.iterdir():
        if not member.is_file():
            continue
        try:
            contained = member.resolve().is_relative_to(root)
        except (OSError, ValueError):
            # Unresolvable (e.g. a symlink loop) is, by definition, not safely
            # contained — fail closed.
            contained = False
        if not contained:
            raise BundleEscapeError(
                f"bundle member {member.name!r} resolves outside the bundle "
                f"directory {str(bundle)!r}; refusing to read (security review F28)."
            )
        files[member.name] = member.read_text(encoding="utf-8")
    return files


interchange_app = typer.Typer(
    help="Export / import portable store bundles.",
    no_args_is_help=True,
)
app.add_typer(interchange_app, name="interchange")


@interchange_app.command("export")
def export_cmd(
    output: Path = typer.Option(..., "-o", "--output", help="Bundle directory to write"),
    store: str = typer.Option("default", "--store", help="Store handle to export"),
    format: str = typer.Option(
        "jsonl",
        "--format",
        help=(
            "Bundle container: jsonl (canonical, one unit per line) or yaml "
            "(human-editable YAML-LD, same data model). Both "
            "round-trip through `interchange import` unchanged."
        ),
    ),
) -> None:
    """Write a store-export bundle (manifest + particles/subjects members)."""
    if format not in ("jsonl", "yaml"):
        typer.echo(f"Unknown --format {format!r}; expected 'jsonl' or 'yaml'.", err=True)
        raise typer.Exit(2)
    run(_export(output, store, format))


async def _export(output: Path, store: str, container: str) -> None:
    from particles.interchange.store import export_store_bundle

    async with session_scope(store) as session:
        files = await export_store_bundle(session, container=container)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    typer.echo(f"Wrote bundle to {output} ({len(files)} files).")


@interchange_app.command("import")
def import_cmd(
    bundle: Path = typer.Argument(..., help="Bundle directory to import"),
    store: str = typer.Option("default", "--store", help="Target store handle"),
) -> None:
    """Import a store-export bundle into a store (single-store writes, §6.6)."""
    if not bundle.is_dir():
        typer.echo(f"Bundle directory {bundle!r} not found.", err=True)
        raise typer.Exit(1)
    run(_import(bundle, store))


async def _import(bundle: Path, store: str) -> None:
    from particles.interchange.store import import_store_bundle

    try:
        files = _read_bundle_members(bundle)
    except BundleEscapeError as exc:
        typer.echo(f"Import failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    async with session_scope(store) as session:
        summary = await import_store_bundle(session, files)
        await session.commit()
    typer.echo(
        f"Imported {summary.imported} particle(s), {summary.subjects_created} new subject(s), "
        f"{summary.dropped} dropped."
    )


@interchange_app.command("restore")
def restore_cmd(
    bundle: Path = typer.Argument(
        ..., help="Bundle directory or a single particles JSONL file to restore"
    ),
    store: str = typer.Option("default", "--store", help="Target store handle (must be empty)"),
) -> None:
    """Faithfully reconstruct a bundle into an EMPTY store, origin ids preserved.

    Unlike ``import`` (claim-fingerprint merge, fresh ids), ``restore`` reconstructs
    the bundle's own store: ids are preserved verbatim and no §6.6 reconcile runs.
    The target must be empty; a populated target is refused.
    """
    if not bundle.exists():
        typer.echo(f"Bundle path {str(bundle)!r} not found.", err=True)
        raise typer.Exit(1)
    run(_restore(bundle, store))


async def _restore(bundle: Path, store: str) -> None:
    from particles.interchange.store import RestoreError, restore_store_bundle

    try:
        if bundle.is_dir():
            files = _read_bundle_members(bundle)
        else:
            files = {bundle.name: bundle.read_text(encoding="utf-8")}
    except BundleEscapeError as exc:
        typer.echo(f"Restore failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    async with session_scope(store) as session:
        try:
            summary = await restore_store_bundle(session, files)
        except RestoreError as exc:
            typer.echo(f"Restore failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        await session.commit()
    typer.echo(
        f"Restored {summary.particles} particle(s) and {summary.subjects} subject(s) "
        "(origin ids preserved)."
    )
