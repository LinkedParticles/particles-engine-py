# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""interchange sub-Typer — export / import / restore portable store bundles.

A bundle is a directory of a ``manifest.json`` envelope plus particles / subjects
members, serialized either as canonical JSONL (``particles.jsonl``) or the
human-editable YAML-LD sibling (``particles.yaml``; ``export --format yaml``).
Import / restore auto-detect the container from the member
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

import errno
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from particles.api.cli import app, run
from particles.db import session_scope

if TYPE_CHECKING:
    from particles.interchange.store import ImportSummary, RestoreSummary


class BundleEscapeError(Exception):
    """A bundle member resolves outside its bundle directory (security review F28)."""


class BundleReadError(Exception):
    """A bundle member is over the byte cap or is not UTF-8 text (security review D26)."""


#: Suffixes of the files a bundle can carry: the manifest, the particles and
#: subjects members in either container, and a restore's ``*.corpus.jsonl``.
_MEMBER_SUFFIXES = (".json", ".jsonl", ".yaml", ".yml")

#: The members a bundle names by convention: the manifest and the particles and
#: subjects members in either container. Only the first spelling of each base in
#: name order (``.jsonl``, then ``.yaml``, then ``.yml``, the order the import
#: reads them in) takes a cap of its own. Every other suffix-matching file (a
#: second spelling of a base, a restore's ``*.corpus.jsonl``) shares one byte
#: budget and one count cap (security review 2026-10-27-1, G57).
_NAMED_MEMBERS = frozenset(
    {"manifest.json"}
    | {f"{base}{ext}" for base in ("particles", "subjects") for ext in (".jsonl", ".yaml", ".yml")}
)
_MAX_OTHER_MEMBERS = 64


def _read_member_file(path: Path, root: Path | None, max_bytes: int | None = None) -> str:
    """Read one bundle member through the size cap and, in a directory, containment.

    The cap is the readers' own per-member cap (``read_jsonl`` / ``read_yaml_ld``),
    or ``max_bytes`` when smaller, checked here on the file's size before a byte
    is buffered (security review N20, D26).
    """
    from particles.corpus.local_read import LocalReadRefused, read_local_file, within
    from particles.interchange import jsonl

    try:
        read = read_local_file(
            path,
            follow_symlinks=True,
            max_bytes=(
                jsonl._MAX_BUNDLE_BYTES
                if max_bytes is None
                else min(max_bytes, jsonl._MAX_BUNDLE_BYTES)
            ),
            allowed=within(root) if root is not None else None,
        )
    except LocalReadRefused as exc:
        if exc.errno == errno.EACCES:
            raise BundleEscapeError(
                f"bundle member {path.name!r} resolves outside the bundle "
                f"directory {str(root)!r}; refusing to read (security review F28)."
            ) from exc
        raise BundleReadError(f"bundle member {path.name!r}: {exc.strerror}") from exc
    try:
        return read.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleReadError(f"bundle member {path.name!r} is not UTF-8 text") from exc


def _read_bundle_members(bundle: Path) -> dict[str, str]:
    """Read a bundle directory's members, refusing escapes (F28) and oversized members.

    Only a regular file with a member suffix (``.json``, ``.jsonl``, ``.yaml``,
    ``.yml``) is read; anything else in the directory is left alone (security
    review N20). Each member is read through one descriptor whose resolved path
    must lie within the bundle root, so a member that is a symlink pointing
    outside the bundle directory cannot redirect the read to an arbitrary local
    file (security review F28), and a member over the per-member byte cap is
    refused before it is buffered (D26). The conventional members (the manifest,
    and the first-read spelling of the particles and subjects members) take the
    per-member cap each, so at most four caps are buffered; every other member
    shares a single per-member cap between them, and at most
    ``_MAX_OTHER_MEMBERS`` are read, so a directory of many members cannot
    buffer past the cap (security review 2026-10-26-1, W19). Raises
    :class:`BundleEscapeError` on a member that escapes and
    :class:`BundleReadError` on one that is too large or not text, or when the
    other members pass their budget; the caller translates both to a clean CLI
    error.
    """
    from particles.interchange import jsonl

    root = bundle.resolve()
    files: dict[str, str] = {}
    other_budget = jsonl._MAX_BUNDLE_BYTES
    other_count = 0
    named_bases: set[str] = set()
    for member in sorted(bundle.iterdir()):
        if member.suffix.lower() not in _MEMBER_SUFFIXES or not member.is_file():
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
        base = member.name.split(".", 1)[0]
        if member.name in _NAMED_MEMBERS and base not in named_bases:
            named_bases.add(base)
            files[member.name] = _read_member_file(member, root)
            continue
        other_count += 1
        if other_count > _MAX_OTHER_MEMBERS or other_budget <= 0:
            raise BundleReadError(
                f"bundle directory {str(bundle)!r} holds more members than one import "
                f"reads (at most {_MAX_OTHER_MEMBERS} besides the manifest, particles "
                "and subjects members, within one member's byte cap in total)"
            )
        content = _read_member_file(member, root, other_budget)
        other_budget -= len(content.encode("utf-8"))
        files[member.name] = content
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


def _echo_dropped_keys(dropped_keys: dict[str, int]) -> None:
    """Name the keys a newer schema minor carried that this SDK does not know."""
    if dropped_keys:
        listed = ", ".join(f"{key} ({count})" for key, count in sorted(dropped_keys.items()))
        typer.echo(
            "Keys this SDK does not know were dropped from the imported particles "
            f"(written by a newer schema minor): {listed}."
        )


async def _import(bundle: Path, store: str) -> None:
    from particles.interchange.store import import_store_bundle

    try:
        files = _read_bundle_members(bundle)
    except (BundleEscapeError, BundleReadError) as exc:
        typer.echo(f"Import failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    async with session_scope(store) as session:
        try:
            summary = await import_store_bundle(session, files, store=store)
        except ValueError as exc:
            # A malformed member (and an incompatible schema version) is refused
            # with its reason, never a traceback (security review 2026-10-28-1, G17).
            typer.echo(f"Import failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        await session.commit()
    typer.echo(
        f"Imported {summary.imported} particle(s), {summary.subjects_created} new subject(s), "
        f"{summary.dropped} dropped."
    )
    _echo_neutralised(summary, "imported")
    _echo_dropped_keys(summary.dropped_keys)


def _echo_neutralised(summary: ImportSummary | RestoreSummary, verb: str) -> None:
    """Say which server-owned marks a bundle carried and what became of each."""
    from particles.interchange.store import CLAIMED_TAG_PREFIX, IMPORT_ASSERTER, ORIGIN_REF_PREFIX

    if summary.restamped:
        typer.echo(
            f"{summary.restamped} record(s) named a Particles server identity as their "
            f"asserter, or the re-anchor pass as a contributor; {verb} as "
            f"{IMPORT_ASSERTER!r}, with that identity kept as an author contributor "
            f"(a contributor id is prefixed {CLAIMED_TAG_PREFIX!r})."
        )
    if summary.retagged:
        typer.echo(
            f"{summary.retagged} particle(s) carried the memory-server migration tag; "
            f"{verb} with it prefixed {CLAIMED_TAG_PREFIX!r}, so the memory-server "
            "facade does not read them as its own memory."
        )
    if summary.reanchored:
        typer.echo(
            f"{summary.reanchored} particle(s) cited a corpus entry or a particle this store "
            f"already holds; {verb} with that id prefixed {ORIGIN_REF_PREFIX!r}, so a bundle "
            "cannot choose which of this store's sources attest its claims."
        )
    if summary.recalibrated:
        typer.echo(
            f"{summary.recalibrated} particle(s) claimed an operator's or an agent's "
            f"calibration; {verb} as IMPORTED."
        )


@interchange_app.command("restore")
def restore_cmd(
    bundle: Path = typer.Argument(
        ..., help="Bundle directory or a single particles JSONL file to restore"
    ),
    store: str = typer.Option("default", "--store", help="Target store handle (must be empty)"),
    trust_bundle: bool = typer.Option(
        False,
        "--trust-bundle",
        help=(
            "Restore the server identities, the migration tag and the calibration sources "
            "as written. Only for a bundle you exported from your own store."
        ),
    ),
) -> None:
    """Faithfully reconstruct a bundle into an EMPTY store, origin ids preserved.

    Unlike ``import`` (claim-fingerprint merge, fresh ids), ``restore`` reconstructs
    the bundle's own store: ids are preserved verbatim and no §6.6 reconcile runs.
    The target must be empty; a populated target is refused.

    A bundle is read as untrusted unless ``--trust-bundle`` is given: a record
    naming a Particles server identity, the memory-server migration tag, an
    operator's or an agent's calibration, or a corpus entry the target already
    holds is restored as attribution, exactly as ``import`` treats it, so a
    bundle you are handed cannot become the memory-server facade's own memory.
    """
    if not bundle.exists():
        typer.echo(f"Bundle path {str(bundle)!r} not found.", err=True)
        raise typer.Exit(1)
    run(_restore(bundle, store, trust_bundle=trust_bundle))


async def _restore(bundle: Path, store: str, *, trust_bundle: bool = False) -> None:
    from particles.interchange.store import RestoreError, restore_store_bundle

    try:
        if bundle.is_dir():
            files = _read_bundle_members(bundle)
        else:
            files = {bundle.name: _read_member_file(bundle, None)}
    except (BundleEscapeError, BundleReadError) as exc:
        typer.echo(f"Restore failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    async with session_scope(store) as session:
        try:
            summary = await restore_store_bundle(session, files, trust_bundle=trust_bundle)
        except (RestoreError, ValueError) as exc:
            typer.echo(f"Restore failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        await session.commit()
    typer.echo(
        f"Restored {summary.particles} particle(s) and {summary.subjects} subject(s) "
        "(origin ids preserved)."
    )
    _echo_neutralised(summary, "restored")
    if summary.restamped or summary.retagged or summary.reanchored or summary.recalibrated:
        typer.echo("Pass --trust-bundle to restore them as written, for your own export only.")
    _echo_dropped_keys(summary.dropped_keys)
