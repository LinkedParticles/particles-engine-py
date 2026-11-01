# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""project verb — render a documentation-projection manifest.

Renders a checked-in manifest (``docs/projection/<name>.yaml``) into a cited
Markdown document: each derived section's current-truth particles synthesised as
clean prose, mechanical blocks spliced verbatim, in manifest order. ``--check``
is the drift gate — it regenerates the deterministic snapshot and fails on
selection / structure drift, without an API key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml
from pydantic import ValidationError

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.db import session_scope

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.operations.projection import DocManifest


@app.command("project")
def project_cmd(
    manifest_path: str = typer.Argument(
        ...,
        metavar="MANIFEST",
        help="Path to a projection manifest (e.g. docs/projection/readme.yaml).",
    ),
    output: str | None = typer.Argument(
        None,
        help="Output Markdown path. Optional when the manifest sets `output:`.",
    ),
    without_synthesis: bool = typer.Option(
        False,
        "--without-synthesis",
        help=(
            "Render the deterministic structured listing: no LLM call, no "
            "ANTHROPIC_API_KEY, reproducible output. The drift gate uses this mode."
        ),
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help=(
            "Drift gate: regenerate the deterministic snapshot and "
            "exit non-zero if it differs from the committed "
            "`<name>.snapshot.md`. Selection + structure are gated; LLM prose "
            "drift is advisory. No API key required."
        ),
    ),
    splice: str | None = typer.Option(
        None,
        "--splice",
        metavar="REGION",
        help=(
            "Block-splice mode: write the rendered body *between* "
            "the `<!-- BEGIN/END PROJECTED: REGION -->` sentinels in the existing "
            "output file, preserving everything outside them, instead of "
            "overwriting the whole file. The output file must already carry the "
            "sentinel pair for REGION. On a manifest with per-section `region:` "
            "bindings, renders only that region's section, the "
            "single-region re-roll path."
        ),
    ),
    splice_all: bool = typer.Option(
        False,
        "--splice-all",
        help=(
            "Multi-region block-splice: render every section that "
            "declares a `region:` and splice each body into its own sentinel "
            "pair in the output file, in one pass. Every derived section must "
            "declare a region."
        ),
    ),
    export_corpus: bool = typer.Option(
        False,
        "--export-corpus",
        help=(
            "Write the manifest's sibling `<name>.corpus.jsonl` gate bundle "
            ": exactly the particles the manifest's deterministic "
            "selection requires, encoded as interchange units the drift gate's "
            "ephemeral restore consumes. No render is performed."
        ),
    ),
    allow_outside_paths: bool = typer.Option(
        False,
        "--allow-outside-paths",
        help=(
            "Let the manifest's own `output:` and block paths be absolute, use "
            "`..`, or resolve outside the repository that holds the manifest. "
            "Without it they are refused, because a manifest a repository ships "
            "is data. It also lets the manifest file itself resolve outside that "
            "repository. Pass it only for a manifest you wrote."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Per-section progress logging."),
) -> None:
    """Render (or drift-check) a documentation projection.

    \b
        particles project docs/projection/readme.yaml README.md
        particles project docs/projection/readme.yaml --without-synthesis
        particles project docs/projection/readme.yaml --check   # CI drift gate
        # Splice every declared region of the README in one pass:
        particles project docs/projection/readme.yaml --splice-all
        # Re-roll a single sentinel region:
        particles project docs/projection/readme.yaml README.md --splice what-is
        # Refresh the committed drift-gate bundle:
        particles project docs/projection/readme.yaml --export-corpus
    """
    configure_logging(verbose, debug=False)

    # (b): projection reads the whole store and writes local files,
    # with no remote bulk-read route — refuse in remote mode rather than project
    # the laptop's near-empty store. (Remote projection is deferred
    # Deferred; the `import project` rough edge tracks the same friction.)
    from particles.api.cli._remote import refuse_remote_sync

    refuse_remote_sync("project")

    if splice_all and splice is not None:
        typer.echo("--splice-all and --splice REGION are mutually exclusive.", err=True)
        raise typer.Exit(2)

    run(
        _project(
            manifest_path,
            output,
            without_synthesis=without_synthesis,
            check=check,
            splice=splice,
            splice_all=splice_all,
            export_corpus=export_corpus,
            allow_outside_paths=allow_outside_paths,
        )
    )


async def _project(
    manifest_path: str,
    output: str | None,
    *,
    without_synthesis: bool,
    check: bool,
    splice: str | None,
    splice_all: bool,
    export_corpus: bool,
    allow_outside_paths: bool = False,
) -> None:
    from particles.operations.projection import load_manifest
    from particles.operations.projection.paths import ManifestPathError, write_root

    mpath = Path(manifest_path).expanduser()
    if not mpath.exists():
        typer.echo(f"Manifest not found: {mpath}", err=True)
        raise typer.Exit(2)
    # The manifest is read inside its own tree unless --allow-outside-paths
    # says otherwise: a manifest committed as a link to a file elsewhere is
    # refused rather than read (security review 2026-10-26-1, W29).
    root = write_root(mpath.parent)
    try:
        manifest = load_manifest(
            mpath,
            allowed=None if allow_outside_paths else (lambda p: p.is_relative_to(root)),
        )
    except ValidationError as exc:
        # Locations and error types only: the input values are the file's own
        # contents, which an error message must not echo.
        problems = "\n".join(
            f"  {'.'.join(str(part) for part in err['loc']) or '(top level)'}: {err['type']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
        typer.echo(f"Invalid manifest {mpath}:\n{problems}", err=True)
        raise typer.Exit(2) from exc
    except yaml.YAMLError as exc:
        typer.echo(f"Invalid manifest {mpath}: not valid YAML.", err=True)
        raise typer.Exit(2) from exc
    except (ValueError, OSError) as exc:
        typer.echo(f"Invalid manifest {mpath}: {exc}", err=True)
        raise typer.Exit(2) from exc

    async with session_scope() as session:
        try:
            await _project_in_session(
                session,
                manifest,
                mpath,
                output,
                without_synthesis=without_synthesis,
                check=check,
                splice=splice,
                splice_all=splice_all,
                export_corpus=export_corpus,
                allow_outside_paths=allow_outside_paths,
            )
        except (ManifestPathError, OSError) as exc:
            # Security review 2026-10-24-2, P3: a manifest path that leaves the
            # manifest's tree, or a symlink where a read or write lands, is
            # refused before anything is written.
            typer.echo(f"Refused: {exc}", err=True)
            raise typer.Exit(2) from exc


def _output_path(
    output: str | None, manifest: DocManifest, base_dir: Path, *, allow_outside_paths: bool
) -> Path:
    """Where the document goes: the operator's OUTPUT, else the manifest's ``output:``.

    An OUTPUT given on the command line is the operator's choice and is taken as
    given. The manifest's own ``output:`` is data a repository can ship, so it
    is resolved against the working directory (as it always was) and must stay
    inside the repository that holds the manifest (security review
    2026-10-24-2, P3).
    """
    from particles.operations.projection.paths import containment_root, resolve_manifest_path

    if output is not None:
        return Path(output).expanduser()
    if manifest.output is None:
        typer.echo(
            "No output path supplied. Pass it as an argument, or set `output:` in the manifest.",
            err=True,
        )
        raise typer.Exit(2)
    return resolve_manifest_path(
        manifest.output,
        against=Path.cwd(),
        root=containment_root(base_dir),
        allow_outside=allow_outside_paths,
        what="output",
    )


async def _project_in_session(
    session: AsyncSession,
    manifest: DocManifest,
    mpath: Path,
    output: str | None,
    *,
    without_synthesis: bool,
    check: bool,
    splice: str | None,
    splice_all: bool,
    export_corpus: bool,
    allow_outside_paths: bool,
) -> None:
    from particles.operations.projection import (
        SpliceError,
        check_drift,
        project_document,
        project_region_bodies,
        project_splice_body,
        snapshot_path_for,
        splice_region,
    )
    from particles.operations.projection.paths import (
        containment_root,
        read_text_no_follow,
        write_root,
        write_text_no_follow,
    )

    base_dir = mpath.parent
    allow = allow_outside_paths
    # The manifest's own `output:` is written contained in its tree, so a
    # parent swapped for a symlink during the render is refused (security
    # review 2026-10-25-1, Q29). An operator's OUTPUT is taken as given.
    out_root = containment_root(base_dir) if output is None and not allow else None
    beside_root = write_root(base_dir)

    if check:
        result = await check_drift(
            session,
            manifest,
            base_dir=base_dir,
            output_root=Path.cwd(),
            allow_outside_paths=allow,
        )
        typer.echo(result.reason)
        if result.drifted:
            raise typer.Exit(1)
        return

    if export_corpus:
        # refresh the drift gate's committed sibling bundle —
        # exactly the deterministic selection, no render.
        bundle_path = await _write_corpus_bundle(session, manifest, base_dir=base_dir)
        typer.echo(f"Exported corpus bundle: {bundle_path}")
        return

    out_path = _output_path(output, manifest, base_dir, allow_outside_paths=allow)

    if splice_all or (splice is not None and manifest.region_sections()):
        # manifest-declared regions — splice each rendered
        # body into its own sentinel pair (--splice-all), or just the one
        # named region (--splice REGION, the re-roll path).
        if not out_path.exists():
            typer.echo(
                f"splice target does not exist: {out_path}. The output file must "
                "already carry the sentinel pair(s) to splice into.",
                err=True,
            )
            raise typer.Exit(2)
        try:
            bodies, used_synth = await project_region_bodies(
                session,
                manifest,
                base_dir=base_dir,
                synthesize=not without_synthesis,
                only_region=splice,
            )
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        text = read_text_no_follow(out_path)
        try:
            for region, body in bodies.items():
                text = splice_region(text, region, body, manifest=str(mpath))
        except SpliceError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        write_text_no_follow(out_path, text, root=out_root)
        await _refresh_snapshot(session, manifest, base_dir=base_dir, allow_outside_paths=allow)
        typer.echo(f"Spliced {manifest.name} → {out_path} (region(s): {', '.join(bodies)})")
        typer.echo(f"  snapshot: {snapshot_path_for(manifest, base_dir=base_dir)}")
        typer.echo("  synthesis: " + ("on" if used_synth else "deterministic (no LLM)"))
        return

    if splice is not None:
        # block-splice: render just the body and write it between the
        # named sentinels in the existing file, preserving everything else.
        if not out_path.exists():
            typer.echo(
                f"--splice target does not exist: {out_path}. The output file must "
                "already carry the sentinel pair to splice into.",
                err=True,
            )
            raise typer.Exit(2)
        body_doc = await project_splice_body(
            session,
            manifest,
            base_dir=base_dir,
            synthesize=not without_synthesis,
            allow_outside_paths=allow,
        )
        existing = read_text_no_follow(out_path)
        try:
            spliced = splice_region(existing, splice, body_doc.document, manifest=str(mpath))
        except SpliceError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        write_text_no_follow(out_path, spliced, root=out_root)
        result_doc = body_doc
    else:
        result_doc = await project_document(
            session,
            manifest,
            base_dir=base_dir,
            synthesize=not without_synthesis,
            allow_outside_paths=allow,
        )
        if out_root is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_no_follow(out_path, result_doc.document, root=out_root)

    # Always refresh the deterministic snapshot — it is the gated contract the
    # `--check` drift gate diffs, so it must stay in lock-step with the doc.
    # The snapshot is the full deterministic document (selection + structure
    # fingerprint), independent of whether we wrote the whole file or spliced.
    snap_path = snapshot_path_for(manifest, base_dir=base_dir)
    if without_synthesis and splice is None:
        snapshot_text = result_doc.document
    else:
        snapshot_text = (
            await project_document(
                session,
                manifest,
                base_dir=base_dir,
                synthesize=False,
                allow_outside_paths=allow,
            )
        ).document
    write_text_no_follow(snap_path, snapshot_text, root=beside_root)

    verb = (
        f"Spliced {manifest.name} → {out_path} (region {splice!r})"
        if splice
        else (f"Projected {manifest.name} → {out_path}")
    )
    typer.echo(verb)
    typer.echo(f"  snapshot: {snap_path}")
    typer.echo("  synthesis: " + ("on" if result_doc.used_synthesis else "deterministic (no LLM)"))


async def _refresh_snapshot(
    session: AsyncSession, manifest: DocManifest, *, base_dir: Path, allow_outside_paths: bool
) -> None:
    """Regenerate the deterministic snapshot beside the manifest (the gated contract)."""
    from particles.operations.projection import project_document, snapshot_path_for
    from particles.operations.projection.paths import write_root, write_text_no_follow

    snapshot = (
        await project_document(
            session,
            manifest,
            base_dir=base_dir,
            synthesize=False,
            allow_outside_paths=allow_outside_paths,
        )
    ).document
    write_text_no_follow(
        snapshot_path_for(manifest, base_dir=base_dir), snapshot, root=write_root(base_dir)
    )


async def _write_corpus_bundle(
    session: AsyncSession, manifest: DocManifest, *, base_dir: Path
) -> Path:
    """Write the manifest's sibling ``<name>.corpus.jsonl`` gate bundle.

    Exactly the manifest's deterministic selection (``required_particle_ids``),
    id-sorted for byte-stable output, encoded as the interchange Particle units
    the drift gate's ephemeral restore consumes.
    """
    from particles.interchange.jsonl import write_jsonl
    from particles.interchange.store import export_particles
    from particles.operations.projection import required_particle_ids
    from particles.operations.projection.paths import write_root, write_text_no_follow
    from particles.store.particle_store import get_active_particle_by_id_or_prefix

    ids = sorted(await required_particle_ids(session, manifest))
    particles = []
    for pid in ids:
        particle = await get_active_particle_by_id_or_prefix(session, pid)
        if particle is not None:
            particles.append(particle)
    units = await export_particles(session, particles)
    path = base_dir / f"{manifest.name}.corpus.jsonl"
    write_text_no_follow(path, write_jsonl(units), root=write_root(base_dir))
    return path


# Module-level logger kept for parity with sibling verbs; projection progress
# routes through ``particles.operations.projection`` loggers, surfaced by
# ``--verbose``.
log = logging.getLogger(__name__)
