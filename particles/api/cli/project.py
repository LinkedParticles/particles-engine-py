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
            "Render the deterministic structured listing — no LLM call, no "
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
            "bindings, renders only that region's section — the "
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
    verbose: bool = typer.Option(False, "--verbose", help="Per-section progress logging."),
) -> None:
    """Render (or drift-check) a documentation projection.

    \b
        particles project docs/projection/readme.yaml README.md
        particles project docs/projection/readme.yaml --without-synthesis
        particles project docs/projection/readme.yaml --check   # CI drift gate
        # splice every declared region of the README in one pass:
        particles project docs/projection/readme.yaml --splice-all
        # re-roll a single sentinel region:
        particles project docs/projection/readme.yaml README.md --splice what-is
        # refresh the committed drift-gate bundle:
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
) -> None:
    from particles.operations.projection import (
        SpliceError,
        check_drift,
        load_manifest,
        project_document,
        project_region_bodies,
        project_splice_body,
        snapshot_path_for,
        splice_region,
    )

    mpath = Path(manifest_path).expanduser()
    if not mpath.exists():
        typer.echo(f"Manifest not found: {mpath}", err=True)
        raise typer.Exit(2)
    try:
        manifest = load_manifest(mpath)
    except (ValueError, ValidationError) as exc:
        typer.echo(f"Invalid manifest {mpath}:\n{exc}", err=True)
        raise typer.Exit(2) from exc
    base_dir = mpath.parent

    async with session_scope() as session:
        if check:
            result = await check_drift(session, manifest, base_dir=base_dir, output_root=Path.cwd())
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

        effective_output = output or manifest.output
        if effective_output is None:
            typer.echo(
                "No output path supplied. Pass it as an argument, or set "
                "`output:` in the manifest.",
                err=True,
            )
            raise typer.Exit(2)
        out_path = Path(effective_output).expanduser()

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
            text = out_path.read_text(encoding="utf-8")
            try:
                for region, body in bodies.items():
                    text = splice_region(text, region, body, manifest=str(mpath))
            except SpliceError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(2) from exc
            out_path.write_text(text, encoding="utf-8")
            await _refresh_snapshot(session, manifest, base_dir=base_dir)
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
                session, manifest, base_dir=base_dir, synthesize=not without_synthesis
            )
            existing = out_path.read_text(encoding="utf-8")
            try:
                spliced = splice_region(existing, splice, body_doc.document, manifest=str(mpath))
            except SpliceError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(2) from exc
            out_path.write_text(spliced, encoding="utf-8")
            result_doc = body_doc
        else:
            result_doc = await project_document(
                session, manifest, base_dir=base_dir, synthesize=not without_synthesis
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result_doc.document, encoding="utf-8")

        # Always refresh the deterministic snapshot — it is the gated contract the
        # `--check` drift gate diffs, so it must stay in lock-step with the doc.
        # The snapshot is the full deterministic document (selection + structure
        # fingerprint), independent of whether we wrote the whole file or spliced.
        snap_path = snapshot_path_for(manifest, base_dir=base_dir)
        if without_synthesis and splice is None:
            snapshot_text = result_doc.document
        else:
            snapshot_text = (
                await project_document(session, manifest, base_dir=base_dir, synthesize=False)
            ).document
        snap_path.write_text(snapshot_text, encoding="utf-8")

        verb = (
            f"Spliced {manifest.name} → {out_path} (region {splice!r})"
            if splice
            else (f"Projected {manifest.name} → {out_path}")
        )
        typer.echo(verb)
        typer.echo(f"  snapshot: {snap_path}")
        typer.echo(
            "  synthesis: " + ("on" if result_doc.used_synthesis else "deterministic (no LLM)")
        )


async def _refresh_snapshot(
    session: AsyncSession, manifest: DocManifest, *, base_dir: Path
) -> None:
    """Regenerate the deterministic snapshot beside the manifest (the gated contract)."""
    from particles.operations.projection import project_document, snapshot_path_for

    snapshot = (
        await project_document(session, manifest, base_dir=base_dir, synthesize=False)
    ).document
    snapshot_path_for(manifest, base_dir=base_dir).write_text(snapshot, encoding="utf-8")


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
    from particles.store.particle_store import get_active_particle_by_id_or_prefix

    ids = sorted(await required_particle_ids(session, manifest))
    particles = []
    for pid in ids:
        particle = await get_active_particle_by_id_or_prefix(session, pid)
        if particle is not None:
            particles.append(particle)
    units = await export_particles(session, particles)
    path = base_dir / f"{manifest.name}.corpus.jsonl"
    path.write_text(write_jsonl(units), encoding="utf-8")
    return path


# Module-level logger kept for parity with sibling verbs; projection progress
# routes through ``particles.operations.projection`` loggers, surfaced by
# ``--verbose``.
log = logging.getLogger(__name__)
