# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""subjects command — manage canonical real-world entities.

This is a single Typer command with manual action dispatch via a positional
argument (list, search, show, alias, merge, confirm, unlink, split, delete,
gc, set-class, fix-labels, find-duplicates, relink-gated, rebuild-aliases,
fold-artifacts) — Typer sub-Typer
style would be more idiomatic but the current shape is preserved for
backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from particles.api.cli import app, run
from particles.api.cli._progress import confirm
from particles.core.schema import Subject
from particles.db import session_scope

if TYPE_CHECKING:
    from particles.api.client.base import Backend
    from particles.operations.artifact_fold import FoldReport
    from particles.operations.subject_relink import RelinkReport
    from particles.store.subject_store import AliasIndexDrift

# (b): subjects actions with no engine endpoint. They refuse in
# remote mode (per-action, inside ``subjects_cmd``) rather than silently
# hitting the local store. The endpoint-backed actions (list / search / show /
# alias / merge / split) route through the backend instead.
_SUBJECTS_LOCAL_ONLY = frozenset(
    {
        "set-class",
        "delete",
        "gc",
        "prune-empty",
        "confirm",
        "unlink",
        "fix-labels",
        "find-duplicates",
        "relink-gated",
        "rebuild-aliases",
        "fold-artifacts",
    }
)


@app.command("subjects")
def subjects_cmd(
    action: str = typer.Argument(
        "list",
        help=(
            "Action: list, search, show, alias, confirm, unlink, merge, split, "
            "delete, gc, set-class, fix-labels, find-duplicates, relink-gated, "
            "rebuild-aliases, fold-artifacts"
        ),
    ),
    rest: list[str] | None = typer.Argument(None, help="Arguments for the chosen action"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview merge/split/gc without committing"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="merge only: skip the confirmation prompt"),
    force: bool = typer.Option(
        False,
        "--force",
        help="delete only: remove a non-phantom subject (one with ACTIVE particles)",
    ),
    phantoms_only: bool = typer.Option(
        False,
        "--phantoms-only",
        help="list only: restrict to phantom subjects (zero ACTIVE particles)",
    ),
    order: str = typer.Option(
        "name",
        "--order",
        help=("list only: 'name' (alphabetical) or 'degree' (most ACTIVE linked particles first)"),
    ),
    particle: list[str] | None = typer.Option(
        None,
        "--particle",
        "-p",
        help="Particle ID to split off from the source (split only; repeat for multiple)",
    ),
    new_name: str | None = typer.Option(
        None,
        "--new-name",
        help="Approximate name for the new Subject (split only). Canonicalised via the resolver.",
    ),
    new_external_id: str | None = typer.Option(
        None,
        "--new-external-id",
        help=(
            "Authoritative external identifier for the new Subject (split only), "
            "e.g. wikidata:Q30297735. Skips resolver search; pulls metadata directly."
        ),
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help=(
            "relink-gated and fold-artifacts only: write the links or merges."
            " Without it the run only reports."
        ),
    ),
    tier: list[int] | None = typer.Option(
        None,
        "--tier",
        help=(
            "relink-gated only: a recovery tier to use (1 record, 2 structured claim, "
            "3 backticks). Repeat for several. Default: subject_gate.relink_tiers."
        ),
    ),
    sample: int = typer.Option(
        0,
        "--sample",
        min=0,
        help="relink-gated only: print this many planned relinks, for a precision check.",
    ),
    seed: int = typer.Option(
        0, "--seed", help="relink-gated only: the seed that picks the --sample relinks."
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help=(
            "rebuild-aliases only: report how the alias index differs from a rebuild,"
            " change nothing, and exit 1 when it does."
        ),
    ),
    namespace: str | None = typer.Option(
        None,
        "--namespace",
        help=(
            "fold-artifacts only: the project key to fold. Defaults to the key of the"
            " --repo checkout."
        ),
    ),
    repo: Path | None = typer.Option(
        None,
        "--repo",
        help=(
            "fold-artifacts only: a git checkout of the project. Its file list decides"
            " which bare names, partial paths and module names fold. Without it only"
            " absolute paths fold."
        ),
    ),
    undo: str | None = typer.Option(
        None,
        "--undo",
        help="fold-artifacts only: reverse the fold that recorded this event id.",
    ),
) -> None:
    """Manage subjects (canonical real-world entities).

    \b
        particles subjects list [--order name|degree] [--phantoms-only]
        particles subjects search QUERY
        particles subjects show ID|NAME
        particles subjects alias ID NAME [NAME ...]
        particles subjects confirm SUBJECT_ID NAMESPACE:ID
        particles subjects unlink SUBJECT_ID NAMESPACE:ID
        particles subjects merge SOURCE_ID [SOURCE_ID ...] TARGET_ID [--dry-run] [--yes]
        particles subjects delete SUBJECT_ID [--force]
        particles subjects gc [--dry-run]          (alias: prune-empty)
        particles subjects set-class SUBJECT_ID CLASS   (e.g. nmo:NumismaticObject)
        particles subjects split SOURCE_ID --particle PID [--particle PID ...] \\
            (--new-name "Applied Optoelectronics" | --new-external-id wikidata:Q30297735) \\
            [--dry-run]
        particles subjects relink-gated [--tier N ...] [--sample N --seed S] [--apply]
        particles subjects rebuild-aliases [--check]
        particles subjects fold-artifacts [--repo DIR] [--namespace KEY] [--apply]
        particles subjects fold-artifacts --undo EVENT_ID
    """
    args = rest or []

    from particles.api.client import get_backend

    backend = get_backend()
    # route or refuse: no-endpoint actions fail loud in remote
    # mode; the endpoint-backed ones below route through the backend.
    if action in _SUBJECTS_LOCAL_ONLY:
        from particles.api.cli._remote import refuse_remote_sync

        refuse_remote_sync(f"subjects {action}")

    if action == "list":
        if phantoms_only:
            # Phantom detection has no engine endpoint — local-only.
            from particles.api.cli._remote import refuse_remote_sync

            refuse_remote_sync("subjects list --phantoms-only")
            subjects = run(_list_phantom_subjects())
        else:
            if order not in ("name", "degree"):
                typer.echo("--order must be 'name' or 'degree'.", err=True)
                raise typer.Exit(code=1)
            subjects = run(backend.subjects_list(order="degree" if order == "degree" else "name"))
        if not subjects:
            typer.echo("No phantom subjects found." if phantoms_only else "No subjects found.")
            return
        # Phantom markers need per-subject particle counts (no endpoint), so they
        # render local-only and degrade in remote mode.
        counts = {} if backend.remote else run(_subject_particle_counts([s.id for s in subjects]))
        for s in subjects:
            ext = ", ".join(f"{r.namespace}:{r.id}" for r in s.external_ids)
            phantom = "  (phantom)" if (not backend.remote and counts.get(s.id, 0) == 0) else ""
            typer.echo(
                f"  {s.id[:8]}…  {s.canonical_name}" + (f"  [{ext}]" if ext else "") + phantom
            )

    elif action == "search":
        if not args:
            typer.echo("Provide a search query.", err=True)
            raise typer.Exit(1)
        subjects = run(backend.subjects_search(args[0]))
        needle = args[0].lower()
        for s in subjects:
            ext = ", ".join(f"{r.namespace}:{r.id}" for r in s.external_ids)
            # Name the alias that matched when the canonical name did not.
            via = (
                ""
                if needle in s.canonical_name.lower()
                else next((f"  (alias: {a})" for a in s.aliases if needle in a.lower()), "")
            )
            typer.echo(f"  {s.id[:8]}…  {s.canonical_name}{via}" + (f"  [{ext}]" if ext else ""))
        if not subjects:
            typer.echo(f"No subjects matching '{args[0]}'.")

    elif action == "show":
        if not args:
            typer.echo("Provide a subject ID or name.", err=True)
            raise typer.Exit(1)
        try:
            subject = run(backend.subject_show(args[0], by_name=True))
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None
        if subject is None:
            typer.echo(f"No subject matches {args[0]!r} by ID, ID prefix, or name.", err=True)
            raise typer.Exit(1)
        typer.echo(f"ID:          {subject.id}")
        typer.echo(f"Name:        {subject.canonical_name}")
        if subject.description:
            typer.echo(f"Description: {subject.description}")
        if subject.aliases:
            typer.echo(f"Aliases:     {', '.join(subject.aliases)}")
        for ref in subject.external_ids:
            typer.echo(
                f"External:    {ref.namespace}:{ref.id}" + (f"  {ref.uri}" if ref.uri else "")
            )

    elif action == "alias":
        if len(args) < 2:
            typer.echo("Usage: subjects alias SUBJECT_ID NAME [NAME ...]", err=True)
            raise typer.Exit(1)
        subject_id, new_aliases = args[0], args[1:]
        try:
            alias_outcome = run(backend.subject_alias(subject_id, new_aliases))
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        subject, added = alias_outcome.subject, alias_outcome.added
        ext = ", ".join(f"{r.namespace}:{r.id}" for r in subject.external_ids)
        typer.echo(f"  {subject.id[:8]}…  {subject.canonical_name}" + (f"  [{ext}]" if ext else ""))
        if added:
            for a in added:
                typer.echo(f"  + alias: {a}")
        else:
            typer.echo("  (no new aliases — all names already present)")

    elif action == "fix-labels":
        fixed, skipped, repaired, cleared = run(_fix_subject_labels())
        typer.echo(f"Fixed: {fixed}  Skipped (no label found): {skipped}")
        typer.echo(
            f"Label cache: {repaired} repaired, {cleared} cleared"
            " (Wikidata unreachable; re-fetched on next use)"
        )
        if repaired or cleared:
            typer.echo(
                "Wikidata snapshots extracted while those labels were cached may have"
                " dropped statements; re-extract them with"
                " `particles reindex --extractor-id wikidata-extractor`."
            )

    elif action == "relink-gated":
        if tier is not None and any(t not in (1, 2, 3) for t in tier):
            typer.echo("--tier must be 1, 2 or 3.", err=True)
            raise typer.Exit(1)
        report = run(_relink_gated(apply=apply, tiers=tier, sample=sample, seed=seed))
        _print_relink_report(report)

    elif action == "rebuild-aliases":
        _rebuild_aliases(check=check)

    elif action == "fold-artifacts":
        _fold_artifacts(namespace=namespace, repo=repo, apply=apply, undo=undo)

    elif action == "find-duplicates":
        pairs = run(_find_duplicate_subjects())
        if pairs is None:
            typer.echo("Embedding model unavailable; cannot compute name similarity.", err=True)
            raise typer.Exit(1)
        if not pairs:
            typer.echo("No candidate-duplicate subjects found.")
            return
        typer.echo(f"{len(pairs)} candidate-duplicate pair(s) (review, then `subjects merge`):")
        for s_a, s_b, sim in pairs:
            typer.echo(
                f"  {sim:.3f}  {s_a.id[:8]}… {s_a.canonical_name}"
                f"  ~  {s_b.id[:8]}… {s_b.canonical_name}"
            )

    elif action == "merge":
        if len(args) < 2:
            typer.echo(
                "Usage: subjects merge SOURCE_ID [SOURCE_ID ...] TARGET_ID [--dry-run] [--yes]",
                err=True,
            )
            raise typer.Exit(1)
        _merge_subjects(backend, args[:-1], args[-1], dry_run=dry_run, yes=yes)

    elif action == "delete":
        if not args:
            typer.echo("Usage: subjects delete SUBJECT_ID [--force]", err=True)
            raise typer.Exit(1)
        subject = run(backend.subject_show(args[0]))
        if subject is None:
            typer.echo(f"Subject {args[0]!r} not found.", err=True)
            raise typer.Exit(1)
        active = run(_subject_particle_counts([subject.id]))[subject.id]
        if active > 0 and not force:
            typer.echo(
                f"Refusing to delete {subject.id[:8]}… ({subject.canonical_name}): "
                f"it has {active} ACTIVE particle(s). It is not a phantom. Re-link or "
                f"retract them first, or pass --force to delete anyway.",
                err=True,
            )
            raise typer.Exit(1)
        detached = run(_delete_subject(subject.id))
        typer.echo(f"Deleted {subject.id[:8]}… ({subject.canonical_name}).")
        if detached:
            typer.echo(f"  Detached from {detached} particle(s).")

    elif action in ("gc", "prune-empty"):
        # : sweep every phantom subject (zero ACTIVE CLAIM
        # particles, the same definition as `list --phantoms-only` and the
        # `delete` guard). Useful after split/merge churn leaves stragglers.
        pruned, detached = run(_gc_phantom_subjects(dry_run=dry_run))
        if not pruned:
            typer.echo("No phantom subjects to prune.")
            return
        typer.echo(f"{'Would prune' if dry_run else 'Pruned'} {len(pruned)} phantom subject(s):")
        for s in pruned:
            typer.echo(f"  {s.id[:8]}…  {s.canonical_name}")
        if dry_run:
            typer.echo("  (no changes made)")
        elif detached:
            typer.echo(f"  Detached from {detached} non-active particle(s).")

    elif action == "set-class":
        # : operator override of the resolver's Nomisma class.
        if len(args) < 2:
            typer.echo(
                "Usage: subjects set-class SUBJECT_ID CLASS (e.g. nmo:NumismaticObject)",
                err=True,
            )
            raise typer.Exit(1)
        subject_id, new_class = args[0], args[1]
        if not new_class.strip():
            typer.echo("CLASS must be a non-empty value (e.g. nmo:Material).", err=True)
            raise typer.Exit(1)
        subject = run(backend.subject_show(subject_id))
        if subject is None:
            typer.echo(f"Subject {subject_id!r} not found.", err=True)
            raise typer.Exit(1)
        updated, previous = run(_set_subject_class_command(subject.id, new_class))
        if previous == new_class:
            typer.echo(
                f"  {updated.id[:8]}… ({updated.canonical_name}) already classed "
                f"{new_class} — no change."
            )
        else:
            prev_label = previous if previous is not None else "(unset)"
            typer.echo(
                f"Reclassified {updated.id[:8]}… ({updated.canonical_name}): "
                f"{prev_label} → {new_class}"
            )
            typer.echo("  Re-export to apply the matching exporter template.")

    elif action == "confirm":
        if len(args) < 2:
            typer.echo("Usage: subjects confirm SUBJECT_ID NAMESPACE:ID", err=True)
            raise typer.Exit(1)
        subject_id, ref_label = args[0], args[1]
        if ":" not in ref_label:
            typer.echo("NAMESPACE:ID must contain a colon (e.g. wikidata:Q47462183).", err=True)
            raise typer.Exit(1)
        namespace, ref_id = ref_label.split(":", 1)
        subject = run(backend.subject_show(subject_id))
        if subject is None:
            typer.echo(f"Subject {subject_id!r} not found.", err=True)
            raise typer.Exit(1)
        ok = run(_confirm_subject_link(subject.id, namespace, ref_id))
        if not ok:
            typer.echo(f"No {namespace}:{ref_id} link found on {subject.canonical_name}.", err=True)
            raise typer.Exit(1)
        typer.echo(
            f"Confirmed {namespace}:{ref_id} on {subject.id[:8]}… ({subject.canonical_name})."
        )
        typer.echo("  Confidence set to 1.0 — re-export to update Obsidian notes.")

    elif action == "unlink":
        if len(args) < 2:
            typer.echo("Usage: subjects unlink SUBJECT_ID NAMESPACE:ID", err=True)
            raise typer.Exit(1)
        subject_id, ref_label = args[0], args[1]
        if ":" not in ref_label:
            typer.echo("NAMESPACE:ID must contain a colon (e.g. wikidata:Q37230).", err=True)
            raise typer.Exit(1)
        namespace, ref_id = ref_label.split(":", 1)
        subject = run(backend.subject_show(subject_id))
        if subject is None:
            typer.echo(f"Subject {subject_id!r} not found.", err=True)
            raise typer.Exit(1)
        ok = run(_unlink_subject_ref(subject.id, namespace, ref_id))
        if not ok:
            typer.echo(
                f"No {namespace}:{ref_id} link found on {subject.canonical_name}.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(
            f"Removed {namespace}:{ref_id} from {subject.id[:8]}… ({subject.canonical_name})."
        )
        typer.echo(
            "  Subject and its particles are unchanged; re-export to drop the "
            "stale link from Obsidian notes."
        )

    elif action == "split":
        # re-link some particles off a source Subject onto a new
        # one created via the resolver (either by-name lookup or by-ID).
        if not args:
            typer.echo(
                "Usage: subjects split SOURCE_ID --particle PID [--particle PID ...] "
                "(--new-name NAME | --new-external-id NS:ID) [--dry-run]",
                err=True,
            )
            raise typer.Exit(1)
        source_id_arg = args[0]
        particle_ids = list(particle or [])
        if not particle_ids:
            typer.echo("subjects split requires at least one --particle PID.", err=True)
            raise typer.Exit(1)
        if not new_name and not new_external_id:
            typer.echo(
                "subjects split requires --new-name or --new-external-id.",
                err=True,
            )
            raise typer.Exit(1)
        source = run(backend.subject_show(source_id_arg))
        if source is None:
            typer.echo(f"Source subject {source_id_arg!r} not found.", err=True)
            raise typer.Exit(1)
        try:
            split_outcome = run(
                backend.subject_split(
                    source_id=source.id,
                    particle_ids=particle_ids,
                    new_name=new_name,
                    new_external_id=new_external_id,
                    dry_run=dry_run,
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        new_subject = split_outcome.new_subject
        split_relinked = split_outcome.relinked_particle_ids
        not_bound = split_outcome.not_bound_particle_ids
        if dry_run:
            typer.echo(
                f"Dry run: would split {len(split_relinked)} particle(s) off "
                f"{source.id[:8]}… ({source.canonical_name})"
            )
            typer.echo(
                f"  New subject: {new_subject.canonical_name}"
                + (
                    f"  [{', '.join(f'{r.namespace}:{r.id}' for r in new_subject.external_ids)}]"
                    if new_subject.external_ids
                    else ""
                )
            )
            if new_subject.subject_class:
                typer.echo(f"  Class: {new_subject.subject_class}")
            if not_bound:
                typer.echo(
                    f"  Warning: {len(not_bound)} particle(s) were not bound to source; skipped:"
                )
                for pid in not_bound:
                    typer.echo(f"    {pid[:8]}…")
            typer.echo("  (no changes made)")
            return
        typer.echo(
            f"Split {len(split_relinked)} particle(s) off {source.id[:8]}…"
            f" ({source.canonical_name})"
        )
        typer.echo(
            f"  → {new_subject.id[:8]}… ({new_subject.canonical_name})"
            + (
                f"  [{', '.join(f'{r.namespace}:{r.id}' for r in new_subject.external_ids)}]"
                if new_subject.external_ids
                else ""
            )
        )
        if not_bound:
            typer.echo(
                f"  Skipped {len(not_bound)} particle(s) not bound to the source: "
                + ", ".join(pid[:8] + "…" for pid in not_bound)
            )
        typer.echo(
            "  Next: re-export to render the corrected attribution; "
            "optionally re-lint for stale relation edges."
        )

    else:
        typer.echo(
            f"Unknown action: {action}."
            " Use list, search, show, alias, merge, split, delete, gc, set-class,"
            " confirm, unlink, fix-labels, find-duplicates, relink-gated, rebuild-aliases,"
            " or fold-artifacts.",
            err=True,
        )
        raise typer.Exit(1)


async def _relink_gated(
    *, apply: bool, tiers: list[int] | None, sample: int, seed: int
) -> RelinkReport:
    from particles.operations.subject_relink import (
        apply_gated_relink,
        build_report,
        plan_gated_relink,
    )

    async with session_scope() as session:
        plan = await plan_gated_relink(session, tiers=tiers)
        if not apply:
            return build_report(plan, sample=sample, seed=seed)
        result = await apply_gated_relink(session, plan, actor="cli:subjects-relink-gated")
        await session.commit()
        return build_report(plan, sample=sample, seed=seed, result=result)


def _merge_subjects(
    backend: Backend, source_ids: list[str], target_id: str, *, dry_run: bool, yes: bool
) -> None:
    """Fold every source into the target after one preview and one confirmation."""
    try:
        plan = run(backend.subject_merge(source_ids, target_id, dry_run=True))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    target = plan.subject
    verb = "Would merge" if dry_run else "Merging"
    typer.echo(
        f"{verb} {len(plan.sources)} subject(s) into {target.id[:8]}… ({target.canonical_name}):"
    )
    for src in plan.sources:
        typer.echo(
            f"  {src.subject_id[:8]}… {src.canonical_name}"
            f"  ({src.particles_relinked} particle link(s))"
        )
        for ref in src.refs_carried:
            typer.echo(f"    carries {ref} to the target")
        for ref in src.refs_withheld:
            typer.echo(f"    drops {ref}: it names another project")
    if plan.aliases_added:
        typer.echo(f"  Aliases to add: {', '.join(plan.aliases_added)}")
    typer.echo(f"  Particle links to re-point: {plan.particles_relinked}")
    typer.echo("  The source subjects are deleted. A merge cannot be undone.")
    if dry_run:
        typer.echo("  (no changes made)")
        return
    if not yes:
        confirm(f"Merge {len(plan.sources)} subject(s)?", abort=True)

    # Re-plan inside the write: the store validates every source again, so a
    # source another session merged meanwhile refuses the whole merge.
    try:
        outcome = run(backend.subject_merge([s.subject_id for s in plan.sources], target.id))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    updated = outcome.subject
    typer.echo(
        f"Merged {len(outcome.sources)} subject(s)"
        f" into {updated.id[:8]}… ({updated.canonical_name})"
    )
    if outcome.aliases_added:
        typer.echo(f"  Aliases added: {', '.join(outcome.aliases_added)}")
    typer.echo(f"  Particles re-linked: {outcome.particles_relinked}")
    typer.echo("  Source subjects deleted.")


def _print_relink_report(report: RelinkReport) -> None:
    tiers = ", ".join(str(t) for t in report.tiers)
    typer.echo(f"Subjectless beliefs that owe a subject: {report.orphans}")
    typer.echo(f"Recoverable with tiers {tiers}: {report.recoverable}")
    for t, n in report.by_tier.items():
        typer.echo(f"  tier {t}: {n}")
    for cls, n in report.by_class.items():
        typer.echo(f"  {cls}: {n} name(s)")
    typer.echo(
        f"Withheld, no project key: {report.fail_closed_no_key}; "
        f"several projects: {report.fail_closed_several}; "
        f"nothing recoverable: {report.unrecovered}"
    )
    for item in report.sample:
        names = ", ".join(f"{n.name} [{n.token_class}, tier {n.tier}]" for n in item.names)
        typer.echo(f"\n  {item.particle_id[:8]}…  {names}\n    {item.content}")
    if report.applied:
        typer.echo(
            f"\nRelinked {report.relinked} belief(s) to {report.subjects} subject(s); "
            f"skipped {report.skipped}."
        )
    else:
        typer.echo("\nNothing written. Re-run with --apply to link them.")


def _checkout_files(repo: Path) -> tuple[str, list[str]]:
    """The project key and tracked file list of the checkout at ``repo``.

    The key is the Claude Code adapter's for the checkout's repository root,
    so a linked worktree names its main checkout's project.
    """
    import subprocess

    from particles.api.cli._claude_code import claude_project_slug, repository_root

    root = repository_root(repo.expanduser().absolute())
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "-C", str(repo), "ls-files", "-z"],  # noqa: S607
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        typer.echo(f"Cannot list the files of {repo}: {exc}", err=True)
        raise typer.Exit(2) from exc
    files = [f for f in done.stdout.decode("utf-8", "replace").split("\0") if f]
    return claude_project_slug(root), files


def _fold_artifacts(
    *, namespace: str | None, repo: Path | None, apply: bool, undo: str | None
) -> None:
    """Fold a file's other spellings into its path Subject, or undo one fold."""
    from particles.operations.artifact_fold import build_fold_report, undo_fold

    if undo is not None:

        async def _undo() -> list[str]:
            async with session_scope(write=True) as session:
                moved = await undo_fold(session, undo)
                await session.commit()
                return moved

        try:
            moved = run(_undo())
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Undid fold {undo}: {len(moved)} particle link(s) moved back.")
        return

    files: list[str] | None = None
    if repo is not None:
        repo_key, files = _checkout_files(repo)
        if namespace is not None and namespace != repo_key:
            typer.echo(
                f"--repo {repo} is project {repo_key}, not {namespace}; refusing to fold"
                " one project against another's files.",
                err=True,
            )
            raise typer.Exit(1)
        namespace = repo_key
    if namespace is None:
        typer.echo("Provide --repo DIR (or --namespace KEY to fold absolute paths only).", err=True)
        raise typer.Exit(1)

    async def _fold() -> FoldReport:
        async with session_scope(write=apply) as session:
            report = await build_fold_report(session, namespace, files, apply=apply)
            if apply:
                await session.commit()
            return report

    _print_fold_report(run(_fold()), applied=apply)


def _print_fold_report(report: FoldReport, *, applied: bool) -> None:
    from collections import Counter

    def _span(subject_id: str) -> str:
        span = report.spans.get(subject_id)
        if span is None or span.first is None or span.last is None:
            return "0 claims"
        return f"{span.claims} claims, {span.first:%Y-%m-%d}..{span.last:%Y-%m-%d}"

    source = f"{report.repo_files} files in the checkout" if report.repo_files else "no checkout"
    typer.echo(f"Project {report.namespace_key} ({source}).")
    for fold in report.folds:
        typer.echo(
            f"  {fold.rule.value:8s} {fold.source_name}  [{_span(fold.source_id)}]"
            f"\n           -> {fold.target_path}  [{_span(fold.target_id)}]"
        )
    by_rule = Counter(f.rule.value for f in report.folds)
    typer.echo(
        f"\n{len(report.folds)} fold(s): "
        + (", ".join(f"{n} {rule}" for rule, n in sorted(by_rule.items())) or "none")
    )
    if report.refusals:
        typer.echo(f"{len(report.refusals)} left as they are:")
        for reason, n in Counter(r.reason.value for r in report.refusals).most_common():
            typer.echo(f"  {n:5d}  {reason}")
        for r in report.refusals:
            typer.echo(f"  {r.rule.value:8s} {r.name}: {r.reason.value}")
    if applied:
        typer.echo(
            f"\nApplied {len(report.applied_events)} fold(s). Undo one with"
            " `particles subjects fold-artifacts --undo EVENT_ID`."
        )
    else:
        typer.echo("\nNothing written. Re-run with --apply to merge them.")


def _rebuild_aliases(*, check: bool) -> None:
    """Rebuild the alias index from the recorded aliases, or report its drift."""
    from particles.store.subject_store import check_alias_index, rebuild_alias_index

    async def _check() -> AliasIndexDrift:
        async with session_scope() as session:
            return await check_alias_index(session)

    async def _rebuild() -> int:
        async with session_scope(write=True) as session:  # writer lock
            written = await rebuild_alias_index(session)
            await session.commit()
            return written

    if not check:
        typer.echo(f"Alias index rebuilt: {run(_rebuild())} row(s).")
        return
    drift = run(_check())
    if drift.clean:
        typer.echo("Alias index matches the recorded aliases.")
        return
    typer.echo(
        f"Alias index differs from a rebuild: {drift.missing} missing, {drift.extra} extra,"
        f" {drift.orphaned} for deleted subjects, across {len(drift.subject_ids)} subject(s)."
        " Run `particles subjects rebuild-aliases` to repair it.",
        err=True,
    )
    for sid in drift.subject_ids[:20]:
        typer.echo(f"  {sid}", err=True)
    raise typer.Exit(1)


async def _find_duplicate_subjects() -> list[tuple[Subject, Subject, float]] | None:
    """Return candidate-duplicate pairs, or None if the embedding model is absent."""
    from particles.config import get_config
    from particles.embeddings import get_embedding_model
    from particles.store.subject_store import find_duplicate_subjects

    if get_embedding_model() is None:
        return None
    threshold = get_config().subjects.find_duplicates_similarity_threshold
    async with session_scope() as session:
        return await find_duplicate_subjects(session, threshold=threshold)


async def _list_phantom_subjects() -> list[Subject]:
    from particles.store.subject_store import get_phantom_subjects

    async with session_scope() as session:
        return await get_phantom_subjects(session)


async def _subject_particle_counts(subject_ids: list[str]) -> dict[str, int]:
    from particles.store.subject_store import get_particle_count_for_subject

    async with session_scope() as session:
        return {sid: await get_particle_count_for_subject(session, sid) for sid in subject_ids}


async def _fix_subject_labels() -> tuple[int, int, int, int]:
    """Repair raw-id labels: subjects named by a QID, and poisoned cache rows.

    A subject whose canonical_name is a raw QID gets its fetched label. A
    ``wikidata_label_cache`` row whose label equals its own id (item or
    property) may be an outage's fallback persisted before, so it is
    re-fetched: a fetched label replaces it, and a failed fetch deletes it so
    the next extraction asks again. A row Wikidata confirms (an entity with no
    English label) is kept.

    Returns:
        (subjects fixed, subjects skipped, cache rows repaired, cache rows cleared).
    """
    import re as _re

    from sqlalchemy import delete as _delete
    from sqlalchemy import select as _select

    from particles.extraction.wikidata import fetch_live_label
    from particles.store.subject_store import SubjectRow
    from particles.store.wikidata_cache import WikidataLabelRow, get_label, set_label

    _QID_RE = _re.compile(r"^Q\d+$")

    # Refuse up front rather than report every row as "no label found".
    from particles import egress

    egress.check(*egress.WIKIDATA_HOST, "subjects fix-labels")

    fixed = skipped = repaired = cleared = 0
    async with session_scope() as session:
        # The label cache first, so a repaired item row serves the subjects below.
        cached = await session.execute(
            _select(WikidataLabelRow.qid).where(WikidataLabelRow.label == WikidataLabelRow.qid)
        )
        for entity_id in list(cached.scalars()):
            fetched = await fetch_live_label(entity_id)
            if fetched is None:
                await session.execute(
                    _delete(WikidataLabelRow).where(WikidataLabelRow.qid == entity_id)
                )
                cleared += 1
            elif fetched != entity_id:
                await set_label(session, entity_id, fetched)
                repaired += 1
                typer.echo(f"  cache {entity_id} → {fetched}")

        result = await session.execute(_select(SubjectRow))
        rows = [r for r in result.scalars() if _QID_RE.match(r.canonical_name)]

        for row in rows:
            qid = row.canonical_name
            # Check DB cache first
            label = await get_label(session, qid)
            if label is None or label == qid:
                label = await fetch_live_label(qid)
                if label is not None and label != qid:
                    await set_label(session, qid, label)

            if label and label != qid:
                row.canonical_name = label
                fixed += 1
                typer.echo(f"  {qid} → {label}")
            else:
                skipped += 1

        await session.commit()
    return fixed, skipped, repaired, cleared


async def _delete_subject(subject_id: str) -> int:
    """Delete a subject; return the number of particles it was detached from."""
    from particles.store.subject_store import delete_subject

    async with session_scope() as session:
        _, detached = await delete_subject(session, subject_id)
        await session.commit()
        return detached


async def _set_subject_class_command(
    subject_id: str, subject_class: str
) -> tuple[Subject, str | None]:
    """Override a subject's Nomisma class; record the operator event."""
    from particles.store.subject_store import reclassify_subject

    async with session_scope() as session:
        updated, previous = await reclassify_subject(session, subject_id, subject_class)
        await session.commit()
        return updated, previous


async def _gc_phantom_subjects(*, dry_run: bool) -> tuple[list[Subject], int]:
    """Sweep every phantom subject (zero ACTIVE CLAIM particles).

    Reuses the same phantom definition as ``list --phantoms-only`` and the
    ``delete`` guard. Each deletion detaches the subject from any non-active /
    non-CLAIM particles still referencing it (recorded in the operator event
    log as a SUBJECT_DELETED event with actor ``subjects-gc``).

    Returns:
        Tuple of (pruned_subjects, total_detached). In dry-run mode nothing is
        committed and ``total_detached`` is 0 (the phantoms are listed only).
    """
    from particles.store.subject_store import delete_subject, get_phantom_subjects

    async with session_scope() as session:
        phantoms = await get_phantom_subjects(session)
        if dry_run or not phantoms:
            return phantoms, 0
        total_detached = 0
        for s in phantoms:
            _, detached = await delete_subject(session, s.id, actor="subjects-gc")
            total_detached += detached
        await session.commit()
        return phantoms, total_detached


async def _confirm_subject_link(subject_id: str, namespace: str, ref_id: str) -> bool:
    from particles.store.subject_store import set_external_ref_confidence

    async with session_scope() as session:
        ok = await set_external_ref_confidence(session, subject_id, namespace, ref_id, 1.0)
        if ok:
            await session.commit()
        return ok


async def _unlink_subject_ref(subject_id: str, namespace: str, ref_id: str) -> bool:
    from particles.store.subject_store import remove_external_ref

    async with session_scope() as session:
        ok = await remove_external_ref(session, subject_id, namespace, ref_id)
        if ok:
            await session.commit()
        return ok
