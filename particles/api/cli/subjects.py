"""subjects command — manage canonical real-world entities.

This is a single Typer command with manual action dispatch via a positional
argument (list, search, show, alias, merge, confirm, unlink, split, delete,
gc, set-class, fix-labels, find-duplicates) — Typer sub-Typer style would be
more idiomatic but the current shape is preserved for backwards compatibility.
"""

from __future__ import annotations

import typer

from particles.api.cli import app, run
from particles.core.schema import Subject
from particles.db import session_scope

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
    }
)


@app.command("subjects")
def subjects_cmd(
    action: str = typer.Argument(
        "list",
        help=(
            "Action: list, search, show, alias, confirm, unlink, merge, split, "
            "delete, gc, set-class, fix-labels, find-duplicates"
        ),
    ),
    rest: list[str] | None = typer.Argument(None, help="Arguments for the chosen action"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview merge/split/gc without committing"
    ),
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
) -> None:
    """Manage subjects (canonical real-world entities).

    \b
        particles subjects list [--order name|degree] [--phantoms-only]
        particles subjects search QUERY
        particles subjects show ID
        particles subjects alias ID NAME [NAME ...]
        particles subjects confirm SUBJECT_ID NAMESPACE:ID
        particles subjects unlink SUBJECT_ID NAMESPACE:ID
        particles subjects merge SOURCE_ID TARGET_ID [--dry-run]
        particles subjects delete SUBJECT_ID [--force]
        particles subjects gc [--dry-run]          (alias: prune-empty)
        particles subjects set-class SUBJECT_ID CLASS   (e.g. nmo:NumismaticObject)
        particles subjects split SOURCE_ID --particle PID [--particle PID ...] \\
            (--new-name "Applied Optoelectronics" | --new-external-id wikidata:Q30297735) \\
            [--dry-run]
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
        for s in subjects:
            ext = ", ".join(f"{r.namespace}:{r.id}" for r in s.external_ids)
            typer.echo(f"  {s.id[:8]}…  {s.canonical_name}" + (f"  [{ext}]" if ext else ""))
        if not subjects:
            typer.echo(f"No subjects matching '{args[0]}'.")

    elif action == "show":
        if not args:
            typer.echo("Provide a subject ID.", err=True)
            raise typer.Exit(1)
        subject = run(backend.subject_show(args[0]))
        if subject is None:
            typer.echo(f"Subject {args[0]} not found.", err=True)
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
        fixed, skipped = run(_fix_subject_labels())
        typer.echo(f"Fixed: {fixed}  Skipped (no label found): {skipped}")

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
            typer.echo("Usage: subjects merge SOURCE_ID TARGET_ID [--dry-run]", err=True)
            raise typer.Exit(1)
        source_id, target_id = args[0], args[1]
        source = run(backend.subject_show(source_id))
        target = run(backend.subject_show(target_id))
        if source is None:
            typer.echo(f"Source subject {source_id!r} not found.", err=True)
            raise typer.Exit(1)
        if target is None:
            typer.echo(f"Target subject {target_id!r} not found.", err=True)
            raise typer.Exit(1)
        if dry_run:
            typer.echo(
                f"Dry run: would merge {source.id[:8]}… ({source.canonical_name})"
                f" → {target.id[:8]}… ({target.canonical_name})"
            )
            typer.echo("  (no changes made)")
            return
        merge_outcome = run(backend.subject_merge(source.id, target.id))
        updated = merge_outcome.subject
        added_aliases = merge_outcome.aliases_added
        relinked = merge_outcome.particles_relinked
        typer.echo(
            f"Merged {source.id[:8]}… ({source.canonical_name})"
            f" → {updated.id[:8]}… ({updated.canonical_name})"
        )
        if added_aliases:
            typer.echo(f"  Aliases added: {', '.join(added_aliases)}")
        typer.echo(f"  Particles re-linked: {relinked}")
        typer.echo("  Source subject deleted.")

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
        #: sweep every phantom subject (zero ACTIVE CLAIM
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
        #: operator override of the resolver's Nomisma class.
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
            " confirm, unlink, fix-labels, or find-duplicates.",
            err=True,
        )
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


async def _fix_subject_labels() -> tuple[int, int]:
    """Find subjects whose canonical_name is a raw QID, fetch proper labels, update."""
    import re as _re

    from sqlalchemy import select as _select

    from particles.http import get_capped, particles_client
    from particles.store.subject_store import SubjectRow
    from particles.store.wikidata_cache import get_label, set_label

    _QID_RE = _re.compile(r"^Q\d+$")
    _WIKIDATA_REST = "https://www.wikidata.org/w/rest.php/wikibase/v1"

    fixed = skipped = 0
    async with session_scope() as session:
        result = await session.execute(_select(SubjectRow))
        rows = [r for r in result.scalars() if _QID_RE.match(r.canonical_name)]

        for row in rows:
            qid = row.canonical_name
            # Check DB cache first
            label = await get_label(session, qid)
            if label is None or label == qid:
                # Fetch from API
                try:
                    async with particles_client(timeout=10.0) as client:
                        resp = await get_capped(client, f"{_WIKIDATA_REST}/entities/items/{qid}")
                        resp.raise_for_status()
                        data = resp.json()
                        label = str(data.get("labels", {}).get("en", qid))
                except Exception:
                    label = qid

                if label != qid:
                    await set_label(session, qid, label)

            if label and label != qid:
                row.canonical_name = label
                fixed += 1
                typer.echo(f"  {qid} → {label}")
            else:
                skipped += 1

        await session.commit()
    return fixed, skipped


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
