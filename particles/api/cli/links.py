# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""links sub-Typer — manage typed relations between particles."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer
from sqlalchemy.ext.asyncio import AsyncSession

from particles.api.cli import app, run
from particles.api.cli._progress import confirm
from particles.core.schema import RelationType
from particles.db import session_scope
from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

if TYPE_CHECKING:
    from particles.core.schema import DedupReport, SuggestReport, UnmergeReport
    from particles.operations.reread_merge import RereadReport

links_app = typer.Typer(
    help="Manage typed relations between particles (e.g. co-evidential links).",
    no_args_is_help=True,
)
app.add_typer(links_app, name="links")


@links_app.command("add")
def links_add_cmd(
    particle_a: str = typer.Argument(..., help="Particle A ID (prefix OK, ≥ 8 chars)"),
    particle_b: str = typer.Argument(..., help="Particle B ID (prefix OK, ≥ 8 chars)"),
    relation_type: str = typer.Option(
        "co-evidential",
        "--type",
        help="Relation type: co-evidential, part-of, or sequence-in. "
        "part-of / sequence-in are directional (A → B).",
    ),
    confidence: float = typer.Option(
        1.0,
        "--confidence",
        help="Link confidence in [0, 1]. Defaults to 1.0 for manual operator links.",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Create a typed relation between two particles."""
    run(_links_add(particle_a, particle_b, relation_type, confidence))


@links_app.command("remove")
def links_remove_cmd(
    particle_a: str = typer.Argument(..., help="Particle A ID (prefix OK, ≥ 8 chars)"),
    particle_b: str = typer.Argument(..., help="Particle B ID (prefix OK, ≥ 8 chars)"),
    relation_type: str = typer.Option(
        "co-evidential",
        "--type",
        help="Relation type to remove.",
    ),
) -> None:
    """Remove a typed relation between two particles."""
    run(_links_remove(particle_a, particle_b, relation_type))


@links_app.command("list")
def links_list_cmd(
    particle_id: str = typer.Argument(..., help="Particle ID (prefix OK, ≥ 8 chars)"),
    kind: str | None = typer.Option(
        None,
        "--kind",
        help="Filter to one relation kind (e.g. co-evidential, part-of, endorses). "
        "Case-insensitive; hyphens and underscores both accepted.",
    ),
) -> None:
    """List all relations incident to a particle, and its full co-evidential group."""
    run(_links_list(particle_id, kind))


@links_app.command("suggest")
def links_suggest_cmd(
    subject: str | None = typer.Option(
        None,
        "--subject",
        help="Restrict to one Subject (ID, ID prefix, or name).",
    ),
    all_subjects: bool = typer.Option(
        False,
        "--all",
        help="Scan every Subject. Mutually exclusive with --subject.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="Cosine-similarity floor (default: links_suggest.candidate_threshold).",
        min=0.0,
        max=1.0,
    ),
    llm_judge: bool = typer.Option(
        False,
        "--llm-judge",
        help="Send each Subject's candidate cluster to the LLM for per-pair verdicts.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Implies --llm-judge; auto-link PARAPHRASE pairs (needs --yes past the cap).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm --apply when it would link more than apply_confirm_threshold pairs.",
    ),
    output_format: str = typer.Option(
        "markdown", "--output-format", help="Output format: markdown or json"
    ),
) -> None:
    """Propose (and optionally resolve) co-evidential candidate links."""
    if subject is None and not all_subjects:
        typer.echo("Specify --subject <id|name> or --all.", err=True)
        raise typer.Exit(1)
    if subject is not None and all_subjects:
        typer.echo("--subject and --all are mutually exclusive.", err=True)
        raise typer.Exit(1)
    run(_links_suggest(subject, threshold, llm_judge, apply, yes, output_format))


@links_app.command("dedup")
def links_dedup_cmd(
    subject: str | None = typer.Option(
        None,
        "--subject",
        help="Restrict to one Subject (ID, ID prefix, or name). Default: whole store.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Merge the groups. Requires links_suggest.auto_merge.enabled in config.yaml. "
        "Without this flag the run is a read-only census.",
    ),
    output_format: str = typer.Option(
        "markdown", "--output-format", help="Output format: markdown or json"
    ),
    limit: int = typer.Option(
        20, "--limit", help="Groups listed in markdown output (counts are always complete)."
    ),
    reread: bool = typer.Option(
        False,
        "--reread",
        help="Work on reworded copies of one passage of an append-only source "
        "(a transcript read twice before 1.166.0) instead of identical text. "
        "Without --judge or --apply the run is a read-only census with a cost estimate.",
    ),
    judge: bool = typer.Option(
        False,
        "--judge",
        help="With --reread: ask the duplicate judge and the update reading for "
        "the pairs that have no recorded answer. Paid; prints the estimate and "
        "asks first. Writes recorded verdicts only.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="With --judge: skip the confirmation"),
    sheet: Path | None = typer.Option(
        None,
        "--sheet",
        help="With --reread: write every SAME verdict to this CSV for labelling.",
    ),
    labels: Path | None = typer.Option(
        None,
        "--labels",
        help="With --reread: score a labelled sheet against the admission bar.",
    ),
) -> None:
    """Merge duplicate beliefs into one survivor.

    Exact content equality only, the same normalized key extract-time
    suppression uses (whitespace and trailing punctuation absorbed, wording and
    case preserved), so no similarity threshold and no LLM call.
    Redundant copies are linked CO_EVIDENTIAL to the survivor and superseded;
    nothing is ever deleted and the survivor is never mutated.

    With --reread the pairs are reworded copies of one passage of an
    append-only source. They merge only when the duplicate judge said they are
    one claim, no update reading said the value changed, and the pair's
    similarity stratum is admitted in links_suggest.reread.admitted.
    """
    if not reread and (judge or sheet is not None or labels is not None):
        typer.echo("--judge, --sheet and --labels need --reread.", err=True)
        raise typer.Exit(1)
    if reread:
        if subject is not None:
            typer.echo("--reread works on the whole store; --subject is not supported.", err=True)
            raise typer.Exit(1)
        if judge and apply:
            typer.echo("--judge and --apply are separate steps; run one at a time.", err=True)
            raise typer.Exit(1)
        if labels is not None:
            _score_labels(labels)
            return
        run(_links_dedup_reread(apply, judge, yes, sheet, output_format, limit))
        return
    run(_links_dedup(subject, apply, output_format, limit))


@links_app.command("unmerge")
def links_unmerge_cmd(
    event_id: str | None = typer.Argument(
        None, help="The DUPLICATES_MERGED event to revert (from `particles events list`)."
    ),
    merge_run: str | None = typer.Option(
        None, "--run", help="Revert every merge stamped with this run id instead of one event."
    ),
    since: datetime | None = typer.Option(
        None,
        "--since",
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        help="Revert every merge at or after this instant. For merges written "
        "before run ids existed.",
    ),
    until: datetime | None = typer.Option(
        None,
        "--until",
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        help="Exclusive upper bound for --since.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan without writing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    output_format: str = typer.Option(
        "markdown", "--output-format", help="Output format: markdown or json"
    ),
    limit: int = typer.Option(
        20, "--limit", help="Groups listed in markdown output (counts are always complete)."
    ),
) -> None:
    """Revert an exact-duplicate auto-merge, restoring the superseded copies.

    The exact inverse of `links dedup --apply`: the retained copies return to
    ACTIVE keeping their ids, the merge's own CO_EVIDENTIAL links are dropped,
    and the survivor is never touched. Copies that moved on since the merge are
    skipped and named rather than restored.
    """
    run(
        _links_unmerge(
            event_id,
            merge_run,
            since,
            until,
            dry_run=dry_run,
            yes=yes,
            output_format=output_format,
            limit=limit,
        )
    )


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _parse_relation_type(rt: str) -> RelationType:
    norm = rt.strip().lower().replace("-", "_").upper()
    try:
        return RelationType(norm)
    except ValueError:
        valid = ", ".join(t.value for t in RelationType)
        typer.echo(f"Unknown relation type {rt!r}. Valid: {valid}.", err=True)
        raise typer.Exit(1) from None


async def _resolve_particle_id(session: AsyncSession, id_prefix: str) -> str:
    """Resolve a particle ID prefix (≥ 8 chars) to a full UUID or exit."""
    from sqlalchemy import select

    from particles.api.cli._id_norm import normalise_particle_id
    from particles.store.particle_store import ParticleRow

    id_prefix = normalise_particle_id(id_prefix)
    if len(id_prefix) < 8:
        typer.echo(f"Particle ID prefix {id_prefix!r} must be at least 8 characters.", err=True)
        raise typer.Exit(1)
    if len(id_prefix) >= 36:
        row = await session.get(ParticleRow, id_prefix)
        if row is None:
            typer.echo(f"Particle {id_prefix!r} not found.", err=True)
            raise typer.Exit(1)
        return str(row.id)
    result = await session.execute(
        select(ParticleRow.id).where(
            ParticleRow.id.like(f"{escape_like_pattern(id_prefix)}%", escape=LIKE_ESCAPE)
        )
    )
    rows = result.scalars().all()
    if not rows:
        typer.echo(f"No particle matches prefix {id_prefix!r}.", err=True)
        raise typer.Exit(1)
    if len(rows) > 1:
        typer.echo(f"Ambiguous prefix {id_prefix!r} matches {len(rows)} particles:", err=True)
        for r in rows[:10]:
            typer.echo(f"  {r}", err=True)
        raise typer.Exit(1)
    return str(rows[0])


async def _links_add(
    particle_a_prefix: str,
    particle_b_prefix: str,
    relation_type_str: str,
    confidence: float,
) -> None:
    from particles.api.cli._id_norm import normalise_particle_id
    from particles.api.client import get_backend

    backend = get_backend()
    rt = _parse_relation_type(relation_type_str)
    # Prefix resolution reads the local store ⇒ local-only; the
    # engine /links endpoint takes full particle UUIDs.
    if backend.remote:
        a_id = normalise_particle_id(particle_a_prefix)
        b_id = normalise_particle_id(particle_b_prefix)
    else:
        async with session_scope() as session:
            a_id = await _resolve_particle_id(session, particle_a_prefix)
            b_id = await _resolve_particle_id(session, particle_b_prefix)
    if a_id == b_id:
        typer.echo("Cannot link a particle to itself.", err=True)
        raise typer.Exit(1)
    try:
        rel = await backend.links_add(a_id, b_id, relation_type=rt.value, confidence=confidence)
    except Exception as exc:
        typer.echo(f"Failed to create relation: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Linked {rel.particle_a[:8]}… ↔ {rel.particle_b[:8]}… "
        f"({rel.relation_type}, created_by={rel.created_by}, confidence={rel.confidence:.2f})"
    )


async def _links_remove(
    particle_a_prefix: str,
    particle_b_prefix: str,
    relation_type_str: str,
) -> None:
    from particles.api.cli._id_norm import normalise_particle_id
    from particles.api.client import get_backend

    backend = get_backend()
    rt = _parse_relation_type(relation_type_str)
    if backend.remote:
        a_id = normalise_particle_id(particle_a_prefix)
        b_id = normalise_particle_id(particle_b_prefix)
    else:
        async with session_scope() as session:
            a_id = await _resolve_particle_id(session, particle_a_prefix)
            b_id = await _resolve_particle_id(session, particle_b_prefix)
    removed = await backend.links_remove(a_id, b_id, relation_type=rt.value)
    if not removed:
        typer.echo(
            f"No {rt.value} relation found between {a_id[:8]}… and {b_id[:8]}….",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"Removed {rt.value} relation: {a_id[:8]}… ↔ {b_id[:8]}…")


async def _links_list(particle_id_prefix: str, kind_str: str | None = None) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.store.relation_store import (
        get_co_evidential_group,
        get_relations_for_particle,
    )

    ensure_local("links list")

    kind = _parse_relation_type(kind_str) if kind_str is not None else None

    async with session_scope() as session:
        pid = await _resolve_particle_id(session, particle_id_prefix)
        rels = await get_relations_for_particle(session, pid, relation_type=kind)

        typer.echo(f"Particle {pid}")
        typer.echo("")

        kind_label = f" of kind {kind.value}" if kind is not None else ""
        if not rels:
            typer.echo(f"No direct relations{kind_label}.")
        else:
            typer.echo(f"Direct relations{kind_label} ({len(rels)}):")
            for r in rels:
                other = r.particle_b if r.particle_a == pid else r.particle_a
                typer.echo(
                    f"  ↔ {other[:8]}…  type={r.relation_type}  "
                    f"created_by={r.created_by}  confidence={r.confidence:.2f}"
                )

        # Show the transitive CO_EVIDENTIAL group (may exceed direct
        # neighbours) — skipped when filtering to a non-co-evidential kind.
        if kind is not None and kind is not RelationType.CO_EVIDENTIAL:
            return
        typer.echo("")
        group = await get_co_evidential_group(session, pid)
        if len(group) > 1:
            typer.echo(f"Co-evidential group ({len(group)} members, transitive closure):")
            for member in sorted(group):
                marker = " ← this particle" if member == pid else ""
                typer.echo(f"  {member[:8]}…{marker}")
        else:
            typer.echo("Co-evidential group: singleton (only this particle).")


async def _resolve_subject_id(session: AsyncSession, subject_arg: str) -> str:
    """Resolve a ``--subject`` argument (ID, ID prefix, or name) to a Subject ID."""
    from particles.store.subject_store import lookup_subject

    try:
        subject = await lookup_subject(session, subject_arg)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    if subject is None:
        typer.echo(f"No subject matches {subject_arg!r} by ID, ID prefix, or name.", err=True)
        raise typer.Exit(1)
    return subject.id


async def _links_suggest(
    subject_arg: str | None,
    threshold: float | None,
    llm_judge: bool,
    apply: bool,
    yes: bool,
    output_format: str,
) -> None:
    from particles.api.client import get_backend
    from particles.api.client.http import EngineHttpError
    from particles.core.schema import SuggestMode
    from particles.operations.links_suggest import ApplyConfirmationRequired

    backend = get_backend()
    mode = (
        SuggestMode.APPLY if apply else SuggestMode.LLM_JUDGE if llm_judge else SuggestMode.REPORT
    )

    # Subject name→id resolution reads the local store ⇒ local-only;
    # in remote mode --subject must be a Subject ID (the engine takes subject_id).
    if subject_arg is None:
        subject_id: str | None = None
    elif backend.remote:
        subject_id = subject_arg
    else:
        async with session_scope() as session:
            subject_id = await _resolve_subject_id(session, subject_arg)

    try:
        report = await backend.links_suggest(
            subject_id=subject_id, threshold=threshold, mode=mode, confirmed=yes
        )
    except ApplyConfirmationRequired as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except EngineHttpError as exc:
        # The engine returns 409 when --apply would exceed the confirm threshold
        # (its own ApplyConfirmationRequired surfaced over HTTP); surface cleanly.
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if output_format == "json":
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(_render_suggest_markdown(report))


async def _links_dedup(
    subject_arg: str | None,
    apply: bool,
    output_format: str,
    limit: int,
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.links_suggest import (
        AutoMergeDisabled,
        auto_merge_exact_duplicates,
    )

    # Store-mutating curation over the local ledger; the engine exposes no
    # dedup route (ships no new operation namespace).
    ensure_local("links dedup")

    # --apply is a short, pure write whose whole session body is the
    # write transaction, so it holds the cross-process writer lock. A dry run
    # opens no write at all.
    async with session_scope(write=apply) as session:
        subject_id = (
            await _resolve_subject_id(session, subject_arg) if subject_arg is not None else None
        )
        try:
            report = await auto_merge_exact_duplicates(
                session, subject_id=subject_id, dry_run=not apply
            )
        except AutoMergeDisabled as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

    if output_format == "json":
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(_render_dedup_markdown(report, limit))


def _score_labels(path: Path) -> None:
    import csv

    from particles.operations.reread_merge import score_labels

    if not path.is_file():
        typer.echo(f"No such sheet: {path}", err=True)
        raise typer.Exit(1)
    with path.open(newline="", encoding="utf-8") as fh:
        scores = score_labels(csv.DictReader(fh))
    typer.echo(f"links dedup --reread --labels  {path}")
    for score in scores:
        verdict = "passes" if score.passes else "does not pass"
        rule = f"{score.rule_correct}/{score.rule_labelled}" if score.rule_labelled else "no labels"
        typer.echo(
            f"  {score.stratum:<4} {score.correct}/{score.labelled} SAME verdicts correct, "
            f"Wilson lower bound {score.lower_bound:.3f}: {verdict} "
            f"(needs at least 100 labels and 0.950). Write rule: {rule}."
        )


async def _links_dedup_reread(
    apply: bool,
    judge: bool,
    yes: bool,
    sheet: Path | None,
    output_format: str,
    limit: int,
) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.operations.links_suggest import AutoMergeDisabled
    from particles.operations.reread_merge import (
        RereadNotAdmitted,
        apply_reread,
        census_reread,
        judge_reread,
        sheet_texts,
        write_sheet,
    )

    ensure_local("links dedup --reread")

    def _confirm(report: RereadReport) -> bool:
        # On stderr under --output-format json, so stdout stays one JSON document.
        typer.echo(_render_reread_markdown(report, limit), err=output_format == "json")
        if yes:
            return True
        return confirm("Ask these questions now?", default=False)

    async def _run() -> RereadReport:
        # --apply is one write and holds the writer lock throughout;
        # the census reads only; --judge commits each answered batch under its
        # own write transaction, so no lock is held across a model call.
        async with session_scope(write=apply) as session:
            try:
                if apply:
                    report = await apply_reread(session)
                elif judge:
                    report = await judge_reread(
                        session,
                        confirm=_confirm,
                        progress=lambda m: typer.echo(f"  {m}", err=True),
                    )
                else:
                    report = await census_reread(session)
            except (AutoMergeDisabled, RereadNotAdmitted) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1) from exc
            if sheet is not None:
                rows = write_sheet(report, await sheet_texts(session, report), sheet)
                typer.echo(f"Wrote {rows} SAME verdict(s) to {sheet}.", err=True)
        return report

    if judge:
        # The judge and the update readings are paid calls: metered as extract
        # is, so `quality` and the digest count them. The census and
        # --apply make no call and record nothing.
        from particles.llm.usage import render_usage_line
        from particles.operations.llm_spend import MeteredExtractRun

        meter = MeteredExtractRun(actor="cli:links-dedup-reread", route="reread-judge")
        try:
            async with meter:
                report = await _run()
        finally:
            if meter.llm_usage is not None:
                typer.echo(render_usage_line(meter.llm_usage), err=True)
    else:
        report = await _run()

    if output_format == "json":
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(_render_reread_markdown(report, limit))


async def _links_unmerge(
    event_id: str | None,
    merge_run: str | None,
    since: datetime | None,
    until: datetime | None,
    *,
    dry_run: bool,
    yes: bool,
    output_format: str,
    limit: int,
) -> None:
    """Plan, show, confirm, then revert.

    Deliberately **not** gated on ``links_suggest.auto_merge.enabled``: that
    flag authorizes *merging*. Gating the undo on it would mean an operator who
    turned the flag off after a bad run cannot clean up — the same reasoning
    A prior design kept the operator retract verb off the agent-facing
    write policy.
    """
    from particles.api.cli._remote import ensure_local
    from particles.operations.links_suggest import (
        UnmergeSelectorError,
        unmerge_exact_duplicates,
    )

    # Store-mutating curation over the local ledger; the engine exposes no
    # unmerge route (keeps this off the agent and HTTP surfaces).
    ensure_local("links unmerge")

    selector = {"event_id": event_id, "run_id": merge_run, "since": since, "until": until}

    # Plan in a read-only session first, so the confirmation shows the real
    # blast radius — including every drift skip — rather than a promise.
    async with session_scope() as session:
        try:
            plan = await unmerge_exact_duplicates(session, **selector, dry_run=True)  # type: ignore[arg-type]
        except UnmergeSelectorError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    if dry_run or not plan.restored_particles:
        if output_format == "json":
            typer.echo(plan.model_dump_json(indent=2))
            return
        typer.echo(_render_unmerge_markdown(plan, limit))
        if not plan.restored_particles:
            typer.echo("\nNothing to restore.")
        elif dry_run:
            typer.echo("\n--dry-run: nothing written.")
        return

    if not yes:
        # Narration on stderr so `--output-format json` still
        # emits a clean artifact on stdout when the operator confirms.
        typer.echo(_render_unmerge_markdown(plan, limit), err=True)
        confirm(
            f"\nRestore {plan.restored_particles} superseded copy/copies "
            f"across {len(plan.groups)} group(s)?",
            abort=True,
        )

    async with session_scope(write=True) as session:  # writer lock
        report = await unmerge_exact_duplicates(session, **selector, dry_run=False)  # type: ignore[arg-type]

    if output_format == "json":
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(_render_unmerge_markdown(report, limit))


def _render_unmerge_markdown(report: UnmergeReport, limit: int) -> str:
    """Human-readable summary of an UnmergeReport for the CLI."""
    mode = "plan — nothing written" if report.dry_run else "applied"
    lines = [
        f"links unmerge ({mode})  {report.run_at.strftime('%Y-%m-%d %H:%M')} UTC",
        f"  selector: {report.selector}",
        f"  {report.total_events} merge event(s): "
        f"{report.restored_particles} copy/copies restored to ACTIVE, "
        f"{report.skipped_particles} skipped, "
        f"{report.relations_deleted} EXACT_DUPLICATE link(s) dropped",
    ]
    for warning in report.warnings:
        lines.append(f"  ⚠ {warning}")
    if not report.groups:
        return "\n".join(lines)

    shown = report.groups[:limit]
    lines.append("")
    lines.append(f"Showing {len(shown)} of {len(report.groups)} group(s):")
    for group in shown:
        marker = " ✓ reverted" if group.reverted else ""
        lines.append(
            f"  survivor {group.survivor_id[:8]}… "
            f"({group.survivor_status or 'missing'}, untouched){marker}"
        )
        if group.restored_ids:
            restored = ", ".join(pid[:8] + "…" for pid in group.restored_ids)
            lines.append(f"        restore {len(group.restored_ids)}: {restored}")
        for skip in group.skipped:
            detail = skip.found_status or "not in store"
            if skip.found_status_reason:
                detail += f" / {skip.found_status_reason}"
            lines.append(f"        skip    {skip.particle_id[:8]}… {skip.reason.value} ({detail})")
    if len(report.groups) > len(shown):
        lines.append(f"  … {len(report.groups) - len(shown)} more group(s) not shown (--limit)")
    return "\n".join(lines)


def _render_reread_markdown(report: RereadReport, limit: int) -> str:
    """Human-readable summary of a RereadReport for the CLI."""
    title = {
        "census": "census, nothing written",
        "judge": "judge, recorded verdicts only",
        "apply": "applied",
    }[report.mode]
    lines = [
        f"links dedup --reread ({title})  {report.run_at.strftime('%Y-%m-%d %H:%M')} UTC",
        f"  {report.population_pairs} pair(s) of {report.finder_pairs} duplicate candidates "
        "are two readings of one append-only source",
        f"  {report.components} group(s), {report.questions} question(s) "
        f"(high {report.by_stratum.get('high', 0)}, low {report.by_stratum.get('low', 0)}); "
        f"not asked: {report.skipped_shared_snapshot} share a snapshot with the survivor, "
        f"{report.skipped_extra_subject} carry a Subject it lacks",
        f"  up to {report.resolvable_pairs} of the {report.population_pairs} pair(s) can resolve",
        f"  recorded: {report.judged} judged ({report.same} SAME, {report.different} DIFFERENT), "
        f"{report.update_read} update reading(s), {report.update_changes} saying CHANGES",
        f"  admitted: {report.admitted}; the write rule allows "
        f"{report.would_merge.get('high', 0)} high and {report.would_merge.get('low', 0)} low",
    ]
    if report.blocked:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(report.blocked.items()))
        lines.append(f"  not merging: {parts}")
    if report.mode == "judge":
        lines.append(
            f"  asked: {report.asked_judge} judge question(s), "
            f"{report.asked_update} update reading(s)"
        )
    if report.mode == "apply":
        lines.append(
            f"  merged {report.merged_groups} group(s): {report.merged_copies} copy/copies "
            f"superseded, {report.links_created} link(s) created; run {report.run_id}"
        )
    est = report.estimate
    if est is not None and (est.judge_questions or est.update_readings):

        def usd(value: float | None) -> str:
            return "unpriced" if value is None else f"US${value:.2f}"

        lines.append(
            f"  to ask: {est.judge_questions} judge question(s) in {est.judge_calls} call(s) "
            f"on {est.judge_model}, up to {est.update_readings} update reading(s) on "
            f"{est.update_model}: about {est.input_tokens:,} input tokens, "
            f"{usd(est.usd_expected)} expected, {usd(est.usd_ceiling)} at the full output "
            "budget"
        )
        if est.second_reading_usd_ceiling_each is not None:
            lines.append(
                f"  plus a second reading on {est.second_model} after each first CHANGES, "
                f"at most {usd(est.second_reading_usd_ceiling_each)} each"
            )
    for warning in report.warnings:
        lines.append(f"  ⚠ {warning}")
    if report.mode == "apply" and limit:
        merged = [i for i in report.items if i.merged][:limit]
        if merged:
            lines.append("")
            lines.append(f"First {len(merged)} merged copy/copies:")
            for item in merged:
                lines.append(f"  {item.copy_id[:8]} → {item.survivor_id[:8]}  ({item.stratum})")
    return "\n".join(lines)


def _render_dedup_markdown(report: DedupReport, limit: int) -> str:
    """Human-readable summary of a DedupReport for the CLI."""
    mode = "dry run — nothing written" if report.dry_run else "applied"
    lines = [
        f"links dedup ({mode})  {report.run_at.strftime('%Y-%m-%d %H:%M')} UTC",
        f"  {report.total_groups} exact-duplicate group(s), "
        f"{report.total_redundant} redundant copy/copies",
    ]
    if not report.dry_run:
        lines.append(
            f"  merged {report.merged_groups} group(s): {report.merged_particles} copy/copies "
            f"superseded, {report.links_created} CO_EVIDENTIAL link(s) created"
        )
    for warning in report.warnings:
        lines.append(f"  ⚠ {warning}")
    if not report.groups:
        lines.append("  (no exact duplicates)")
        return "\n".join(lines)

    # break the groups out by Subject composition. `orphan` and
    # `mixed` are the reach this ADR added — an operator whose config already
    # said auto_merge.enabled: true consented to the Subject-scoped pass, so
    # the widening is disclosed here rather than behind a second config knob.
    by_class = Counter(g.subject_class for g in report.groups)
    if by_class["orphan"] or by_class["mixed"]:
        redundant_by_class: Counter[str] = Counter()
        for group in report.groups:
            redundant_by_class[group.subject_class] += len(group.redundant_ids)
        lines.append("  by subject composition:")
        classes: tuple[tuple[Literal["linked", "mixed", "orphan"], str], ...] = (
            ("linked", "every copy subject-linked"),
            ("mixed", "some copies subject-less"),
            ("orphan", "no copy subject-linked"),
        )
        for cls, label in classes:
            if by_class[cls]:
                lines.append(
                    f"    {cls:<7} {by_class[cls]:>4} group(s), "
                    f"{redundant_by_class[cls]:>4} redundant  ({label})"
                )

    shown = sorted(report.groups, key=lambda g: (-len(g.redundant_ids), g.survivor_id))[:limit]
    lines.append("")
    lines.append(f"Largest {len(shown)} of {report.total_groups} group(s):")
    for group in shown:
        marker = " ✓ merged" if group.merged else ""
        excerpt = group.content_excerpt.replace("\n", " ")
        if len(excerpt) > 96:
            excerpt = excerpt[:95] + "…"
        lines.append(f"  ×{len(group.redundant_ids) + 1:<3} keep {group.survivor_id[:8]}…{marker}")
        lines.append(f"        {excerpt}")
    if report.total_groups > len(shown):
        lines.append(f"  … {report.total_groups - len(shown)} more group(s) not shown (--limit)")
    if report.dry_run:
        lines.append("")
        lines.append(
            "Nothing was written. Re-run with --apply (needs "
            "links_suggest.auto_merge.enabled: true) to merge."
        )
    return "\n".join(lines)


def _render_suggest_markdown(report: SuggestReport) -> str:
    """Human-readable summary of a SuggestReport for the CLI."""
    lines: list[str] = []
    head = (
        f"links suggest ({report.mode.value.lower()})  "
        f"{report.run_at.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    lines.append(head)
    lines.append(
        f"  {report.total_candidates} candidate pair(s) across {len(report.clusters)} subject(s)"
    )
    if report.judged_pairs:
        lines.append(f"  {report.judged_pairs} pair(s) LLM-judged")
    if report.applied_pairs:
        lines.append(f"  {report.applied_pairs} pair(s) linked CO_EVIDENTIAL")
    for warning in report.warnings:
        lines.append(f"  ⚠ {warning}")
    for cluster in report.clusters:
        lines.append("")
        lines.append(f"▸ {cluster.subject_name or cluster.subject_id}")
        for c in cluster.candidates:
            verdict = f"  [{c.verdict.value}]" if c.verdict is not None else ""
            applied = "  ✓ linked" if c.applied else ""
            lines.append(
                f"    {c.particle_a[:8]}… ↔ {c.particle_b[:8]}…  "
                f"sim={c.similarity:.3f}{verdict}{applied}"
            )
    if not report.clusters:
        lines.append("  (no candidates)")
    return "\n".join(lines)
