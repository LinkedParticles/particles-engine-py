# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the two-project observer fixture (gate B).

Per seed: a scratch store; both projects' ``MEMORY.md`` deposited each day they
change, exactly as the SessionEnd harvest deposits them (``LOCAL_MARKDOWN``,
``MUTABLE``, ``file://`` URI, ``claude-code`` + ``project:<key>`` tags), then
extracted through the **real** pipeline with scripted perception; a keyless
hand deposit for the global lines; the store rescoped so the lens may engage.

At the end of every day, for each project observer, every line that project's
file currently states is probed: is a particle holding it in view? If not, why
— which mechanism retired it, and did the retiring event come
from the other project? Alongside, the no-lens control (is it ACTIVE anywhere?) and
the lens's own correctness (does the project see anything it never stated?).

Beside the read-side measurement, every extraction is bracketed by a scope
reading of the ACTIVE set, so each retirement is attributed as own or
cross-project at the moment it happened (:class:`WriteCensus`); after the last
day each project contests one global line, which must reach review rather than
retire the operator's claim.

Report-only: scratch stores, an isolated blob dir, zero LLM calls (every
purpose is scripted or refused and counted).
"""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.benchmark.memory.runner import isolated_blob_dir, scratch_store
from particles.benchmark.observer.generator import (
    GLOBAL_CONTESTS,
    days_with_changes,
    generate_world,
    lines_on_day,
    render_memory_file,
)
from particles.benchmark.observer.oracle import (
    ChunkedLinesExtractor,
    LinesExtractor,
    SlotProbeProvider,
)
from particles.benchmark.observer.schema import (
    GENERATOR_VERSION,
    PROJECT_A,
    PROJECT_B,
    PROJECTS,
    LineKind,
    LineObservation,
    ObserverArm,
    ObserverMetrics,
    ObserverReport,
    ObserverWorld,
    VanishCause,
    WorldResult,
    WriteCensus,
)
from particles.benchmark.rot.oracle import RefusingProvider
from particles.benchmark.rot.schema import Rate
from particles.config import get_config
from particles.core.observer_scope import project_tag
from particles.core.schema import Mutability, RelationCreatedBy, RelationType
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.store import get_tags_for_entries
from particles.embeddings import get_embedding_model
from particles.ingest.observer_gate import DivergenceTally
from particles.ingest.pipeline import extract_snapshot
from particles.llm import CompletionProvider, LLMPurpose, override_providers
from particles.operations.observer_scope import rescope
from particles.operations.query.observer_scope import filter_visible, load_scopes
from particles.store.particle_store import ParticleRow, ProvenanceEdgeRow, get_particles_by_status
from particles.store.relation_store import ParticleRelationRow

_PROJECTS_ROOT = "/home/me/.claude/projects"
GLOBAL_URI = "file:///home/me/notes/how-i-work.md"


class ObserverFixtureError(ValueError):
    """A run that cannot measure anything (no encoder)."""


def memory_uri(project: str) -> str:
    return f"file://{_PROJECTS_ROOT}/{project}/memory/MEMORY.md"


@contextlib.contextmanager
def _scripted_perception(refused: dict[str, int]) -> Iterator[SlotProbeProvider]:
    """Route the contradiction probe to the slot table and every other purpose to a refusal."""
    overrides: dict[LLMPurpose, CompletionProvider] = {
        purpose: RefusingProvider(purpose, refused) for purpose in get_args(LLMPurpose)
    }
    probe = SlotProbeProvider()
    overrides["semantic_lint"] = probe
    with override_providers(overrides):
        yield probe


@contextlib.contextmanager
def _offline_subjects() -> Iterator[None]:
    """Keep subject resolution local for the run: memory files are ``LOCAL_MARKDOWN``,
    which the default ``subjects.skip_live_authorities_source_types`` does not cover,
    and a benchmark must not query Wikidata for "the repository"."""
    cfg = get_config().subjects
    before = list(cfg.skip_live_authorities_source_types)
    if "LOCAL_MARKDOWN" not in before:
        cfg.skip_live_authorities_source_types = [*before, "LOCAL_MARKDOWN"]
    try:
        yield
    finally:
        cfg.skip_live_authorities_source_types = before


def thresholds() -> dict[str, float]:
    """Every write-side knob the result depends on."""
    cfg = get_config()
    return {
        "reconciliation.store_mode.single": float(cfg.reconciliation.store_mode == "single"),
        "reconciliation.update_supersession.enabled": float(
            cfg.reconciliation.update_supersession.enabled
        ),
        "reconciliation.update_supersession.subject_floor": (
            cfg.reconciliation.update_supersession.subject_floor
        ),
        "extraction.duplicate_suppression.enabled": float(
            cfg.extraction.duplicate_suppression.enabled
        ),
        "extraction.similarity_threshold": cfg.extraction.similarity_threshold,
        "trust.differential_threshold": cfg.trust.differential_threshold,
    }


async def run_observer_fixture(
    *,
    seeds: list[int],
    days: int,
    work_dir: Path,
    arm: ObserverArm = ObserverArm.LINES,
    keep_stores: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ObserverReport:
    """Run every seed on one arm and pool the metrics."""
    if get_embedding_model() is None:
        raise ObserverFixtureError(
            "no embedding model is available; the pipeline's candidate search needs one "
            "(install the `embeddings` extra or set one with set_embedding_model)."
        )
    started = datetime.now(UTC)
    refused: dict[str, int] = {}
    worlds: list[WorldResult] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    with isolated_blob_dir(work_dir), _offline_subjects(), _scripted_perception(refused):
        for seed in seeds:
            world = generate_world(seed, days)
            db_path = work_dir / f"observer-seed-{seed}.db"
            result = await _run_world(world, db_path=db_path, arm=arm)
            if not keep_stores:
                db_path.unlink(missing_ok=True)
            worlds.append(result)
            if progress is not None:
                progress(
                    f"seed {seed}: {result.deposits} deposits; own lines in view "
                    f"{_pct(result.metrics.own_visible)}"
                )
    census = WriteCensus()
    for w in worlds:
        census = census.merged(w.write_census)
    report = ObserverReport(
        seeds=seeds,
        days=days,
        arm=arm,
        generator_version=GENERATOR_VERSION,
        store_mode=get_config().reconciliation.store_mode,
        thresholds=thresholds(),
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        metrics=_pool([w.metrics for w in worlds]),
        write_census=census,
        worlds=worlds,
        refused_llm_calls=dict(refused),
    )
    if refused:
        report.quality_notes.append(
            "Some LLM purposes were called and refused (see refused_llm_calls); the pipeline "
            "took its documented fallback for each."
        )
    return report


def _pct(rate: Rate) -> str:
    return "n/a" if rate.value is None else f"{100 * rate.value:.0f}%"


async def _run_world(world: ObserverWorld, *, db_path: Path, arm: ObserverArm) -> WorldResult:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    extractor = ChunkedLinesExtractor() if arm is ObserverArm.CHUNKED else LinesExtractor()
    result = WorldResult(seed=world.seed, fingerprint=world.fingerprint)
    stated: dict[str, set[str]] = {p: set() for p in PROJECTS}
    changes = {p: days_with_changes(world, p) for p in PROJECTS}

    async with scratch_store(db_path) as factory, factory() as db:
        # The global lines: a hand deposit, keyless, so they are in view everywhere.
        global_text = render_memory_file(world.global_lines)
        extractor.register(global_text, world.global_lines)
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            db,
            text=global_text,
            uri_r=GLOBAL_URI,
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.STABLE,
            deposited_by="operator",
            content_published_at=epoch,
        )
        await db.commit()
        await extract_snapshot(db, entry_id, snapshot_id, extractor=extractor)
        result.deposits += 1
        # What lets a project observer engage on this store.
        await rescope(db, key_for=lambda _e, _u, _t: None, actor="benchmark-observer")
        await db.commit()

        entry_by_project: dict[str, str] = {}
        for day in range(1, world.days + 1):
            for seq, project in enumerate(PROJECTS):
                if day not in changes[project]:
                    continue
                lines = lines_on_day(world, project, day)
                stated[project].update(line.text for line in lines)
                text = render_memory_file(lines)
                extractor.register(text, lines)
                stamp = epoch + timedelta(days=day, minutes=10 * seq)
                entry_id, snapshot_id, unchanged = await deposit_text_versioned(
                    db,
                    text=text,
                    uri_r=memory_uri(project),
                    source_type="LOCAL_MARKDOWN",
                    mutability=Mutability.MUTABLE,
                    tags=["claude-code", "memory-file", project_tag(project)],
                    deposited_by="claude-code-hook",
                    content_published_at=stamp,
                )
                entry_by_project[project] = entry_id
                await db.commit()
                if not unchanged:
                    await _extract_attributed(
                        db, entry_id, snapshot_id, extractor, project, result.write_census
                    )
                    result.deposits += 1
            for project in PROJECTS:
                result.observations.extend(
                    await _observe(db, world, project, day, stated, entry_by_project)
                )

        await _contest_global_lines(db, world, extractor, epoch, result)
        result.write_census.divergences_recorded = int(
            (
                await db.execute(
                    select(func.count()).where(
                        ParticleRelationRow.relation_type == RelationType.CONTRADICTS.value,
                        ParticleRelationRow.created_by
                        == RelationCreatedBy.OBSERVER_DIVERGENCE.value,
                    )
                )
            ).scalar_one()
        )
        if isinstance(extractor, ChunkedLinesExtractor):
            result.write_census.chunks_carried = extractor.chunks_carried
            result.write_census.chunks_extracted = extractor.chunks_extracted

        rows = await db.execute(
            select(ParticleRow.status, ParticleRow.status_reason, func.count()).group_by(
                ParticleRow.status, ParticleRow.status_reason
            )
        )
        for status, reason, n in rows.all():
            result.store_census[status if reason is None else f"{status}/{reason}"] = int(n)

    result.metrics = metrics_for(result.observations)
    return result


async def _extract_attributed(
    db: AsyncSession,
    entry_id: str,
    snapshot_id: str,
    extractor: LinesExtractor,
    project: str,
    census: WriteCensus,
) -> None:
    """Extract one deposit, attributing every retirement it causes.

    The ACTIVE set's scopes are read just before the extraction, through the
    lens's own join: a retired claim that some *other* project observed at that
    moment is a cross-project retirement.
    """
    actives = await get_particles_by_status(db, Status.ACTIVE)
    before = await load_scopes(db, actives)
    tally = DivergenceTally()
    await extract_snapshot(db, entry_id, snapshot_id, extractor=extractor, divergences_out=tally)
    await db.commit()
    census.declined_pairs += tally.declined
    census.declined_without_relation += tally.declined - tally.recorded

    ids = list(before)
    retired: list[tuple[str, str | None]] = []
    for start in range(0, len(ids), 500):
        rows = await db.execute(
            select(ParticleRow.id, ParticleRow.status_reason).where(
                ParticleRow.id.in_(ids[start : start + 500]),
                ParticleRow.status != Status.ACTIVE.value,
            )
        )
        retired.extend((pid, reason) for pid, reason in rows.all())
    for particle_id, reason in retired:
        cross = bool(before[particle_id].keys - {project})
        if reason == StatusReason.SUPERSEDED_BY_UPDATE.value:
            if cross:
                census.cross_project_supersessions += 1
            else:
                census.own_supersessions += 1
        elif reason == StatusReason.RETRACTED_DEPENDENCY.value:
            if cross:
                census.cross_project_cascades += 1
            else:
                census.own_cascades += 1

    # A candidate this extraction stored already demoted (rung 2.5's mirror).
    born = await db.execute(
        select(func.count(func.distinct(ParticleRow.id)))
        .join(ProvenanceEdgeRow, ProvenanceEdgeRow.particle_id == ParticleRow.id)
        .where(
            ProvenanceEdgeRow.snapshot_id == snapshot_id,
            ParticleRow.status_reason == StatusReason.SUPERSEDED_BY_UPDATE.value,
        )
    )
    census.candidates_born_superseded += int(born.scalar_one())


async def _contest_global_lines(
    db: AsyncSession,
    world: ObserverWorld,
    extractor: LinesExtractor,
    epoch: datetime,
    result: WorldResult,
) -> None:
    """After the last day, each project contests one global line (§8).

    The operator's own claim must not be retired by a project's update: the pair
    goes to review, an ``INCONSISTENCY`` with the contesting line quarantined.
    """
    for seq, (project, (global_text, contest)) in enumerate(
        zip(PROJECTS, GLOBAL_CONTESTS.items(), strict=True)
    ):
        lines = [*lines_on_day(world, project, world.days), contest]
        text = render_memory_file(lines)
        extractor.register(text, lines)
        entry_id, snapshot_id, unchanged = await deposit_text_versioned(
            db,
            text=text,
            uri_r=memory_uri(project),
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            tags=["claude-code", "memory-file", project_tag(project)],
            deposited_by="claude-code-hook",
            content_published_at=epoch + timedelta(days=world.days + 1, minutes=10 * seq),
        )
        await db.commit()
        if not unchanged:
            await _extract_attributed(
                db, entry_id, snapshot_id, extractor, project, result.write_census
            )
        held = await _global_held_in_review(db, global_text)
        result.write_census.global_contests.add(held)


async def _global_held_in_review(db: AsyncSession, global_text: str) -> bool:
    """The global claim is still ACTIVE and an INCONSISTENCY names it."""
    rows = (
        (await db.execute(select(ParticleRow).where(ParticleRow.content == global_text)))
        .scalars()
        .all()
    )
    active = [r for r in rows if r.status == Status.ACTIVE.value]
    if not active:
        return False
    ids = {r.id for r in active}
    inconsistencies = (
        (
            await db.execute(
                select(ParticleRow.provenance_json).where(
                    ParticleRow.status == Status.INCONSISTENCY.value
                )
            )
        )
        .scalars()
        .all()
    )
    return any(any(pid in prov for pid in ids) for prov in inconsistencies)


async def _observe(
    db: AsyncSession,
    world: ObserverWorld,
    project: str,
    day: int,
    stated: dict[str, set[str]],
    entry_by_project: dict[str, str],
) -> list[LineObservation]:
    """Probe every line ``project`` currently states, through its observer."""
    actives = await get_particles_by_status(db, Status.ACTIVE)
    scope = await filter_visible(db, actives, project)
    active_texts = {p.content for p in actives}
    visible_texts = {p.content for p in actives if p.id in scope.visible_ids}
    own = lines_on_day(world, project, day)
    global_texts = {line.text for line in world.global_lines}

    observations: list[LineObservation] = []
    for line in own:
        obs = LineObservation(
            day=day,
            project=project,
            kind=line.kind,
            text=line.text,
            visible=line.text in visible_texts,
            visible_store_wide=line.text in active_texts,
        )
        if not obs.visible:
            await _explain(db, obs, project, entry_by_project, visible_texts)
        observations.append(obs)

    # The lens's correctness: anything in view this project never stated?
    other = PROJECT_B if project == PROJECT_A else PROJECT_A
    for text in sorted(visible_texts - stated[project] - global_texts):
        if text in stated[other]:
            observations.append(
                LineObservation(
                    day=day,
                    project=project,
                    kind=LineKind.RULE,
                    text=text,
                    visible=True,
                    visible_store_wide=True,
                    cause=None,
                    cross_project=True,
                    winner_in_view=None,
                )
            )
    return observations


async def _explain(
    db: AsyncSession,
    obs: LineObservation,
    project: str,
    entry_by_project: dict[str, str],
    visible_texts: set[str],
) -> None:
    """Why a line this project states is not in view for it."""
    rows = (
        (
            await db.execute(
                select(ParticleRow)
                .where(ParticleRow.content == obs.text)
                .order_by(ParticleRow.asserted_at.desc())
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        obs.cause = VanishCause.NEVER_MINTED
        return
    latest = rows[0]
    reason = latest.status_reason
    if reason == StatusReason.SUPERSEDED_BY_UPDATE.value:
        obs.cause = VanishCause.SUPERSEDED_BY_UPDATE
        winner = (
            await db.execute(select(ParticleRow).where(ParticleRow.supersedes == latest.id))
        ).scalar_one_or_none()
        if winner is not None:
            keys = await _project_keys_of(db, winner.id)
            obs.cross_project = project not in keys
            obs.winner_in_view = winner.content in visible_texts
        else:
            obs.cross_project = True
    elif reason == StatusReason.RETRACTED_DEPENDENCY.value:
        obs.cause = VanishCause.CASCADE
        # The line is in this project's current file, so its own entry's current
        # snapshot still attests it: the generation that moved was the other's.
        obs.cross_project = True
    elif reason == StatusReason.CONFLICT_PENDING.value:
        obs.cause = VanishCause.QUARANTINED
        obs.cross_project = True
    elif latest.status == Status.ACTIVE.value:
        # ACTIVE but not in view: the surviving particle is attested only by the
        # other project's sources. This project stated the same claim, but its
        # own particle was retired earlier and the restatement has not been
        # re-deposited since (folds only onto ACTIVE particles).
        obs.cause = VanishCause.ACTIVE_ELSEWHERE
        obs.cross_project = project not in await _project_keys_of(db, latest.id)
    else:
        obs.cause = VanishCause.OTHER


async def _project_keys_of(db: AsyncSession, particle_id: str) -> set[str]:
    entries = (
        (
            await db.execute(
                select(ProvenanceEdgeRow.corpus_entry_id).where(
                    ProvenanceEdgeRow.particle_id == particle_id
                )
            )
        )
        .scalars()
        .all()
    )
    tags = await get_tags_for_entries(db, set(entries))
    return {
        t[len("project:") :]
        for entry_tags in tags.values()
        for t in entry_tags
        if t.startswith("project:")
    }


def metrics_for(observations: list[LineObservation]) -> ObserverMetrics:
    m = ObserverMetrics()
    causes: Counter[str] = Counter()
    for obs in observations:
        if obs.cause is None and obs.cross_project and obs.visible:
            # A leak row (in view, never stated by this project).
            m.leaked.add(True)
            continue
        m.own_visible.add(obs.visible)
        m.own_visible_store_wide.add(obs.visible_store_wide)
        m.by_kind.setdefault(obs.kind.value, Rate()).add(obs.visible)
        m.vanished_cross_project.add(not obs.visible and bool(obs.cross_project))
        if obs.visible:
            m.leaked.add(False)
        else:
            causes[(obs.cause or VanishCause.OTHER).value] += 1
            if obs.cause is VanishCause.SUPERSEDED_BY_UPDATE and obs.cross_project:
                m.winner_in_view.add(bool(obs.winner_in_view))
    m.vanished_by_cause = dict(causes)
    return m


def _pool(all_metrics: list[ObserverMetrics]) -> ObserverMetrics:
    pooled = ObserverMetrics()
    causes: Counter[str] = Counter()
    for m in all_metrics:
        pooled.own_visible = pooled.own_visible.merged(m.own_visible)
        pooled.own_visible_store_wide = pooled.own_visible_store_wide.merged(
            m.own_visible_store_wide
        )
        pooled.vanished_cross_project = pooled.vanished_cross_project.merged(
            m.vanished_cross_project
        )
        pooled.winner_in_view = pooled.winner_in_view.merged(m.winner_in_view)
        pooled.leaked = pooled.leaked.merged(m.leaked)
        causes.update(m.vanished_by_cause)
        for kind, rate in m.by_kind.items():
            pooled.by_kind[kind] = pooled.by_kind.get(kind, Rate()).merged(rate)
    pooled.vanished_by_cause = dict(causes)
    return pooled
