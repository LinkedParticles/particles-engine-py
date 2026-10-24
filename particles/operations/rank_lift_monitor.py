# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The nightly rank-lift monitor, the Engine half.

The rank-lift ``λ`` was calibrated against one named belief, and that
belief left the rendered head three times with nobody noticing. This module
checks what the named belief stood for, every night, without anyone naming
anything: in each view a session renders, does the head hold the beliefs the
agent keeps applying *outside* it, and does it keep room for beliefs that have
not earned use yet?

The gather is here; the judgement is the pure
:func:`particles.core.scoring.utility.evaluate_rank_lift_monitor`:

- **Rows.** :func:`~particles.operations.utility_sweep.load_sweep_rows` once
  for every ACTIVE belief. Effective confidence and ``R`` are per-belief, so a
  project view's rows are a subset of the store-wide ones, and one observer
  scope join (:func:`~particles.store.observer_scope_join.load_scopes`) serves
  every view.
- **The off-head cohort.** Mined utility events within the half-life, against
  the session-start record: a session counts for a belief when it
  has a record and the record does not include the belief. A session with no
  record counts for nobody, since whether the head showed it is unknown.
- **Named targets.** ``utility.monitor.targets`` plus beliefs with an
  explicit usefulness gesture within the half-life. Reported,
  never judged.

Read-only: it writes nothing, calls no model, and never changes ``λ``.
The consolidation cycle records its report under the run
record's ``rank_lift`` key; the curation queue raises one card from that key
when the alert is set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import visible
from particles.core.schema import Particle
from particles.core.scoring.relevance import sweep_owner_rank_lift
from particles.core.scoring.utility import (
    OUT_OF_BAND,
    LiftRanker,
    MonitorHead,
    MonitorThresholds,
    RankLiftBand,
    SweepRow,
    evaluate_rank_lift_monitor,
    plan_alert,
    rank_lift_grid,
    target_transitions,
)
from particles.core.status import Status, StatusReason
from particles.operations.utility_sweep import (
    DEFAULT_GRID_MAX,
    DEFAULT_GRID_STEPS,
    load_sweep_rows,
)
from particles.store.event_store import OperatorEventType, list_events
from particles.store.observer_scope_join import lens_may_engage, load_scopes
from particles.store.particle_store import ParticleRow, get_particles_by_status
from particles.store.session_exposure_store import SessionExposureRow
from particles.store.utility_store import SOURCE_EXPLICIT, SOURCE_MINED, UtilityEventRow

#: The run record key the report is stored under.
PAYLOAD_KEY = "rank_lift"

#: The view label for the store-wide head.
STORE_WIDE = "store-wide"

#: The projection manifest's default ``top_k`` (the ``memory.yaml`` that
#: ``particles init claude-code`` writes). The Engine cannot read a manifest.
PROJECTION_TOP_K = 60

_IN_CHUNK = 500


class Band(BaseModel):
    """A grid band, as the run record stores it."""

    low: float | None = None
    high: float | None = None
    contiguous: bool = True

    @classmethod
    def of(cls, band: RankLiftBand) -> Band:
        return cls(low=band.low, high=band.high, contiguous=band.contiguous)


class HeadReading(BaseModel):
    """One view's reading at one head size."""

    head_size: int
    verdict: str
    cohort_size: int
    captured: int
    entry_share: float
    floor: Band
    ceiling: Band
    band: Band
    retained: int
    retention_size: int
    distinct_contents: int

    @classmethod
    def of(cls, head: MonitorHead) -> HeadReading:
        return cls(
            head_size=head.head_size,
            verdict=head.verdict,
            cohort_size=head.cohort_size,
            captured=head.captured,
            entry_share=head.entry_share,
            floor=Band.of(head.floor),
            ceiling=Band.of(head.ceiling),
            band=Band.of(head.band),
            retained=head.retained,
            retention_size=head.retention_size,
            distinct_contents=head.distinct_contents,
        )


class TargetReading(BaseModel):
    """A named target in one view: where it ranks, its use, the heads it is in."""

    particle_id: str
    rank: int = Field(description="1-based rank at the configured λ; 0 when not in this view.")
    reinforcement: float
    in_heads: list[int] = Field(default_factory=list)


class ViewReading(BaseModel):
    """The monitor's reading of one view: store-wide, or one project's."""

    view: str
    scored: int = 0
    heads: list[HeadReading] = Field(default_factory=list)
    targets: list[TargetReading] = Field(default_factory=list)
    carried: bool = Field(
        default=False,
        description="Not evaluated this run (the max_views rotation); heads are the last run's.",
    )


class CohortMember(BaseModel):
    """One off-head cohort belief, store-wide."""

    particle_id: str
    off_head_sessions: int
    reinforcement: float


class OwnerLensLine(BaseModel):
    """Whether the configured ``ω`` still sits in its band at the configured ``λ``."""

    rank_lift: float
    admissible: bool
    band: Band


class RankLiftMonitorReport(BaseModel):
    """What one run of the monitor found. Additive on the run record."""

    configured_rank_lift: float
    thresholds: dict[str, float | int]
    head_sizes: list[int]
    window_days: float
    cohort: list[CohortMember] = Field(default_factory=list)
    views: list[ViewReading] = Field(default_factory=list)
    skipped_views: list[str] = Field(default_factory=list)
    next_view_offset: int = 0
    transitions: list[dict[str, Any]] = Field(default_factory=list)
    record_coverage: float | None = Field(
        default=None,
        description="Share of the window's ACTIVE-belief events from sessions with a record.",
    )
    retained_credit: float | None = Field(
        default=None,
        description=(
            "Share of the window's events on a still-ACTIVE belief, among events "
            "not on a belief retracted at its source."
        ),
    )
    owner_lens: OwnerLensLine | None = None
    alert: str | None = None
    alert_new: bool = False
    baseline: bool = Field(
        default=False, description="No earlier record to compare against; nothing was raised."
    )

    def readings(self) -> dict[str, str]:
        """``"<view>@<N>"`` → verdict, the shape :func:`plan_alert` compares."""
        return {f"{v.view}@{h.head_size}": h.verdict for v in self.views for h in v.heads}

    def out_of_band(self) -> list[tuple[str, HeadReading]]:
        return [(v.view, h) for v in self.views for h in v.heads if h.verdict in OUT_OF_BAND]

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def thresholds_from_config() -> tuple[MonitorThresholds, int]:
    """The configured thresholds and the cohort's session minimum."""
    cfg = get_config().utility.monitor
    return (
        MonitorThresholds(
            capture=cfg.capture, min_cohort=cfg.min_cohort, entry_share=cfg.entry_share
        ),
        cfg.min_sessions,
    )


def resolve_head_sizes() -> list[int]:
    """``utility.monitor.head_sizes``, or the sizes this install renders."""
    cfg = get_config()
    if cfg.utility.monitor.head_sizes:
        return sorted({n for n in cfg.utility.monitor.head_sizes if n > 0})
    sizes = {PROJECTION_TOP_K, cfg.agent_memory.projection.max_lines}
    if cfg.mcp.recall.digest_max_beliefs > 0:
        sizes.add(cfg.mcp.recall.digest_max_beliefs)
    return sorted(sizes)


def _as_utc(when: datetime) -> datetime:
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


async def _mined_events_since(session: AsyncSession, since: datetime) -> list[tuple[str, str]]:
    rows = await session.execute(
        select(UtilityEventRow.particle_id, UtilityEventRow.session_id).where(
            UtilityEventRow.source == SOURCE_MINED, UtilityEventRow.observed_at >= since
        )
    )
    return [(pid, sid) for pid, sid in rows.all()]


async def _shown_by_session(session: AsyncSession, session_ids: set[str]) -> dict[str, set[str]]:
    """Every belief each session's start-context rows name, unioned."""
    shown: dict[str, set[str]] = {}
    ids = sorted(session_ids)
    for start in range(0, len(ids), _IN_CHUNK):
        rows = await session.execute(
            select(SessionExposureRow.session_id, SessionExposureRow.beliefs).where(
                SessionExposureRow.session_id.in_(ids[start : start + _IN_CHUNK])
            )
        )
        for sid, beliefs in rows.all():
            names = shown.setdefault(sid, set())
            for item in json.loads(beliefs or "[]"):
                pid = item.get("particle_id")
                if pid:
                    names.add(pid)
    return shown


def _was_shown(pid: str, shown: set[str]) -> bool:
    """Whether a session's start context named ``pid``, by full id or a display prefix."""
    if pid in shown:
        return True
    return any(len(s) < len(pid) and pid.startswith(s.removeprefix("p-")) for s in shown)


async def off_head_sessions(
    session: AsyncSession,
    active_ids: set[str],
    *,
    since: datetime,
) -> tuple[dict[str, int], float | None]:
    """Per ACTIVE belief, the sessions that applied it without being shown it at start.

    Also returns the share of the window's ACTIVE-belief events whose session has
    a start-context record (``None`` with no such events), so a thin cohort reads
    as thin rather than as a verdict.
    """
    events = [
        (pid, sid) for pid, sid in await _mined_events_since(session, since) if pid in active_ids
    ]
    if not events:
        return {}, None
    shown = await _shown_by_session(session, {sid for _pid, sid in events})
    counts: dict[str, set[str]] = defaultdict(set)
    recorded = 0
    for pid, sid in events:
        if sid not in shown:
            continue
        recorded += 1
        if not _was_shown(pid, shown[sid]):
            counts[pid].add(sid)
    return {pid: len(sids) for pid, sids in counts.items()}, recorded / len(events)


async def retained_credit(session: AsyncSession, since: datetime) -> float | None:
    """Share of the window's mined events on a still-ACTIVE belief (finding 1).

    Events on a belief retired because its source was retracted are left out of
    the denominator: that retraction was a judgement about the source, and the
    credit going with it is the intended outcome.
    """
    rows = await session.execute(
        select(ParticleRow.status, ParticleRow.status_reason)
        .join(UtilityEventRow, UtilityEventRow.particle_id == ParticleRow.id)
        .where(UtilityEventRow.source == SOURCE_MINED, UtilityEventRow.observed_at >= since)
    )
    active = counted = 0
    for status, reason in rows.all():
        if reason == StatusReason.SOURCE_RETRACTED.value:
            continue
        counted += 1
        if status == Status.ACTIVE.value:
            active += 1
    return active / counted if counted else None


async def explicit_targets(session: AsyncSession, since: datetime) -> set[str]:
    """Beliefs an operator marked useful within the window."""
    rows = await session.execute(
        select(UtilityEventRow.particle_id)
        .where(UtilityEventRow.source == SOURCE_EXPLICIT, UtilityEventRow.observed_at >= since)
        .distinct()
    )
    return {pid for (pid,) in rows.all()}


async def latest_monitor_block(session: AsyncSession) -> dict[str, Any] | None:
    """The most recent run record's ``rank_lift`` block, or ``None``."""
    events = await list_events(session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=100)
    for event in events:
        block = (event.payload or {}).get(PAYLOAD_KEY)
        if isinstance(block, dict):
            return block
    return None


async def live_project_keys(session: AsyncSession, since: datetime) -> set[str]:
    """Project keys a session read the store through within the window.

    The record names each session's project and whether its read went
    through that project's observer. A key no session read through is a view
    nobody sees (a removed worktree, say), so it is not judged.
    """
    rows = await session.execute(
        select(SessionExposureRow.project_key)
        .where(
            SessionExposureRow.recorded_at >= since,
            SessionExposureRow.observer_scope_applied.is_(True),
        )
        .distinct()
    )
    return {key for (key,) in rows.all() if key}


async def view_ids(
    session: AsyncSession, particles: Sequence[Particle], key: str
) -> frozenset[str] | None:
    """The ids in view for one project, or ``None`` when the store was never rescoped."""
    if not await lens_may_engage(session):
        return None
    scopes = await load_scopes(session, particles)
    return frozenset(pid for pid, s in scopes.items() if visible(s, key))


async def project_views(
    session: AsyncSession, particles: Sequence[Particle], *, since: datetime
) -> list[tuple[str, frozenset[str]]]:
    """Each live project key with the ids in view for it, largest view first.

    Empty when the store was never rescoped: every project read then stays
    store-wide, so there is no separate view to judge.
    """
    if not await lens_may_engage(session):
        return []
    keys = sorted(await live_project_keys(session, since))
    if not keys:
        return []
    scopes = await load_scopes(session, particles)
    views = [(key, frozenset(pid for pid, s in scopes.items() if visible(s, key))) for key in keys]
    views.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    return views


def _target_reading(
    rows: Sequence[SweepRow], pid: str, lam: float, head_sizes: Sequence[int]
) -> TargetReading:
    by_id = {r.particle_id: r for r in rows}
    if pid not in by_id:
        return TargetReading(particle_id=pid, rank=0, reinforcement=0.0)
    order = LiftRanker(rows).ordered(lam)
    rank = next(i for i, r in enumerate(order, start=1) if r.particle_id == pid)
    return TargetReading(
        particle_id=pid,
        rank=rank,
        reinforcement=round(by_id[pid].reinforcement, 4),
        in_heads=[n for n in head_sizes if rank <= n],
    )


def _cold_view(label: str, scored: int, cohort_size: int, head_sizes: Sequence[int]) -> ViewReading:
    """A view whose cohort is below ``min_cohort``: no verdict, and no ranking needed."""
    empty = Band()
    return ViewReading(
        view=label,
        scored=scored,
        heads=[
            HeadReading(
                head_size=n,
                verdict="cold",
                cohort_size=cohort_size,
                captured=0,
                entry_share=1.0,
                floor=empty,
                ceiling=empty,
                band=empty,
                retained=0,
                retention_size=0,
                distinct_contents=0,
            )
            for n in head_sizes
        ],
    )


def read_view(
    label: str,
    rows: Sequence[SweepRow],
    *,
    cohort_ids: set[str],
    head_sizes: Sequence[int],
    lam: float,
    thresholds: MonitorThresholds,
    targets: Sequence[str] = (),
) -> ViewReading:
    """Judge one view's rows (pure apart from the model construction)."""
    heads = evaluate_rank_lift_monitor(
        rows,
        cohort_ids=cohort_ids,
        grid=rank_lift_grid(DEFAULT_GRID_MAX, DEFAULT_GRID_STEPS),
        head_sizes=head_sizes,
        configured_rank_lift=lam,
        thresholds=thresholds,
    )
    return ViewReading(
        view=label,
        scored=len(rows),
        heads=[HeadReading.of(h) for h in heads],
        targets=[_target_reading(rows, pid, lam, head_sizes) for pid in targets],
    )


async def select_cohort(
    session: AsyncSession,
    particles: Sequence[Particle],
    rows: Sequence[SweepRow],
    *,
    min_sessions: int,
    now: datetime,
) -> tuple[list[CohortMember], float | None, datetime]:
    """The store-wide off-head cohort, the record coverage, and the window's start."""
    window = get_config().utility.default.half_life_uses_days
    since = now - timedelta(days=window)
    counts, coverage = await off_head_sessions(session, {p.id for p in particles}, since=since)
    by_id = {r.particle_id: r for r in rows}
    cohort = sorted(
        (
            CohortMember(
                particle_id=pid,
                off_head_sessions=n,
                reinforcement=round(by_id[pid].reinforcement, 4) if pid in by_id else 0.0,
            )
            for pid, n in counts.items()
            if n >= min_sessions
        ),
        key=lambda m: (-m.off_head_sessions, -m.reinforcement, m.particle_id),
    )
    return cohort, coverage, since


def _previous_state(
    previous: Mapping[str, Any] | None,
) -> tuple[dict[str, str] | None, dict[str, list[int]] | None]:
    """The last block's readings and store-wide target heads, read leniently.

    A block another version wrote may lack fields this one has; what can be
    read is compared and the rest is treated as absent, so an old record never
    fails the pass.
    """
    if previous is None:
        return None, None
    readings: dict[str, str] = {}
    targets: dict[str, list[int]] = {}
    for view in previous.get("views") or []:
        if not isinstance(view, dict):
            continue
        label = view.get("view")
        for head in view.get("heads") or []:
            if isinstance(head, dict) and "head_size" in head and "verdict" in head:
                readings[f"{label}@{head['head_size']}"] = str(head["verdict"])
        if label == STORE_WIDE:
            for target in view.get("targets") or []:
                if isinstance(target, dict) and "particle_id" in target:
                    targets[str(target["particle_id"])] = list(target.get("in_heads") or [])
    return readings, targets


async def run_rank_lift_monitor(
    session: AsyncSession,
    *,
    run_id: str,
    previous: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    thresholds: MonitorThresholds | None = None,
    min_sessions: int | None = None,
    head_sizes: Sequence[int] | None = None,
    only: str | None | Literal[False] = False,
) -> RankLiftMonitorReport:
    """One night's reading of every view. Read-only.

    ``previous`` is the last run record's ``rank_lift`` block (``None`` before
    the first), against which the alert and the target transitions are
    computed and from which a view this run's rotation skipped is carried.
    ``run_id`` names a new alert.

    The keyword overrides serve ``particles memory sweep-rank-lift
    --off-head-cohort``, which reproduces a verdict by hand: thresholds and head
    sizes default to config, and ``only`` reads one view instead of every live
    one (``None`` for store-wide, or a project key).
    """
    cfg = get_config()
    monitor = cfg.utility.monitor
    lam = cfg.utility.default.rank_lift
    configured, configured_sessions = thresholds_from_config()
    thresholds = thresholds or configured
    min_sessions = min_sessions or configured_sessions
    head_sizes = sorted({n for n in head_sizes if n > 0}) if head_sizes else resolve_head_sizes()
    moment = _as_utc(now or datetime.now(UTC))

    particles = await get_particles_by_status(session, Status.ACTIVE)
    rows = await load_sweep_rows(session, particles)
    cohort, coverage, since = await select_cohort(
        session, particles, rows, min_sessions=min_sessions, now=moment
    )
    cohort_ids = {m.particle_id for m in cohort}
    targets = sorted(set(monitor.targets) | await explicit_targets(session, since))

    report = RankLiftMonitorReport(
        configured_rank_lift=lam,
        thresholds={
            "capture": thresholds.capture,
            "min_cohort": thresholds.min_cohort,
            "entry_share": thresholds.entry_share,
            "min_sessions": min_sessions,
        },
        head_sizes=head_sizes,
        window_days=cfg.utility.default.half_life_uses_days,
        cohort=cohort,
        record_coverage=round(coverage, 4) if coverage is not None else None,
    )
    credit = await retained_credit(session, since)
    report.retained_credit = round(credit, 4) if credit is not None else None

    def judge(label: str, view_rows: Sequence[SweepRow]) -> ViewReading:
        return read_view(
            label,
            view_rows,
            cohort_ids=cohort_ids,
            head_sizes=head_sizes,
            lam=lam,
            thresholds=thresholds,
            targets=targets,
        )

    if only is not False:
        if only is None:
            report.views.append(judge(STORE_WIDE, rows))
        else:
            ids = await view_ids(session, particles, only)
            report.views.append(
                judge(only, [r for r in rows if ids is None or r.particle_id in ids])
            )
        report.baseline = True
        return report

    report.views.append(judge(STORE_WIDE, rows))

    projects = await project_views(session, particles, since=since)
    slots = max(0, monitor.max_views - 1)
    offset = int((previous or {}).get("next_view_offset", 0)) if projects else 0
    offset = offset % len(projects) if projects else 0
    chosen = [projects[(offset + i) % len(projects)] for i in range(min(slots, len(projects)))]
    chosen_keys = {key for key, _ids in chosen}
    report.next_view_offset = (offset + len(chosen)) % len(projects) if projects else 0
    previous_views = {v.get("view"): v for v in (previous or {}).get("views", [])}
    for key, ids in projects:
        if key in chosen_keys:
            in_view = len(cohort_ids & ids)
            if in_view < thresholds.min_cohort:
                # Cold without ranking the view: the cohort alone decides it.
                report.views.append(_cold_view(key, len(ids), in_view, head_sizes))
            else:
                report.views.append(judge(key, [r for r in rows if r.particle_id in ids]))
            continue
        report.skipped_views.append(key)
        carried = previous_views.get(key)
        if carried is not None:
            try:
                view = ViewReading.model_validate(carried)
            except ValidationError:
                continue  # an older record's shape: nothing to carry
            view.carried = True
            report.views.append(view)

    owner = cfg.owner_lens
    if owner.enabled and owner.rank_lift > 0.0 and any(r.owner_relevant for r in rows):
        sweep = sweep_owner_rank_lift(
            rows,
            grid=rank_lift_grid(DEFAULT_GRID_MAX, DEFAULT_GRID_STEPS),
            head_sizes=head_sizes,
            lambda_=lam,
            configured_rank_lift=owner.rank_lift,
        )
        report.owner_lens = OwnerLensLine(
            rank_lift=owner.rank_lift,
            admissible=sweep.configured_admissible,
            band=Band.of(sweep.intersection),
        )

    previous_readings, previous_targets = _previous_state(previous)
    report.baseline = previous is None
    report.alert = plan_alert(
        report.readings(),
        previous_readings,
        (previous or {}).get("alert"),
        run_id,
    )
    report.alert_new = report.alert is not None and report.alert == run_id
    store_wide = report.views[0]
    current_targets = {t.particle_id: t.in_heads for t in store_wide.targets}
    report.transitions = [
        {"target": pid, "view": STORE_WIDE, "head_size": n, "change": change}
        for pid, n, change in target_transitions(current_targets, previous_targets, head_sizes)
    ]
    return report


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def reproduce_command(view: str, report: RankLiftMonitorReport) -> str:
    """The ``sweep-rank-lift`` invocation that reproduces one view's verdict."""
    t = report.thresholds
    parts = ["particles memory sweep-rank-lift --off-head-cohort"]
    if view != STORE_WIDE:
        parts.append(f"--project {view}")
    parts.extend(f"--head {n}" for n in report.head_sizes)
    parts.append(f"--min-sessions {t['min_sessions']}")
    parts.append(f"--capture {t['capture']}")
    parts.append(f"--entry-share {t['entry_share']}")
    return " ".join(parts)


def render_rank_lift_lines(report: RankLiftMonitorReport) -> list[str]:
    """The consolidation report's rank-lift line, and the reproducing command when out of band."""
    lam = report.configured_rank_lift
    out = report.out_of_band()
    evaluated = [v for v in report.views if not v.carried]
    cold = [v.view for v in evaluated if v.heads and all(h.verdict == "cold" for h in v.heads)]
    if not out:
        judged = [v for v in evaluated if v.view not in cold]
        if not judged:
            line = (
                f"  rank-lift        λ {lam} not judged: too few beliefs used outside "
                f"the head (cold in {len(cold)} view(s))"
            )
        else:
            line = f"  rank-lift        λ {lam} in band in {len(judged)} view(s)"
            if cold:
                line += f", {len(cold)} cold"
        return [line]
    view, head = out[0]
    views_out = list(dict.fromkeys(v for v, _h in out))
    what = {
        "below_floor": f"below the use-capture floor {_fmt(head.floor.low)}",
        "above_ceiling": f"above the entry-share ceiling {_fmt(head.ceiling.high)}",
        "no_band": "with no admissible band",
    }[head.verdict]
    line = (
        f"  rank-lift        λ {lam} {what} ({head.captured} of {head.cohort_size} "
        f"off-head-used beliefs in the {head.head_size}-head; "
        f"{', '.join(views_out)})"
    )
    if report.alert_new:
        line += " [new]"
    return [line, f"                   reproduce: {reproduce_command(view, report)}"]


def render_rank_lift_monitor(report: RankLiftMonitorReport) -> str:
    """The full Markdown reading, for ``sweep-rank-lift --off-head-cohort``."""
    t = report.thresholds
    lines = [
        f"# Rank-lift monitor: λ = {report.configured_rank_lift}",
        "",
        "Read-only. The floor: the head holds at least "
        f"{t['capture']} of the beliefs applied outside it in at least "
        f"{t['min_sessions']} sessions within {report.window_days:g} days. The ceiling: at "
        f"least {t['entry_share']} of head slots hold beliefs with no use yet. A view "
        f"with fewer than {t['min_cohort']} cohort beliefs is cold.",
        "",
        f"Cohort (store-wide): {len(report.cohort)} belief(s); record coverage "
        f"{_fmt(report.record_coverage)}; retained use credit {_fmt(report.retained_credit)}.",
        "",
        "| view | N | verdict | cohort in head | R = 0 share | floor | ceiling | band |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for view in report.views:
        for h in view.heads:
            band = "—" if h.band.low is None else f"{h.band.low:.3f} – {h.band.high:.3f}"
            lines.append(
                f"| {view.view}{' (carried)' if view.carried else ''} | {h.head_size} "
                f"| {h.verdict} | {h.captured} of {h.cohort_size} | {h.entry_share:.2f} "
                f"| {_fmt(h.floor.low)} | {_fmt(h.ceiling.high)} | {band} |"
            )
    targets = report.views[0].targets if report.views else []
    if targets:
        lines.extend(["", "## Named targets (disclosed, never judged)", ""])
        for tr in targets:
            note = " (earning no use)" if tr.reinforcement <= 0.0 else ""
            rank = tr.rank or "not in view"
            lines.append(f"- `{tr.particle_id[:8]}`: rank {rank}, R {tr.reinforcement}{note}")
    if report.owner_lens is not None:
        o = report.owner_lens
        state = "inside" if o.admissible else "OUTSIDE"
        lines.extend(
            [
                "",
                f"Owner lens: ω = {o.rank_lift} is {state} its band "
                f"{_fmt(o.band.low)} – {_fmt(o.band.high)} at this λ.",
            ]
        )
    return "\n".join(lines) + "\n"
