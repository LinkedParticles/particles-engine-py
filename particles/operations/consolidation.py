"""Dream cycle — the scheduled consolidation operation.

``run_consolidation`` composes the **existing** engine passes in the fixed §3
order and adds no detection of its own: extract catch-up (capped), the reconcile sweep, one ``collect_cards(semantic=…)`` census pass under
the probe cap + the §4 delta scope, the curation-queue
refresh over the *same* card collection, the utility-mining pass, and
the projection re-render (injected by the Surface caller — the
render-splice tail is CLI-side, so the Engine never imports upward).

Each completed run writes one ``CONSOLIDATION_RUN`` operator event (
§7 — the fold): a versioned payload carrying per-pass status /
durations / LLM call counts, the machine-readable census, degradation
disclosures, and ``started_at`` — the instant the next run's delta window
opens from (correction, v1.74.1: the window opens at the previous
run's *start*, not its completion, so particles minted mid-cycle — by pass 1
itself or by an interleaving harvest — are re-probed by the next run rather
than falling into a permanent gap; overlap re-probes are idempotent and
cheap). Only a successful, non-degraded run by this verb's own actor is
watermark-eligible. ``particles audit`` records the same event shape via
:func:`record_audit_run` (``actor: audit``) for the §7 delta report, but an
audit event neither advances the watermark nor satisfies ``--if-due``.

The cron contract (§8): one cycle at a time (the ``consolidate.lock`` file in
the integration state directory, stale-reclaimed), ``--if-due`` cadence guard,
and continue-and-report on pass failure — a flaky night never leaves the
zero-LLM passes unrun or the run record unwritten.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import SuggestMode, WarcRecordType
from particles.core.status import Status
from particles.llm.errors import AccountLevelLLMError
from particles.operations._llm import llm_circuit_open
from particles.operations.abstraction import AbstractionReport, run_abstraction_pass
from particles.operations.curation.cards import CardKind, CurationCard
from particles.operations.curation.collect import collect_cards
from particles.operations.curation.session import _suppressed_keys, build_curation_queue
from particles.operations.curation.snapshot import collect_and_persist
from particles.operations.lint import ContradictionProbeControl
from particles.operations.reconcile import reconcile_supersession
from particles.operations.utility_mining import mine_session, session_id_from_uri
from particles.store.curation_snapshot_store import CollectionScope
from particles.store.event_store import (
    OperatorEvent,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    get_particle_ids_changed_since,
    get_particle_ids_for_entries,
    get_particles_by_status,
)

if TYPE_CHECKING:
    from particles.operations.audit import AuditReport

log = logging.getLogger(__name__)

#: Version stamp of the ``CONSOLIDATION_RUN`` event payload.
RUN_PAYLOAD_FORMAT = 1

#: The LLM purposes the cycle's passes route through (§5: existing purposes,
#: no new "consolidation" purpose) — recorded on the run record so a routing
#: change is visible in the audit trail.
_RUN_PURPOSES: tuple[str, ...] = ("extraction", "semantic_lint", "abstraction")

ProjectionRunner = Callable[[], Awaitable[dict[str, Any]]]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(dt: datetime) -> datetime:
    """Normalize a possibly-naive stored datetime to UTC for comparisons."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# The cycle lockfile — one cycle at a time, stale-reclaimed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleLock:
    """A held cycle lock; release with :func:`release_cycle_lock`."""

    path: Path


def cycle_lock_path() -> Path:
    """``consolidate.lock`` in the integration state directory."""
    return Path(get_config().claude_code.state_dir).expanduser() / "consolidate.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _lock_is_stale(path: Path, timeout_minutes: int) -> bool:
    """True when the held lock's pid is dead or its age exceeds the timeout.

    An unreadable / malformed lock is stale by definition — a crashed run must
    never wedge the cadence forever.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    pid = data.get("pid")
    if isinstance(pid, int) and not _pid_alive(pid):
        return True
    try:
        started = _as_utc(datetime.fromisoformat(str(data.get("started_at"))))
    except (TypeError, ValueError):
        return True
    return _utcnow() - started > timedelta(minutes=timeout_minutes)


def _reclaim_stale_lock(path: Path, timeout_minutes: int) -> bool:
    """Unlink ``path`` only if it is stale AND unchanged since it was judged.

    Closes the reclaim TOCTOU in the protocol: two contenders can
    both judge the same old lock stale; without the re-verify, the slower one
    would unlink the faster one's FRESH lock. The lock's identity (inode +
    mtime) is recorded before the staleness judgement and re-checked just
    before the unlink — if it changed, another contender already reclaimed
    and re-created the lock, so this contender backs off (returns ``False``).
    Duplicate-run prevention stays best-effort, but a live contender's lock
    is never deleted on the strength of a pre-race judgement.

    Returns ``True`` when the caller may retry the exclusive create (the
    stale lock was removed, or vanished on its own).
    """
    try:
        before = path.stat()
    except OSError:
        return True  # gone already — the exclusive-create retry decides
    if not _lock_is_stale(path, timeout_minutes):
        return False
    try:
        after = path.stat()
    except OSError:
        return True
    if (after.st_ino, after.st_mtime_ns) != (before.st_ino, before.st_mtime_ns):
        # Replaced mid-judgement: a rival contender reclaimed first and holds
        # a fresh lock. Do NOT unlink it — lost the race.
        return False
    log.warning("Reclaiming stale consolidation lock at %s", path)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return True


def acquire_cycle_lock(path: Path, *, timeout_minutes: int) -> CycleLock | None:
    """Atomically acquire the cycle lock, reclaiming a stale one.

    Returns ``None`` when a live cycle holds the lock — the caller exits 0
    with "already running — skipped" (contention is normal under cron, not an
    alarm). Acquisition is ``O_CREAT | O_EXCL`` (atomic); a stale-lock reclaim
    re-verifies the lock's identity before unlinking (:func:`_reclaim_stale_lock`)
    and treats a failed re-create as a lost race.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "started_at": _utcnow().isoformat()})
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt > 0 or not _reclaim_stale_lock(path, timeout_minutes):
                return None
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        return CycleLock(path)
    return None


def release_cycle_lock(lock: CycleLock) -> None:
    """Release a held cycle lock (idempotent)."""
    with contextlib.suppress(FileNotFoundError):
        lock.path.unlink()


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class ConsolidationPass(BaseModel):
    """One §3 pass: what ran, for how long, at what LLM spend."""

    name: str
    status: Literal["ran", "skipped", "failed"]
    #: Skip reason or error text; None for a clean "ran".
    detail: str | None = None
    duration_seconds: float = 0.0
    #: The §11 spend side. Extraction counts extracted snapshots (a lower
    #: bound: a chunked source makes one call per chunk); the census counts
    #: probes run; reconcile counts replacement-signal probes; utility counts
    #: behavioural matcher calls.
    llm_calls: int = 0

    def payload_status(self) -> str:
        """The run-record status string: ``ran | skipped(<r>) | failed(<e>)``."""
        if self.status == "ran":
            return "ran"
        return f"{self.status}({self.detail or 'unknown'})"


class ConsolidationReport(BaseModel):
    """The output of :func:`run_consolidation`."""

    store: str = "default"
    actor: str = "memory-consolidate"
    outcome: Literal["ran", "skipped"] = "ran"
    #: Set when ``outcome == "skipped"`` (lock held / --if-due not due).
    skip_reason: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    # §4 scope. ``scope`` records what was asked for; ``effective_scope`` what
    # actually ran ("store" on a first run with no watermark). ``watermark``
    # is the instant the delta scope was computed FROM: the previous
    # watermark-eligible run's ``started_at`` (correction v1.74.1 — see
    # :func:`latest_run_event`).
    scope: Literal["delta", "store"] = "delta"
    effective_scope: Literal["delta", "store"] = "delta"
    watermark: datetime | None = None
    scope_particle_count: int | None = None

    passes: list[ConsolidationPass] = Field(default_factory=list)

    # §6 degradation disclosure — mirrors ``LintReport.semantic_skipped``.
    semantic_degraded: bool = False
    semantic_degraded_reason: str | None = None
    #: provider:model per purpose, so a routing change is visible (§7).
    providers: dict[str, str] = Field(default_factory=dict)

    # Pass 0.5 — local-source refresh. Zero-LLM, so it runs on
    # degraded nights too. ``refresh_unchanged_mtime`` is the no-I/O tier-1
    # short-circuit; ``refresh_unchanged_hash`` read the bytes and matched.
    refresh_checked: int = 0
    refresh_unchanged_mtime: int = 0
    refresh_unchanged_hash: int = 0
    refresh_updated: int = 0
    refresh_missing: int = 0
    refresh_remaining: int = 0

    # Pass 1 — extract catch-up.
    pending_total: int = 0
    pending_extracted: int = 0
    pending_failed: int = 0
    pending_remaining: int = 0

    # Pass 2 — reconcile sweep (probe-bearing: one semantic_lint call per
    # candidate pair, capped at consolidation.max_reconcile_probes).
    reconcile_demoted: int = 0
    reconcile_candidate_pairs: int = 0
    reconcile_probes_run: int = 0

    # Pass 3 — census (the machine-readable fields).
    card_counts: dict[str, int] = Field(default_factory=dict)
    # the CONTESTED class basis, so a delta of "+2
    # contested" is attributable to the instrument that produced it rather
    # than being one unattributed total. A card firing two bases counts under
    # both. Additive on the versioned run-record payload — no format bump.
    contested_bases: dict[str, int] = Field(default_factory=dict)
    contradiction_candidate_pairs: int = 0
    contradiction_intra_scope_pairs: int = 0
    contradiction_probes_run: int = 0
    duplicate_candidate_pairs_total: int = 0
    duplicate_in_scope: int = 0

    # Pass 4 — curation-queue refresh. Since the card collection is
    # **persisted**, so this pass is what makes the morning's `GET /curation`
    # fast rather than a report line that is thrown away. The rendered lines
    # below stay in the run record; the cards live in `curation_snapshots`.
    curation_queue: list[str] = Field(default_factory=list)
    curation_queue_total: int = 0
    # the collection this run wrote. A pointer, not the blob —
    # the audit trail says which snapshot the night produced without carrying
    # megabytes of derived cards in an append-only event.
    curation_snapshot_id: str | None = None

    # Pass 5 — utility mining. ``utility_behavioural_exhausted_after`` is the
    # session count at which the shared per-run behavioural budget ran out
    # (None = never) — the truncation disclosure's numerator.
    utility_literal: int = 0
    utility_behavioural: int = 0
    utility_behavioural_calls: int = 0
    utility_sessions_mined: int = 0
    utility_behavioural_exhausted_after: int | None = None

    # Pass 5b — abstraction promotion: the pass's own sub-report
    # (clusters, promotions/proposals, the §5 revalidation ladder outcomes).
    abstraction: AbstractionReport | None = None

    # Pass 6 — projection re-render (the runner's telemetry dict).
    projection: dict[str, Any] | None = None

    # §7 delta report against the most recent prior CONSOLIDATION_RUN.
    previous_run_at: datetime | None = None
    deltas: dict[str, int] = Field(default_factory=dict)

    #: Event id of the written run record.
    event_id: str | None = None

    def count(self, kind: CardKind) -> int:
        """The census count for one card class (0 when absent)."""
        return self.card_counts.get(kind.value, 0)

    @property
    def headline_contradictions(self) -> int:
        return self.count(CardKind.CONTRADICTION) + self.count(CardKind.CONTESTED)

    @property
    def headline_stale(self) -> int:
        return (
            self.count(CardKind.STALE)
            + self.count(CardKind.RECENCY_DECAY)
            + self.count(CardKind.CONFIDENCE_DECAY)
        )

    def failed_passes(self) -> list[str]:
        """Names of the passes that failed (drives the CLI's exit 1)."""
        return [p.name for p in self.passes if p.status == "failed"]


# ---------------------------------------------------------------------------
# Run-record helpers
# ---------------------------------------------------------------------------


def _event_completed_at(event: OperatorEvent) -> datetime | None:
    """The run's ``completed_at`` (cadence-age basis), falling back to ``occurred_at``."""
    raw = (event.payload or {}).get("completed_at")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return _as_utc(datetime.fromisoformat(raw))
    return _as_utc(event.occurred_at)


def _event_started_at(event: OperatorEvent) -> datetime:
    """The run's ``started_at`` — the §4 delta-watermark basis (correction v1.74.1).

    The next run's delta window opens here, not at ``completed_at``: particles
    minted while the run was executing (pass 1's own output, an interleaving
    SessionEnd harvest) must land in *some* run's scope. Overlap re-probes are
    idempotent and cheap; permanent gaps are not. Falls back to ``occurred_at``
    (conservative — earlier than any payload timestamp is impossible, and the
    event row is written at completion).
    """
    raw = (event.payload or {}).get("started_at")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return _as_utc(datetime.fromisoformat(raw))
    return _as_utc(event.occurred_at)


def _event_successful(event: OperatorEvent) -> bool:
    """A run record with no ``failed(...)`` pass counts as successful."""
    passes = (event.payload or {}).get("passes")
    if not isinstance(passes, list):
        return True
    return not any(
        isinstance(p, dict) and str(p.get("status", "")).startswith("failed") for p in passes
    )


def _event_degraded(event: OperatorEvent) -> bool:
    """True when the run record disclosed a degraded (structural-only) run."""
    return bool((event.payload or {}).get("semantic_degraded"))


async def latest_run_event(
    session: AsyncSession,
    *,
    actor: str | None = None,
    successful_only: bool = False,
    exclude_degraded: bool = False,
) -> OperatorEvent | None:
    """The most recent ``CONSOLIDATION_RUN`` event passing the given filters.

    Correction (v1.74.1) — eligibility is filtered, not blanket:

    - The ``--if-due`` guard (§2) reads the last *successful* run **by this
      verb's own actor** — an interactive ``particles audit`` writes the same
      event type but runs none of the cross-session passes, so it must not
      satisfy the cadence. A *degraded* consolidation run still satisfies
      cadence — deliberate, so a key-less setup does not hot-loop.
    - The §4 watermark additionally requires ``exclude_degraded``: a
      structural-only night that advanced the watermark would silently convert
      its disclosed "not probed this run" into "never probed".
    - The delta report (§7) compares against the most recent prior run with
      no filters at all.
    """
    events = await list_events(session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=100)
    for event in events:
        if actor is not None and event.actor != actor:
            continue
        if successful_only and not _event_successful(event):
            continue
        if exclude_degraded and _event_degraded(event):
            continue
        return event
    return None


def build_run_payload(
    *,
    store: str,
    actor: str,
    scope: str,
    watermark: datetime | None,
    started_at: datetime,
    completed_at: datetime,
    semantic_degraded: bool,
    semantic_degraded_reason: str | None,
    providers: dict[str, str],
    passes: list[ConsolidationPass],
    census: dict[str, Any],
) -> dict[str, Any]:
    """The versioned ``CONSOLIDATION_RUN`` payload (``format: 1``).

    Shared verbatim by the scheduled cycle and the interactive audit's
    recording (:func:`record_audit_run`) so the delta chain reads one shape.
    """
    return {
        "format": RUN_PAYLOAD_FORMAT,
        "store": store,
        "actor": actor,
        "scope": scope,
        "watermark": watermark.isoformat() if watermark is not None else None,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "semantic_degraded": semantic_degraded,
        "semantic_degraded_reason": semantic_degraded_reason,
        "providers": providers,
        "passes": [
            {
                "name": p.name,
                "status": p.payload_status(),
                "duration_seconds": round(p.duration_seconds, 3),
                "llm_calls": p.llm_calls,
            }
            for p in passes
        ],
        "census": census,
    }


def _current_providers() -> dict[str, str]:
    """provider:model per cycle purpose — a config read, no client construction."""
    llm_cfg = get_config().llm
    out: dict[str, str] = {}
    for purpose in _RUN_PURPOSES:
        selection = llm_cfg.for_purpose(purpose)
        out[purpose] = f"{selection.provider}:{selection.model}"
    return out


def _semantic_availability(structural_only: bool) -> tuple[bool, str | None]:
    """(available, degradation reason) for the cycle's LLM passes (§6).

    Degraded when the operator asked (``--structural-only``), when
    ``consolidation.semantic`` is off (the §11 demotion switch), when the breaker is open, or when an Anthropic-routed purpose has no key.
    A purpose routed to the local provider needs no Anthropic key.
    """
    if structural_only:
        return False, "--structural-only"
    cfg = get_config()
    if not cfg.consolidation.semantic:
        return False, "consolidation.semantic is false"
    if llm_circuit_open():
        return False, "LLM unavailable (circuit breaker open)"
    needs_key = any(cfg.llm.for_purpose(p).provider == "anthropic" for p in _RUN_PURPOSES)
    if needs_key:
        from particles.secrets import get_anthropic_api_key_optional

        if get_anthropic_api_key_optional() is None:
            return False, "no API key (structural-only run)"
    return True, None


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


async def run_consolidation(
    session: AsyncSession,
    *,
    store: str = "default",
    scope: Literal["delta", "store"] = "delta",
    structural_only: bool = False,
    if_due: bool = False,
    actor: str = "memory-consolidate",
    projection_runner: ProjectionRunner | None = None,
    projection_skip_reason: str | None = None,
) -> ConsolidationReport:
    """Run the fixed consolidation pass list and write the run record.

    Composes existing operations only; the ordering is
    load-bearing — reconcile before census so the census sees the demoted
    state, everything before the projection render so ``MEMORY.md`` reflects
    this run's work. Pass failures are caught, recorded, and the cycle
    continues (§8); the run record is written even for a partially-failed run.
    Commits are per-pass (a later failure must not roll back completed work);
    the final run-record commit happens here too, so a cron run persists its
    record regardless of the caller.

    ``projection_runner`` is the Surface-injected harvest-then-render tail
    (the SessionEnd cycle's, reused) — the Engine cannot import the CLI-side
    projection helpers without inverting the layer contract. ``None`` records
    pass 6 as skipped with ``projection_skip_reason``.
    """
    cfg = get_config().consolidation
    report = ConsolidationReport(store=store, actor=actor, scope=scope, effective_scope=scope)

    # ---------------------------------------------------------- pass 0: gate
    # §2 cadence — only this verb's own successful runs count (an interactive
    # audit writes the same event type but runs none of the cross-session
    # passes). A degraded run still satisfies cadence: deliberate, so a
    # key-less setup retries next interval instead of hot-looping.
    if if_due:
        last_for_cadence = await latest_run_event(session, actor=actor, successful_only=True)
        last_completed = (
            _event_completed_at(last_for_cadence) if last_for_cadence is not None else None
        )
        if last_completed is not None:
            age = _utcnow() - last_completed
            if age < timedelta(hours=cfg.min_interval_hours):
                report.outcome = "skipped"
                report.skip_reason = (
                    f"not due: last successful run {last_completed:%Y-%m-%d %H:%M} UTC is "
                    f"younger than consolidation.min_interval_hours ({cfg.min_interval_hours}h)"
                )
                return report

    lock = acquire_cycle_lock(cycle_lock_path(), timeout_minutes=cfg.lock_timeout_minutes)
    if lock is None:
        report.outcome = "skipped"
        report.skip_reason = "consolidation already running — skipped"
        return report

    try:
        semantic_ok, degrade_reason = _semantic_availability(structural_only)
        report.semantic_degraded = not semantic_ok
        report.semantic_degraded_reason = degrade_reason
        report.providers = _current_providers()

        # §4 watermark basis (correction v1.74.1): the previous
        # watermark-eligible run — same actor, successful, NOT degraded (a
        # structural-only night must not convert its disclosed "not probed
        # this run" into "never probed") — and its *started_at*, so nothing
        # written mid-cycle ever falls between two runs' windows.
        watermark_event = await latest_run_event(
            session, actor=actor, successful_only=True, exclude_degraded=True
        )
        prior_started = _event_started_at(watermark_event) if watermark_event is not None else None

        # ---------------------------------------- pass 0.5: local refresh
        # Placed BEFORE extract so a rule file edited today is
        # re-snapshotted, extracted by pass 1, and reconciled by pass 2 in the
        # SAME run — anywhere later and a change would take three nights to
        # reach the projection. Zero-LLM, so it is not gated on ``semantic_ok``:
        # a degraded night still notices that the rules changed.
        if not get_config().local_refresh.enabled:
            _skip(report, "refresh", "local_refresh.enabled is false")
        else:
            await _run_pass(session, report, "refresh", lambda: _pass_refresh(session, report))

        # ------------------------------------------------ pass 1: extract
        if not get_config().consolidation.extract_pending:
            _skip(report, "extract", "consolidation.extract_pending is false")
        elif not semantic_ok:
            # §6: extraction is LLM-priced; the backlog is disclosed instead.
            _skip(report, "extract", f"extraction is LLM-priced ({degrade_reason})")
            with contextlib.suppress(Exception):
                await _count_backlog(session, report)
        else:
            await _run_pass(session, report, "extract", lambda: _pass_extract(session, report))

        # §4 delta scope — computed AFTER pass 1 (correction v1.74.1), so the
        # particles pass 1 just minted (asserted_at > the previous run's
        # started_at) are in THIS run's census scope rather than in no run's
        # scope ever. Particles from corpus entries deposited since the
        # watermark are folded in. Store-wide on the first run or --scope
        # store.
        scope_ids: frozenset[str] | None = None
        if scope == "delta":
            if prior_started is None:
                report.effective_scope = "store"  # first run: nothing to delta against
            else:
                report.watermark = prior_started
                scope_ids = await _delta_scope_ids(session, prior_started)
                report.scope_particle_count = len(scope_ids)

        # ---------------------------------------------- pass 2: reconcile
        if not semantic_ok:
            # §6 (correction v1.74.1): the sweep makes one replacement-signal
            # probe per candidate pair — LLM-priced, so a degraded run skips
            # it with a disclosure rather than fail-opening every probe to
            # "keep both" behind a clean-looking bill.
            _skip(
                report, "reconcile", f"replacement-signal probes are LLM-priced ({degrade_reason})"
            )
        else:
            await _run_pass(session, report, "reconcile", lambda: _pass_reconcile(session, report))

        # ------------------------------------------------- pass 3: census
        cards: list[CurationCard] = []

        async def _census() -> int:
            nonlocal cards
            cards = await _pass_census(session, report, semantic=semantic_ok, scope_ids=scope_ids)
            return report.contradiction_probes_run

        await _run_pass(session, report, "census", _census)

        # ----------------------------------------- pass 4: curation queue
        if report.passes[-1].status == "failed":
            _skip(report, "curation", "census failed — no card collection to rank")
        else:
            # the snapshot records the scope its finders ran under,
            # so tonight's delta-scoped contradiction set unions with (rather
            # than erases) the store-wide picture earlier runs built.
            collection_scope = CollectionScope.STORE if scope_ids is None else CollectionScope.DELTA
            await _run_pass(
                session,
                report,
                "curation",
                lambda: _pass_curation(
                    session, report, cards, scope=collection_scope, semantic=semantic_ok
                ),
            )

        # ------------------------------------------------ pass 5: utility
        if not get_config().utility.mining.enabled:
            _skip(report, "utility", "utility.mining.enabled is false")
        else:
            await _run_pass(
                session,
                report,
                "utility",
                lambda: _pass_utility(
                    session,
                    report,
                    watermark=report.watermark,
                    behavioural=semantic_ok,
                ),
            )

        # ------------------------------------- pass 5b: abstraction
        # Between utility mining (whose signals inform future eligibility) and
        # the projection render (so the projection reflects this run's
        # promotions), ahead of the reserved retention slot.
        if not get_config().consolidation.abstraction.enabled:
            _skip(report, "abstraction", "consolidation.abstraction.enabled is false")
        elif not semantic_ok:
            _skip(
                report,
                "abstraction",
                f"synthesis + judges are LLM-priced ({degrade_reason})",
            )
        else:
            await _run_pass(
                session,
                report,
                "abstraction",
                lambda: _pass_abstraction(session, report, scope_ids=scope_ids),
            )

        # --------------------------------------------- pass 6: projection
        if projection_runner is None:
            _skip(
                report,
                "projection",
                projection_skip_reason or "projection disabled or no memory directory",
            )
        else:
            runner: ProjectionRunner = projection_runner
            await _run_pass(session, report, "projection", lambda: _pass_projection(report, runner))

        # ---------------------------------------- pass 7: record + report
        report.completed_at = _utcnow()
        prior = await latest_run_event(session)
        if prior is not None:
            report.previous_run_at = _event_completed_at(prior)
            report.deltas = _compute_deltas(report, prior)
        payload = build_run_payload(
            store=store,
            actor=actor,
            scope=report.effective_scope,
            watermark=report.watermark,
            started_at=report.started_at,
            completed_at=report.completed_at,
            semantic_degraded=report.semantic_degraded,
            semantic_degraded_reason=report.semantic_degraded_reason,
            providers=report.providers,
            passes=report.passes,
            census=_census_payload(report),
        )
        event = await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.CONSOLIDATION_RUN,
            payload=payload,
        )
        await session.commit()
        report.event_id = event.event_id
    finally:
        release_cycle_lock(lock)
    return report


# ---------------------------------------------------------------------------
# Pass bodies — each an existing verb's operation, composed (§3)
# ---------------------------------------------------------------------------


def _skip(report: ConsolidationReport, name: str, reason: str) -> None:
    report.passes.append(ConsolidationPass(name=name, status="skipped", detail=reason))


async def _run_pass(
    session: AsyncSession,
    report: ConsolidationReport,
    name: str,
    body: Callable[[], Awaitable[int]],
) -> None:
    """Run one pass under the §8 continue-and-report contract.

    ``body`` returns the pass's LLM call count. A failure is caught, the
    session rolled back, and the pass recorded as ``failed(<error>)`` — the
    cycle continues, so the zero-LLM tail (projection) still runs on a flaky
    night and the run record is written regardless.
    """
    start = time.monotonic()
    entry = ConsolidationPass(name=name, status="ran")
    report.passes.append(entry)
    try:
        entry.llm_calls = await body()
    except Exception as exc:  # noqa: BLE001 — §8: continue and report
        with contextlib.suppress(Exception):
            await session.rollback()
        entry.status = "failed"
        entry.detail = f"{type(exc).__name__}: {exc}"
        log.warning("consolidation: pass %s failed: %s", name, exc)
    finally:
        entry.duration_seconds = time.monotonic() - start


async def _delta_scope_ids(session: AsyncSession, watermark: datetime) -> frozenset[str]:
    """The §4 delta scope: particles changed since ``watermark`` + particles
    from corpus entries deposited since, threaded through the at-least-one-side-in-scope seams unchanged."""
    from particles.corpus.store import list_entry_ids_created_since

    changed = await get_particle_ids_changed_since(session, watermark)
    entry_ids = await list_entry_ids_created_since(session, watermark)
    if entry_ids:
        changed |= await get_particle_ids_for_entries(session, entry_ids)
    return frozenset(changed)


async def _count_backlog(session: AsyncSession, report: ConsolidationReport) -> int:
    """Disclose the PENDING backlog without extracting (the degraded pass 1)."""
    from particles.corpus.store import list_pending_snapshots_oldest_first

    pending = await list_pending_snapshots_oldest_first(session)
    report.pending_total = len(pending)
    report.pending_remaining = len(pending)
    return 0


async def _pass_refresh(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 0.5: re-check every LAZY ``file://`` entry against the file on disk.

    The pass — the wire that makes the loop close unattended: edit
    ``AGENTS.md`` → tonight's run stats a changed mtime → the hash differs → a
    RESPONSE snapshot lands PENDING → pass 1 extracts it → the §2 generation
    cascade retires the prior generation → pass 2 sweeps cross-entry → pass 6
    re-renders the projection.

    Returns 0: the pass makes no LLM calls at all — its cost is a ``stat`` per
    entry plus a read + SHA-256 only for the files whose mtime moved.
    Per-entry failures are disclosed and never fatal (§8).
    """
    from particles.corpus.fetch import maybe_refetch
    from particles.corpus.store import list_refreshable_local_entries

    entries = await list_refreshable_local_entries(session)
    cap = get_config().local_refresh.max_entries
    report.refresh_remaining = max(0, len(entries) - cap)

    for entry_id, _uri_r in entries[:cap]:
        report.refresh_checked += 1
        try:
            prior_before = await _latest_snapshot_id(session, entry_id)
            snap = await maybe_refetch(session, entry_id)
            if snap is None:
                report.refresh_missing += 1
            elif snap.snapshot_id == prior_before:
                # maybe_refetch returned the existing snapshot untouched: the
                # tier-1 mtime compare short-circuited before any read.
                report.refresh_unchanged_mtime += 1
            elif snap.warc_record_type is WarcRecordType.REVISIT:
                report.refresh_unchanged_hash += 1
            else:
                report.refresh_updated += 1
            await session.commit()
        except Exception as exc:  # noqa: BLE001 — §8: continue the sweep, disclose
            await session.rollback()
            log.warning("consolidation: refresh failed for %s: %s", entry_id[:8], exc)
    return 0


async def _latest_snapshot_id(session: AsyncSession, entry_id: str) -> str | None:
    """The newest snapshot id for an entry, or None when it has none yet."""
    from particles.corpus.store import list_snapshots_for_entry

    snapshots = await list_snapshots_for_entry(session, entry_id)
    if not snapshots:
        return None
    return max(snapshots, key=lambda s: s.captured_at).snapshot_id


async def _pass_extract(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 1: extract PENDING snapshots, oldest first, capped per run (§3.1).

    Level-triggered like the harvest — the corpus is the state; a
    capped run discloses the remainder and the next run continues. Per-snapshot
    failures are disclosed, never fatal (mirroring the audit's extract loop).

    Under ``consolidation.extract_batching`` (default on) the capped
    set runs as concurrent per-snapshot tasks whose LLM requests merge into
    one pooled batch; ``false`` restores this serial loop exactly.
    """
    # Deferred import: the pipeline pulls the extractor registry / LLM stack;
    # load it only when there is something to extract (AGENTS.md case 2).
    from particles.corpus.store import list_pending_snapshots_oldest_first
    from particles.operations.extract import extract_snapshot

    pending = await list_pending_snapshots_oldest_first(session)
    report.pending_total = len(pending)
    cap = get_config().consolidation.max_pending_entries
    if get_config().consolidation.extract_batching:
        return await _pass_extract_pooled(pending[:cap], report)
    for entry_id, snapshot_id in pending[:cap]:
        try:
            await extract_snapshot(session, entry_id, snapshot_id, agent_id=report.actor)
            await session.commit()
            report.pending_extracted += 1
        except AccountLevelLLMError as exc:
            # §8 says continue-and-disclose on a per-snapshot failure, but an
            # account-level failure is not per-snapshot: every remaining
            # extraction in the cap would fail identically. Stop the pass and
            # disclose once — the untried snapshots are still PENDING, so the
            # next night (or the next `extract --all-pending`) resumes.
            await session.rollback()
            report.pending_failed += 1
            log.error("consolidation: extraction unavailable (account-level): %s", exc)
            break
        except Exception as exc:  # noqa: BLE001 — §8: continue the batch, disclose
            await session.rollback()
            report.pending_failed += 1
            log.warning(
                "consolidation: extraction failed for %s/%s: %s",
                entry_id[:8],
                snapshot_id[:8],
                exc,
            )
    report.pending_remaining = report.pending_total - report.pending_extracted
    # A lower bound on the spend (a chunked source makes one call per chunk);
    # token telemetry lives with the OTel spans.
    return report.pending_extracted


async def _pass_extract_pooled(batch: list[tuple[str, str]], report: ConsolidationReport) -> int:
    """The pooled twin of the serial extract loop.

    One asyncio task per snapshot, each on its own session, all registered on
    one :class:`~particles.llm.CompletionPool` — so every snapshot's chunk
    requests land in the same nightly batch and the turnarounds
    overlap instead of serialising. Concurrency leans entirely on machinery
    that already serves the multi-process case: the IN_PROGRESS claim
    protocol and the write lock around the pipeline's write phase.

    Two rules:

    * **At most one snapshot per corpus entry per run** — a second pending
      snapshot of the same entry stays PENDING for the next run
      (level-triggered), preserving the ordering where a later snapshot's
      carry-forward lookup sees the earlier one's persisted chunk hashes.
    * **An account-level failure stops the pass once.** The pool raises it in
      every parked task; each task's cleanup has already reset its snapshot
      IN_PROGRESS → PENDING, and it is counted as one failure here, exactly
      as the serial loop's single ``break`` discloses it.
    """
    # Deferred import: the pipeline pulls the extractor registry / LLM stack
    # (AGENTS.md case 2), mirroring the serial loop above.
    from particles.db import session_scope
    from particles.llm import CompletionPool
    from particles.operations.extract import extract_snapshot

    seen_entries: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for entry_id, snapshot_id in batch:
        if entry_id in seen_entries:
            continue
        seen_entries.add(entry_id)
        chosen.append((entry_id, snapshot_id))
    if len(chosen) < len(batch):
        log.info(
            "consolidation: %d pending snapshot(s) deferred to the next run "
            "(one snapshot per corpus entry per pooled pass)",
            len(batch) - len(chosen),
        )

    # expected_participants closes the startup race: no wave dispatches until
    # every task below has registered, however quickly the first ones park.
    pool = CompletionPool("extraction", expected_participants=len(chosen))

    async def _run_one(entry_id: str, snapshot_id: str) -> None:
        async with pool.participant(), session_scope() as task_session:
            await extract_snapshot(
                task_session,
                entry_id,
                snapshot_id,
                agent_id=report.actor,
                completion_pool=pool,
            )
            await task_session.commit()

    outcomes = await asyncio.gather(
        *(_run_one(entry_id, snapshot_id) for entry_id, snapshot_id in chosen),
        return_exceptions=True,
    )

    account_level_seen = False
    for (entry_id, snapshot_id), outcome in zip(chosen, outcomes, strict=True):
        if outcome is None:
            report.pending_extracted += 1
        elif isinstance(outcome, AccountLevelLLMError):
            # Not per-snapshot: every task failed identically and each
            # snapshot was already reset to PENDING on the way out. Disclose
            # once below, mirroring the serial loop's single break.
            account_level_seen = True
        elif isinstance(outcome, BaseException):
            report.pending_failed += 1
            log.warning(
                "consolidation: extraction failed for %s/%s: %s",
                entry_id[:8],
                snapshot_id[:8],
                outcome,
            )
    if account_level_seen:
        report.pending_failed += 1
        log.error(
            "consolidation: extraction unavailable (account-level); "
            "untried snapshots remain PENDING for the next run"
        )
    report.pending_remaining = report.pending_total - report.pending_extracted
    # A lower bound on the spend, as in the serial loop; per-chunk counts ride
    # the OTel spans.
    return report.pending_extracted


async def _pass_reconcile(session: AsyncSession, report: ConsolidationReport) -> int:
    """Pass 2: the cross-entry document-supersession sweep. Idempotent.

    Probe-bearing (one ``semantic_lint`` call per candidate pair, routed
    through the breaker seam and capped at
    ``consolidation.max_reconcile_probes``, highest-similarity-first) — so the
    caller only runs it when ``semantic_ok``, and a capped run's truncation is
    disclosed in the report ("probed X of Y candidate pairs").
    """
    summary = await reconcile_supersession(session)
    demoted = summary.get("demoted", 0)
    probed = summary.get("probed", 0)
    candidates = summary.get("candidate_pairs", 0)
    report.reconcile_demoted = demoted if isinstance(demoted, int) else 0
    report.reconcile_candidate_pairs = candidates if isinstance(candidates, int) else 0
    report.reconcile_probes_run = probed if isinstance(probed, int) else 0
    return probed if isinstance(probed, int) else 0


async def _pass_census(
    session: AsyncSession,
    report: ConsolidationReport,
    *,
    semantic: bool,
    scope_ids: frozenset[str] | None,
) -> list[CurationCard]:
    """Pass 3: one ``collect_cards`` pass, capped + scoped (§3.3).

    The re-audit composition: the probe control carries
    ``audit.max_contradiction_probes`` and the §4 delta scope; the duplicate partition keeps the store-wide tail count beside the in-scope
    headline. Duplicates run in REPORT mode (unjudged candidates — the audit's
    default; ``--judge`` remains an interactive choice). Granularity probes
    stay off on the card path (via ``collect_cards``).
    """
    probe_control: ContradictionProbeControl | None = None
    if semantic:
        probe_control = ContradictionProbeControl(
            max_probes=get_config().audit.max_contradiction_probes,
            scope_particle_ids=scope_ids,
            # The dream cycle runs unattended at 03:30: nobody is
            # waiting on these probes, so they go out as one half-price batch
            #. ``particles lint`` and the interactive first-run
            # audit leave this off and keep the sequential loop.
            latency_tolerant=True,
        )
    cards = await collect_cards(
        session,
        semantic=semantic,
        duplicate_mode=SuggestMode.REPORT,
        contradiction_probe=probe_control,
        duplicate_scope_ids=scope_ids,
    )

    counts: dict[str, int] = {}
    bases: dict[str, int] = {}
    for card in cards:
        counts[card.kind.value] = counts.get(card.kind.value, 0) + 1
        for basis in card.contested_bases or ():
            bases[basis] = bases.get(basis, 0) + 1
    report.card_counts = counts
    report.contested_bases = bases

    # hybrid: in-scope headline, store-wide tail always disclosed.
    dup_total = sum(1 for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
    report.duplicate_candidate_pairs_total = dup_total
    if scope_ids is None:
        report.duplicate_in_scope = dup_total
    else:
        report.duplicate_in_scope = sum(
            1
            for c in cards
            if c.kind is CardKind.DUPLICATE_PAIR and any(pid in scope_ids for pid in c.particle_ids)
        )

    if probe_control is not None:
        report.contradiction_candidate_pairs = probe_control.candidate_pairs
        report.contradiction_intra_scope_pairs = probe_control.intra_scope_pairs
        report.contradiction_probes_run = probe_control.probes_run
    return cards


async def _pass_curation(
    session: AsyncSession,
    report: ConsolidationReport,
    cards: list[CurationCard],
    *,
    scope: CollectionScope,
    semantic: bool,
) -> int:
    """Pass 4: persist the collection pass 3 paid for, and report the top worklist.

    **The original §3.4 honesty note is superseded.** The queue used to be
    computed on demand and persist nothing, so this pass ended with a rendered
    worklist and threw 13,000+ fully-formed cards away — while `GET /curation`
    rebuilt them from scratch on every request (172 s measured). The collection
    is now stored, so the night's work is what the morning's queue serves.

    ``scope`` is this run's §4 scope, which drives the §4 per-kind
    replace-vs-carry-forward rule: a delta run must not let its narrower
    contradiction set erase the store-wide picture built by earlier runs.
    """
    merged, snapshot_id = await collect_and_persist(
        session, semantic=semantic, scope=scope, cards=cards
    )
    report.curation_snapshot_id = snapshot_id

    suppressed = await _suppressed_keys(session)
    eligible = [c for c in merged if c.key not in suppressed]
    result = await build_curation_queue(session, cards=eligible)
    report.curation_queue_total = len(eligible)
    report.curation_queue = [_queue_line(card) for card in result.cards]
    return 0


def _queue_line(card: CurationCard) -> str:
    """One rendered worklist line: kind, claim text (when briefed), diagnostic."""
    text = card.particles[0].content if card.particles else None
    body = f'"{text}"' if text else (card.corpus_url or card.key)
    detail = f" — {card.diagnostic}" if card.diagnostic else ""
    return f"[{card.kind.value}] {body}{detail}"


async def _pass_utility(
    session: AsyncSession,
    report: ConsolidationReport,
    *,
    watermark: datetime | None,
    behavioural: bool,
) -> int:
    """Pass 5: the utility-mining pass over CONVERSATION entries.

    Delta runs mine transcripts harvested since the last run (idempotent
    regardless — events are keyed on ``(particle_id, session_id)``); a
    store-wide / first run re-mines everything reachable. A degraded run
    forces the literal tier only (§6).

    ONE behavioural budget for the whole pass (correction v1.74.1):
    ``utility.mining.max_behavioural_calls`` is a per-*run* cap, so the
    remaining budget is threaded into every ``mine_session`` call and the
    behavioural tier stops when it is spent — the literal tier (LLM-free)
    keeps mining every session. Exhaustion is disclosed in the report
    ("behavioural budget exhausted after N of M sessions").
    """
    from particles.corpus.deposit import load_blob
    from particles.corpus.store import list_entries, list_snapshots_for_entry

    actives = await get_particles_by_status(session, Status.ACTIVE)
    entries = await list_entries(session, limit=1_000_000, source_type="CONVERSATION")
    budget = get_config().utility.mining.max_behavioural_calls
    calls = 0
    for entry in entries:
        snapshots = [
            s
            for s in await list_snapshots_for_entry(session, entry.entry_id)
            if s.content_hash and s.archive_path
        ]
        if not snapshots:
            continue
        latest = max(snapshots, key=lambda s: s.captured_at)
        if watermark is not None and _as_utc(latest.captured_at) < watermark:
            continue  # already mined by a previous run's pass
        try:
            text = load_blob(latest.content_hash).decode("utf-8", errors="replace")
        except (OSError, FileNotFoundError):
            log.warning("consolidation: blob for entry %s missing; skipping", entry.entry_id)
            continue
        sid = session_id_from_uri(entry.uri_r) or entry.entry_id
        result = await mine_session(
            session,
            sid,
            text,
            actives,
            behavioural_matching=None if behavioural else False,
            max_behavioural_calls=max(0, budget - calls),
            # Unattended run — the matcher's calls may be batched.
            latency_tolerant=True,
        )
        report.utility_literal += result.literal
        report.utility_behavioural += result.behavioural
        report.utility_sessions_mined += 1
        calls += result.behavioural_calls
        if (
            behavioural
            and result.behavioural_truncated
            and report.utility_behavioural_exhausted_after is None
        ):
            report.utility_behavioural_exhausted_after = report.utility_sessions_mined
    await session.commit()
    report.utility_behavioural_calls = calls
    return calls


async def _pass_abstraction(
    session: AsyncSession,
    report: ConsolidationReport,
    *,
    scope_ids: frozenset[str] | None,
) -> int:
    """Pass 5b: the abstraction-promotion pass.

    Revalidation ladder first, then new-cluster promotion, both under
    ``consolidation.abstraction.max_promotions_per_run``. Delta scope gates
    cluster discovery only (revalidation is store-wide by design — a premise
    change outside the window still needs repair).
    """
    report.abstraction = await run_abstraction_pass(session, scope_ids=scope_ids)
    await session.commit()
    return report.abstraction.llm_calls


async def _pass_projection(report: ConsolidationReport, runner: ProjectionRunner) -> int:
    """Pass 6: the render via the Surface-injected harvest-then-render tail."""
    report.projection = await runner()
    return 0


# ---------------------------------------------------------------------------
# Delta report (§7 — the fold)
# ---------------------------------------------------------------------------

#: Headline keys the delta report tracks, and how each is computed.
_DELTA_KEYS = ("contradictions", "duplicates", "stale")


def _headline_values(report: ConsolidationReport) -> dict[str, int]:
    return {
        "contradictions": report.headline_contradictions,
        "duplicates": report.duplicate_candidate_pairs_total,
        "stale": report.headline_stale,
    }


def _headline_from_payload(payload: dict[str, Any]) -> dict[str, int] | None:
    census = payload.get("census")
    if not isinstance(census, dict):
        return None
    cards = census.get("cards")
    if not isinstance(cards, dict):
        return None

    def _count(kind: CardKind) -> int:
        value = cards.get(kind.value, 0)
        return int(value) if isinstance(value, int) else 0

    return {
        "contradictions": _count(CardKind.CONTRADICTION) + _count(CardKind.CONTESTED),
        "duplicates": int(census.get("duplicate_candidate_pairs_total", 0) or 0),
        "stale": (
            _count(CardKind.STALE)
            + _count(CardKind.RECENCY_DECAY)
            + _count(CardKind.CONFIDENCE_DECAY)
        ),
    }


def _compute_deltas(report: ConsolidationReport, prior: OperatorEvent) -> dict[str, int]:
    """Per-headline-class delta against the prior run's recorded census."""
    previous = _headline_from_payload(prior.payload or {})
    if previous is None:
        return {}
    current = _headline_values(report)
    return {key: current[key] - previous.get(key, 0) for key in _DELTA_KEYS}


def _census_payload(report: ConsolidationReport) -> dict[str, Any]:
    """The machine-readable census block of the run record (§7)."""
    return {
        "cards": dict(report.card_counts),
        "contested_bases": dict(report.contested_bases),
        "contradiction_candidate_pairs": report.contradiction_candidate_pairs,
        "contradiction_intra_scope_pairs": report.contradiction_intra_scope_pairs,
        "contradiction_probes_run": report.contradiction_probes_run,
        "duplicate_candidate_pairs_total": report.duplicate_candidate_pairs_total,
        "duplicate_in_scope": report.duplicate_in_scope,
        "pending_backlog": report.pending_remaining,
        "pending_extracted": report.pending_extracted,
        "refresh_checked": report.refresh_checked,
        "refresh_updated": report.refresh_updated,
        "refresh_missing": report.refresh_missing,
        "reconcile_demoted": report.reconcile_demoted,
        "reconcile_candidate_pairs": report.reconcile_candidate_pairs,
        "reconcile_probes_run": report.reconcile_probes_run,
        "utility_literal": report.utility_literal,
        "utility_behavioural": report.utility_behavioural,
        "utility_behavioural_calls": report.utility_behavioural_calls,
        "utility_behavioural_exhausted_after": report.utility_behavioural_exhausted_after,
        "curation_queue_total": report.curation_queue_total,
        "curation_snapshot_id": report.curation_snapshot_id,
        **(
            {
                "abstraction_clusters": report.abstraction.clusters_found,
                "abstraction_promoted": len(report.abstraction.promoted_particle_ids),
                "abstraction_proposed": len(report.abstraction.proposed_event_ids),
                "abstraction_rejected_entailment": report.abstraction.rejected_entailment,
                "abstraction_rejected_duplicate": report.abstraction.rejected_duplicate,
                "abstraction_revalidated": (
                    report.abstraction.revalidation.refreshed_structural
                    + report.abstraction.revalidation.refreshed_entailed
                    + report.abstraction.revalidation.refreshed_paraphrase
                ),
                "abstraction_superseded": report.abstraction.revalidation.superseded,
                "abstraction_retired": report.abstraction.revalidation.retired,
            }
            if report.abstraction is not None
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# The interactive audit's recording (§7 — ``actor: audit``)
# ---------------------------------------------------------------------------


async def record_audit_run(
    session: AsyncSession,
    audit_report: AuditReport,
    *,
    started_at: datetime,
    actor: str = "audit",
) -> None:
    """Record an interactive audit as a ``CONSOLIDATION_RUN`` event (§7).

    The audit contributes to the §7 *delta report* (the most-recent-prior-run
    comparison) — a request asked for exactly this record and is discharged
    here. Correction (v1.74.1): an ``actor: audit`` event is **not**
    watermark-eligible and does **not** satisfy ``--if-due`` — the audit runs
    neither reconcile, utility mining, curation refresh, nor projection, so it
    must not stand in for a consolidation run (see :func:`latest_run_event`).
    Flushes via
    ``record_event``; the caller owns the commit, exactly like the audit's
    other writes. (``AuditReport`` is imported under ``TYPE_CHECKING`` only:
    ``operations.audit`` imports this module at runtime for the shared payload
    shape, so a runtime import here would be a cycle.)
    """
    completed_at = _utcnow()
    harvested = audit_report.files_audited is not None or audit_report.transcripts_audited > 0
    extract_pass = ConsolidationPass(
        name="extract",
        status="ran" if harvested else "skipped",
        detail=None if harvested else "re-audit — no harvest",
        llm_calls=int(audit_report.extracted_snapshots),
    )
    census_pass = ConsolidationPass(
        name="census",
        status="ran",
        llm_calls=int(audit_report.contradiction_probes_run),
    )
    counts = {bucket.kind.value: bucket.count for bucket in audit_report.buckets}
    report = ConsolidationReport(
        store=audit_report.store,
        actor=actor,
        card_counts=counts,
        contradiction_candidate_pairs=audit_report.contradiction_candidate_pairs,
        contradiction_intra_scope_pairs=audit_report.contradiction_intra_scope_pairs,
        contradiction_probes_run=audit_report.contradiction_probes_run,
        duplicate_candidate_pairs_total=audit_report.duplicate_candidate_pairs_total,
        pending_extracted=audit_report.extracted_snapshots,
    )
    payload = build_run_payload(
        store=audit_report.store,
        actor=actor,
        scope=audit_report.contradiction_probe_scope or "store",
        watermark=None,
        started_at=started_at,
        completed_at=completed_at,
        semantic_degraded=bool(audit_report.semantic_skipped),
        semantic_degraded_reason=audit_report.semantic_skip_reason,
        providers=_current_providers(),
        passes=[extract_pass, census_pass],
        census=_census_payload(report),
    )
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.CONSOLIDATION_RUN,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# The renderer (§7) — one renderer for terminal and --output
# ---------------------------------------------------------------------------


def _fmt_delta(deltas: dict[str, int], key: str) -> str:
    if key not in deltas:
        return ""
    value = deltas[key]
    sign = "+" if value > 0 else ""
    return f"  ({sign}{value} since last run)"


def render_consolidation_report(report: ConsolidationReport) -> str:
    """Render the §7 report shape — headline deltas, disclosures, the queue."""
    lines: list[str] = []
    when = (report.completed_at or report.started_at).strftime("%Y-%m-%d %H:%M")
    if report.previous_run_at is not None:
        prev = report.previous_run_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"Consolidated store '{report.store}' — {when} (previous run: {prev})")
    else:
        lines.append(f"Consolidated store '{report.store}' — {when} (first recorded run)")
    lines.append("")

    # --- Headline counts + deltas (a degraded run never reads "0"; §6) -----
    if report.semantic_degraded:
        reason = report.semantic_degraded_reason or "LLM unavailable"
        lines.append(f"  contradictions   not probed this run ({reason})")
    else:
        lines.append(
            f"  contradictions   {report.headline_contradictions}"
            f"{_fmt_delta(report.deltas, 'contradictions')}"
        )
    dup_line = f"  duplicates       {report.duplicate_candidate_pairs_total}"
    dup_line += _fmt_delta(report.deltas, "duplicates")
    if (
        report.effective_scope == "delta"
        and report.duplicate_in_scope < report.duplicate_candidate_pairs_total
    ):
        dup_line += f"  [{report.duplicate_in_scope} touch this run's delta]"
    lines.append(dup_line)
    lines.append(f"  stale            {report.headline_stale}{_fmt_delta(report.deltas, 'stale')}")
    if report.refresh_checked:
        refresh_line = f"  local sources    {report.refresh_checked} checked"
        if report.refresh_updated:
            refresh_line += f", {report.refresh_updated} changed → re-extracting"
        else:
            refresh_line += ", none changed"
        if report.refresh_missing:
            refresh_line += f", {report.refresh_missing} missing"
        lines.append(refresh_line)
    lines.append(
        f"  pending          extracted {report.pending_extracted}"
        + (
            f" ({report.pending_remaining} remain — next run continues)"
            if report.pending_remaining
            else ""
        )
    )
    if report.reconcile_demoted:
        lines.append(f"  reconcile        demoted {report.reconcile_demoted} superseded claim(s)")
    utility = f"  utility events   +{report.utility_literal + report.utility_behavioural}"
    if report.utility_behavioural:
        utility += f" ({report.utility_behavioural} behavioural)"
    lines.append(utility)
    if report.abstraction is not None:
        ab = report.abstraction
        parts: list[str] = []
        if ab.proposed_event_ids:
            parts.append(f"{len(ab.proposed_event_ids)} proposed")
        if ab.promoted_particle_ids:
            parts.append(f"{len(ab.promoted_particle_ids)} promoted")
        reval = ab.revalidation
        repaired = (
            reval.refreshed_structural + reval.refreshed_entailed + reval.refreshed_paraphrase
        )
        if repaired:
            parts.append(f"{repaired} revalidated")
        if reval.superseded:
            parts.append(f"{reval.superseded} superseded")
        if reval.retired:
            parts.append(f"{reval.retired} retired")
        if not parts:
            parts.append("nothing to do")
        lines.append(f"  abstraction      {', '.join(parts)}")
    if report.projection is not None:
        rendered = report.projection.get("rendered", 0)
        lines.append(f"  projection       re-rendered ({rendered} memory file(s))")

    # --- Disclosures (§6: every skip, cap, and degradation named) ----------
    if report.semantic_degraded:
        reason = report.semantic_degraded_reason or "LLM unavailable"
        lines.append("")
        lines.append(f"  semantic passes skipped: {reason}")
    if (
        not report.semantic_degraded
        and report.contradiction_probes_run < report.contradiction_candidate_pairs
    ):
        cap = get_config().audit.max_contradiction_probes
        lines.append("")
        lines.append(
            f"  contradiction probe capped: probed {report.contradiction_probes_run} of "
            f"{report.contradiction_candidate_pairs} candidate pairs "
            f"(audit.max_contradiction_probes = {cap})"
        )
    if (
        not report.semantic_degraded
        and report.reconcile_probes_run < report.reconcile_candidate_pairs
    ):
        reconcile_cap = get_config().consolidation.max_reconcile_probes
        lines.append("")
        lines.append(
            f"  reconcile probe capped: probed {report.reconcile_probes_run} of "
            f"{report.reconcile_candidate_pairs} candidate pairs "
            f"(consolidation.max_reconcile_probes = {reconcile_cap})"
        )
    if report.refresh_remaining:
        refresh_cap = get_config().local_refresh.max_entries
        lines.append("")
        lines.append(
            f"  local refresh capped: checked {report.refresh_checked} entries, "
            f"{report.refresh_remaining} not reached this run "
            f"(local_refresh.max_entries = {refresh_cap})"
        )
    if report.utility_behavioural_exhausted_after is not None:
        behavioural_cap = get_config().utility.mining.max_behavioural_calls
        lines.append("")
        lines.append(
            f"  behavioural budget exhausted after "
            f"{report.utility_behavioural_exhausted_after} of "
            f"{report.utility_sessions_mined} sessions "
            f"(utility.mining.max_behavioural_calls = {behavioural_cap})"
        )
    if report.abstraction is not None and report.abstraction.warnings:
        lines.append("")
        for warning in report.abstraction.warnings:
            lines.append(f"  abstraction: {warning}")
    if report.effective_scope == "delta" and report.watermark is not None:
        lines.append("")
        lines.append(
            f"  semantic scope: delta since {report.watermark:%Y-%m-%d %H:%M} UTC "
            f"({report.scope_particle_count or 0} beliefs) — pass --scope store for "
            f"the whole store"
        )
    for entry in report.passes:
        if entry.status == "skipped":
            lines.append(f"  pass skipped: {entry.name} — {entry.detail}")
        elif entry.status == "failed":
            lines.append(f"  pass FAILED: {entry.name} — {entry.detail}")
    if report.pending_failed:
        lines.append(
            f"  {report.pending_failed} snapshot(s) failed to extract — "
            f"run `particles reindex` to retry them."
        )

    # --- The morning's bus stop (§3.4) --------------------------------------
    if report.curation_queue:
        lines.append("")
        shown = len(report.curation_queue)
        lines.append(f"Curation queue — top {shown} of {report.curation_queue_total}:")
        for line in report.curation_queue:
            lines.append(f"  • {line}")

    lines.append("")
    lines.append("Run 'particles curate' to work these down.")
    return "\n".join(lines) + "\n"
