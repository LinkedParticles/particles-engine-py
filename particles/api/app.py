"""FastAPI application — Core endpoints (§C.6)."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from particles import __version__
from particles.api._middleware import BodySizeLimitMiddleware
from particles.api._rate_limit import RateLimitDep, enforce_rate_limit
from particles.api.auth import (
    AuthDep,
    ReadAuthDep,
    enforce_fail_closed_on_startup,
    warn_if_dev_auth_in_use,
)
from particles.api.daemon import DaemonStatus, daemon_status, start_daemon, stop_daemon
from particles.api.web_app import WEB_APP_MOUNT, build_web_ui_static_app
from particles.config import get_config
from particles.core.schema import (
    ContestedBadge,
    CorpusEntry,
    ExternalRef,
    FetchPolicy,
    GraphData,
    LintReport,
    Mutability,
    Particle,
    ParticleRelation,
    ParticleType,
    QualityReport,
    QueryRequest,
    QueryResponse,
    ResolutionAction,
    ReviewParticle,
    SourceType,
    Subject,
    SuggestMode,
    SuggestReport,
    TaxonomyDefinition,
)
from particles.core.status import Status
from particles.db import (
    DEFAULT_STORE,
    WriteLockTimeout,
    create_tables,
    get_session,
    get_write_session,
    session_scope,
)
from particles.http import SourceFetchError
from particles.operations._llm import llm_circuit_open
from particles.operations.agent_write import (
    AgentWriteResult,
    assert_belief,
    assign_subject_belief,
    retract_belief,
    supersede_belief,
)
from particles.operations.curation import (
    CurationCard,
    QueueSource,
    build_curation_queue,
    rebuild_curation_snapshot,
)
from particles.operations.curation.cards import CardKind
from particles.operations.deposit import (
    deposit_file,
    deposit_text,
    deposit_text_versioned,
    deposit_url,
)
from particles.operations.deposit_suggest import DepositSuggestReport
from particles.operations.extract import extract_snapshot
from particles.operations.graph_view import build_graph_data
from particles.operations.lint import run_lint
from particles.operations.quality import get_quality_report
from particles.operations.query import query
from particles.operations.reconcile import reconcile_supersession
from particles.operations.reindex import ReindexPlan, reindex
from particles.operations.review import list_inconsistencies, resolve
from particles.store.event_store import OperatorEvent

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan (FastAPI's startup/shutdown successor).

    Replaces the deprecated ``@app.on_event("startup")`` hook. Startup work
    runs before ``yield``; the shutdown half cancels the daemon tasks
    (a no-op unless daemon mode is on).
    """
    # Fail closed before anything else: if bearer auth is disabled and the
    # bind host is not loopback, refuse to start rather than serve
    # unauthenticated traffic beyond loopback. Raising here aborts startup.
    enforce_fail_closed_on_startup()
    warn_if_dev_auth_in_use()
    # Bootstrap observability. No-op unless enabled + the `otel` extra
    # is installed. This covers a raw `uvicorn particles.api.app:app` launch
    # (providers + httpx / SQLAlchemy / log correlation); the FastAPI server span
    # itself is added by `engine serve` via instrument_fastapi_app() before the
    # app starts (middleware cannot be added once serving has begun).
    from particles.observability import setup_observability

    setup_observability()
    await create_tables()
    # Resident daemon mode — opt-in, off by default. Started after
    # create_tables() so the first tick never races schema creation.
    start_daemon()
    try:
        yield
    finally:
        await stop_daemon()


app = FastAPI(
    title="Particles API",
    version=__version__,
    description="Python SDK for the Particles epistemic knowledge standard — Core Loop.",
    lifespan=lifespan,
)

# Defense-in-depth: cap the request body before any handler runs (F-7).
# The limit is read per-request from config, so this single registration
# honours reloads and the PARTICLES_MAX_REQUEST_BODY_BYTES override.
app.add_middleware(BodySizeLimitMiddleware)


@app.middleware("http")
async def _log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Interim request observability ahead of the OpenTelemetry work.

    uvicorn's access log only fires on *response*, so a request that hangs
    (e.g. blocked on a SQLite write lock held by a concurrent writer
) leaves no trace. Logging arrival makes a stuck request visible,
    and the ``database is locked`` branch records write-lock contention
    explicitly. This is a stopgap the OTel tracing work is expected to subsume.
    """
    started = time.monotonic()
    client = request.client.host if request.client else "-"
    log.info("request start: %s %s (client=%s)", request.method, request.url.path, client)
    try:
        response = await call_next(request)
    except OperationalError as exc:
        elapsed_ms = (time.monotonic() - started) * 1000
        if "database is locked" in str(exc).lower():
            log.warning(
                "request SQLITE_BUSY after %.0fms: %s %s — a concurrent writer holds the "
                "single SQLite write lock",
                elapsed_ms,
                request.method,
                request.url.path,
            )
        else:
            log.warning(
                "request DB error after %.0fms: %s %s — %s",
                elapsed_ms,
                request.method,
                request.url.path,
                exc.__class__.__name__,
            )
        raise
    elapsed_ms = (time.monotonic() - started) * 1000
    log.info(
        "request end: %s %s -> %s (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(WriteLockTimeout)
async def _write_lock_timeout_handler(_request: Request, exc: WriteLockTimeout) -> Response:
    """Translate a writer-lock timeout into a clean 503.

    A write endpoint could not acquire the cross-process writer lock within
    ``storage.write_lock.timeout_seconds`` — another process holds it. 503 marks
    it retryable; the thin client surfaces the message rather than a stack trace.
    """
    from fastapi.responses import JSONResponse

    log.warning("write-lock timeout: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


SessionDep = Annotated[AsyncSession, Depends(get_session)]
#: A session held under the cross-process writer lock — for short,
#: pure-write endpoints (the handler commits before returning, no slow I/O before
#: the write). ``/extract`` / ``/reindex`` keep ``SessionDep``: the extract
#: pipeline acquires the lock around its own write phase, never across the LLM.
WriteSessionDep = Annotated[AsyncSession, Depends(get_write_session)]


def _bounded_limit(limit: int | None, *, default: int | None = None) -> int:
    """Resolve a caller-supplied ``limit`` query param to a safe page size (F5).

    Any value is clamped to ``storage.max_page_size`` so a huge ``limit`` — or,
    via SQLite's negative-limit = all-rows quirk, a negative one — cannot force a
    full row scan + full Pydantic-list materialization. Non-positive values are
    rejected upstream by the endpoints' ``Query(..., ge=1)`` bound, so this only
    clamps the upper end. When ``limit`` is omitted (``None``): ``default`` is
    used if given (the endpoint's historical page size), else the cap itself —
    so a formerly-unbounded endpoint now returns at most ``max_page_size``. The
    cap is read from config at call time so operators can retune
    ``storage.max_page_size`` without a code change.
    """
    max_page = get_config().storage.max_page_size
    if limit is None:
        limit = max_page if default is None else default
    return min(limit, max_page)


# ---------------------------------------------------------------------------
# Common response models — typed contracts for error paths + small responses
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope returned by every non-2xx endpoint.

    The ``detail`` field is the operator-facing message; the HTTP status
    code carries the failure category (400 invalid input, 404 not found,
    500 server error).
    """

    detail: str


class HealthResponse(BaseModel):
    """Liveness probe response — returned only when the server is healthy."""

    status: str
    version: str
    #: When the artifact was built, if it stamped itself (``config.build.date``,
    #: which the container image sets). ``None`` from source, and the
    #: pair *version + built_at* is what lets a caller tell a long-running engine
    #: is behind — the version names the release, the date names the artifact.
    built_at: str | None = None
    #: Present only in daemon mode: the background tasks' live state,
    #: so a crashed tick is disclosed here rather than dying silently. ``None``
    #: for a plain ``engine serve``, which is the default.
    daemon: DaemonStatus | None = None


class StatusResponse(BaseModel):
    """Generic status envelope for write endpoints that have nothing else to return."""

    status: str


class ReindexResponse(BaseModel):
    """Summary of a reindex run.

    ``scope`` is the total number of corpus entries visited;
    ``succeeded`` / ``failed`` partition that scope; ``failed_entries``
    lists the entry IDs that failed (for operator-pull retry).
    ``lint_summary`` is the structural lint count-by-status the
    post-reindex lint pass produced (empty dict when lint was skipped
    because ``scope`` was 0). ``plan`` is the upfront work plan computed
    before extraction, per-snapshot detail included (dry and live runs
    alike — the missing-blob list must survive into the envelope because
    the CLI's human rendering caps it)."""

    scope: int
    succeeded: int
    failed: int
    failed_entries: list[str] = []
    lint_summary: dict[str, int] = {}
    dry_run: bool = False
    plan: ReindexPlan | None = None


class ReconcileResponse(BaseModel):
    """Summary of a document-supersession reconcile sweep.

    ``scope_pairs`` is the number of corpus-entry pairs in a (transitive)
    document-supersession relation; ``candidate_pairs`` the particle pairs that
    cleared the similarity floor; ``probed`` the replacement-signal probes run;
    ``demoted`` the count transitioned ACTIVE → PROVENANCE_STALE /
    DOCUMENT_SUPERSEDED. ``demotions`` is the per-demotion audit list (winner /
    loser / entries / similarity). ``enabled`` / ``single_trust_order`` echo the
    v1 gates — a ``False`` on either means the sweep was a no-op."""

    enabled: bool
    single_trust_order: bool
    dry_run: bool
    scope_pairs: int
    candidate_pairs: int
    probed: int
    demoted: int
    demotions: list[dict[str, Any]] = []


_ERR404: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Resource not found"}
}
_ERR400: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid input"}
}
_ERR500: dict[int | str, dict[str, Any]] = {
    500: {"model": ErrorResponse, "description": "Server error"}
}
_ERR502: dict[int | str, dict[str, Any]] = {
    502: {"model": ErrorResponse, "description": "Upstream source fetch failed"}
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check. Returns ``status="ok"``, the SDK version, and the build date.

    In daemon mode the response also carries the background tasks'
    state, and ``status`` becomes ``"degraded"`` once any of them has crashed.
    The process is still serving requests, so this stays a 200 — the API must
    not take itself down because a scheduled tick died — but the disclosure is
    machine-readable for an operator alert or a readiness gate.

    ``built_at`` is present only when the artifact stamped itself (the
    container image does; a source checkout does not), so the route stays
    unauthenticated with nothing new behind it: the version was already
    disclosed here, and a build timestamp says no more about the deployment
    than the version it accompanies.
    """
    built_at = get_config().build.date
    daemon = daemon_status()
    if not daemon.enabled:
        return HealthResponse(status="ok", version=__version__, built_at=built_at)
    return HealthResponse(
        status="ok" if daemon.healthy else "degraded",
        version=__version__,
        built_at=built_at,
        daemon=daemon,
    )


@app.get("/quality", response_model=QualityReport)
async def quality_dashboard(session: SessionDep, _auth: ReadAuthDep) -> QualityReport:
    """Extraction quality dashboard — calibration distribution, corpus status, subject coverage."""
    return await get_quality_report(session)


# ---------------------------------------------------------------------------
# Curation — the bus-stop-editing queue over HTTP. Exposes the
# existing `build_curation_queue` Engine operation so the deferred mobile / web
# client can render the leverage-ranked "today's N" feed.
# ---------------------------------------------------------------------------


class CurationQueueResponse(BaseModel):
    """The leverage-ranked, finite, snooze-filtered curation queue.

    ``cards`` is the ranked "today's N" worklist (highest leverage first), each a
    normalized projection of an existing read diagnostic. ``count`` is
    ``len(cards)`` for convenience. The HTTP mirror of `particles curate`.

    The cards come from a persisted collection, so the response
    also carries the §5 staleness stamp: ``built_at`` / ``age_seconds`` /
    ``stale`` say how old the *detection* is (suppression, belief status and
    post-build resolutions are always live), and ``per_kind_scope`` discloses
    which kinds a delta-scoped nightly build did not re-census store-wide. A
    client renders "as of 03:30" rather than implying freshness it lacks."""

    cards: list[CurationCard]
    count: int
    # True when semantic finders were requested but skipped because the LLM was
    # unavailable (account-level failure). A client shows "semantic
    # finders unavailable" rather than silently presenting fewer cards.
    semantic_skipped: bool = False
    # --- staleness stamp ---
    source: str = "live"
    snapshot_id: str | None = None
    built_at: datetime | None = None
    age_seconds: float | None = None
    stale: bool = False
    scope: str = "store"
    per_kind_scope: dict[str, str] = Field(default_factory=dict)
    collection_size: int = 0


@app.get("/curation", response_model=CurationQueueResponse)
async def get_curation_queue(
    session: SessionDep,
    _auth: AuthDep,
    limit: int | None = None,
    kind: str | None = None,
    semantic: bool | None = None,
    no_snapshot: bool = False,
) -> CurationQueueResponse:
    """Return the bus-stop-editing curation queue (HTTP mirror of `curate`).

    Composes the existing read diagnostics (lint, links/corpus-links suggest, the
    contested digest, quality) into one leverage-ranked, snooze-filtered worklist
    — no new detection logic. ``limit`` overrides ``curation.session_size`` for
    one call; ``kind`` restricts to a single ``CardKind`` (e.g. ``stale``,
    ``contested``); ``semantic`` runs the LLM-assisted finders (defaults to
    ``curation.semantic``). 400 on an unknown ``kind``.

    Served from the persisted collection, which the nightly
    consolidation cycle or ``POST /curation/rebuild`` writes. A store with no
    collection yet is served **live** — correct, but as slow as it was before
    the persisted-collection cutover — and labelled ``source:
    live``; this route never writes the cache, so a GET stays a read.
    ``no_snapshot=true`` forces that live path
    for one request. The response carries the staleness stamp; see
    ``CurationQueueResponse``."""
    kind_enum: CardKind | None = None
    if kind is not None:
        try:
            kind_enum = CardKind(kind.lower())
        except ValueError as exc:
            valid = ", ".join(k.value for k in CardKind)
            raise HTTPException(
                status_code=400, detail=f"Unknown kind {kind!r}. Valid: {valid}."
            ) from exc

    result = await build_curation_queue(
        session,
        limit=limit,
        kind=kind_enum,
        semantic=semantic,
        source=QueueSource.LIVE if no_snapshot else QueueSource.SNAPSHOT,
    )
    # Did the semantic pass actually run? It defaults to curation.semantic when the
    # caller leaves it unset. If it was on but the LLM circuit breaker is
    # open (account-level failure), report it so the client can say so.
    # A snapshot records the breaker state at *build* time, which is the honest
    # answer for cards it is serving.
    semantic_on = get_config().curation.semantic if semantic is None else semantic
    return CurationQueueResponse(
        cards=result.cards,
        count=result.count,
        semantic_skipped=result.semantic_degraded or (semantic_on and llm_circuit_open()),
        source=result.source,
        snapshot_id=result.snapshot_id,
        built_at=result.built_at,
        age_seconds=result.age_seconds,
        stale=result.stale,
        scope=result.scope,
        per_kind_scope=result.per_kind_scope,
        collection_size=result.collection_size,
    )


class CurationRebuildResponse(BaseModel):
    """Outcome of an explicit curation-collection rebuild."""

    snapshot_id: str | None
    built_at: datetime | None
    collection_size: int
    scope: str
    semantic: bool


@app.post("/curation/rebuild", response_model=CurationRebuildResponse)
async def rebuild_curation_endpoint(
    session: SessionDep,
    _auth: AuthDep,
    semantic: bool | None = None,
) -> CurationRebuildResponse:
    """Rebuild the persisted curation collection store-wide.

    **Synchronous and slow by design** — it runs every finder over the whole
    store (minutes on a large one). It is not how the queue becomes fast; the
    nightly consolidation cycle is. This is the operator's escape hatch when they
    need the queue to reflect right now, and the web UI already has an honest
    loading state for it. An async 202 + poll variant waits on a resident
    process to host job state (tracked).

    Requires the write bearer: it mutates the cache."""
    result = await rebuild_curation_snapshot(session, semantic=semantic)
    return CurationRebuildResponse(
        snapshot_id=result.snapshot_id,
        built_at=result.built_at,
        collection_size=result.collection_size,
        scope=result.scope,
        semantic=result.semantic,
    )


# ---------------------------------------------------------------------------
# Curation operator-event writes — affirm / snooze. These write
# operator EVENTS, not belief mutations: the queue's existing snooze/affirm
# filter (`_suppressed_keys`) consumes them. Gated exactly like the belief-write
# endpoints (`mcp.write.enabled_stores` → 403; the bearer) — they are
# operator-judgment writes against the canonical store, so the same write gate
# applies and they never ride the dev-key loopback skip.
# ---------------------------------------------------------------------------


class CurationAffirmRequest(BaseModel):
    """Affirm a belief still holds: records ``BELIEF_AFFIRMED``,
    suppressing the belief's card. Does NOT touch confidence — first-class
    corroboration is vouch."""

    particle_id: str
    card_key: str | None = None


class CurationSnoozeRequest(BaseModel):
    """Snooze a curation card: records ``CURATION_CARD_SNOOZED``
    keyed by ``card_key``. ``snooze_days=None`` is a permanent dismiss; a positive
    value hides the card for that many days (defaults to ``curation.snooze_days``)."""

    card_key: str
    particle_ids: list[str] = []
    snooze_days: int | None = None


class CurationEventResponse(BaseModel):
    """Outcome of a curation operator-event write."""

    event_id: str
    card_key: str | None = None
    snoozed_until: datetime | None = None


@app.post("/curation/affirm", response_model=CurationEventResponse, responses=_ERR400)
async def curation_affirm_endpoint(
    req: CurationAffirmRequest, session: SessionDep, _auth: AuthDep
) -> CurationEventResponse:
    """Affirm a belief still holds (``BELIEF_AFFIRMED`` operator event).

    Suppresses the belief's curation card without mutating the belief. 403 when
    curation writes are disabled (``mcp.write.enabled_stores`` default-deny)."""
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event

    _require_belief_writes_enabled()
    card_key = req.card_key or f"contested:{req.particle_id}"
    event = await record_event(
        session,
        actor="http:/curation/affirm",
        event_type=OperatorEventType.BELIEF_AFFIRMED,
        refs=[(EventRefKind.PARTICLE, req.particle_id)],
        payload={"card_key": card_key, "particle_id": req.particle_id},
    )
    await session.commit()
    return CurationEventResponse(event_id=event.event_id, card_key=card_key)


class MemoryUsefulRequest(BaseModel):
    """Mark a belief useful: records ``BELIEF_MARKED_USEFUL`` and credits
    the belief on the explicit utility channel. Does NOT touch confidence and does
    NOT claim the belief is true — it lifts the belief in the projection / digest
    ranking only. For "still true", use ``/curation/affirm``."""

    particle_id: str
    reason: str | None = None


class MemoryUsefulResponse(BaseModel):
    """Outcome of an explicit usefulness gesture.

    ``counted`` is False when this principal already credited this belief today —
    the rate bound. The event is recorded either way."""

    event_id: str
    particle_id: str
    credit_key: str
    counted: bool


@app.post("/memory/useful", response_model=MemoryUsefulResponse, responses=_ERR400)
async def memory_useful_endpoint(
    req: MemoryUsefulRequest, session: SessionDep, _auth: AuthDep
) -> MemoryUsefulResponse:
    """Mark a belief useful (``BELIEF_MARKED_USEFUL`` operator event).

    The explicit second utility channel, for the belief class the transcript
    miner cannot observe — prohibitions and design stances, complied
    with by *not* acting. Promotion-only and projection-only. 403 when belief
    writes are disabled (``mcp.write.enabled_stores`` default-deny)."""
    from particles.operations.utility_feedback import (
        HTTP_ACTOR,
        BeliefNotCreditable,
        mark_belief_useful,
    )

    _require_belief_writes_enabled()
    try:
        result = await mark_belief_useful(
            session, req.particle_id, actor=HTTP_ACTOR, reason=req.reason
        )
    except BeliefNotCreditable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return MemoryUsefulResponse(
        event_id=result.event_id,
        particle_id=result.particle_id,
        credit_key=result.credit_key,
        counted=result.counted,
    )


@app.post("/curation/snooze", response_model=CurationEventResponse, responses=_ERR400)
async def curation_snooze_endpoint(
    req: CurationSnoozeRequest, session: SessionDep, _auth: AuthDep
) -> CurationEventResponse:
    """Snooze / dismiss a curation card (``CURATION_CARD_SNOOZED`` event).

    ``snooze_days=None`` permanently dismisses; a positive value hides the card
    for that window. 403 when curation writes are disabled."""
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event

    _require_belief_writes_enabled()
    snooze_days = req.snooze_days
    until = None if snooze_days is None else datetime.now(UTC) + timedelta(days=snooze_days)
    event = await record_event(
        session,
        actor="http:/curation/snooze",
        event_type=OperatorEventType.CURATION_CARD_SNOOZED,
        refs=[(EventRefKind.PARTICLE, pid) for pid in req.particle_ids],
        payload={
            "card_key": req.card_key,
            "snoozed_until": None if until is None else until.isoformat(),
            "snooze_days": snooze_days,
        },
    )
    await session.commit()
    return CurationEventResponse(
        event_id=event.event_id, card_key=req.card_key, snoozed_until=until
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class DepositUrlRequest(BaseModel):
    """Deposit-from-URL request. The importer registry routes the URL to
    a domain-specific extractor when applicable (Reddit, GitHub, Numista, …);
    otherwise the generic fetcher takes it."""

    url: str
    source_type: str | None = None
    mutability: Mutability | None = None
    deposited_by: str = "api-user"
    tags: list[str] = []
    # deposit-time link-follow opt-in; None ⇒ the extractor's default
    # (Reddit / HN / Mastodon default True, everything else False). Exposed over
    # HTTP so the thin client's `deposit --follow-post-links` reaches the engine
    #.
    follow_post_links: bool | None = None
    follow_comment_links: bool | None = None


class DepositTextRequest(BaseModel):
    """Deposit raw text directly (no fetch). Used for conversation transcripts
    and operator-pasted snippets.

    ``author_id`` carries the originating principal so the agent
    belief-write ``deposit_text`` (the cheap deposit) attributes the
    CONVERSATION entry to its asserter over HTTP, matching the in-process path.

    ``uri_r`` switches the handler onto the *versioned* deposit
    path: the text lands on one corpus entry identified by the URI, with the
    caller-chosen ``mutability`` and an unchanged re-deposit skipped
    (``DepositResponse.unchanged``). Used by the Claude Code harvest hook over
    the remote transport (``claude_code.harvest.allow_remote``)."""

    text: str
    source_type: str = SourceType.CONVERSATION
    deposited_by: str = "api-user"
    tags: list[str] = []
    author_id: str | None = None
    uri_r: str | None = None
    mutability: str | None = None
    # enrols the entry in the local refresh loop
    # (``"LAZY"``). Reconciled onto an existing entry even when the content is
    # unchanged; ``None`` leaves any stored policy alone.
    fetch_policy: str | None = None
    content_published_at: datetime | None = None


class DepositResponse(BaseModel):
    """Identifiers for the corpus entry + initial snapshot created by a deposit.

    ``unchanged`` is set only by the versioned text-deposit path:
    the entry's latest snapshot already carried the deposited content hash, so
    nothing was written and the existing IDs are echoed."""

    entry_id: str
    snapshot_id: str
    unchanged: bool = False


@app.post(
    "/corpus/deposit/url",
    response_model=DepositResponse,
    responses={**_ERR400, **_ERR502},
)
async def corpus_deposit_url(
    req: DepositUrlRequest, session: WriteSessionDep, _auth: AuthDep
) -> DepositResponse:
    """Deposit a URL into the corpus. Returns the entry + snapshot IDs."""
    try:
        entry_id, snapshot_id = await deposit_url(
            session,
            req.url,
            deposited_by=req.deposited_by,
            source_type=req.source_type,
            mutability=req.mutability,
            tags=req.tags,
            follow_post_links=req.follow_post_links,
            follow_comment_links=req.follow_comment_links,
        )
        await session.commit()
    except SourceFetchError as exc:
        # Expected upstream condition — the origin refused/failed the fetch (e.g.
        # Reddit's 403 bot-wall), not an SDK bug. Log without a stack trace and
        # map to 502 Bad Gateway with the origin detail (the client request was
        # well-formed; the upstream is the problem).
        log.warning("deposit_url: upstream fetch failed for %r: %s", req.url, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        log.warning("deposit_url failed for %r: %s", req.url, exc, exc_info=True)
        raise HTTPException(status_code=400, detail="Could not deposit the provided URL") from exc
    return DepositResponse(entry_id=entry_id, snapshot_id=snapshot_id)


@app.post("/corpus/deposit/text", response_model=DepositResponse, responses={**_ERR400})
async def corpus_deposit_text(
    req: DepositTextRequest, session: WriteSessionDep, _auth: AuthDep
) -> DepositResponse:
    """Deposit raw text into the corpus (no fetch)."""
    if req.uri_r is not None:
        # Versioned path: stable per-source entry identity,
        # caller-chosen mutability, unchanged re-deposit skipped.
        try:
            mutability = Mutability(req.mutability) if req.mutability else Mutability.STABLE
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Unknown mutability {req.mutability!r}"
            ) from exc
        try:
            fetch_policy = FetchPolicy(req.fetch_policy) if req.fetch_policy else None
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Unknown fetch_policy {req.fetch_policy!r}"
            ) from exc
        entry_id, snapshot_id, unchanged = await deposit_text_versioned(
            session,
            text=req.text,
            uri_r=req.uri_r,
            source_type=req.source_type,
            mutability=mutability,
            tags=req.tags,
            deposited_by=req.deposited_by,
            content_published_at=req.content_published_at,
            fetch_policy=fetch_policy,
        )
        await session.commit()
        return DepositResponse(entry_id=entry_id, snapshot_id=snapshot_id, unchanged=unchanged)

    entry_id, snapshot_id = await deposit_text(
        session, req.text, req.deposited_by, req.source_type, req.tags, author_id=req.author_id
    )
    await session.commit()
    return DepositResponse(entry_id=entry_id, snapshot_id=snapshot_id)


@app.post("/corpus/deposit/file", response_model=DepositResponse)
async def corpus_deposit_file(
    file: UploadFile,
    session: WriteSessionDep,
    _auth: AuthDep,
    deposited_by: Annotated[str, Form()] = "api-user",
    source_type: Annotated[str | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
    content_date: Annotated[datetime | None, Form()] = None,
) -> DepositResponse:
    """Deposit a file upload into the corpus. The file is hashed +
    stored under the blob directory; metadata + initial snapshot are
    created from the upload's content-type.

    ``deposited_by`` / ``source_type`` / ``tags`` / ``content_date`` are
    optional multipart form fields so the thin client's local-file ``deposit``
    (which uploads the bytes over HTTP) keeps its flags; ``content_date``
    is the archival authorship date."""
    import tempfile

    # Create + write the temp file *inside* the try so a crash mid-read/write is
    # still covered by the finally unlink — a ``delete=False`` temp that leaked
    # before the try would accrete under repeated failed uploads (F20).
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "upload").suffix
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())
        entry_id, snapshot_id = await deposit_file(
            session,
            tmp_path,
            deposited_by=deposited_by,
            source_type=source_type,
            tags=tags or [],
            content_date=content_date,
        )
        await session.commit()
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return DepositResponse(entry_id=entry_id, snapshot_id=snapshot_id)


@app.get("/corpus", response_model=list[CorpusEntry])
async def list_corpus(
    session: SessionDep,
    _auth: ReadAuthDep,
    limit: int | None = Query(None, ge=1),
    source_type: str | None = None,
) -> list[CorpusEntry]:
    """List corpus entries, most-recently-deposited first (HTTP mirror of the MCP
    ``list_corpus_entries`` tool). Optional ``source_type`` filter.
    ``limit`` defaults to 50 and is capped at ``storage.max_page_size`` (F5)."""
    from particles.corpus.store import list_entries

    return await list_entries(
        session, limit=_bounded_limit(limit, default=50), source_type=source_type
    )


@app.get(
    "/corpus/{entry_id}",
    response_model=CorpusEntry,
    responses=_ERR404,
)
async def corpus_get(entry_id: str, session: SessionDep, _auth: ReadAuthDep) -> CorpusEntry:
    """Fetch a corpus entry by ID. 404 if not present."""
    from particles.corpus.store import get_entry as _get_entry

    entry = await _get_entry(session, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@app.get(
    "/corpus/blob/{selector}",
    responses={**_ERR404, **_ERR400},
)
async def corpus_blob(selector: str, session: SessionDep, _auth: ReadAuthDep) -> Response:
    """Return the raw stored bytes of a snapshot's blob (backs `corpus cat`).

    ``selector`` is a full ID or unambiguous prefix of a snapshot or a corpus
    entry; an entry resolves to its most-recent snapshot. The body is the exact
    deposited bytes (``application/octet-stream``); ``X-Snapshot-Id`` and
    ``X-Content-Hash`` response headers identify what was served. 404 when
    nothing matches or the blob is missing on disk; 400 for an ambiguous prefix.
    """
    from particles.corpus.deposit import load_blob
    from particles.corpus.store import resolve_snapshot_for_blob

    try:
        snap = await resolve_snapshot_for_blob(session, selector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if snap is None:
        raise HTTPException(status_code=404, detail="No snapshot or entry matches that ID")
    try:
        content = load_blob(snap.content_hash)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Blob missing for snapshot {snap.snapshot_id}")
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "X-Snapshot-Id": snap.snapshot_id,
            "X-Content-Hash": snap.content_hash,
        },
    )


class CorpusRetractRequest(BaseModel):
    """Retract every live particle sourced from a corpus entry.

    ``reason`` is recorded on the SOURCE_RETRACTED event. ``dry_run`` returns the
    plan (which particles *would* be retracted) without writing anything."""

    reason: str | None = None
    dry_run: bool = False


class CorpusRetractResponse(BaseModel):
    """Outcome of a corpus retraction. ``retracted_ids`` are the particles
    actually retracted, or — when ``dry_run`` is true — the ones that *would*
    be. ``skipped`` maps each non-live status to how many particles were left
    untouched. The corpus entry and its snapshots are always preserved."""

    entry_id: str
    dry_run: bool
    retracted_ids: list[str]
    skipped: dict[str, int]


@app.post(
    "/corpus/{entry_id}/retract",
    response_model=CorpusRetractResponse,
    responses=_ERR404,
)
async def corpus_retract_endpoint(
    entry_id: str, req: CorpusRetractRequest, session: WriteSessionDep, _auth: AuthDep
) -> CorpusRetractResponse:
    """Retract every live particle from a source, preserving the corpus +
    snapshots (HTTP mirror of `corpus retract`). Live particles
    (ACTIVE / INCONSISTENCY) become RETRACTED with reason SOURCE_RETRACTED;
    idempotent. 404 if the entry is absent. Run `POST /lint` afterwards to
    cascade PROVENANCE_STALE to downstream particles."""
    from particles.corpus.store import get_entry as _get_entry
    from particles.operations.retract import plan_retraction, retract_entry

    if await _get_entry(session, entry_id) is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    if req.dry_run:
        plan = await plan_retraction(session, entry_id)
        return CorpusRetractResponse(
            entry_id=entry_id,
            dry_run=True,
            retracted_ids=[item.particle_id for item in plan.to_retract],
            skipped=plan.skipped,
        )

    result = await retract_entry(session, entry_id, reason=req.reason)
    await session.commit()
    return CorpusRetractResponse(
        entry_id=entry_id,
        dry_run=False,
        retracted_ids=result.retracted_ids,
        skipped=result.skipped,
    )


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    """Identify the corpus entry + snapshot to run extraction against."""

    entry_id: str
    snapshot_id: str
    agent_id: str = "api-user"


@app.post(
    "/extract",
    response_model=list[Particle],
    responses={**_ERR404, **_ERR500},
)
async def extract(
    req: ExtractRequest, session: SessionDep, _auth: AuthDep, _rl: RateLimitDep
) -> list[Particle]:
    """Run the extractor pipeline against a corpus snapshot. Returns
    the ACTIVE particles produced by the run (excludes any that
    immediately transitioned to SUPERSEDED / INCONSISTENCY)."""
    try:
        particles = await extract_snapshot(
            session, req.entry_id, req.snapshot_id, agent_id=req.agent_id
        )
        await session.commit()
    except ValueError as exc:
        log.info(
            "extract: entry %r / snapshot %r not found: %s",
            req.entry_id,
            req.snapshot_id,
            exc,
        )
        raise HTTPException(status_code=404, detail="Corpus entry or snapshot not found") from exc
    except Exception as exc:
        log.exception("extract failed for entry %r / snapshot %r", req.entry_id, req.snapshot_id)
        raise HTTPException(status_code=500, detail="Extraction failed") from exc
    return particles


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


class NarrativeResponse(BaseModel):
    """A NARRATIVE particle plus its constituents in ``SEQUENCE_IN`` order.

    ``constituents`` is the linear narrative order; an empty list means the
    NARRATIVE has no ``PART_OF`` children linked yet.
    """

    narrative: Particle
    constituents: list[Particle]


@app.get(
    "/particles/{particle_id}/narrative",
    response_model=NarrativeResponse,
    responses=_ERR404,
)
async def get_narrative(
    particle_id: str, session: SessionDep, _auth: ReadAuthDep
) -> NarrativeResponse:
    """Return a NARRATIVE particle and its constituents in SEQUENCE_IN order.

    404 if the particle does not exist or is not a NARRATIVE."""
    from particles.operations.narrative import get_narrative_sequence
    from particles.store.particle_store import get_particle

    narrative = await get_particle(session, particle_id)
    if narrative is None:
        raise HTTPException(status_code=404, detail="Particle not found")
    if narrative.particle_type != ParticleType.NARRATIVE:
        raise HTTPException(status_code=404, detail="Particle is not a NARRATIVE")
    constituents = await get_narrative_sequence(session, particle_id)
    return NarrativeResponse(narrative=narrative, constituents=constituents)


@app.get(
    "/particles/{particle_id}/narratives",
    response_model=list[Particle],
    responses=_ERR404,
)
async def get_containing_narratives(
    particle_id: str, session: SessionDep, _auth: ReadAuthDep
) -> list[Particle]:
    """Return the NARRATIVE particles this particle is ``PART_OF``.

    404 if the particle does not exist; an empty list if it belongs to none."""
    from particles.operations.narrative import get_narratives_containing
    from particles.store.particle_store import get_particle

    if await get_particle(session, particle_id) is None:
        raise HTTPException(status_code=404, detail="Particle not found")
    return await get_narratives_containing(session, particle_id)


class NarrativeArticleResponse(BaseModel):
    """A NARRATIVE rendered as one cited prose article.

    ``body`` is the Markdown article (cited footnotes + References).
    ``used_synthesis`` is False when the deterministic structured-listing
    fallback was used (no key, LLM error, or a validation failure).
    ``constituent_count`` is the number of SEQUENCE_IN members rendered
    (0 ⇒ the narrative has no constituents yet and ``body`` is empty).
    """

    narrative: Particle
    body: str
    used_synthesis: bool
    constituent_count: int


@app.get(
    "/particles/{particle_id}/narrative/synthesis",
    response_model=NarrativeArticleResponse,
    responses=_ERR404,
)
async def get_narrative_synthesis(
    particle_id: str, session: SessionDep, _auth: AuthDep
) -> NarrativeArticleResponse:
    """Render a NARRATIVE as one cited prose article.

    Traverses the SEQUENCE_IN chain and runs the article-synthesis engine
    (LLM call; structured-listing fallback on failure). 404 if the particle
    does not exist or is not a NARRATIVE."""
    from particles.operations.narrative_synthesis import synthesize_narrative
    from particles.store.particle_store import get_particle

    article = await synthesize_narrative(session, particle_id)
    if article is None:
        if await get_particle(session, particle_id) is None:
            raise HTTPException(status_code=404, detail="Particle not found")
        raise HTTPException(status_code=404, detail="Particle is not a NARRATIVE")
    return NarrativeArticleResponse(
        narrative=article.narrative,
        body=article.body,
        used_synthesis=article.used_synthesis,
        constituent_count=len(article.constituents),
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def run_query(
    req: QueryRequest, session: SessionDep, _auth: AuthDep, _rl: RateLimitDep
) -> QueryResponse:
    """Tag-aware semantic query. Embeds the question, runs cosine
    similarity over ACTIVE particles, generates a natural-language
    answer with citations.

    Rate-limited per client (``api.rate_limit_per_minute``; security review F6)
    — this drives a paid embedding + completion per request. Bearer-gated
    regardless of ``api.require_auth_for_reads`` (security review F2): the paid
    Anthropic completion runs on the operator's ``ANTHROPIC_API_KEY``, so an open
    ``/query`` is an unbounded-spend vector. The dev-key loopback skip keeps it
    open for local development."""
    return await query(session, req)


# ---------------------------------------------------------------------------
# Graph view — the served half of the one-build-
# two-presentations contract: the same GraphData the static exporter embeds,
# returned on the wire for the unified web UI's #/browse route and the MCP
# graph_view tool.
# ---------------------------------------------------------------------------

#: The four scope values, all resolved since landed:
#: `subject` / `query` (v0), `inconsistency` (a contradiction's evidence) and
#: `projection` (a manifest section's selection).
GraphScope = Literal["subject", "query", "inconsistency", "projection"]

#: scope value → the selector param(s) it requires (the validation matrix).
_GRAPH_SELECTORS: dict[str, tuple[str, ...]] = {
    "subject": ("subject_id",),
    "query": ("q",),
    "inconsistency": ("inconsistency_id",),
    "projection": ("manifest", "section"),
}


@app.get("/graph", response_model=GraphData)
async def get_graph(
    _auth: AuthDep,
    _rl: RateLimitDep,
    scope: GraphScope,
    subject_id: str | None = None,
    q: str | None = None,
    inconsistency_id: str | None = None,
    manifest: str | None = None,
    section: str | None = None,
    hops: Annotated[int, Query(ge=0)] = 1,
    history: bool = False,
    as_of: datetime | None = None,
    max_nodes: Annotated[int | None, Query(ge=1)] = None,
    store: str = DEFAULT_STORE,
) -> GraphData:
    """One scoped epistemic subgraph: the same ``GraphData`` the
    static ``export graph`` artifact embeds, computed fresh per request.

    Scope is mandatory (the anti-hairball invariant — a whole-store render does
    not exist): ``scope=subject`` needs ``subject_id``, ``scope=query`` needs
    ``q``, ``scope=inconsistency`` needs ``inconsistency_id`` (a contradiction's
    evidence: the INCONSISTENCY anchor, its disputants, their
    subjects), ``scope=projection`` needs ``manifest`` + ``section`` (a
    manifest section's deterministic selection — the manifest path is
    resolved on the engine host, which the bearer gate makes operator-equivalent
    access); anything else is 422. ``subject_id`` accepts a subject id or an
    exact (case-insensitive) canonical name / alias; ``inconsistency_id``
    accepts a full id or unique prefix; ``scope_ref`` in the response is always
    the resolved anchor address. ``hops`` / ``max_nodes`` are clamped to the
    ``graph.*`` caps; ``as_of`` renders the graph as believed at T;
    ``history`` adds supersession-chain ghosts; ``store`` selects the target
    store (404 on an unknown handle).

    Bearer-gated regardless of ``api.require_auth_for_reads`` and rate-limited
    (same posture as ``/query``): the query scope drives a paid embedding per
    request. All epistemics are computed server-side — clients only render
    ."""
    given = {
        "subject_id": subject_id,
        "q": q,
        "inconsistency_id": inconsistency_id,
        "manifest": manifest,
        "section": section,
    }
    required = _GRAPH_SELECTORS[scope]
    for name in required:
        if given[name] is None:
            raise HTTPException(
                status_code=422,
                detail=f"scope={scope} requires {' and '.join(required)}",
            )
    # Passing another scope's selector is an error, not an ignored param —
    # a caller sending both is confused about what renders.
    for name, value in given.items():
        if value is not None and name not in required:
            raise HTTPException(status_code=422, detail=f"scope={scope} does not take {name}")

    try:
        async with session_scope(store) as session:
            return await build_graph_data(
                session,
                subject_id=subject_id,
                query=q,
                inconsistency_id=inconsistency_id,
                manifest=manifest,
                section=section,
                hops=hops,
                history=history,
                as_of=as_of,
                max_nodes=max_nodes,
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown store {store!r}") from exc
    except ValueError as exc:
        # The operation's scope-resolution errors: an unknown anchor (subject /
        # inconsistency / manifest / section) is a 404 — nothing at that
        # address in this store; anything else (a future as_of instant, an
        # ambiguous prefix, …) is a request-validation 422.
        status = 404 if str(exc).startswith("unknown ") else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


class LintRequest(BaseModel):
    """Lint-run options. ``fix=True`` applies auto-fixable findings
    (status transitions); ``semantic=True`` invokes the LLM
    contradiction check (costs tokens).

    ``fix`` defaults to ``False``: lint is read-only by
    default and only mutates Particle status when the caller opts in."""

    fix: bool = False
    semantic: bool = True
    low_coverage_threshold: int = 3


@app.post("/lint", response_model=LintReport)
async def run_lint_endpoint(
    req: LintRequest, request: Request, session: SessionDep, _auth: AuthDep
) -> LintReport:
    """Run the lint pipeline. Returns the structured findings report.

    ``fix`` defaults to ``False`` — pass ``fix: true`` to
    apply structural status transitions (STALENESS, RETRACTION_CASCADE,
    CORPUS_LINK_INTEGRITY)."""
    # Rate-limit only the semantic path (security review F6): it drives the paid
    # LLM contradiction check, whereas a structural-only lint (``semantic=False``)
    # makes no model call and stays unthrottled.
    if req.semantic:
        enforce_rate_limit(request)
    return await run_lint(
        session,
        fix=req.fix,
        semantic=req.semantic,
        low_coverage_threshold=req.low_coverage_threshold,
    )


@app.get("/lint/report", response_model=LintReport)
async def get_lint_report(session: SessionDep, _auth: ReadAuthDep) -> LintReport:
    """Return the structural-only lint report — no LLM call, no mutations."""
    return await run_lint(session, fix=False, semantic=False)


# ---------------------------------------------------------------------------
# Links — co-evidential candidate suggestion
# ---------------------------------------------------------------------------


class LinksSuggestRequest(BaseModel):
    """Options for ``POST /links/suggest``.

    ``mode=REPORT`` lists candidate pairs (no LLM, no mutation);
    ``LLM_JUDGE`` adds per-pair verdicts; ``APPLY`` auto-links PARAPHRASE
    pairs. ``confirmed`` must be ``true`` for ``APPLY`` to link more than
    ``links_suggest.apply_confirm_threshold`` pairs."""

    subject_id: str | None = None
    threshold: float | None = None
    mode: SuggestMode = SuggestMode.REPORT
    confirmed: bool = False


@app.post("/links/suggest", response_model=SuggestReport)
async def run_links_suggest(
    req: LinksSuggestRequest, session: SessionDep, _auth: AuthDep
) -> SuggestReport:
    """Propose (and optionally resolve) co-evidential candidate links.

    Mirrors the ``particles links suggest`` CLI. Returns ``409`` when
    ``APPLY`` would link more pairs than the confirm threshold and
    ``confirmed`` was not set."""
    from particles.operations.links_suggest import (
        ApplyConfirmationRequired,
        suggest_co_evidential,
    )

    try:
        return await suggest_co_evidential(
            session,
            subject_id=req.subject_id,
            threshold=req.threshold,
            mode=req.mode,
            confirmed=req.confirmed,
        )
    except ApplyConfirmationRequired as exc:
        # Handler-owned message built from the exception's typed fields (not a
        # str(exc) passthrough): APPLY links more pairs than the confirm
        # threshold; the caller must re-request with confirmed=true.
        raise HTTPException(
            status_code=409,
            detail=(
                f"Applying would link {exc.pair_count} pairs, more than the "
                f"confirm threshold of {exc.threshold}. Re-request with confirmed=true."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Citation-signal deposit suggestions
# ---------------------------------------------------------------------------


class CorpusLinksSuggestRequest(BaseModel):
    """Options for ``POST /corpus/links/suggest``.

    ``limit`` / ``min_sources`` default to ``citation_signal.rank_cap`` /
    ``citation_signal.min_distinct_sources`` when omitted."""

    limit: int | None = None
    min_sources: int | None = None


class CorpusLinksDismissRequest(BaseModel):
    """Options for ``POST /corpus/links/dismiss``.

    ``snooze_days=None`` is a permanent dismiss; a positive value snoozes for
    that many days."""

    url: str
    snooze_days: int | None = None


class CorpusLinksDismissResponse(BaseModel):
    canonical_url: str
    suppressed_until: datetime


@app.post("/corpus/links/suggest", response_model=DepositSuggestReport)
async def run_corpus_links_suggest(
    req: CorpusLinksSuggestRequest, session: SessionDep, _auth: ReadAuthDep
) -> DepositSuggestReport:
    """Rank undeposited but frequently-cited URLs as deposit suggestions.

    Mirrors ``particles corpus links suggest``. Read-only — nothing is fetched
    or deposited."""
    from particles.operations.deposit_suggest import suggest_deposits

    return await suggest_deposits(session, limit=req.limit, min_sources=req.min_sources)


@app.post("/corpus/links/dismiss", response_model=CorpusLinksDismissResponse)
async def run_corpus_links_dismiss(
    req: CorpusLinksDismissRequest, session: SessionDep, _auth: AuthDep
) -> CorpusLinksDismissResponse:
    """Dismiss / snooze a deposit suggestion (audited via the event log).

    Returns ``400`` when ``url`` is not a usable http(s) URL."""
    from particles.operations.deposit_suggest import dismiss_suggestion
    from particles.url_canonical import canonicalize_url

    canon = canonicalize_url(req.url)
    if canon is None:
        raise HTTPException(status_code=400, detail=f"Not a usable http(s) URL: {req.url!r}")
    until = await dismiss_suggestion(
        session,
        canonical_url=canon,
        actor="http:/corpus/links/dismiss",
        snooze_days=req.snooze_days,
    )
    await session.commit()
    return CorpusLinksDismissResponse(canonical_url=canon, suppressed_until=until)


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


@app.get("/review", response_model=list[Particle])
async def list_review_queue(session: SessionDep, _auth: AuthDep) -> list[Particle]:
    """Return all INCONSISTENCY particles awaiting operator resolution."""
    return await list_inconsistencies(session)


class ReviewRequest(BaseModel):
    """Resolve an INCONSISTENCY: pick a ``ResolutionAction`` (trust A,
    trust B, both wrong, both right) and optionally annotate."""

    action: ResolutionAction
    reviewer_id: str
    domain: str = "general"
    note: str | None = None


@app.post(
    "/review/{particle_id}",
    response_model=ReviewParticle,
    responses=_ERR404,
)
async def review_particle(
    particle_id: str, req: ReviewRequest, session: SessionDep, _auth: AuthDep
) -> ReviewParticle:
    """Resolve an INCONSISTENCY particle. The resolution is annotation-
    only by default; the originals keep their status. A new
    SourceTrustStatement may be created depending on the action."""
    try:
        return await resolve(
            session,
            particle_id,
            req.action,
            req.reviewer_id,
            req.domain,
            req.note,
        )
    except ValueError as exc:
        log.info("review: particle %r not resolvable: %s", particle_id, exc)
        raise HTTPException(
            status_code=404, detail="No INCONSISTENCY particle with that id"
        ) from exc


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------


class ReindexRequest(BaseModel):
    """Reindex options. ``entry_ids=None`` reindexes every stale /
    failed snapshot; ``extractor_version`` (when set) restricts to
    particles emitted by that extractor version (used after an
    extractor upgrade). The particle-selecting fields —
    ``extractor_version`` / ``extractor_id`` / ``provider_model`` — union
    with each other and intersect with ``entry_ids`` when both are sent
    , so naming entries can only narrow the scope."""

    entry_ids: list[str] | None = None
    extractor_version: str | None = None
    extractor_id: str | None = None
    include_failed: bool = True
    # re-extract what one "<provider>:<model>" pairing produced.
    provider_model: str | None = None
    rate_limit_per_minute: int = 100
    # Compute and return the work plan without extracting (zero LLM calls,
    # zero writes); the response carries per-snapshot detail in ``plan``.
    dry_run: bool = False


@app.post("/reindex", response_model=ReindexResponse)
async def run_reindex(
    req: ReindexRequest, session: SessionDep, _auth: AuthDep, _rl: RateLimitDep
) -> ReindexResponse:
    """Re-extract stale / failed snapshots. Honours chunk-hash
    carry-forward so unchanged source chunks skip
    the LLM call."""
    result = await reindex(
        session,
        entry_ids=req.entry_ids,
        extractor_version=req.extractor_version,
        extractor_id=req.extractor_id,
        include_failed=req.include_failed,
        provider_model=req.provider_model,
        rate_limit_per_minute=req.rate_limit_per_minute,
        dry_run=req.dry_run,
    )
    return ReindexResponse.model_validate(result)


class ReconcileRequest(BaseModel):
    """Document-supersession reconcile-sweep options.

    ``dry_run=True`` reports what would be demoted without mutating the store."""

    dry_run: bool = False


@app.post("/reconcile", response_model=ReconcileResponse)
async def run_reconcile(
    req: ReconcileRequest, session: SessionDep, _auth: AuthDep
) -> ReconcileResponse:
    """Run the cross-entry document-supersession sweep: demote
    superseded claims the intra-entry extract path never reconciles. v1 is
    single-trust-order only and covers the document-supersession mode."""
    result = await reconcile_supersession(session, dry_run=req.dry_run)
    return ReconcileResponse.model_validate(result)


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


class CreateSubjectRequest(BaseModel):
    """Operator-supplied fields for creating a new Subject by hand
    (the normal path is automatic resolution during extraction)."""

    canonical_name: str
    description: str | None = None
    aliases: list[str] = []
    external_ids: list[ExternalRef] = []


@app.get("/subjects", response_model=list[Subject])
async def list_subjects(
    session: SessionDep,
    _auth: ReadAuthDep,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
    order: Literal["name", "degree"] = Query("name"),
) -> list[Subject]:
    """List Subjects (alphabetical). ``limit``/``offset`` paginate (MCP-friendly);
    an omitted ``limit`` returns the first ``storage.max_page_size`` and any
    ``limit`` is clamped to that cap (security review F5 — formerly unbounded).
    ``order=degree`` sorts by descending ACTIVE-particle link count instead
    (most-connected first — e.g. the web UI's Browse seed)."""
    from particles.store.subject_store import list_all_subjects

    return await list_all_subjects(session, limit=_bounded_limit(limit), offset=offset, order=order)


@app.get("/subjects/search", response_model=list[Subject])
async def search_subjects_endpoint(
    session: SessionDep,
    _auth: ReadAuthDep,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
) -> list[Subject]:
    """Full-text search Subjects by canonical name or alias. ``limit``/``offset``
    paginate; an omitted ``limit`` returns up to ``storage.max_page_size``
    matches and any ``limit`` is clamped to it. ``q`` is bounded to 1–200 chars
    and its LIKE wildcards are escaped in the store (security review F5 / F7)."""
    from particles.store.subject_store import search_subjects

    return await search_subjects(session, q, limit=_bounded_limit(limit), offset=offset)


@app.get(
    "/subjects/{subject_id}",
    response_model=Subject,
    responses=_ERR404,
)
async def get_subject_endpoint(subject_id: str, session: SessionDep, _auth: ReadAuthDep) -> Subject:
    """Fetch a Subject by ID. 404 if not present."""
    from particles.store.subject_store import get_subject

    s = await get_subject(session, subject_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    return s


@app.get("/subjects/{subject_id}/particles", response_model=list[Particle])
async def get_particles_for_subject(
    subject_id: str,
    session: SessionDep,
    _auth: ReadAuthDep,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
) -> list[Particle]:
    """List ACTIVE particles whose subject set includes this subject.

    ``limit``/``offset`` paginate over the linked particle IDs; an omitted
    ``limit`` returns up to ``storage.max_page_size`` and any ``limit`` is
    clamped to it (security review F5 — formerly returned every linked
    particle)."""
    from particles.store.particle_store import get_particle
    from particles.store.subject_store import get_particles_for_subject as _get

    particle_ids = await _get(session, subject_id, limit=_bounded_limit(limit), offset=offset)
    particles: list[Particle] = []
    for pid in particle_ids:
        p = await get_particle(session, pid)
        if p is not None:
            particles.append(p)
    return particles


class SubjectParticleIdsResponse(BaseModel):
    """Particle IDs linked to a subject (any status) + the true total.

    Backs the MCP ``subjects_show`` tool: ``particle_ids`` is capped at ``limit``
    for response-size control, while ``particle_count`` is the full total so the
    caller can detect truncation. (Distinct from ``/subjects/{id}/particles``,
    which returns full ACTIVE particle bodies.)"""

    particle_ids: list[str]
    particle_count: int


@app.get("/subjects/{subject_id}/particle-ids", response_model=SubjectParticleIdsResponse)
async def get_subject_particle_ids(
    subject_id: str, session: SessionDep, _auth: ReadAuthDep, limit: int | None = Query(None, ge=1)
) -> SubjectParticleIdsResponse:
    """Return up to ``limit`` particle IDs linked to a subject, plus the true total.

    ``limit`` defaults to 100 and is clamped to ``storage.max_page_size`` (F5)."""
    from particles.store.subject_store import (
        count_particles_for_subject,
    )
    from particles.store.subject_store import (
        get_particles_for_subject as _get,
    )

    particle_ids = await _get(session, subject_id, limit=_bounded_limit(limit, default=100))
    particle_count = await count_particles_for_subject(session, subject_id)
    return SubjectParticleIdsResponse(particle_ids=particle_ids, particle_count=particle_count)


@app.post("/subjects", response_model=Subject)
async def create_subject(req: CreateSubjectRequest, session: SessionDep, _auth: AuthDep) -> Subject:
    """Create a Subject by hand. The normal path is automatic
    resolution at extraction time; this endpoint exists for operator
    bootstrapping and corrections."""
    from datetime import datetime

    from particles.store.subject_store import insert_subject

    subject = Subject(
        canonical_name=req.canonical_name,
        description=req.description,
        aliases=req.aliases,
        external_ids=req.external_ids,
        created_at=datetime.now(UTC),
        asserted_by="api-user",
    )
    await insert_subject(session, subject)
    await session.commit()
    return subject


class SplitSubjectRequest(BaseModel):
    """Re-bind some particles off a source Subject onto a new one.

    Supply the new Subject's identity via ``new_external_id`` (authoritative —
    metadata pulled directly from the identifier, e.g. ``wikidata:Q30297735``)
    or ``new_name`` (canonicalised via the resolver). ``particle_ids`` lists the
    particles to move off the source."""

    particle_ids: list[str]
    new_name: str | None = None
    new_external_id: str | None = None


class SplitSubjectResponse(BaseModel):
    """Outcome of a split: the resolved/created target Subject plus which
    particles actually moved and which were skipped (not bound to the source)."""

    new_subject: Subject
    relinked_particle_ids: list[str]
    not_bound_particle_ids: list[str]


@app.post(
    "/subjects/{subject_id}/split",
    response_model=SplitSubjectResponse,
    responses=_ERR404,
)
async def split_subject_endpoint(
    subject_id: str, req: SplitSubjectRequest, session: WriteSessionDep, _auth: AuthDep
) -> SplitSubjectResponse:
    """Re-bind misjoined particles off a source Subject onto a new Subject the
    resolver canonicalises (HTTP mirror of `subjects split`). 404 if
    the source Subject is absent; 400 on an empty particle list or a bad target
    identity."""
    from particles.ingest.subject_resolver import split_subject_resolving
    from particles.store.subject_store import get_subject

    if not req.particle_ids:
        raise HTTPException(status_code=400, detail="particle_ids must be non-empty")
    if await get_subject(session, subject_id) is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    try:
        new_subject, relinked, not_bound = await split_subject_resolving(
            session,
            source_id=subject_id,
            particle_ids=req.particle_ids,
            new_name=req.new_name,
            new_external_id=req.new_external_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return SplitSubjectResponse(
        new_subject=new_subject,
        relinked_particle_ids=relinked,
        not_bound_particle_ids=not_bound,
    )


# ---------------------------------------------------------------------------
# Operator event log — read-only, mirrors the CLI `events` group
# ---------------------------------------------------------------------------


@app.get("/events", response_model=list[OperatorEvent])
async def list_events_endpoint(
    session: SessionDep,
    _auth: AuthDep,
    particle: str | None = None,
    subject: str | None = None,
    entry: str | None = None,
    type: str | None = None,
    limit: int | None = Query(None, ge=1),
) -> list[OperatorEvent]:
    """List operator events newest-first, optionally filtered.

    At most one of ``particle`` / ``subject`` / ``entry`` may be set (the
    record the event touched); ``type`` filters by event type.
    """
    from particles.store.event_store import (
        OperatorEventType,
        list_events,
        ref_filter,
    )

    try:
        ref_kind, ref_id = ref_filter(particle=particle, subject=subject, corpus_entry=entry)
    except ValueError as exc:
        # Handler-owned constraint message (not a str(exc) passthrough).
        raise HTTPException(
            status_code=400, detail="Provide at most one of particle / subject / entry"
        ) from exc
    etype: OperatorEventType | None = None
    if type is not None:
        try:
            etype = OperatorEventType(type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown event type {type!r}") from exc
    return await list_events(
        session,
        ref_kind=ref_kind,
        ref_id=ref_id,
        event_type=etype,
        limit=_bounded_limit(limit, default=50),
    )


@app.get("/events/{event_id}", response_model=OperatorEvent, responses=_ERR404)
async def get_event_endpoint(event_id: str, session: SessionDep, _auth: AuthDep) -> OperatorEvent:
    """Fetch one operator event by id. 404 if not present.

    Bearer-gated like ``GET /events`` (the operator audit trail; F2)."""
    from particles.store.event_store import get_event

    event = await get_event(session, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


# ---------------------------------------------------------------------------
# MCP read parity — thin GETs the routed MCP read tools need that
# no pre-existing endpoint covered. Declared BEFORE `/particles/{particle_id}`
# so the static `/particles/search` and `/particles/contested` paths win the
# route match over the `{particle_id}` capture. These reads carry `ReadAuthDep`
# — unauthenticated by default, but bearer-gated when `api.require_auth_for_reads`
# is set (security review F2). The bulk read surface (`/quality`, `/subjects`,
# `/corpus`, …) follows the same posture; `/query`, `/events`, and `/digest`
# carry the unconditional `AuthDep` instead (gated regardless of the flag).
# ---------------------------------------------------------------------------


@app.get("/particles", response_model=list[Particle])
async def list_particles_endpoint(
    session: SessionDep,
    _auth: ReadAuthDep,
    status: str | None = None,
    subject_id: str | None = None,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
) -> list[Particle]:
    """List particles, optionally filtered by status / subject (HTTP mirror of the
    MCP ``particles_list`` tool). Newest first; no embeddings.

    ``limit`` defaults to 50 and is clamped to ``storage.max_page_size`` (F5).
    400 on an unknown ``status``."""
    from particles.store.particle_store import list_particles_filtered

    status_enum: Status | None = None
    if status is not None:
        try:
            status_enum = Status(status)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in Status)
            raise HTTPException(
                status_code=400, detail=f"Unknown status {status!r}. Allowed: {allowed}."
            ) from exc
    return await list_particles_filtered(
        session,
        status=status_enum,
        subject_id=subject_id,
        limit=_bounded_limit(limit, default=50),
        offset=offset,
    )


@app.get("/particles/search", response_model=list[Particle])
async def search_particles_by_fingerprint(
    fingerprint: str,
    session: SessionDep,
    _auth: ReadAuthDep,
    limit: int | None = Query(None, ge=1),
) -> list[Particle]:
    """List particles sharing a context fingerprint (HTTP mirror of the
    MCP ``particle_search`` tool). ``fingerprint`` is a full 64-char SHA-256 hex
    or a prefix (≥ 8 chars; validated in-handler → 400). ``limit`` defaults to 50
    and is clamped to ``storage.max_page_size`` (F5)."""
    from sqlalchemy import select

    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern
    from particles.store.particle_store import ParticleRow

    fp = fingerprint.lower().strip()
    if len(fp) < 8:
        raise HTTPException(
            status_code=400, detail="Fingerprint prefix must be ≥ 8 hex characters."
        )
    if any(ch not in "0123456789abcdef" for ch in fp):
        raise HTTPException(status_code=400, detail="Fingerprint must be hexadecimal.")
    if len(fp) == 64:
        stmt = select(ParticleRow).where(ParticleRow.context_fingerprint == fp)
    else:
        stmt = select(ParticleRow).where(
            ParticleRow.context_fingerprint.like(f"{escape_like_pattern(fp)}%", escape=LIKE_ESCAPE)
        )
    result = await session.execute(stmt.limit(_bounded_limit(limit, default=50)))
    return [row.to_model() for row in result.scalars()]


@app.get("/particles/contested", response_model=dict[str, str])
async def particle_contested_backrefs(session: SessionDep, _auth: ReadAuthDep) -> dict[str, str]:
    """Map each ACTIVE belief an open INCONSISTENCY references → that INCONSISTENCY
    id. Backs the contested marker the routed MCP ``query`` /
    ``particles_list`` tools apply to recalled beliefs."""
    from particles.store.particle_store import get_inconsistency_backrefs

    return await get_inconsistency_backrefs(session)


class ContestedBadgesRequest(BaseModel):
    """The ids to compose badges for — POST because the list can be long."""

    particle_ids: list[str] = Field(default_factory=list)


@app.post("/particles/contested-badges", response_model=dict[str, ContestedBadge])
async def particle_contested_badges(
    body: ContestedBadgesRequest,
    session: SessionDep,
    _auth: ReadAuthDep,
) -> dict[str, ContestedBadge]:
    """Compose the contested badge for the requested ids.

    The one composer for every recall surface: the routed MCP
    ``particles_list`` tool used to hand-roll a one-basis badge of its own,
    which made the listing disagree with ``query`` on the same server. Keyed by
    id and **sparse** — an id no available basis fired on (or one that does not
    exist) is simply absent, exactly as ``/particles/contested`` omits an
    uncontested belief. 400 above ``storage.max_page_size`` ids (F5)."""
    from particles.operations.query.contested import compute_contested_badges
    from particles.store.particle_store import get_particles_by_ids

    cap = _bounded_limit(None)
    if len(body.particle_ids) > cap:
        raise HTTPException(status_code=400, detail=f"At most {cap} particle_ids per request.")
    loaded = list((await get_particles_by_ids(session, body.particle_ids)).values())
    badges = await compute_contested_badges(session, loaded)
    return {p.id: b for p, b in zip(loaded, badges, strict=True) if b is not None}


# ---------------------------------------------------------------------------
# Taxonomies — read-only; HTTP mirror of the MCP `list_taxonomies`.
# ---------------------------------------------------------------------------


@app.get("/taxonomies", response_model=list[TaxonomyDefinition])
async def list_taxonomies_endpoint(
    session: SessionDep,
    _auth: ReadAuthDep,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
) -> list[TaxonomyDefinition]:
    """List taxonomies + their full tag trees. ``limit``/``offset``
    paginate (each tree can be large); an omitted ``limit`` returns up to
    ``storage.max_page_size`` and any ``limit`` is clamped to it (F5)."""
    from particles.store.taxonomy_store import list_taxonomies

    return await list_taxonomies(session, limit=_bounded_limit(limit), offset=offset)


# ---------------------------------------------------------------------------
# Memory digest — the MCP `particles://digest/<store>` resource,
# rendered server-side so a routed client recalls the canonical store.
# ---------------------------------------------------------------------------


class DigestResponse(BaseModel):
    """The compiled session-start memory digest as one Markdown document."""

    markdown: str


@app.get("/digest/{store}", response_model=DigestResponse, responses=_ERR404)
async def get_digest_endpoint(store: str, _auth: AuthDep) -> DigestResponse:
    """Render the session-start memory digest for a store.

    Read-only, zero LLM/embeddings, rendered fresh. 404 if the store handle is
    not configured on this engine. ``build_digest`` manages its own per-store
    session, so this handler takes no ``SessionDep``.

    Bearer-gated regardless of ``api.require_auth_for_reads`` (F2): the digest
    is the provenance-ranked roll-up of the full belief store, including
    contested beliefs — confidential operator context. The dev-key loopback
    skip keeps it open for local development."""
    from particles.operations.digest import build_digest

    try:
        markdown = await build_digest(store)
    except (KeyError, ValueError) as exc:
        # db.py raises KeyError for an unknown store handle.
        raise HTTPException(status_code=404, detail=f"Unknown store {store!r}") from exc
    return DigestResponse(markdown=markdown)


# ---------------------------------------------------------------------------
# Operator-verb HTTP parity — thin mirrors of CLI write verbs.
# These expose the operator-judgment write surface (particle read + tags, links,
# subject alias / merge, trust rules / statements) over HTTP so they are
# reachable against the remote engine (fires trigger). Each is a thin
# mirror of one store / operation function. The particle belief-write trio
# (assert / supersede / retract) is mirrored further down — it
# wraps the §6.6 reconciliation ladder via particles.operations.agent_write.
# ---------------------------------------------------------------------------


@app.get("/particles/{particle_id}", response_model=Particle, responses=_ERR404)
async def get_particle_endpoint(
    particle_id: str, session: SessionDep, _auth: ReadAuthDep
) -> Particle:
    """Fetch a single particle by ID (HTTP mirror of `particle show`). 404 if absent."""
    from particles.store.particle_store import get_particle

    p = await get_particle(session, particle_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Particle not found")
    return p


class ParticleTagsRequest(BaseModel):
    """Tags to add to / remove from a particle."""

    tags: list[str]


class ParticleTagsResponse(BaseModel):
    """The particle and the tags actually added / removed (idempotent ⇒ may be fewer)."""

    particle_id: str
    tags: list[str]


@app.post("/particles/{particle_id}/tags", response_model=ParticleTagsResponse)
async def add_particle_tags_endpoint(
    particle_id: str, req: ParticleTagsRequest, session: WriteSessionDep, _auth: AuthDep
) -> ParticleTagsResponse:
    """Add tags to a particle (HTTP mirror of `particle tag`). Idempotent."""
    from particles.store.taxonomy_store import add_particle_tags

    added = await add_particle_tags(session, particle_id, req.tags)
    await session.commit()
    return ParticleTagsResponse(particle_id=particle_id, tags=added)


@app.delete("/particles/{particle_id}/tags", response_model=ParticleTagsResponse)
async def remove_particle_tags_endpoint(
    particle_id: str, req: ParticleTagsRequest, session: WriteSessionDep, _auth: AuthDep
) -> ParticleTagsResponse:
    """Remove tags from a particle (HTTP mirror of `particle untag`). Idempotent."""
    from particles.store.taxonomy_store import remove_particle_tags

    removed = await remove_particle_tags(session, particle_id, req.tags)
    await session.commit()
    return ParticleTagsResponse(particle_id=particle_id, tags=removed)


class LinkRequest(BaseModel):
    """A typed relation between two particles. ``relation_type`` is a
    registry kind (e.g. ``CO_EVIDENTIAL``)."""

    particle_a: str
    particle_b: str
    relation_type: str = "CO_EVIDENTIAL"
    confidence: float = 1.0


class LinkDeleteResponse(BaseModel):
    deleted: bool


@app.post("/links", response_model=ParticleRelation, responses=_ERR400)
async def create_link_endpoint(
    req: LinkRequest, session: WriteSessionDep, _auth: AuthDep
) -> ParticleRelation:
    """Create a typed relation between two particles (HTTP mirror of `links add`)."""
    from particles.core.schema import RelationCreatedBy, RelationType
    from particles.store.relation_store import create_relation

    try:
        rtype = RelationType(req.relation_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown relation_type {req.relation_type!r}"
        ) from exc
    try:
        relation = await create_relation(
            session,
            req.particle_a,
            req.particle_b,
            rtype,
            RelationCreatedBy.MANUAL_CLI,
            confidence=req.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return relation


@app.delete("/links", response_model=LinkDeleteResponse, responses=_ERR400)
async def delete_link_endpoint(
    req: LinkRequest, session: WriteSessionDep, _auth: AuthDep
) -> LinkDeleteResponse:
    """Delete a typed relation between two particles (HTTP mirror of `links remove`)."""
    from particles.core.schema import RelationType
    from particles.store.relation_store import delete_relation

    try:
        rtype = RelationType(req.relation_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown relation_type {req.relation_type!r}"
        ) from exc
    deleted = await delete_relation(session, req.particle_a, req.particle_b, rtype)
    await session.commit()
    return LinkDeleteResponse(deleted=deleted)


class SubjectAliasRequest(BaseModel):
    aliases: list[str]


class SubjectAliasResponse(BaseModel):
    """The updated subject and the alias names actually added (idempotent)."""

    subject: Subject
    added: list[str]


@app.post(
    "/subjects/{subject_id}/aliases",
    response_model=SubjectAliasResponse,
    responses=_ERR404,
)
async def add_subject_aliases_endpoint(
    subject_id: str, req: SubjectAliasRequest, session: WriteSessionDep, _auth: AuthDep
) -> SubjectAliasResponse:
    """Append aliases to a subject (HTTP mirror of `subjects alias`). 404 if absent."""
    from particles.store.subject_store import add_aliases, get_subject

    if await get_subject(session, subject_id) is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    subject, added = await add_aliases(session, subject_id, req.aliases)
    await session.commit()
    return SubjectAliasResponse(subject=subject, added=added)


class SubjectMergeRequest(BaseModel):
    target_id: str


class SubjectMergeResponse(BaseModel):
    """Outcome of a merge: the surviving target, the aliases it absorbed, and how
    many particle links were re-pointed. The source subject is deleted."""

    subject: Subject
    aliases_added: list[str]
    particles_relinked: int


@app.post(
    "/subjects/{source_id}/merge",
    response_model=SubjectMergeResponse,
    responses={**_ERR404, **_ERR400},
)
async def merge_subjects_endpoint(
    source_id: str, req: SubjectMergeRequest, session: WriteSessionDep, _auth: AuthDep
) -> SubjectMergeResponse:
    """Merge a source subject into a target (HTTP mirror of `subjects merge`).

    Irreversible: the source is deleted and its particles + aliases fold into the
    target. 404 if either subject is absent; 400 on an invalid merge."""
    from particles.store.subject_store import get_subject, merge_subjects

    if await get_subject(session, source_id) is None:
        raise HTTPException(status_code=404, detail="Source subject not found")
    if await get_subject(session, req.target_id) is None:
        raise HTTPException(status_code=404, detail="Target subject not found")
    try:
        target, aliases_added, relinked = await merge_subjects(session, source_id, req.target_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return SubjectMergeResponse(
        subject=target, aliases_added=aliases_added, particles_relinked=relinked
    )


class TrustRuleRequest(BaseModel):
    """A source-trust rule. ``scope`` is ``"domain"`` (use ``score``)
    or ``"url_pattern"`` (use ``modifier``)."""

    scope: str
    pattern: str
    score: float | None = None
    modifier: float | None = None
    rationale: str | None = None


@app.post("/trust/rules", response_model=StatusResponse, responses=_ERR400)
async def set_trust_rule_endpoint(
    req: TrustRuleRequest, session: WriteSessionDep, _auth: AuthDep
) -> StatusResponse:
    """Add or update a source-trust rule (HTTP mirror of `trust set`)."""
    from particles.store.trust_store import upsert_trust_rule

    if req.scope not in ("domain", "url_pattern"):
        raise HTTPException(status_code=400, detail="scope must be 'domain' or 'url_pattern'")
    await upsert_trust_rule(
        session,
        scope=req.scope,
        pattern=req.pattern,
        score=req.score,
        modifier=req.modifier,
        rationale=req.rationale,
    )
    await session.commit()
    return StatusResponse(status="trust rule saved")


class TrustStatementRequest(BaseModel):
    """An OPERATOR_DIRECT SourceTrustStatement (HTTP mirror of `trust statement-set`).

    ``source_ref_type`` is ``CORPUS_ENTRY`` / ``SOURCE_TYPE`` / ``AUTHOR``."""

    domain: str
    source_ref_type: str
    source_ref_value: str
    trust_rank: float
    basis: str | None = None


class TrustStatementResponse(BaseModel):
    """Number of INCONSISTENCY particles the statement's cascade resolved."""

    cascade_resolved: int


@app.post("/trust/statements", response_model=TrustStatementResponse, responses=_ERR400)
async def set_trust_statement_endpoint(
    req: TrustStatementRequest, session: WriteSessionDep, _auth: AuthDep
) -> TrustStatementResponse:
    """Write an OPERATOR_DIRECT SourceTrustStatement and run its cascade."""
    from particles.core.schema import (
        PolicyProvenance,
        SourceRef,
        SourceRefType,
        SourceTrustStatement,
    )
    from particles.operations.trust import set_trust_statement

    if not 0.0 <= req.trust_rank <= 1.0:
        raise HTTPException(status_code=400, detail="trust_rank must be in [0.0, 1.0]")
    try:
        ref_type = SourceRefType(req.source_ref_type.upper())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="source_ref_type must be CORPUS_ENTRY, SOURCE_TYPE, or AUTHOR",
        ) from exc
    stmt = SourceTrustStatement(
        domain=req.domain,
        source_ref=SourceRef(type=ref_type, value=req.source_ref_value),
        trust_rank=req.trust_rank,
        policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
        asserted_by="api-user",
        basis=req.basis,
    )
    resolved = await set_trust_statement(session, stmt, actor="api-trust-statement")
    await session.commit()
    return TrustStatementResponse(cascade_resolved=resolved)


# ---------------------------------------------------------------------------
# Particle belief writes — assert / supersede / retract over HTTP.
# Not thin mirrors: each wraps the §6.6 reconciliation ladder, server-side field
# construction (§4a), contributor identity, and event logging — all in
# particles.operations.agent_write, the single convergence point the local MCP
# write tools also reach. Resolves circular deferral (belief-writes
# were "already remoteable via MCP", but MCP covered them only locally).
#
# Write-enablement is server-authoritative: the engine gates these
# on its own `mcp.write.enabled_stores` for its canonical (default) store,
# independent of the local server's offering gate.
# ---------------------------------------------------------------------------


def _require_belief_writes_enabled() -> str:
    """Engine-side write-enablement gate; returns the canonical store handle.

    The belief-write endpoints operate on the engine's default store (the MVP
    canonical archive; multi-store routing is deferred). 403 if that
    store is not in the engine's ``mcp.write.enabled_stores``."""
    if DEFAULT_STORE not in get_config().mcp.write.enabled_stores:
        raise HTTPException(
            status_code=403,
            detail=(
                "Belief writes are disabled on this engine: the default store is not "
                "in mcp.write.enabled_stores (default-deny)."
            ),
        )
    return DEFAULT_STORE


class ParticleAssertRequest(BaseModel):
    """Assert one belief (flagship). Trust-/status-/identity-bearing
    fields are constructed server-side (§4a) — only the claim, its subjects,
    self-reported confidence, and provenance are caller-supplied."""

    content: str
    subject_names: list[str]
    confidence: float
    source_excerpt: str | None = None
    corpus_entry_id: str | None = None
    uncertainty_nature: str = "EPISTEMIC"
    tags: list[str] | None = None


class ParticleSupersedeRequest(ParticleAssertRequest):
    """Revise a belief: ``supersedes_id`` is retired to SUPERSEDED, then a
    successor is asserted."""

    supersedes_id: str


class ParticleRetractRequest(BaseModel):
    """Retract a belief: ACTIVE → RETRACTED. ``reason`` is recorded on
    the audit event."""

    particle_id: str
    reason: str


class ParticleRetractEndpointResponse(BaseModel):
    """Outcome of a belief retraction."""

    particle_id: str
    verdict: str


@app.post("/particles/assert", response_model=AgentWriteResult, responses=_ERR400)
async def particle_assert_endpoint(
    req: ParticleAssertRequest, session: SessionDep, _auth: AuthDep
) -> AgentWriteResult:
    """Assert a belief through the §6.6 ladder (HTTP mirror of MCP ``particle_assert``).

    A confirmed contradiction returns ``verdict=INCONSISTENCY_RAISED`` (consensus
    mode) — a first-class result, not an error. 400 on a granularity / provenance
    violation; 403 when belief writes are disabled."""
    store = _require_belief_writes_enabled()
    try:
        result = await assert_belief(
            session,
            store=store,
            content=req.content,
            subject_names=req.subject_names,
            confidence=req.confidence,
            source_excerpt=req.source_excerpt,
            corpus_entry_id=req.corpus_entry_id,
            uncertainty_nature=req.uncertainty_nature,
            tags=req.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return result


@app.post("/particles/supersede", response_model=AgentWriteResult, responses=_ERR400)
async def particle_supersede_endpoint(
    req: ParticleSupersedeRequest, session: SessionDep, _auth: AuthDep
) -> AgentWriteResult:
    """Revise a belief (HTTP mirror of MCP ``particle_supersede``). Own-beliefs-only;
    operator (HUMAN_REVIEW) particles are not agent-mutable. 400 on a guard or
    granularity violation; 403 when belief writes are disabled."""
    store = _require_belief_writes_enabled()
    try:
        result = await supersede_belief(
            session,
            store=store,
            supersedes_id=req.supersedes_id,
            content=req.content,
            subject_names=req.subject_names,
            confidence=req.confidence,
            source_excerpt=req.source_excerpt,
            corpus_entry_id=req.corpus_entry_id,
            uncertainty_nature=req.uncertainty_nature,
            tags=req.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return result


@app.post("/particles/retract", response_model=ParticleRetractEndpointResponse, responses=_ERR400)
async def particle_retract_endpoint(
    req: ParticleRetractRequest, session: SessionDep, _auth: AuthDep
) -> ParticleRetractEndpointResponse:
    """Retract a belief (HTTP mirror of MCP ``particle_retract``). Own-beliefs-only;
    operator (HUMAN_REVIEW) particles are not agent-mutable. 400 on a guard
    violation or empty reason; 403 when belief writes are disabled."""
    store = _require_belief_writes_enabled()
    try:
        await retract_belief(session, store=store, particle_id=req.particle_id, reason=req.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return ParticleRetractEndpointResponse(particle_id=req.particle_id, verdict="RETRACTED")


# ---------------------------------------------------------------------------
# Operator-scoped belief mutation — supersede / retract a belief
# the operator does NOT own (incl. extracted beliefs), recorded as operator
# events. This is the curation write surface a bus-stop queue actually needs: a
# queue is overwhelmingly extracted beliefs, which the own-beliefs-only agent
# path (above) cannot touch.
#
# SECURITY: the operator path can mutate ANY belief, so it MUST stay gated by
# `mcp.write.enabled_stores` + the bearer exactly like the agent
# belief-writes — `_require_belief_writes_enabled()` enforces the store gate and
# the route's `AuthDep` the bearer. It NEVER widens the dev-key loopback skip.
# The agent_write operator path skips only the ownership check; the HUMAN_REVIEW
# guard and ACTIVE-status check still apply.
# ---------------------------------------------------------------------------


class OperatorSupersedeRequest(BaseModel):
    """Operator-supersede a belief by path id. The predecessor is
    retired to SUPERSEDED and a successor is asserted with the operator's revised
    content + the confidence the operator sets (operator-attributed standard
    supersession). For a content-unchanged subject-assign use
    ``POST /particles/{id}/subjects`` (provenance-preserving)."""

    content: str
    subject_names: list[str]
    confidence: float
    source_excerpt: str | None = None
    corpus_entry_id: str | None = None
    uncertainty_nature: str = "EPISTEMIC"
    tags: list[str] | None = None


class OperatorRetractRequest(BaseModel):
    """Operator-retract a belief by path id. ``reason`` is recorded
    on the audit event."""

    reason: str


@app.post("/particles/{particle_id}/supersede", response_model=AgentWriteResult, responses=_ERR400)
async def operator_supersede_endpoint(
    particle_id: str, req: OperatorSupersedeRequest, session: SessionDep, _auth: AuthDep
) -> AgentWriteResult:
    """Operator-supersede any belief, incl. an extracted one.

    The "edit" gesture finally working on the extracted beliefs that fill a
    curation queue. Skips only the ownership check; the HUMAN_REVIEW guard and
    ACTIVE-status check still apply. 400 on a guard / granularity violation; 403
    when curation writes are disabled."""
    store = _require_belief_writes_enabled()
    try:
        result = await supersede_belief(
            session,
            store=store,
            supersedes_id=particle_id,
            content=req.content,
            subject_names=req.subject_names,
            confidence=req.confidence,
            source_excerpt=req.source_excerpt,
            corpus_entry_id=req.corpus_entry_id,
            uncertainty_nature=req.uncertainty_nature,
            tags=req.tags,
            operator=True,
            actor="http:/particles/{id}/supersede",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return result


@app.post(
    "/particles/{particle_id}/retract",
    response_model=ParticleRetractEndpointResponse,
    responses=_ERR400,
)
async def operator_retract_endpoint(
    particle_id: str, req: OperatorRetractRequest, session: SessionDep, _auth: AuthDep
) -> ParticleRetractEndpointResponse:
    """Operator-retract any belief, incl. an extracted one.

    Retract one extraction-/operator-sourced belief by id — the per-particle
    retract a curation card needs (vs the blunt per-source
    ``POST /corpus/{id}/retract``). Skips only the ownership check; the
    HUMAN_REVIEW guard and ACTIVE-status check still apply. 400 on a guard
    violation or empty reason; 403 when curation writes are disabled."""
    store = _require_belief_writes_enabled()
    try:
        await retract_belief(
            session,
            store=store,
            particle_id=particle_id,
            reason=req.reason,
            operator=True,
            actor="http:/particles/{id}/retract",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return ParticleRetractEndpointResponse(particle_id=particle_id, verdict="RETRACTED")


# ---------------------------------------------------------------------------
# Per-particle subject assignment — attach a subject to a NO_SUBJECT
# orphan via a provenance-preserving operator-supersede. The successor copies the
# predecessor's confidence record + extractor_ref + source verbatim (the same
# extracted claim, only the subject linkage corrected); the operator's act is the
# recorded PARTICLE_SUPERSEDED event. Same write gate as the other belief-writes.
# ---------------------------------------------------------------------------


class ParticleSubjectAssignRequest(BaseModel):
    """Assign a subject to an orphan particle. Resolution accepts an
    explicit ``subject_id`` (link a Subject the operator picked) OR a
    ``subject_name`` run through the standard resolver (local → Wikidata →
    bare-local) — provide exactly one."""

    subject_id: str | None = None
    subject_name: str | None = None


@app.post("/particles/{particle_id}/subjects", response_model=AgentWriteResult, responses=_ERR400)
async def assign_subject_endpoint(
    particle_id: str, req: ParticleSubjectAssignRequest, session: SessionDep, _auth: AuthDep
) -> AgentWriteResult:
    """Attach a subject to a NO_SUBJECT orphan (provenance-preserving).

    Supersedes the orphan with a successor that has the same content + the
    resolved subject(s), carrying over the predecessor's confidence record,
    extractor_ref, and source — it is the same extracted claim with a corrected
    linkage. 400 on a bad resolution / guard violation; 403 when writes are
    disabled."""
    store = _require_belief_writes_enabled()
    try:
        result = await assign_subject_belief(
            session,
            store=store,
            particle_id=particle_id,
            subject_id=req.subject_id,
            subject_name=req.subject_name,
            actor="http:/particles/{id}/subjects",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    return result


# ---------------------------------------------------------------------------
# DB init helper
# ---------------------------------------------------------------------------


@app.post("/db/init", response_model=StatusResponse)
async def db_init(_auth: AuthDep) -> StatusResponse:
    """Idempotently create database tables. Safe to call on a
    fully-initialised store; existing tables are not touched."""
    await create_tables()
    return StatusResponse(status="tables created")


# ---------------------------------------------------------------------------
# Unified web UI — same-origin static serving.
#
# Mount the built web-UI bundle (`clients/web-ui/dist`) at `/app`, behind the
# auth gate (AuthStaticFiles), so the app's authenticated `fetch`
# calls to `/curation`, `/query`, `/graph`, … are same-origin and trigger no
# CORS preflight (no CORS middleware is added; the standalone-host
# variant is deferred). The bundle is built separately and is NOT in
# the wheel, so the mount degrades gracefully when `dist/` is absent: the factory
# returns None and the route is simply not registered (the API is unaffected).
# Mounted last so the `/app` prefix never shadows an API route. A pure static
# mount is a Starlette sub-app, not a path operation, so it does not appear in
# the OpenAPI contract surface and leaves the snapshot untouched.
# ---------------------------------------------------------------------------

_web_ui_static_app = build_web_ui_static_app()
if _web_ui_static_app is not None:
    app.mount(WEB_APP_MOUNT, _web_ui_static_app, name="web-ui")

    # Convenience only, and only when the UI is actually mounted: a browser
    # landing on the engine root gets the app instead of a bare 404 JSON.
    # Excluded from the OpenAPI snapshot — this is navigation sugar, not a
    # contract capability (the parity rules are unaffected); an
    # API-only install (no dist/) keeps its plain 404 at the root.
    @app.get("/", include_in_schema=False)
    async def _root_to_web_ui() -> RedirectResponse:
        return RedirectResponse(url=f"{WEB_APP_MOUNT}/")
