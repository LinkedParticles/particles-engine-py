# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application — Core endpoints (§C.6)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlsplit

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, BeforeValidator, Field
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import ASGIApp, Receive, Scope, Send

from particles import __version__
from particles.api import _keys
from particles.api._middleware import BodySizeLimitMiddleware, HostBoundMiddleware
from particles.api._rate_limit import ModelCallMeterMiddleware, RateLimitDep, enforce_rate_limit
from particles.api.auth import (
    AGENT_GATE,
    AUTH_GATE,
    OPERATOR_GATE,
    READ_GATE,
    Capability,
    Principal,
    current_principal,
    enforce_fail_closed_on_startup,
    enforce_key_material_on_startup,
    request_holds_a_key,
    require_operator_credential,
    startup_open_reads_warning,
    warn_if_dev_auth_in_use,
)
from particles.api.daemon import DaemonStatus, daemon_status, start_daemon, stop_daemon
from particles.api.web_app import WEB_APP_MOUNT, build_web_ui_static_app
from particles.config import get_config
from particles.core.observer_scope import (
    PROJECT_TAG_PREFIX,
    UNATTRIBUTED_MARKER,
    EntryScope,
    canonical_project_key,
    classify_entry,
    hides_a_character,
    lookalike_reason,
    names_a_project,
    project_keys,
    project_tag,
)
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
    VocabularyReport,
)
from particles.core.status import Status
from particles.corpus.deposit import (
    ForeignEntryRefused,
    own_entries_only_for,
    resolve_upload_source_type,
)
from particles.corpus.harvest_marks import reserved_marks
from particles.corpus.rule_sources import RULE_SOURCE_TAG
from particles.db import (
    DEFAULT_STORE,
    WriteLockTimeout,
    create_tables,
    get_session,
    get_write_session,
    session_scope,
    write_transaction,
)
from particles.egress import EgressRefused, ensure_backstop, require_covered_loop
from particles.extraction.mcp_memory import SOURCE_TYPE as MIGRATION_SOURCE_TYPE
from particles.http import SourceFetchError
from particles.operations._llm import llm_circuit_open
from particles.operations.agent_write import (
    AgentWriteResult,
    assert_belief,
    assign_subject_belief,
    check_belief_body,
    check_operator_supersede,
    deposit_conversation_text,
    link_beliefs,
    require_project_binding,
    retract_belief,
    supersede_belief,
    tag_belief,
    unlink_beliefs,
    untag_belief,
)
from particles.operations.curation import (
    CurationCard,
    QueueSource,
    build_curation_queue,
    dismiss_source_card,
    get_curation_card,
    rebuild_curation_snapshot,
    record_demotion_ruling,
)
from particles.operations.curation.cards import CardKind
from particles.operations.deposit import (
    deposit_file,
    deposit_text,
    deposit_text_versioned,
    deposit_url,
    writes_committed_under_lock,
)
from particles.operations.deposit_suggest import DepositSuggestReport
from particles.operations.digest import DigestLine
from particles.operations.extract import extract_snapshot
from particles.operations.graph_view import build_graph_data
from particles.operations.lint import run_lint
from particles.operations.llm_spend import MeteredExtractRun, reindexed_snapshots
from particles.operations.projection import DocManifest
from particles.operations.quality import get_quality_report
from particles.operations.query import query
from particles.operations.reconcile import reconcile_supersession
from particles.operations.reindex import ReindexPlan, reindex
from particles.operations.review import ResolutionRefused, list_inconsistencies, resolve
from particles.operations.source_passage import SourcePassage, hydrate_source_passage
from particles.operations.subject_relink import (
    RelinkReport,
    apply_gated_relink,
    build_report,
    plan_gated_relink,
)
from particles.store.event_store import HTTP_KEY_ID_FIELD, OperatorEvent
from particles.store.session_exposure_store import SessionExposure
from particles.url_safety import is_network_uri

log = logging.getLogger(__name__)


#: The config the serving checks last passed under. ``None`` until they first
#: pass; a reload (a new config object) runs them again.
_serving_checks_passed_for: object | None = None


def ensure_serving_checks() -> None:
    """Run the checks that must hold before the engine serves, once per config.

    The lifespan runs them at startup, where a failure aborts it. A server run
    with its lifespan off (``uvicorn --lifespan off``) never enters the
    lifespan, so :class:`ServingChecksMiddleware` runs them on the first
    request too, and answers 503 for as long as they fail (security review
    2026-10-25, Q10 and Q46). Raises what the failing check raises.
    """
    global _serving_checks_passed_for
    config = get_config()
    if _serving_checks_passed_for is config:
        return
    # Fail closed before anything else: if bearer auth is disabled and the
    # bind host is not loopback, refuse to start rather than serve
    # unauthenticated traffic beyond loopback. Raising here aborts startup.
    enforce_fail_closed_on_startup()
    # Refuse key material the engine cannot use as intended.
    enforce_key_material_on_startup()
    warn_if_dev_auth_in_use()
    reads_warning = startup_open_reads_warning()
    if reads_warning is not None:
        log.warning("%s", reads_warning)
    # Arm the no-egress backstop before anything can connect.
    ensure_backstop()
    # The backstop cannot see uvloop's sockets; a raw `uvicorn` launch picks
    # uvloop when it is installed, so a refusing posture refuses to start on it.
    require_covered_loop()
    _serving_checks_passed_for = config


class ServingChecksMiddleware:
    """Serve nothing until :func:`ensure_serving_checks` has passed."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            try:
                ensure_serving_checks()
            except Exception as exc:  # noqa: BLE001 - every failing check refuses alike
                log.error("engine refused a request: a serving check failed: %s", exc)
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1011})
                    return
                await JSONResponse(
                    {"detail": "the engine is not serving: a startup check failed (see its log)"},
                    status_code=503,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan (FastAPI's startup/shutdown successor).

    Replaces the deprecated ``@app.on_event("startup")`` hook. Startup work
    runs before ``yield``; the shutdown half cancels the daemon tasks
    (a no-op unless daemon mode is on).
    """
    ensure_serving_checks()
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
    description="Python SDK for the Particles epistemic knowledge standard: Core Loop.",
    lifespan=lifespan,
    # No /docs or /redoc: FastAPI's pages load floating-version scripts from a
    # CDN onto the engine's origin, where the web UI keeps its bearer (security
    # review 2026-10-24, N7). /openapi.json stays; the docs site renders it.
    docs_url=None,
    redoc_url=None,
)

# Defense-in-depth: cap the request body before any handler runs (F-7).
# The limit is read per-request from config, so this single registration
# honours reloads and the PARTICLES_MAX_REQUEST_BODY_BYTES override.
app.add_middleware(BodySizeLimitMiddleware)
# A request without the bearer has each model call charged to its rate-limit
# bucket and writes no probe verdict (security review 2026-10-25-2, R2 and R5).
app.add_middleware(ModelCallMeterMiddleware)
# Outermost (added last): refuse a disallowed Host, or a non-loopback client under
# the dev-key skip, on every path before the body is read (security review
# 2026-10-24, N1). The route gates repeat it; this covers /health and /app.
app.add_middleware(HostBoundMiddleware)
# Outermost of all: nothing is served before the startup checks have passed,
# including under `uvicorn --lifespan off` (security review 2026-10-25, Q10).
app.add_middleware(ServingChecksMiddleware)


@app.middleware("http")
async def _log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Interim request observability ahead of the OpenTelemetry work.

    uvicorn's access log only fires on *response*, so a request that hangs
    (e.g. blocked on a SQLite write lock held by a concurrent writer)
    leaves no trace. Logging arrival makes a stuck request visible,
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


@app.exception_handler(EgressRefused)
async def _egress_refused_handler(_request: Request, exc: EgressRefused) -> Response:
    """Translate a no-egress refusal into a 403 that names the destination.

    Distinct from the SSRF 400 and the upstream 502: the request was well-formed
    and the origin may be up, but this engine's ``network.egress`` posture does
    not allow the connection.
    """
    from fastapi.responses import JSONResponse

    log.info("egress refused: %s", exc)
    return JSONResponse(
        status_code=403,
        content={
            "detail": str(exc),
            "error": "egress_refused",
            "host": exc.host,
            "port": exc.port,
        },
    )


SessionDep = Annotated[AsyncSession, Depends(get_session)]
#: A session held under the cross-process writer lock — for short,
#: pure-write endpoints (the handler commits before returning, no slow I/O before
#: the write). ``/extract`` / ``/reindex`` keep ``SessionDep``: the extract
#: pipeline acquires the lock around its own write phase, never across the LLM.
WriteSessionDep = Annotated[AsyncSession, Depends(get_write_session)]


def _store_handle(store: Annotated[str, Query(min_length=1)] = DEFAULT_STORE) -> str:
    """The store a routed write names: ``default`` unless the caller names one.

    The thin client sends the handle its verb was given, so a hook installed on
    a named store writes where its digest reads. 404 on
    a handle this engine does not configure, as ``GET /digest/{store}`` does.
    """
    if store != DEFAULT_STORE and store not in get_config().storage.stores:
        raise HTTPException(status_code=404, detail=f"Unknown store {store!r}")
    return store


StoreDep = Annotated[str, Depends(_store_handle)]


async def _store_session(store: StoreDep) -> AsyncIterator[AsyncSession]:
    async with session_scope(store) as session:
        yield session


async def _store_write_session(store: StoreDep) -> AsyncIterator[AsyncSession]:
    async with session_scope(store, write=True) as session:
        yield session


async def _belief_write_session(store: StoreDep) -> AsyncIterator[AsyncSession]:
    """A write session on a store that takes belief writes, checked before the lock.

    A refused write on a store outside ``mcp.write.enabled_stores`` answers 403
    without taking the store's writer lock (security review 2026-10-27-1, G36).
    """
    _require_belief_writes_enabled(store)
    async with session_scope(store, write=True) as session:
        yield session


#: ``SessionDep`` / ``WriteSessionDep`` bound to the store the ``store`` query
#: parameter names, for the writes the thin client routes by handle.
StoreSessionDep = Annotated[AsyncSession, Depends(_store_session)]
StoreWriteSessionDep = Annotated[AsyncSession, Depends(_store_write_session)]
#: ``StoreWriteSessionDep`` for a belief write: the write-enable 403 comes first.
BeliefWriteSessionDep = Annotated[AsyncSession, Depends(_belief_write_session)]


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
    """Liveness probe response, returned only when the server is healthy."""

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
    alike: the missing-blob list must survive into the envelope because
    the CLI's human rendering caps it). ``skipped_newer_schema`` counts the
    snapshots left alone because they hold a particle a newer schema minor
    wrote, which this SDK does not re-extract (D2)."""

    scope: int
    succeeded: int
    failed: int
    failed_entries: list[str] = []
    lint_summary: dict[str, int] = {}
    dry_run: bool = False
    skipped_newer_schema: int = 0
    plan: ReindexPlan | None = None


class ReconcileResponse(BaseModel):
    """Summary of a document-supersession reconcile sweep.

    ``scope_pairs`` is the number of corpus-entry pairs in a (transitive)
    document-supersession relation; ``candidate_pairs`` the particle pairs that
    cleared the similarity floor; ``probed`` the replacement-signal probes run;
    ``demoted`` the count transitioned ACTIVE → PROVENANCE_STALE /
    DOCUMENT_SUPERSEDED. ``demotions`` is the per-demotion audit list (winner /
    loser / entries / similarity). ``enabled`` / ``single_trust_order`` echo the
    v1 gates; a ``False`` on either means the sweep was a no-op."""

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


def _blank_is_omitted(value: object) -> object:
    """Read a string with no visible character (empty, whitespace, zero-width) as omitted.

    A key that names a project is stored in NFC, and one that hides an
    invisible character among visible ones is refused (422): it would render
    exactly as another project (security review 2026-10-28-1, O46).
    """
    if not isinstance(value, str):
        return value
    if not names_a_project(value):
        return None
    reason = lookalike_reason(value)
    if reason is not None:
        raise ValueError(f"project_key {reason}")
    return canonical_project_key(value)


#: The project an agent surface writes as. An empty key names no
#: project, so it is read as omitted and meets the rescoped-store binding
#: check like one (security review 2026-10-25-1, Q54); so does a key of only
#: invisible characters (security review 2026-10-26-1, E9).
ProjectKey = Annotated[str | None, BeforeValidator(_blank_is_omitted)]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(holds_key: Annotated[bool, Depends(request_holds_a_key)]) -> HealthResponse:
    """Liveness check. Returns ``status="ok"``, the SDK version, and the build date.

    In daemon mode the response also carries the background tasks'
    state, and ``status`` becomes ``"degraded"`` once any of them has crashed.
    The process is still serving requests, so this stays a 200 (the API must
    not take itself down because a scheduled tick died), but the disclosure is
    machine-readable for an operator alert or a readiness gate.

    ``built_at`` is present only when the artifact stamped itself (the
    container image does; a source checkout does not), so the route stays
    unauthenticated with nothing new behind it: the version was already
    disclosed here, and a build timestamp says no more about the deployment
    than the version it accompanies.

    With ``api.require_auth_for_reads`` on, the daemon's per-task detail (runs,
    failures, each task's last outcome) is served only to a request whose key
    authenticates; a caller without one still gets ``status``, so a readiness
    gate keeps working (security review 2026-10-28-2, AA50).
    """
    built_at = get_config().build.date
    daemon = daemon_status()
    if not daemon.enabled:
        return HealthResponse(status="ok", version=__version__, built_at=built_at)
    if get_config().api.require_auth_for_reads and not holds_key:
        return HealthResponse(
            status="ok" if daemon.healthy else "degraded",
            version=__version__,
            built_at=built_at,
        )
    return HealthResponse(
        status="ok" if daemon.healthy else "degraded",
        version=__version__,
        built_at=built_at,
        daemon=daemon,
    )


@app.get("/quality", dependencies=[READ_GATE], response_model=QualityReport)
async def quality_dashboard(session: SessionDep) -> QualityReport:
    """Extraction quality dashboard: calibration distribution, corpus status, subject coverage."""
    return await get_quality_report(session)


@app.get("/vocabulary", dependencies=[READ_GATE], response_model=VocabularyReport)
async def vocabulary_report_route(
    session: SessionDep,
    observer_project: Annotated[str | None, Query(min_length=1)] = None,
    include_document_meta: bool = False,
    include_non_asserted: bool = False,
) -> VocabularyReport:
    """The read-time vocabulary report: the evidence an ontology is built from.

    Each canonical predicate with its surface forms, claim count, object value
    shapes, the subject classes it attaches to, and its alignment, under a
    store header of subject alignment by namespace and link-confidence band,
    class counts, and the confirmed modelling decisions in the
    operator log.
    Computed per request and never stored; deterministic, with no embedding
    and no LLM call. ``observer_project`` reads the claims through a project
    observer; the subject header and the decision counts are
    store-wide. The same report rides ``POST /query`` with ``list_vocabulary``.
    """
    response = await query(
        session,
        QueryRequest(
            list_vocabulary=True,
            observer_project=observer_project,
            include_document_meta=include_document_meta,
            include_non_asserted=include_non_asserted,
        ),
    )
    if response.vocabulary_report is None:  # set on every list_vocabulary path
        raise RuntimeError("list_vocabulary returned no vocabulary report")
    return response.vocabulary_report


# ---------------------------------------------------------------------------
# Curation — the curation queue over HTTP. Exposes the
# existing `build_curation_queue` Engine operation so the deferred mobile / web
# client can render the leverage-ranked "today's N" feed.
# ---------------------------------------------------------------------------


class CurationQueueResponse(BaseModel):
    """The leverage-ranked, finite, snooze-filtered curation queue.

    ``cards`` is the ranked batch (highest leverage first), each a normalized
    projection of an existing read diagnostic. ``count`` is ``len(cards)`` for
    convenience; ``open_count`` is the backlog the batch was cut from. The HTTP
    mirror of `particles curate`.

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
    # How many cards are open in all (after the kind filter and suppression),
    # of which ``cards`` is the leverage-ranked head. Lets a client say
    # "7 of 154" instead of implying the capped slice is the whole backlog.
    open_count: int = 0
    # When each kind's finder last ran for the cards served. A ``carried`` kind
    # in ``per_kind_scope`` keeps the date of the build that last ran its
    # finder, so a client can say "contradictions as of the census of …".
    kind_as_of: dict[str, datetime] = Field(default_factory=dict)


@app.get("/curation", dependencies=[AUTH_GATE], response_model=CurationQueueResponse)
async def get_curation_queue(
    request: Request,
    session: SessionDep,
    limit: int | None = Query(None, ge=1),
    kind: str | None = None,
    semantic: bool | None = None,
    no_snapshot: bool = False,
) -> CurationQueueResponse:
    """Return the curation queue (HTTP mirror of `curate`).

    Composes the existing read diagnostics (lint, links/corpus-links suggest, the
    contested digest, quality) into one leverage-ranked, snooze-filtered worklist;
    no new detection logic. ``limit`` overrides ``curation.session_size`` for
    one call and is clamped to ``storage.max_page_size``;
    ``kind`` restricts to a single ``CardKind`` (e.g. ``stale``,
    ``contested``); ``semantic`` runs the LLM-assisted finders (defaults to
    ``curation.semantic``), and a request that runs them is rate-limited like
    ``/query``. 400 on an unknown ``kind``.

    Served from the persisted collection, which the nightly
    consolidation cycle or ``POST /curation/rebuild`` writes. A store with no
    collection yet is served **live** (correct, but as slow as it was before
    the persisted-collection cutover) and labelled ``source:
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

    semantic_on = get_config().curation.semantic if semantic is None else semantic
    if semantic_on:
        enforce_rate_limit(request)
    result = await build_curation_queue(
        session,
        limit=_bounded_limit(limit, default=get_config().curation.session_size),
        kind=kind_enum,
        semantic=semantic,
        source=QueueSource.LIVE if no_snapshot else QueueSource.SNAPSHOT,
    )
    # Did the semantic pass actually run? It defaults to curation.semantic when the
    # caller leaves it unset. If it was on but the LLM circuit breaker is
    # open (account-level failure), report it so the client can say so.
    # A snapshot records the breaker state at *build* time, which is the honest
    # answer for cards it is serving.
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
        open_count=result.open_count,
        kind_as_of=result.kind_as_of,
    )


@app.get(
    "/curation/cards/{key:path}",
    dependencies=[AUTH_GATE],
    response_model=CurationCard,
    responses=_ERR404,
)
async def get_curation_card_endpoint(
    key: str, request: Request, session: SessionDep
) -> CurationCard:
    """Return one open curation card by its key, with every belief it names briefed.

    The queue previews five beliefs of a ``subjectless_source`` card;
    this read lists them all, each resolvable on its own ``no_subject`` key. It
    runs the same live checks as ``GET /curation``, and is rate-limited like it
    when ``curation.semantic`` is on, since a lookup with no snapshot collects the
    queue with the model. 404 when no open card has the key."""
    if get_config().curation.semantic:
        enforce_rate_limit(request)
    card = await get_curation_card(session, key)
    if card is None:
        raise HTTPException(status_code=404, detail=f"No open card with key {key!r}.")
    return card


class CurationRebuildResponse(BaseModel):
    """Outcome of an explicit curation-collection rebuild."""

    snapshot_id: str | None
    built_at: datetime | None
    collection_size: int
    scope: str
    semantic: bool


@app.post("/curation/rebuild", dependencies=[OPERATOR_GATE], response_model=CurationRebuildResponse)
async def rebuild_curation_endpoint(
    session: SessionDep,
    _rl: RateLimitDep,
    semantic: bool | None = None,
) -> CurationRebuildResponse:
    """Rebuild the persisted curation collection store-wide.

    **Synchronous and slow by design**: it runs every finder over the whole
    store (minutes on a large one). It is not how the queue becomes fast; the
    nightly consolidation cycle is. This is the operator's escape hatch when they
    need the queue to reflect right now, and the web UI already has an honest
    loading state for it. An async 202 + poll variant waits on a resident
    process to host job state (tracked).

    Requires the write bearer: it mutates the cache. Rate-limited like
    ``/query``, since a semantic rebuild drives paid probes store-wide."""
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
# applies: `OPERATOR_GATE` refuses them (503) while PARTICLES_API_KEY is the
# dev-key sentinel, as every write route is refused on a keyless engine.
# ---------------------------------------------------------------------------


class CurationAffirmRequest(BaseModel):
    """Affirm a belief still holds: records ``BELIEF_AFFIRMED``,
    suppressing the belief's card. Does NOT touch confidence; agreeing with a
    belief is an endorsement stance, which never feeds
    confidence either."""

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
    #: Set when the gesture ruled on a ``demotion`` card and the ruling was
    #: kept as a labelled benchmark pair: the disclosure line.
    benchmark_fixture: str | None = None


@app.post(
    "/curation/affirm",
    dependencies=[OPERATOR_GATE],
    response_model=CurationEventResponse,
    responses=_ERR400,
)
async def curation_affirm_endpoint(req: CurationAffirmRequest) -> CurationEventResponse:
    """Affirm a belief still holds (``BELIEF_AFFIRMED`` operator event).

    Suppresses the belief's curation card without mutating the belief. 403 when
    curation writes are disabled (``mcp.write.enabled_stores`` default-deny).
    Refused (503) under the dev-key skip, even from loopback: it needs a real bearer."""
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event

    _require_belief_writes_enabled()
    card_key = req.card_key or f"contested:{req.particle_id}"
    # The event commits under the store's writer lock, after the 403 (security
    # review 2026-10-28-2, Z47).
    async with session_scope(write=True) as session:
        event = await record_event(
            session,
            actor="http:/curation/affirm",
            event_type=OperatorEventType.BELIEF_AFFIRMED,
            refs=[(EventRefKind.PARTICLE, req.particle_id)],
            payload={"card_key": card_key, "particle_id": req.particle_id},
        )
        fixture = await _record_demotion_ruling(
            session, card_key, "affirm", "http:/curation/affirm"
        )
        await session.commit()
    return CurationEventResponse(
        event_id=event.event_id, card_key=card_key, benchmark_fixture=fixture
    )


async def _record_demotion_ruling(
    session: AsyncSession, card_key: str, gesture: str, actor: str
) -> str | None:
    """Keep a ruling on a ``demotion`` card as a labelled benchmark pair.

    The HTTP gestures write their events directly rather than through
    ``apply_gesture``, so they make the same side-effect call here. ``None``
    for every other card.
    """
    if not card_key.startswith(f"{CardKind.DEMOTION.value}:"):
        return None
    card = CurationCard.from_key(card_key)
    return await record_demotion_ruling(session, card, gesture, actor=actor, store="default")


class MemoryUsefulRequest(BaseModel):
    """Mark a belief useful: records ``BELIEF_MARKED_USEFUL`` and, for a
    press made with an operator key, credits the belief on the explicit utility
    channel. Does NOT touch confidence and does NOT claim the belief is true; it
    lifts the belief in the projection / digest ranking only. For "still true",
    use ``/curation/affirm``."""

    particle_id: str
    reason: str | None = None


class MemoryUsefulResponse(BaseModel):
    """Outcome of an explicit usefulness gesture.

    ``counted`` is False when this interface already credited this belief today,
    the rate bound, and on every press made without an operator
    key, which earns no credit and has no ``credit_key`` (amended
    2026-10-27). The event is recorded either way."""

    event_id: str
    particle_id: str
    credit_key: str | None
    counted: bool


@app.post(
    "/memory/useful",
    dependencies=[AGENT_GATE],
    response_model=MemoryUsefulResponse,
    responses=_ERR400,
)
async def memory_useful_endpoint(req: MemoryUsefulRequest) -> MemoryUsefulResponse:
    """Mark a belief useful (``BELIEF_MARKED_USEFUL`` operator event).

    The explicit second utility channel, for the belief class the transcript
    miner cannot observe: prohibitions and design stances, complied
    with by *not* acting. Promotion-only and projection-only. Only a press made
    with an operator key earns the utility credit, at most one per belief per
    day. An agent key's press is recorded with its principal and earns none,
    so it neither adds to the credit nor uses up the operator's.
    403 when belief writes are disabled (``mcp.write.enabled_stores``
    default-deny)."""
    from particles.operations.utility_feedback import (
        HTTP_ACTOR,
        BeliefNotCreditable,
        mark_belief_useful,
    )

    store = _require_belief_writes_enabled()
    principal = current_principal()
    operator = principal is not None and principal.can(Capability.OPERATOR_WRITE)
    # The press writes an event, so it commits under the writer lock like
    # every other agent write; the 403 above comes first (security review
    # 2026-10-28-1, O47, E18).
    async with session_scope(store, write=True) as session:
        try:
            result = await mark_belief_useful(
                session, req.particle_id, actor=HTTP_ACTOR, reason=req.reason, operator=operator
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


@app.post(
    "/curation/snooze",
    dependencies=[OPERATOR_GATE],
    response_model=CurationEventResponse,
    responses=_ERR400,
)
async def curation_snooze_endpoint(req: CurationSnoozeRequest) -> CurationEventResponse:
    """Snooze / dismiss a curation card (``CURATION_CARD_SNOOZED`` event).

    ``snooze_days=None`` permanently dismisses; a positive value hides the card
    for that window. 403 when curation writes are disabled.
    Refused (503) under the dev-key skip, even from loopback: it needs a real
    bearer."""
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event

    _require_belief_writes_enabled()
    snooze_days = req.snooze_days
    # Every write commits under the store's writer lock, after the 403 (security
    # review 2026-10-28-2, Z47).
    async with session_scope(write=True) as session:
        if snooze_days is None and req.card_key.startswith(f"{CardKind.SUBJECTLESS_SOURCE.value}:"):
            # dismissing a source card rules on the beliefs it names.
            card = CurationCard.from_key(req.card_key)
            card.particle_ids = list(req.particle_ids)
            try:
                dismissal = await dismiss_source_card(session, card, actor="http:/curation/snooze")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            await session.commit()
            return CurationEventResponse(event_id=dismissal.event_id, card_key=req.card_key)
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
        fixture = (
            await _record_demotion_ruling(session, req.card_key, "dismiss", "http:/curation/snooze")
            if snooze_days is None
            else None
        )
        await session.commit()
    return CurationEventResponse(
        event_id=event.event_id,
        card_key=req.card_key,
        snoozed_until=until,
        benchmark_fixture=fixture,
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
    # Absent: an agent key's principal, or ``api-user`` for an operator key.
    deposited_by: str | None = None
    tags: list[str] = []
    # deposit-time link-follow opt-in; None ⇒ the extractor's default
    # (Reddit / HN / Mastodon default True, everything else False). Exposed over
    # HTTP so the thin client's `deposit --follow-post-links` reaches the engine.
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
    # Absent: an agent key's principal, or ``api-user`` for an operator key.
    deposited_by: str | None = None
    tags: list[str] = []
    author_id: str | None = None
    uri_r: str | None = None
    mutability: str | None = None
    # ``"LAZY"`` asks the engine to re-check the source later.
    # Reconciled onto an existing entry even when the content is unchanged;
    # ``None`` leaves any stored policy alone. Over HTTP, LAZY is refused (400)
    # unless ``uri_r`` is http(s): the local refresh loop would read a file:// path
    # on the engine host, which a remote caller must never choose (D2).
    fetch_policy: str | None = None
    content_published_at: datetime | None = None


class DepositResponse(BaseModel):
    """Identifiers for the corpus entry + initial snapshot created by a deposit.

    ``unchanged``: the entry's latest snapshot already
    carried the deposited content hash, so nothing was written and the existing
    IDs are echoed. Every deposit route sets it, except a URL deposit handled by
    a domain importer, which is deduplicated the same way but always reports
    ``false``.

    ``follow_refused`` (a URL deposit only): each primary URL a link post
    names that the deposit did not follow, because another principal's entry
    holds it and an agent key writes only its own entries. The
    primary deposit stands; empty when every follow was made or none was due."""

    entry_id: str
    snapshot_id: str
    unchanged: bool = False
    follow_refused: list[str] = []


def _actor_label() -> str:
    """The acting key's principal, for the audit labels an operator route records.

    ``POST /subjects``, ``POST /trust/statements`` and ``POST /extract`` recorded a
    fixed ``api-user`` before (§3 rule 6). Nothing reads these labels as
    evidence, unlike a depositor, so they name who acted.
    """
    principal = current_principal()
    return principal.id if principal is not None else _ANONYMOUS_DEPOSITOR


@dataclass(frozen=True)
class _DepositProvenance:
    """Who a deposit is recorded as, decided from the key, not the body.

    ``confined`` is true for an agent key: its deposit is wrapped in
    :func:`own_entries_only_for` and may name no URI, no source type of its own
    and no reserved mark it does not hold.
    """

    principal: Principal
    deposited_by: str
    author_id: str | None
    confined: bool
    #: The deposit's tags, with a project-bound key's project stamped on (§4).
    tags: list[str]


#: The anonymous depositor label an operator key's deposit records when the
#: body names none, exactly as before.
_ANONYMOUS_DEPOSITOR = "api-user"

#: The ``file://`` paths a harvester principal's text deposit may name: a
#: memory file under a Claude Code ``projects/<slug>/memory/`` directory, and
#: one project's fold-and-archive file, ``projects/<slug>/MEMORY.archive.md``
#: under the hook's state directory (rule 4; security review
#: 2026-10-26-1, W41). The shape is checked here, and the directories the key
#: records at mint by :func:`_harvest_root_refusal`.
_HARVEST_FILE_PATH_RE = re.compile(
    r"(?:/[^/]+)*/projects/[^/]+/(?:memory/(?:[^/]+/)*[^/]+\.md|MEMORY\.archive\.md)"
)

#: The one ``claude-code://`` shape the harvest sends: a session transcript.
_HARVEST_SESSION_URI_RE = re.compile(r"claude-code://session/[0-9A-Za-z][0-9A-Za-z_-]*")

#: The source types a file can declare in its own bytes that set policy.
_POLICY_SOURCE_TYPES = frozenset(
    {
        SourceType.TAXONOMY_DEFINITION,
        SourceType.TRUST_LENS_DEFINITION,
        SourceType.VOCABULARY_DOCUMENT,
    }
)


def _provenance_refusal(principal: Principal, what: str) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail=(
            f"This key ({principal.role.value}, principal {principal.id}) deposits as "
            f"{principal.id}, so {what}; that needs an operator key."
        ),
    )


def _canonical_project_tags(tags: list[str] | None) -> list[str]:
    """``tags`` with each ``project:`` tag that names a project in its canonical spelling."""
    return [
        project_tag(canonical_project_key(tag[len(PROJECT_TAG_PREFIX) :]))
        if tag.startswith(PROJECT_TAG_PREFIX) and names_a_project(tag[len(PROJECT_TAG_PREFIX) :])
        else tag
        for tag in tags or []
    ]


def _deposit_provenance(
    deposited_by: str | None, author_id: str | None = None, tags: list[str] | None = None
) -> _DepositProvenance:
    """Stamp a deposit with the key's principal, or keep an operator's named attribution.

    An operator key may name any depositor and author (attribution authority);
    when it names no depositor the deposit records ``api-user``, as before,
    because a depositor is evidence trust and the refresh tier read. An agent
    key's deposit is recorded as its principal: a body that names anyone else
    is refused with 403 and nothing is stored, so a client that thinks it
    deposits as someone else is told so. The reserved harvest tags need
    ``HARVEST_MARKS``: an operator, or an agent whose principal is a harvester
    (rules 1 and 5; security review 2026-10-24-2, P5).
    """
    principal = current_principal()
    if principal is None:  # the agent gate admits no anonymous write
        raise HTTPException(status_code=401, detail="Invalid API key")
    if principal.can(Capability.ATTRIBUTE_OTHERS):
        # An operator names what tags it likes, but a project tag is stored in
        # its canonical spelling, as an agent's is, so one project is never two
        # by normalisation form (security review 2026-10-28-2, AA25).
        return _DepositProvenance(
            principal=principal,
            deposited_by=deposited_by or _ANONYMOUS_DEPOSITOR,
            author_id=author_id,
            confined=False,
            tags=_canonical_project_tags(tags),
        )
    for field, value in (("deposited_by", deposited_by), ("author_id", author_id)):
        if value is not None and value != principal.id:
            raise _provenance_refusal(principal, f"{field} may not name {value!r}")
    marks = reserved_marks(tags, None)
    if marks and not principal.can(Capability.HARVEST_MARKS):
        raise _provenance_refusal(
            principal,
            f"it may not name {', '.join(marks)}, which only the owner's harvesters write",
        )
    if RULE_SOURCE_TAG in marks:
        # The rule-file intake (`rules sync`, `init claude-code`) runs on the
        # engine's host only; the remote harvest never sends the mark
        # (security review 2026-10-26-1, W41).
        raise _provenance_refusal(
            principal,
            f"it may not name {RULE_SOURCE_TAG}, which only the local rule-file intake writes",
        )
    for tag in tags or []:
        # A project tag that renders as another project's (security review
        # 2026-10-28-1, O46). The key is judged alone, as the mint judges it,
        # so a key in one non-Latin script is not read as mixing it with the
        # Latin prefix (security review 2026-10-28-2, AA26, AA48).
        reason = (
            lookalike_reason(tag[len(PROJECT_TAG_PREFIX) :])
            if tag.startswith(PROJECT_TAG_PREFIX)
            else None
        )
        if reason is not None:
            raise _provenance_refusal(principal, f"the tag {tag!r} {reason}")
    stamped = _canonical_project_tags(tags)
    if principal.project is not None:
        foreign = project_keys(stamped) - {principal.project}
        if foreign:
            raise _provenance_refusal(
                principal,
                f"its writes are bound to project {principal.project!r} and may not name "
                f"{', '.join(sorted(project_tag(k) for k in foreign))}",
            )
        if principal.project not in project_keys(stamped):
            stamped.append(project_tag(principal.project))
    elif classify_entry(stamped, get_config().observer_scope.harness_tags) is EntryScope.GLOBAL:
        # An agent's material is never global by omission: a deposit that
        # names no project and carries no harness tag is marked unattributed,
        # as the agent routes mark it (security review
        # 2026-10-26-1, W4, W36).
        stamped.append(UNATTRIBUTED_MARKER)
    return _DepositProvenance(
        principal=principal,
        deposited_by=principal.id,
        author_id=principal.id,
        confined=True,
        tags=stamped,
    )


def _is_harvest_uri(uri_r: str) -> bool:
    """Whether ``uri_r`` has the shape of a URI the SessionEnd harvest deposits.

    A session transcript (``claude-code://session/<id>``), or a ``file://``
    memory file or archive file (:data:`_HARVEST_FILE_PATH_RE`). A ``file://``
    URI with a host, a query, a fragment, a ``.`` / ``..`` segment or a control
    character is refused, and the whole path must match, so the shape cannot be
    met by a path that leaves it.
    """
    if any(unicodedata.category(ch)[0] == "C" for ch in uri_r):
        # Judged on the raw string, the one the deposit stores: ``urlsplit``
        # drops a raw tab, CR or LF before the path check below could see it
        # (security review 2026-10-28-1, O44).
        return False
    if uri_r.startswith("claude-code://"):
        return _HARVEST_SESSION_URI_RE.fullmatch(uri_r) is not None
    if not uri_r.startswith("file://"):
        return False
    if any(unicodedata.category(ch)[0] == "C" for ch in uri_r):
        # ``urlsplit`` strips a raw tab, CR or LF before the path is judged,
        # while the raw ``uri_r`` is what is stored (security review
        # 2026-10-28-1, G41).
        return False
    parts = urlsplit(uri_r)
    if parts.netloc or parts.query or parts.fragment:
        return False
    path = unquote(parts.path)
    if any(segment in {".", ".."} for segment in path.split("/")):
        return False
    if any(unicodedata.category(ch)[0] == "C" for ch in path):
        # ``$`` matched before a trailing newline, so an encoded one passed the
        # shape (security review 2026-10-27-1, G41).
        return False
    return _HARVEST_FILE_PATH_RE.fullmatch(path) is not None


def _harvest_root_refusal(principal: Principal, path: str) -> str | None:
    """Why a harvester's ``file://`` path lies under none of its key's roots, or None.

    ``path`` has passed :func:`_is_harvest_uri`. A memory file must sit at
    ``<memory root>/<slug>/memory/…/<name>.md`` and an archive file at
    ``<archive root>/<slug>/MEMORY.archive.md``, the leading segments equal to a
    recorded root's segment for segment: never a string prefix, case-sensitive,
    and with no link resolved, since the harvest sends resolved paths and the
    mint records resolved roots. A key that records no root names no file.
    """
    if principal.memory_roots is None and principal.archive_roots is None:
        return (
            "records no harvest roots, so it names no memory or archive file; mint the hooks "
            "a key with --memory-root and --archive-root naming the client's directories"
        )
    segments = path.split("/")[1:]

    def rest_under(roots: tuple[str, ...] | None) -> list[list[str]]:
        found: list[list[str]] = []
        for root in roots or ():
            head = root.split("/")[1:]
            if segments[: len(head)] == head:
                found.append(segments[len(head) :])
        return found

    for rest in rest_under(principal.memory_roots):
        if len(rest) >= 3 and rest[1] == "memory" and rest[-1].endswith(".md"):
            return None
    for rest in rest_under(principal.archive_roots):
        if len(rest) == 2 and rest[1] == "MEMORY.archive.md":
            return None
    named = [
        f"{label} {', '.join(roots)}"
        for label, roots in (
            ("memory roots", principal.memory_roots),
            ("archive roots", principal.archive_roots),
        )
        if roots
    ]
    return f"may not name {path!r}, which lies under none of its {'; '.join(named)}"


def _confine_text_deposit(prov: _DepositProvenance, uri_r: str | None, source_type: str) -> None:
    """An agent's text deposit names no URI and no source type of its own (rule 4).

    A URI is evidence: an ``http(s)`` one takes that domain's trust and joins its
    update lineage without a fetch (Q2, Q9), and a ``file://`` one is what the
    local refresh keys on and what a document collection is keyed on (Q7).
    Only a harvester principal names one, and only the shapes the harvest
    sends (:func:`_is_harvest_uri`). The engine records the type:
    ``CONVERSATION``, or ``LOCAL_MARKDOWN`` for a harvester's ``file://``
    memory file.
    """
    if not prov.confined:
        return
    harvester = prov.principal.can(Capability.HARVEST_MARKS)
    if uri_r is not None and not (harvester and _is_harvest_uri(uri_r)):
        raise _provenance_refusal(prov.principal, f"it may not name the URI {uri_r!r}")
    if uri_r is not None and uri_r.startswith("file://"):
        path = unquote(urlsplit(uri_r).path)
        refusal = _harvest_root_refusal(prov.principal, path)
        if refusal is None and uri_r != Path(path).as_uri():
            # The one spelling the harvest sends, so the path the roots judged is
            # the URI every lookup keys on and the slug every consumer reads
            # (security review 2026-10-30-1, AB52).
            refusal = f"may name a file only as {Path(path).as_uri()!r}, the harvest's spelling"
        if refusal is not None:
            principal = prov.principal
            raise HTTPException(
                status_code=403,
                detail=f"This key ({principal.role.value}, principal {principal.id}) {refusal}.",
            )
    allowed = {SourceType.CONVERSATION}
    if harvester and uri_r is not None and uri_r.startswith("file://"):
        allowed.add(SourceType.LOCAL_MARKDOWN)
    if source_type not in allowed:
        raise _provenance_refusal(prov.principal, f"it may not type a deposit {source_type!r}")


@contextlib.contextmanager
def _confined(prov: _DepositProvenance) -> Iterator[None]:
    """Run an agent key's deposit under :func:`own_entries_only_for`; 409 on another's entry."""
    if not prov.confined:
        yield
        return
    try:
        with own_entries_only_for(prov.deposited_by):
            yield
    except ForeignEntryRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _refuse_migration_source_type(source_type: str | None) -> None:
    """Refuse an HTTP deposit typed as a memory-server export (400).

    The memory-server façade reads a belief extracted from such an entry as its
    own memory, so the type is a mark only the local migration verb
    (``particles import mcp-memory``, which refuses remote mode) may write.
    Over HTTP, with a key or without one, the type is refused (security review
    2026-10-25-2, R26; the bearer half of Q6).
    """
    if source_type == MIGRATION_SOURCE_TYPE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Refusing a deposit typed {MIGRATION_SOURCE_TYPE}: only the local "
                "`particles import mcp-memory` verb deposits a memory-server export, "
                "because the memory-server facade reads what it yields as its own memory. "
                "Run that verb on the engine's host, or deposit the file under another type."
            ),
        )


@app.post(
    "/corpus/deposit/url",
    dependencies=[AGENT_GATE],
    response_model=DepositResponse,
    responses={**_ERR400, **_ERR502},
)
async def corpus_deposit_url(
    req: DepositUrlRequest, _rl: RateLimitDep, session: SessionDep
) -> DepositResponse:
    """Deposit a URL into the corpus. Returns the entry + snapshot IDs.

    An agent key's deposit is recorded as its principal, names no source type
    (the engine records the one it fetches) and no harvester's tag, and may
    not write another principal's entry: a URL another principal's entry holds
    answers 409. An operator key names what it likes. The store's
    writer lock is taken around each write, never across the fetch. Rate-limited
    per principal like the agents' deposit (``api.rate_limit_per_minute``)."""
    prov = _deposit_provenance(req.deposited_by, tags=req.tags)
    _refuse_migration_source_type(req.source_type)
    if prov.confined and req.source_type is not None:
        raise _provenance_refusal(
            prov.principal, "it may not type a deposit; the engine records the type it fetches"
        )
    unchanged: list[bool] = []
    follow_refused: list[str] = []
    try:
        with _confined(prov), writes_committed_under_lock():
            entry_id, snapshot_id = await deposit_url(
                session,
                req.url,
                deposited_by=prov.deposited_by,
                source_type=req.source_type,
                mutability=req.mutability,
                tags=prov.tags,
                follow_post_links=req.follow_post_links,
                follow_comment_links=req.follow_comment_links,
                out_unchanged=unchanged,
                out_follow_refused=follow_refused,
            )
        # Each write unit committed itself; this commits, under the lock, any
        # write a unit did not cover.
        async with write_transaction(session):
            pass
    except SourceFetchError as exc:
        # Expected upstream condition — the origin refused/failed the fetch (e.g.
        # a Reddit thread the Arctic Shift archive has not captured, the
        # archive's rate limit, or Reddit's own 403 bot-wall on the direct
        # backend), not an SDK bug. Log without a stack trace and map to 502 Bad
        # Gateway with the origin detail, which names the source and the remedy
        # (the client request was well-formed; the upstream is the problem).
        log.warning("deposit_url: upstream fetch failed for %r: %s", req.url, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (EgressRefused, HTTPException):
        # A policy refusal, mapped to 403 by the app-level handler,
        # or this route's own answer (the 409 for another principal's entry).
        raise
    except Exception as exc:
        log.warning("deposit_url failed for %r: %s", req.url, exc, exc_info=True)
        raise HTTPException(status_code=400, detail="Could not deposit the provided URL") from exc
    return DepositResponse(
        entry_id=entry_id,
        snapshot_id=snapshot_id,
        unchanged=any(unchanged),
        follow_refused=follow_refused,
    )


@app.post(
    "/corpus/deposit/text",
    dependencies=[AGENT_GATE],
    response_model=DepositResponse,
    responses={
        **_ERR400,
        409: {
            "model": ErrorResponse,
            "description": "uri_r is held by an entry this route cannot change",
        },
    },
)
async def corpus_deposit_text(
    req: DepositTextRequest, _rl: RateLimitDep, store: StoreDep
) -> DepositResponse:
    """Deposit raw text into the corpus (no fetch).

    ``store`` selects the target store (404 on an unknown handle).
    ``fetch_policy`` ``LAZY`` is refused with 400 unless ``uri_r`` is an http(s)
    URL, because the engine would later re-read a ``file://`` path from its own
    disk. A ``uri_r`` already held by a non-http(s) ``LAZY`` entry (a file the
    engine refreshes from its own disk) is refused with 409, as is a deposit
    whose ``source_type`` differs from the entry it would join.

    An agent key's deposit is recorded as its principal, depositor and author
    alike, and a body naming anyone else is refused (403). It names no
    ``uri_r`` (a harvester principal may name a session transcript's
    ``claude-code://session/<id>``, or the ``file://`` URI of a memory file
    under a ``projects/<slug>/memory/`` directory or of a project's
    ``MEMORY.archive.md``), no source type but ``CONVERSATION``
    (``LOCAL_MARKDOWN`` for a harvester's memory file), and no harvester's tag
    (``claude-code``, ``memory-file``, ``memory-archive``) unless its
    principal is a harvester. No agent key names ``rule-file``, which only the
    local rule-file intake writes. An agent key's deposit that names no project
    is marked ``scope:unattributed``. It may not write another principal's
    entry, unchanged bytes included (409 on a ``uri_r`` such an entry holds). An operator
    key names what it likes. A ``content_published_at`` after the
    deposit is stored as the deposit time. Rate-limited per principal like the
    agents' deposit (``api.rate_limit_per_minute``)."""
    # Every refusal that reads no row comes before the store's writer lock, so
    # a refused or malformed request holds up no other writer (security review
    # 2026-10-28-2, AA6, AA43).
    prov = _deposit_provenance(req.deposited_by, req.author_id, req.tags)
    _refuse_migration_source_type(req.source_type)
    _confine_text_deposit(prov, req.uri_r, req.source_type)
    if req.uri_r is None:
        text_unchanged: list[bool] = []
        async with session_scope(store, write=True) as session:
            with _confined(prov):
                entry_id, snapshot_id = await deposit_text(
                    session,
                    req.text,
                    prov.deposited_by,
                    req.source_type,
                    prov.tags,
                    author_id=prov.author_id,
                    out_unchanged=text_unchanged,
                )
            await session.commit()
        return DepositResponse(
            entry_id=entry_id, snapshot_id=snapshot_id, unchanged=any(text_unchanged)
        )

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
    if fetch_policy is FetchPolicy.LAZY and not is_network_uri(req.uri_r):
        # A LAZY entry is re-read later by the engine itself. Over HTTP the
        # caller is not on the engine host, so a file:// URI-R names the
        # engine's files, not the caller's (security review 2026-10-23, D2).
        # A local deposit (CLI, hook, rules sync) never comes through here.
        raise HTTPException(
            status_code=400,
            detail=(
                "fetch_policy LAZY needs an http(s) uri_r over HTTP: the engine "
                "cannot re-read a file on the caller's host. Deposit with "
                "fetch_policy NEVER, or run the deposit against a local store."
            ),
        )
    from particles.corpus.store import get_entry_by_uri

    async with session_scope(store, write=True) as session:
        existing = await get_entry_by_uri(session, req.uri_r)
        if (
            existing is not None
            and existing.fetch_policy is FetchPolicy.LAZY
            and not is_network_uri(existing.uri_r or "")
        ):
            # The engine enrolled this file from its own disk (rules sync, the
            # local harvest). An HTTP caller appending to it would replace the
            # file's content until the next refresh (security review N19).
            raise HTTPException(
                status_code=409,
                detail=(
                    "uri_r names a file the engine refreshes from its own disk; it "
                    "cannot be written over HTTP."
                ),
            )
        if existing is not None and existing.source_type != req.source_type:
            # A retype would also move the entry's re-fetch floor (N19).
            raise HTTPException(
                status_code=409,
                detail=(
                    f"uri_r is held by a {existing.source_type} entry; a deposit over "
                    f"HTTP cannot change it to {req.source_type}."
                ),
            )
        with _confined(prov):
            entry_id, snapshot_id, unchanged = await deposit_text_versioned(
                session,
                text=req.text,
                uri_r=req.uri_r,
                source_type=req.source_type,
                mutability=mutability,
                tags=prov.tags,
                deposited_by=prov.deposited_by,
                content_published_at=req.content_published_at,
                fetch_policy=fetch_policy,
            )
        await session.commit()
    return DepositResponse(entry_id=entry_id, snapshot_id=snapshot_id, unchanged=unchanged)


#: The block size the file-upload route copies in.
_UPLOAD_CHUNK = 1024 * 1024


@app.post("/corpus/deposit/file", dependencies=[AGENT_GATE], response_model=DepositResponse)
async def corpus_deposit_file(
    file: UploadFile,
    _rl: RateLimitDep,
    deposited_by: Annotated[str | None, Form()] = None,
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
    is the archival authorship date, held to the deposit time. An
    agent key's upload is recorded as its principal, takes the type the engine
    infers from the bytes (a body type must equal it, and an inferred policy
    type, a taxonomy, trust lens or vocabulary document, is refused), names no
    harvester's tag unless its principal is a harvester, and touches only the
    agent's own entries. An operator key names what it likes.
    Rate-limited per principal like the agents' deposit (``api.rate_limit_per_minute``).

    An upload larger than ``local_files.max_bytes`` is refused with 413. An
    upload whose bytes match a file the engine enrolled from its own disk gets
    an entry of its own rather than appending to that entry or changing its
    ``source_type``."""
    import tempfile

    prov = _deposit_provenance(deposited_by, tags=tags)
    _refuse_migration_source_type(source_type)

    max_bytes = get_config().local_files.max_bytes
    # Create + write the temp file *inside* the try so a crash mid-read/write is
    # still covered by the finally unlink — a ``delete=False`` temp that leaked
    # before the try would accrete under repeated failed uploads (F20).
    tmp_path: Path | None = None
    file_unchanged: list[bool] = []
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "upload").suffix
        ) as tmp:
            tmp_path = Path(tmp.name)
            # Copied in blocks and capped, never read whole into memory
            # (security review 2026-10-24, the critic's S4 item).
            written = 0
            while block := await file.read(_UPLOAD_CHUNK):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload is over the {max_bytes}-byte local_files.max_bytes cap",
                    )
                tmp.write(block)
        if prov.confined:
            # The engine records the type the bytes resolve to; a body type
            # must agree, and a file declaring policy is the operator's. The
            # sniff reads and parses, so it runs off the event loop (AC2).
            inferred = await asyncio.to_thread(resolve_upload_source_type, tmp_path)
            if source_type is not None and source_type != inferred:
                raise _provenance_refusal(
                    prov.principal, f"it may not type an upload {source_type!r}"
                )
            if inferred in _POLICY_SOURCE_TYPES or inferred == MIGRATION_SOURCE_TYPE:
                raise _provenance_refusal(prov.principal, f"it may not deposit a {inferred} file")
            source_type = inferred
        elif source_type is None:
            # Any caller's untyped upload is sniffed the same way: off the event
            # loop and before the writer lock (security review 2026-10-31-1, AD23).
            source_type = await asyncio.to_thread(resolve_upload_source_type, tmp_path)
        # The caller chose ``deposited_by``, so a content-hash match may not
        # join or retype an entry the engine enrolled itself (security review P19).
        # The writer lock is taken only now, after the upload is copied and every
        # refusal made (security review 2026-10-28-2, AA6).
        async with session_scope(write=True) as session:
            with _confined(prov):
                entry_id, snapshot_id = await deposit_file(
                    session,
                    tmp_path,
                    deposited_by=prov.deposited_by,
                    source_type=source_type,
                    tags=prov.tags,
                    content_date=content_date,
                    out_unchanged=file_unchanged,
                    untrusted_caller=True,
                )
            await session.commit()
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return DepositResponse(
        entry_id=entry_id, snapshot_id=snapshot_id, unchanged=any(file_unchanged)
    )


@app.get("/corpus", dependencies=[READ_GATE], response_model=list[CorpusEntry])
async def list_corpus(
    session: SessionDep,
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
    dependencies=[READ_GATE],
    response_model=CorpusEntry,
    responses=_ERR404,
)
async def corpus_get(entry_id: str, session: SessionDep) -> CorpusEntry:
    """Fetch a corpus entry by ID. 404 if not present."""
    from particles.corpus.store import get_entry as _get_entry

    entry = await _get_entry(session, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@app.get(
    "/corpus/blob/{selector}",
    dependencies=[READ_GATE],
    responses={**_ERR404, **_ERR400},
)
async def corpus_blob(selector: str, session: SessionDep) -> Response:
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
            # The bytes are a depositor's, served on the origin that holds the
            # web UI's bearer: a browser must never sniff them into a script or
            # a page (security review 2026-10-24-2, N7).
            "X-Content-Type-Options": "nosniff",
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
    actually retracted, or, when ``dry_run`` is true, the ones that *would*
    be. ``skipped`` maps each non-live status to how many particles were left
    untouched. The corpus entry and its snapshots are always preserved."""

    entry_id: str
    dry_run: bool
    retracted_ids: list[str]
    skipped: dict[str, int]


@app.post(
    "/corpus/{entry_id}/retract",
    dependencies=[AUTH_GATE],
    response_model=CorpusRetractResponse,
    responses=_ERR404,
)
async def corpus_retract_endpoint(
    entry_id: str, req: CorpusRetractRequest, session: SessionDep
) -> CorpusRetractResponse:
    """Retract every live particle from a source, preserving the corpus +
    snapshots (HTTP mirror of `corpus retract`). Live particles
    (ACTIVE / INCONSISTENCY) become RETRACTED with reason SOURCE_RETRACTED;
    idempotent. 404 if the entry is absent. Run `POST /lint` afterwards to
    cascade PROVENANCE_STALE to downstream particles. A real retraction needs
    a real bearer: it is refused (503) under the dev-key skip, even from
    loopback, while a `dry_run` plan is served as before."""
    from particles.corpus.store import get_entry as _get_entry
    from particles.operations.retract import plan_retraction, retract_entry

    if not req.dry_run:
        require_operator_credential()
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

    # The writer lock is taken only here, after the credential check, so a
    # refused retract never holds it (security review 2026-10-25, Q12).
    async with write_transaction(session):
        result = await retract_entry(session, entry_id, reason=req.reason)
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
    # Absent: the acting key's principal (rule 6).
    agent_id: str | None = None


@app.post(
    "/extract",
    dependencies=[OPERATOR_GATE],
    response_model=list[Particle],
    responses={**_ERR404, **_ERR500},
)
async def extract(req: ExtractRequest, session: SessionDep, _rl: RateLimitDep) -> list[Particle]:
    """Run the extractor pipeline against a corpus snapshot. Returns
    the ACTIVE particles produced by the run (excludes any that
    immediately transitioned to SUPERSEDED / INCONSISTENCY).

    The run's measured LLM usage is recorded as an ``EXTRACT_RUN`` event, so a
    remote caller's extraction counts toward the store's recorded spend."""
    try:
        meter = MeteredExtractRun(actor="http:/extract", route="http")
        async with meter:
            meter.snapshots = 1
            particles = await extract_snapshot(
                session, req.entry_id, req.snapshot_id, agent_id=req.agent_id or _actor_label()
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
    dependencies=[READ_GATE],
    response_model=NarrativeResponse,
    responses=_ERR404,
)
async def get_narrative(particle_id: str, session: SessionDep) -> NarrativeResponse:
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
    dependencies=[READ_GATE],
    response_model=list[Particle],
    responses=_ERR404,
)
async def get_containing_narratives(particle_id: str, session: SessionDep) -> list[Particle]:
    """Return the NARRATIVE particles this particle is ``PART_OF``.

    404 if the particle does not exist; an empty list if it belongs to none."""
    from particles.operations.narrative import get_narratives_containing
    from particles.store.particle_store import get_particle

    if await get_particle(session, particle_id) is None:
        raise HTTPException(status_code=404, detail="Particle not found")
    return await get_narratives_containing(session, particle_id)


@app.get(
    "/particles/{particle_id}/source",
    dependencies=[READ_GATE],
    response_model=SourcePassage,
    responses=_ERR404,
)
async def get_particle_source(particle_id: str, session: SessionDep) -> SourcePassage:
    """Return the source passage behind a particle, with how it was found.

    ``match`` is ``EXACT`` (the hash-verified chunk the extractor saw),
    ``LOCATED`` (best term-overlap paragraph; a reading aid, not
    verification), ``WHOLE`` (the snapshot text), or ``UNAVAILABLE`` with a
    ``note``. Display only: never a ranking input. 404 if the particle does
    not exist."""
    passage = await hydrate_source_passage(session, particle_id)
    if passage is None:
        raise HTTPException(status_code=404, detail="Particle not found")
    return passage


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
    dependencies=[AUTH_GATE],
    response_model=NarrativeArticleResponse,
    responses=_ERR404,
)
async def get_narrative_synthesis(
    particle_id: str, session: SessionDep, _rl: RateLimitDep
) -> NarrativeArticleResponse:
    """Render a NARRATIVE as one cited prose article.

    Traverses the SEQUENCE_IN chain and runs the article-synthesis engine
    (LLM call; structured-listing fallback on failure), rate-limited like
    ``/query``. 404 if the particle does not exist or is not a NARRATIVE.

    A synthesised body is cached, so a repeat request makes no model call. Only
    a key that can write persists the cache row: a reader key writes nothing
    (security review 2026-10-27-1, G7), and a keyless engine
    serves reads only."""
    from particles.operations.narrative_synthesis import synthesize_narrative
    from particles.store.particle_store import get_particle

    principal = current_principal()
    persist = principal is not None and principal.can(Capability.AGENT_WRITE)
    article = await synthesize_narrative(session, particle_id, persist_cache=persist)
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


@app.post("/query", dependencies=[AUTH_GATE], response_model=QueryResponse)
async def run_query(req: QueryRequest, session: SessionDep, _rl: RateLimitDep) -> QueryResponse:
    """Tag-aware semantic query. Embeds the question, runs cosine
    similarity over ACTIVE particles, generates a natural-language
    answer with citations.

    Rate-limited per client (``api.rate_limit_per_minute``; security review F6):
    this drives a paid embedding + completion per request. Bearer-gated
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


@app.get("/graph", dependencies=[AUTH_GATE], response_model=GraphData)
async def get_graph(
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
    observer_project: Annotated[str | None, Query(min_length=1)] = None,
) -> GraphData:
    """One scoped epistemic subgraph: the same ``GraphData`` the
    static ``export graph`` artifact embeds, computed fresh per request.

    Scope is mandatory (the anti-hairball invariant: a whole-store render does
    not exist): ``scope=subject`` needs ``subject_id``, ``scope=query`` needs
    ``q``, ``scope=inconsistency`` needs ``inconsistency_id`` (a contradiction's
    evidence: the INCONSISTENCY anchor, its disputants, their
    subjects), ``scope=projection`` needs ``manifest`` + ``section`` (a
    manifest section's deterministic selection); anything else is
    422. ``manifest`` is a relative name under ``graph.manifest_root``; with no
    root configured the projection scope is refused (404), and a name that is
    absolute, climbs out with ``..``, or names a missing, non-regular,
    oversized or invalid file gets one fixed 404 with no exception text, so the
    route reads nothing on the engine host outside that directory and does not
    say which files exist (security review 2026-10-24-2, P1; 2026-10-25, Q43).
    ``subject_id`` accepts a subject id or an exact (case-insensitive) canonical
    name / alias; ``inconsistency_id``
    accepts a full id or unique prefix; ``scope_ref`` in the response is always
    the resolved anchor address. ``hops`` / ``max_nodes`` are clamped to the
    ``graph.*`` caps; ``as_of`` renders the graph as believed at T;
    ``history`` adds supersession-chain ghosts; ``store`` selects the target
    store (404 on an unknown handle). ``observer_project`` reads the
    subject, query and projection scopes through a project observer;
    the inconsistency scope addresses its evidence by id and is never filtered.

    Bearer-gated regardless of ``api.require_auth_for_reads`` and rate-limited
    (same posture as ``/query``): the query scope drives a paid embedding per
    request. All epistemics are computed server-side; clients only render."""
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

    manifest_root: Path | None = None
    if scope == "projection":
        root = get_config().graph.manifest_root
        if root is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "unknown manifest: the projection scope is served over HTTP only "
                    "from graph.manifest_root, which is not set"
                ),
            )
        manifest_root = Path(root)

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
                observer_project=observer_project,
                manifest_root=manifest_root,
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


@app.post("/lint", dependencies=[AUTH_GATE], response_model=LintReport)
async def run_lint_endpoint(req: LintRequest, request: Request, session: SessionDep) -> LintReport:
    """Run the lint pipeline. Returns the structured findings report.

    ``fix`` defaults to ``False``; pass ``fix: true`` to
    apply structural status transitions (STALENESS, RETRACTION_CASCADE,
    CORPUS_LINK_INTEGRITY). A fixing run changes belief status with the
    operator's authority, so it is refused (503) under the dev-key skip, even
    from loopback; a report-only run is served as before."""
    if req.fix:
        require_operator_credential()
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


@app.get("/lint/report", dependencies=[READ_GATE], response_model=LintReport)
async def get_lint_report(session: SessionDep) -> LintReport:
    """Return the structural-only lint report: no LLM call, no mutations."""
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


@app.post("/links/suggest", dependencies=[AUTH_GATE], response_model=SuggestReport)
async def run_links_suggest(
    req: LinksSuggestRequest, request: Request, session: SessionDep
) -> SuggestReport:
    """Propose (and optionally resolve) co-evidential candidate links.

    Mirrors the ``particles links suggest`` CLI. Returns ``409`` when
    ``APPLY`` would link more pairs than the confirm threshold and
    ``confirmed`` was not set. The judging modes call the model per pair, so
    they are rate-limited like ``/query``; ``REPORT`` is not. ``APPLY`` links
    pairs with the operator's authority, so it is refused (503) under the
    dev-key skip, even from loopback."""
    from particles.operations.links_suggest import (
        ApplyConfirmationRequired,
        suggest_co_evidential,
    )

    if req.mode is SuggestMode.APPLY:
        require_operator_credential()
    if req.mode != SuggestMode.REPORT:
        enforce_rate_limit(request)
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

    limit: int | None = Field(default=None, ge=1)
    min_sources: int | None = Field(default=None, ge=1)


class CorpusLinksDismissRequest(BaseModel):
    """Options for ``POST /corpus/links/dismiss``.

    ``snooze_days=None`` is a permanent dismiss; a positive value snoozes for
    that many days."""

    url: str
    snooze_days: int | None = None


class CorpusLinksDismissResponse(BaseModel):
    canonical_url: str
    suppressed_until: datetime


@app.post("/corpus/links/suggest", dependencies=[READ_GATE], response_model=DepositSuggestReport)
async def run_corpus_links_suggest(
    req: CorpusLinksSuggestRequest, session: SessionDep
) -> DepositSuggestReport:
    """Rank undeposited but frequently-cited URLs as deposit suggestions.

    Mirrors ``particles corpus links suggest``. Read-only: nothing is fetched
    or deposited."""
    from particles.operations.deposit_suggest import suggest_deposits

    limit = None if req.limit is None else _bounded_limit(req.limit)
    return await suggest_deposits(session, limit=limit, min_sources=req.min_sources)


@app.post(
    "/corpus/links/dismiss", dependencies=[OPERATOR_GATE], response_model=CorpusLinksDismissResponse
)
async def run_corpus_links_dismiss(req: CorpusLinksDismissRequest) -> CorpusLinksDismissResponse:
    """Dismiss / snooze a deposit suggestion (audited via the event log).

    Returns ``400`` when ``url`` is not a usable http(s) URL."""
    from particles.operations.deposit_suggest import dismiss_suggestion
    from particles.url_canonical import canonicalize_url

    canon = canonicalize_url(req.url)
    if canon is None:
        raise HTTPException(status_code=400, detail=f"Not a usable http(s) URL: {req.url!r}")
    # Commits under the store's writer lock, after the 400 (security review
    # 2026-10-30-1, AB68).
    async with session_scope(write=True) as session:
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


@app.get("/review", dependencies=[AUTH_GATE], response_model=list[Particle])
async def list_review_queue(session: SessionDep) -> list[Particle]:
    """Return all INCONSISTENCY particles awaiting operator resolution."""
    return await list_inconsistencies(session)


class ReviewRequest(BaseModel):
    """Resolve an INCONSISTENCY: pick a ``ResolutionAction`` (prefer A,
    prefer B, both valid, defer, or discard both) and optionally annotate."""

    action: ResolutionAction
    reviewer_id: str
    domain: str = "general"
    note: str | None = None


@app.post(
    "/review/{particle_id}",
    dependencies=[OPERATOR_GATE],
    response_model=ReviewParticle,
    responses={
        **_ERR404,
        409: {"model": ErrorResponse, "description": "Action refused on this record"},
    },
)
async def review_particle(particle_id: str, req: ReviewRequest) -> ReviewParticle:
    """Resolve an INCONSISTENCY particle (§9.6).

    Every action except DEFER closes the record. PREFER_A and PREFER_B demote
    the other claim and may write a SourceTrustStatement. BOTH_VALID keeps
    both claims as ALEATORY. DISCARD retracts both claims and writes no trust
    statement. An action that would keep a claim it cannot
    restore, such as PREFER_A on a record whose claim A has left ACTIVE, is
    refused with 409 and nothing is written.
    Refused (503) under the dev-key skip, even from loopback: it needs a real
    bearer."""
    try:
        # The resolution commits under the store's writer lock (security review
        # 2026-10-30-1, AB4).
        async with session_scope(write=True) as session:
            return await resolve(
                session,
                particle_id,
                req.action,
                req.reviewer_id,
                req.domain,
                req.note,
            )
    except ResolutionRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    extractor upgrade). The particle-selecting fields
    (``extractor_version`` / ``extractor_id`` / ``provider_model``) union
    with each other and intersect with ``entry_ids`` when both are sent,
    so naming entries can only narrow the scope.
    ``include_schema_stale`` also re-extracts the snapshots whose particles
    carry an older minor of the schema; it is off by default because after a
    minor bump the whole store qualifies (D5)."""

    entry_ids: list[str] | None = None
    extractor_version: str | None = None
    extractor_id: str | None = None
    include_failed: bool = True
    # re-extract what one "<provider>:<model>" pairing produced.
    provider_model: str | None = None
    # D5: also re-extract snapshots whose particles carry an older
    # minor of the SDK's schema major. Opt-in, like the CLI flag; auto-discovery
    # only, like ``include_failed``.
    include_schema_stale: bool = False
    # Pace of the re-extraction loop. Bounded (security review D37): 0 used to
    # mean unpaced, and a caller could pick any pace at all.
    rate_limit_per_minute: int = Field(default=100, ge=1, le=600)
    # Compute and return the work plan without extracting (zero LLM calls,
    # zero writes); the response carries per-snapshot detail in ``plan``.
    dry_run: bool = False


@app.post("/reindex", dependencies=[AUTH_GATE], response_model=ReindexResponse)
async def run_reindex(
    req: ReindexRequest, session: SessionDep, _rl: RateLimitDep
) -> ReindexResponse:
    """Re-extract stale / failed snapshots. Honours chunk-hash
    carry-forward so unchanged source chunks skip
    the LLM call.

    A run that is not a dry run records its measured LLM usage, the
    re-extraction and the post-reindex lint pass together, as an
    ``EXTRACT_RUN`` event with ``route: reindex``, so it counts toward the
    store's recorded spend."""

    async def _run() -> dict[str, object]:
        return await reindex(
            session,
            entry_ids=req.entry_ids,
            extractor_version=req.extractor_version,
            extractor_id=req.extractor_id,
            include_failed=req.include_failed,
            provider_model=req.provider_model,
            rate_limit_per_minute=req.rate_limit_per_minute,
            dry_run=req.dry_run,
            include_schema_stale=req.include_schema_stale,
        )

    if req.dry_run:
        result = await _run()
    else:
        require_operator_credential()
        meter = MeteredExtractRun(actor="http:/reindex", route="reindex")
        async with meter:
            result = await _run()
            meter.snapshots = reindexed_snapshots(result)
    return ReindexResponse.model_validate(result)


class ReconcileRequest(BaseModel):
    """Document-supersession reconcile-sweep options.

    ``dry_run=True`` reports what would be demoted without mutating the store."""

    dry_run: bool = False


@app.post("/reconcile", dependencies=[AUTH_GATE], response_model=ReconcileResponse)
async def run_reconcile(
    req: ReconcileRequest, session: SessionDep, _rl: RateLimitDep
) -> ReconcileResponse:
    """Run the cross-entry document-supersession sweep: demote
    superseded claims the intra-entry extract path never reconciles. v1 is
    single-trust-order only and covers the document-supersession mode. A run that
    demotes is refused (503) under the dev-key skip, even from loopback; a
    ``dry_run`` report is served as before."""
    if not req.dry_run:
        require_operator_credential()
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


@app.get("/subjects", dependencies=[READ_GATE], response_model=list[Subject])
async def list_subjects(
    session: SessionDep,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
    order: Literal["name", "degree"] = Query("name"),
) -> list[Subject]:
    """List Subjects (alphabetical). ``limit``/``offset`` paginate (MCP-friendly);
    an omitted ``limit`` returns the first ``storage.max_page_size`` and any
    ``limit`` is clamped to that cap (security review F5; formerly unbounded).
    ``order=degree`` sorts by descending ACTIVE-particle link count instead
    (most-connected first, e.g. the web UI's Browse seed)."""
    from particles.store.subject_store import list_all_subjects

    return await list_all_subjects(session, limit=_bounded_limit(limit), offset=offset, order=order)


@app.get("/subjects/search", dependencies=[READ_GATE], response_model=list[Subject])
async def search_subjects_endpoint(
    session: SessionDep,
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
    dependencies=[READ_GATE],
    response_model=Subject,
    responses=_ERR404,
)
async def get_subject_endpoint(subject_id: str, session: SessionDep) -> Subject:
    """Fetch a Subject by ID. 404 if not present."""
    from particles.store.subject_store import get_subject

    s = await get_subject(session, subject_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    return s


@app.get(
    "/subjects/{subject_id}/particles", dependencies=[READ_GATE], response_model=list[Particle]
)
async def get_particles_for_subject(
    subject_id: str,
    session: SessionDep,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
) -> list[Particle]:
    """List ACTIVE particles whose subject set includes this subject.

    ``limit``/``offset`` paginate over the linked particle IDs; an omitted
    ``limit`` returns up to ``storage.max_page_size`` and any ``limit`` is
    clamped to it (security review F5; formerly returned every linked
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


@app.get(
    "/subjects/{subject_id}/particle-ids",
    dependencies=[READ_GATE],
    response_model=SubjectParticleIdsResponse,
)
async def get_subject_particle_ids(
    subject_id: str, session: SessionDep, limit: int | None = Query(None, ge=1)
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


@app.post("/subjects", dependencies=[OPERATOR_GATE], response_model=Subject)
async def create_subject(req: CreateSubjectRequest) -> Subject:
    """Create a Subject by hand. The normal path is automatic
    resolution at extraction time; this endpoint exists for operator
    bootstrapping and corrections.

    ``aliases`` and ``external_ids`` need operator authority (503 under the
    dev-key skip), as ``/subjects/{id}/aliases`` does: the local alias index is
    the resolver's first step, so either steers every later extraction that
    names those strings onto this Subject (security review 2026-10-24-2, P7).
    Every Subject needs a real key, since a keyless engine serves reads only:
    a canonical name alone can capture another Subject's alias
    (security review 2026-10-25, Q3)."""
    from datetime import datetime

    if req.aliases or req.external_ids:
        require_operator_credential()

    from particles.store.subject_store import insert_subject

    subject = Subject(
        canonical_name=req.canonical_name,
        description=req.description,
        aliases=req.aliases,
        external_ids=req.external_ids,
        created_at=datetime.now(UTC),
        asserted_by=_actor_label(),
    )
    # Commits under the store's writer lock, after the credential refusal
    # (security review 2026-10-30-1, AB68).
    async with session_scope(write=True) as session:
        await insert_subject(session, subject)
        await session.commit()
    return subject


class SplitSubjectRequest(BaseModel):
    """Re-bind some particles off a source Subject onto a new one.

    Supply the new Subject's identity via ``new_external_id`` (authoritative:
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
    dependencies=[OPERATOR_GATE],
    response_model=SplitSubjectResponse,
    responses=_ERR404,
)
async def split_subject_endpoint(subject_id: str, req: SplitSubjectRequest) -> SplitSubjectResponse:
    """Re-bind misjoined particles off a source Subject onto a new Subject the
    resolver canonicalises (HTTP mirror of `subjects split`). 404 if
    the source Subject is absent; 400 on an empty particle list or a bad target
    identity. Refused (503) under the dev-key skip, even from loopback: it
    needs a real bearer."""
    from particles.ingest.subject_resolver import gather_split_target, split_subject_resolving
    from particles.store.subject_store import get_subject

    if not req.particle_ids:
        raise HTTPException(status_code=400, detail="particle_ids must be non-empty")
    # The writer lock is taken after the body's refusals (security review
    # 2026-10-30-1, AB54), and after the target's live lookups and label fetch
    # (security review 2026-10-31-1).
    async with session_scope() as session:
        if await get_subject(session, subject_id) is None:
            raise HTTPException(status_code=404, detail="Subject not found")
        try:
            gathered = await gather_split_target(
                session, new_name=req.new_name, new_external_id=req.new_external_id
            )
            async with write_transaction(session):
                new_subject, relinked, not_bound = await split_subject_resolving(
                    session,
                    source_id=subject_id,
                    particle_ids=req.particle_ids,
                    new_name=req.new_name,
                    new_external_id=req.new_external_id,
                    gathered=gathered,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SplitSubjectResponse(
        new_subject=new_subject,
        relinked_particle_ids=relinked,
        not_bound_particle_ids=not_bound,
    )


# ---------------------------------------------------------------------------
# Operator event log — read-only, mirrors the CLI `events` group
# ---------------------------------------------------------------------------


class WhoAmIResponse(BaseModel):
    """The presented key, as the engine resolved it."""

    anonymous: bool
    principal: str | None = None
    role: str | None = None
    key_id: str | None = None
    project: str | None = None
    #: A harvester key's recorded client directories, so the
    #: client's doctor can compare them with its own; absent otherwise.
    memory_roots: list[str] | None = None
    archive_roots: list[str] | None = None


@app.get("/auth/whoami", dependencies=[AUTH_GATE], response_model=WhoAmIResponse)
async def whoami() -> WhoAmIResponse:
    """Say which principal, role and project the presented key stands for.

    A client checks its key with it before any write, and the remote-aware
    doctor uses it as its authenticated probe. On a keyed engine the bearer is
    required (401 otherwise); a keyless engine answers ``anonymous`` to a
    loopback caller. ``key_id`` is ``env`` for ``PARTICLES_API_KEY``, which is
    the operator's key. A harvester key also returns the client directories it
    records, which its own memory and archive files must lie under.
    """
    principal = current_principal()
    if principal is None:
        return WhoAmIResponse(anonymous=True)
    return WhoAmIResponse(
        anonymous=False,
        principal=principal.id,
        role=principal.role.value,
        key_id=principal.key_id,
        project=principal.project,
        memory_roots=list(principal.memory_roots) if principal.memory_roots else None,
        archive_roots=list(principal.archive_roots) if principal.archive_roots else None,
    )


@app.get("/events", dependencies=[AUTH_GATE], response_model=list[OperatorEvent])
async def list_events_endpoint(
    session: SessionDep,
    particle: str | None = None,
    subject: str | None = None,
    entry: str | None = None,
    type: str | None = None,
    principal: str | None = None,
    key: str | None = None,
    limit: int | None = Query(None, ge=1),
) -> list[OperatorEvent]:
    """List operator events newest-first, optionally filtered.

    At most one of ``particle`` / ``subject`` / ``entry`` may be set (the
    record the event touched); ``type`` filters by event type; ``principal``
    and ``key`` keep the events an HTTP caller with that principal, or that
    one key, wrote. A reader or agent key is served no other key's id, and may
    filter by its own key only (403 otherwise).
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
    caller = current_principal()
    if (
        key is not None
        and caller is not None
        and _hides_other_key_ids(caller)
        and key != caller.key_id
    ):
        raise HTTPException(
            status_code=403,
            detail="This key may filter the event log by its own key id only.",
        )
    events = await list_events(
        session,
        ref_kind=ref_kind,
        ref_id=ref_id,
        event_type=etype,
        http_principal=principal,
        http_key_id=key,
        limit=_bounded_limit(limit, default=50),
    )
    return [_event_for_caller(event, caller) for event in events]


@app.get(
    "/events/{event_id}", dependencies=[AUTH_GATE], response_model=OperatorEvent, responses=_ERR404
)
async def get_event_endpoint(event_id: str, session: SessionDep) -> OperatorEvent:
    """Fetch one operator event by id. 404 if not present.

    Bearer-gated like ``GET /events`` (the operator audit trail; F2)."""
    from particles.store.event_store import get_event

    event = await get_event(session, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _event_for_caller(event, current_principal())


def _hides_other_key_ids(caller: Principal) -> bool:
    """Whether ``caller`` is served no other key's id from the event log.

    Every key but an operator key: the ids are the engine's key inventory,
    and with them a reader could list one other client's acts (security
    review 2026-10-27-1, G47).
    """
    return not caller.can(Capability.OPERATOR_WRITE)


def _event_for_caller(event: OperatorEvent, caller: Principal | None) -> OperatorEvent:
    """``event`` as ``caller`` may read it: another key's id removed for a non-operator.

    A keyless engine (no ``caller``) has no keys to hide.
    """
    if caller is None or not _hides_other_key_ids(caller) or not event.payload:
        return event
    key_id = event.payload.get(HTTP_KEY_ID_FIELD)
    if key_id is None or key_id == caller.key_id:
        return event
    payload = {k: v for k, v in event.payload.items() if k != HTTP_KEY_ID_FIELD}
    return event.model_copy(update={"payload": payload})


# ---------------------------------------------------------------------------
# MCP read parity — thin GETs the routed MCP read tools need that
# no pre-existing endpoint covered. Declared BEFORE `/particles/{particle_id}`
# so the static `/particles/search` and `/particles/contested` paths win the
# route match over the `{particle_id}` capture. These reads carry `READ_GATE`
# — unauthenticated by default, but bearer-gated when `api.require_auth_for_reads`
# is set (security review F2). The bulk read surface (`/quality`, `/subjects`,
# `/corpus`, …) follows the same posture; `/query`, `/events`, and `/digest`
# carry the unconditional `AUTH_GATE` instead (gated regardless of the flag).
# ---------------------------------------------------------------------------


@app.get("/particles", dependencies=[READ_GATE], response_model=list[Particle])
async def list_particles_endpoint(
    session: SessionDep,
    status: str | None = None,
    subject_id: str | None = None,
    limit: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
    observer_project: Annotated[str | None, Query(min_length=1)] = None,
) -> list[Particle]:
    """List particles, optionally filtered by status / subject (HTTP mirror of the
    MCP ``particles_list`` tool). Newest first; no embeddings.

    ``limit`` defaults to 50 and is clamped to ``storage.max_page_size`` (F5).
    400 on an unknown ``status``. ``observer_project`` lists only what is in
    view for that project observer, paged after the predicate."""
    from particles.operations.query.observer_scope import list_particles_in_view

    status_enum: Status | None = None
    if status is not None:
        try:
            status_enum = Status(status)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in Status)
            raise HTTPException(
                status_code=400, detail=f"Unknown status {status!r}. Allowed: {allowed}."
            ) from exc
    return await list_particles_in_view(
        session,
        status=status_enum,
        subject_id=subject_id,
        limit=_bounded_limit(limit, default=50),
        offset=offset,
        observer_project=observer_project,
    )


@app.get("/particles/search", dependencies=[READ_GATE], response_model=list[Particle])
async def search_particles_by_fingerprint(
    fingerprint: str,
    session: SessionDep,
    limit: int | None = Query(None, ge=1),
    observer_project: Annotated[str | None, Query(min_length=1)] = None,
) -> list[Particle]:
    """List particles sharing a context fingerprint (HTTP mirror of the
    MCP ``particle_search`` tool). ``fingerprint`` is a full 64-char SHA-256 hex
    or a prefix (≥ 8 chars; validated in-handler → 400). ``limit`` defaults to 50
    and is clamped to ``storage.max_page_size`` (F5). ``observer_project`` keeps
    only what is in view for that project observer."""
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
    bounded = _bounded_limit(limit, default=50)
    if observer_project is None:
        result = await session.execute(stmt.limit(bounded))
        return [row.to_model() for row in result.scalars()]
    from particles.operations.query.observer_scope import in_view

    found = [row.to_model() for row in (await session.execute(stmt)).scalars()]
    return (await in_view(session, found, observer_project))[:bounded]


@app.get("/particles/contested", dependencies=[READ_GATE], response_model=dict[str, str])
async def particle_contested_backrefs(session: SessionDep) -> dict[str, str]:
    """Map each ACTIVE belief an open INCONSISTENCY references → that INCONSISTENCY
    id. Backs the contested marker the routed MCP ``query`` /
    ``particles_list`` tools apply to recalled beliefs."""
    from particles.store.particle_store import get_inconsistency_backrefs

    return await get_inconsistency_backrefs(session)


class ContestedBadgesRequest(BaseModel):
    """The ids to compose badges for; POST because the list can be long."""

    particle_ids: list[str] = Field(default_factory=list)


@app.post(
    "/particles/contested-badges",
    dependencies=[READ_GATE],
    response_model=dict[str, ContestedBadge],
)
async def particle_contested_badges(
    body: ContestedBadgesRequest,
    session: SessionDep,
) -> dict[str, ContestedBadge]:
    """Compose the contested badge for the requested ids.

    The one composer for every recall surface: the routed MCP
    ``particles_list`` tool used to hand-roll a one-basis badge of its own,
    which made the listing disagree with ``query`` on the same server. Keyed by
    id and **sparse**: an id no available basis fired on (or one that does not
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


@app.get("/taxonomies", dependencies=[READ_GATE], response_model=list[TaxonomyDefinition])
async def list_taxonomies_endpoint(
    session: SessionDep,
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
    lines: list[DigestLine] = Field(
        default_factory=list,
        description=(
            "The belief behind each digest line, in order, with the line's character "
            "offset in `markdown`. A caller that cuts the text keeps the lines whose "
            "offset fell inside the cut."
        ),
    )


@app.get(
    "/digest/{store}", dependencies=[AUTH_GATE], response_model=DigestResponse, responses=_ERR404
)
async def get_digest_endpoint(
    store: str,
    project: Annotated[str | None, Query(min_length=1)] = None,
) -> DigestResponse:
    """Render the session-start memory digest for a store.

    Read-only, zero LLM/embeddings, rendered fresh. 404 if the store handle is
    not configured on this engine. ``project`` reads the store through that
    project observer, evaluated against this engine's own sources.
    ``build_digest`` manages its own per-store session, so this handler takes no
    ``SessionDep``.

    Bearer-gated regardless of ``api.require_auth_for_reads`` (F2): the digest
    is the provenance-ranked roll-up of the full belief store, including
    contested beliefs, which is confidential operator context. The dev-key loopback
    skip keeps it open for local development."""
    from particles.operations.digest import build_digest_located

    try:
        rendered = await build_digest_located(store, project)
    except (KeyError, ValueError) as exc:
        # db.py raises KeyError for an unknown store handle.
        raise HTTPException(status_code=404, detail=f"Unknown store {store!r}") from exc
    return DigestResponse(markdown=rendered.markdown, lines=rendered.lines)


class MemoryProjectionRequest(BaseModel):
    """The client's projection policy for one ``MEMORY.md`` render."""

    manifest: DocManifest = Field(
        description=(
            "The client's parsed memory manifest. A section that includes a file is refused."
        )
    )
    observer_project: str | None = Field(
        default=None, min_length=1, description="Render through this project observer."
    )
    exclude_source_tags: list[str] = Field(
        default_factory=list,
        description="Drop candidates held only by sources with one of these tags.",
    )
    fold_targets: list[str] = Field(
        default_factory=list,
        description="File URIs whose fold verdicts the client needs.",
    )


class MemoryProjectionResponse(BaseModel):
    """A rendered region body, its hash, and the fold verdicts asked of it."""

    body: str
    content_hash: str = Field(description="SHA-256 of `body`'s UTF-8 bytes.")
    fold_verdicts: dict[str, Literal["not-extracted", "not-in-region"] | None] = Field(
        default_factory=dict,
        description="Per fold target: null when the region represents the file, else why not.",
    )


@app.post(
    "/memory/projection/{store}",
    dependencies=[AUTH_GATE],
    response_model=MemoryProjectionResponse,
    responses={
        **_ERR404,
        422: {
            "model": ErrorResponse,
            "description": "Manifest names a client file this engine cannot read",
        },
    },
)
async def memory_projection_endpoint(
    store: str, req: MemoryProjectionRequest
) -> MemoryProjectionResponse:
    """Render the ``MEMORY.md`` region body for a thin client.

    The client sends its own manifest, so its projection policy governs, and
    splices the returned body into its memory directory itself; this route
    writes nothing. Deterministic, with no model call. ``fold_targets`` are
    answered against the body just rendered. 404 if the store handle is not
    configured on this engine; 422 for a manifest with a file section, which
    names a client path this engine cannot read.

    Bearer-gated regardless of ``api.require_auth_for_reads``, like
    ``GET /digest/{store}``: the region is the same confidential roll-up."""
    from particles.operations.projection.memory_region import (
        ClientFileSectionError,
        render_memory_region,
    )

    try:
        async with session_scope(store) as session:
            region = await render_memory_region(
                session,
                req.manifest,
                base_dir=None,
                observer_project=req.observer_project,
                exclude_source_tags=req.exclude_source_tags,
                fold_targets=req.fold_targets,
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown store {store!r}") from exc
    except ClientFileSectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MemoryProjectionResponse(
        body=region.body, content_hash=region.content_hash, fold_verdicts=region.fold_verdicts
    )


# ---------------------------------------------------------------------------
# Session-start exposure: the SessionStart hook records what each
# session was shown. Write-only here: no route reads the rows back (§5).
# ---------------------------------------------------------------------------


class SessionExposureResponse(BaseModel):
    """The id of the appended session-start row."""

    id: int


@app.post(
    "/session-exposures",
    dependencies=[AGENT_GATE],
    response_model=SessionExposureResponse,
    responses={
        403: {
            "model": ErrorResponse,
            "description": "Another project than the key's, or writes disabled on the store",
        },
        422: {"model": ErrorResponse, "description": "More beliefs than one request may name"},
    },
)
async def record_session_exposure_endpoint(
    req: SessionExposure, _rl: RateLimitDep, store: StoreDep
) -> SessionExposureResponse:
    """Append the beliefs one session was shown at session start.

    The SessionStart hook calls this after it has cut what it injects to its
    byte budget. Short ``p-`` ids are resolved to full ids here. Rows are only
    appended, and no route reads them back. ``store`` selects the store the
    digest was read from (404 on an unknown handle).

    The row records the key's principal and the engine's own receipt time, not
    a time the body names. A project-bound key records its own project, and a
    body naming another is refused (403). The store must be in
    ``mcp.write.enabled_stores`` (403), a request names at most
    ``storage.max_page_size`` beliefs (422), and the route is rate-limited per
    principal (``api.rate_limit_per_minute``).
    """
    from particles.store.session_exposure_store import record_session_exposure

    # Every refusal (the write-enable and project 403s, the size 422, and the
    # body's own 422) comes before the writer lock (security review
    # 2026-10-28-1, O47).
    _require_belief_writes_enabled(store)
    if len(req.beliefs) > get_config().storage.max_page_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"A session exposure names at most {get_config().storage.max_page_size} "
                f"beliefs (storage.max_page_size); this one names {len(req.beliefs)}."
            ),
        )
    principal = current_principal()
    project = _bound_project(req.project_key) or req.project_key
    if names_a_project(project):
        # The project key every other agent route applies: refused when it
        # reads as another project's, stored in NFC (security review
        # 2026-10-28-2, AA47, AA61).
        reason = lookalike_reason(project)
        if reason is not None:
            raise HTTPException(status_code=422, detail=f"project_key {reason}")
        project = canonical_project_key(project)
    else:
        project = ""
    exposure = req.model_copy(update={"project_key": project, "recorded_at": datetime.now(UTC)})
    async with session_scope(store, write=True) as session:
        row_id = await record_session_exposure(
            session, exposure, recorded_by=principal.id if principal is not None else None
        )
        await session.commit()
    return SessionExposureResponse(id=row_id)


# ---------------------------------------------------------------------------
# Operator-verb HTTP parity — thin mirrors of CLI write verbs.
# These expose the operator-judgment write surface (particle read + tags, links,
# subject alias / merge, trust rules / statements) over HTTP so they are
# reachable against the remote engine (fires trigger). Each is a thin
# mirror of one store / operation function. The particle belief-write trio
# (assert / supersede / retract) is mirrored further down — it
# wraps the §6.6 reconciliation ladder via particles.operations.agent_write.
# ---------------------------------------------------------------------------


@app.get(
    "/particles/{particle_id}", dependencies=[READ_GATE], response_model=Particle, responses=_ERR404
)
async def get_particle_endpoint(particle_id: str, session: SessionDep) -> Particle:
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


@app.post(
    "/particles/{particle_id}/tags",
    dependencies=[OPERATOR_GATE],
    response_model=ParticleTagsResponse,
)
async def add_particle_tags_endpoint(
    particle_id: str, req: ParticleTagsRequest
) -> ParticleTagsResponse:
    """Add tags to a particle (HTTP mirror of `particle tag`). Idempotent.

    The operator form: it tags any belief, so it needs a real bearer (503 under
    the dev-key skip) and 403 when belief writes are disabled. An agent
    tags its own beliefs through `POST /particles/tag`. The memory-server
    migration's tag is refused (400): only the migration stamps it."""
    from particles.store.taxonomy_store import ReservedTagError, add_particle_tags

    # Refused before the writer lock (security review 2026-10-28-2, AA6).
    _require_belief_writes_enabled()
    async with session_scope(write=True) as session:
        try:
            added = await add_particle_tags(session, particle_id, req.tags)
        except ReservedTagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return ParticleTagsResponse(particle_id=particle_id, tags=added)


@app.delete(
    "/particles/{particle_id}/tags",
    dependencies=[OPERATOR_GATE],
    response_model=ParticleTagsResponse,
)
async def remove_particle_tags_endpoint(
    particle_id: str, req: ParticleTagsRequest
) -> ParticleTagsResponse:
    """Remove tags from a particle (HTTP mirror of `particle untag`). Idempotent.

    The operator form, gated like `POST /particles/{particle_id}/tags`."""
    from particles.store.taxonomy_store import remove_particle_tags

    _require_belief_writes_enabled()
    async with session_scope(write=True) as session:
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
    # the project a project-bound agent surface links as. Set by
    # that surface (the MCP server's launch binding), not by the agent. With it,
    # a CO_EVIDENTIAL link between beliefs whose observer scopes differ is
    # refused (400), because the link would change those scopes.
    project_key: ProjectKey = None


class LinkDeleteResponse(BaseModel):
    deleted: bool


@app.post(
    "/links", dependencies=[OPERATOR_GATE], response_model=ParticleRelation, responses=_ERR400
)
async def create_link_endpoint(req: LinkRequest) -> ParticleRelation:
    """Create a typed relation between two particles (HTTP mirror of `links add`).

    The operator form: it links any two beliefs, so it needs a real bearer (503
    under the dev-key skip) and 403 when belief writes are disabled.
    An agent links its own beliefs through `POST /particles/link`."""
    from particles.core.schema import RelationType
    from particles.operations.agent_write import link_particles

    _require_belief_writes_enabled()
    try:
        rtype = RelationType(req.relation_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown relation_type {req.relation_type!r}"
        ) from exc
    # Both refusals come before the writer lock (security review 2026-10-28-2, AA6).
    async with session_scope(write=True) as session:
        try:
            relation = await link_particles(
                session,
                req.particle_a,
                req.particle_b,
                relation_type=rtype.value,
                confidence=req.confidence,
                project_key=req.project_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return relation


@app.delete(
    "/links", dependencies=[OPERATOR_GATE], response_model=LinkDeleteResponse, responses=_ERR400
)
async def delete_link_endpoint(req: LinkRequest) -> LinkDeleteResponse:
    """Delete a typed relation between two particles (HTTP mirror of `links remove`).

    It removes a relation of any provenance, a judged one included, so it is
    gated like `POST /links`."""
    from particles.core.schema import RelationType
    from particles.store.relation_store import delete_relation

    _require_belief_writes_enabled()
    try:
        rtype = RelationType(req.relation_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown relation_type {req.relation_type!r}"
        ) from exc
    async with session_scope(write=True) as session:
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
    dependencies=[OPERATOR_GATE],
    response_model=SubjectAliasResponse,
    responses=_ERR404,
)
async def add_subject_aliases_endpoint(
    subject_id: str, req: SubjectAliasRequest
) -> SubjectAliasResponse:
    """Append aliases to a subject (HTTP mirror of `subjects alias`). 404 if absent.

    An alias steers every later resolution, so it needs a real bearer (503 under
    the dev-key skip)."""
    from particles.store.subject_store import add_aliases, get_subject

    # The writer lock is taken in the handler, after the body validated, never
    # as a parameter dependency (security review 2026-10-30-1, AB54).
    async with session_scope(write=True) as session:
        if await get_subject(session, subject_id) is None:
            raise HTTPException(status_code=404, detail="Subject not found")
        subject, added = await add_aliases(session, subject_id, req.aliases)
        await session.commit()
    return SubjectAliasResponse(subject=subject, added=added)


class SubjectMergeRequest(BaseModel):
    target_id: str


class SubjectMergeManyRequest(BaseModel):
    """An N→1 merge: every source folds into ``target_id`` in one pass."""

    source_ids: list[str]
    target_id: str
    dry_run: bool = False


class SubjectMergeSource(BaseModel):
    """One merged source: its id and name (it is deleted), the links it moved, and
    its ``artifact`` refs: carried to the target, or withheld as another project's."""

    subject_id: str
    canonical_name: str
    particles_relinked: int
    refs_carried: list[str] = []
    refs_withheld: list[str] = []


class SubjectMergeResponse(BaseModel):
    """Outcome of a merge: the surviving target, the aliases it absorbed, and how
    many particle links were re-pointed. The sources are deleted, unless
    ``dry_run`` is set, when nothing was written."""

    subject: Subject
    aliases_added: list[str]
    particles_relinked: int
    sources: list[SubjectMergeSource] = []
    dry_run: bool = False


async def _merge_subjects_response(
    session: AsyncSession, source_ids: list[str], target_id: str, *, dry_run: bool
) -> SubjectMergeResponse:
    from particles.store.subject_store import (
        SubjectNotFoundError,
        merge_subjects_many,
        plan_subject_merge,
    )

    try:
        if dry_run:
            plan = await plan_subject_merge(session, source_ids, target_id)
            target = plan.target
        else:
            target, plan = await merge_subjects_many(session, source_ids, target_id)
    except SubjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not dry_run:
        await session.commit()
    return SubjectMergeResponse(
        subject=target,
        aliases_added=list(plan.aliases_added),
        particles_relinked=plan.particles_relinked,
        sources=[
            SubjectMergeSource(
                subject_id=s.subject_id,
                canonical_name=s.canonical_name,
                particles_relinked=len(s.particle_ids),
                refs_carried=[f"{r.namespace}:{r.id}" for r in s.refs_carried],
                refs_withheld=[f"{r.namespace}:{r.id}" for r in s.refs_withheld],
            )
            for s in plan.sources
        ],
        dry_run=dry_run,
    )


@app.post(
    "/subjects/merge",
    dependencies=[AUTH_GATE],
    response_model=SubjectMergeResponse,
    responses={**_ERR404, **_ERR400},
)
async def merge_subjects_many_endpoint(
    req: SubjectMergeManyRequest, session: SessionDep
) -> SubjectMergeResponse:
    """Merge several source subjects into one target (HTTP mirror of `subjects merge`).

    Irreversible unless `dry_run` is set: every source is deleted and its
    particles and aliases fold into the target, recorded as one operator event.
    Every source is checked before anything is written, so one bad source
    refuses the whole merge. 404 if a subject is absent or was already merged;
    400 if a source is the target or is listed twice. A real merge is refused
    (503) under the dev-key skip, even from loopback; a `dry_run` plan is served
    as before."""
    if req.dry_run:
        return await _merge_subjects_response(session, req.source_ids, req.target_id, dry_run=True)
    require_operator_credential()
    # The writer lock is taken only after the credential check, so a refused
    # merge never holds it (security review 2026-10-25, Q12).
    async with write_transaction(session):
        return await _merge_subjects_response(session, req.source_ids, req.target_id, dry_run=False)


@app.post(
    "/subjects/{source_id}/merge",
    dependencies=[OPERATOR_GATE],
    response_model=SubjectMergeResponse,
    responses={**_ERR404, **_ERR400},
)
async def merge_subjects_endpoint(source_id: str, req: SubjectMergeRequest) -> SubjectMergeResponse:
    """Merge one source subject into a target (the one-source form of `POST /subjects/merge`).

    Irreversible: the source is deleted and its particles + aliases fold into the
    target. 404 if either subject is absent; 400 on an invalid merge. Refused
    (503) under the dev-key skip, even from loopback."""
    # The writer lock is taken after the body validated (security review
    # 2026-10-30-1, AB54).
    async with session_scope(write=True) as session:
        return await _merge_subjects_response(session, [source_id], req.target_id, dry_run=False)


class TrustRuleRequest(BaseModel):
    """A source-trust rule. ``scope`` is ``"domain"`` (use ``score``)
    or ``"url_pattern"`` (use ``modifier``)."""

    scope: str
    pattern: str
    score: float | None = None
    modifier: float | None = None
    rationale: str | None = None


@app.post(
    "/trust/rules", dependencies=[OPERATOR_GATE], response_model=StatusResponse, responses=_ERR400
)
async def set_trust_rule_endpoint(req: TrustRuleRequest) -> StatusResponse:
    """Add or update a source-trust rule (HTTP mirror of `trust set`).

    Trust policy is the operator's judgment, and it decides future conflicts.
    Refused (503) under the dev-key skip, even from loopback: it needs a real
    bearer."""
    from particles.store.trust_store import upsert_trust_rule

    if req.scope not in ("domain", "url_pattern"):
        raise HTTPException(status_code=400, detail="scope must be 'domain' or 'url_pattern'")
    # The writer lock is taken after the body's refusals (security review
    # 2026-10-30-1, AB54).
    async with session_scope(write=True) as session:
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


@app.post(
    "/trust/statements",
    dependencies=[OPERATOR_GATE],
    response_model=TrustStatementResponse,
    responses=_ERR400,
)
async def set_trust_statement_endpoint(req: TrustStatementRequest) -> TrustStatementResponse:
    """Write an OPERATOR_DIRECT SourceTrustStatement and run its cascade.

    The cascade can resolve a held conflict, so this route carries a review's
    authority. Refused (503) under the dev-key skip, even from loopback: it
    needs a real bearer."""
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
        asserted_by=_actor_label(),
        basis=req.basis,
    )
    # The writer lock is taken after the body's refusals (security review
    # 2026-10-30-1, AB54).
    async with session_scope(write=True) as session:
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


def _require_belief_writes_enabled(store: str = DEFAULT_STORE) -> str:
    """Engine-side write-enablement gate; returns the store handle it cleared.

    The agent belief writes name their store; the operator routes
    still operate on the engine's default store. 403 if the store is
    not in the engine's ``mcp.write.enabled_stores``."""
    if store not in get_config().mcp.write.enabled_stores:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Belief writes are disabled on this engine: store {store!r} is not "
                "in mcp.write.enabled_stores (default-deny)."
            ),
        )
    return store


class ParticleAssertRequest(BaseModel):
    """Assert one belief (flagship). Trust-/status-/identity-bearing
    fields are constructed server-side (§4a); only the claim, its subjects,
    self-reported confidence, and provenance are caller-supplied."""

    content: str
    subject_names: list[str]
    confidence: float
    source_excerpt: str | None = None
    corpus_entry_id: str | None = None
    uncertainty_nature: str = "EPISTEMIC"
    tags: list[str] | None = None
    # the project a project-bound agent surface asserts as. Set by
    # that surface (the MCP server's launch binding), not by the agent. With it,
    # the belief's deposited excerpt carries the key, and a ``corpus_entry_id``
    # that is not one of that project's sources is refused (400).
    project_key: ProjectKey = None


class ParticleSupersedeRequest(ParticleAssertRequest):
    """Revise a belief: ``supersedes_id`` is retired to SUPERSEDED, then a
    successor is asserted. ``reason`` is optional on this
    agent-attributed path and, when given, is recorded on the audit event."""

    supersedes_id: str
    reason: str | None = None


class ParticleRetractRequest(BaseModel):
    """Retract a belief: ACTIVE → RETRACTED. ``reason`` is recorded on
    the audit event."""

    particle_id: str
    reason: str
    # the project a project-bound agent surface retracts as. With it,
    # a belief not in view for that project is refused (400).
    project_key: ProjectKey = None


class ParticleRetractEndpointResponse(BaseModel):
    """Outcome of a belief retraction."""

    particle_id: str
    verdict: str


def _bound_project(body_project: str | None) -> str | None:
    """The project an agent route writes as: a bound key's, else the body's.

    A body that names no project takes the key's; one that names the key's own
    is accepted; one that names another is refused with 403 and changes
    nothing. An unbound key, and an operator key, keep the body's value, which
    ``require_project_binding`` then checks as before.
    """
    principal = current_principal()
    if principal is None or principal.project is None:
        return body_project
    if body_project is None or not names_a_project(body_project):
        return principal.project
    if hides_a_character(body_project) or canonical_project_key(body_project) != (
        principal.project
    ):
        raise _provenance_refusal(
            principal,
            f"its writes are bound to project {principal.project!r}, not {body_project!r}",
        )
    return principal.project


def _agent_identity() -> str | None:
    """Who an agent route asserts as: an agent key's principal (rule 2).

    ``None`` for an operator key, so the operation falls back to
    ``mcp.write.asserter_identity`` exactly as every caller did before: the
    agent routes are the agent's surface whoever calls them, and an MCP server
    still holding the legacy key never gains the operator's attribution. The
    operator's own edits belong on the operator routes.
    """
    principal = current_principal()
    if principal is None or principal.can(Capability.OPERATOR_WRITE):
        return None
    return principal.id


def _mutating_identity() -> str | None:
    """:func:`_agent_identity` for a route that changes a belief; 403 for a writer's label.

    An agent key minted under a store writer's label (an extractor, the
    interchange import, a maintenance pass) matches every belief that writer
    asserted, so the own-beliefs rule alone let it retire or rewrite them
    (security review 2026-10-27-1, G30, G39). Changing those is an operator's
    act.
    """
    identity = _agent_identity()
    if identity is not None and identity in _keys.store_writer_labels():
        raise HTTPException(
            status_code=403,
            detail=(
                f"This key's principal, {identity}, is a store writer's label; an agent key "
                "under it changes no existing belief. The operator routes make that change."
            ),
        )
    return identity


def _agent_mutation_checks(store: str, body_project: str | None) -> tuple[str | None, str | None]:
    """Every refusal an agent's belief change can meet before its session opens.

    The write-enable 403, the writer-label 403 (:func:`_mutating_identity`) and
    the bound-project 403 (:func:`_bound_project`) all come before the store's
    writer lock is taken, so a refused request never holds it (security review
    2026-10-28-1, O47). Returns ``(identity, project_key)``.
    """
    _require_belief_writes_enabled(store)
    return _mutating_identity(), _bound_project(body_project)


@app.post(
    "/particles/assert",
    dependencies=[AGENT_GATE],
    response_model=AgentWriteResult,
    responses=_ERR400,
)
async def particle_assert_endpoint(
    req: ParticleAssertRequest,
    store: StoreDep,
    _rl: RateLimitDep,
) -> AgentWriteResult:
    """Assert a belief through the §6.6 ladder (HTTP mirror of MCP ``particle_assert``).

    A confirmed contradiction returns ``verdict=INCONSISTENCY_RAISED`` (consensus
    mode), a first-class result, not an error; so does re-asserting a claim an
    operator or reviewer retired (it is held for review), and
    ``HELD_FOR_REVIEW`` when the claim is already on an open hold. 400 on a
    granularity / provenance violation; 403 when belief writes are disabled for
    the store; 404 on an unknown ``store``. Rate-limited like
    ``/query``: the reconciliation ladder's contradiction and update probes call
    the model."""
    _require_belief_writes_enabled(store)
    project_key = _bound_project(req.project_key)
    identity = _agent_identity()
    try:
        # The body's own refusals come before the session opens, so a refused
        # body takes no lock (security review 2026-10-30-2, AC7).
        check_belief_body(
            content=req.content,
            subject_names=req.subject_names,
            uncertainty_nature=req.uncertainty_nature,
        )
        async with session_scope(store) as session:
            await require_project_binding(session, project_key)
            # The belief's writes (its subjects, excerpt, trust seed and the
            # ladder's insert) commit under the store's writer lock, so SQLite's
            # write lock is never held while the advisory lock is free (security
            # review 2026-10-30-1, AB4, AB68). The subject lookups, embedding
            # and model probes run before it (security review 2026-10-30-2,
            # AC1, AC3).
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
                project_key=project_key,
                identity=identity,
                commit_under_lock=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post(
    "/particles/supersede",
    dependencies=[AGENT_GATE],
    response_model=AgentWriteResult,
    responses=_ERR400,
)
async def particle_supersede_endpoint(
    req: ParticleSupersedeRequest,
    store: StoreDep,
    _rl: RateLimitDep,
) -> AgentWriteResult:
    """Revise a belief (HTTP mirror of MCP ``particle_supersede``). Own-beliefs-only;
    operator (HUMAN_REVIEW) particles are not agent-mutable. 400 on a guard or
    granularity violation; 403 when belief writes are disabled for
    the store; 404 on an unknown ``store``. Rate-limited like
    ``particle_assert``, whose ladder it runs."""
    identity, project_key = _agent_mutation_checks(store, req.project_key)
    try:
        check_belief_body(
            content=req.content,
            subject_names=req.subject_names,
            uncertainty_nature=req.uncertainty_nature,
        )
        async with session_scope(store) as session:
            await require_project_binding(session, project_key)
            # Retiring the predecessor and asserting its successor commit
            # together under the store's writer lock (security review
            # 2026-10-30-1, AB4, AB68); the successor's lookups and probes run
            # before it (security review 2026-10-30-2, AC1, AC3).
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
                reason=req.reason,
                project_key=project_key,
                identity=identity,
                commit_under_lock=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post(
    "/particles/retract",
    dependencies=[AGENT_GATE],
    response_model=ParticleRetractEndpointResponse,
    responses=_ERR400,
)
async def particle_retract_endpoint(
    req: ParticleRetractRequest, store: StoreDep
) -> ParticleRetractEndpointResponse:
    """Retract a belief (HTTP mirror of MCP ``particle_retract``). Own-beliefs-only;
    operator (HUMAN_REVIEW) particles are not agent-mutable. 400 on a guard
    violation or empty reason; 403 when belief writes are disabled for
    the store; 404 on an unknown ``store``."""
    identity, project_key = _agent_mutation_checks(store, req.project_key)
    async with session_scope(store, write=True) as session:
        try:
            await require_project_binding(session, project_key)
            await retract_belief(
                session,
                store=store,
                particle_id=req.particle_id,
                reason=req.reason,
                project_key=project_key,
                identity=identity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return ParticleRetractEndpointResponse(particle_id=req.particle_id, verdict="RETRACTED")


class AgentDepositRequest(BaseModel):
    """Deposit conversational material as the agent (MCP ``deposit_text``). The
    depositor, author, source type and scope tags are set server-side."""

    text: str
    tags: list[str] | None = None
    # set by a project-bound agent surface, not by the agent.
    project_key: ProjectKey = None


class AgentDepositResponse(BaseModel):
    """Identifiers for the agent's corpus entry and its snapshot."""

    entry_id: str
    snapshot_id: str


@app.post(
    "/particles/deposit",
    dependencies=[AGENT_GATE],
    response_model=AgentDepositResponse,
    responses=_ERR400,
)
async def agent_deposit_endpoint(
    req: AgentDepositRequest,
    _rl: RateLimitDep,
    store: StoreDep,
) -> AgentDepositResponse:
    """Deposit text as the agent (HTTP mirror of MCP ``deposit_text``). The entry
    is a CONVERSATION deposit attributed to the agent identity and deduplicated
    only onto the agent's own entries in the same scope, so identical bytes
    another principal or project deposited are never reused or rescoped. 400 on
    a guard violation; 403 when belief writes are disabled for the store; 404 on
    an unknown ``store``; 429 past ``api.rate_limit_per_minute``, since each
    deposit queues a snapshot for model extraction."""
    # The 403s come before the writer lock (security review 2026-10-28-1, O47).
    _require_belief_writes_enabled(store)
    project_key = _bound_project(req.project_key)
    async with session_scope(store, write=True) as session:
        try:
            await require_project_binding(session, project_key)
            entry_id, snapshot_id = await deposit_conversation_text(
                session,
                text=req.text,
                tags=req.tags,
                project_key=project_key,
                identity=_agent_identity(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return AgentDepositResponse(entry_id=entry_id, snapshot_id=snapshot_id)


class AgentLinkRequest(BaseModel):
    """An agent's link or unlink between two of its beliefs (MCP ``link_add`` /
    ``link_remove``). Only ``CO_EVIDENTIAL`` is agent-writable."""

    particle_a: str
    particle_b: str
    relation_type: str = "CO_EVIDENTIAL"
    # set by a project-bound agent surface, not by the agent.
    project_key: ProjectKey = None


class AgentTagRequest(BaseModel):
    """An agent's tag or untag of one of its beliefs (MCP ``particle_tag`` /
    ``particle_untag``)."""

    particle_id: str
    tags: list[str]
    # set by a project-bound agent surface, not by the agent.
    project_key: ProjectKey = None


@app.post(
    "/particles/link", dependencies=[AGENT_GATE], response_model=ParticleRelation, responses=_ERR400
)
async def agent_link_endpoint(req: AgentLinkRequest, store: StoreDep) -> ParticleRelation:
    """Link two beliefs as an agent (HTTP mirror of MCP ``link_add``). Both
    endpoints are own-beliefs-only, never HUMAN_REVIEW, and in view for
    ``project_key``. 400 on a guard violation; 403 when belief writes are
    disabled for the store; 404 on an unknown ``store``."""
    identity, project_key = _agent_mutation_checks(store, req.project_key)
    async with session_scope(store, write=True) as session:
        try:
            await require_project_binding(session, project_key)
            relation = await link_beliefs(
                session,
                req.particle_a,
                req.particle_b,
                relation_type=req.relation_type,
                project_key=project_key,
                identity=identity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return relation


@app.post(
    "/particles/unlink",
    dependencies=[AGENT_GATE],
    response_model=LinkDeleteResponse,
    responses=_ERR400,
)
async def agent_unlink_endpoint(req: AgentLinkRequest, store: StoreDep) -> LinkDeleteResponse:
    """Withdraw an agent-written link (HTTP mirror of MCP ``link_remove``). 400 on
    a guard violation, including a link an agent did not write; 403 when belief
    writes are disabled for the store; 404 on an unknown ``store``."""
    identity, project_key = _agent_mutation_checks(store, req.project_key)
    async with session_scope(store, write=True) as session:
        try:
            await require_project_binding(session, project_key)
            deleted = await unlink_beliefs(
                session,
                req.particle_a,
                req.particle_b,
                relation_type=req.relation_type,
                project_key=project_key,
                identity=identity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return LinkDeleteResponse(deleted=deleted)


@app.post(
    "/particles/tag",
    dependencies=[AGENT_GATE],
    response_model=ParticleTagsResponse,
    responses=_ERR400,
)
async def agent_tag_endpoint(req: AgentTagRequest, store: StoreDep) -> ParticleTagsResponse:
    """Tag a belief as an agent (HTTP mirror of MCP ``particle_tag``). 400 on a
    guard violation; 403 when belief writes are disabled for the store; 404 on
    an unknown ``store``."""
    identity, project_key = _agent_mutation_checks(store, req.project_key)
    async with session_scope(store, write=True) as session:
        try:
            await require_project_binding(session, project_key)
            added = await tag_belief(
                session,
                particle_id=req.particle_id,
                tags=req.tags,
                project_key=project_key,
                identity=identity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return ParticleTagsResponse(particle_id=req.particle_id, tags=added)


@app.post(
    "/particles/untag",
    dependencies=[AGENT_GATE],
    response_model=ParticleTagsResponse,
    responses=_ERR400,
)
async def agent_untag_endpoint(req: AgentTagRequest, store: StoreDep) -> ParticleTagsResponse:
    """Untag a belief as an agent (HTTP mirror of MCP ``particle_untag``). 400 on
    a guard violation; 403 when belief writes are disabled for the store; 404
    on an unknown ``store``."""
    identity, project_key = _agent_mutation_checks(store, req.project_key)
    async with session_scope(store, write=True) as session:
        try:
            await require_project_binding(session, project_key)
            removed = await untag_belief(
                session,
                particle_id=req.particle_id,
                tags=req.tags,
                project_key=project_key,
                identity=identity,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await session.commit()
    return ParticleTagsResponse(particle_id=req.particle_id, tags=removed)


# ---------------------------------------------------------------------------
# Operator-scoped belief mutation — supersede / retract a belief
# the operator does NOT own (incl. extracted beliefs), recorded as operator
# events. This is the curation write surface a curation queue actually needs: a
# queue is overwhelmingly extracted beliefs, which the own-beliefs-only agent
# path (above) cannot touch.
#
# SECURITY: the operator path can mutate ANY belief, so it MUST stay gated by
# `mcp.write.enabled_stores` + the bearer exactly like the agent
# belief-writes — `_require_belief_writes_enabled()` enforces the store gate and
# the route's `OPERATOR_GATE` the bearer. It refuses (503) while
# PARTICLES_API_KEY is the dev-key sentinel, so a loopback process with no key
# cannot reach it; since the agent own-beliefs routes above
# (`AGENT_GATE`) are refused on a keyless engine too.
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
    reason: str
    source_excerpt: str | None = None
    corpus_entry_id: str | None = None
    uncertainty_nature: str = "EPISTEMIC"
    tags: list[str] | None = None


class OperatorRetractRequest(BaseModel):
    """Operator-retract a belief by path id. ``reason`` is recorded
    on the audit event."""

    reason: str


@app.post(
    "/particles/{particle_id}/supersede",
    dependencies=[OPERATOR_GATE],
    response_model=AgentWriteResult,
    responses=_ERR400,
)
async def operator_supersede_endpoint(
    particle_id: str,
    req: OperatorSupersedeRequest,
    _rl: RateLimitDep,
) -> AgentWriteResult:
    """Operator-supersede any belief, incl. an extracted one.

    The "edit" gesture finally working on the extracted beliefs that fill a
    curation queue. Skips only the ownership check; the HUMAN_REVIEW guard and
    ACTIVE-status check still apply. ``reason`` is required and recorded on the
    audit event. 400 on a guard / granularity violation or an empty reason; 403
    when curation writes are disabled.
    Refused (503) under the dev-key skip, even from loopback: it needs a real
    bearer. Rate-limited like the agent ``/particles/supersede``: the
    replacement runs the same reconciliation ladder and its model calls
    (security review 2026-10-24-2, N10)."""
    store = _require_belief_writes_enabled()
    try:
        # The body's own refusals come before the session opens (security
        # review 2026-10-30-2, AC7).
        check_operator_supersede(reason=req.reason, confidence=req.confidence)
        check_belief_body(
            content=req.content,
            subject_names=req.subject_names,
            uncertainty_nature=req.uncertainty_nature,
        )
        # The retirement and its successor commit together under the store's
        # writer lock, after the 403 (security review 2026-10-30-1, AB68); the
        # successor's lookups and probes run before it (AC1, AC3).
        async with session_scope(store) as session:
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
                reason=req.reason,
                commit_under_lock=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post(
    "/particles/{particle_id}/retract",
    dependencies=[OPERATOR_GATE],
    response_model=ParticleRetractEndpointResponse,
    responses=_ERR400,
)
async def operator_retract_endpoint(
    particle_id: str, req: OperatorRetractRequest
) -> ParticleRetractEndpointResponse:
    """Operator-retract any belief, incl. an extracted one.

    Retract one extraction-/operator-sourced belief by id: the per-particle
    retract a curation card needs (vs the blunt per-source
    ``POST /corpus/{id}/retract``). Skips only the ownership check; the
    HUMAN_REVIEW guard and ACTIVE-status check still apply. 400 on a guard
    violation or empty reason; 403 when curation writes are disabled.
    Refused (503) under the dev-key skip, even from loopback: it needs a real bearer."""
    # The write-enable 403 comes before the writer lock (security review
    # 2026-10-28-1, O47).
    store = _require_belief_writes_enabled()
    async with session_scope(store, write=True) as session:
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
    """Assign a subject to an orphan particle, in place. Resolution accepts an
    explicit ``subject_id`` (link a Subject the operator picked) OR a
    ``subject_name`` run through the standard resolver (local → Wikidata →
    bare-local). Provide exactly one."""

    subject_id: str | None = None
    subject_name: str | None = None


@app.post(
    "/particles/{particle_id}/subjects",
    dependencies=[OPERATOR_GATE],
    response_model=AgentWriteResult,
    responses=_ERR400,
)
async def assign_subject_endpoint(
    particle_id: str, req: ParticleSubjectAssignRequest
) -> AgentWriteResult:
    """Attach a subject to a belief that has none (in place since).

    Writes the subject link on the same belief, which keeps its id, confidence,
    provenance and utility evidence; the response names that same id. 400 on a
    bad resolution, a guard violation, or a belief that already has a subject;
    403 when writes are disabled.
    Refused (503) under the dev-key skip, even from loopback: it needs a real
    bearer."""
    store = _require_belief_writes_enabled()
    # The subject link commits under the store's writer lock, after the 403, as
    # the extract pipeline's subject resolution does (security review
    # 2026-10-28-2, Z47). A name's live authority lookups run before the lock
    # (security review 2026-10-31-1).
    async with session_scope(store) as session:
        try:
            result = await assign_subject_belief(
                session,
                store=store,
                particle_id=particle_id,
                subject_id=req.subject_id,
                subject_name=req.subject_name,
                actor="http:/particles/{id}/subjects",
                commit_under_lock=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


class RelinkGatedRequest(BaseModel):
    """Relink the subjectless beliefs whose gated names a relink recovers.

    ``dry_run`` (the default) plans and reports without writing. ``tiers``
    defaults to ``subject_gate.relink_tiers``. ``sample`` returns that many
    planned relinks, chosen reproducibly by ``seed``, for a precision check."""

    dry_run: bool = True
    tiers: list[Literal[1, 2, 3]] | None = None
    sample: int = Field(default=0, ge=0, le=1000)
    seed: int = 0


@app.post(
    "/subjects/relink-gated",
    dependencies=[AUTH_GATE],
    response_model=RelinkReport,
    responses=_ERR400,
)
async def relink_gated_endpoint(req: RelinkGatedRequest, session: SessionDep) -> RelinkReport:
    """Link subjectless beliefs to the project-scoped subjects they name.

    Recovers each belief's names from its own gated-subjects record, its
    structured claim's subject term, or its backtick spans, and links them in
    place, one event per run. The `relink` gesture of the `gated_subjects`
    curation card. 403 when writes are disabled and `dry_run` is false. A
    real relink is refused (503) under the dev-key skip, even from loopback;
    a `dry_run` plan is served as before."""
    if not req.dry_run:
        require_operator_credential()
        _require_belief_writes_enabled()
    plan = await plan_gated_relink(session, tiers=req.tiers)
    if req.dry_run:
        return build_report(plan, sample=req.sample, seed=req.seed)
    # The relink commits under the store's writer lock; the plan's reads ran
    # outside it (security review 2026-10-30-1, AB68).
    async with write_transaction(session):
        result = await apply_gated_relink(session, plan, actor="http:/subjects/relink-gated")
    return build_report(plan, sample=req.sample, seed=req.seed, result=result)


# ---------------------------------------------------------------------------
# DB init helper
# ---------------------------------------------------------------------------


@app.post("/db/init", dependencies=[OPERATOR_GATE], response_model=StatusResponse)
async def db_init() -> StatusResponse:
    """Idempotently create database tables. Safe to call on a
    fully-initialised store; existing tables are not touched. It migrates the
    store, an operator step, so it is refused (503) under the dev-key skip;
    the local CLI's `db init` runs in-process."""
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
