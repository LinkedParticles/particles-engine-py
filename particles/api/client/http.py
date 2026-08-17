"""``HttpBackend`` — the remote backend.

Each method issues an HTTP/JSON request to the FastAPI engine over the OpenAPI contract and parses the response back into the *same* core Pydantic
models the local backend returns. It is **store-free**: it never opens a
session or imports ``operations`` — the engine runs that code server-side.

Connection settings come from ``config.engine`` (``base_url`` / ``timeout_seconds``);
the bearer token is a secret read via ``particles.secrets.get_engine_token_optional``
(omitted when unset, for a loopback dev engine running the dev-key skip).

Display-only extras the thin endpoints do not return (deposit follow-targets,
extract page-stats / carry-forward) come back empty here — a graceful
degradation whose extras the local backend still populates.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx

from particles.api.client.base import (
    AliasOutcome,
    BlobResult,
    DepositOutcome,
    DismissOutcome,
    ExtractOutcome,
    MergeOutcome,
    NotYetRemoteError,
    ParticleDetail,
    RetractOutcome,
    SplitOutcome,
    SubjectDetail,
    TextDepositOutcome,
)
from particles.config import get_config
from particles.core.schema import (
    ContestedBadge,
    CorpusEntry,
    GraphData,
    LintReport,
    Particle,
    ParticleRelation,
    QualityReport,
    QueryRequest,
    QueryResponse,
    ResolutionAction,
    ReviewParticle,
    SourceTrustStatement,
    Subject,
    SuggestMode,
    SuggestReport,
    TaxonomyDefinition,
)
from particles.secrets import get_engine_token_optional

if TYPE_CHECKING:
    from particles.operations.agent_write import AgentWriteResult
    from particles.operations.deposit_suggest import DepositSuggestReport
    from particles.store.event_store import OperatorEvent

log = logging.getLogger(__name__)


def _host_is_loopback(host: str | None) -> bool:
    """True iff ``host`` is loopback (``127.0.0.0/8`` / ``::1`` / ``localhost``).

    Fail-closed like ``particles.api.auth._is_loopback_host``: a non-IP
    hostname (other than the literal ``localhost``) is treated as non-loopback,
    so the F18 warning fires for it. We do not resolve DNS — a custom hostname
    aliasing loopback warns, the safe direction for a security notice.
    """
    if not host:
        return False
    h = host.strip().lower()
    if h == "localhost":
        return True
    h = h.removeprefix("[").removesuffix("]")
    h = h.split("%", 1)[0]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


@cache
def _warn_plaintext_token_once(base_url: str) -> None:
    """Emit a loud one-time warning for a bearer token over non-loopback ``http://`` (F18).

    ``lru_cache`` makes the warning fire at most once per distinct ``base_url``
    for the process lifetime. The documented remote-engine setup tunnels
    ``http://`` over Tailscale / SSH, so loopback ``http://`` is legitimate and
    stays silent — the caller only invokes this for the non-loopback plaintext
    case.
    """
    log.warning(
        "Engine bearer token is being sent over an unencrypted connection to %s. "
        "The token is exposed to anyone on the network path. Use https://, or "
        "tunnel http:// over loopback (Tailscale / SSH) so the token only "
        "traverses an encrypted hop.",
        base_url,
    )


def _maybe_warn_plaintext_token(base_url: str, *, has_token: bool) -> None:
    """Warn when a bearer token rides an unencrypted, non-loopback ``http://`` URL (F18).

    No-op unless a token is present, the scheme is not ``https``, and the host
    is not loopback. Never raises — the documented loopback/tunnel deployments
    legitimately use ``http://`` and must keep working silently.
    """
    if not has_token:
        return
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        return
    if _host_is_loopback(parsed.hostname):
        return
    _warn_plaintext_token_once(base_url)


def _detail(response: httpx.Response) -> str:
    """Best-effort engine error message — the ``{"detail": …}`` envelope or the body."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)


class EngineHttpError(RuntimeError):
    """A non-2xx response from the remote engine, carrying the engine's detail."""


class EngineUnreachableError(EngineHttpError):
    """The remote engine could not be reached at all — no response was received.

    Distinct from a plain :class:`EngineHttpError` (a non-2xx *response*): here
    the connection was refused, timed out, or DNS-failed, so the engine is down,
    the SSH tunnel is closed, or ``engine.base_url`` points somewhere wrong.
    Uncaught, the underlying ``httpx.TransportError`` dumps a raw stack trace
    whose message ("All connection attempts failed") names nothing actionable;
    this carries a message that names ``base_url`` and what to check. Subclasses
    ``EngineHttpError`` so a single ``except EngineHttpError`` (in the CLI
    ``run()`` helper and the ``links`` verb) covers both "couldn't talk to the
    engine" cases uniformly.
    """


class HttpBackend:
    """Remote backend: HTTP/JSON to the FastAPI engine, parsed into shared models."""

    remote = True

    async def health(self) -> str:
        body = await self._get("/health")
        return str(body["version"])

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        engine = get_config().engine
        if not engine.base_url:
            # The factory only builds HttpBackend when base_url is set; this
            # guards a direct instantiation with a misconfigured environment.
            raise RuntimeError("HttpBackend requires engine.base_url to be configured.")
        token = get_engine_token_optional()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        _maybe_warn_plaintext_token(engine.base_url, has_token=bool(token))
        async with httpx.AsyncClient(
            base_url=engine.base_url,
            timeout=engine.timeout_seconds,
            headers=headers,
        ) as client:
            yield client

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue one request to the engine, translating an unreachable engine.

        A refused connection, timeout, or DNS failure (the engine isn't running,
        the SSH tunnel is closed, ``engine.base_url`` is wrong) reaches us as an
        ``httpx.TransportError`` — no response was received. Uncaught, that dumps
        a stack trace ending in ``ConnectError: All connection attempts failed``,
        which tells the operator nothing. Translate it to an
        :class:`EngineUnreachableError` that names ``base_url`` and what to check.
        Non-2xx *responses* are the callers' concern (they carry the engine's
        detail via :func:`_detail`); this only covers the no-response case.

        ``httpx``'s ``get``/``post``/``delete`` shortcuts variously forbid a body
        or fix the verb, so every helper routes through ``client.request`` here.
        """
        try:
            async with self._client() as client:
                return await client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            base_url = get_config().engine.base_url
            raise EngineUnreachableError(
                f"Could not reach the Particles engine at {base_url}: {exc}. "
                f"Check that the engine is running and reachable — e.g. your SSH "
                f"tunnel is open and the engine is serving — or unset "
                f"engine.base_url to use the local store."
            ) from exc

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        resp = await self._request("POST", path, json=payload)
        if resp.status_code >= 400:
            raise EngineHttpError(f"{resp.status_code} from {path}: {_detail(resp)}")
        return resp.json()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._request("GET", path, params=params)
        if resp.status_code >= 400:
            raise EngineHttpError(f"{resp.status_code} from {path}: {_detail(resp)}")
        return resp.json()

    async def _delete(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        resp = await self._request("DELETE", path, json=payload)
        if resp.status_code >= 400:
            raise EngineHttpError(f"{resp.status_code} from {path}: {_detail(resp)}")
        return resp.json()

    async def _get_optional(self, path: str) -> Any | None:
        """GET that maps a 404 to ``None`` — for the ``show`` reads.

        The local ``get_*`` functions return ``None`` for an absent record; the
        engine 404s. This translates the 404 back to ``None`` so both backends
        return the same shape and the verb renders the same not-found message.
        """
        resp = await self._request("GET", path)
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise EngineHttpError(f"{resp.status_code} from {path}: {_detail(resp)}")
        return resp.json()

    async def deposit_url(
        self,
        url: str,
        *,
        deposited_by: str,
        source_type: str | None,
        tags: list[str],
        follow_post_links: bool | None,
        follow_comment_links: bool | None,
    ) -> DepositOutcome:
        body = await self._post(
            "/corpus/deposit/url",
            {
                "url": url,
                "deposited_by": deposited_by,
                "source_type": source_type,
                "tags": tags,
                "follow_post_links": follow_post_links,
                "follow_comment_links": follow_comment_links,
            },
        )
        return DepositOutcome(entry_id=body["entry_id"], snapshot_id=body["snapshot_id"])

    async def deposit_file(
        self,
        path: Path,
        *,
        deposited_by: str,
        source_type: str | None,
        tags: list[str],
        content_date: datetime | None,
        mutability: str | None = None,
        fetch_policy: str | None = None,
    ) -> DepositOutcome:
        # a remote deposit uploads the *bytes*; the engine writes them
        # to a temp path and records that as the URI-R. There is no durable local
        # file for the refresh ladder to re-stat, so opting one in would record a
        # promise the engine cannot keep. Refuse rather than silently ignore.
        if mutability is not None or fetch_policy is not None:
            raise EngineHttpError(
                "--mutability / --fetch-policy are local-mode only: a remote engine "
                "receives the uploaded bytes, not your file, so it cannot re-read the "
                "path later. Run the deposit against a local store."
            )
        data: dict[str, Any] = {"deposited_by": deposited_by, "tags": tags}
        if source_type is not None:
            data["source_type"] = source_type
        if content_date is not None:
            data["content_date"] = content_date.isoformat()
        with path.open("rb") as fh:
            resp = await self._request(
                "POST",
                "/corpus/deposit/file",
                files={"file": (path.name, fh)},
                data=data,
            )
        if resp.status_code >= 400:
            raise EngineHttpError(f"{resp.status_code} from /corpus/deposit/file: {_detail(resp)}")
        body = resp.json()
        return DepositOutcome(entry_id=body["entry_id"], snapshot_id=body["snapshot_id"])

    async def deposit_file_split(
        self,
        path: Path,
        *,
        deposited_by: str,
        source_type: str | None,
        tags: list[str],
    ) -> list[DepositOutcome]:
        # The /corpus/deposit/file endpoint returns one entry; there is no
        # multi-entry split endpoint yet. Refuse rather than silently depositing
        # the file whole (which would drop every entry's date but the first).
        raise NotYetRemoteError(
            "`deposit --split-by-date` is not available against a remote engine "
            "yet: the engine has no multi-entry split "
            "endpoint. Run it on the engine host directly, or unset "
            "engine.base_url to split against the local store."
        )

    async def extract(self, entry_id: str, snapshot_id: str, *, agent_id: str) -> ExtractOutcome:
        body = await self._post(
            "/extract",
            {"entry_id": entry_id, "snapshot_id": snapshot_id, "agent_id": agent_id},
        )
        particles = [Particle.model_validate(p) for p in body]
        # page_stats / carry_forward_ids are local-only display extras.
        return ExtractOutcome(entry_id=entry_id, particles=particles)

    async def query(self, request: QueryRequest) -> QueryResponse:
        body = await self._post("/query", request.model_dump(mode="json"))
        return QueryResponse.model_validate(body)

    async def graph(
        self,
        *,
        subject_id: str | None = None,
        query: str | None = None,
        inconsistency_id: str | None = None,
        manifest: str | None = None,
        section: str | None = None,
        hops: int = 1,
        history: bool = False,
        as_of: datetime | None = None,
        max_nodes: int | None = None,
        store: str = "default",
    ) -> GraphData:
        if subject_id is not None:
            scope = "subject"
        elif inconsistency_id is not None:
            scope = "inconsistency"
        elif manifest is not None:
            scope = "projection"
        else:
            scope = "query"
        params: dict[str, Any] = {
            "scope": scope,
            "hops": hops,
            "history": history,
            "store": store,
        }
        if subject_id is not None:
            params["subject_id"] = subject_id
        if query is not None:
            params["q"] = query
        if inconsistency_id is not None:
            params["inconsistency_id"] = inconsistency_id
        if manifest is not None:
            params["manifest"] = manifest
        if section is not None:
            params["section"] = section
        if as_of is not None:
            params["as_of"] = as_of.isoformat()
        if max_nodes is not None:
            params["max_nodes"] = max_nodes
        body = await self._get("/graph", params)
        return GraphData.model_validate(body)

    async def lint(self, *, fix: bool, semantic: bool, low_coverage_threshold: int) -> LintReport:
        body = await self._post(
            "/lint",
            {"fix": fix, "semantic": semantic, "low_coverage_threshold": low_coverage_threshold},
        )
        return LintReport.model_validate(body)

    async def review_list(self) -> list[Particle]:
        body = await self._get("/review")
        return [Particle.model_validate(p) for p in body]

    async def review_resolve(
        self,
        particle_id: str,
        action: ResolutionAction,
        reviewer_id: str,
        domain: str,
        note: str | None,
    ) -> ReviewParticle:
        body = await self._post(
            f"/review/{particle_id}",
            {
                "action": action.value,
                "reviewer_id": reviewer_id,
                "domain": domain,
                "note": note,
            },
        )
        return ReviewParticle.model_validate(body)

    async def quality(self) -> QualityReport:
        body = await self._get("/quality")
        return QualityReport.model_validate(body)

    async def reindex(
        self,
        *,
        entry_ids: list[str] | None,
        extractor_version: str | None,
        extractor_id: str | None,
        include_failed: bool,
        provider_model: str | None = None,
        progress: Callable[[str], None] | None = None,
        dry_run: bool = False,
        on_plan: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        # progress/on_plan/on_status are local-stderr streams with no HTTP
        # analogue; ignored here. The plan still comes back in the response body.
        body = await self._post(
            "/reindex",
            {
                "entry_ids": entry_ids,
                "extractor_version": extractor_version,
                "extractor_id": extractor_id,
                "include_failed": include_failed,
                "provider_model": provider_model,
                "dry_run": dry_run,
            },
        )
        return dict(body)

    # ------------------------------------------------------------------
    # Operator-verb surface — thin mirrors of the §5 + corpus
    # parity endpoints, parsed back into the same shared models LocalBackend
    # returns. The engine takes full IDs (no prefix resolution).
    # ------------------------------------------------------------------

    async def particle_show(self, particle_id: str) -> Particle | None:
        body = await self._get_optional(f"/particles/{particle_id}")
        return Particle.model_validate(body) if body is not None else None

    async def subjects_list(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
        order: Literal["name", "degree"] = "name",
    ) -> list[Subject]:
        params: dict[str, Any] = {"offset": offset}
        if limit is not None:
            params["limit"] = limit
        # Only sent when non-default, so an older engine (pre-`order`) keeps
        # working — FastAPI would ignore the unknown param anyway, but the
        # request should transcribe intent, not defaults.
        if order != "name":
            params["order"] = order
        body = await self._get("/subjects", params=params)
        return [Subject.model_validate(s) for s in body]

    async def subjects_search(
        self, query: str, *, limit: int | None = None, offset: int = 0
    ) -> list[Subject]:
        params: dict[str, Any] = {"q": query, "offset": offset}
        if limit is not None:
            params["limit"] = limit
        body = await self._get("/subjects/search", params=params)
        return [Subject.model_validate(s) for s in body]

    async def subject_show(self, subject_id: str) -> Subject | None:
        body = await self._get_optional(f"/subjects/{subject_id}")
        return Subject.model_validate(body) if body is not None else None

    async def corpus_show(self, entry_id: str) -> CorpusEntry | None:
        body = await self._get_optional(f"/corpus/{entry_id}")
        return CorpusEntry.model_validate(body) if body is not None else None

    async def corpus_blob(self, selector: str) -> BlobResult | None:
        resp = await self._request("GET", f"/corpus/blob/{selector}")
        if resp.status_code == 404:
            return None
        if resp.status_code == 400:
            # Ambiguous prefix — mirror the local backend's ValueError so the
            # CLI renders one friendly message across both transports.
            raise ValueError(_detail(resp))
        if resp.status_code >= 400:
            raise EngineHttpError(
                f"{resp.status_code} from /corpus/blob/{selector}: {_detail(resp)}"
            )
        return BlobResult(
            content=resp.content,
            snapshot_id=resp.headers.get("X-Snapshot-Id", ""),
            content_hash=resp.headers.get("X-Content-Hash", ""),
        )

    # ------------------------------------------------------------------
    # MCP-read surface — thin GETs against the engine's read
    # endpoints. Store-only enrichment (subject names, provenance URIs) degrades
    # gracefully here, the same way the CLI's reads do in remote mode.
    # ------------------------------------------------------------------

    async def inconsistency_backrefs(self) -> dict[str, str]:
        body = await self._get("/particles/contested")
        return {str(k): str(v) for k, v in body.items()}

    async def contested_badges(self, particle_ids: list[str]) -> list[ContestedBadge | None]:
        if not particle_ids:
            return []
        body = await self._post("/particles/contested-badges", {"particle_ids": particle_ids})
        badges = {str(k): ContestedBadge.model_validate(v) for k, v in body.items()}
        return [badges.get(pid) for pid in particle_ids]

    async def particle_detail(self, particle_id: str) -> ParticleDetail:
        body = await self._get_optional(f"/particles/{particle_id}")
        if body is None:
            raise ValueError(f"Particle {particle_id!r} not found.")
        particle = Particle.model_validate(body)
        # Subject names and provenance URIs need store reads the engine GET does
        # not return — local-only enrichment (graceful degradation).
        provenance = [
            {
                "type": ref.type.value,
                "corpus_entry_id": ref.corpus_entry_id,
                "uri_r": None,
                "source_type": None,
                "snapshot_id": ref.snapshot_id,
            }
            for ref in particle.provenance
        ]
        return ParticleDetail(particle=particle, subjects=[], provenance=provenance)

    async def particles_list(
        self, *, status: str | None, subject_id: str | None, limit: int, offset: int
    ) -> list[Particle]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        if subject_id is not None:
            params["subject_id"] = subject_id
        body = await self._get("/particles", params=params)
        return [Particle.model_validate(p) for p in body]

    async def particles_by_fingerprint(self, fingerprint: str, *, limit: int) -> list[Particle]:
        body = await self._get(
            "/particles/search", params={"fingerprint": fingerprint, "limit": limit}
        )
        return [Particle.model_validate(p) for p in body]

    async def subject_detail(self, subject_id: str, *, particle_id_limit: int) -> SubjectDetail:
        subject_body = await self._get_optional(f"/subjects/{subject_id}")
        if subject_body is None:
            raise ValueError(f"Subject {subject_id!r} not found.")
        ids_body = await self._get(
            f"/subjects/{subject_id}/particle-ids", params={"limit": particle_id_limit}
        )
        return SubjectDetail(
            subject=Subject.model_validate(subject_body),
            particle_ids=list(ids_body["particle_ids"]),
            particle_count=int(ids_body["particle_count"]),
        )

    async def list_taxonomies(
        self, *, limit: int | None = None, offset: int = 0
    ) -> list[TaxonomyDefinition]:
        params: dict[str, Any] = {"offset": offset}
        if limit is not None:
            params["limit"] = limit
        body = await self._get("/taxonomies", params=params)
        return [TaxonomyDefinition.model_validate(t) for t in body]

    async def list_corpus_entries(
        self, *, limit: int, source_type: str | None
    ) -> list[CorpusEntry]:
        params: dict[str, Any] = {"limit": limit}
        if source_type is not None:
            params["source_type"] = source_type
        body = await self._get("/corpus", params=params)
        return [CorpusEntry.model_validate(e) for e in body]

    async def digest(self, store: str) -> str:
        body = await self._get(f"/digest/{store}")
        return str(body["markdown"])

    async def events_list(
        self,
        *,
        particle: str | None,
        subject: str | None,
        entry: str | None,
        event_type: str | None,
        limit: int,
    ) -> list[OperatorEvent]:
        from particles.store.event_store import OperatorEvent

        params: dict[str, Any] = {"limit": limit}
        if particle is not None:
            params["particle"] = particle
        if subject is not None:
            params["subject"] = subject
        if entry is not None:
            params["entry"] = entry
        if event_type is not None:
            params["type"] = event_type
        body = await self._get("/events", params=params)
        return [OperatorEvent.model_validate(e) for e in body]

    async def event_show(self, event_id: str) -> OperatorEvent | None:
        from particles.store.event_store import OperatorEvent

        body = await self._get_optional(f"/events/{event_id}")
        return OperatorEvent.model_validate(body) if body is not None else None

    async def trust_set(
        self,
        *,
        scope: str,
        pattern: str,
        score: float | None,
        modifier: float | None,
        rationale: str | None,
    ) -> None:
        await self._post(
            "/trust/rules",
            {
                "scope": scope,
                "pattern": pattern,
                "score": score,
                "modifier": modifier,
                "rationale": rationale,
            },
        )

    async def trust_statement_set(self, statement: SourceTrustStatement) -> int:
        body = await self._post(
            "/trust/statements",
            {
                "domain": statement.domain,
                "source_ref_type": statement.source_ref.type.value,
                "source_ref_value": statement.source_ref.value,
                "trust_rank": statement.trust_rank,
                "basis": statement.basis,
            },
        )
        return int(body["cascade_resolved"])

    async def subject_alias(self, subject_id: str, aliases: list[str]) -> AliasOutcome:
        body = await self._post(f"/subjects/{subject_id}/aliases", {"aliases": aliases})
        return AliasOutcome(
            subject=Subject.model_validate(body["subject"]), added=list(body["added"])
        )

    async def subject_merge(self, source_id: str, target_id: str) -> MergeOutcome:
        body = await self._post(f"/subjects/{source_id}/merge", {"target_id": target_id})
        return MergeOutcome(
            subject=Subject.model_validate(body["subject"]),
            aliases_added=list(body["aliases_added"]),
            particles_relinked=int(body["particles_relinked"]),
        )

    async def subject_split(
        self,
        *,
        source_id: str,
        particle_ids: list[str],
        new_name: str | None,
        new_external_id: str | None,
        dry_run: bool,
    ) -> SplitOutcome:
        if dry_run:
            # POST /subjects/{id}/split always commits; there is no dry-run
            # query. Refuse rather than silently performing a real split.
            raise NotYetRemoteError(
                "`subjects split --dry-run` is not available against a remote "
                "engine yet: the split endpoint has no "
                "dry-run. Run it on the engine host, or drop --dry-run to "
                "perform the split."
            )
        body = await self._post(
            f"/subjects/{source_id}/split",
            {
                "particle_ids": particle_ids,
                "new_name": new_name,
                "new_external_id": new_external_id,
            },
        )
        return SplitOutcome(
            new_subject=Subject.model_validate(body["new_subject"]),
            relinked_particle_ids=list(body["relinked_particle_ids"]),
            not_bound_particle_ids=list(body["not_bound_particle_ids"]),
        )

    async def links_add(
        self, particle_a: str, particle_b: str, *, relation_type: str, confidence: float
    ) -> ParticleRelation:
        body = await self._post(
            "/links",
            {
                "particle_a": particle_a,
                "particle_b": particle_b,
                "relation_type": relation_type,
                "confidence": confidence,
            },
        )
        return ParticleRelation.model_validate(body)

    async def links_remove(self, particle_a: str, particle_b: str, *, relation_type: str) -> bool:
        body = await self._delete(
            "/links",
            {"particle_a": particle_a, "particle_b": particle_b, "relation_type": relation_type},
        )
        return bool(body["deleted"])

    async def particle_tag(self, particle_id: str, tags: list[str]) -> list[str]:
        body = await self._post(f"/particles/{particle_id}/tags", {"tags": tags})
        return list(body["tags"])

    async def particle_untag(self, particle_id: str, tags: list[str]) -> list[str]:
        body = await self._delete(f"/particles/{particle_id}/tags", {"tags": tags})
        return list(body["tags"])

    async def corpus_retract(
        self, entry_id: str, *, reason: str | None, dry_run: bool
    ) -> RetractOutcome:
        body = await self._post(
            f"/corpus/{entry_id}/retract", {"reason": reason, "dry_run": dry_run}
        )
        return RetractOutcome(
            entry_id=body["entry_id"],
            dry_run=bool(body["dry_run"]),
            retracted_ids=list(body["retracted_ids"]),
            skipped=dict(body["skipped"]),
        )

    async def links_suggest(
        self,
        *,
        subject_id: str | None,
        threshold: float | None,
        mode: SuggestMode,
        confirmed: bool,
    ) -> SuggestReport:
        body = await self._post(
            "/links/suggest",
            {
                "subject_id": subject_id,
                "threshold": threshold,
                "mode": mode.value,
                "confirmed": confirmed,
            },
        )
        return SuggestReport.model_validate(body)

    async def corpus_links_suggest(
        self, *, limit: int | None, min_sources: int | None
    ) -> DepositSuggestReport:
        from particles.operations.deposit_suggest import DepositSuggestReport

        body = await self._post(
            "/corpus/links/suggest", {"limit": limit, "min_sources": min_sources}
        )
        return DepositSuggestReport.model_validate(body)

    async def corpus_links_dismiss(self, *, url: str, snooze_days: int | None) -> DismissOutcome:
        body = await self._post("/corpus/links/dismiss", {"url": url, "snooze_days": snooze_days})
        return DismissOutcome(
            canonical_url=body["canonical_url"],
            suppressed_until=datetime.fromisoformat(body["suppressed_until"]),
        )

    # ------------------------------------------------------------------
    # Agent belief-write surface — POST to the engine belief-write
    # endpoints, which run §6.6 reconciliation + construction server-side and
    # govern write-enablement themselves (§5). ``store`` is ignored here: the
    # engine writes its canonical (default) store; multi-store routing is
    # deferred.
    # ------------------------------------------------------------------

    async def particle_assert(
        self,
        *,
        content: str,
        subject_names: list[str],
        confidence: float,
        source_excerpt: str | None,
        corpus_entry_id: str | None,
        uncertainty_nature: str,
        tags: list[str] | None,
        store: str,
    ) -> AgentWriteResult:
        from particles.operations.agent_write import AgentWriteResult

        body = await self._post(
            "/particles/assert",
            {
                "content": content,
                "subject_names": subject_names,
                "confidence": confidence,
                "source_excerpt": source_excerpt,
                "corpus_entry_id": corpus_entry_id,
                "uncertainty_nature": uncertainty_nature,
                "tags": tags,
            },
        )
        return AgentWriteResult.model_validate(body)

    async def particle_supersede(
        self,
        *,
        supersedes_id: str,
        content: str,
        subject_names: list[str],
        confidence: float,
        source_excerpt: str | None,
        corpus_entry_id: str | None,
        uncertainty_nature: str,
        tags: list[str] | None,
        store: str,
    ) -> AgentWriteResult:
        from particles.operations.agent_write import AgentWriteResult

        body = await self._post(
            "/particles/supersede",
            {
                "supersedes_id": supersedes_id,
                "content": content,
                "subject_names": subject_names,
                "confidence": confidence,
                "source_excerpt": source_excerpt,
                "corpus_entry_id": corpus_entry_id,
                "uncertainty_nature": uncertainty_nature,
                "tags": tags,
            },
        )
        return AgentWriteResult.model_validate(body)

    async def particle_retract(self, *, particle_id: str, reason: str, store: str) -> None:
        await self._post("/particles/retract", {"particle_id": particle_id, "reason": reason})

    async def deposit_text(
        self,
        *,
        text: str,
        tags: list[str] | None,
        store: str,
        deposited_by: str | None = None,
        source_type: str | None = None,
    ) -> tuple[str, str]:
        # The agent-attributed deposit reuses the generic corpus deposit endpoint
        #, carrying the asserter identity as deposited_by + author_id so
        # the CONVERSATION entry is attributed exactly as the local path attributes
        # it. The engine governs write-enablement at the belief layer; a deposit is
        # benign archival, gated by the local offering allowlist + bearer auth.
        # An explicit deposited_by is the operator path and attributes
        # to that principal instead — same endpoint, different identity.
        identity = deposited_by or get_config().mcp.write.asserter_identity
        body = await self._post(
            "/corpus/deposit/text",
            {
                "text": text,
                "source_type": source_type or "CONVERSATION",
                "deposited_by": identity,
                "tags": tags or [],
                "author_id": identity,
            },
        )
        return body["entry_id"], body["snapshot_id"]

    async def deposit_text_at_uri(
        self,
        *,
        text: str,
        uri_r: str,
        source_type: str,
        mutability: str,
        tags: list[str],
        deposited_by: str,
        content_published_at: datetime | None,
        store: str,
        fetch_policy: str | None = None,
    ) -> TextDepositOutcome:
        # The harvest deposit rides the same generic corpus deposit
        # endpoint; ``uri_r`` switches the engine handler onto the versioned
        # path (one entry per logical source, unchanged re-deposit skipped).
        body = await self._post(
            "/corpus/deposit/text",
            {
                "text": text,
                "uri_r": uri_r,
                "source_type": source_type,
                "mutability": mutability,
                "fetch_policy": fetch_policy,
                "deposited_by": deposited_by,
                "tags": tags,
                "content_published_at": (
                    content_published_at.isoformat() if content_published_at else None
                ),
            },
        )
        return TextDepositOutcome(
            entry_id=body["entry_id"],
            snapshot_id=body["snapshot_id"],
            unchanged=bool(body.get("unchanged", False)),
        )
