"""The ``Backend`` protocol + its result types.

Every daily CLI verb targets one :class:`Backend` whose methods return the
*shared* core Pydantic models. Two implementations satisfy it:

* :class:`particles.api.client.local.LocalBackend` — opens ``session_scope()``
  and calls ``operations.*`` in-process. This is today's path, unchanged, and
  the **default** when no engine is configured.
* :class:`particles.api.client.http.HttpBackend` — calls the FastAPI engine
  over HTTP/JSON and parses the responses back into the same models.

``operations/`` is the single convergence point: ``LocalBackend`` reaches it
directly; ``HttpBackend`` reaches it across the wire (HTTP → ``app.py`` handler
→ ``operations``). No verb logic forks on transport.

*Naming.* This was called "the ``Client`` protocol"; it is named
``Backend`` here to avoid colliding with "Client layer" (the
store-free package substrate). ``HttpBackend`` is store-free and could be
promoted into that enforced substrate later; for now the whole seam lives with
its consumer, the CLI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

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
from particles.extraction.general import PageStat

if TYPE_CHECKING:
    # Engine-layer result models the operator-verb / MCP methods return
    #. Imported only for annotations (this module is
    # ``from __future__`` lazy), so the seam stays import-light and the protocol
    # carries no Engine import at runtime.
    from particles.operations.agent_write import AgentWriteResult
    from particles.operations.deposit_suggest import DepositSuggestReport
    from particles.store.event_store import OperatorEvent


class NotYetRemoteError(RuntimeError):
    """Raised when a CLI feature has no engine endpoint and cannot run remotely.

    The graceful-degradation contract: a verb (or a verb sub-feature
    such as ``extract --all-pending`` or the author-enriched ``review`` detail
    view) that reads the store in ways the HTTP surface does not yet expose
    fails loudly in remote mode rather than silently doing the wrong thing. The
    message names the limitation and what the operator can do instead.
    """


@dataclass(frozen=True)
class FollowTarget:
    """A secondary corpus entry created deposit-time link following."""

    entry_id: str
    snapshot_id: str
    uri: str


@dataclass(frozen=True)
class DepositOutcome:
    """Result of a deposit: the entry + snapshot IDs, plus any followed targets.

    ``follow_targets`` is populated only by ``LocalBackend`` (the follow
    machinery runs against the local store). Over HTTP it is empty — the
    display-only follow list degrades gracefully; the secondary
    entries are still deposited engine-side.
    """

    entry_id: str
    snapshot_id: str
    follow_targets: list[FollowTarget] = field(default_factory=list)


@dataclass(frozen=True)
class TextDepositOutcome:
    """Result of a versioned text deposit.

    ``unchanged=True`` means the target entry's latest snapshot already carried
    this content hash, so nothing was written — the level-triggered harvest's
    skipped-as-duplicate signal. Over HTTP the flag is parsed from the
    ``DepositResponse.unchanged`` field.
    """

    entry_id: str
    snapshot_id: str
    unchanged: bool = False


@dataclass(frozen=True)
class ExtractOutcome:
    """Result of an extraction run: the particles plus local-only display extras.

    ``page_stats``, ``carry_forward_ids`` and ``suppressed_ids`` are populated
    only by ``LocalBackend`` — the ``/extract`` endpoint returns just the
    particles, so over HTTP these display extras are empty (graceful
    degradation). ``entry_id`` echoes the (possibly prefix-resolved) entry the
    run targeted.
    """

    entry_id: str
    particles: list[Particle]
    page_stats: list[PageStat] = field(default_factory=list)
    carry_forward_ids: list[str] = field(default_factory=list)
    #: one entry per candidate suppressed as an exact duplicate,
    #: holding the id of the existing ACTIVE particle it was folded into.
    suppressed_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator-verb result models — structured returns the operator
# verbs render. Each mirrors the JSON shape of its §5 / corpus parity endpoint,
# so ``LocalBackend`` (store call) and ``HttpBackend`` (HTTP parse) return the
# same dataclass and the verb body renders it identically across transports.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasOutcome:
    """Result of ``subjects alias``: the updated subject + aliases actually added."""

    subject: Subject
    added: list[str]


@dataclass(frozen=True)
class MergeOutcome:
    """Result of ``subjects merge``: the surviving target + what folded in."""

    subject: Subject
    aliases_added: list[str]
    particles_relinked: int


@dataclass(frozen=True)
class SplitOutcome:
    """Result of ``subjects split``: the new subject + which particles moved."""

    new_subject: Subject
    relinked_particle_ids: list[str]
    not_bound_particle_ids: list[str]


@dataclass(frozen=True)
class RetractOutcome:
    """Result of ``corpus retract``: which particles were (or would be) retracted.

    ``retracted_ids`` are the particles retracted, or — when ``dry_run`` — the
    ones that *would* be. ``skipped`` maps each non-live status to a count.
    """

    entry_id: str
    dry_run: bool
    retracted_ids: list[str]
    skipped: dict[str, int]


@dataclass(frozen=True)
class DismissOutcome:
    """Result of ``corpus links dismiss``: the canonical URL + suppression end."""

    canonical_url: str
    suppressed_until: datetime


@dataclass(frozen=True)
class BlobResult:
    """Result of ``corpus cat``: the raw stored bytes of a snapshot's blob.

    ``content`` is the exact deposited bytes; the CLI either renders a text
    preview locally (the default) or writes them raw. ``snapshot_id`` and
    ``content_hash`` identify which blob was served — over HTTP they come from
    the ``X-Snapshot-Id`` / ``X-Content-Hash`` response headers.
    """

    content: bytes
    snapshot_id: str
    content_hash: str


# ---------------------------------------------------------------------------
# MCP-read result models — the enriched shapes the routed MCP
# read tools render. ``LocalBackend`` fills the enrichment from the store;
# ``HttpBackend`` populates the core data from the engine and degrades the
# store-only enrichment (subject names, provenance URIs) gracefully, the same
# way the CLI's ``particle show`` does in remote mode.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleDetail:
    """Result of MCP ``particle_show``: the particle + resolved subjects + provenance.

    ``subjects`` is a list of ``{id, canonical_name}``; ``provenance`` a list of
    ``{type, corpus_entry_id, uri_r, source_type, snapshot_id}``. Over HTTP the
    name / URI enrichment is local-only: ``subjects`` is empty and the
    provenance ``uri_r`` / ``source_type`` are ``None``.
    """

    particle: Particle
    subjects: list[dict[str, str]]
    provenance: list[dict[str, Any]]


@dataclass(frozen=True)
class SubjectDetail:
    """Result of MCP ``subjects_show``: a subject + linked particle ids + true count."""

    subject: Subject
    particle_ids: list[str]
    particle_count: int


class Backend(Protocol):
    """The transport-agnostic surface every daily CLI verb targets."""

    #: ``True`` for the HTTP backend, ``False`` for the in-process local backend.
    #: Verbs consult this only to gate local-only rich features (prefix
    #: resolution, ``--all-pending``, author-enriched review) — never to fork the
    #: core operation call.
    remote: bool

    async def health(self) -> str:
        """Return the backend's SDK version — the ``version`` a ``/health`` probe reports.

        For :class:`LocalBackend` this is the in-process ``particles.__version__``;
        for :class:`HttpBackend` it is the version reported by the remote engine's
        ``GET /health``. ``particles --version`` uses this to surface the engine
        version alongside the client's in remote mode.
        """
        ...

    async def deposit_url(
        self,
        url: str,
        *,
        deposited_by: str,
        source_type: str | None,
        tags: list[str],
        follow_post_links: bool | None,
        follow_comment_links: bool | None,
    ) -> DepositOutcome: ...

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
    ) -> DepositOutcome: ...

    async def deposit_file_split(
        self,
        path: Path,
        *,
        deposited_by: str,
        source_type: str | None,
        tags: list[str],
    ) -> list[DepositOutcome]: ...

    async def extract(
        self, entry_id: str, snapshot_id: str, *, agent_id: str
    ) -> ExtractOutcome: ...

    async def query(self, request: QueryRequest) -> QueryResponse: ...

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
        """One scoped epistemic subgraph.

        Exactly one scope selector — ``subject_id`` / ``query`` /
        ``inconsistency_id`` / ``manifest``+``section`` — an
        unscoped render does not exist. Locally this runs
        ``operations.graph_view.build_graph_data`` in-process; remotely it is
        ``GET /graph`` on the canonical engine.
        """
        ...

    async def lint(
        self, *, fix: bool, semantic: bool, low_coverage_threshold: int
    ) -> LintReport: ...

    async def review_list(self) -> list[Particle]: ...

    async def review_resolve(
        self,
        particle_id: str,
        action: ResolutionAction,
        reviewer_id: str,
        domain: str,
        note: str | None,
    ) -> ReviewParticle: ...

    async def quality(self) -> QualityReport: ...

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
    ) -> dict[str, object]: ...

    # ------------------------------------------------------------------
    # Operator-verb surface (§2(a)) — the endpoint-backed operator
    # verbs route through these. Reads return ``None`` when absent; writes
    # return the structured outcome the verb renders. Prefix resolution and
    # rich (multi-store) rendering stay verb-side and local-only — these
    # methods take the IDs the §5 endpoints take and return the shared models.
    # ------------------------------------------------------------------

    # Reads ----------------------------------------------------------------

    async def particle_show(self, particle_id: str) -> Particle | None: ...

    async def subjects_list(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
        order: Literal["name", "degree"] = "name",
    ) -> list[Subject]: ...

    async def subjects_search(
        self, query: str, *, limit: int | None = None, offset: int = 0
    ) -> list[Subject]: ...

    async def subject_show(self, subject_id: str) -> Subject | None: ...

    async def corpus_show(self, entry_id: str) -> CorpusEntry | None: ...

    async def corpus_blob(self, selector: str) -> BlobResult | None: ...

    # MCP-read surface — the routed MCP read tools target these.

    async def inconsistency_backrefs(self) -> dict[str, str]: ...

    async def contested_badges(self, particle_ids: list[str]) -> list[ContestedBadge | None]: ...

    async def particle_detail(self, particle_id: str) -> ParticleDetail: ...

    async def particles_list(
        self, *, status: str | None, subject_id: str | None, limit: int, offset: int
    ) -> list[Particle]: ...

    async def particles_by_fingerprint(self, fingerprint: str, *, limit: int) -> list[Particle]: ...

    async def subject_detail(self, subject_id: str, *, particle_id_limit: int) -> SubjectDetail: ...

    async def list_taxonomies(
        self, *, limit: int | None = None, offset: int = 0
    ) -> list[TaxonomyDefinition]: ...

    async def list_corpus_entries(
        self, *, limit: int, source_type: str | None
    ) -> list[CorpusEntry]: ...

    async def digest(self, store: str) -> str: ...

    async def events_list(
        self,
        *,
        particle: str | None,
        subject: str | None,
        entry: str | None,
        event_type: str | None,
        limit: int,
    ) -> list[OperatorEvent]: ...

    async def event_show(self, event_id: str) -> OperatorEvent | None: ...

    # Writes ---------------------------------------------------------------

    async def trust_set(
        self,
        *,
        scope: str,
        pattern: str,
        score: float | None,
        modifier: float | None,
        rationale: str | None,
    ) -> None: ...

    async def trust_statement_set(self, statement: SourceTrustStatement) -> int: ...

    async def subject_alias(self, subject_id: str, aliases: list[str]) -> AliasOutcome: ...

    async def subject_merge(self, source_id: str, target_id: str) -> MergeOutcome: ...

    async def subject_split(
        self,
        *,
        source_id: str,
        particle_ids: list[str],
        new_name: str | None,
        new_external_id: str | None,
        dry_run: bool,
    ) -> SplitOutcome: ...

    async def links_add(
        self, particle_a: str, particle_b: str, *, relation_type: str, confidence: float
    ) -> ParticleRelation: ...

    async def links_remove(
        self, particle_a: str, particle_b: str, *, relation_type: str
    ) -> bool: ...

    async def particle_tag(self, particle_id: str, tags: list[str]) -> list[str]: ...

    async def particle_untag(self, particle_id: str, tags: list[str]) -> list[str]: ...

    async def corpus_retract(
        self, entry_id: str, *, reason: str | None, dry_run: bool
    ) -> RetractOutcome: ...

    # Analysis (writes under --apply) --------------------------------------

    async def links_suggest(
        self,
        *,
        subject_id: str | None,
        threshold: float | None,
        mode: SuggestMode,
        confirmed: bool,
    ) -> SuggestReport: ...

    async def corpus_links_suggest(
        self, *, limit: int | None, min_sources: int | None
    ) -> DepositSuggestReport: ...

    async def corpus_links_dismiss(
        self, *, url: str, snooze_days: int | None
    ) -> DismissOutcome: ...

    # Agent belief-write surface — the routed MCP write tools target
    # these. ``store`` is the (already allowlist-resolved) handle the local backend
    # writes; the HTTP backend ignores it and the engine writes its canonical
    # store, gated by the engine's own write-enablement.

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
    ) -> AgentWriteResult: ...

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
    ) -> AgentWriteResult: ...

    async def particle_retract(self, *, particle_id: str, reason: str, store: str) -> None: ...

    async def deposit_text(
        self,
        *,
        text: str,
        tags: list[str] | None,
        store: str,
        deposited_by: str | None = None,
        source_type: str | None = None,
    ) -> tuple[str, str]:
        """Deposit a literal string as one corpus entry; no fetch, no extraction.

        Two callers with deliberately different attribution:

        * **Agent (cheap deposit)** — both arguments omitted. The entry
          is attributed to the server-bound asserter identity and typed
          ``CONVERSATION``, so it lands under the §6.4 AUTHOR trust tier that
          ranks agent-asserted content below operator content.
        * **Operator (the CLI ``deposit --text`` half)**
          — ``deposited_by`` names the operator principal and is stamped as both
          ``deposited_by`` and ``author_id``, so a note the operator typed is
          not silently filed at agent trust.

        Passing ``deposited_by`` is what selects the operator path; it is not a
        cosmetic label.
        """
        ...

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
        """Deposit text under a stable URI-R identity; no-op when unchanged.

        The Claude Code harvest seam: repeated deposits of the same logical
        source (a growing session transcript, an edited memory file) land on
        one corpus entry, with unchanged content skipped engine-side.

        ``fetch_policy`` enrols the entry in the local
        refresh loop — ``"LAZY"`` for a source the nightly cycle should re-check
        on disk. It is reconciled onto an existing entry even when the content
        is unchanged; ``None`` leaves any stored policy alone.
        """
        ...
