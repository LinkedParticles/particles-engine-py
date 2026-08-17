"""``LocalBackend`` — the in-process backend.

This is today's path, unchanged: each method opens ``session_scope()`` and
calls the matching ``operations.*`` function, committing exactly where the verb
bodies did before the seam existed. It is the **default** backend (selected when
``engine.base_url`` is unset), so the single-machine local experience is
byte-for-byte preserved.

``LocalBackend`` touches the store, so it is an Engine-layer component (it may
import ``operations`` / ``db`` freely).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from particles.api.client.base import (
    AliasOutcome,
    BlobResult,
    DepositOutcome,
    DismissOutcome,
    ExtractOutcome,
    FollowTarget,
    MergeOutcome,
    ParticleDetail,
    RetractOutcome,
    SplitOutcome,
    SubjectDetail,
    TextDepositOutcome,
)
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
from particles.db import session_scope
from particles.extraction.general import PageStat

if TYPE_CHECKING:
    from particles.operations.agent_write import AgentWriteResult
    from particles.operations.deposit_suggest import DepositSuggestReport
    from particles.store.event_store import OperatorEvent


class LocalBackend:
    """In-process backend: ``session_scope()`` + ``operations.*`` (the default)."""

    remote = False

    async def health(self) -> str:
        from particles import __version__

        return __version__

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
        from particles.operations.deposit import deposit_url

        follow: list[tuple[str, str, str]] = []
        async with session_scope(write=True) as session:  # writer lock
            entry_id, snapshot_id = await deposit_url(
                session,
                url,
                deposited_by=deposited_by,
                source_type=source_type,
                tags=tags,
                follow_post_links=follow_post_links,
                follow_comment_links=follow_comment_links,
                out_follow_targets=follow,
            )
            await session.commit()
        return DepositOutcome(
            entry_id=entry_id,
            snapshot_id=snapshot_id,
            follow_targets=[FollowTarget(e, s, u) for e, s, u in follow],
        )

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
        from particles.core.schema import FetchPolicy, Mutability
        from particles.operations.deposit import deposit_file

        async with session_scope(write=True) as session:  # writer lock
            entry_id, snapshot_id = await deposit_file(
                session,
                path,
                deposited_by=deposited_by,
                source_type=source_type,
                tags=tags,
                content_date=content_date,
                mutability=Mutability(mutability) if mutability else None,
                fetch_policy=FetchPolicy(fetch_policy) if fetch_policy else None,
            )
            await session.commit()
        return DepositOutcome(entry_id=entry_id, snapshot_id=snapshot_id)

    async def deposit_file_split(
        self,
        path: Path,
        *,
        deposited_by: str,
        source_type: str | None,
        tags: list[str],
    ) -> list[DepositOutcome]:
        from particles.operations.deposit import split_file_by_date

        async with session_scope(write=True) as session:  # writer lock
            rows = await split_file_by_date(
                session,
                path,
                deposited_by=deposited_by,
                source_type=source_type,
                tags=tags,
            )
            await session.commit()
        return [DepositOutcome(entry_id=e, snapshot_id=s) for e, s in rows]

    async def extract(self, entry_id: str, snapshot_id: str, *, agent_id: str) -> ExtractOutcome:
        from particles.operations.extract import extract_snapshot

        page_stats: list[PageStat] = []
        carry_forward_ids: list[str] = []
        suppressed_ids: list[str] = []
        async with session_scope() as session:
            particles = await extract_snapshot(
                session,
                entry_id,
                snapshot_id,
                agent_id=agent_id,
                page_stats_out=page_stats,
                carry_forward_ids_out=carry_forward_ids,
                suppressed_ids_out=suppressed_ids,
            )
            await session.commit()
        return ExtractOutcome(
            entry_id=entry_id,
            particles=particles,
            page_stats=page_stats,
            carry_forward_ids=carry_forward_ids,
            suppressed_ids=suppressed_ids,
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        from particles.operations.query import query as query_op

        async with session_scope() as session:
            return await query_op(session, request)

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
        from particles.operations.graph_view import build_graph_data

        async with session_scope(store) as session:
            return await build_graph_data(
                session,
                subject_id=subject_id,
                query=query,
                inconsistency_id=inconsistency_id,
                manifest=manifest,
                section=section,
                hops=hops,
                history=history,
                as_of=as_of,
                max_nodes=max_nodes,
            )

    async def lint(self, *, fix: bool, semantic: bool, low_coverage_threshold: int) -> LintReport:
        from particles.operations.lint import run_lint

        async with session_scope() as session:
            return await run_lint(
                session,
                fix=fix,
                semantic=semantic,
                low_coverage_threshold=low_coverage_threshold,
            )

    async def review_list(self) -> list[Particle]:
        from particles.operations.review import list_inconsistencies

        async with session_scope() as session:
            return await list_inconsistencies(session)

    async def review_resolve(
        self,
        particle_id: str,
        action: ResolutionAction,
        reviewer_id: str,
        domain: str,
        note: str | None,
    ) -> ReviewParticle:
        from particles.operations.review import resolve

        async with session_scope() as session:
            return await resolve(session, particle_id, action, reviewer_id, domain, note)

    async def quality(self) -> QualityReport:
        from particles.operations.quality import get_quality_report

        async with session_scope() as session:
            return await get_quality_report(session)

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
        from particles.operations.reindex import reindex as reindex_op

        async with session_scope() as session:
            return await reindex_op(
                session,
                entry_ids=entry_ids,
                extractor_version=extractor_version,
                extractor_id=extractor_id,
                include_failed=include_failed,
                provider_model=provider_model,
                progress=progress,
                dry_run=dry_run,
                on_plan=on_plan,
                on_status=on_status,
            )

    # ------------------------------------------------------------------
    # Operator-verb surface — each lifts the verb's current
    # session_scope() store call behind the protocol, behaviour-preserving.
    # ------------------------------------------------------------------

    async def particle_show(self, particle_id: str) -> Particle | None:
        from particles.store.particle_store import get_particle

        async with session_scope() as session:
            return await get_particle(session, particle_id)

    async def subjects_list(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
        order: Literal["name", "degree"] = "name",
    ) -> list[Subject]:
        from particles.store.subject_store import list_all_subjects

        async with session_scope() as session:
            return await list_all_subjects(session, limit=limit, offset=offset, order=order)

    async def subjects_search(
        self, query: str, *, limit: int | None = None, offset: int = 0
    ) -> list[Subject]:
        from particles.store.subject_store import search_subjects

        async with session_scope() as session:
            return await search_subjects(session, query, limit=limit, offset=offset)

    async def subject_show(self, subject_id: str) -> Subject | None:
        from particles.store.subject_store import get_subject

        async with session_scope() as session:
            return await get_subject(session, subject_id)

    async def corpus_show(self, entry_id: str) -> CorpusEntry | None:
        from particles.corpus.store import get_entry

        async with session_scope() as session:
            return await get_entry(session, entry_id)

    async def corpus_blob(self, selector: str) -> BlobResult | None:
        from particles.corpus.deposit import load_blob
        from particles.corpus.store import resolve_snapshot_for_blob

        async with session_scope() as session:
            snap = await resolve_snapshot_for_blob(session, selector)
        if snap is None:
            return None
        try:
            content = load_blob(snap.content_hash)
        except FileNotFoundError:
            return None
        return BlobResult(
            content=content,
            snapshot_id=snap.snapshot_id,
            content_hash=snap.content_hash,
        )

    # ------------------------------------------------------------------
    # MCP-read surface — each lifts the routed MCP tool's store
    # call behind the protocol, behaviour-preserving for the local default path.
    # ------------------------------------------------------------------

    async def inconsistency_backrefs(self) -> dict[str, str]:
        from particles.store.particle_store import get_inconsistency_backrefs

        async with session_scope() as session:
            return await get_inconsistency_backrefs(session)

    async def contested_badges(self, particle_ids: list[str]) -> list[ContestedBadge | None]:
        from particles.operations.query.contested import compute_contested_badges
        from particles.store.particle_store import get_particles_by_ids

        if not particle_ids:
            return []
        async with session_scope() as session:
            present = list((await get_particles_by_ids(session, particle_ids)).values())
            badges = dict(
                zip(
                    [p.id for p in present],
                    await compute_contested_badges(session, present),
                    strict=True,
                )
            )
        return [badges.get(pid) for pid in particle_ids]

    async def particle_detail(self, particle_id: str) -> ParticleDetail:
        from sqlalchemy import select

        from particles.corpus.store import CorpusEntryRow
        from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern
        from particles.store.particle_store import ParticleRow
        from particles.store.subject_store import SubjectRow

        async with session_scope() as session:
            if len(particle_id) < 36:
                pattern = f"{escape_like_pattern(particle_id)}%"
                result = await session.execute(
                    select(ParticleRow).where(ParticleRow.id.like(pattern, escape=LIKE_ESCAPE))
                )
                rows = list(result.scalars())
                if not rows:
                    raise ValueError(f"No particle matches prefix {particle_id!r}.")
                if len(rows) > 1:
                    raise ValueError(
                        f"Ambiguous prefix {particle_id!r} matches {len(rows)} particles."
                    )
                row = rows[0]
            else:
                fetched = await session.get(ParticleRow, particle_id)
                if fetched is None:
                    raise ValueError(f"Particle {particle_id!r} not found.")
                row = fetched

            particle = row.to_model()

            subjects: list[dict[str, str]] = []
            for sid in particle.subject_ids:
                srow = await session.get(SubjectRow, sid)
                if srow is not None:
                    subjects.append({"id": srow.id, "canonical_name": srow.canonical_name})

            provenance: list[dict[str, Any]] = []
            for ref in particle.provenance:
                entry_uri: str | None = None
                source_type: str | None = None
                if ref.corpus_entry_id:
                    entry = await session.get(CorpusEntryRow, ref.corpus_entry_id)
                    if entry is not None:
                        entry_uri = entry.uri_r
                        source_type = entry.source_type
                provenance.append(
                    {
                        "type": ref.type.value,
                        "corpus_entry_id": ref.corpus_entry_id,
                        "uri_r": entry_uri,
                        "source_type": source_type,
                        "snapshot_id": ref.snapshot_id,
                    }
                )
        return ParticleDetail(particle=particle, subjects=subjects, provenance=provenance)

    async def particles_list(
        self, *, status: str | None, subject_id: str | None, limit: int, offset: int
    ) -> list[Particle]:
        from particles.core.status import Status
        from particles.store.particle_store import list_particles_filtered

        status_enum: Status | None = None
        if status is not None:
            try:
                status_enum = Status(status)
            except ValueError as exc:
                allowed = ", ".join(s.value for s in Status)
                raise ValueError(f"Unknown status {status!r}. Allowed: {allowed}.") from exc

        async with session_scope() as session:
            return await list_particles_filtered(
                session, status=status_enum, subject_id=subject_id, limit=limit, offset=offset
            )

    async def particles_by_fingerprint(self, fingerprint: str, *, limit: int) -> list[Particle]:
        from sqlalchemy import select

        from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern
        from particles.store.particle_store import ParticleRow

        if len(fingerprint) == 64:
            stmt = select(ParticleRow).where(ParticleRow.context_fingerprint == fingerprint)
        else:
            stmt = select(ParticleRow).where(
                ParticleRow.context_fingerprint.like(
                    f"{escape_like_pattern(fingerprint)}%", escape=LIKE_ESCAPE
                )
            )
        async with session_scope() as session:
            result = await session.execute(stmt.limit(limit))
            return [row.to_model() for row in result.scalars()]

    async def subject_detail(self, subject_id: str, *, particle_id_limit: int) -> SubjectDetail:
        from particles.store.subject_store import (
            count_particles_for_subject,
            get_particles_for_subject,
            get_subject,
        )

        async with session_scope() as session:
            subject = await get_subject(session, subject_id)
            if subject is None:
                raise ValueError(f"Subject {subject_id!r} not found.")
            particle_ids = await get_particles_for_subject(
                session, subject.id, limit=particle_id_limit
            )
            particle_count = await count_particles_for_subject(session, subject.id)
        return SubjectDetail(
            subject=subject, particle_ids=particle_ids, particle_count=particle_count
        )

    async def list_taxonomies(
        self, *, limit: int | None = None, offset: int = 0
    ) -> list[TaxonomyDefinition]:
        from particles.store.taxonomy_store import list_taxonomies as list_taxonomies_op

        async with session_scope() as session:
            return await list_taxonomies_op(session, limit=limit, offset=offset)

    async def list_corpus_entries(
        self, *, limit: int, source_type: str | None
    ) -> list[CorpusEntry]:
        from particles.corpus.store import list_entries

        async with session_scope() as session:
            return await list_entries(session, limit=limit, source_type=source_type)

    async def digest(self, store: str) -> str:
        from particles.operations.digest import build_digest

        return await build_digest(store)

    async def events_list(
        self,
        *,
        particle: str | None,
        subject: str | None,
        entry: str | None,
        event_type: str | None,
        limit: int,
    ) -> list[OperatorEvent]:
        from particles.store.event_store import OperatorEventType, list_events, ref_filter

        ref_kind, ref_id = ref_filter(particle=particle, subject=subject, corpus_entry=entry)
        etype = OperatorEventType(event_type) if event_type is not None else None
        async with session_scope() as session:
            return await list_events(
                session, ref_kind=ref_kind, ref_id=ref_id, event_type=etype, limit=limit
            )

    async def event_show(self, event_id: str) -> OperatorEvent | None:
        from particles.store.event_store import get_event

        async with session_scope() as session:
            return await get_event(session, event_id)

    async def trust_set(
        self,
        *,
        scope: str,
        pattern: str,
        score: float | None,
        modifier: float | None,
        rationale: str | None,
    ) -> None:
        from particles.store.trust_store import upsert_trust_rule

        async with session_scope(write=True) as session:  # writer lock
            await upsert_trust_rule(
                session,
                scope=scope,
                pattern=pattern,
                score=score,
                modifier=modifier,
                rationale=rationale,
            )
            await session.commit()

    async def trust_statement_set(self, statement: SourceTrustStatement) -> int:
        from particles.operations.trust import set_trust_statement

        async with session_scope(write=True) as session:  # writer lock
            resolved = await set_trust_statement(session, statement)
            await session.commit()
            return resolved

    async def subject_alias(self, subject_id: str, aliases: list[str]) -> AliasOutcome:
        from particles.store.subject_store import add_aliases, get_subject

        async with session_scope(write=True) as session:  # writer lock
            if await get_subject(session, subject_id) is None:
                raise ValueError(f"Subject {subject_id!r} not found")
            subject, added = await add_aliases(session, subject_id, aliases)
            await session.commit()
            return AliasOutcome(subject=subject, added=added)

    async def subject_merge(self, source_id: str, target_id: str) -> MergeOutcome:
        from particles.store.subject_store import merge_subjects

        async with session_scope(write=True) as session:  # writer lock
            subject, aliases_added, relinked = await merge_subjects(session, source_id, target_id)
            await session.commit()
            return MergeOutcome(
                subject=subject, aliases_added=aliases_added, particles_relinked=relinked
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
        from particles.ingest.subject_resolver import split_subject_resolving

        async with session_scope(write=True) as session:  # writer lock
            new_subject, relinked, not_bound = await split_subject_resolving(
                session,
                source_id=source_id,
                particle_ids=particle_ids,
                new_name=new_name,
                new_external_id=new_external_id,
            )
            # Dry-run: the resolver inserted the new Subject and re-linked
            # particles in this transaction; roll it back so nothing persists.
            if dry_run:
                await session.rollback()
            else:
                await session.commit()
            return SplitOutcome(
                new_subject=new_subject,
                relinked_particle_ids=relinked,
                not_bound_particle_ids=not_bound,
            )

    async def links_add(
        self, particle_a: str, particle_b: str, *, relation_type: str, confidence: float
    ) -> ParticleRelation:
        from particles.core.schema import RelationCreatedBy, RelationType
        from particles.store.relation_store import create_relation

        async with session_scope(write=True) as session:  # writer lock
            rel = await create_relation(
                session,
                particle_a,
                particle_b,
                RelationType(relation_type),
                RelationCreatedBy.MANUAL_CLI,
                confidence=confidence,
            )
            await session.commit()
            return rel

    async def links_remove(self, particle_a: str, particle_b: str, *, relation_type: str) -> bool:
        from particles.core.schema import RelationType
        from particles.store.relation_store import delete_relation

        async with session_scope(write=True) as session:  # writer lock
            removed = await delete_relation(
                session, particle_a, particle_b, RelationType(relation_type)
            )
            await session.commit()
            return removed

    async def particle_tag(self, particle_id: str, tags: list[str]) -> list[str]:
        from particles.store.taxonomy_store import add_particle_tags

        async with session_scope(write=True) as session:  # writer lock
            added = await add_particle_tags(session, particle_id, tags)
            await session.commit()
            return added

    async def particle_untag(self, particle_id: str, tags: list[str]) -> list[str]:
        from particles.store.taxonomy_store import remove_particle_tags

        async with session_scope(write=True) as session:  # writer lock
            removed = await remove_particle_tags(session, particle_id, tags)
            await session.commit()
            return removed

    async def corpus_retract(
        self, entry_id: str, *, reason: str | None, dry_run: bool
    ) -> RetractOutcome:
        from particles.operations.retract import plan_retraction, retract_entry

        async with session_scope(write=True) as session:  # writer lock
            if dry_run:
                plan = await plan_retraction(session, entry_id)
                return RetractOutcome(
                    entry_id=entry_id,
                    dry_run=True,
                    retracted_ids=[item.particle_id for item in plan.to_retract],
                    skipped=plan.skipped,
                )
            result = await retract_entry(session, entry_id, reason=reason)
            await session.commit()
            return RetractOutcome(
                entry_id=entry_id,
                dry_run=False,
                retracted_ids=result.retracted_ids,
                skipped=result.skipped,
            )

    async def links_suggest(
        self,
        *,
        subject_id: str | None,
        threshold: float | None,
        mode: SuggestMode,
        confirmed: bool,
    ) -> SuggestReport:
        from particles.operations.links_suggest import suggest_co_evidential

        async with session_scope() as session:
            return await suggest_co_evidential(
                session,
                subject_id=subject_id,
                threshold=threshold,
                mode=mode,
                confirmed=confirmed,
            )

    async def corpus_links_suggest(
        self, *, limit: int | None, min_sources: int | None
    ) -> DepositSuggestReport:
        from particles.operations.deposit_suggest import suggest_deposits

        async with session_scope() as session:
            return await suggest_deposits(session, limit=limit, min_sources=min_sources)

    async def corpus_links_dismiss(self, *, url: str, snooze_days: int | None) -> DismissOutcome:
        from particles.operations.deposit_suggest import dismiss_suggestion
        from particles.url_canonical import canonicalize_url

        canon = canonicalize_url(url)
        if canon is None:
            raise ValueError(f"Not a usable http(s) URL: {url!r}")
        async with session_scope() as session:
            until = await dismiss_suggestion(
                session,
                canonical_url=canon,
                actor="cli:corpus-links-dismiss",
                snooze_days=snooze_days,
            )
            await session.commit()
            return DismissOutcome(canonical_url=canon, suppressed_until=until)

    # ------------------------------------------------------------------
    # Agent belief-write surface — open ``session_scope(store)``,
    # delegate to ``operations.agent_write`` (the convergence point the engine
    # endpoints also call), and own the commit. Behaviour-preserving for the
    # in-process MCP write path lifted out of ``mcp/tools/write.py``.
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
        from particles.operations.agent_write import assert_belief

        async with session_scope(store) as session:
            result = await assert_belief(
                session,
                store=store,
                content=content,
                subject_names=subject_names,
                confidence=confidence,
                source_excerpt=source_excerpt,
                corpus_entry_id=corpus_entry_id,
                uncertainty_nature=uncertainty_nature,
                tags=tags,
            )
            await session.commit()
            return result

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
        from particles.operations.agent_write import supersede_belief

        async with session_scope(store) as session:
            result = await supersede_belief(
                session,
                store=store,
                supersedes_id=supersedes_id,
                content=content,
                subject_names=subject_names,
                confidence=confidence,
                source_excerpt=source_excerpt,
                corpus_entry_id=corpus_entry_id,
                uncertainty_nature=uncertainty_nature,
                tags=tags,
            )
            await session.commit()
            return result

    async def particle_retract(self, *, particle_id: str, reason: str, store: str) -> None:
        from particles.operations.agent_write import retract_belief

        async with session_scope(store) as session:
            await retract_belief(session, store=store, particle_id=particle_id, reason=reason)
            await session.commit()

    async def deposit_text(
        self,
        *,
        text: str,
        tags: list[str] | None,
        store: str,
        deposited_by: str | None = None,
        source_type: str | None = None,
    ) -> tuple[str, str]:
        from particles.core.schema import SourceType
        from particles.operations.agent_write import deposit_conversation_text
        from particles.operations.deposit import deposit_text as deposit_text_op

        async with session_scope(store, write=True) as session:  # writer lock
            if deposited_by is None:
                # Agent path: the asserter identity is resolved inside
                # deposit_conversation_text and stamped as deposited_by + author_id.
                entry_id, snapshot_id = await deposit_conversation_text(
                    session, text=text, tags=tags
                )
            else:
                # Operator path. Attribute to the named principal on
                # both axes so the §6.4 AUTHOR tier reads it as operator content.
                entry_id, snapshot_id = await deposit_text_op(
                    session,
                    text,
                    deposited_by,
                    source_type or SourceType.CONVERSATION,
                    tags,
                    author_id=deposited_by,
                )
            await session.commit()
            return entry_id, snapshot_id

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
        from particles.core.schema import FetchPolicy, Mutability
        from particles.operations.deposit import deposit_text_versioned

        async with session_scope(store, write=True) as session:  # writer lock
            entry_id, snapshot_id, unchanged = await deposit_text_versioned(
                session,
                text=text,
                uri_r=uri_r,
                source_type=source_type,
                mutability=Mutability(mutability),
                tags=tags,
                deposited_by=deposited_by,
                content_published_at=content_published_at,
                fetch_policy=FetchPolicy(fetch_policy) if fetch_policy else None,
            )
            await session.commit()
            return TextDepositOutcome(
                entry_id=entry_id, snapshot_id=snapshot_id, unchanged=unchanged
            )
