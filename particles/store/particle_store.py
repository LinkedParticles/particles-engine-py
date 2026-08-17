"""SQLAlchemy ORM models and repository for the particle store (§8).

Embeddings are stored as numpy arrays serialized to bytes and loaded into memory
for cosine similarity search — acceptable for the v0.2 target of ≤10⁵ particles.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import DateTime, Float, Index, String, Text, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.duplicate_key import content_hash
from particles.core.fingerprint import context_fingerprint
from particles.core.schema import (
    AssertionModality,
    CanonicalForm,
    ClaimTerm,
    Confidence,
    ContributorRef,
    ExtractorRef,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    StructuredClaim,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.db import Base
from particles.embeddings import EmbeddingProfile, get_embedding_model_id, get_embedding_profile
from particles.extraction.incremental import register_carry_forward_lookup

log = logging.getLogger(__name__)

# the model a NULL ``embedding_model_id`` is grandfathered to —
# all-MiniLM-L6-v2 was the only embedding model in use before the marker
# existed, so legacy vectors are treated as having come from it. Distinct from
# the *current* id (``embeddings.get_embedding_model_id()``) so that a future
# swap away from this default correctly flags every legacy vector as stale.
LEGACY_EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"


def structured_claim_payload(claim: StructuredClaim) -> str:
    """Serialise the *triple half* of an annotation.

    The generator stamp is deliberately excluded — it lives in its own columns
    so the backfill's scope query and the coverage report stay SQL. Storing it
    twice would let the two copies drift.
    """
    payload: dict[str, Any] = {
        "subject": claim.subject.model_dump(mode="json", exclude_none=True),
        "predicate": claim.predicate.model_dump(mode="json", exclude_none=True),
        "object": claim.object.model_dump(mode="json", exclude_none=True),
    }
    if claim.subject_id is not None:
        payload["subject_id"] = claim.subject_id
    return json.dumps(payload)


class ParticleRow(Base):
    __tablename__ = "particles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_value: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    confidence_variance: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_calibration_source: Mapped[str] = mapped_column(String, nullable=False)
    confidence_calibration_method: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence_calibration_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    uncertainty_nature: Mapped[str] = mapped_column(String, nullable=False)
    asserted_by: Mapped[str] = mapped_column(String, nullable=False)
    asserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    schema_version: Mapped[str] = mapped_column(String, nullable=False)
    particle_type: Mapped[str] = mapped_column(String, nullable=False, default="CLAIM")
    # truth-aptness axis. Defaults to FALSIFIABLE so pre-0124 rows
    # (no column) and existing inserts are unchanged.
    assertion_modality: Mapped[str] = mapped_column(
        String, nullable=False, default="FALSIFIABLE", server_default="FALSIFIABLE"
    )
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supersedes: Mapped[str | None] = mapped_column(String, nullable=True)
    # transaction-time end of the belief — the instant the
    # particle FIRST left ACTIVE. Write-once: stamped by update_particle_status
    # and never overwritten on later hops (ACTIVE → PROVENANCE_STALE →
    # SUPERSEDED keeps the departure-from-ACTIVE instant). NULL while ACTIVE,
    # on born-retired rows (never believed), and on pre-migration history.
    # Storage metadata beside the embedding, NOT a core Particle field — it is
    # never serialized to the schema artifacts or interchange.
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # SHA-256 of the normalized ``content`` (
    # particles.core.duplicate_key.content_hash). The index key for extract-time
    # exact-duplicate suppression: without it the rung would scan every ACTIVE
    # particle per pass. Derived, never authoritative — ``content`` remains the
    # source of truth and the hash is recomputed on every write in
    # ``from_model``, so it cannot drift. Nullable only for rows written before
    # migration 031's backfill; the suppression lookup simply misses those.
    content_norm_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Serialised JSON fields
    provenance_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    extractor_ref_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # the ``"<provider>:<model>"`` pairing that produced this
    # particle. Its OWN column, not a key inside ``extractor_ref_json``, for
    # the reason the ADR §2.1 gives: the ref names the code, this names the
    # runtime substrate, and the JSON column is selected by SQL substring —
    # pairings nest (``openai:gpt-5.6`` ⊂ ``openai:gpt-5.6-luna``), so a
    # nested scope would over-select the sibling model it exists to separate.
    # Indexed below because the reason this exists is a scope query.
    #
    # Unlike ``embedding_model_id`` there is NO legacy tier and no sentinel:
    # NULL means UNRECORDED (deterministic extractor, direct assertion, or
    # pre-0229 row) and is never coalesced to a default pairing — the
    # pre-stamp population is already model-mixed, so any default would be
    # wrong for roughly half of it. Never backfilled.
    extraction_provider_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Embedding stored as JSON-encoded float list (384-dim all-MiniLM-L6-v2)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Id of the embedding model that produced ``embedding_json``.
    # NULL on legacy rows / rows with no embedding; the cosine query path treats
    # NULL as the historical default and skips any vector whose id != current.
    embedding_model_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Subject UUIDs this particle is a statement about
    subject_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # Structured ontology properties, e.g. {"nmo:hasWeight": 0.75}
    properties_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Domain hint for efficient cascade queries (Extension B); set on INCONSISTENCY rows
    domain_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    # SHA-256 fingerprint of the ACTIVE-particle baseline at extraction time
    # (Extension C). NULL for pre-0058 particles; stamped by
    # extract_snapshot() going forward.
    context_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Operator-curated tag paths (Extension C.2). NULL when the
    # particle has never been tagged; an empty list serialises as "[]".
    # The denormalised companion is ``particle_tag_edges`` in taxonomy_store.
    tags_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Extension D/E contributor attribution. NULL ≡ no attribution
    # recorded; serialised as a JSON list of {id, role, at} objects.
    contributors_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # the derived S-P-O annotation, split on the embedding pattern —
    # payload in JSON, generator stamp in its own columns so the backfill's
    # scope query and the coverage report are SQL rather than a JSON scan. The
    # stamp is never duplicated inside the payload: one storage location per
    # fact, so the two cannot drift.
    #
    # Unlike ``embedding_model_id`` there is NO legacy tier and no
    # ``LEGACY_STRUCTURIZER_ID`` sentinel: a structured claim is *born*
    # stamped (``from_model`` writes payload and stamp together or writes
    # neither), so a payload with a NULL stamp is unreachable. Rows written
    # before migration 033 are simply un-annotated, which is a legal permanent
    # state.
    structured_claim_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    structurizer_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    structurizer_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    structured_claim_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Which of the prose/structured pair is the assertion.
    # Defaults to PROSE so pre-0218 rows (no column) and every particle this
    # SDK produces today are unchanged.
    canonical_form: Mapped[str] = mapped_column(
        String, nullable=False, default="PROSE", server_default="PROSE"
    )

    __table_args__ = (
        Index("ix_particles_status_confidence", "status", "confidence_value"),
        # Created by migration 007 for Extension B cascade queries
        # (find_open_inconsistencies_for_domain); declared here so
        # autogenerate and create_all agree with migration-built DBs.
        Index("ix_particles_status_domain_hint", "status", "domain_hint"),
        # suppression lookup: "is this exact claim already ACTIVE?".
        # Status leads because the rung only ever asks about ACTIVE particles.
        Index("ix_particles_status_content_hash", "status", "content_norm_hash"),
        # reindex --provider-model scope: "which ACTIVE particles did
        # this pairing produce?". Status leads, mirroring the indexes above.
        Index("ix_particles_status_provider_model", "status", "extraction_provider_model"),
    )

    def to_model(self) -> Particle:
        provenance = [
            ProvenanceRef(
                type=ProvenanceRefType(p["type"]),
                corpus_entry_id=p["corpus_entry_id"],
                snapshot_id=p.get("snapshot_id"),
                location=p.get("location"),
                chunk_hash=p.get("chunk_hash"),
            )
            for p in json.loads(self.provenance_json)
        ]
        return Particle(
            id=self.id,
            content=self.content,
            confidence=Confidence(
                value=self.confidence_value,
                variance=self.confidence_variance,
                calibration_source=CalibrationSource(self.confidence_calibration_source),
                calibration_method=self.confidence_calibration_method,
                calibration_ref=self.confidence_calibration_ref,
            ),
            uncertainty_nature=UncertaintyNature(self.uncertainty_nature),
            provenance=provenance,
            asserted_by=self.asserted_by,
            asserted_at=self.asserted_at,
            status=Status(self.status),
            status_reason=StatusReason(self.status_reason) if self.status_reason else None,
            schema_version=self.schema_version,
            particle_type=ParticleType(self.particle_type),
            assertion_modality=AssertionModality(self.assertion_modality),
            valid_until=self.valid_until,
            supersedes=self.supersedes,
            extractor_ref=self._extractor_ref_to_model(),
            extraction_provider_model=self.extraction_provider_model,
            subject_ids=json.loads(self.subject_ids_json),
            properties=json.loads(self.properties_json) if self.properties_json else None,
            context_fingerprint=self.context_fingerprint,
            tags=json.loads(self.tags_json) if self.tags_json else None,
            contributors=(
                [ContributorRef(**c) for c in json.loads(self.contributors_json)]
                if self.contributors_json
                else None
            ),
            structured_claim=self._structured_claim_to_model(),
            canonical_form=CanonicalForm(self.canonical_form),
        )

    def _extractor_ref_to_model(self) -> ExtractorRef | None:
        """Parse the stored ``extractor_ref`` payload into the model.

        **Coerces rather than raises** (owner decision at sign-off):
        a stored payload that is not a well-formed ``{name, version}`` reads as
        ``None`` with a warning, never an exception. The field was an untyped
        dict through 1.109.x, so this is the one place where a row written
        before the type existed meets a validator — and a read path that could
        wedge on one malformed legacy row would take out query, export and
        lint alike for a field whose absence every consumer already handles
        (``extractor_ref is None`` is the normal operator-assertion state,
        §9.1a).

        The cost is stated plainly: a coerced row loses its trust attribution
        and falls back to ``general-extractor`` weighting like any unstamped
        particle. The warning is the operator's signal to re-extract it; it
        names the particle so the row is findable.
        """
        if not self.extractor_ref_json:
            return None
        try:
            return ExtractorRef(**json.loads(self.extractor_ref_json))
        except (ValueError, TypeError) as exc:
            log.warning(
                "particle %s: malformed extractor_ref %r — reading as None; "
                "re-extract to restore extractor attribution: %s",
                self.id,
                self.extractor_ref_json,
                exc,
            )
            return None

    def _structured_claim_to_model(self) -> StructuredClaim | None:
        """Reassemble the annotation from its payload + stamp columns.

        Returns ``None`` when the row is un-annotated — the legal permanent
        state for most rows. A payload with a NULL stamp is unreachable
        (``from_model`` writes both or neither, and there is no legacy tier),
        so this reads the stamp rather than grandfathering a default; a row in
        that state is corrupt and says so.

        Raises:
            ValueError: If the payload is present but the stamp is not.
        """
        if self.structured_claim_json is None:
            return None
        if (
            self.structurizer_id is None
            or self.structurizer_version is None
            or self.structured_claim_generated_at is None
        ):
            raise ValueError(
                f"particle {self.id}: structured_claim payload without a complete "
                "structurizer stamp — the row was not written by from_model()"
            )
        payload = json.loads(self.structured_claim_json)
        return StructuredClaim(
            subject=ClaimTerm(**payload["subject"]),
            predicate=ClaimTerm(**payload["predicate"]),
            object=ClaimTerm(**payload["object"]),
            subject_id=payload.get("subject_id"),
            structurizer_id=self.structurizer_id,
            structurizer_version=self.structurizer_version,
            generated_at=self.structured_claim_generated_at,
        )

    @classmethod
    def from_model(cls, p: Particle, embedding: list[float] | None = None) -> ParticleRow:
        prov_list: list[dict[str, Any]] = [
            {
                "type": ref.type.value,
                "corpus_entry_id": ref.corpus_entry_id,
                "snapshot_id": ref.snapshot_id,
                "location": ref.location,
                "chunk_hash": ref.chunk_hash,
            }
            for ref in p.provenance
        ]
        return cls(
            id=p.id,
            content=p.content,
            # Recomputed on every write, so the derived hash can never drift
            # from the content it indexes.
            content_norm_hash=content_hash(p.content),
            confidence_value=p.confidence.value,
            confidence_variance=p.confidence.variance,
            confidence_calibration_source=p.confidence.calibration_source.value,
            confidence_calibration_method=p.confidence.calibration_method,
            confidence_calibration_ref=p.confidence.calibration_ref,
            uncertainty_nature=p.uncertainty_nature.value,
            asserted_by=p.asserted_by,
            asserted_at=p.asserted_at,
            status=p.status.value,
            status_reason=p.status_reason.value if p.status_reason else None,
            schema_version=p.schema_version,
            particle_type=p.particle_type.value,
            assertion_modality=p.assertion_modality.value,
            valid_until=p.valid_until,
            supersedes=p.supersedes,
            provenance_json=json.dumps(prov_list),
            extractor_ref_json=(
                json.dumps(p.extractor_ref.model_dump(mode="json")) if p.extractor_ref else None
            ),
            extraction_provider_model=p.extraction_provider_model,
            embedding_json=json.dumps(embedding) if embedding else None,
            embedding_model_id=get_embedding_model_id() if embedding else None,
            subject_ids_json=json.dumps(p.subject_ids),
            properties_json=json.dumps(p.properties) if p.properties is not None else None,
            context_fingerprint=p.context_fingerprint,
            tags_json=json.dumps(p.tags) if p.tags is not None else None,
            contributors_json=(
                json.dumps([c.model_dump(mode="json") for c in p.contributors])
                if p.contributors is not None
                else None
            ),
            # payload and stamp are written together or not at all.
            structured_claim_json=(
                structured_claim_payload(p.structured_claim)
                if p.structured_claim is not None
                else None
            ),
            structurizer_id=(
                p.structured_claim.structurizer_id if p.structured_claim is not None else None
            ),
            structurizer_version=(
                p.structured_claim.structurizer_version if p.structured_claim is not None else None
            ),
            structured_claim_generated_at=(
                p.structured_claim.generated_at if p.structured_claim is not None else None
            ),
            canonical_form=p.canonical_form.value,
        )


class ProvenanceEdgeRow(Base):
    """Denormalised provenance index: one row per (particle, corpus_entry) pair.

    Enables O(k) corpus-entry → particle lookup without JSON parsing.
    """

    __tablename__ = "particle_provenance_edges"

    particle_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # SHA-256 of the LLM-prompt text that produced this particle.
    # Indexed jointly with corpus_entry_id so chunk-hash carry-forward lookup
    # is a single index probe rather than a JSON scan.
    chunk_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_prov_edges_entry", "corpus_entry_id"),
        Index("ix_prov_edges_chunk", "corpus_entry_id", "chunk_hash"),
    )


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------


async def insert_particle(
    session: AsyncSession,
    particle: Particle,
    embedding: list[float] | None = None,
    domain_hint: str | None = None,
) -> None:
    """Persist a particle and its provenance edges. Flushes but does not commit.

    Args:
        session: Active async SQLAlchemy session.
        particle: The particle to insert.
        embedding: Optional pre-computed embedding vector for cosine search.
        domain_hint: Domain label for INCONSISTENCY particles (Extension B).

    Raises:
        ValueError: If the particle is born ``PROVENANCE_STALE`` without
            ``status_reason = CONFLICT_PENDING``. The §6.6 table admits the
            quarantine birth only under that reason; the table is
            keyed on status alone, so the reason condition is enforced here.
    """
    if (
        particle.status is Status.PROVENANCE_STALE
        and particle.status_reason is not StatusReason.CONFLICT_PENDING
    ):
        raise ValueError(
            "A particle may be born PROVENANCE_STALE only as a quarantined"
            " conflict loser with status_reason = CONFLICT_PENDING"
            " (§6.6); got"
            f" status_reason = {particle.status_reason!r}"
        )
    row = ParticleRow.from_model(particle, embedding)
    if domain_hint is not None:
        row.domain_hint = domain_hint
    session.add(row)
    for pref in particle.provenance:
        edge = ProvenanceEdgeRow(
            particle_id=particle.id,
            corpus_entry_id=pref.corpus_entry_id,
            snapshot_id=pref.snapshot_id,
            chunk_hash=pref.chunk_hash,
        )
        session.add(edge)
    if particle.subject_ids:
        # defer: cycle — particle_store and subject_store import each other's
        # ORM Row classes. See root AGENTS.md § Code conventions → Deferred
        # imports for the convention.
        from particles.store.subject_store import link_particle_to_subjects

        await link_particle_to_subjects(session, particle.id, particle.subject_ids)
    await session.flush()


async def get_particle(session: AsyncSession, particle_id: str) -> Particle | None:
    """Fetch a particle by UUID. Returns None if not found."""
    row = await session.get(ParticleRow, particle_id)
    return row.to_model() if row else None


async def get_active_particle_by_id_or_prefix(
    session: AsyncSession, particle_id: str
) -> Particle | None:
    """Load a single ACTIVE particle by full id or short-id prefix.

    The projection ``select.allow`` pin is written in either the full store id
    or the ``p-<shortid>`` display form (the 8-char prefix the sources trailer /
    ``particle_show`` use). The caller strips the ``p-`` display prefix before
    calling; this resolves the remaining string against **ACTIVE** rows as an
    exact id or, failing that, a unique id prefix.

    Returns the matching ``Particle`` or ``None`` when no ACTIVE row matches —
    which is exactly the stale-pin signal the selector turns into a hard error
    (``allow``) or warning (``deny``). A prefix that matches more than one
    ACTIVE id is ambiguous and resolves to ``None`` (treated as unresolved); a
    full id never collides.
    """
    exact = await session.get(ParticleRow, particle_id)
    if exact is not None and Status(exact.status) is Status.ACTIVE:
        return exact.to_model()
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.id.startswith(particle_id),
        )
    )
    rows = result.scalars().all()
    if len(rows) == 1:
        return rows[0].to_model()
    return None


async def update_particle_status(
    session: AsyncSession,
    particle_id: str,
    new_status: Status,
    reason: StatusReason | None = None,
    uncertainty_nature: UncertaintyNature | None = None,
) -> None:
    """Transition a particle's status, enforcing the normative transition table.

    Args:
        particle_id: UUID of the particle to update.
        new_status: Target status. Invalid transitions raise ValueError.
        reason: Optional StatusReason recorded alongside the transition.
        uncertainty_nature: If provided, also updates uncertainty_nature
            (used by Review to mark BOTH_VALID resolutions as ALEATORY).

    Raises:
        ValueError: If the transition is not permitted or particle not found.
    """
    from particles.core.status import REVERSIBLE_SUPERSESSION_REASON, validate_transition

    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")
    from_status = Status(row.status)
    validate_transition(from_status, new_status)
    # the §6.6 table is keyed on status alone, so the reason
    # condition on SUPERSEDED → ACTIVE is enforced here — the same split
    # uses for the CONFLICT_PENDING birth gate. Only an
    # auto-merge supersession is reversible; every judgment-bearing reason
    # (EXPLICIT_SUPERSESSION, SUPERSEDED_BY_REINDEX, LOWER_TRUST_SOURCE,
    # DOCUMENT_SUPERSEDED, CONFLICT_RESOLVED) keeps SUPERSEDED terminal.
    if from_status is Status.SUPERSEDED and new_status is Status.ACTIVE:
        current = StatusReason(row.status_reason) if row.status_reason else None
        if current is not REVERSIBLE_SUPERSESSION_REASON:
            raise ValueError(
                f"Invalid status transition: {from_status!r} → {new_status!r} requires "
                f"status_reason {REVERSIBLE_SUPERSESSION_REASON.value}, "
                f"found {current.value if current else 'none'}. See §6.6 normative table."
            )
    # write-once retirement stamp. Set at the moment a particle
    # first leaves ACTIVE — this choke point is the single path every status
    # write routes through — and never overwritten on later hops (a multi-hop
    # chain must keep the departure-from-ACTIVE instant). Born-retired rows
    # (quarantine losers, INCONSISTENCY records) never pass through an
    # ACTIVE→X transition, so they are never stamped — no special case.
    if from_status is Status.ACTIVE and new_status is not Status.ACTIVE and row.retired_at is None:
        row.retired_at = datetime.now(UTC)
    # write-once is *per retirement*, not per row. A return to
    # ACTIVE withdraws the retirement, so the stamp must go with it — left in
    # place it would survive as an ACTIVE row's retirement instant and, worse,
    # the guard above would preserve the *withdrawn* instant when the row is
    # later genuinely retired, silently misdating the as-of lens.
    if new_status is Status.ACTIVE:
        row.retired_at = None
    row.status = new_status.value
    row.status_reason = reason.value if reason else None
    if uncertainty_nature is not None:
        row.uncertainty_nature = uncertainty_nature.value
    await session.flush()

    # §"Effects on Core operations": when a member of a co-evidential
    # group is RETRACTED, the group survives but the retracted particle leaves.
    # A singleton group dissolves naturally (a particle with no relations is
    # the BFS result for an unlinked particle). Local import to avoid a
    # circular dependency between particle_store and relation_store.
    if new_status == Status.RETRACTED:
        from particles.store.relation_store import remove_particle_from_relations

        await remove_particle_from_relations(session, particle_id)


async def update_status_reason(
    session: AsyncSession,
    particle_id: str,
    reason: StatusReason,
) -> None:
    """Reason-only update — no status transition.

    Narrow seam for resolving a quarantined conflict loser in place: when a
    conflict resolves *against* the quarantined candidate, its reason flips
    ``CONFLICT_PENDING → CONFLICT_RESOLVED`` while the status stays
    ``PROVENANCE_STALE``. The §6.6 table has no same-status re-set edge for
    PROVENANCE_STALE, and widening the table for a reason flip would loosen
    the status machine — hence this helper, restricted to PROVENANCE_STALE
    rows.

    Raises:
        ValueError: If the particle is missing or not PROVENANCE_STALE.
    """
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")
    if Status(row.status) is not Status.PROVENANCE_STALE:
        raise ValueError(
            "Reason-only update is permitted only on PROVENANCE_STALE rows;"
            f" particle {particle_id} has status {row.status!r}"
        )
    row.status_reason = reason.value
    await session.flush()


async def copy_particle_embedding(
    session: AsyncSession,
    source_id: str,
    target_id: str,
) -> None:
    """Copy ``embedding_json`` and ``embedding_model_id`` between rows.

    Used by quarantine promotion: the minted particle carries the
    quarantined row's vector verbatim, so the model marker must travel with
    it — re-stamping with the *current* model id would corrupt cosine search
    after an embedding-model swap. No-op when the source row
    has no embedding.

    Raises:
        ValueError: If either particle is missing.
    """
    source = await session.get(ParticleRow, source_id)
    target = await session.get(ParticleRow, target_id)
    if source is None or target is None:
        raise ValueError(f"copy_particle_embedding: particle not found ({source_id} → {target_id})")
    if source.embedding_json is None:
        return
    target.embedding_json = source.embedding_json
    target.embedding_model_id = source.embedding_model_id
    await session.flush()


async def get_snapshot_ids_with_particles(session: AsyncSession) -> set[str]:
    """Return the set of snapshot_ids that produced at least one particle.

    Read from the provenance edge table (``ProvenanceEdgeRow.snapshot_id``).
    Used by the ``EMPTY_COMPLETE_SNAPSHOT`` lint check to find COMPLETE
    snapshots whose extraction yielded zero particles — the signature of the
    F4.1 silent-loss bug (a fully-failed chunked/PDF extraction stamped
    COMPLETE). NULL snapshot_ids are ignored.
    """
    result = await session.execute(
        select(ProvenanceEdgeRow.snapshot_id)
        .where(ProvenanceEdgeRow.snapshot_id.is_not(None))
        .distinct()
    )
    return {str(s) for s in result.scalars() if s is not None}


async def get_active_particles_for_entry(
    session: AsyncSession, corpus_entry_id: str
) -> list[Particle]:
    result = await session.execute(
        select(ParticleRow)
        .join(
            ProvenanceEdgeRow,
            ProvenanceEdgeRow.particle_id == ParticleRow.id,
        )
        .where(
            ProvenanceEdgeRow.corpus_entry_id == corpus_entry_id,
            ParticleRow.status == Status.ACTIVE.value,
        )
    )
    return [r.to_model() for r in result.scalars()]


async def get_active_particle_ids_from_other_snapshots(
    session: AsyncSession, corpus_entry_id: str, current_snapshot_id: str
) -> list[str]:
    """ACTIVE particle ids anchored to a *superseded* snapshot of one entry.

    The generation cascade's candidate set: particles whose provenance edge
    names this corpus entry but a snapshot other than ``current_snapshot_id``.
    Reads the denormalised edge index, so it is one indexed probe rather than a
    JSON scan.

    Edges with a NULL ``snapshot_id`` are excluded — snapshot-less provenance
    carries no generation, so there is nothing to supersede it.
    """
    result = await session.execute(
        select(ParticleRow.id)
        .join(ProvenanceEdgeRow, ProvenanceEdgeRow.particle_id == ParticleRow.id)
        .where(
            ProvenanceEdgeRow.corpus_entry_id == corpus_entry_id,
            ProvenanceEdgeRow.snapshot_id.is_not(None),
            ProvenanceEdgeRow.snapshot_id != current_snapshot_id,
            ParticleRow.status == Status.ACTIVE.value,
        )
        .distinct()
    )
    return [str(pid) for pid in result.scalars()]


async def get_active_particles_by_content_hashes(
    session: AsyncSession, content_hashes: Sequence[str]
) -> list[Particle]:
    """ACTIVE particles whose normalized-content hash is in ``content_hashes``.

    The suppression lookup. One indexed probe per extraction pass
    (chunked to stay under SQLite's bound-parameter limit) rather than a scan
    over every ACTIVE particle — the rung sits on the write path, so an O(N)
    read here would be paid by every extraction.

    Returns *candidates*, not matches: the hash settles content identity only.
    The caller still has to compare the subject-id set and ``stance:holder``
    (:func:`particles.core.duplicate_key.duplicate_key`) and apply the
    truth-apt / asserted gates before treating a row as the same claim.

    Rows written before migration 031's backfill have a NULL hash and are
    simply not returned.
    """
    keys = [h for h in dict.fromkeys(content_hashes) if h]
    if not keys:
        return []
    out: list[Particle] = []
    for start in range(0, len(keys), 500):
        result = await session.execute(
            select(ParticleRow).where(
                ParticleRow.status == Status.ACTIVE.value,
                ParticleRow.content_norm_hash.in_(keys[start : start + 500]),
            )
        )
        out.extend(row.to_model() for row in result.scalars())
    return out


async def append_provenance_ref(
    session: AsyncSession, particle_id: str, ref: ProvenanceRef
) -> bool:
    """Record an additional source observation on an existing particle.

    When extraction suppresses a candidate as an exact duplicate, the new
    source's evidence has to land somewhere — a suppression that dropped it
    would be a source-faithfulness regression (techspec §6.10). It lands here,
    on the ACTIVE read surface, rather than behind a graph walk.

    **Append-only and idempotent.** The ref is appended to the end of the list,
    so ``provenance[0]`` — the decay anchor — stays the *earliest*
    observation: re-reading a claim must not silently refresh its age. An
    identical ref already present is a no-op, which is what makes re-harvesting
    an unchanged snapshot safe (runs a level-triggered catch-up sweep,
    so the same snapshot is re-offered routinely).

    Nothing else about the particle is touched — in particular
    ``confidence.value`` stays immutable and ``asserted_at`` /
    ``asserted_by`` keep recording who first minted the claim.

    The denormalised ``particle_provenance_edges`` index is keyed
    ``(particle_id, corpus_entry_id)``, so it already carries one row per entry:
    a second *snapshot* of an entry the particle is already linked to needs no
    new edge, and the existing edge is deliberately **not** re-pointed at the
    newer snapshot (see ``cascade_superseded_generation``'s ``exclude_ids``,
    which is how the generation cascade is told to leave a
    still-current particle alone — the same treatment carry-forward
    gets).

    Returns:
        ``True`` if the ref was added, ``False`` if it was already present.
    """
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        return False
    refs: list[dict[str, Any]] = json.loads(row.provenance_json)
    incoming = {
        "type": ref.type.value,
        "corpus_entry_id": ref.corpus_entry_id,
        "snapshot_id": ref.snapshot_id,
        "location": ref.location,
        "chunk_hash": ref.chunk_hash,
    }
    if any(existing == incoming for existing in refs):
        return False
    refs.append(incoming)
    row.provenance_json = json.dumps(refs)
    existing_edge = await session.get(ProvenanceEdgeRow, (particle_id, ref.corpus_entry_id))
    if existing_edge is None:
        session.add(
            ProvenanceEdgeRow(
                particle_id=particle_id,
                corpus_entry_id=ref.corpus_entry_id,
                snapshot_id=ref.snapshot_id,
                chunk_hash=ref.chunk_hash,
            )
        )
    await session.flush()
    return True


async def get_particle_ids_for_entries(
    session: AsyncSession, corpus_entry_ids: Sequence[str]
) -> set[str]:
    """Particle ids with a provenance edge to any of the given corpus entries.

    Reads the denormalised provenance-edge index in one shot (chunked to stay
    under SQLite's bound-parameter limit). Used by the audit's harvested-scope
    contradiction probe to bound candidate pairs to the
    beliefs this harvest produced or touched.
    """
    ids = list(dict.fromkeys(corpus_entry_ids))
    out: set[str] = set()
    for start in range(0, len(ids), 500):
        result = await session.execute(
            select(ProvenanceEdgeRow.particle_id)
            .where(ProvenanceEdgeRow.corpus_entry_id.in_(ids[start : start + 500]))
            .distinct()
        )
        out.update(str(pid) for pid in result.scalars())
    return out


async def get_particle_ids_changed_since(session: AsyncSession, since: datetime) -> set[str]:
    """Particle ids created or modified at/after ``since`` (delta scope).

    "Created" = ``asserted_at``; "modified" = ``retired_at`` (the write-once
    departure-from-ACTIVE stamp — the only mutation the delta scope
    cares about, since content is immutable and edits are supersessions, which
    mint a new ``asserted_at``). Feeds ``scope_particle_ids`` on the seams so a scheduled census probes only what moved since the
    previous run's watermark.
    """
    result = await session.execute(
        select(ParticleRow.id).where(
            or_(ParticleRow.asserted_at >= since, ParticleRow.retired_at >= since)
        )
    )
    return {str(pid) for pid in result.scalars()}


async def get_particles_for_entry(session: AsyncSession, corpus_entry_id: str) -> list[Particle]:
    """Return every particle (any status) whose provenance traces to the entry.

    Keyed on ``corpus_entry_id`` via the provenance edge, so particles
    re-extracted from later snapshots of the same entry are included. Used by
    the ``corpus retract`` operation to partition by status.
    """
    result = await session.execute(
        select(ParticleRow)
        .join(ProvenanceEdgeRow, ProvenanceEdgeRow.particle_id == ParticleRow.id)
        .where(ProvenanceEdgeRow.corpus_entry_id == corpus_entry_id)
        .distinct()
    )
    return [r.to_model() for r in result.scalars()]


async def stamp_scope_exemption_for_entry(session: AsyncSession, corpus_entry_id: str) -> int:
    """Stamp the scope exemption on one entry's stored particles.

    Returns the number of rows changed. The stamp is a deterministic function
    of the entry's tags — no LLM, no re-classification — so applying it to
    particles an earlier run already wrote is a *restamp*, not a reindex; that
    is why rule-sourcing departs from "reindex is the relabelling
    path" (which is right for a classification, wrong for a policy over stored
    labels).

    Idempotent: only rows the exemption actually changes are written, so a
    second pass reports 0. Never touches ``confidence`` — scope governs
    visibility only (§Decision 3).

    Whether the entry *is* exempt is the caller's decision (see
    ``particles.extraction.scope.is_scope_exempt_source``); this accessor only
    performs the write.
    """
    from particles.extraction.scope import apply_source_exemption

    result = await session.execute(
        select(ParticleRow)
        .join(ProvenanceEdgeRow, ProvenanceEdgeRow.particle_id == ParticleRow.id)
        .where(ProvenanceEdgeRow.corpus_entry_id == corpus_entry_id)
        .distinct()
    )
    changed = 0
    for row in result.scalars():
        properties = json.loads(row.properties_json) if row.properties_json else None
        stamped = apply_source_exemption(properties)
        if stamped is not properties:
            row.properties_json = json.dumps(stamped)
            changed += 1
    return changed


async def get_particles_by_status(session: AsyncSession, status: Status) -> list[Particle]:
    result = await session.execute(select(ParticleRow).where(ParticleRow.status == status.value))
    return [r.to_model() for r in result.scalars()]


async def get_active_narratives(session: AsyncSession) -> list[Particle]:
    """Return every ACTIVE NARRATIVE particle.

    The per-narrative note loops in the prose exporters need only the NARRATIVE
    set. The Obsidian and Logseq exporters filter it out of the full ACTIVE list
    they already hold; the wiki exporter has no such list, so it asks for the
    narratives directly rather than paying a second full-store load.
    """
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.particle_type == ParticleType.NARRATIVE.value,
        )
    )
    return [r.to_model() for r in result.scalars()]


async def list_particles_filtered(
    session: AsyncSession,
    *,
    status: Status | None = None,
    subject_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Particle]:
    """Paginated, status/subject-filtered particle listing — no embeddings.

    Read-only accessor used by the MCP ``particles_list`` tool (and any
    other caller that needs lightweight particle summaries without the
    embedding blob). Particles are ordered by ``asserted_at`` descending
    so the most recently extracted items appear first; ties are broken by
    ``id`` for a stable page boundary across calls.

    Args:
        session: Active async SQLAlchemy session.
        status: Optional ``Status`` filter (e.g. ``Status.INCONSISTENCY``).
        subject_id: Optional subject filter via the ``particle_subjects``
            join table.
        limit: Maximum particles to return (default 50). Must be > 0.
        offset: Number of particles to skip before returning results
            (default 0). Combine with ``limit`` to page through the full set.

    Returns:
        List of ``Particle`` models. The embedding column is never read,
        keeping responses small for MCP-client consumption.
    """
    from particles.store.subject_store import ParticleSubjectRow

    stmt = select(ParticleRow)
    if status is not None:
        stmt = stmt.where(ParticleRow.status == status.value)
    if subject_id is not None:
        stmt = stmt.join(
            ParticleSubjectRow,
            ParticleSubjectRow.particle_id == ParticleRow.id,
        ).where(ParticleSubjectRow.subject_id == subject_id)
    stmt = stmt.order_by(ParticleRow.asserted_at.desc(), ParticleRow.id).offset(offset).limit(limit)
    result = await session.execute(stmt)
    return [row.to_model() for row in result.scalars()]


async def compute_context_fingerprint(session: AsyncSession) -> str:
    """Compute the SHA-256 fingerprint of the current ACTIVE-particle baseline.

    Per spec §16.1, the algorithm is:
      1. Identify all ACTIVE particles in the store.
      2. Sort their UUIDs lexicographically.
      3. SHA-256 of the concatenated sorted UUIDs (no delimiter).

    Step 1 is the query below; steps 2–3 are
    :func:`particles.core.fingerprint.context_fingerprint`, shared with the conformance runner so the procedure exists once.

    Returns a 64-character hex digest. An empty store returns the SHA-256
    of the empty string, which is the canonical baseline for a fresh store.
    """
    result = await session.execute(
        select(ParticleRow.id).where(ParticleRow.status == Status.ACTIVE.value)
    )
    return context_fingerprint(row[0] for row in result.all())


async def get_particles_by_context_fingerprint(
    session: AsyncSession, fingerprint: str
) -> list[Particle]:
    """Return all particles sharing a given context_fingerprint."""
    result = await session.execute(
        select(ParticleRow).where(ParticleRow.context_fingerprint == fingerprint)
    )
    return [r.to_model() for r in result.scalars()]


async def get_inconsistency_particles(session: AsyncSession) -> list[Particle]:
    return await get_particles_by_status(session, Status.INCONSISTENCY)


async def get_inconsistency_backrefs(session: AsyncSession) -> dict[str, str]:
    """Map each particle id referenced by an open INCONSISTENCY -> that INCONSISTENCY id.

    The §6.6 INCONSISTENCY particle records its conflicting pair as PARTICLE-type
    provenance refs whose ``corpus_entry_id`` carries the referenced particle
    UUID (the field name is legacy; see ``build_inconsistency_particle``). This
    backref lets a read surface mark a returned ACTIVE belief as *contested*
    : the surviving (ACTIVE) side of a conflict is referenced here;
    the quarantined loser is PROVENANCE_STALE and never reaches a query result.
    Cost is one INCONSISTENCY-status scan — acceptable at memory-store scale
    (is the general scaling lever).
    """
    backrefs: dict[str, str] = {}
    for inc in await get_inconsistency_particles(session):
        for ref in inc.provenance:
            if ref.type == ProvenanceRefType.PARTICLE:
                backrefs.setdefault(ref.corpus_entry_id, inc.id)
    return backrefs


async def count_active_dependents(session: AsyncSession, particle_ids: set[str]) -> dict[str, int]:
    """Count ACTIVE particles depending on each id through the provenance DAG.

    A particle *depends on* another when its ``provenance`` carries a
    ``PARTICLE``-type ref whose ``corpus_entry_id`` is the other's UUID (the
    field name is legacy; see ``build_inconsistency_particle``). This is the
    reverse of the edge ``_check_retraction_propagation`` walks forward — pure
    aggregation of the existing DAG, no new detection logic. Used as the
    dependency-count leverage signal: a wrong belief many ACTIVE particles rest
    on outranks an isolated one.

    Returns a count for every id in ``particle_ids`` (0 when nothing depends on
    it). A particle is not counted as its own dependent.
    """
    counts: dict[str, int] = {pid: 0 for pid in particle_ids}
    if not particle_ids:
        return counts
    for p in await get_active_particles(session):
        seen: set[str] = set()
        for ref in p.provenance:
            if ref.type != ProvenanceRefType.PARTICLE:
                continue
            target = ref.corpus_entry_id
            if target in particle_ids and target != p.id and target not in seen:
                seen.add(target)
                counts[target] += 1
    return counts


async def get_active_particles_with_embeddings(
    session: AsyncSession,
    min_confidence: float = 0.0,
    subject_id: str | None = None,
) -> list[tuple[Particle, np.ndarray[Any, np.dtype[np.float32]]]]:
    """Return all ACTIVE particles that have embeddings, for cosine similarity search."""
    from particles.store.subject_store import ParticleSubjectRow

    q = select(ParticleRow).where(
        ParticleRow.status == Status.ACTIVE.value,
        ParticleRow.confidence_value >= min_confidence,
        ParticleRow.embedding_json.isnot(None),
    )
    if subject_id is not None:
        q = q.join(
            ParticleSubjectRow,
            ParticleSubjectRow.particle_id == ParticleRow.id,
        ).where(ParticleSubjectRow.subject_id == subject_id)
    result = await session.execute(q)
    rows = result.scalars().all()
    current_id = get_embedding_model_id()
    out: list[tuple[Particle, np.ndarray[Any, np.dtype[np.float32]]]] = []
    skipped = 0
    for row in rows:
        # a vector embedded under a different model lives in a
        # different space; comparing it to the current query embedding is
        # meaningless. Skip it loudly rather than corrupt the ranking. NULL is a
        # legacy row, grandfathered as the historical default model.
        stored_id = row.embedding_model_id or LEGACY_EMBEDDING_MODEL_ID
        if stored_id != current_id:
            skipped += 1
            continue
        emb = np.array(json.loads(row.embedding_json), dtype=np.float32)  # type: ignore[arg-type]
        out.append((row.to_model(), emb))
    if skipped:
        log.warning(
            "Skipped %d ACTIVE particle(s) whose embedding model id != %r during "
            "cosine search — their vectors live in a different embedding space. "
            "Re-extract / re-embed them to make them searchable again.",
            skipped,
            current_id,
        )
    return out


async def get_active_particles_with_claims(
    session: AsyncSession,
    min_confidence: float = 0.0,
    subject_id: str | None = None,
) -> list[Particle]:
    """All ACTIVE particles carrying an annotated structured claim.

    The structural-scan loader: unlike the cosine loaders it does
    **not** require an embedding — a structural filter selects claims by their
    form, and an un-embedded particle's claim is as filterable as any other.
    ``min_confidence`` is the same raw-value SQL superset prefilter the
    semantic path uses (every effective-confidence factor is ≤ 1.0, so
    effective ≤ raw and the prefilter can never drop a row the read-time
    effective check would keep).
    """
    from particles.store.subject_store import ParticleSubjectRow

    q = select(ParticleRow).where(
        ParticleRow.status == Status.ACTIVE.value,
        ParticleRow.confidence_value >= min_confidence,
        ParticleRow.structured_claim_json.isnot(None),
    )
    if subject_id is not None:
        q = q.join(
            ParticleSubjectRow,
            ParticleSubjectRow.particle_id == ParticleRow.id,
        ).where(ParticleSubjectRow.subject_id == subject_id)
    result = await session.execute(q)
    return [row.to_model() for row in result.scalars().all()]


async def get_particles_with_claims_as_of(
    session: AsyncSession,
    as_of: datetime,
    min_confidence: float = 0.0,
    subject_id: str | None = None,
) -> list[tuple[Particle, datetime | None]]:
    """All-statuses claim-carrying loader for the as-of lens.

    The structural sibling of :func:`get_particles_with_embeddings_as_of`
    (modes compose with ``--as-of`` unchanged): every particle
    carrying a structured claim whose ``asserted_at <= as_of``, with the §2a
    ``retired_at`` storage column threaded alongside for the ``AsOfView``
    ladder. No embedding is required — the structural modes never embed.
    """
    from particles.store.subject_store import ParticleSubjectRow

    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    q = select(ParticleRow).where(
        ParticleRow.confidence_value >= min_confidence,
        ParticleRow.structured_claim_json.isnot(None),
    )
    if subject_id is not None:
        q = q.join(
            ParticleSubjectRow,
            ParticleSubjectRow.particle_id == ParticleRow.id,
        ).where(ParticleSubjectRow.subject_id == subject_id)
    result = await session.execute(q)
    out: list[tuple[Particle, datetime | None]] = []
    for row in result.scalars().all():
        asserted = row.asserted_at
        if asserted.tzinfo is None:
            asserted = asserted.replace(tzinfo=UTC)
        if asserted > as_of:
            continue
        out.append((row.to_model(), row.retired_at))
    return out


async def get_particles_with_embeddings_as_of(
    session: AsyncSession,
    as_of: datetime,
    min_confidence: float = 0.0,
    subject_id: str | None = None,
) -> list[tuple[Particle, np.ndarray[Any, np.dtype[np.float32]], datetime | None]]:
    """All-statuses candidate loader for the as-of read lens.

    Sibling of :func:`get_active_particles_with_embeddings` (which remains the
    ``as_of=None`` fast path, so the default query plan is untouched): loads
    every particle **across statuses** with an embedding present whose
    ``asserted_at <= as_of`` — the "it had been asserted" half of the visibility predicate. A timezone-naive stored ``asserted_at`` is
    assumed UTC, matching the query path's existing comparison. The
    "not yet retired" half (the §2 reconstruction ladder) is the
    ``AsOfView``'s job in ``operations/query/as_of.py`` — this loader is
    deliberately mechanical.

    Returns ``(particle, embedding, retired_at)`` triples; ``retired_at`` is
    the §2a storage column (rung 0 of the ladder), threaded alongside because
    it is storage metadata, not a ``Particle`` field. Vectors embedded under a
    different model are skipped, exactly as the ACTIVE-only loader does.
    """
    from particles.store.subject_store import ParticleSubjectRow

    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    q = select(ParticleRow).where(
        ParticleRow.confidence_value >= min_confidence,
        ParticleRow.embedding_json.isnot(None),
    )
    if subject_id is not None:
        q = q.join(
            ParticleSubjectRow,
            ParticleSubjectRow.particle_id == ParticleRow.id,
        ).where(ParticleSubjectRow.subject_id == subject_id)
    result = await session.execute(q)
    rows = result.scalars().all()
    current_id = get_embedding_model_id()
    out: list[tuple[Particle, np.ndarray[Any, np.dtype[np.float32]], datetime | None]] = []
    skipped = 0
    for row in rows:
        asserted = row.asserted_at
        if asserted.tzinfo is None:
            asserted = asserted.replace(tzinfo=UTC)
        if asserted > as_of:
            continue
        stored_id = row.embedding_model_id or LEGACY_EMBEDDING_MODEL_ID
        if stored_id != current_id:
            skipped += 1
            continue
        emb = np.array(json.loads(row.embedding_json), dtype=np.float32)  # type: ignore[arg-type]
        out.append((row.to_model(), emb, row.retired_at))
    if skipped:
        log.warning(
            "Skipped %d particle(s) whose embedding model id != %r during as-of "
            "cosine search — their vectors live in a different embedding space "
            ".",
            skipped,
            current_id,
        )
    return out


async def get_retired_at(session: AsyncSession, particle_id: str) -> datetime | None:
    """Read one particle's ``retired_at`` storage column.

    Storage metadata is not carried on the ``Particle`` model, so callers that
    need the stamp (the as-of ladder's rung 0, tests, diagnostics) read it
    through this narrow accessor. Returns ``None`` for an ACTIVE / born-retired
    / pre-migration row — and raises for a missing particle.
    """
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")
    return row.retired_at


async def get_active_particles_with_stale_embedding_model(
    session: AsyncSession, current_embedding_model_id: str
) -> list[Particle]:
    """Return ACTIVE particles whose embedding model id != the current one.

    Mirrors :func:`get_active_particles_with_stale_schema_version`:
    the read-side surface a future re-embed operation consumes to find vectors
    that need recomputing after an embedding-model swap. A NULL
    ``embedding_model_id`` is a legacy row, grandfathered as
    :data:`LEGACY_EMBEDDING_MODEL_ID`; it counts as stale only when the current
    model differs from that historical default. Rows without an embedding are
    not returned (nothing to re-embed).
    """
    from sqlalchemy import func

    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.embedding_json.isnot(None),
            func.coalesce(ParticleRow.embedding_model_id, LEGACY_EMBEDDING_MODEL_ID)
            != current_embedding_model_id,
        )
    )
    return [row.to_model() for row in result.scalars()]


async def get_store_embedding_profile(session: AsyncSession) -> EmbeddingProfile | None:
    """Return the structured ``embedding_profile`` recorded in store metadata.

    The profile is the ``{model, dim, normalization}`` triple that determines the
    embedding space this store's vectors occupy. The ``model`` component is
    persisted per-row as ``embedding_model_id``; ``dim`` and
    ``normalization`` are store-uniform facts read from ``config.embeddings``
    (a profile change requires re-embedding the whole store, so they cannot vary
    row-to-row). This accessor reads the model marker off the most recent
    embedded particle and combines it with the active config to surface the full
    structured profile.

    Returns ``None`` when the store holds no embedded particle yet (no profile
    has been recorded). A NULL ``embedding_model_id`` on an embedded row is the
    legacy default (:data:`LEGACY_EMBEDDING_MODEL_ID`), grandfathered the same
    way the cosine query path treats it.
    """
    result = await session.execute(
        select(ParticleRow.embedding_model_id)
        .where(ParticleRow.embedding_json.isnot(None))
        .order_by(ParticleRow.asserted_at.desc())
        .limit(1)
    )
    row = result.first()
    if row is None:
        return None
    stored_model = row[0] or LEGACY_EMBEDDING_MODEL_ID
    # dim / normalization are store-uniform; take them from the active profile.
    active = get_embedding_profile()
    return EmbeddingProfile(model=stored_model, dim=active.dim, normalization=active.normalization)


async def update_uncertainty_nature(
    session: AsyncSession,
    particle_id: str,
    uncertainty_nature: UncertaintyNature,
) -> None:
    """Update only the uncertainty_nature of a particle — no status transition required."""
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle {particle_id} not found")
    row.uncertainty_nature = uncertainty_nature.value
    await session.flush()


async def get_inconsistency_particles_by_domain(
    session: AsyncSession, domain: str, limit: int = 500
) -> list[Particle]:
    """Return open INCONSISTENCY particles whose domain_hint matches domain (Extension B)."""
    result = await session.execute(
        select(ParticleRow)
        .where(
            ParticleRow.status == Status.INCONSISTENCY.value,
            ParticleRow.domain_hint == domain,
        )
        .limit(limit)
    )
    return [row.to_model() for row in result.scalars().all()]


async def count_particles_by_status(session: AsyncSession) -> dict[str, int]:
    from sqlalchemy import func

    result = await session.execute(
        select(ParticleRow.status, func.count(ParticleRow.id)).group_by(ParticleRow.status)
    )
    return {row[0]: row[1] for row in result}


async def count_active_particles_by_calibration_source(
    session: AsyncSession,
) -> dict[str, int]:
    """Distribution of calibration_source across ACTIVE particles."""
    from sqlalchemy import func

    result = await session.execute(
        select(
            ParticleRow.confidence_calibration_source,
            func.count(ParticleRow.id),
        )
        .where(ParticleRow.status == Status.ACTIVE.value)
        .group_by(ParticleRow.confidence_calibration_source)
    )
    return {row[0]: row[1] for row in result}


async def get_active_particles_with_extractor_version(
    session: AsyncSession, extractor_version: str
) -> list[Particle]:
    """Return ACTIVE particles whose ``extractor_ref.version`` equals this value.

    The SQL ``contains`` clause is a superset prefilter only; the parsed
    ``extractor_ref`` ``version`` key is then exact-matched in Python — the
    same pattern (and for the same reason) as
    :func:`get_active_particles_for_chunk_hash`. Substring matching alone has
    no key discrimination, so a version string false-hits on an extractor
    *name* that happens to contain it, and ``1.0`` false-hits ``1.0.0``.
    Over-selection is not benign here: this helper backs
    ``reindex --extractor-version``, and reindex supersedes what it
    re-extracts.
    """
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.extractor_ref_json.contains(extractor_version),
        )
    )
    particles: list[Particle] = []
    for row in result.scalars():
        p = row.to_model()
        if p.extractor_ref is not None and p.extractor_ref.version == extractor_version:
            particles.append(p)
    return particles


async def get_active_particles_with_extractor_id(
    session: AsyncSession, extractor_id: str
) -> list[Particle]:
    """Return ACTIVE particles whose ``extractor_ref.name`` equals this value.

    Use this when a shared upstream (e.g. a prompt change in general.py) affects
    multiple extractors that delegate to it and you want to re-extract every
    particle produced by a specific extractor regardless of recorded version.

    As with :func:`get_active_particles_with_extractor_version`, the SQL
    ``contains`` clause is a prefilter and the parsed ``name`` key is
    exact-matched in Python: ids nest as substrings (``gist`` inside
    ``github-gist-extractor``), and a bare ``contains`` would also hit a
    *version* that spelled the id.
    """
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.extractor_ref_json.contains(extractor_id),
        )
    )
    particles: list[Particle] = []
    for row in result.scalars():
        p = row.to_model()
        if p.extractor_ref is not None and p.extractor_ref.name == extractor_id:
            particles.append(p)
    return particles


async def get_active_particles_with_provider_model(
    session: AsyncSession, provider_model: str
) -> list[Particle]:
    """Return ACTIVE particles stamped with this ``"<provider>:<model>"`` pairing.

    The reindex scope. **Exact equality on a dedicated column** — no
    substring prefilter over ``extractor_ref_json`` and no Python-side
    re-check, as the two helpers above need. Pairings nest as substrings
    (``openai:gpt-5.6`` is a prefix of ``openai:gpt-5.6-luna``), so a
    substring scope would silently sweep in the sibling model this exists to
    separate. Rows with a NULL stamp never match:
    unrecorded is not a pairing, and re-extracting them is the operator's
    remedy, not something a model-scoped query may guess at.
    """
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.extraction_provider_model == provider_model,
        )
    )
    return [row.to_model() for row in result.scalars()]


async def get_active_particles_for_chunk_hash(
    session: AsyncSession,
    corpus_entry_id: str,
    chunk_hash: str,
    extractor_id: str,
    extractor_version: str,
) -> list[Particle]:
    """Return ACTIVE particles eligible for chunk-hash carry-forward.

    Filters by (corpus_entry_id, chunk_hash) via the indexed edge table and
    ParticleRow.status=ACTIVE, then exact-matches the parsed ``extractor_ref``
    ``name``/``version`` fields in Python. The SQL ``contains`` clauses are a
    superset prefilter only — substring matching alone would false-hit on ids
    that contain other ids (e.g. ``gist`` inside ``github-gist-extractor``).
    A name or version mismatch is treated as a cache miss so that
    EXTRACTOR_VERSION bumps still force re-extraction.
    """
    result = await session.execute(
        select(ParticleRow)
        .join(
            ProvenanceEdgeRow,
            ProvenanceEdgeRow.particle_id == ParticleRow.id,
        )
        .where(
            ProvenanceEdgeRow.corpus_entry_id == corpus_entry_id,
            ProvenanceEdgeRow.chunk_hash == chunk_hash,
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.extractor_ref_json.contains(extractor_id),
            ParticleRow.extractor_ref_json.contains(extractor_version),
        )
    )
    particles: list[Particle] = []
    for row in result.scalars():
        p = row.to_model()
        ref = p.extractor_ref
        if ref is not None and ref.name == extractor_id and ref.version == extractor_version:
            particles.append(p)
    return particles


# Inverted carry-forward coupling: the Engine registers this
# store lookup with the Client-layer ``incremental`` module at import time, so
# chunked extraction (``extract_with_carry_forward``) consults the store
# without ``incremental`` importing any Engine module. Registering from the
# store side keeps it robust: any store access imports this module, so the hook
# is set before carry-forward ever runs. Engine→Client import is allowed.
register_carry_forward_lookup(get_active_particles_for_chunk_hash)


async def get_active_particles_with_stale_schema_version(
    session: AsyncSession, current_schema_version: str
) -> list[Particle]:
    """Return ACTIVE particles whose schema_version does not match the current value."""
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.schema_version != current_schema_version,
        )
    )
    return [row.to_model() for row in result.scalars()]


async def get_active_particles(session: AsyncSession) -> list[Particle]:
    """Return all ACTIVE particles as models. Loads all rows into memory."""
    result = await session.execute(
        select(ParticleRow).where(ParticleRow.status == Status.ACTIVE.value)
    )
    return [row.to_model() for row in result.scalars()]


async def get_active_derived_particles(session: AsyncSession) -> list[Particle]:
    """Return ACTIVE particles with ``calibration_source == DERIVED``.

    The derived population is expected to stay small (bounded by the per-cycle
    promotion cap), so consumers — the revalidation ladder, the read-time
    stale-support discount, the projection premise suppression — load it whole.
    """
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.confidence_calibration_source == CalibrationSource.DERIVED.value,
        )
    )
    return [row.to_model() for row in result.scalars()]


async def get_particles_by_ids(
    session: AsyncSession, particle_ids: Sequence[str]
) -> dict[str, Particle]:
    """Bulk-fetch particles by id (any status). Missing ids are absent from the map."""
    if not particle_ids:
        return {}
    out: dict[str, Particle] = {}
    ids = list(dict.fromkeys(particle_ids))
    for i in range(0, len(ids), 500):
        result = await session.execute(
            select(ParticleRow).where(ParticleRow.id.in_(ids[i : i + 500]))
        )
        for row in result.scalars():
            out[row.id] = row.to_model()
    return out


async def get_superseding_particle(session: AsyncSession, superseded_id: str) -> Particle | None:
    """Return the particle whose ``supersedes`` points at ``superseded_id``.

    Follows one hop of the revision chain (a SUPERSEDED premise is
    replaced by its successor in the updated premise set). When several rows
    claim the same predecessor (re-extraction races), the most recently
    asserted one wins.
    """
    result = await session.execute(
        select(ParticleRow)
        .where(ParticleRow.supersedes == superseded_id)
        .order_by(ParticleRow.asserted_at.desc())
    )
    row = result.scalars().first()
    return row.to_model() if row else None


async def update_particle_provenance(
    session: AsyncSession, particle_id: str, provenance: list[ProvenanceRef]
) -> None:
    """Replace a particle's provenance refs (premise-ref refresh).

    Used by the revalidation ladder to point a still-valid derived particle at
    its updated premise set (retracted premises dropped, superseded premises
    replaced by their successors). Touches only ``provenance_json`` and the
    denormalised edge index — never content, status, or the immutable
    ``confidence.value``.

    Raises:
        ValueError: If the particle does not exist.
    """
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"Particle not found: {particle_id}")
    row.provenance_json = json.dumps(
        [
            {
                "type": ref.type.value,
                "corpus_entry_id": ref.corpus_entry_id,
                "snapshot_id": ref.snapshot_id,
                "location": ref.location,
                "chunk_hash": ref.chunk_hash,
            }
            for ref in provenance
        ]
    )
    # Rebuild this particle's rows in the denormalised provenance-edge index.
    await session.execute(
        delete(ProvenanceEdgeRow).where(ProvenanceEdgeRow.particle_id == particle_id)
    )
    seen: set[str] = set()
    for ref in provenance:
        if ref.corpus_entry_id in seen:
            continue
        seen.add(ref.corpus_entry_id)
        session.add(ProvenanceEdgeRow(particle_id=particle_id, corpus_entry_id=ref.corpus_entry_id))
    await session.flush()


async def get_active_particles_with_valid_until(session: AsyncSession) -> list[Particle]:
    """Return ACTIVE particles that have a non-null valid_until (for staleness checks)."""
    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.valid_until.isnot(None),
        )
    )
    return [row.to_model() for row in result.scalars()]


async def get_active_epistemic_particles_with_variance(
    session: AsyncSession,
) -> list[Particle]:
    """Return ACTIVE EPISTEMIC particles with a non-null confidence variance."""
    from particles.core.schema import UncertaintyNature

    result = await session.execute(
        select(ParticleRow).where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.uncertainty_nature == UncertaintyNature.EPISTEMIC.value,
            ParticleRow.confidence_variance.isnot(None),
        )
    )
    return [row.to_model() for row in result.scalars()]


async def get_particles_needing_structured_claim(
    session: AsyncSession,
    *,
    structurizer_version: str | None = None,
    limit: int | None = None,
) -> list[Particle]:
    """The backfill scope, in one indexed scan.

    Default: ACTIVE particles carrying no structured claim. With
    ``structurizer_version``, ACTIVE particles whose *annotation* was produced
    by a different version — the regeneration scope, mirroring
    ``reindex --extractor-version``.

    Args:
        session: Active async SQLAlchemy session.
        structurizer_version: when given, select annotated particles stamped
            with a version other than this one instead of unannotated ones.
        limit: cap the number of rows returned.
    """
    stmt = select(ParticleRow).where(
        ParticleRow.status == Status.ACTIVE.value,
        *_needs_structured_claim(structurizer_version),
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return [row.to_model() for row in result.scalars()]


def _needs_structured_claim(structurizer_version: str | None) -> tuple[Any, ...]:
    """The backfill scope predicate, shared by the count and the fetch.

    Default (no ``structurizer_version``): particles this structurizer has not
    *attempted* — no stored triple **and** no recorded decline. Excluding
    recorded declines is what makes the pass terminable: a particle whose prose
    has no honest triple would otherwise be re-selected and re-paid on every
    future run, crowding fresh work out of each batch until progress stopped
    (observed: a 500-particle run declined 207, and those 207 led the next
    run's scan).

    With a ``structurizer_version``: everything a *different* version touched —
    both its triples and its declines. A better structurizer deserves another
    look at the prose an older one gave up on, which is the whole point of
    stamping the attempt.

    **Structure-canonical particles are out of scope in both branches.** For a
    ``STRUCTURED`` particle the triple is the *assertion* a parser read from the
    source, not a derived annotation over prose, so the
    content-structurizer regenerating it would overwrite an assertion with an
    LLM's guess at it — precisely what the spec forbids ("backfill writes
    only derived storage"). The right way to redo one is the extractor path:
    bump ``EXTRACTOR_VERSION`` and ``reindex --extractor-version <old>``, which
    mints new particles through the §6.6 ladder. Without the filter, a routine
    ``structure --structurizer-version <v>`` would sweep up every RDF- and
    Wikidata-derived triple, since none of them carries the standalone
    structurizer's version.
    """
    prose_canonical = ParticleRow.canonical_form != CanonicalForm.STRUCTURED.value
    attempted_by_current = ParticleRow.structurizer_version.is_(None)
    if structurizer_version is None:
        # Un-attempted: no payload and no stamp. (A payload without a stamp is
        # unreachable; a stamp without a payload is a recorded decline.)
        return (
            prose_canonical,
            ParticleRow.structured_claim_json.is_(None),
            attempted_by_current,
        )
    return (
        prose_canonical,
        ParticleRow.structurizer_version.isnot(None),
        ParticleRow.structurizer_version != structurizer_version,
    )


async def record_structured_claim_declined(
    session: AsyncSession,
    particle_id: str,
    *,
    structurizer_id: str,
    structurizer_version: str,
) -> None:
    """Record that a structurizer looked at this claim and found no honest triple.

    Writes the stamp with **no payload** — the mirror of a stored annotation.
    The three column states are therefore:

    ===========================  ==================  ============================
    ``structured_claim_json``    ``structurizer_*``  Meaning
    ===========================  ==================  ============================
    NULL                         NULL                never attempted
    NULL                         set                 attempted; declined (§2.9)
    set                          set                 annotated
    set                          NULL                unreachable (corrupt)
    ===========================  ==================  ============================

    This does not make absence illegal — the annotation is still absent, and
    nothing lints for it. It records *that we asked*, so the
    question is not re-paid for on every future run. A later structurizer
    version re-asks by construction (see :func:`_needs_structured_claim`).

    Touches only the stamp columns: never ``content``, ``confidence``, or
    provenance.

    Raises:
        ValueError: If the particle is missing.
    """
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"record_structured_claim_declined: particle {particle_id} not found")
    row.structured_claim_json = None
    row.structurizer_id = structurizer_id
    row.structurizer_version = structurizer_version
    row.structured_claim_generated_at = datetime.now(UTC)
    await session.flush()


async def count_particles_needing_structured_claim(
    session: AsyncSession, *, structurizer_version: str | None = None
) -> int:
    """How many ACTIVE particles the backfill still has to do.

    The **uncapped** backlog, deliberately separate from
    :func:`get_particles_needing_structured_claim`, whose ``limit`` is the
    per-run batch cap. Reporting the capped list length as "the scope" would
    tell an operator with a 21 k backlog that 200 particles need annotating —
    a silent truncation, and the number they would size the job from.

    A ``COUNT(*)``, so the dry-run probe does not load rows it will not use.
    """
    from sqlalchemy import func

    stmt = select(func.count(ParticleRow.id)).where(
        ParticleRow.status == Status.ACTIVE.value,
        *_needs_structured_claim(structurizer_version),
    )
    return await session.scalar(stmt) or 0


async def set_structured_claim(
    session: AsyncSession, particle_id: str, claim: StructuredClaim
) -> None:
    """Write (or replace) a particle's annotation. Flushes.

    Touches the payload column and the three stamp columns and **nothing
    else** — never ``content``, ``confidence``, provenance, or ``status``
    . ``canonical_form`` is not written here either: which form
    is the assertion is decided at creation, not by annotating.

    Raises:
        ValueError: If the particle is missing.
    """
    row = await session.get(ParticleRow, particle_id)
    if row is None:
        raise ValueError(f"set_structured_claim: particle {particle_id} not found")
    row.structured_claim_json = structured_claim_payload(claim)
    row.structurizer_id = claim.structurizer_id
    row.structurizer_version = claim.structurizer_version
    row.structured_claim_generated_at = claim.generated_at
    await session.flush()


async def count_structured_claim_coverage(session: AsyncSession) -> dict[str, Any]:
    """coverage: how many ACTIVE particles carry an annotation.

    Returns ``{"active": n, "annotated": n, "by_structurizer": {"id@version": n}}``.
    Absence is a legal permanent state, so this is a *count*, never a finding.
    """
    from sqlalchemy import func

    active = await session.scalar(
        select(func.count(ParticleRow.id)).where(ParticleRow.status == Status.ACTIVE.value)
    )
    result = await session.execute(
        select(
            ParticleRow.structurizer_id,
            ParticleRow.structurizer_version,
            func.count(ParticleRow.id),
        )
        .where(
            ParticleRow.status == Status.ACTIVE.value,
            ParticleRow.structured_claim_json.isnot(None),
        )
        .group_by(ParticleRow.structurizer_id, ParticleRow.structurizer_version)
    )
    by_structurizer = {f"{sid}@{ver}": count for sid, ver, count in result}
    return {
        "active": active or 0,
        "annotated": sum(by_structurizer.values()),
        "by_structurizer": by_structurizer,
    }


async def count_active_particles_by_schema_version(session: AsyncSession) -> dict[str, int]:
    """ACTIVE particle counts grouped by schema_version."""
    from sqlalchemy import func

    result = await session.execute(
        select(ParticleRow.schema_version, func.count(ParticleRow.id))
        .where(ParticleRow.status == Status.ACTIVE.value)
        .group_by(ParticleRow.schema_version)
    )
    return {row[0]: row[1] for row in result}
