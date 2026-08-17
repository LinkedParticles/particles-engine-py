"""SQLAlchemy ORM models and repositories for source trust (§6.4)."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

from sqlalchemy import DateTime, Float, Index, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.schema import PolicyProvenance, SourceRef, SourceRefType, SourceTrustStatement
from particles.db import Base
from particles.store.event_store import OperatorEventType, record_event


class TrustStatementRow(Base):
    __tablename__ = "trust_statements"

    statement_id: Mapped[str] = mapped_column(String, primary_key=True)
    domain: Mapped[str] = mapped_column(String, nullable=False)
    source_ref_type: Mapped[str] = mapped_column(String, nullable=False)
    source_ref_value: Mapped[str] = mapped_column(String, nullable=False)
    trust_rank: Mapped[float] = mapped_column(Float, nullable=False)
    policy_provenance: Mapped[str] = mapped_column(String, nullable=False)
    asserted_by: Mapped[str] = mapped_column(String, nullable=False)
    asserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    basis: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_id: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        # Declared explicitly under the name migration 001 created (index=True
        # would imply ix_trust_statements_domain and make autogenerate churn).
        Index("ix_trust_domain", "domain"),
        Index("ix_trust_domain_ref", "domain", "source_ref_type", "source_ref_value"),
    )

    def to_model(self) -> SourceTrustStatement:
        return SourceTrustStatement(
            statement_id=self.statement_id,
            domain=self.domain,
            source_ref=SourceRef(
                type=SourceRefType(self.source_ref_type),
                value=self.source_ref_value,
            ),
            trust_rank=self.trust_rank,
            policy_provenance=PolicyProvenance(self.policy_provenance),
            asserted_by=self.asserted_by,
            asserted_at=self.asserted_at,
            basis=self.basis,
            review_id=self.review_id,
        )

    @classmethod
    def from_model(cls, s: SourceTrustStatement) -> TrustStatementRow:
        return cls(
            statement_id=s.statement_id,
            domain=s.domain,
            source_ref_type=s.source_ref.type.value,
            source_ref_value=s.source_ref.value,
            trust_rank=s.trust_rank,
            policy_provenance=s.policy_provenance.value,
            asserted_by=s.asserted_by,
            asserted_at=s.asserted_at,
            basis=s.basis,
            review_id=s.review_id,
        )


async def insert_trust_statement(session: AsyncSession, stmt: SourceTrustStatement) -> None:
    session.add(TrustStatementRow.from_model(stmt))
    await session.flush()


async def get_trust_statements_for_domain(
    session: AsyncSession, domain: str
) -> list[SourceTrustStatement]:
    result = await session.execute(
        select(TrustStatementRow).where(TrustStatementRow.domain == domain)
    )
    return [r.to_model() for r in result.scalars()]


# ---------------------------------------------------------------------------
# Source trust rules
# ---------------------------------------------------------------------------


class SourceTrustRow(Base):
    """One trust rule: either a domain baseline score or a URL-pattern modifier."""

    __tablename__ = "source_trust"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scope: Mapped[str] = mapped_column(String, nullable=False)  # "domain" | "url_pattern"
    pattern: Mapped[str] = mapped_column(String, nullable=False, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # domain rows only
    modifier: Mapped[float | None] = mapped_column(Float, nullable=True)  # url_pattern rows only
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    asserted_by: Mapped[str] = mapped_column(String, nullable=False)


def _extract_domain(uri_r: str) -> str:
    return urlparse(uri_r).netloc or uri_r


async def _lookup_domain_score(session: AsyncSession, domain: str) -> float:
    """Return the domain-specific score, falling back to the wildcard '*' row, then 0.50."""
    result = await session.execute(
        select(SourceTrustRow).where(
            SourceTrustRow.scope == "domain", SourceTrustRow.pattern.in_([domain, "*"])
        )
    )
    rows = result.scalars().all()
    # Prefer exact match over wildcard
    for row in rows:
        if row.pattern == domain and row.score is not None:
            return row.score
    for row in rows:
        if row.pattern == "*" and row.score is not None:
            return row.score
    return 0.50


async def _lookup_url_modifier(session: AsyncSession, uri_r: str) -> float:
    """Sum all URL-pattern modifiers whose regex matches the URI."""
    result = await session.execute(
        select(SourceTrustRow).where(SourceTrustRow.scope == "url_pattern")
    )
    total = 0.0
    for row in result.scalars():
        try:
            if re.search(row.pattern, uri_r) and row.modifier is not None:
                total += row.modifier
        except re.error:
            pass
    return total


async def resolve_trust_score(session: AsyncSession, uri_r: str | None) -> float:
    """Return the effective trust score [0.0, 1.0] for a corpus entry URI."""
    if uri_r is None or uri_r.startswith("file://"):
        return 0.50
    domain = _extract_domain(uri_r)
    base = await _lookup_domain_score(session, domain)
    mod = await _lookup_url_modifier(session, uri_r)
    return max(0.0, min(1.0, base + mod))


async def upsert_trust_rule(
    session: AsyncSession,
    scope: str,
    pattern: str,
    score: float | None,
    modifier: float | None,
    rationale: str | None = None,
    asserted_by: str = "operator",
    actor: str = "trust-set",
) -> SourceTrustRow:
    """Insert or replace a trust rule for the given scope+pattern.

    Records a ``TRUST_CHANGED`` operator event. A domain/pattern
    rule has no particle/subject/entry ref, so the event carries its target
    in ``payload`` only.
    """
    existing = (
        await session.execute(
            select(SourceTrustRow).where(
                SourceTrustRow.scope == scope, SourceTrustRow.pattern == pattern
            )
        )
    ).scalar_one_or_none()
    if existing:
        existing.score = score
        existing.modifier = modifier
        existing.rationale = rationale
        existing.asserted_by = asserted_by
        row = existing
    else:
        row = SourceTrustRow(
            id=str(uuid.uuid4()),
            scope=scope,
            pattern=pattern,
            score=score,
            modifier=modifier,
            rationale=rationale,
            created_at=datetime.now(UTC),
            asserted_by=asserted_by,
        )
        session.add(row)
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.TRUST_CHANGED,
        reason=rationale,
        payload={
            "kind": "rule",
            "scope": scope,
            "pattern": pattern,
            "score": score,
            "modifier": modifier,
        },
    )
    return row


async def get_trust_rank(
    session: AsyncSession,
    domain: str,
    source_ref_type: str,
    source_ref_value: str,
) -> float | None:
    """Return the most authoritative trust_rank for a source within a domain, or None."""
    result = await session.execute(
        select(TrustStatementRow)
        .where(
            TrustStatementRow.domain == domain,
            TrustStatementRow.source_ref_type == source_ref_type,
            TrustStatementRow.source_ref_value == source_ref_value,
        )
        .order_by(TrustStatementRow.asserted_at.desc())
    )
    row = result.scalar_one_or_none()
    return row.trust_rank if row else None


async def get_layered_trust_rank(
    session: AsyncSession,
    domain: str,
    entry_id: str,
    source_type: str,
    uri_r: str | None = None,
    author_id: str | None = None,
) -> float:
    """Return trust_rank using the Extension B layered lookup.

    The §6.4 four-tier cascade, first match wins: CORPUS_ENTRY-scoped →
    AUTHOR-scoped (the snapshot's ``author_id``, §6.5) → SOURCE_TYPE-scoped →
    the URL baseline. A particle with no ``author_id`` skips tier 2.
    """
    rank = await get_trust_rank(session, domain, "CORPUS_ENTRY", entry_id)
    if rank is not None:
        return rank
    if author_id is not None:
        rank = await get_trust_rank(session, domain, "AUTHOR", author_id)
        if rank is not None:
            return rank
    rank = await get_trust_rank(session, domain, "SOURCE_TYPE", source_type)
    if rank is not None:
        return rank
    return await resolve_trust_score(session, uri_r)


async def count_reviewer_confirmations(
    session: AsyncSession,
    domain: str,
    source_ref_type: str,
    source_ref_value: str,
) -> int:
    """Count distinct REVIEWER_DERIVED SourceTrustStatements for a source in a domain."""
    from sqlalchemy import func

    result = await session.execute(
        select(func.count(TrustStatementRow.statement_id)).where(
            TrustStatementRow.domain == domain,
            TrustStatementRow.source_ref_type == source_ref_type,
            TrustStatementRow.source_ref_value == source_ref_value,
            TrustStatementRow.policy_provenance == PolicyProvenance.REVIEWER_DERIVED.value,
        )
    )
    return result.scalar_one() or 0
