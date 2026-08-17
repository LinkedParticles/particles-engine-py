"""Operator trust-statement operation with event log.

Thin seam wrapping ``insert_trust_statement`` + the cascade so the operator
``trust statement-set`` action records a single ``TRUST_CHANGED`` event in the
same transaction — regardless of front-end. The logging lives here (not in
``insert_trust_statement``) because that store helper is *also* called by
``operations/review`` when a review resolves an INCONSISTENCY; that path is a
``REVIEW_RESOLVED`` event, not a standalone trust change.

``SourceTrustStatement.basis`` stays on the statement record (its standing
rationale); the event snapshots it so the change history survives later
overwrites.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import SourceTrustStatement
from particles.operations.cascade import run_trust_cascade
from particles.store.event_store import EventRefKind, OperatorEventType, record_event
from particles.store.trust_store import insert_trust_statement


async def set_trust_statement(
    session: AsyncSession,
    stmt: SourceTrustStatement,
    *,
    actor: str = "trust-statement-set",
) -> int:
    """Insert an operator SourceTrustStatement, run the cascade, log the event.

    Caller commits. Returns the number of INCONSISTENCY particles the cascade
    resolved.
    """
    await insert_trust_statement(session, stmt)
    resolved = await run_trust_cascade(session, stmt)
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.TRUST_CHANGED,
        reason=stmt.basis,
        refs=[(EventRefKind.TRUST_STATEMENT, stmt.statement_id)],
        payload={
            "kind": "statement",
            "domain": stmt.domain,
            "source_ref_type": stmt.source_ref.type.value,
            "source_ref_value": stmt.source_ref.value,
            "trust_rank": stmt.trust_rank,
            "policy_provenance": stmt.policy_provenance.value,
            "basis": stmt.basis,
        },
    )
    return resolved
