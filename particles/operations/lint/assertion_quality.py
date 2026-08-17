"""Compound-assertion lint — structural, no LLM.

Flags ACTIVE, agent-asserted particles whose ``content`` breaches the same
claim-granularity soft-gate the MCP write surface applies at assert time
(``particles/core/granularity.py``). It surfaces the compound particles
already sitting in a store — e.g. beliefs asserted before the gate existed —
for operator review. Read-only: no status transition, severity ``WARNING``.

"Agent-asserted" is keyed off ``asserted_by == mcp.write.asserter_identity``
(the server-bound identity), so the check catches both new ``AGENT_ASSERTED``
particles and any legacy MCP-asserted ones. The same ``mcp.write`` thresholds
the gate uses are read here, so the gate and the lint cannot diverge.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.granularity import granularity_violation
from particles.core.schema import LintFinding
from particles.store.particle_store import get_active_particles


async def _check_compound_assertions(session: AsyncSession) -> list[LintFinding]:
    """Flag agent-asserted ACTIVE particles that breach the claim-granularity gate."""
    cfg = get_config().mcp.write
    identity = cfg.asserter_identity
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        if p.asserted_by != identity:
            continue  # not an assertion from this server's bound identity
        reason = granularity_violation(
            p.content,
            max_chars=cfg.max_assertion_chars,
            max_sentences=cfg.max_assertion_sentences,
        )
        if reason is None:
            continue
        findings.append(
            LintFinding(
                particle_id=p.id,
                particle_content=p.content,
                finding_type="COMPOUND_ASSERTION",
                severity="WARNING",
                detail=f"Agent-asserted particle: {reason}.",
                recommended_action=(
                    "Operator-review only. If genuinely multi-claim, "
                    "`particle_supersede` it with one atomic claim per call "
                    "(or RETRACT). New compound assertions are now rejected by "
                    "the assert-time soft-gate."
                ),
            )
        )
    return findings
