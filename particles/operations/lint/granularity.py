"""Granularity detectors (structural length outliers + LLM violation check).

A particle is meant to express one atomic, separately-falsifiable claim.

  - ``_check_granularity_length`` — flag particles whose ``content`` length is
    more than 3× the median for ACTIVE particles. No LLM call; cheap pre-check.
  - ``_check_granularity_violations`` — LLM-assisted; asks the model whether
    the content contains multiple independent claims. Skips particles under
    100 chars as a noise filter.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import LintFinding
from particles.operations._llm import _llm_call
from particles.store.particle_store import get_active_particles

_GRANULARITY_LENGTH_MULTIPLIER = 3.0  # flag if content > 3× median length


async def _check_granularity_length(session: AsyncSession) -> list[LintFinding]:
    """Flag particles whose content length significantly exceeds the median."""
    particles = await get_active_particles(session)
    if not particles:
        return []
    lengths = [len(p.content) for p in particles]
    median_len = sorted(lengths)[len(lengths) // 2]
    threshold = median_len * _GRANULARITY_LENGTH_MULTIPLIER

    findings: list[LintFinding] = []
    for p in particles:
        if len(p.content) > threshold:
            findings.append(
                LintFinding(
                    particle_id=p.id,
                    particle_content=p.content,
                    finding_type="GRANULARITY_VIOLATION_CANDIDATE",
                    severity="WARNING",
                    detail=(
                        f"Content length {len(p.content)} chars >"
                        f" {_GRANULARITY_LENGTH_MULTIPLIER}× median ({median_len});"
                        " may contain multiple claims"
                    ),
                    recommended_action=(
                        "Inspect with `particles particle show <id>`. "
                        "Operator-review only; no automated split exists. "
                        "If genuinely multi-claim: RETRACT the particle and re-extract "
                        "its source entry (`particles reindex --entry-ids <entry_id>`)."
                    ),
                )
            )
    return findings


async def _check_granularity_violations(session: AsyncSession) -> list[LintFinding]:
    """Detect particles with multiple independent claims using LLM."""
    findings: list[LintFinding] = []
    for p in await get_active_particles(session):
        if len(p.content) < 100:
            continue  # too short to contain multiple claims; skip
        violation = await _llm_check_granularity(p.content)
        if violation:
            findings.append(
                LintFinding(
                    particle_id=p.id,
                    particle_content=p.content,
                    finding_type="GRANULARITY_VIOLATION",
                    severity="WARNING",
                    detail=f"Particle may contain multiple independent claims: {violation}",
                    recommended_action=(
                        "Operator-review only; no automated split exists. "
                        "If confirmed: RETRACT and re-extract its source entry "
                        "(`particles reindex --entry-ids <entry_id>`)."
                    ),
                )
            )
    return findings


async def _llm_check_granularity(content: str) -> str | None:
    """Return a description of the violation or None if the particle is atomic."""
    prompt = (
        "Does this text contain multiple independent, separately-falsifiable claims? "
        "Answer with exactly one of:\n"
        "- 'YES: <list the separate claims briefly>' if it contains multiple\n"
        "- 'NO' if it contains a single atomic claim\n\n"
        f"Text: {content}"
    )
    response = await _llm_call(prompt, max_tokens=150)
    if response and response.upper().startswith("YES"):
        return response[4:].strip()
    return None
