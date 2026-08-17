"""Citation-signal lint check — L-CITE-01.

Surfaces undeposited primary sources the corpus leans on: a URL cited by many
distinct sources but never deposited means the database represents *hearsay
about* the source rather than the source itself. The lint framing (preferred) anchors the suggestion to a real grounding gap, not URL
popularity. Structural / read-only — no LLM, no mutation.

The check reuses the ranking operation (``suggest_deposits``) with the more
conservative ``citation_signal.lint_min_distinct_sources`` gate, so lint stays
quieter than the ``corpus links suggest`` verb.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import LintFinding
from particles.operations.deposit_suggest import suggest_deposits


async def _check_undeposited_cited_sources(session: AsyncSession) -> list[LintFinding]:
    """Flag undeposited URLs cited by ≥ ``lint_min_distinct_sources`` sources."""
    cfg = get_config().citation_signal
    report = await suggest_deposits(session, min_sources=cfg.lint_min_distinct_sources)
    findings: list[LintFinding] = []
    for s in report.suggestions:
        findings.append(
            LintFinding(
                finding_type="UNDEPOSITED_CITED_SOURCE",
                severity="INFO",
                detail=(
                    f"{s.distinct_sources} distinct sources cite {s.canonical_url}, "
                    "but it is not deposited — the corpus knows this source only by hearsay."
                ),
                recommended_action=(
                    f"Deposit the primary source to ground these claims: "
                    f"particles deposit {s.canonical_url}"
                ),
            )
        )
    return findings
