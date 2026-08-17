"""L-SEM-03: flag Wikidata ExternalRefs with low confidence.

These links were created by the subject resolver but the embedding similarity
between the Wikidata entity description and the particle content was low,
suggesting a context mismatch (e.g. 'poet' → Q49757 vs POET Technologies).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import LintFinding
from particles.store.subject_store import list_all_subjects


async def _check_wikidata_link_confidence(
    session: AsyncSession,
) -> list[LintFinding]:
    """L-SEM-03: flag subjects whose Wikidata ExternalRef has confidence below threshold.

    These links were created by the subject resolver but the embedding similarity
    between the Wikidata entity description and the particle content was low,
    suggesting a context mismatch (e.g. 'poet' → Q49757 vs POET Technologies).
    """
    from particles.config import get_config

    threshold = get_config().subjects.wikidata_link_suppress_threshold
    findings: list[LintFinding] = []

    for subject in await list_all_subjects(session):
        for ref in subject.external_ids:
            if ref.namespace == "wikidata" and ref.confidence < threshold:
                findings.append(
                    LintFinding(
                        particle_id=None,
                        subject_id=subject.id,
                        finding_type="WIKIDATA_LINK_MISMATCH",
                        severity="WARNING",
                        detail=(
                            f"Subject '{subject.canonical_name}' has Wikidata link "
                            f"{ref.id} (confidence {ref.confidence:.2f} < {threshold:.2f}). "
                            f"Wikidata description: '{subject.description or 'unknown'}'. "
                            # Both resolutions are emitted fully-specified: the
                            # subject's short id and the offending
                            # ``wikidata:QID`` are both known here, so the
                            # operator runs the command verbatim rather than
                            # substituting a placeholder by hand. CLI
                            # invocations are wrapped in backticks so Markdown
                            # renderers (Obsidian, GFM) treat them as inline
                            # code; nothing is angle-bracketed, so nothing is
                            # parsed as a raw HTML tag. This mirrors the
                            # convention in ``_render_subject_note`` (which
                            # writes the same commands into the unverified-link
                            # callout).
                            f"If the link is wrong, remove it: "
                            f"`particles subjects unlink {subject.id[:8]} wikidata:{ref.id}`. "
                            f"If it is correct, confirm it: "
                            f"`particles subjects confirm {subject.id[:8]} wikidata:{ref.id}`."
                        ),
                    )
                )

    return findings
