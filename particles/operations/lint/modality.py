# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The adjudicability-default findings: stale stamps and the lens queue.

Both are structural, make no LLM call, and are ``INFO``: neither is a defect in
a claim.

- ``MODALITY_CLASSIFIER_STALE`` names, per classifier identity, how many ACTIVE
  claims carry a modality set by a rule other than today's. The regeneration
  verb is the remedy.
- ``MODALITY_GRANT_PENDING`` names each ACTIVE claim a regeneration run would
  have made adjudicable. Regeneration never writes that direction unattended,
  so the verdict waits here for ``particle reclassify``.
- ``MODALITY_LENS_DIVERGENCE`` names each ACTIVE claim whose adjudicability
  under the adopted lenses differs from its stored default. It is the queue a
  lens reading reaches the write path through: only an operator verdict
  (``particle reclassify``) changes what the write path arbitrates.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.modality import diverges
from particles.core.schema import LintFinding
from particles.operations.modality import load_modality_lens, modality_census, pending_grants
from particles.store.particle_store import get_active_particles, get_particles_by_ids


async def _report_modality_classifiers(session: AsyncSession) -> list[LintFinding]:
    """One INFO finding per stale classifier identity, with its count."""
    census = await modality_census(session)
    current = ", ".join(census.current)
    return [
        LintFinding(
            finding_type="MODALITY_CLASSIFIER_STALE",
            severity="INFO",
            detail=(
                f"{count} ACTIVE particle(s) carry an assertion_modality set by classifier "
                f"{classifier!r}, not a current one ({current})."
            ),
            recommended_action=(
                "Run `particles modality --dry-run` to size the backlog, then "
                "`particles modality` to reclassify it. Operator verdicts are never touched. "
                "Journal-extractor claims are left to re-extraction "
                "(`particles reindex --extractor-id journal-extractor`)."
            ),
        )
        for classifier, count in census.stale_by_classifier.items()
    ]


async def _check_modality_lens_divergence(session: AsyncSession) -> list[LintFinding]:
    """Each ACTIVE claim an adopted lens reads differently from its stored default."""
    lens = await load_modality_lens(session)
    if lens.empty:
        return []
    particles = await get_active_particles(session)
    readings = await lens.readings(session, particles)
    findings: list[LintFinding] = []
    for p in particles:
        reading = readings[p.id]
        if not diverges(reading, p.assertion_modality):
            continue
        stored = p.assertion_modality.value
        if reading.adjudicable:
            detail = (
                f"{reading.basis} reads this claim as adjudicable ({reading.modality.value}); "
                f"its stored default {stored} keeps it out of arbitration."
            )
            action = (
                "If the store should arbitrate it, run `particles particle reclassify "
                f"{p.id[:8]} --modality FALSIFIABLE --reason …`; the next consolidation "
                "run re-pairs it."
            )
        else:
            detail = (
                f"{reading.basis} reads this claim as not adjudicable "
                f"({reading.modality.value}); its stored default {stored} still lets "
                "the write path arbitrate it."
            )
            action = (
                "If the store should abstain, run `particles particle reclassify "
                f"{p.id[:8]} --modality {reading.modality.value} --reason …`."
            )
        findings.append(
            LintFinding(
                particle_id=p.id,
                particle_content=p.content,
                finding_type="MODALITY_LENS_DIVERGENCE",
                severity="INFO",
                detail=detail,
                recommended_action=action,
            )
        )
    return findings


async def _report_pending_modality_grants(session: AsyncSession) -> list[LintFinding]:
    """Each ACTIVE claim with a queued regeneration grant awaiting an operator verdict."""
    grants = await pending_grants(session)
    if not grants:
        return []
    particles = await get_particles_by_ids(session, sorted(grants))
    findings: list[LintFinding] = []
    for pid in sorted(grants):
        grant = grants[pid]
        p = particles.get(pid)
        model = f" ({grant.model})" if grant.model else ""
        findings.append(
            LintFinding(
                particle_id=pid,
                particle_content=p.content if p is not None else None,
                finding_type="MODALITY_GRANT_PENDING",
                severity="INFO",
                detail=(
                    f"Classifier {grant.classifier}{model} reads this claim as adjudicable "
                    f"(FALSIFIABLE); its stored default {grant.stored.value} keeps it out of "
                    "arbitration. Regeneration never makes a claim adjudicable unattended."
                ),
                recommended_action=(
                    f"If the store should arbitrate it, run `particles particle reclassify "
                    f"{pid[:8]} --modality FALSIFIABLE --reason …`; the next consolidation "
                    "run re-pairs it."
                ),
            )
        )
    return findings
