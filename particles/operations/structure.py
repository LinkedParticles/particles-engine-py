"""Structured-claim backfill — annotate particles that carry no triple.

Built in the ``operations/reindex.py`` mold, because that module is this repo's
worked answer to the same problem: walk a discovered scope, pay one LLM call per
item, hold a rate limit, report what happened, and never let one failure end the
run. Scope discovery is a single indexed scan; the rate limit is the same fixed
inter-item delay; failures are collected, not raised.

**What this pass may write.** The payload column and the three stamp columns,
via ``set_structured_claim`` — and nothing else. It never re-extracts, never
mints or supersedes a particle, never touches ``status``, and cannot touch
``content``, ``confidence`` or provenance. A structurization failure
and a structurization success are epistemically identical states of the belief:
the claim is unchanged either way, and absence of an annotation is a legal
permanent state.

Deliberately **not** a dream-cycle pass: ``run_consolidation``
composes *existing* passes and adds no detection of its own, so scheduling this
one is a separate decision to make once its per-item cost is measured on a real
store.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.extraction.structure import (
    STRUCTURIZER_ID,
    STRUCTURIZER_VERSION,
    bind_subject_id,
    structure_content,
)
from particles.observability import traced
from particles.store.particle_store import (
    count_particles_needing_structured_claim,
    count_structured_claim_coverage,
    get_particles_needing_structured_claim,
    record_structured_claim_declined,
    set_structured_claim,
)
from particles.store.subject_store import get_subject

log = logging.getLogger(__name__)


@traced("structure")
async def backfill_structured_claims(
    session: AsyncSession,
    *,
    limit: int | None = None,
    rate_limit_per_minute: int | None = None,
    structurizer_version: str | None = None,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Annotate ACTIVE particles that carry no structured claim.

    Args:
        limit: cap on particles annotated this run. Defaults to
            ``structured_claim.backfill_batch_limit``. The pass is resumable —
            a small cap is a feature, not a limitation.
        rate_limit_per_minute: max structurizer calls per minute. Defaults to
            ``structured_claim.backfill_rate_limit_per_minute``; 0 disables the
            delay.
        structurizer_version: regenerate annotations stamped with a version
            *other* than this one, instead of annotating unannotated particles
            (mirrors ``reindex --extractor-version``).
        dry_run: report the **uncapped** backlog plus the batch cap and how
            many runs it implies, write nothing. This is the operator's
            job-sizing probe, so it must never report the cap as the job.
        progress: optional callback for human-readable progress lines.

    Returns:
        A summary dict. A dry run reports ``backlog`` / ``batch_limit`` /
        ``runs_needed``; a real run reports the ``scope`` it took, what
        ``remaining`` after it, counts of annotated / skipped / failed, the
        stamp applied, and the resulting coverage census.
    """
    cfg = get_config().structured_claim
    # ``--limit 0`` means the whole backlog. Without it the operator has to
    # paste a magic number from the dry run, and the natural thing to type
    # ("0") would have meant "annotate nothing" — a useless meaning for a
    # value someone will plausibly reach for.
    effective_limit = cfg.backfill_batch_limit if limit is None else limit
    uncapped = effective_limit <= 0
    rate = (
        cfg.backfill_rate_limit_per_minute
        if rate_limit_per_minute is None
        else rate_limit_per_minute
    )

    # The uncapped backlog is counted separately from the capped batch, so a
    # binding cap is always *disclosed* rather than silently reported as the
    # whole job (the "probed X of Y" convention). Without this, a
    # 21 k-particle store's dry run says "scope: 200".
    backlog = await count_particles_needing_structured_claim(
        session, structurizer_version=structurizer_version
    )
    coverage = await count_structured_claim_coverage(session)

    if dry_run:
        log.info("Structured-claim backlog: %d particles", backlog)
        if progress is not None:
            progress(f"Structured-claim backlog: {backlog} particles")
        return {
            "dry_run": True,
            "backlog": backlog,
            "batch_limit": 0 if uncapped else effective_limit,
            "runs_needed": 1 if uncapped else -(-backlog // effective_limit),
            "annotated": 0,
            "skipped": 0,
            "failed": 0,
            "structurizer": f"{STRUCTURIZER_ID}@{STRUCTURIZER_VERSION}",
            "coverage": coverage,
        }

    scope = await get_particles_needing_structured_claim(
        session,
        structurizer_version=structurizer_version,
        limit=None if uncapped else effective_limit,
    )
    total = len(scope)
    log.info("Structured-claim backfill: annotating %d of %d", total, backlog)
    if progress is not None:
        progress(f"Structured-claim backfill: annotating {total} of {backlog} particles")

    # The rate cap is a *ceiling on calls per minute*, so the delay is the
    # remainder of the interval after the call itself — not a flat sleep on top
    # of it. Sleeping the full interval regardless made a nominal 60/min run at
    # ~1.3 s latency actually issue ~26/min, better than doubling the wall clock
    # of a 21 k-particle backfill. (``reindex`` still has the flat-sleep shape;
    # fixing it there is a separate change to a separate pass.)
    interval = 60.0 / rate if rate > 0 else 0.0
    annotated = 0
    # A particle the structurizer declines to triple-ize is *skipped*, not
    # failed: "this prose has no honest triple" is a valid, permanent answer
    #, not an error to alarm the operator with. The decline is
    # RECORDED so it is not re-asked on every future run — see
    # ``record_structured_claim_declined``.
    skipped = 0
    failed: list[str] = []

    for i, particle in enumerate(scope, start=1):
        started = time.monotonic()
        if progress is not None:
            progress(f"[{i}/{total}] {particle.id[:8]}… {particle.content[:60]}")
        try:
            names, ids = await _resolved_subjects(session, particle.subject_ids)
            claim = await structure_content(particle.content, names)
            if claim is None:
                await record_structured_claim_declined(
                    session,
                    particle.id,
                    structurizer_id=STRUCTURIZER_ID,
                    structurizer_version=STRUCTURIZER_VERSION,
                )
                skipped += 1
            else:
                await set_structured_claim(session, particle.id, bind_subject_id(claim, names, ids))
                annotated += 1
        except Exception as exc:
            log.error("Structurizing particle %s failed: %s", particle.id, exc)
            if progress is not None:
                progress(f"[{i}/{total}] FAILED: {exc}")
            failed.append(particle.id)
        # Commit as we go. The pass is long by construction (hours, for a store
        # of any size), so a single commit at the end meant an interrupt — or an
        # account-level LLM failure — discarded every call already paid for.
        if i % cfg.backfill_commit_interval == 0:
            await session.commit()
        if interval > 0:
            remaining_delay = interval - (time.monotonic() - started)
            if remaining_delay > 0:
                await asyncio.sleep(remaining_delay)

    await session.commit()
    return {
        "dry_run": False,
        "scope": total,
        # What is left after this run — so a capped run never reads as "done".
        "remaining": await count_particles_needing_structured_claim(
            session, structurizer_version=structurizer_version
        ),
        "annotated": annotated,
        "skipped": skipped,
        "failed": len(failed),
        "failed_ids": failed,
        "structurizer": f"{STRUCTURIZER_ID}@{STRUCTURIZER_VERSION}",
        "coverage": await count_structured_claim_coverage(session),
    }


async def _resolved_subjects(
    session: AsyncSession, subject_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Return (canonical names, ids) for a particle's subjects, kept aligned.

    The names are offered to the structurizer so its subject term matches an
    entity the store already knows, and used again by ``bind_subject_id`` to
    turn that term back into a UUID — which only works if the two lists stay
    positionally aligned. A subject that no longer exists is therefore dropped
    from *both*, never from one.
    """
    names: list[str] = []
    ids: list[str] = []
    for subject_id in subject_ids:
        subject = await get_subject(session, subject_id)
        if subject is not None:
            names.append(subject.canonical_name)
            ids.append(subject_id)
    return names, ids
