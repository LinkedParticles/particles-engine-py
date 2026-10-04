# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles reindex --estimate``: what a version bump changes, measured on a sample.

A version-scoped reindex re-extracts every snapshot stamped with the old
version and supersedes what it re-extracts, at full price for each. Whether a
snapshot's output would differ is unknowable without running it, so the
estimate runs a few: it re-extracts a seeded sample of the snapshots the sweep
would re-run, compares each sample's new claims with its stored ones under the
equivalence judge, and projects the share of the scope that changes
and what the whole sweep costs. The operator can then confirm, narrow, or skip.

**It spends only on the sample and writes nothing.** The re-extraction calls
the extractor directly, never the pipeline, so no particle, relation, status
or snapshot row is touched; the one record is the ``EXTRACT_RUN`` event the
caller's meter appends (``route: reindex``).

Gather, decide, apply (D2): :func:`plan_reindex_estimate` reads the
scope and chooses the sample, free; :func:`run_reindex_estimate` spends on it.
The decisions between them are pure functions of plain values
(:func:`choose_sample`, :func:`pair_claims`, :func:`claim_change`,
:func:`project_sweep`), tested without a store or a model.

**Read the share as an upper bound.** Two re-extractions of the same text
under the same prompt rarely return the same claim set, so a sample changes
partly because the model is not deterministic. A claim the pipeline would have
folded into another entry's identical claim also reads as added
here. Neither inflation applies to the projected cost, which is the price of
re-running the whole scope whatever it would change.
"""

from __future__ import annotations

import difflib
import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import JudgeVerdictKind, Particle
from particles.corpus.deposit import load_blob
from particles.corpus.store import get_entry, get_snapshot
from particles.extraction.registry import select_extractor
from particles.llm import AccountLevelLLMError
from particles.llm.usage import LLMUsage, format_usd, track_usage
from particles.operations.links_suggest import judge_claim_pairs
from particles.operations.reindex import (
    ReindexPlan,
    SnapshotPlan,
    resolve_reindex_work,
)
from particles.operations.spend_estimate import (
    PassEstimate,
    context_call_estimate,
    extraction_estimate,
)
from particles.store.particle_store import get_active_particles_for_entry

log = logging.getLogger(__name__)

#: Two-sided 95% normal quantile, for the share's Wilson interval.
_Z95 = 1.96


# ---------------------------------------------------------------------------
# Pure decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePopulation:
    """Which of a plan's snapshots a sample may be drawn from, and why the rest may not."""

    eligible: list[SnapshotPlan]
    #: Append-only entries replayed whole: one snapshot read
    #: alone is not what the sweep would do with it.
    replayed: int = 0
    #: Snapshots whose blob is missing: the sweep fails on them anyway.
    blob_missing: int = 0
    #: Snapshots with no stored claim (FAILED, PENDING, or extracted to
    #: nothing): there is nothing to compare, and the sweep re-runs them anyway.
    no_claims: int = 0


def sample_population(plan: ReindexPlan) -> SamplePopulation:
    """Split the plan's snapshots into the sampleable and the counted-out."""
    eligible: list[SnapshotPlan] = []
    replayed = blob_missing = no_claims = 0
    for sp in plan.snapshot_plans:
        if sp.replayed:
            replayed += 1
        elif sp.blob_missing:
            blob_missing += 1
        elif sp.particles == 0:
            no_claims += 1
        else:
            eligible.append(sp)
    return SamplePopulation(
        eligible=eligible, replayed=replayed, blob_missing=blob_missing, no_claims=no_claims
    )


def choose_sample(population: Sequence[SnapshotPlan], size: int, seed: int) -> list[SnapshotPlan]:
    """A seeded sample of ``size`` snapshots, the same one for the same scope and seed.

    The population is put in snapshot-id order first, so the sample does not
    depend on the order the scope was gathered in.
    """
    ordered = sorted(population, key=lambda sp: (sp.snapshot_id, sp.entry_id))
    if size >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, size)


def normalise_claim(text: str) -> str:
    """The form two claims are compared in verbatim: case, spacing and a final stop ignored."""
    return " ".join(text.casefold().split()).rstrip(".")


@dataclass(frozen=True)
class ClaimPairing:
    """How a snapshot's stored claims line up with a re-extraction's, before any judging."""

    #: Claims present on both sides verbatim (after :func:`normalise_claim`).
    exact: int
    #: ``(stored, new)`` pairs left after the verbatim matches, each new claim
    #: paired with its most similar remaining stored claim: what the judge reads.
    pairs: list[tuple[str, str]]
    #: Stored claims with no counterpart at all (the re-extraction has fewer).
    unmatched_stored: int
    #: New claims with no counterpart at all (the re-extraction has more).
    unmatched_new: int


def pair_claims(stored: Sequence[str], new: Sequence[str]) -> ClaimPairing:
    """Line up two claim lists: verbatim matches first, then greedy closest pairs.

    Verbatim matches cost nothing to confirm. The rest are paired greedily by
    lexical similarity, most similar first, so the judge reads each new claim
    beside the stored claim it most plausibly restates; anything left over on
    the longer side is unmatched.
    """
    remaining_stored = [normalise_claim(c) for c in stored]
    stored_text = list(stored)
    residual_new: list[tuple[str, str]] = []
    exact = 0
    for claim in new:
        key = normalise_claim(claim)
        if key in remaining_stored:
            index = remaining_stored.index(key)
            del remaining_stored[index]
            del stored_text[index]
            exact += 1
        else:
            residual_new.append((key, claim))

    scored = sorted(
        (
            (difflib.SequenceMatcher(None, new_key, stored_key).ratio(), n, s)
            for n, (new_key, _) in enumerate(residual_new)
            for s, stored_key in enumerate(remaining_stored)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    taken_new: set[int] = set()
    taken_stored: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, n, s in scored:
        if n in taken_new or s in taken_stored:
            continue
        taken_new.add(n)
        taken_stored.add(s)
        matched.append((n, s))
    matched.sort()
    return ClaimPairing(
        exact=exact,
        pairs=[(stored_text[s], residual_new[n][1]) for n, s in matched],
        unmatched_stored=len(remaining_stored) - len(taken_stored),
        unmatched_new=len(residual_new) - len(taken_new),
    )


@dataclass(frozen=True)
class ClaimChange:
    """One sample's outcome: how many claims differ, of how many."""

    #: Claims on either side with no equivalent on the other: unmatched ones,
    #: plus both claims of every pair the judge did not call a paraphrase.
    differing: int
    #: Claims on both sides together.
    total: int

    @property
    def changed(self) -> bool:
        """Whether the re-extraction's claim set differs from the stored one at all."""
        return self.differing > 0


def claim_change(pairing: ClaimPairing, verdicts: Sequence[JudgeVerdictKind]) -> ClaimChange:
    """Decide one sample from its pairing and the judge's verdict on each pair.

    Only ``PARAPHRASE`` counts as the same claim; ``UNSURE`` counts as
    changed, the direction that errs toward re-running.
    """
    if len(verdicts) != len(pairing.pairs):
        raise ValueError(f"{len(verdicts)} verdict(s) for {len(pairing.pairs)} pair(s)")
    non_equivalent = sum(1 for v in verdicts if v is not JudgeVerdictKind.PARAPHRASE)
    total = (
        2 * (pairing.exact + len(pairing.pairs)) + pairing.unmatched_stored + pairing.unmatched_new
    )
    differing = 2 * non_equivalent + pairing.unmatched_stored + pairing.unmatched_new
    return ClaimChange(differing=differing, total=total)


def wilson_interval(successes: int, trials: int, z: float = _Z95) -> tuple[float, float]:
    """The Wilson score interval for a binomial share (``(0, 1)`` with no trials)."""
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


class SweepProjection(BaseModel):
    """What the full sweep would change and cost, projected from a measured sample."""

    #: Samples measured (re-extracted without a transient failure).
    measured: int
    #: Of those, samples whose claim set changed.
    changed: int
    #: ``changed / measured``; ``None`` with nothing measured.
    share: float | None
    #: The share's 95% Wilson interval.
    share_low: float
    share_high: float
    #: ``share`` applied to the sampleable population, rounded.
    projected_changed: int | None
    #: Claims differing over claims compared, across the whole sample.
    claim_change_rate: float | None
    #: US$ the sample's re-extractions cost, measured at list price.
    sample_extraction_usd: float | None
    #: The sample's re-extraction cost scaled by the sweep's bytes over the
    #: sample's: what re-running the whole scope would cost at the same rate.
    projected_sweep_usd: float | None


def project_sweep(
    *,
    outcomes: Sequence[tuple[int, ClaimChange]],
    population: int,
    sweep_bytes: int,
    sample_extraction_usd: float | None,
) -> SweepProjection:
    """Project the measured sample onto the scope. Pure arithmetic.

    Args:
        outcomes: ``(source_bytes, change)`` for each measured sample.
        population: snapshots the sample was drawn from.
        sweep_bytes: raw bytes of every snapshot the sweep would re-extract,
            the sampleable ones and the rest alike.
        sample_extraction_usd: what the sample's re-extractions cost, or
            ``None`` when a model it ran on is unpriced.

    The cost scales by bytes, not by snapshot count, because extraction is
    priced per token and a sample of short snapshots would otherwise
    under-price a scope of long ones.
    """
    measured = len(outcomes)
    changed = sum(1 for _, change in outcomes if change.changed)
    share = changed / measured if measured else None
    low, high = wilson_interval(changed, measured)
    compared = sum(change.total for _, change in outcomes)
    differing = sum(change.differing for _, change in outcomes)
    sample_bytes = sum(size for size, _ in outcomes)
    projected_usd = (
        sample_extraction_usd * sweep_bytes / sample_bytes
        if sample_extraction_usd is not None and sample_bytes > 0
        else None
    )
    return SweepProjection(
        measured=measured,
        changed=changed,
        share=share,
        share_low=low,
        share_high=high,
        projected_changed=round(share * population) if share is not None else None,
        claim_change_rate=differing / compared if compared else None,
        sample_extraction_usd=sample_extraction_usd,
        projected_sweep_usd=projected_usd,
    )


# ---------------------------------------------------------------------------
# Gather: the scope and the sample (free)
# ---------------------------------------------------------------------------


class EstimatePlan(BaseModel):
    """The sample an estimate would spend on, chosen before any spend."""

    plan: ReindexPlan
    sample: list[SnapshotPlan]
    population: int
    replayed: int = 0
    blob_missing: int = 0
    no_claims: int = 0
    seed: int
    #: Pre-spend list-price estimate of the sample: its re-extractions plus
    #: the judge calls at most one per stored-claim batch could need.
    sample_estimate_calls: int = 0
    sample_estimate_usd: float | None = 0.0
    #: Pre-spend list-price estimate of the whole sweep, for comparison with
    #: the projection the sample measures.
    sweep_estimate_usd: float | None = 0.0

    def format_lines(self) -> list[str]:
        """The human plan printed before the operator is asked to spend."""
        excluded = []
        if self.replayed:
            excluded.append(f"{self.replayed} replayed append-only")
        if self.no_claims:
            excluded.append(f"{self.no_claims} with no stored claim")
        if self.blob_missing:
            excluded.append(f"{self.blob_missing} missing their blob")
        lines = [self.plan.format_line()]
        line = (
            f"Estimate: re-extract {len(self.sample)} of {self.population} sampleable "
            f"snapshot(s) (seed {self.seed}) and judge their claims against the stored ones"
        )
        if excluded:
            line += f"; not sampled: {', '.join(excluded)} (re-run by the sweep regardless)"
        lines.append(line)
        lines.append(
            f"Estimated sample spend: ~{self.sample_estimate_calls} LLM call(s), "
            f"{_usd(self.sample_estimate_usd)} at list price; the full sweep "
            f"{_usd(self.sweep_estimate_usd)} at list price."
        )
        return lines


def _usd(value: float | None) -> str:
    return "≈ " + format_usd(value) if value is not None else "unpriced"


async def plan_reindex_estimate(
    session: AsyncSession,
    *,
    extractor_version: str,
    entry_ids: list[str] | None = None,
    extractor_id: str | None = None,
    include_failed: bool = True,
    provider_model: str | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
) -> EstimatePlan:
    """Resolve the scope a live reindex would sweep and choose the sample. Reads only.

    The scope is resolved exactly as ``reindex --dry-run`` resolves it
    (``resolve_reindex_work``), so the sample is drawn from what the sweep
    would actually re-run. ``sample_size`` and ``seed`` default to
    ``reindex.estimate_sample_size`` and ``reindex.estimate_seed``.
    """
    cfg = get_config().reindex
    size = sample_size if sample_size is not None else cfg.estimate_sample_size
    if size < 1:
        raise ValueError("the estimate's sample size must be at least 1")
    seed_value = seed if seed is not None else cfg.estimate_seed
    resolved = await resolve_reindex_work(
        session,
        entry_ids=entry_ids,
        extractor_version=extractor_version,
        extractor_id=extractor_id,
        include_failed=include_failed,
        provider_model=provider_model,
        dry_run=True,
    )
    population = sample_population(resolved.plan)
    sample = choose_sample(population.eligible, size, seed_value)
    batch = cfg.estimate_judge_batch_pairs
    judge_calls = sum(math.ceil(sp.particles / batch) for sp in sample)
    sample_cost: PassEstimate = extraction_estimate(
        [sp.source_bytes for sp in sample]
    ) + context_call_estimate("semantic_lint", judge_calls)
    sweep_cost = extraction_estimate(
        [sp.source_bytes for sp in resolved.plan.snapshot_plans if not sp.blob_missing]
    )
    return EstimatePlan(
        plan=resolved.plan,
        sample=sample,
        population=len(population.eligible),
        replayed=population.replayed,
        blob_missing=population.blob_missing,
        no_claims=population.no_claims,
        seed=seed_value,
        sample_estimate_calls=sample_cost.calls,
        sample_estimate_usd=sample_cost.usd,
        sweep_estimate_usd=sweep_cost.usd,
    )


# ---------------------------------------------------------------------------
# Apply: re-extract and judge the sample (spends)
# ---------------------------------------------------------------------------


class SampleResult(BaseModel):
    """One sampled snapshot, re-extracted and compared."""

    entry_id: str
    snapshot_id: str
    source_bytes: int
    stored_claims: int = 0
    new_claims: int = 0
    #: Claims matched verbatim, and pairs the judge read.
    exact: int = 0
    judged: int = 0
    differing: int = 0
    changed: bool | None = None
    #: Why the sample was not measured (a transient LLM failure, a missing
    #: snapshot); an unmeasured sample is left out of the share.
    error: str | None = None


class ReindexEstimate(BaseModel):
    """The estimate's report: the plan it ran, each sample, and the projection."""

    plan: EstimatePlan
    samples: list[SampleResult] = Field(default_factory=list)
    projection: SweepProjection
    #: The sample's measured usage, every purpose (re-extraction and judge).
    llm_usage: LLMUsage | None = None

    def format_lines(self) -> list[str]:
        """The human report printed after the sample ran."""
        p = self.projection
        lines: list[str] = []
        if p.share is None:
            lines.append("Estimate: no sample could be measured; see --format json for why.")
        else:
            lines.append(
                f"Estimate: {p.changed} of {p.measured} sampled snapshot(s) changed their "
                f"claims ({p.share:.0%}; 95% interval {p.share_low:.0%}–{p.share_high:.0%}), "
                f"so about {p.projected_changed} of {self.plan.population} sampleable "
                "snapshot(s) would change."
            )
        if p.claim_change_rate is not None:
            lines.append(
                f"  {p.claim_change_rate:.0%} of the sample's claims had no equivalent on the "
                "other side. Both figures are upper bounds: re-extracting unchanged text "
                "also varies run to run."
            )
        if p.projected_sweep_usd is not None and p.sample_extraction_usd is not None:
            lines.append(
                f"  Full sweep ≈ {format_usd(p.projected_sweep_usd)} at the sample's measured "
                f"rate (the sample's re-extraction cost {format_usd(p.sample_extraction_usd)}; "
                f"list-price estimate {_usd(self.plan.sweep_estimate_usd)})."
            )
        else:
            lines.append(
                f"  Full sweep: list-price estimate {_usd(self.plan.sweep_estimate_usd)} "
                "(the sample's cost was not measurable)."
            )
        errors = [s for s in self.samples if s.error]
        if errors:
            lines.append(f"  {len(errors)} sample(s) not measured (see --format json).")
        lines.append("Nothing was written; reindex without --estimate to run the sweep.")
        return lines


def _snapshot_claims(particles: list[Particle], snapshot_id: str) -> list[Particle]:
    """The ACTIVE claims a reindex of ``snapshot_id`` would supersede (``_reindex_snapshot``)."""
    return [p for p in particles if any(ref.snapshot_id == snapshot_id for ref in p.provenance)]


async def _measure_sample(session: AsyncSession, sp: SnapshotPlan, batch: int) -> SampleResult:
    """Re-extract one sampled snapshot without writing, and compare its claims."""
    result = SampleResult(
        entry_id=sp.entry_id, snapshot_id=sp.snapshot_id, source_bytes=sp.source_bytes
    )
    entry = await get_entry(session, sp.entry_id)
    snapshot = await get_snapshot(session, sp.snapshot_id)
    if entry is None or snapshot is None:
        return result.model_copy(update={"error": "entry or snapshot not found"})
    stored = _snapshot_claims(
        await get_active_particles_for_entry(session, sp.entry_id), sp.snapshot_id
    )
    extractor = select_extractor(entry.source_type)
    # The extractor reads the store only for the carry-forward
    # lookup; naming the stored claims as superseded makes it read every
    # chunk, as a reindex would.
    extracted = await extractor.extract(
        snapshot,
        load_blob(snapshot.content_hash),
        session=session,
        corpus_entry_id=entry.entry_id,
        source_type=entry.source_type,
        entry_uri_r=entry.uri_r,
        deposited_by=entry.deposited_by,
        supersede_ids=frozenset(p.id for p in stored),
    )
    if extracted.transient_error_count:
        return result.model_copy(
            update={"error": f"{extracted.transient_error_count} extraction call(s) failed"}
        )
    pairing = pair_claims([p.content for p in stored], [c.content for c in extracted.candidates])
    verdicts = await judge_claim_pairs(pairing.pairs, batch_size=batch) if pairing.pairs else []
    change = claim_change(pairing, verdicts)
    return result.model_copy(
        update={
            "stored_claims": len(stored),
            "new_claims": len(extracted.candidates),
            "exact": pairing.exact,
            "judged": len(pairing.pairs),
            "differing": change.differing,
            "changed": change.changed,
        }
    )


def _extraction_usd(usage: LLMUsage) -> float | None:
    """The list price of the usage's ``extraction``-purpose rows, ``None`` if any is unpriced."""
    total = 0.0
    for row in usage.rows:
        if row.purpose != "extraction":
            continue
        if row.cost_usd is None:
            return None
        total += row.cost_usd
    return total


async def run_reindex_estimate(session: AsyncSession, plan: EstimatePlan) -> ReindexEstimate:
    """Re-extract and judge the plan's sample, then project. Spends; writes nothing.

    The caller meters the run (``MeteredExtractRun``, ``route: reindex``).
    The sample's own usage is read from a nested scope so the projection can
    price re-extraction apart from judging. A sample whose re-extraction
    raises is recorded as unmeasured and the estimate goes on; an
    account-level LLM failure raises out, since every later sample would fail
    the same way.
    """
    batch = get_config().reindex.estimate_judge_batch_pairs
    samples: list[SampleResult] = []
    with track_usage() as accumulator:
        for sp in plan.sample:
            try:
                samples.append(await _measure_sample(session, sp, batch))
            except AccountLevelLLMError:
                raise
            except Exception as exc:
                log.warning("Estimate sample %s failed: %s", sp.snapshot_id, exc)
                samples.append(
                    SampleResult(
                        entry_id=sp.entry_id,
                        snapshot_id=sp.snapshot_id,
                        source_bytes=sp.source_bytes,
                        error=str(exc),
                    )
                )
    usage = accumulator.snapshot()
    measured = [s for s in samples if s.changed is not None]
    projection = project_sweep(
        outcomes=[
            (
                s.source_bytes,
                ClaimChange(differing=s.differing, total=s.stored_claims + s.new_claims),
            )
            for s in measured
        ],
        population=plan.population,
        sweep_bytes=sum(sp.source_bytes for sp in plan.plan.snapshot_plans),
        sample_extraction_usd=_extraction_usd(usage),
    )
    return ReindexEstimate(plan=plan, samples=samples, projection=projection, llm_usage=usage)
