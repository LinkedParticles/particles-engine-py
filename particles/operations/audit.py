"""First-run memory audit — the activation-moment census.

``run_memory_audit`` composes the **existing** machinery and adds no detection
of its own: the standard extract pipeline over the harvested snapshots, then
``collect_cards`` finder-normalization seam — **uncapped and
snooze-unfiltered** (an audit is a census, not a worklist) — leverage-ranked
exemplars per class, and ``get_quality_report`` header counts, assembled into
an :class:`AuditReport`.

The harvest itself (file walk, sentinel filter, transcript distillation) is
Surface-side — it reuses the helpers in
``particles/api/cli/_claude_code.py`` and lives with the ``particles audit``
verb (``particles/api/cli/audit.py``); this Engine module receives the
harvested entry ids. ``estimate_extraction`` is the §4 dry-run cost estimate
(byte counts × the extraction chunker), computed before anything touches the
store.

The renderer bakes the ADR's honesty stance into the copy (§5/§6, approved
verbatim 2026-07-11): hedged class labels ("potential", "likely-",
"probably-"), the uncalibrated-confidence footnote, disclosed skips, and a
next verb per class — the report is a door into the existing loops, not a
dead end.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    ExtractionStatus,
    JudgeVerdictKind,
    QualityReport,
    SuggestMode,
)
from particles.operations._llm import llm_circuit_open, llm_failure_count
from particles.operations.curation.cards import CardKind, CurationCard
from particles.operations.curation.collect import collect_cards
from particles.operations.curation.leverage import contested_ids_from, score_cards
from particles.operations.curation.session import _attach_particle_briefs
from particles.operations.lint import ContradictionProbeControl
from particles.operations.quality import get_quality_report
from particles.store.particle_store import get_particle_ids_for_entries

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Cost estimate — pure, computed before anything is deposited
# ---------------------------------------------------------------------------


class AuditEstimate(BaseModel):
    """Dry-run extraction-cost estimate from harvested byte counts.

    Mirrors the general extractor's chunking: one LLM call per source at or
    under ``extraction.html_chunk_size`` characters, else one call per chunk,
    bounded by ``extraction.max_llm_calls_per_source``. The contradiction probes are additional and density-dependent, so they are
    disclosed in prose rather than counted here.
    """

    entries: int = 0
    total_chars: int = 0
    estimated_llm_calls: int = 0
    estimated_tokens: int = 0


def estimate_extraction(char_counts: Sequence[int]) -> AuditEstimate:
    """Estimate extraction LLM calls for texts of the given character counts."""
    cfg = get_config().extraction
    calls = 0
    entries = 0
    total = 0
    for n in char_counts:
        if n <= 0:
            continue
        entries += 1
        total += n
        if n <= cfg.html_chunk_size:
            calls += 1
        else:
            calls += min(math.ceil(n / cfg.html_chunk_size), cfg.max_llm_calls_per_source)
    # ~4 chars per token is the standard rough conversion; the estimate is a
    # magnitude signal for the confirm gate, not a billing quote.
    return AuditEstimate(
        entries=entries,
        total_chars=total,
        estimated_llm_calls=calls,
        estimated_tokens=total // 4,
    )


def render_estimate(estimate: AuditEstimate) -> str:
    """One-line human rendering of the §4 estimate (always printed pre-extraction)."""
    cap = get_config().audit.max_contradiction_probes
    return (
        f"Estimate: {estimate.entries} entr{'y' if estimate.entries == 1 else 'ies'} to "
        f"extract → ~{estimate.estimated_llm_calls} LLM call(s), "
        f"~{estimate.estimated_tokens:,} tokens of source text, plus similarity-gated "
        f"contradiction probes (scales with near-duplicate density, not n²; capped at "
        f"{cap} — audit.max_contradiction_probes). "
        f"Already-captured content is skipped automatically."
    )


# ---------------------------------------------------------------------------
# The report model (SDK-internal, like SuggestReport / QualityReport)
# ---------------------------------------------------------------------------


class AuditBucket(BaseModel):
    """One card class in the census: the full count plus a few exemplars."""

    kind: CardKind
    count: int
    # Top ``audit.exemplars_per_class`` cards by leverage, briefs attached
    # so every exemplar carries its claim text.
    exemplars: list[CurationCard] = Field(default_factory=list)
    # for the CONTESTED class, how many cards each contested basis
    # fired on (a card firing two bases counts under both, so these need not
    # sum to ``count``). Empty for every other kind. The class composes three
    # incommensurable signals, so reporting one unattributed total would hide
    # which instrument produced the census — the same auditability
    # built into the badge itself.
    bases: dict[str, int] = Field(default_factory=dict)


class AuditReport(BaseModel):
    """The census output of ``run_memory_audit``."""

    generated_at: datetime = Field(default_factory=_utcnow)
    store: str = "default"
    # Harvest header. ``files_audited`` is None on a re-audit (no PATH).
    files_audited: int | None = None
    transcripts_audited: int = 0
    harvested_new: int = 0
    harvested_unchanged: int = 0
    extracted_snapshots: int = 0
    extraction_failures: int = 0
    # Store-level counts (the get_quality_report header).
    beliefs: int = 0
    subjects: int = 0
    snapshots_failed: int = 0
    # The census: complete per-kind counts + leverage-ranked exemplars.
    buckets: list[AuditBucket] = Field(default_factory=list)
    # ``--judge``: duplicate pairs carry LLM_JUDGE verdicts and the
    # label upgrades to "verified duplicates".
    judged: bool = False
    # §6/§7 disclosure flags: the contradiction probe did not run (no API key,
    # or the circuit breaker is open) — never a silent clean bill.
    semantic_skipped: bool = False
    semantic_skip_reason: str | None = None
    # Per-probe failures during the finding scan (refusals, transient errors):
    # each skips one candidate pair without tripping the breaker, so counts can
    # read low — disclosed, never silent (§6).
    semantic_probe_failures: int = 0
    # contradiction-probe census. ``scope`` is the
    # candidate-pair scope the probe ran under ("harvested" = at least one side
    # of every pair traces to this harvest's entries; "store" = store-wide);
    # None when the probe didn't run. When ``probes_run`` <
    # ``candidate_pairs``, the ``audit.max_contradiction_probes`` cap bound and
    # the report discloses "probed X of Y candidate pairs" (§6) — the
    # contradiction count is a lower bound, not a census. Under harvested
    # scope, ``intra_scope_pairs`` is the both-sides-in-scope subset of the
    # candidates — probed first, and named in the capped
    # disclosure's tier split ("N intra-harvest, M cross-store").
    contradiction_probe_scope: Literal["harvested", "store"] | None = None
    contradiction_candidate_pairs: int = 0
    contradiction_intra_scope_pairs: int = 0
    contradiction_probes_run: int = 0
    # duplicate-scan census — the DUPLICATE class analogue of the
    # contradiction fields above. ``duplicate_scope`` is "harvested" when the
    # headline / exemplars were filtered to pairs touching this harvest,
    # "store" when store-wide (re-audit / ``--scope store``).
    # ``duplicate_candidate_pairs_total`` is the store-wide candidate count M
    # (before scoping); when it exceeds the harvest-scoped headline count, the
    # report discloses the store-wide tail so the scan's full reach — e.g.
    # pre-existing store pollution — is never hidden behind the scoped count.
    duplicate_scope: Literal["harvested", "store"] | None = None
    duplicate_candidate_pairs_total: int = 0
    # The §4 estimate the run was gated on, when a harvest happened.
    estimate: AuditEstimate | None = None
    # ``--judge`` verdict split over ALL duplicate cards (not just exemplars):
    # verdict value → count. Empty unless ``judged``.
    duplicate_verdicts: dict[str, int] = Field(default_factory=dict)
    # Set by the CLI when the projection cycle re-rendered MEMORY.md
    # at the end of the run.
    projection_rendered: bool = False

    def count(self, kind: CardKind) -> int:
        """The census count for one card class (0 when absent)."""
        for bucket in self.buckets:
            if bucket.kind is kind:
                return bucket.count
        return 0

    def bucket(self, kind: CardKind) -> AuditBucket | None:
        """The bucket for one card class, or None when the class is empty."""
        for b in self.buckets:
            if b.kind is kind:
                return b
        return None

    def contested_split(self) -> tuple[int, int]:
        """(self-inconsistent, observer-contested) counts for the CONTESTED class.

        the first stays in the "potential contradictions" headline
        — an open INCONSISTENCY *is* the store contradicting itself. The second
        is every belief the badge fired on without that basis (a lens spread or
        a declared DISPUTES), which is not a contradiction and must not inflate
        the headline the wedge's trust claim rests on.
        """
        bucket = self.bucket(CardKind.CONTESTED)
        if bucket is None:
            return 0, 0
        inconsistency = bucket.bases.get("inconsistency", bucket.count)
        return inconsistency, max(0, bucket.count - inconsistency)


# ---------------------------------------------------------------------------
# Assembly (pure given cards + quality) — unit-testable without a store
# ---------------------------------------------------------------------------

# Stable bucket order: headline classes first, then the secondary kinds in the
# §5 "Also:" order.
_BUCKET_ORDER: tuple[CardKind, ...] = (
    CardKind.CONTRADICTION,
    CardKind.CONTESTED,
    CardKind.DUPLICATE_PAIR,
    CardKind.STALE,
    CardKind.RECENCY_DECAY,
    CardKind.CONFIDENCE_DECAY,
    CardKind.UNCITED_URL,
    CardKind.NO_SUBJECT,
    CardKind.RETRACTION_CASCADE,
    CardKind.BROKEN_PROVENANCE,
    CardKind.FAILED_SNAPSHOTS,
)


def build_buckets(cards: Sequence[CurationCard], exemplars_per_class: int) -> list[AuditBucket]:
    """Group scored cards by kind: full counts, top-N exemplars by leverage."""
    by_kind: dict[CardKind, list[CurationCard]] = {}
    for card in cards:
        by_kind.setdefault(card.kind, []).append(card)

    buckets: list[AuditBucket] = []
    kinds = [k for k in _BUCKET_ORDER if k in by_kind]
    kinds += [k for k in by_kind if k not in _BUCKET_ORDER]  # future kinds never vanish
    for kind in kinds:
        group = sorted(by_kind[kind], key=lambda c: (-c.leverage, c.key))
        bases: dict[str, int] = {}
        if kind is CardKind.CONTESTED:
            for card in group:
                for basis in card.contested_bases or ():
                    bases[basis] = bases.get(basis, 0) + 1
        buckets.append(
            AuditBucket(
                kind=kind,
                count=len(group),
                exemplars=group[:exemplars_per_class],
                bases=bases,
            )
        )
    return buckets


# ---------------------------------------------------------------------------
# Progress events (rendered by the CLI; the operation itself never prints)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditProgress:
    """One progress event from :func:`run_memory_audit`.

    The extraction phase can take many minutes (one LLM call per pending
    snapshot, plus subject resolution); without feedback the activation-moment
    audit is indistinguishable from a hang. The same goes for the contradiction probe on a populated store, so it emits per-pair ``probe``
    events (``done``/``total`` over the planned probe set
    — proposed). The Engine emits these events and the Surface renders them —
    the operation itself never prints (AGENTS.md § Code conventions).
    """

    phase: Literal["extract", "census", "probe"]
    done: int
    total: int
    label: str
    #: Beliefs written for this unit (``extract`` phase, successful units only).
    particles: int | None = None
    failed: bool = False


ProgressCallback = Callable[[AuditProgress], None]


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


async def run_memory_audit(
    session: AsyncSession,
    *,
    store: str = "default",
    files_audited: int | None = None,
    transcripts_audited: int = 0,
    harvested_new: int = 0,
    harvested_unchanged: int = 0,
    harvested_entry_ids: Sequence[str] | None = None,
    semantic: bool = True,
    judge: bool = False,
    semantic_skip_reason: str | None = None,
    estimate: AuditEstimate | None = None,
    agent_id: str = "audit",
    on_progress: ProgressCallback | None = None,
    contradiction_scope: Literal["harvested", "store"] = "store",
) -> AuditReport:
    """Extract the harvested entries, census the findings, assemble the report.

    Composes existing pieces only: the standard extract pipeline
    over PENDING snapshots of ``harvested_entry_ids`` (COMPLETE snapshots skip
    — idempotence against the harvest is structural), then the
    ``collect_cards(semantic=...)`` **uncapped and snooze-unfiltered**,
    leverage scoring to rank exemplars *within* each class, brief attachment
    , and ``get_quality_report`` for the header.

    ``semantic`` gates the contradiction probe; ``judge`` runs the
    duplicate finder in ``LLM_JUDGE`` mode (default ``REPORT`` — unjudged
    similarity candidates). Commits after each extracted snapshot (a later
    failure must not roll back completed extraction work); otherwise the
    caller owns the transaction.

    ``contradiction_scope`` bounds the probe's candidate
    pairs: ``"harvested"`` keeps only pairs where at least one side's particle
    traces to ``harvested_entry_ids`` (the CLI's default on a harvest run);
    ``"store"`` is the store-wide set (the re-audit default).
    ``"harvested"`` with no harvested entries probes nothing — the caller
    should not ask for it. The ``audit.max_contradiction_probes`` cap applies
    in both scopes; the census fields on the report disclose what was probed.
    Under ``"harvested"``, both the contradiction probe and the ``--judge``
    duplicate pass consume their candidates in two tiers:
    intra-harvest pairs (both sides in scope) first, mixed pairs second,
    highest similarity within each — so a binding cap never starves the
    harvest's own pairs behind coincidental cross-store neighbours.

    Each run also records one ``CONSOLIDATION_RUN`` operator event
    (``actor: audit``) so the interactive audit contributes to —
    and its census deltas are readable from — the same delta chain the
    scheduled consolidation cycle keys off.
    """
    started_at = _utcnow()
    # 1. Extract — PENDING snapshots scoped to this harvest's entries.
    extracted = 0
    failures = 0
    if harvested_entry_ids:
        # Deferred import: the pipeline pulls the extractor registry / LLM
        # stack; load it only when there is something to extract (AGENTS.md
        # deferred-import case 2).
        from particles.corpus.store import get_entry, list_snapshots_for_entry
        from particles.operations.extract import extract_snapshot

        pending: list[tuple[str, str, str]] = []  # (entry_id, snapshot_id, label)
        for entry_id in dict.fromkeys(harvested_entry_ids):  # de-dupe, keep order
            label: str | None = None
            for snap in await list_snapshots_for_entry(session, entry_id):
                if snap.extraction_status is not ExtractionStatus.PENDING:
                    continue
                if label is None:
                    entry = await get_entry(session, entry_id)
                    # Human handle: the URI tail (filename / session id), never
                    # the opaque entry uuid.
                    uri = entry.uri_r if entry is not None else None
                    label = uri.rstrip("/").rsplit("/", 1)[-1] if uri else entry_id[:8]
                pending.append((entry_id, snap.snapshot_id, label))

        for index, (entry_id, snapshot_id, label) in enumerate(pending, start=1):
            try:
                written = await extract_snapshot(session, entry_id, snapshot_id, agent_id=agent_id)
                await session.commit()
                extracted += 1
                if on_progress is not None:
                    on_progress(
                        AuditProgress(
                            phase="extract",
                            done=index,
                            total=len(pending),
                            label=label,
                            particles=len(written),
                        )
                    )
            except Exception as exc:  # noqa: BLE001 — census the rest; disclose the failure
                await session.rollback()
                failures += 1
                log.warning(
                    "audit: extraction failed for %s/%s: %s",
                    entry_id[:8],
                    snapshot_id[:8],
                    exc,
                )
                if on_progress is not None:
                    on_progress(
                        AuditProgress(
                            phase="extract",
                            done=index,
                            total=len(pending),
                            label=label,
                            failed=True,
                        )
                    )

    # 2–3. Collect findings through the shared finder-normalization seam.
    duplicate_mode = SuggestMode.LLM_JUDGE if (judge and semantic) else SuggestMode.REPORT
    if on_progress is not None:
        census_label = (
            "contradiction probe + duplicate scan"
            if semantic and not llm_circuit_open()
            else "structural checks + duplicate scan"
        )
        on_progress(AuditProgress(phase="census", done=0, total=1, label=census_label))

    # Harvested-scope particle-id set: pairs where at least one side traces to
    # this harvest's entries. Shared by the contradiction probe and the
    # duplicate scan (computed once, independent of ``semantic`` since
    # the duplicate scan runs even when semantic is off). ``None`` ⇒ store-wide
    # (the re-audit / ``--scope store`` path).
    harvested_scope_ids: frozenset[str] | None = None
    if contradiction_scope == "harvested":
        harvested_scope_ids = frozenset(
            await get_particle_ids_for_entries(session, list(harvested_entry_ids or []))
        )

    # Bound the probe: the audit — unlike lint —
    # is cost-gated, so it always caps, optionally scopes to this harvest's
    # beliefs, and streams per-pair progress. The control carries the
    # candidate-pair census back out for the §6 "probed X of Y" disclosure.
    probe_control: ContradictionProbeControl | None = None
    if semantic:

        def _probe_progress(done: int, total: int) -> None:
            if on_progress is not None:
                on_progress(
                    AuditProgress(
                        phase="probe", done=done, total=total, label="contradiction probe"
                    )
                )

        probe_control = ContradictionProbeControl(
            max_probes=get_config().audit.max_contradiction_probes,
            scope_particle_ids=harvested_scope_ids,
            on_progress=_probe_progress if on_progress is not None else None,
        )

    failures_before = llm_failure_count()
    cards = await collect_cards(
        session,
        semantic=semantic,
        duplicate_mode=duplicate_mode,
        contradiction_probe=probe_control,
        duplicate_scope_ids=harvested_scope_ids,
    )
    probe_failures = llm_failure_count() - failures_before
    # read the composed-badge set off the full collection, before
    # the duplicate harvest-scoping below narrows it.
    contested_ids = contested_ids_from(cards)

    # harvest-scope the duplicate headline/exemplars. The scan ran
    # store-wide (REPORT enumeration is pure cosine); keep the store-wide total
    # M for the tail disclosure, then drop DUPLICATE_PAIR cards that don't touch
    # this harvest so the headline / exemplars / verdict census describe the
    # harvest — never store-wide pollution mislabelled as a harvest finding.
    duplicate_total = sum(1 for c in cards if c.kind is CardKind.DUPLICATE_PAIR)
    if harvested_scope_ids is not None:
        cards = [
            c
            for c in cards
            if c.kind is not CardKind.DUPLICATE_PAIR
            or any(pid in harvested_scope_ids for pid in c.particle_ids)
        ]

    await score_cards(session, cards, contested_ids=contested_ids)

    # 4. Assemble: full per-kind counts, top exemplars with claim text.
    exemplars_per_class = get_config().audit.exemplars_per_class
    buckets = build_buckets(cards, exemplars_per_class)
    await _attach_particle_briefs(
        session, [card for bucket in buckets for card in bucket.exemplars]
    )
    quality: QualityReport = await get_quality_report(session)

    skipped = (not semantic) or llm_circuit_open()
    reason = semantic_skip_reason
    if skipped and reason is None:
        reason = "LLM unavailable" if llm_circuit_open() else "semantic checks disabled"

    verdict_counts: dict[str, int] = {}
    if judge and semantic:
        for card in cards:
            if card.kind is not CardKind.DUPLICATE_PAIR:
                continue
            value = card.verdict.verdict.value if card.verdict else JudgeVerdictKind.UNSURE.value
            verdict_counts[value] = verdict_counts.get(value, 0) + 1

    report = AuditReport(
        store=store,
        files_audited=files_audited,
        transcripts_audited=transcripts_audited,
        harvested_new=harvested_new,
        harvested_unchanged=harvested_unchanged,
        extracted_snapshots=extracted,
        extraction_failures=failures,
        beliefs=quality.active_particles,
        subjects=quality.total_subjects,
        snapshots_failed=quality.snapshots_failed,
        buckets=buckets,
        judged=judge and semantic,
        duplicate_verdicts=verdict_counts,
        semantic_skipped=skipped,
        semantic_skip_reason=reason if skipped else None,
        semantic_probe_failures=probe_failures,
        contradiction_probe_scope=contradiction_scope if probe_control is not None else None,
        contradiction_candidate_pairs=(
            probe_control.candidate_pairs if probe_control is not None else 0
        ),
        contradiction_intra_scope_pairs=(
            probe_control.intra_scope_pairs if probe_control is not None else 0
        ),
        contradiction_probes_run=probe_control.probes_run if probe_control is not None else 0,
        duplicate_scope="harvested" if harvested_scope_ids is not None else "store",
        duplicate_candidate_pairs_total=duplicate_total,
        estimate=estimate,
    )

    # the run-record fold. Deferred import — the
    # consolidation module pulls the reconcile/ingest stack this census path
    # otherwise never loads (AGENTS.md deferred-import case 2).
    from particles.operations.consolidation import record_audit_run

    await record_audit_run(session, report, started_at=started_at, actor=agent_id)
    return report


# ---------------------------------------------------------------------------
# The renderer (§6) — one renderer for terminal and --output
# ---------------------------------------------------------------------------

# Per-class next verbs (§5): every class ends with its door into an existing loop.
_NEXT_VERBS: dict[str, str] = {
    "contradictions": "next: particles review · particles curate --kind contradiction",
    "duplicates": "next: particles links suggest --judge · particles curate --kind duplicate_pair",
    "stale": "next: particles curate --kind stale",
    # a divergence- or stance-only card has no INCONSISTENCY for
    # `review` to resolve, so its door is the drill-down that shows the reading.
    "contested": ("next: particles query --contestedness · particles curate --kind contested"),
}

_STALE_KINDS = (CardKind.STALE, CardKind.RECENCY_DECAY, CardKind.CONFIDENCE_DECAY)


def _short(pid: str) -> str:
    return f"{pid[:8]}…"


def _exemplar_lines(card: CurationCard) -> list[str]:
    """Render one exemplar: claim text, partner claim for pairs, short id."""
    lines: list[str] = []
    briefs = {b.particle_id: b for b in card.particles}
    if card.particle_ids:
        first = card.particle_ids[0]
        content = briefs[first].content if first in briefs else "(claim text unavailable)"
        lines.append(f'  • "{content}"  [{_short(first)}]')
        for partner in card.particle_ids[1:]:
            partner_content = (
                briefs[partner].content if partner in briefs else "(claim text unavailable)"
            )
            lines.append(f'    ↔ "{partner_content}"  [{_short(partner)}]')
    elif card.corpus_url is not None:
        lines.append(f"  • {card.corpus_url}")
    if card.diagnostic:
        lines.append(f"    — {card.diagnostic}")
    return lines


def _judged_duplicate_split(report: AuditReport) -> tuple[int, int, int]:
    """(paraphrase, distinct, unsure) counts from the report's verdict census."""
    counts = report.duplicate_verdicts
    paraphrase = counts.get(JudgeVerdictKind.PARAPHRASE.value, 0)
    distinct = counts.get(JudgeVerdictKind.DISTINCT.value, 0)
    unsure = report.count(CardKind.DUPLICATE_PAIR) - paraphrase - distinct
    return paraphrase, distinct, max(0, unsure)


def render_audit_report(report: AuditReport) -> str:
    """Render the census in the §5 approved shape — terminal and Markdown share it."""
    lines: list[str] = []

    # --- Header -----------------------------------------------------------
    if report.files_audited is not None:
        sources = f"{report.files_audited} memory file{'s' if report.files_audited != 1 else ''}"
        if report.transcripts_audited:
            sources += (
                f" + {report.transcripts_audited} "
                f"transcript{'s' if report.transcripts_audited != 1 else ''}"
            )
        lines.append(
            f"Audited {sources} → {report.beliefs} beliefs about {report.subjects} subjects."
        )
    else:
        lines.append(
            f"Re-audited store '{report.store}' → {report.beliefs} beliefs "
            f"about {report.subjects} subjects."
        )
    lines.append("")

    # --- Headline classes (always shown, zero or not; §5) ------------------
    contradiction_n = report.count(CardKind.CONTRADICTION)
    # only the inconsistency basis belongs in this headline.
    contested_n, observer_contested_n = report.contested_split()
    dup_n = report.count(CardKind.DUPLICATE_PAIR)
    expired_n = report.count(CardKind.STALE)
    aged_n = report.count(CardKind.RECENCY_DECAY) + report.count(CardKind.CONFIDENCE_DECAY)

    headline: list[tuple[str, str]] = []
    headline.append(
        (
            f"{contradiction_n + contested_n} potential contradictions",
            f"({contradiction_n} cross-file, {contested_n} contested at extract time)",
        )
    )
    if report.judged:
        paraphrase, distinct, unsure = _judged_duplicate_split(report)
        headline.append(
            (
                f"{dup_n} verified duplicate belief pairs",
                f"(LLM-judged: {paraphrase} paraphrase, {distinct} distinct, {unsure} unsure)",
            )
        )
    else:
        headline.append(
            (
                f"{dup_n} likely-duplicate belief pairs",
                "(unjudged similarity candidates; --judge to verify)",
            )
        )
    headline.append(
        (
            f"{expired_n + aged_n} probably-stale facts",
            f"({aged_n} aged past their source's decay horizon, {expired_n} expired)",
        )
    )
    width = max(len(label) for label, _ in headline)
    for label, detail in headline:
        lines.append(f"  {label.ljust(width)}  {detail}")

    if report.semantic_skipped:
        reason = report.semantic_skip_reason or "LLM unavailable"
        lines.append(f"  contradiction check skipped: {reason}")
    if report.semantic_probe_failures:
        n = report.semantic_probe_failures
        noun = "probe" if n == 1 else "probes"
        lines.append(
            f"  {n} semantic {noun} failed or were declined by the model and were "
            f"skipped — contradiction and duplicate counts may read low"
        )
    # Proposed disclosures: a scoped or capped probe is a lower
    # bound, never a silent partial census (§6).
    if report.contradiction_probe_scope == "harvested":
        lines.append(
            "  contradiction probe scoped to this harvest's beliefs (each probed "
            "pair touches at least one; intra-harvest pairs probed first) — "
            "pass --scope store to probe the whole store"
        )
    if (
        not report.semantic_skipped
        and report.contradiction_probes_run < report.contradiction_candidate_pairs
    ):
        cap = get_config().audit.max_contradiction_probes
        # under harvested scope, name the tier split so the operator
        # can see whether the cap ever reached the cross-store tier.
        split = ""
        if report.contradiction_probe_scope == "harvested":
            intra = report.contradiction_intra_scope_pairs
            cross = report.contradiction_candidate_pairs - intra
            split = f"{intra} intra-harvest, {cross} cross-store; "
        lines.append(
            f"  contradiction probe capped: probed {report.contradiction_probes_run} of "
            f"{report.contradiction_candidate_pairs} candidate pairs "
            f"({split}audit.max_contradiction_probes = {cap}) — the contradiction count "
            f"may read low"
        )
    # a harvest-scoped duplicate headline discloses the store-wide
    # total so the scan's full reach (e.g. pre-existing store pollution) is
    # never hidden behind the harvest-scoped count.
    if (
        report.duplicate_scope == "harvested"
        and report.duplicate_candidate_pairs_total > report.count(CardKind.DUPLICATE_PAIR)
    ):
        harvested_dups = report.count(CardKind.DUPLICATE_PAIR)
        store_dups = report.duplicate_candidate_pairs_total
        lines.append(
            f"  duplicate scan is store-wide; {store_dups} candidate pairs total, "
            f"{harvested_dups} involve this harvest — pass --scope store to surface "
            f"all {store_dups}"
        )

    # --- Secondary line (only nonzero kinds; §5) ----------------------------
    also: list[str] = []
    uncited_n = report.count(CardKind.UNCITED_URL)
    no_subject_n = report.count(CardKind.NO_SUBJECT)
    cascade_n = report.count(CardKind.RETRACTION_CASCADE)
    provenance_n = report.count(CardKind.BROKEN_PROVENANCE)
    if uncited_n:
        also.append(f"{uncited_n} cited sources never captured")
    if no_subject_n:
        also.append(f"{no_subject_n} beliefs have no resolvable subject")
    if cascade_n:
        also.append(f"{cascade_n} beliefs depend on a retracted belief")
    if provenance_n:
        also.append(f"{provenance_n} beliefs cite a missing corpus entry")
    if report.snapshots_failed:
        also.append(f"{report.snapshots_failed} snapshots failed extraction")
    if observer_contested_n:
        contested_bucket = report.bucket(CardKind.CONTESTED)
        counted = contested_bucket.bases if contested_bucket is not None else {}
        by_basis = ", ".join(
            f"{counted[b]} {b}" for b in ("stance", "divergence") if counted.get(b)
        )
        also.append(f"{observer_contested_n} beliefs contested by observer signal ({by_basis})")
    if also:
        lines.append("")
        lines.append("  Also: " + " · ".join(also))
        if uncited_n:
            lines.append("  next: particles deposit <url> · particles curate --kind uncited_url")

    # --- Exemplars per headline class (§5) ----------------------------------
    def _class_block(
        title: str,
        kinds: Sequence[CardKind],
        verbs: str,
        where: Callable[[CurationCard], bool] | None = None,
    ) -> None:
        exemplars: list[CurationCard] = []
        for kind in kinds:
            bucket = report.bucket(kind)
            if bucket is not None:
                exemplars.extend(c for c in bucket.exemplars if where is None or where(c))
        if not exemplars:
            return
        exemplars.sort(key=lambda c: (-c.leverage, c.key))
        limit = get_config().audit.exemplars_per_class
        lines.append("")
        lines.append(title)
        for card in exemplars[:limit]:
            lines.extend(_exemplar_lines(card))
        lines.append(f"  {verbs}")

    dup_title = "Verified duplicates" if report.judged else "Likely-duplicate belief pairs"
    _class_block(
        "Potential contradictions",
        (CardKind.CONTRADICTION, CardKind.CONTESTED),
        _NEXT_VERBS["contradictions"],
        # a contested card belongs here only via its inconsistency
        # basis; the observer-signal ones get their own block below.
        where=lambda c: (
            c.kind is not CardKind.CONTESTED
            or "inconsistency" in (c.contested_bases or ["inconsistency"])
        ),
    )
    if observer_contested_n:
        _class_block(
            "Contested by observer signal",
            (CardKind.CONTESTED,),
            _NEXT_VERBS["contested"],
            where=lambda c: "inconsistency" not in (c.contested_bases or ["inconsistency"]),
        )
    _class_block(dup_title, (CardKind.DUPLICATE_PAIR,), _NEXT_VERBS["duplicates"])
    _class_block("Probably-stale facts", _STALE_KINDS, _NEXT_VERBS["stale"])

    # --- Disclosures + footer (§5/§6) ---------------------------------------
    if report.extraction_failures:
        lines.append("")
        lines.append(
            f"  {report.extraction_failures} snapshot(s) failed to extract during this "
            f"audit — run `particles reindex` to retry them."
        )
    lines.append("")
    if report.beliefs:
        lines.append(
            "note: confidence on this content is self-reported and capped, "
            "not benchmark-calibrated."
        )
    lines.append("Run 'particles curate' to work these down a few at a time.")
    if report.projection_rendered:
        lines.append(
            "MEMORY.md was re-projected from the audited store — its memory-index "
            "region now reflects these beliefs."
        )
    return "\n".join(lines) + "\n"
