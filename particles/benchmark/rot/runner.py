# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-rot benchmark runner (§5): deposit in time order, probe at checkpoints.

Per seed, one ephemeral scratch store (never a user store).
Sessions are deposited **in timeline order** and extracted through the
standard ``extract_snapshot`` + §6.6 path; at each checkpoint every slot is
probed through ``retrieve_ranked`` with a default ``QueryRequest`` — the
selection half the ``query`` op uses, sharing its filter, scoring, and
co-evidential collapse by construction. Default config throughout; the only
store-side setup is the ``source`` channel's domain trust rule,
which is ordinary operator policy (skipped under ``trust_policy=False``).

Three perception arms (§3): ``oracle`` scripts extraction *and* the §6.6
contradiction probe (zero LLM calls, deterministic); ``probe`` scripts
extraction and pays only for the live probe; ``live`` is the product. The
scripted arms route every purpose they must not pay for to a refusing
provider through :func:`particles.llm.override_providers`, so the zero-spend
property holds by construction and any unanticipated call is disclosed.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import pickle
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, date, datetime, time, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.benchmark.memory.runner import (
    SameModelViolation,
    decay_quality_note,
    isolated_blob_dir,
    scratch_store,
)
from particles.benchmark.rot.generator import (
    GENERATOR_VERSION,
    UNTRUSTED_DOMAIN,
    generate_world,
    render_session,
    slot_state,
)
from particles.benchmark.rot.oracle import (
    OracleExtractor,
    OracleProbeProvider,
    RefusingProvider,
    oracle_claims,
)
from particles.benchmark.rot.schema import (
    HitClass,
    HitRecord,
    ProbeResult,
    RotBenchmarkReport,
    RotRunSelection,
    RotSession,
    RotWorld,
    SessionKind,
    SlotState,
    WorldResult,
)
from particles.benchmark.rot.scoring import (
    SCORER_VERSION,
    breakdowns,
    classify,
    first_value_bearing,
    floor_sweep,
    merge_metrics,
    metrics_for,
)
from particles.config import get_config
from particles.core.schema import Mutability, QueryRequest, SourceType
from particles.corpus.deposit import deposit_text_versioned
from particles.embeddings import get_embedding_model, get_embedding_model_id
from particles.extraction.general import ExtractionResult
from particles.extraction.registry import ExtractorPlugin, select_extractor
from particles.ingest.pipeline import extract_snapshot
from particles.llm import CompletionProvider, LLMPurpose, get_provider, override_providers
from particles.operations.query.contested import compute_contested_badges
from particles.operations.query.main import retrieve_ranked
from particles.store.particle_store import ParticleRow
from particles.store.trust_store import upsert_trust_rule

log = logging.getLogger(__name__)

#: The perception arms.
ARMS: tuple[str, ...] = ("oracle", "probe", "live")

#: Extraction id recorded for the scripted arms.
SCRIPTED_EXTRACTION_ID = "rot-oracle:scripted-extractor"

TOOL_TURN_NOTE = (
    "The tool_turn poison channel is measured on the raw-transcript deposit "
    "path (deposit_text / MCP deposit_text, which do not distill). The "
    "Claude Code harvester drops tool results, so on the wedge path that "
    "channel is structurally zero."
)


class ExtractionCache:
    """A disk-persisted candidate cache for the paid arms.

    A `live` run's bill is extraction; a candidacy or ladder change leaves
    extraction untouched but re-pays it (~US$12 per three worlds). Candidate
    production is Client-layer and store-free, so a session's
    candidates depend only on its bytes and the extractor that read them: the
    key is ``sha256(text)`` plus the extractor id/version, the resolved
    extraction model, and the SDK version, so a prompt change (which ships
    with a version bump) or a model change is a miss rather than a silent
    replay. Results are deep-copied on both store and replay — the pipeline
    mutates candidates in place — and only clean results are kept.

    Pickle is the format: a ``CandidateParticle`` is a dataclass of enums and
    pydantic models, and this cache is a benchmark-local, single-machine
    artifact, never an interchange surface. ``--no-cache`` skips it.

    **One file per stamp, and a flush that merges.** Both halves are repairs
    for the same destroyed artifact: a run under a *different* stamp — the
    scripted arms resolve their extraction model to ``rot-refused:extraction``
    — read nothing, then wrote its own empty cache over a paid run's 549
    entries, and a diagnostic that was meant to cost nothing cost the whole
    US$12 re-extraction. A stamp now names its own file, so an arm can never
    reach another arm's, and a flush folds into whatever is on disk rather
    than replacing it, so a partial or failed run cannot erase a complete one
    and the concurrent worlds of one run no longer overwrite each other.
    """

    def __init__(self, path: Path, *, extractor: ExtractorPlugin, model: str) -> None:
        self._stamp = "|".join(
            (
                extractor.EXTRACTOR_ID,
                extractor.EXTRACTOR_VERSION,
                model,
                version("linkedparticles"),
            )
        )
        digest = hashlib.sha256(self._stamp.encode("utf-8")).hexdigest()[:12]
        self.path = path.with_name(f"{path.stem}-{digest}{path.suffix}")
        self._mem: dict[str, ExtractionResult] = {}
        self.hits = 0
        self.misses = 0
        # The pre-1.146.1 single-file layout is still read when it happens to
        # carry this stamp, so a cache written before the split is not orphaned.
        for candidate in (self.path, path):
            self._mem = self._load(candidate)
            if self._mem:
                break

    def _load(self, path: Path) -> dict[str, ExtractionResult]:
        """Entries on disk at ``path`` under this stamp, or an empty dict."""
        if not path.exists():
            return {}
        with contextlib.suppress(Exception):
            loaded = pickle.loads(path.read_bytes())
            if isinstance(loaded, dict) and loaded.get("stamp") == self._stamp:
                entries: dict[str, ExtractionResult] = loaded["entries"]
                return entries
        return {}

    def key(self, text: str) -> str:
        """Cache key for one session's deposited text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> ExtractionResult | None:
        """A cached result for ``text``, deep-copied, or ``None``."""
        hit = self._mem.get(self.key(text))
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return copy.deepcopy(hit)

    def put(self, text: str, result: ExtractionResult) -> None:
        """Keep a clean result for ``text``."""
        if result.transient_error_count:
            return
        self._mem[self.key(text)] = copy.deepcopy(result)

    def flush(self) -> None:
        """Fold this world's entries into the stamp's cache file.

        Never a replacement: the three worlds of one run hold separate
        instances over one path, and a run that extracted nothing must leave a
        complete cache alone.
        """
        if not self._mem:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._load(self.path) | self._mem
        self.path.write_bytes(pickle.dumps({"stamp": self._stamp, "entries": merged}))


class _CachedExtractor:
    """``extractor`` with :class:`ExtractionCache` in front of it."""

    def __init__(self, inner: ExtractorPlugin, cache: ExtractionCache) -> None:
        self._inner = inner
        self._cache = cache
        self.EXTRACTOR_ID = inner.EXTRACTOR_ID
        self.EXTRACTOR_VERSION = inner.EXTRACTOR_VERSION

    def accepts(self, source_type: str) -> bool:
        """Delegate to the wrapped extractor."""
        return self._inner.accepts(source_type)

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        """Replay a cached result for this content, or extract and keep it."""
        text = content.decode("utf-8", errors="replace")
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        result = await self._inner.extract(snapshot, content, **kwargs)
        self._cache.put(text, result)
        return result


class RotArmError(ValueError):
    """An unknown arm, or a run that cannot measure anything (no encoder)."""


# ---------------------------------------------------------------------------
# Estimate / confirm gate (shape)
# ---------------------------------------------------------------------------


class RotRunEstimate(BaseModel):
    """Projected LLM spend of a run, computed before any call is made."""

    arm: str
    worlds: int = 0
    sessions: int = 0
    extraction_calls: int = 0
    extraction_input_tokens: int = 0
    extraction_output_tokens: int = 0
    probe_calls: int = 0
    probe_input_tokens: int = 0
    probe_output_tokens: int = 0
    extraction_model: str = ""
    probe_model: str = ""
    #: ``None`` when any priced component's model has no
    #: ``benchmark_memory.price_per_mtok`` entry — never a partial total.
    cost_usd: float | None = None
    assumptions: list[str] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        """Total projected LLM calls."""
        return self.extraction_calls + self.probe_calls


def _price(purpose: LLMPurpose) -> tuple[str, float, float] | None:
    """``(model, input $/MTok, output $/MTok)`` for a purpose, or ``None`` if unpriced."""
    sel = get_config().llm.for_purpose(purpose)
    prices = get_config().benchmark_memory.price_per_mtok
    for key in (f"{sel.provider}:{sel.model}", sel.model):
        if key in prices:
            return sel.model, prices[key].input, prices[key].output
    return None


def _chars_per_token(purpose: LLMPurpose) -> float:
    sel = get_config().llm.for_purpose(purpose)
    cfg = get_config().benchmark_memory
    for key in (f"{sel.provider}:{sel.model}", sel.model):
        if key in cfg.chars_per_token_by_model:
            return cfg.chars_per_token_by_model[key]
    return cfg.chars_per_token


def estimate_rot_run(
    arm: str,
    *,
    seeds: list[int],
    days: int,
    checkpoints: list[int],
) -> RotRunEstimate:
    """Project calls, tokens, and — when every model is priced — dollars.

    ``oracle`` projects zero. ``probe`` pays only the contradiction probes;
    ``live`` adds one extraction call per session (a rot session is far below
    the chunking threshold). Probe count is bounded from the world's
    value-bearing events × ``benchmark_rot.estimate_probe_calls_per_update``;
    every assumption is listed on the estimate.
    """
    if arm not in ARMS:
        raise RotArmError(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    cfg = get_config().benchmark_rot
    llm = get_config().llm
    est = RotRunEstimate(
        arm=arm,
        worlds=len(seeds),
        extraction_model=llm.for_purpose("extraction").model if arm == "live" else "scripted",
        probe_model=llm.for_purpose("semantic_lint").model if arm != "oracle" else "scripted",
    )
    session_chars = 0
    events = 0
    for seed in seeds:
        world = generate_world(seed, days=days, checkpoints=checkpoints)
        est.sessions += len(world.sessions)
        events += len(world.events)
        session_chars += sum(len(render_session(s, "2026-01-01")) for s in world.sessions)
    if arm == "live":
        est.extraction_calls = est.sessions
        est.extraction_input_tokens = int(
            session_chars / _chars_per_token("extraction")
            + est.extraction_calls * cfg.estimate_extraction_prompt_overhead_tokens
        )
        est.extraction_output_tokens = (
            est.extraction_calls * cfg.estimate_output_tokens_per_extraction_call
        )
        est.assumptions.append(
            f"{cfg.estimate_output_tokens_per_extraction_call} output + "
            f"{cfg.estimate_extraction_prompt_overhead_tokens} prompt-overhead tokens "
            f"per extraction call (benchmark_rot.estimate_*)"
        )
    if arm in ("probe", "live"):
        est.probe_calls = int(events * cfg.estimate_probe_calls_per_update)
        est.probe_input_tokens = est.probe_calls * cfg.estimate_probe_input_tokens
        est.probe_output_tokens = est.probe_calls * cfg.estimate_probe_output_tokens
        est.assumptions.append(
            f"{cfg.estimate_probe_calls_per_update} probe calls per world event "
            f"({events} events), {cfg.estimate_probe_input_tokens} in / "
            f"{cfg.estimate_probe_output_tokens} out tokens each — an upper-leaning "
            f"bound; the real count is the candidates over the similarity threshold"
        )
    total = 0.0
    priced = True
    if est.extraction_calls:
        p = _price("extraction")
        if p is None:
            priced = False
        else:
            total += (
                est.extraction_input_tokens * p[1] + est.extraction_output_tokens * p[2]
            ) / 1e6
    if est.probe_calls:
        p = _price("semantic_lint")
        if p is None:
            priced = False
        else:
            total += (est.probe_input_tokens * p[1] + est.probe_output_tokens * p[2]) / 1e6
    est.cost_usd = total if priced else None
    return est


def render_estimate(est: RotRunEstimate) -> str:
    """The estimate as printed before any call."""
    lines = [
        f"Rot benchmark estimate — arm {est.arm}: {est.worlds} world(s), {est.sessions} sessions",
        f"  extraction: {est.extraction_calls} call(s) [{est.extraction_model}], "
        f"~{est.extraction_input_tokens:,} in / ~{est.extraction_output_tokens:,} out tokens",
        f"  §6.6 probe: {est.probe_calls} call(s) [{est.probe_model}], "
        f"~{est.probe_input_tokens:,} in / ~{est.probe_output_tokens:,} out tokens",
    ]
    if est.llm_calls == 0:
        lines.append("  cost: US$0.00 — no LLM call is made (scripted perception)")
    elif est.cost_usd is None:
        lines.append(
            "  cost: no price configured for every model (benchmark_memory.price_per_mtok)"
        )
    else:
        lines.append(f"  cost: ~US${est.cost_usd:,.2f}")
    lines += [f"  assumption: {a}" for a in est.assumptions]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider routing per arm
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _arm_routing(
    arm: str, world: RotWorld, refused: dict[str, int]
) -> Iterator[OracleProbeProvider | None]:
    """Install the arm's provider overrides for one world."""
    if arm == "live":
        yield None
        return
    overrides: dict[LLMPurpose, CompletionProvider] = {}
    for purpose in get_args(LLMPurpose):
        overrides[purpose] = RefusingProvider(purpose, refused)
    probe: OracleProbeProvider | None = None
    if arm == "oracle":
        probe = OracleProbeProvider(world)
        overrides["semantic_lint"] = probe
    else:  # probe arm: the live contradiction probe is the one paid purpose
        del overrides["semantic_lint"]
    with override_providers(overrides):
        yield probe


def _resolved_models(arm: str) -> tuple[str, str]:
    """``(extraction id, semantic_lint id)`` as the arm resolves them right now."""
    extraction = (
        get_provider("extraction").provider_model if arm == "live" else SCRIPTED_EXTRACTION_ID
    )
    return extraction, get_provider("semantic_lint").provider_model


def _thresholds() -> dict[str, float]:
    """Every read- and write-side knob the result depends on (the run tuple)."""
    cfg = get_config()
    return {
        "extraction.similarity_threshold": cfg.extraction.similarity_threshold,
        "extraction.duplicate_suppression.enabled": float(
            cfg.extraction.duplicate_suppression.enabled
        ),
        "trust.differential_threshold": cfg.trust.differential_threshold,
        "reconciliation.store_mode.single": float(cfg.reconciliation.store_mode == "single"),
        "document_supersession.enabled": float(cfg.document_supersession.enabled),
        "query.similarity_weight": cfg.query.similarity_weight,
        "query.confidence_weight": cfg.query.confidence_weight,
        "query.relevance_floor": cfg.query.relevance_floor,
        "confidence.uncalibrated_cap.enabled": float(cfg.confidence.uncalibrated_cap.enabled),
        "contestedness.badge_enabled": float(cfg.contestedness.badge_enabled),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_rot_benchmark(  # noqa: PLR0913 — the run tuple is the API
    *,
    arm: str,
    seeds: list[int] | None = None,
    days: int | None = None,
    checkpoints: list[int] | None = None,
    top_k: int | None = None,
    trust_policy: bool = True,
    work_dir: Path | None = None,
    keep_stores: bool = False,
    run_date: date | None = None,
    cache_dir: Path | None = None,
    attribute: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RotBenchmarkReport:
    """Run the memory-rot benchmark and return the report of record.

    ``cache_dir`` persists the paid arms' extraction results, so
    a re-run that changes only candidacy or the ladder pays probes alone.
    ``attribute`` stamps one author id on every session, which is what the
    attribution rule needs to fire in a ``multi`` store.
    """
    if arm not in ARMS:
        raise RotArmError(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    if get_embedding_model() is None:
        raise RotArmError(
            "No embedding model: retrieval would rank by confidence alone and the "
            "§6.6 ladder would never pair a claim with its predecessor — the run "
            "would measure nothing. Install the sentence-transformers extra."
        )
    cfg = get_config().benchmark_rot
    seeds = list(seeds) if seeds else list(cfg.seeds)
    days = days if days is not None else cfg.days
    cps = list(checkpoints) if checkpoints else [c for c in cfg.checkpoints if c <= days]
    top_k = top_k if top_k is not None else cfg.top_k
    today = run_date if run_date is not None else datetime.now(UTC).date()
    started = datetime.now(UTC)
    refused: dict[str, int] = {}

    probe_world = generate_world(seeds[0], days=days, checkpoints=cps)
    with _arm_routing(arm, probe_world, refused):
        extraction_id, lint_id = _resolved_models(arm)

    own_dir = work_dir is None
    base = Path(tempfile.mkdtemp(prefix="particles-rot-")) if work_dir is None else work_dir
    base.mkdir(parents=True, exist_ok=True)
    worlds: list[WorldResult] = []
    try:
        with isolated_blob_dir(base):
            for seed in seeds:
                worlds.append(
                    await _run_seed(
                        seed,
                        arm=arm,
                        days=days,
                        cps=cps,
                        top_k=top_k,
                        trust_policy=trust_policy,
                        epoch=datetime.combine(today - timedelta(days=days), time(0), UTC),
                        db_path=base / f"rot-seed{seed}.db",
                        pinned=(extraction_id, lint_id),
                        refused=refused,
                        cache_dir=cache_dir,
                        attribute=attribute,
                        progress=progress,
                    )
                )
                if not keep_stores:
                    for suffix in ("", "-wal", "-shm"):
                        Path(str(base / f"rot-seed{seed}.db") + suffix).unlink(missing_ok=True)
    finally:
        if own_dir and not keep_stores:
            shutil.rmtree(base, ignore_errors=True)

    all_probes = [p for w in worlds for p in w.probes]
    by_cp, by_ph, by_ch = breakdowns(all_probes)
    notes = _run_notes(arm, trust_policy, refused)
    selection = RotRunSelection(
        arm=arm,
        seeds=seeds,
        days=days,
        checkpoints=cps,
        top_k=top_k,
        trust_policy=trust_policy,
        untrusted_domain_trust=cfg.untrusted_domain_trust,
        generator_version=GENERATOR_VERSION,
        scorer_version=SCORER_VERSION,
        extraction_model_id=extraction_id,
        semantic_lint_model_id=lint_id,
        embedding_model_id=get_embedding_model_id(),
        thresholds=_thresholds(),
    )
    return RotBenchmarkReport(
        selection=selection,
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        metrics=merge_metrics(w.metrics for w in worlds),
        by_checkpoint=by_cp,
        by_phrasing=by_ph,
        by_channel=by_ch,
        floor_sweep=floor_sweep(all_probes, list(cfg.floor_sweep)),
        worlds=worlds,
        refused_llm_calls=dict(sorted(refused.items())),
        quality_notes=notes,
    )


async def _run_seed(  # noqa: PLR0913 — one world's slice of the run tuple
    seed: int,
    *,
    arm: str,
    days: int,
    cps: list[int],
    top_k: int,
    trust_policy: bool,
    epoch: datetime,
    db_path: Path,
    pinned: tuple[str, str],
    refused: dict[str, int],
    cache_dir: Path | None,
    attribute: str | None,
    progress: Callable[[str], None] | None,
) -> WorldResult:
    """Build one seed's world and run it under the arm's routing, pins enforced."""
    world = generate_world(seed, days=days, checkpoints=cps)
    with _arm_routing(arm, world, refused):
        now_ids = _resolved_models(arm)
        if now_ids != pinned:
            raise SameModelViolation(f"resolved models changed mid-run: {pinned} → {now_ids}")
        if progress is not None:
            progress(f"seed {seed}: {len(world.sessions)} sessions, arm {arm}")
        return await _run_world(
            world,
            arm=arm,
            top_k=top_k,
            trust_policy=trust_policy,
            epoch=epoch,
            db_path=db_path,
            cache_dir=cache_dir,
            attribute=attribute,
            progress=progress,
        )


def _run_notes(arm: str, trust_policy: bool, refused: Mapping[str, int]) -> list[str]:
    notes: list[str] = []
    if arm == "oracle":
        notes.append(
            "ORACLE arm: extraction and the §6.6 contradiction probe are scripted "
            "from the world's ground truth. This measures the ladder, source trust, "
            "and ranking given perfect perception — not the product."
        )
    elif arm == "probe":
        notes.append(
            "PROBE arm: extraction is scripted; the §6.6 contradiction probe is the "
            "live semantic_lint model. Isolates the probe's judgement on updates."
        )
    if not trust_policy:
        notes.append(
            "Trust policy OFF: no domain rule was written for the untrusted source, "
            "so it scores under the neutral-when-silent default."
        )
    notes.append(TOOL_TURN_NOTE)
    decay = decay_quality_note()
    if decay is not None:
        notes.append(decay)
    if refused:
        total = sum(refused.values())
        detail = ", ".join(f"{k}={v}" for k, v in sorted(refused.items()))
        notes.append(
            f"{total} LLM call(s) were refused by the scripted arm ({detail}); each "
            f"failed open through its call site's documented fallback. Nothing billed."
        )
    return notes


async def _run_world(
    world: RotWorld,
    *,
    arm: str,
    top_k: int,
    trust_policy: bool,
    epoch: datetime,
    db_path: Path,
    cache_dir: Path | None,
    attribute: str | None,
    progress: Callable[[str], None] | None,
) -> WorldResult:
    """Deposit one world in time order and probe it at each checkpoint."""
    cfg = get_config()
    extractor: ExtractorPlugin | None = OracleExtractor() if arm != "live" else None
    cache: ExtractionCache | None = None
    if arm == "live" and cache_dir is not None:
        inner = select_extractor(SourceType.CONVERSATION)
        cache = ExtractionCache(
            cache_dir / "rot-extraction.pickle",
            extractor=inner,
            model=get_provider("extraction").provider_model,
        )
        extractor = _CachedExtractor(inner, cache)
    by_day: dict[int, list[RotSession]] = {}
    for sess in world.sessions:
        by_day.setdefault(sess.day, []).append(sess)
    result = WorldResult(
        seed=world.seed, fingerprint=world.fingerprint, epoch=epoch.date().isoformat()
    )

    async with scratch_store(db_path) as factory, factory() as db:
        if trust_policy:
            await upsert_trust_rule(
                db,
                "domain",
                UNTRUSTED_DOMAIN,
                score=cfg.benchmark_rot.untrusted_domain_trust,
                modifier=None,
                rationale="rot benchmark: untrusted source channel",
                asserted_by="benchmark-rot",
                actor="benchmark-rot",
            )
            await db.commit()
        for day in range(1, world.days + 1):
            for sess in sorted(by_day.get(day, []), key=lambda s: s.seq):
                await _deposit_and_extract(db, world, sess, epoch, extractor, attribute)
                result.sessions_deposited += 1
            if day in world.checkpoints:
                result.probes += await _probe_checkpoint(db, world, day, top_k)
                if progress is not None:
                    progress(f"seed {world.seed}: checkpoint day {day} probed")
        rows = await db.execute(
            select(ParticleRow.status, ParticleRow.status_reason, func.count()).group_by(
                ParticleRow.status, ParticleRow.status_reason
            )
        )
        for status, reason, n in rows.all():
            key = status if reason is None else f"{status}/{reason}"
            result.store_census[key] = int(n)

    if cache is not None:
        cache.flush()
        result.quality_notes.append(
            f"Extraction cache: {cache.hits} hit(s), {cache.misses} miss(es) at {cache.path}"
        )
    result.metrics = metrics_for(result.probes)
    result.by_checkpoint, result.by_phrasing, result.by_channel = breakdowns(result.probes)
    return result


async def _deposit_and_extract(
    db: AsyncSession,
    world: RotWorld,
    sess: RotSession,
    epoch: datetime,
    extractor: ExtractorPlugin | None,
    attribute: str | None = None,
) -> None:
    stamp = epoch + timedelta(days=sess.day, minutes=10 * sess.seq)
    text = render_session(sess, stamp.date().isoformat())
    if isinstance(extractor, OracleExtractor):
        extractor.register(text, oracle_claims(world, sess))
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        db,
        text=text,
        uri_r=sess.uri,
        source_type=(
            SourceType.WEB_PAGE if sess.kind is SessionKind.WEB_PAGE else SourceType.CONVERSATION
        ),
        mutability=Mutability.STABLE,
        deposited_by="benchmark-rot",
        content_published_at=stamp,
    )
    if attribute is not None:
        # needs a positively attributed lineage before rung 2.5
        # fires in a ``multi`` store; no deposit path stamps one yet (that is
        # the multi-user MVP's), so the harness writes it where a harvester
        # would.
        from particles.corpus.store import SnapshotRow

        row = await db.get(SnapshotRow, snapshot_id)
        if row is not None:
            row.author_id = attribute
    await db.commit()
    if not unchanged:
        await extract_snapshot(db, entry_id, snapshot_id, extractor=extractor)
        await db.commit()


async def _probe_checkpoint(
    db: AsyncSession, world: RotWorld, day: int, top_k: int
) -> list[ProbeResult]:
    """Ask every probe for ``day`` and classify each top-k hit."""
    cfg = get_config()
    record_text = cfg.benchmark.record_claim_text
    out: list[ProbeResult] = []
    for probe in (p for p in world.probes if p.checkpoint == day):
        scored = await retrieve_ranked(db, QueryRequest(question=probe.question, top_k=top_k))
        particles = [p for p, _, _ in scored]
        badges = (
            await compute_contested_badges(db, particles)
            if cfg.contestedness.badge_enabled
            else [None] * len(particles)
        )
        state = (
            SlotState(slot=probe.slot, current=None, superseded=[], poison=[], poison_channels=[])
            if probe.negative
            else slot_state(world, probe.slot, day)
        )
        hits = [
            HitRecord(
                rank=i + 1,
                particle_id=p.id,
                text=p.content if record_text else None,
                hit_class=HitClass.NONE if probe.negative else classify(p.content, state),
                status=p.status.value,
                cosine=round(sim, 6),
                effective_confidence=round(eff, 6),
                contested=badge is not None,
            )
            for i, ((p, sim, eff), badge) in enumerate(zip(scored, badges, strict=True))
        ]
        result = ProbeResult(
            checkpoint=day,
            slot=probe.slot,
            question=probe.question,
            negative=probe.negative,
            current=state.current,
            superseded=state.superseded,
            poison=state.poison,
            poison_channels=state.poison_channels,
            last_phrasing=state.last_phrasing,
            top_cosine=max((sim for _, sim, _ in scored), default=None),
            hits=hits,
        )
        result.first_hit, result.first_hit_contested = first_value_bearing(result)
        out.append(result)
    return out
