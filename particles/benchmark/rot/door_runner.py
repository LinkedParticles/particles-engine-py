# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rot benchmark's door arm: other memories through the rot world.

Each memory is built fresh per world: the world's sessions are written in
time order through the memory's own write path, the memory settles after each
write, and at each checkpoint every probe is asked through the memory's
default door. The door text is recorded whole and scored by :mod:`.door`.

Three tiers share this driver and differ only in the world, the
seeds, the build count and what is rebuilt; no tier changes a memory's
configuration. A competitor's builds are cached on its pin and the world, so
during development only Particles is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from particles.benchmark.rot.answer_runner import (
    AssumedReads,
    ReadEstimate,
    estimate_read,
    plan_reads,
    readings_for,
)
from particles.benchmark.rot.contract import (
    ChildContractAdapter,
    DoorAdapterError,
    DoorMemory,
    ManagedServer,
    child_env,
    parent_environ,
    wait_healthy,
)
from particles.benchmark.rot.door import (
    BUDGETS_CHARS,
    DOOR_SEGMENTER_VERSION,
    evaluate_predictions,
    finalize_memory,
    score_build,
)
from particles.benchmark.rot.door_schema import (
    DoorBuild,
    DoorMemoryResult,
    DoorProbeResult,
    DoorReport,
    DoorRunSelection,
    ReaderPin,
)
from particles.benchmark.rot.generator import (
    GENERATOR_VERSION,
    UNTRUSTED_DOMAIN,
    generate_world,
    render_session,
)
from particles.benchmark.rot.manifest import ArmsManifest, MemorySpec
from particles.benchmark.rot.runner import estimate_rot_run
from particles.benchmark.rot.schema import RotWorld, SessionKind
from particles.benchmark.rot.scoring import SCORER_VERSION
from particles.config import McpConfig, get_config

#: tier → (world profile, builds per seed for a model-backed memory).
TIERS: dict[str, tuple[str, int]] = {
    "quick": ("quick", 1),
    "standard": ("full", 1),
    "milestone": ("full", 3),
}

DISCLOSURES: tuple[str, ...] = (
    "Written by the Particles authors: the world, the scorer, the segmenter and "
    "the driver are theirs, and the world's untrusted channels were designed "
    "around Particles' own trust model.",
    "Competitors run through adebench's own adapters, unmodified, each at its "
    "defaults with the model its adebench entry uses; every configuration is in "
    "the run tuple.",
)
TUNED_ON = (
    "These numbers come from development seeds Particles was tuned on. They are "
    "internal and are never a comparative claim."
)
HELD_OUT = (
    "These numbers come from held-out seeds of a generator Particles was tuned on "
    ", not from a held-out world."
)


@dataclass
class MemoryPlan:
    """How to build one row's memory, and what its row says about it."""

    name: str
    factory: Callable[[Path, RotWorld], tuple[DoorMemory, ManagedServer | None]]
    role: str = "headline"
    variant_note: str | None = None
    adapter: str = ""
    version: str = ""
    model: str | None = None
    model_backed: bool = False
    cacheable: bool = False
    wait_s: float = 30.0
    usd_per_full_build: tuple[float, float] | None = None
    maintainer_review: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    requires_session: bool = False


def competitor_plan(name: str, spec: MemorySpec, manifest: ArmsManifest) -> MemoryPlan:
    """A plan that runs one adebench adapter in its own process per build."""
    assert manifest.adebench is not None
    ade = manifest.adebench
    clone = Path(ade.clone).expanduser()
    python = Path(ade.python).expanduser()

    def factory(store: Path, world: RotWorld) -> tuple[DoorMemory, ManagedServer | None]:
        env, _ = child_env(parent_environ(), spec.env, store=store)
        Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
        for argv in spec.prepare:
            subprocess.run(argv, cwd=store, env=env, check=True)  # noqa: S603 — operator's manifest
        server = None
        if spec.server is not None:
            senv, _ = child_env(parent_environ(), spec.server.env, store=store)
            server = ManagedServer(
                [a.replace("{store}", str(store)) for a in spec.server.cmd],
                cwd=Path(spec.server.cwd.replace("{store}", str(store))),
                env={**env, **senv},
                log=store / "server.log",
            )
        try:
            memory = ChildContractAdapter(
                name,
                spec=spec.adapter,
                clone=clone,
                python=python,
                env=env,
                log=store / "adapter.log",
            )
        except DoorAdapterError:
            if server is not None:
                server.stop()
            raise
        return memory, server

    _, record = child_env(
        {k: "" for k in _env_sources(spec.env)} | parent_environ(), spec.env, store=Path("{store}")
    )
    return MemoryPlan(
        name=name,
        factory=factory,
        role="variant" if spec.patched else "headline",
        variant_note="patched adapter" if spec.patched else None,
        adapter=spec.adapter,
        version=spec.version,
        model=spec.model,
        model_backed=spec.model_backed,
        cacheable=True,
        wait_s=spec.wait_s,
        usd_per_full_build=spec.usd_per_full_build,
        maintainer_review=spec.maintainer_review,
        configuration={
            "env": record,
            "prepare": spec.prepare,
            "server": spec.server.model_dump() if spec.server else None,
            "adebench_sha": ade.sha,
        },
    )


def _env_sources(declared: dict[str, str]) -> list[str]:
    return [v[5:-1] for v in declared.values() if v.startswith("{env:") and v.endswith("}")]


def door_query_shape(mcp: McpConfig) -> dict[str, Any]:
    """The MCP ``query`` defaults a Particles door reads at."""
    return {
        "detail": mcp.query_detail,
        "list": mcp.query_list,
        "hits": mcp.query_hits,
        "grounded": mcp.query_detail == "compact" and mcp.query_list == "cited",
    }


def particles_plan(
    *,
    role: str = "headline",
    variant_note: str | None = None,
    parity: tuple[str, str] | None = None,
    scripted: bool = False,
    cache_dir: Path | None = None,
    trust_rule: bool = False,
    maintainer_review: str | None = None,
) -> MemoryPlan:
    """A plan for one Particles row (headline, trust-rule, parity or scripted)."""
    from particles.benchmark.rot.particles_door import ParticlesDoorMemory

    trust = (UNTRUSTED_DOMAIN, get_config().benchmark_rot.untrusted_domain_trust)

    def factory(store: Path, world: RotWorld) -> tuple[DoorMemory, ManagedServer | None]:
        return (
            ParticlesDoorMemory(
                store,
                world=(world.seed, world.days, world.profile),
                parity=parity,
                scripted=scripted,
                cache_dir=cache_dir,
                trust_rule=trust if trust_rule else None,
            ),
            None,
        )

    llm = get_config().llm
    model = (
        "scripted (oracle perception)"
        if scripted
        else (parity[1] if parity else llm.for_purpose("extraction").model)
    )
    configuration: dict[str, Any] = {"compiled_defaults": True}
    # the shape of the door the child reads, from the compiled
    # defaults the child runs under, so the report can footnote a compact row.
    configuration["mcp_query"] = door_query_shape(McpConfig())
    if parity is not None:
        configuration["every_purpose_pinned_to"] = f"{parity[0]}:{parity[1]}"
    if trust_rule:
        configuration["domain_trust_rule"] = {UNTRUSTED_DOMAIN: trust[1]}
    if cache_dir is not None:
        configuration["extraction_replay_cache"] = str(cache_dir)
    if scripted:
        configuration["perception"] = "oracle (scripted extraction and contradiction probe)"
    return MemoryPlan(
        name="particles",
        factory=factory,
        role=role,
        variant_note=variant_note,
        adapter="particles.benchmark.rot.particles_door:ParticlesDoorMemory",
        version=version("linkedparticles"),
        model=model,
        model_backed=not scripted,
        cacheable=False,
        wait_s=120.0,
        maintainer_review=maintainer_review,
        configuration=configuration,
        requires_session=True,
    )


def baseline_plans() -> list[MemoryPlan]:
    """The two zero-cost baselines."""
    from particles.benchmark.rot.baselines import RawChunksMemory, RecentMentionsMemory

    def raw(store: Path, world: RotWorld) -> tuple[DoorMemory, ManagedServer | None]:
        return RawChunksMemory(), None

    def recent(store: Path, world: RotWorld) -> tuple[DoorMemory, ManagedServer | None]:
        return RecentMentionsMemory(), None

    return [
        MemoryPlan(name="raw-chunks", factory=raw, role="baseline", adapter="baselines"),
        MemoryPlan(name="recent-mentions", factory=recent, role="baseline", adapter="baselines"),
    ]


# ---------------------------------------------------------------------------
# The competitor cache
# ---------------------------------------------------------------------------


def cache_key(plan: MemoryPlan, *, world: RotWorld, build: int) -> str:
    """A build's cache key: the memory's pin and the world, nothing else."""
    payload = {
        "name": plan.name,
        "adapter": plan.adapter,
        "version": plan.version,
        "model": plan.model,
        "configuration": plan.configuration,
        "profile": world.profile,
        "seed": world.seed,
        "days": world.days,
        "generator_version": GENERATOR_VERSION,
        "build": build,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(cache_dir: Path, plan: MemoryPlan, key: str) -> Path:
    return cache_dir / f"door-{plan.name}-{key[:16]}.json"


def load_cached(cache_dir: Path | None, plan: MemoryPlan, key: str) -> DoorBuild | None:
    """A cached build of a competitor, unscored, or ``None``."""
    if cache_dir is None or not plan.cacheable:
        return None
    path = _cache_path(cache_dir, plan, key)
    if not path.exists():
        return None
    build = DoorBuild.model_validate_json(path.read_text(encoding="utf-8"))
    build.cached = True
    return build


def store_cached(cache_dir: Path | None, plan: MemoryPlan, key: str, build: DoorBuild) -> None:
    """Keep a completed competitor build (its door texts and timings)."""
    failed = bool(build.notes) and not build.probes
    if cache_dir is None or not plan.cacheable or failed:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, plan, key).write_text(build.model_dump_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


class DoorEstimateRow(BaseModel):
    """One row's projected builds and spend."""

    name: str
    role: str
    builds: int = 0
    cached: int = 0
    usd_lo: float | None = 0.0
    usd_hi: float | None = 0.0
    basis: str = ""


class DoorEstimate(BaseModel):
    """What a door run will build and roughly cost, before anything runs."""

    tier: str
    profile: str
    seeds: list[int]
    rows: list[DoorEstimateRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def paid_builds(self) -> int:
        """Builds that will call a model (uncached, model-backed)."""
        return sum(r.builds - r.cached for r in self.rows if r.basis not in ("free", ""))

    @property
    def total(self) -> tuple[float | None, float | None]:
        """Low and high totals; ``None`` when any paid row is unpriced."""
        lo: float | None = 0.0
        hi: float | None = 0.0
        for r in self.rows:
            lo = None if lo is None or r.usd_lo is None else lo + r.usd_lo
            hi = None if hi is None or r.usd_hi is None else hi + r.usd_hi
        return lo, hi


def _builds_for(plan: MemoryPlan, tier: str) -> int:
    return TIERS[tier][1] if plan.model_backed else 1


def _particles_build_usd(world: RotWorld) -> float | None:
    """One Particles build: the live arm's projection plus the door's answers."""
    cfg = get_config()
    est = estimate_rot_run(
        "live",
        seeds=[world.seed],
        days=world.days,
        checkpoints=world.checkpoints,
        profile=world.profile,
    )
    if est.cost_usd is None:
        return None
    llm = cfg.llm
    price = llm.price_for(llm.for_purpose("query_response"))
    if price is None:
        return None
    rot = cfg.benchmark_rot
    answers = len(world.probes)
    return (
        est.cost_usd
        + answers
        * (
            rot.estimate_answer_input_tokens * price.input
            + rot.estimate_answer_output_tokens * price.output
        )
        / 1e6
    )


def estimate_door_run(
    plans: list[MemoryPlan], *, tier: str, seeds: list[int], cache_dir: Path | None
) -> DoorEstimate:
    """Project builds and spend per row; nothing is called."""
    profile = TIERS[tier][0]
    days = get_config().benchmark_rot.days
    est = DoorEstimate(tier=tier, profile=profile, seeds=list(seeds))
    worlds = {s: generate_world(s, days=days, profile=profile) for s in seeds}
    full_sessions = {s: len(generate_world(s, days=days).sessions) for s in seeds}
    for plan in plans:
        n = _builds_for(plan, tier)
        row = DoorEstimateRow(name=plan.name, role=plan.role, builds=n * len(seeds))
        lo: float | None = 0.0
        hi: float | None = 0.0
        for seed, world in worlds.items():
            for b in range(n):
                if load_cached(cache_dir, plan, cache_key(plan, world=world, build=b)) is not None:
                    row.cached += 1
                    continue
                if not plan.model_backed:
                    continue
                if plan.name == "particles":
                    usd = _particles_build_usd(world)
                    lo = None if lo is None or usd is None else lo + usd
                    hi = None if hi is None or usd is None else hi + usd
                    row.basis = "estimated (rot live projection + answers)"
                elif plan.usd_per_full_build is not None:
                    scale = len(world.sessions) / full_sessions[seed]
                    lo = None if lo is None else lo + plan.usd_per_full_build[0] * scale
                    hi = None if hi is None else hi + plan.usd_per_full_build[1] * scale
                    row.basis = "estimated (manifest figure, scaled by world size)"
                else:
                    lo = hi = None
                    row.basis = "unpriced (no usd_per_full_build in the manifest)"
        if not plan.model_backed:
            row.basis = "free"
        row.usd_lo, row.usd_hi = lo, hi
        est.rows.append(row)
    est.notes.append(
        "Order-of-magnitude figures, not quotes: competitor figures come from the "
        "arms manifest; Particles' from the rot live-arm projection plus one "
        "generated answer per probe at benchmark_rot.estimate_answer_* tokens."
    )
    return est


def estimate_run_reads(
    plans: list[MemoryPlan],
    readers: list[ReaderPin],
    *,
    tier: str,
    seeds: list[int],
    cache_dir: Path | None,
    reader_cache_dir: Path | None,
) -> ReadEstimate:
    """The answer stage's estimate for a ``door run``, before any build.

    A cached competitor build is priced from its recorded door texts (and the
    reader cache); every other build's doors at an assumed size.
    """
    profile = TIERS[tier][0]
    days = get_config().benchmark_rot.days
    known: list[DoorMemoryResult] = []
    assumed: list[AssumedReads] = []
    for plan in plans:
        row = DoorMemoryResult(name=plan.name, role=plan.role, variant_note=plan.variant_note)
        for seed in seeds:
            world = generate_world(seed, days=days, profile=profile)
            for b in range(_builds_for(plan, tier)):
                cached = load_cached(cache_dir, plan, cache_key(plan, world=world, build=b))
                if cached is not None and cached.probes:
                    row.builds.append(cached)
                    continue
                for pin in readers:
                    assumed += [
                        AssumedReads(pin=pin, reading=reading, count=len(world.probes))
                        for reading in readings_for(row, pin.role)
                    ]
        if row.builds:
            known.append(row)
    jobs = plan_reads(
        DoorReport(
            selection=DoorRunSelection(
                tier=tier,
                profile=profile,
                seeds=list(seeds),
                generator_version=GENERATOR_VERSION,
                scorer_version=SCORER_VERSION,
                door_segmenter_version=DOOR_SEGMENTER_VERSION,
            ),
            started_at="",
            finished_at="",
            memories=known,
        ),
        readers,
    )
    return estimate_read(jobs, cache_dir=reader_cache_dir, assumed=assumed)


def render_door_estimate(est: DoorEstimate) -> str:
    """The estimate as printed before any build."""
    lines = [
        f"Rot door arm estimate — tier {est.tier} ({est.profile} world), seeds {est.seeds}",
    ]
    for r in est.rows:
        if r.basis == "free":
            money = "US$0"
        elif r.usd_lo is None or r.usd_hi is None:
            money = "unpriced"
        else:
            money = f"~US${r.usd_lo:,.2f}–{r.usd_hi:,.2f}"
        cached = f", {r.cached} cached" if r.cached else ""
        lines.append(f"  {r.name} [{r.role}]: {r.builds} build(s){cached}, {money} ({r.basis})")
    lo, hi = est.total
    total = "unpriced" if lo is None or hi is None else f"~US${lo:,.2f}–{hi:,.2f}"
    lines.append(f"  total: {total}; {est.paid_builds} build(s) call a model")
    lines += [f"  note: {n}" for n in est.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def _probe_once(memory: DoorMemory, door: str, probe: Any, day: int) -> DoorProbeResult:
    result = DoorProbeResult(
        checkpoint=day, slot=probe.slot, question=probe.question, negative=probe.negative, door=door
    )
    started = time.monotonic()
    try:
        text, raw = memory.door_text(probe.question, door)
    except DoorAdapterError as exc:
        result.error = str(exc)
    else:
        result.door_text = text
        result.raw = raw
        result.chars = len(text)
    result.door_ms = round((time.monotonic() - started) * 1000, 1)
    return result


def run_build(
    plan: MemoryPlan,
    world: RotWorld,
    *,
    build: int,
    store: Path,
    epoch: datetime,
    progress: Callable[[str], None] | None = None,
) -> DoorBuild:
    """Write one world into a fresh memory in time order and probe it at each checkpoint."""
    out = DoorBuild(
        seed=world.seed,
        build=build,
        fingerprint=world.fingerprint,
        profile=world.profile,
        epoch=epoch.date().isoformat(),
    )
    store.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    memory: DoorMemory | None = None
    server: ManagedServer | None = None
    current = ""
    try:
        memory, server = plan.factory(store, world)
        # adebench calls health() on an imported memory, and several adapters
        # (mem0, cognee, engram) answer False on an empty store. A server must
        # answer before the first write; every memory is checked after it,
        # before any probe.
        healthy = server is not None
        if healthy:
            wait_healthy(memory, wait_s=plan.wait_s, server=server)
        if plan.requires_session and not memory.supports("import_session"):
            raise DoorAdapterError(f"{plan.name}: import_session is required and missing")
        door = memory.doors()[0]
        usd_before = memory.usage_usd() if memory.supports("usage") else None
        by_day: dict[int, list[Any]] = {}
        for sess in world.sessions:
            by_day.setdefault(sess.day, []).append(sess)
        for day in range(1, world.days + 1):
            for sess in sorted(by_day.get(day, []), key=lambda s: s.seq):
                current = sess.session_id
                stamp = epoch + timedelta(days=sess.day, minutes=10 * sess.seq)
                text = render_session(sess, stamp.date().isoformat())
                t0 = time.monotonic()
                if sess.kind is SessionKind.WEB_PAGE and memory.supports("import_page"):
                    memory.import_page(
                        text, stamp.isoformat(), uri=sess.uri, session_id=sess.session_id
                    )
                    method = "import_page"
                elif sess.kind is SessionKind.CONVERSATION and memory.supports("import_session"):
                    turns = [{"role": t.role, "content": t.content} for t in sess.turns]
                    memory.import_session(
                        turns, stamp.isoformat(), text=text, session_id=sess.session_id
                    )
                    method = "import_session"
                else:
                    memory.import_memory(text, stamp.isoformat())
                    method = "import_memory"
                memory.settle()
                out.write_ms_samples.append(round((time.monotonic() - t0) * 1000, 1))
                out.write_methods[method] = out.write_methods.get(method, 0) + 1
                out.sessions_written += 1
                if not healthy:
                    wait_healthy(memory, wait_s=plan.wait_s, server=server)
                    healthy = True
            if day in world.checkpoints:
                current = f"checkpoint day {day}"
                for probe in (p for p in world.probes if p.checkpoint == day):
                    out.probes.append(_probe_once(memory, door, probe, day))
                if progress is not None:
                    progress(f"{plan.name} seed {world.seed} build {build}: day {day} probed")
        if usd_before is not None:
            usd_after = memory.usage_usd()
            if usd_after is not None:
                out.usd = round(usd_after - usd_before, 4)
                out.usd_basis = "measured"
    except (DoorAdapterError, subprocess.CalledProcessError, OSError) as exc:
        out.notes.append(f"build failed at {current or 'start'}: {exc}")
        out.probes = []
    finally:
        if memory is not None:
            memory.close()
        if server is not None:
            server.stop()
    out.build_minutes = round((time.monotonic() - started) / 60, 2)
    return out


def run_door_benchmark(  # noqa: PLR0913 — the run tuple is the API
    plans: list[MemoryPlan],
    *,
    tier: str,
    seeds: list[int],
    work_dir: Path,
    manifest: ArmsManifest,
    cache_dir: Path | None = None,
    held_out: bool = False,
    seed_commitment: str | None = None,
    predictions: dict[str, str] | None = None,
    run_date: date | None = None,
    keep_stores: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DoorReport:
    """Build every row's memory on every seed and return the report of record."""
    profile, model_builds = TIERS[tier]
    days = get_config().benchmark_rot.days
    today = run_date if run_date is not None else datetime.now(UTC).date()
    epoch = datetime.combine(today - timedelta(days=days), dtime(0), UTC)
    started = datetime.now(UTC)
    worlds = {s: generate_world(s, days=days, profile=profile) for s in seeds}
    memories: list[DoorMemoryResult] = []
    for plan in plans:
        row = DoorMemoryResult(
            name=plan.name,
            role=plan.role,
            variant_note=plan.variant_note,
            adapter=plan.adapter,
            version=plan.version,
            model=plan.model,
            model_backed=plan.model_backed,
            configuration=plan.configuration,
            maintainer_review=plan.maintainer_review,
        )
        label = plan.name if plan.variant_note is None else f"{plan.name}-{plan.variant_note}"
        for seed, world in worlds.items():
            for b in range(_builds_for(plan, tier)):
                key = cache_key(plan, world=world, build=b)
                build = load_cached(cache_dir, plan, key)
                if build is None:
                    store = work_dir / label.replace(" ", "-") / f"seed{seed}-b{b}"
                    if progress is not None:
                        progress(f"{label}: seed {seed}, build {b}, {len(world.sessions)} sessions")
                    build = run_build(
                        plan, world, build=b, store=store, epoch=epoch, progress=progress
                    )
                    store_cached(cache_dir, plan, key, build)
                    if not keep_stores:
                        shutil.rmtree(store, ignore_errors=True)
                if plan.model_backed and plan.usd_per_full_build and build.usd_basis == "none":
                    full = len(generate_world(seed, days=days).sessions)
                    mid = sum(plan.usd_per_full_build) / 2 * len(world.sessions) / full
                    build.usd, build.usd_basis = round(mid, 4), "estimated"
                if build.notes and not build.probes:
                    row.builds.append(build)
                    continue
                score_build(build, world)
                if plan.model_backed and _builds_for(plan, tier) == 1:
                    build.notes.append(
                        "single build: no range (one sample of a model-backed memory)"
                    )
                row.builds.append(build)
        row.builds = sorted(row.builds, key=lambda b: (b.seed, b.build))
        scored = DoorMemoryResult.model_validate(
            row.model_dump() | {"builds": [b for b in row.builds if b.probes]}
        )
        finalize_memory(scored)
        row.metrics, row.ranges = scored.metrics, scored.ranges
        memories.append(row)
    selection = DoorRunSelection(
        tier=tier,
        profile=profile,
        days=days,
        seeds=list(seeds),
        held_out=held_out,
        seed_commitment=seed_commitment,
        builds_model_backed=model_builds,
        builds_deterministic=1,
        generator_version=GENERATOR_VERSION,
        scorer_version=SCORER_VERSION,
        door_segmenter_version=DOOR_SEGMENTER_VERSION,
        budgets_chars=list(BUDGETS_CHARS),
        adebench_sha=manifest.adebench.sha if manifest.adebench else None,
        manifest_sha256=manifest.sha256,
        manifest=manifest.raw,
        particles_version=version("linkedparticles"),
    )
    preds = dict(predictions or {})
    return DoorReport(
        selection=selection,
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        disclosures=[*DISCLOSURES, HELD_OUT if held_out else TUNED_ON],
        predictions=preds,
        prediction_results=evaluate_predictions(preds, memories),
        memories=memories,
    )
