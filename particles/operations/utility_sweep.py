"""The rank-lift calibration sweep — the harness, shipped.

This calibrated ``utility.default.rank_lift`` (``λ``) by hand and left the
sweep as a recipe in its §Validation section. A question then asked whether ``λ``
could be *fitted* from the store instead, "analogous to the temperature
fit". That was declined by measurement: temperature scaling minimises a
real loss against **labelled** benchmark pairs, and nothing here supplies the
analogue — no label says which belief *should* occupy a head slot. Every
candidate closed form either overshot the admissible band by 3–8×, returned zero
(the cap flattens the head's ``effective_confidence`` spread to exactly
0.0000, so a spread-keyed fit disables the feature it is calibrating), or keyed
on head diversity — which measures over-extraction, not utility policy,
and would quietly re-tune ``λ`` downward to conceal duplicate clusters.

So this module ships the *harness* rather than a fit: it loads the two maps recipe names, hands them to the pure sweep in ``core/scoring/utility.py``, and
lets the operator supply the one input a fit cannot manufacture — which beliefs
they assert ought to reach the head.

It is **read-only**: no writes, no LLM calls, no embeddings. Scoring reuses the
query path's own :func:`score_effective_confidence` and the projection's own
reinforcement lookup, so the sweep can never disagree with what the digest and
the projection actually render.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle
from particles.core.scoring.relevance import OwnerRankLiftSweep, sweep_owner_rank_lift
from particles.core.scoring.utility import (
    RankLiftSweep,
    SweepRow,
    content_dedup_key,
    rank_lift_grid,
    sweep_rank_lift,
)
from particles.core.status import Status

#: Default sweep grid — 120 points to ``0.12``, so band edges resolve to 0.001.
#: The range brackets every value measured as interesting on a real
#: store (its admissible bands ran 0.005–0.031) with an order of magnitude of
#: headroom above, since a store whose confidence axis is *not* capped needs a
#: larger ``λ`` to move its head at all.
DEFAULT_GRID_MAX = 0.12
DEFAULT_GRID_STEPS = 120

#: Fraction of head slots that must hold distinct content for a ``λ`` to pass.
#: Not 1.0 — see :func:`particles.core.scoring.utility.sweep_rank_lift`.
DEFAULT_DISTINCT_RATIO = 0.95


async def load_sweep_rows(
    session: AsyncSession,
    particles: list[Particle] | None = None,
) -> list[SweepRow]:
    """Load the two-map recipe for every ACTIVE belief.

    The base map is :func:`score_effective_confidence` with
    ``apply_utility_factor=False`` (the truth-axis score the lift is added to);
    ``R`` comes from :func:`get_reinforcement_scores` under each belief's own
    resolved half-life, grouped exactly as ``apply_utility`` groups it so a
    per-``source_type`` or ``url_pattern`` utility rule is honoured here too.

    Args:
        session: Active store session.
        particles: Pre-loaded ACTIVE particles to reuse; ``None`` loads them.
    """
    from particles.operations.query.effective_confidence import score_effective_confidence
    from particles.operations.query.owner_policy import load_owner_policy
    from particles.operations.query.source_info import load_source_rows
    from particles.operations.query.utility_policy import (
        EMPTY_UTILITY_POLICY,
        load_utility_policy,
    )
    from particles.store.particle_store import get_particles_by_status
    from particles.store.utility_store import get_reinforcement_scores

    if particles is None:
        particles = await get_particles_by_status(session, Status.ACTIVE)
    if not particles:
        return []

    eff = await score_effective_confidence(
        session, particles, populate_cache=True, apply_utility_factor=False
    )

    # Reinforcement, grouped by each belief's resolved half-life — the same
    # single-pass shape `apply_utility` uses, so no per-particle round trips.
    policy = await load_utility_policy(session)
    scores: dict[str, float] = {}
    if policy is not EMPTY_UTILITY_POLICY:
        source_rows = await load_source_rows(session, particles)
        by_half_life: dict[float, list[str]] = {}
        for p in particles:
            _pub, source_type, _eid, uri_r, _author = source_rows.get(
                p.id, (None, "", None, None, None)
            )
            half_life = policy.half_life_uses_days(source_type, uri_r)
            if half_life is not None:
                by_half_life.setdefault(half_life, []).append(p.id)
        for half_life, pids in by_half_life.items():
            scores.update(await get_reinforcement_scores(session, pids, half_life))

    # A(p): resolved once per sweep, exactly as a render resolves it,
    # so the ω sweep measures the same cohort the lens will actually promote.
    owner_policy = await load_owner_policy(session, require_rank_lift=False)

    return [
        SweepRow(
            particle_id=p.id,
            effective_confidence=eff.get(p.id, p.confidence.value),
            reinforcement=scores.get(p.id, 0.0),
            content_key=content_dedup_key(p.content),
            owner_relevant=owner_policy.is_owner_relevant(p.subject_ids),
        )
        for p in particles
    ]


async def sweep_store_rank_lift(
    session: AsyncSession,
    *,
    head_sizes: list[int],
    target_ids: list[str] | None = None,
    grid_max: float = DEFAULT_GRID_MAX,
    grid_steps: int = DEFAULT_GRID_STEPS,
    distinct_ratio: float = DEFAULT_DISTINCT_RATIO,
) -> RankLiftSweep:
    """Sweep ``λ`` over one store and return the per-surface admissible bands.

    Read-only. The configured ``utility.default.rank_lift`` is carried into the
    result so the caller can report whether the value in force actually sits
    inside the band it is supposed to.

    Args:
        session: Active store session.
        head_sizes: Each rendered surface's ``N``: the band is a
            property of the surface, not the store, so passing every ``N`` the
            deployment actually renders is what makes the answer meaningful.
        target_ids: Beliefs the operator asserts ought to reach the head.
        grid_max: Largest ``λ`` to evaluate.
        grid_steps: Non-zero grid points (band edges resolve to one step).
        distinct_ratio: Fraction of head slots that must hold distinct content.
    """
    from particles.config import get_config

    rows = await load_sweep_rows(session)
    return sweep_rank_lift(
        rows,
        grid=rank_lift_grid(grid_max, grid_steps),
        head_sizes=head_sizes,
        target_ids=target_ids or [],
        distinct_ratio=distinct_ratio,
        configured_rank_lift=get_config().utility.default.rank_lift,
    )


def render_rank_lift_sweep(sweep: RankLiftSweep) -> str:
    """Render a sweep as the operator-facing Markdown report.

    Three parts: the per-``λ`` grid table (one column group per surface), the
    resolved band per surface plus their intersection, and the verdict on the
    configured value. Rows whose outcome is identical to the previous row are
    still printed — the flat stretches *are* the finding (the
    rendered head is identical across a wide interior of the band, which is why
    a fit would be optimising something that changes nothing).
    """
    if not sweep.points:
        return "No ACTIVE beliefs to sweep — nothing to calibrate."

    sizes = [n for n, _band in sweep.bands]
    targets = [pid for pid, _rank in sweep.points[0].heads[0].target_ranks]

    lines = [
        f"# Rank-lift calibration sweep — {sweep.scored} ACTIVE beliefs",
        "",
        "Read-only. `λ` is not fitted; this reports where the",
        "acceptance criteria hold so the operator can choose it.",
        "",
    ]

    header = ["| `λ`"]
    rule = ["|---"]
    for pid in targets:
        header.append(f"| rank `{pid[:8]}`")
        rule.append("|---")
    for n in sizes:
        header.append(f"| distinct/{n}")
        rule.append("|---")
    header.append("| admissible |")
    rule.append("|---|")
    lines.extend(["".join(header), "".join(rule)])

    for point in sweep.points:
        cells = [f"| {point.rank_lift:.4f}"]
        for _pid, rank in point.heads[0].target_ranks:
            cells.append(f"| {rank if rank else '—'}")
        for head in point.heads:
            cells.append(f"| {head.distinct_contents}")
        ok = all(h.admissible for h in point.heads)
        cells.append(f"| {'yes' if ok else 'no'} |")
        lines.append("".join(cells))

    lines.extend(["", "## Admissible band", ""])
    for n, band in sweep.bands:
        if band.empty:
            lines.append(f"- **N = {n}** — empty: no `λ` on the grid satisfies both criteria.")
        else:
            note = "" if band.contiguous else " (**not contiguous** — interior points fail)"
            lines.append(f"- **N = {n}** — `{band.low:.4f}` – `{band.high:.4f}`{note}")

    inter = sweep.intersection
    if inter.empty:
        lines.append(
            "- **all surfaces** — empty: no single `λ` satisfies every rendered head "
            "size at once (the band is a property of the surface, not "
            "the store)."
        )
    else:
        lines.append(f"- **all surfaces** — `{inter.low:.4f}` – `{inter.high:.4f}`")

    lines.extend(["", "## Configured value", ""])
    configured = sweep.configured_rank_lift
    if configured is None:
        lines.append("No configured `utility.default.rank_lift` supplied.")
    elif sweep.configured_admissible:
        lines.append(
            f"`utility.default.rank_lift = {configured}` is **inside** the all-surface band."
        )
    else:
        detail = ", ".join(
            f"N={n} {'in' if band.contains(configured) else 'OUT'}" for n, band in sweep.bands
        )
        lines.append(
            f"`utility.default.rank_lift = {configured}` is **outside** the all-surface "
            f"band ({detail}). Choose a value inside it, or accept that one store-global "
            "knob cannot satisfy every rendered `N` and record which surface it is "
            "calibrated for."
        )
    return "\n".join(lines) + "\n"


#: The largest fraction of a rendered head the viewer cohort may occupy
#: (criterion 2). Half is deliberately permissive: the lens exists
#: to make the viewer's slice reachable, and a recall head that is *mostly*
#: about the reader is a legitimate configuration for a personal store. It bites
#: on the failure this criterion actually guards — a flat step promoting a whole
#: cohort at once, leaving no room for the domain knowledge that is the rest of
#: the store's value.
DEFAULT_MAX_OWNER_SHARE = 0.5


async def sweep_store_owner_rank_lift(
    session: AsyncSession,
    *,
    head_sizes: list[int],
    target_ids: list[str] | None = None,
    grid_max: float = DEFAULT_GRID_MAX,
    grid_steps: int = DEFAULT_GRID_STEPS,
    min_owner_in_head: int = 1,
    max_owner_share: float = DEFAULT_MAX_OWNER_SHARE,
) -> OwnerRankLiftSweep:
    """Sweep ``ω`` over one store and return the per-surface admissible bands.

    Read-only. The utility ``λ`` in force is held **fixed** across the sweep, so
    what is measured is what aboutness does to the head *utility has already
    shaped* — which is what makes third criterion (the targets must not regress out of the head) mean anything.

    Args:
        session: Active store session.
        head_sizes: Each rendered surface's ``N`` (the band is a
            property of the surface, not the store).
        target_ids: Beliefs that must **stay** in the head — pass the utility targets here to check them for non-regression.
        grid_max: Largest ``ω`` to evaluate.
        grid_steps: Non-zero grid points (band edges resolve to one step).
        min_owner_in_head: Criterion 1's floor — viewer beliefs the head must
            hold for an ``ω`` to be admissible.
        max_owner_share: Criterion 2's ceiling on the cohort's head share.
    """
    from particles.config import get_config

    cfg = get_config()
    rows = await load_sweep_rows(session)
    return sweep_owner_rank_lift(
        rows,
        grid=rank_lift_grid(grid_max, grid_steps),
        head_sizes=head_sizes,
        lambda_=cfg.utility.default.rank_lift if cfg.utility.enabled else 0.0,
        target_ids=target_ids or [],
        min_owner_in_head=min_owner_in_head,
        max_owner_share=max_owner_share,
        configured_rank_lift=cfg.owner_lens.rank_lift,
    )


def render_owner_rank_lift_sweep(sweep: OwnerRankLiftSweep) -> str:
    """Render an ``ω`` sweep as the operator-facing Markdown report.

    Sibling of :func:`render_rank_lift_sweep`, reporting the quantity that
    matters for *this* lens: the viewer cohort's **share of each rendered
    head** at every candidate ``ω``, plus the three acceptance criteria.
    """
    if not sweep.points:
        return "No ACTIVE beliefs to sweep — nothing to calibrate."
    if sweep.owner_population == 0:
        return (
            "No viewer-relevant beliefs in the scored set — nothing for the owner\n"
            "lens to promote. Check `owner_lens.subjects`: the configured viewer\n"
            "must resolve to a local Subject that beliefs are actually linked to\n"
            "(`particles subjects search <name>`)."
        )

    sizes = [n for n, _band in sweep.bands]
    targets = [pid for pid, _rank in sweep.points[0].heads[0].target_ranks]
    share_pct = 100.0 * sweep.owner_population / sweep.scored if sweep.scored else 0.0

    lines = [
        f"# Owner-lens calibration sweep — {sweep.scored} ACTIVE beliefs",
        "",
        f"Viewer cohort: **{sweep.owner_population}** beliefs ({share_pct:.1f}% of the store).",
        "",
        "Read-only. `ω` is not fitted (a cohort-normalised lift would",
        "put a store-wide aggregate into a per-belief score); this reports where the",
        "three acceptance criteria hold so the operator can choose it. The utility `λ`",
        "in force is held fixed, so the third criterion — the utility targets must not",
        "regress out of the head — measures what it claims to.",
        "",
    ]

    header = ["| `ω`"]
    rule = ["|---"]
    for pid in targets:
        header.append(f"| rank `{pid[:8]}`")
        rule.append("|---")
    for n in sizes:
        header.append(f"| owner/{n}")
        rule.append("|---")
    header.append("| admissible |")
    rule.append("|---|")
    lines.extend(["".join(header), "".join(rule)])

    for point in sweep.points:
        cells = [f"| {point.rank_lift:.4f}"]
        for _pid, rank in point.heads[0].target_ranks:
            cells.append(f"| {rank if rank else '—'}")
        for head in point.heads:
            cells.append(f"| {head.owner_in_head} ({head.owner_share:.0%})")
        ok = all(h.admissible for h in point.heads)
        cells.append(f"| {'yes' if ok else 'no'} |")
        lines.append("".join(cells))

    # Disclose any target that was ALREADY outside a head at ω = 0. Criterion 3
    # skips those (it is a non-regression check, not a membership check), so
    # without this the pre-existing regression would vanish into a passing band.
    stale: list[str] = []
    for head in sweep.points[0].heads:
        for pid in head.targets_absent_at_baseline:
            rank = dict(head.target_ranks).get(pid, 0)
            stale.append(
                f"- `{pid[:8]}` was **already outside** the N = {head.head_size} head "
                f"before this lens (rank {rank or '—'} at `ω = 0`). Criterion 3 is a "
                "non-regression check, so it is skipped for that surface — but the "
                "regression is real and belongs to whatever moved it, not to `ω`."
            )
    if stale:
        lines.extend(["", "## Targets already out of the head at `ω = 0`", "", *stale])

    # "No silent caps": if the top grid point still passes, the band's upper edge
    # is an artifact of the grid rather than a measured ceiling. Reporting it as
    # a ceiling would invite an operator to read a hard constraint into a number
    # that only marks where they stopped looking.
    open_top = bool(sweep.points) and all(h.admissible for h in sweep.points[-1].heads)
    open_note = " — **open at the top** (grid edge)" if open_top else ""

    lines.extend(["", "## Admissible band", ""])
    for n, band in sweep.bands:
        if band.empty:
            lines.append(
                f"- **N = {n}** — empty: no `ω` on the grid satisfies all three criteria. "
                "Because `A(p)` is a flat step, `ω` is a threshold over the whole cohort — "
                "if the cohort is large relative to the head there may be no value that "
                "both surfaces the viewer and leaves room for anything else. The named "
                "escape hatch is to grade `A(p)`, not to raise `ω`."
            )
        else:
            note = "" if band.contiguous else " (**not contiguous** — interior points fail)"
            lines.append(f"- **N = {n}** — `{band.low:.4f}` – `{band.high:.4f}`{note}{open_note}")

    inter = sweep.intersection
    if inter.empty:
        lines.append(
            "- **all surfaces** — empty: no single `ω` satisfies every rendered head size "
            "at once (the band is a property of the surface, not the store)."
        )
    else:
        note = "" if inter.contiguous else " (**not contiguous**)"
        lines.append(
            f"- **all surfaces** — `{inter.low:.4f}` – `{inter.high:.4f}`{note}{open_note}"
        )
    if open_top:
        lines.extend(
            [
                "",
                f"The largest `ω` swept (`{sweep.points[-1].rank_lift:.4f}`) is still "
                "admissible, so the upper edge above is **where the grid stopped, not where a "
                "criterion failed**. Raise `--grid-max` to find the real ceiling — though on a "
                "store whose viewer cohort is small the share criterion may never bite at any "
                "plausible `ω`, in which case the ceiling is not the thing to calibrate "
                "against. Use the amplification below instead.",
            ]
        )

    if sweep.scored and sweep.owner_population:
        store_share = sweep.owner_population / sweep.scored
        lines.extend(
            [
                "",
                "## Choosing a value — amplification",
                "",
                "Divide the viewer's share of a head by their share of the store "
                f"(**{store_share:.1%}** here). That ratio is the **amplification** the lens "
                "applies, and unlike a raw `ω` it means the same thing on a store of any size "
                "or genre — so it is the quantity to have an opinion about. Prefer the "
                "interior of a flat stretch over a value sitting on a jump: a plateau is what "
                "keeps the outcome stable as the store grows.",
                "",
            ]
        )
        for probe in sweep.points[0].heads:
            n = probe.head_size
            shares = [
                h.owner_share
                for p in sweep.points
                for h in p.heads
                if h.head_size == n and h.admissible
            ]
            if shares:
                lines.append(
                    f"- **N = {n}** — admissible `ω` spans roughly "
                    f"**{shares[0] / store_share:.0f}×** to "
                    f"**{shares[-1] / store_share:.0f}×** amplification."
                )

    lines.extend(["", "## Configured value", ""])
    configured = sweep.configured_rank_lift
    if configured is None:
        lines.append("- `owner_lens.rank_lift` is unset.")
    elif configured == 0.0:
        lines.append(
            "- `owner_lens.rank_lift` is `0.0` — the lens is **inert** (the shipped "
            "default). Pick a value from the band above to enable it."
        )
    elif sweep.configured_admissible:
        lines.append(f"- `owner_lens.rank_lift = {configured}` is **inside** the band.")
    else:
        lines.append(f"- `owner_lens.rank_lift = {configured}` is **outside** the admissible band.")
    return "\n".join(lines)
