"""Query-time usefulness (utility) policy — the lens layer.

The usefulness analogue of the ``DecayPolicy`` (``decay_policy.py``): a
once-per-render snapshot resolving each belief's utility parameters
``(half_life_uses_days, rank_lift)`` from the store's **local** ``utility``
config composed over the ``utility_rules`` of every adopted trust lens. It turns
the per-belief utility *evidence* (the mined reinforcement count,
``store/utility_store.py``) into an additive rank bonus (``core/utility.py``).

Composition, mirroring lens composition:

- **Three scopes, most-specific wins.** A ``url_pattern`` rule matching the
  belief's URI overrides a ``source_type`` rule, which overrides the
  ``default`` rule.
- **Local wins; across lenses, most-skeptical-wins.** The store's local
  ``utility`` config is the base for the ``default`` scope; for keys it does not
  set, adopted lenses contribute the *least promotion* — shortest
  ``half_life_uses_days`` and lowest ``rank_lift`` (each minimised
  independently). So an adopted lens can only make utility promote *less*,
  never inflate another store's recall.

**Applied only on the projection / digest ranking path** — never the
semantic-search retrieval rank. This split makes the boundary
sharper than the superseded multiplier did: the bonus is added to form a
*ranking score*, and neither the stored ``confidence.value`` nor the read-time
``effective_confidence`` is ever scaled. The resulting score is an
ordering key, **not** a probability — it is not confined to ``[0, 1]``.

With ``utility.enabled = False``, or a belief with no utility evidence, the
bonus is ``+0``, so the ranking is byte-for-byte the pre-0190 order (cold-start).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.scoring.utility import utility_rank_bonus
from particles.store.lens_store import get_adopted_lenses
from particles.store.utility_store import get_reinforcement_scores

#: A resolved utility rule: (half_life_uses_days, rank_lift).
UtilityParams = tuple[float, float]


def _more_skeptical(a: UtilityParams, b: UtilityParams) -> UtilityParams:
    """Most-skeptical (least-promoting) composition: min each parameter.

    With the additive form, minimising ``rank_lift`` is exactly
    minimising promotion — a smaller ``λ`` scales the whole bonus down, so the
    "an adopted lens can only promote less" guarantee is now a single
    comparison rather than three independent clamps.
    """
    return (min(a[0], b[0]), min(a[1], b[1]))


@dataclass(frozen=True)
class UtilityPolicy:
    """One store's usefulness policy, snapshotted for in-process projection scoring.

    Attributes:
        default: the store-wide ``(half_life_uses_days, rank_lift)``
            (local config base, min-composed with adopted lenses' ``default``
            rules for a key local does not set).
        source_type_rules: ``{source_type: params}`` from adopted lenses,
            min-composed per source type.
        url_rules: compiled ``url_pattern`` rules ``(pattern, params)`` from
            adopted lenses, min-composed per pattern.
    """

    default: UtilityParams | None
    source_type_rules: dict[str, UtilityParams]
    url_rules: tuple[tuple[re.Pattern[str], UtilityParams], ...]

    def resolve(self, source_type: str, uri_r: str | None) -> UtilityParams | None:
        """Return the effective utility params for a belief, most-specific-wins."""
        if uri_r is not None and not uri_r.startswith("file://"):
            best: UtilityParams | None = None
            for pattern, params in self.url_rules:
                if pattern.search(uri_r):
                    best = params if best is None else _more_skeptical(best, params)
            if best is not None:
                return best
        return self.source_type_rules.get(source_type, self.default)

    def bonus(self, reinforcement: float, source_type: str, uri_r: str | None) -> float:
        """Additive projection rank-lift for one belief under this composed policy.

        ``0.0`` when no rule resolves — the belief keeps its base position
        (promotion-only).
        """
        params = self.resolve(source_type, uri_r)
        if params is None:
            return 0.0
        return utility_rank_bonus(reinforcement, params[1])

    def half_life_uses_days(self, source_type: str, uri_r: str | None) -> float | None:
        """The reinforcement half-life for a belief, used to weight its event ages."""
        params = self.resolve(source_type, uri_r)
        return None if params is None else params[0]


#: Neutral policy — utility disabled or unconfigured (bonus +0 everywhere).
EMPTY_UTILITY_POLICY = UtilityPolicy(default=None, source_type_rules={}, url_rules=())


async def load_utility_policy(session: AsyncSession) -> UtilityPolicy:
    """Snapshot one store's utility policy: local config composed over adopted lenses.

    Returns :data:`EMPTY_UTILITY_POLICY` when ``utility.enabled`` is False — the
    caller then adds bonus 0.0 (pre-0190 ranking). Called once per render;
    the result is reused across every belief.
    """
    from particles.config import get_config

    cfg = get_config().utility
    if not cfg.enabled:
        return EMPTY_UTILITY_POLICY

    d = cfg.default
    default: UtilityParams = (d.half_life_uses_days, d.rank_lift)

    lens_source_type: dict[str, UtilityParams] = {}
    lens_url: dict[str, UtilityParams] = {}
    lens_default: UtilityParams | None = None
    for lens in await get_adopted_lenses(session):
        for rule in lens.utility_rules:
            params: UtilityParams = (rule.half_life_uses_days, rule.rank_lift)
            if rule.scope == "default":
                lens_default = (
                    params if lens_default is None else _more_skeptical(lens_default, params)
                )
            elif rule.scope == "source_type" and rule.pattern:
                current = lens_source_type.get(rule.pattern)
                lens_source_type[rule.pattern] = (
                    params if current is None else _more_skeptical(current, params)
                )
            elif rule.scope == "url_pattern" and rule.pattern:
                current = lens_url.get(rule.pattern)
                lens_url[rule.pattern] = (
                    params if current is None else _more_skeptical(current, params)
                )

    # Local config wins for the default scope; a lens default only applies where
    # local is silent — but local always sets a default, so a lens can only make
    # the default *more* skeptical if it is stricter than local's own base.
    if lens_default is not None:
        default = _more_skeptical(default, lens_default)

    url_rules: list[tuple[re.Pattern[str], UtilityParams]] = []
    for pattern, params in lens_url.items():
        with contextlib.suppress(re.error):
            url_rules.append((re.compile(pattern), params))

    return UtilityPolicy(
        default=default,
        source_type_rules=lens_source_type,
        url_rules=tuple(url_rules),
    )


async def apply_utility(
    session: AsyncSession,
    eff_by_id: dict[str, float],
    source_info: dict[str, tuple[str, str | None]],
    *,
    policy: UtilityPolicy | None = None,
) -> dict[str, float]:
    """Add the projection/digest utility rank-lift to an effective-confidence map.

    ``eff_by_id`` maps ``particle_id → effective_confidence`` (from the trust /
    decay formula); ``source_info`` maps ``particle_id →
    (source_type, uri_r)`` for scope resolution. Returns a new map
    ``rank_score = effective_confidence + λ·ln(1 + R)``. A belief with no utility
    evidence keeps its score unchanged (bonus ``+0``). Loads the policy and the
    reinforcement counts once for the whole set — no per-particle round trips.

    The returned values are **ranking scores, not confidences**: the additive
    bonus is unbounded above, so a score may exceed ``1.0``. Callers must use
    them for ordering only — a surface that *displays* a confidence must score
    separately with the utility bonus off (see ``operations/digest.py``).

    This is the single seam that keeps utility on the projection / digest path
    only: the semantic-search ``query`` path never calls it.
    """
    if not eff_by_id:
        return dict(eff_by_id)
    if policy is None:
        policy = await load_utility_policy(session)
    if policy is EMPTY_UTILITY_POLICY:
        return dict(eff_by_id)

    # One reinforcement half-life per belief (its resolved scope); read all scores
    # in a single pass. Beliefs share a half-life in the common one-rule case, so
    # group by half-life to minimise store reads.
    ids = list(eff_by_id)
    hl_by_id: dict[str, float] = {}
    for pid in ids:
        st, uri = source_info.get(pid, ("", None))
        hl = policy.half_life_uses_days(st, uri)
        if hl is not None:
            hl_by_id[pid] = hl
    scores: dict[str, float] = {}
    by_hl: dict[float, list[str]] = {}
    for pid, hl in hl_by_id.items():
        by_hl.setdefault(hl, []).append(pid)
    for hl, pids in by_hl.items():
        scores.update(await get_reinforcement_scores(session, pids, hl))

    out: dict[str, float] = {}
    for pid, eff in eff_by_id.items():
        st, uri = source_info.get(pid, ("", None))
        out[pid] = eff + policy.bonus(scores.get(pid, 0.0), st, uri)
    return out
