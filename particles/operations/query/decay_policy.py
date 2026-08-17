"""Query-time content-age decay policy.

The decay analogue of the ``TrustPolicy`` (``source_trust.py``): a
once-per-query snapshot resolving each candidate's recency ``(half_life, floor)``
from the store's **local** ``content_age_decay`` config composed over the
``decay_rules`` of every adopted trust lens. Like the trust snapshot, in
``query_federated`` it is loaded from the **viewer's** store and applied to
every store's candidates (the per-viewer rendering promised).

Composition, mirrored:

- **Two layers, URL is the more specific.** A ``url_pattern`` decay rule that
  matches the candidate's URI overrides the ``source_type`` default — in either
  direction, so a per-subreddit rule can make content *more* durable than its
  source-type default (a plain global ``min`` could only shorten it).
- **Local wins; across lenses, most-skeptical-wins.** The store's own
  ``content_age_decay`` config (source_type layer) wins for a key it configures;
  for keys it does not, adopted lenses contribute the shortest ``half_life_days``
  and lowest ``floor`` (each minimised independently — both make decay *more*
  skeptical). A key neither local config nor any lens asserts → no decay
  (factor 1.0), exactly as today.

With no decay-bearing lens adopted the snapshot *is* the global config, so every
existing store scores byte-for-byte identically (backward-compat).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.scoring.decay import recency_factor_from_params
from particles.store.lens_store import get_adopted_lenses

#: A resolved decay rule: (half_life_days, floor).
DecayParams = tuple[float, float]


@dataclass(frozen=True)
class DecayPolicy:
    """One store's content-age decay policy, snapshotted for in-process scoring.

    Attributes:
        source_type_rules: ``{source_type: (half_life_days, floor)}`` — local
            ``content_age_decay`` config (local-wins) plus the min-composed
            ``source_type`` decay rules of adopted lenses for keys local config
            does not set.
        url_rules: compiled ``url_pattern`` decay rules from adopted lenses,
            ``(pattern, half_life_days, floor)``, min-composed per pattern.
    """

    source_type_rules: dict[str, DecayParams]
    url_rules: tuple[tuple[re.Pattern[str], float, float], ...]

    def resolve(self, source_type: str, uri_r: str | None) -> DecayParams | None:
        """Return the effective ``(half_life_days, floor)`` for a candidate, or None.

        The URL layer is consulted first and wins when it matches (most
        specific); across multiple matching URL rules the most-skeptical
        (shortest half-life / lowest floor) applies. Otherwise the
        ``source_type`` layer answers; a ``None`` return means "no decay" → the
        caller applies factor 1.0.
        """
        if uri_r is not None and not uri_r.startswith("file://"):
            best: DecayParams | None = None
            for pattern, half_life, floor in self.url_rules:
                if pattern.search(uri_r):
                    best = (
                        (half_life, floor)
                        if best is None
                        else (min(best[0], half_life), min(best[1], floor))
                    )
            if best is not None:
                return best
        return self.source_type_rules.get(source_type)

    def recency_factor(
        self,
        content_published_at: datetime | None,
        source_type: str,
        uri_r: str | None,
        now: datetime | None = None,
    ) -> float:
        """Recency multiplier for one candidate under this composed decay policy."""
        params = self.resolve(source_type, uri_r)
        if params is None:
            return 1.0
        return recency_factor_from_params(content_published_at, params[0], params[1], now)


#: Neutral policy — no source decays (factor 1.0 everywhere).
EMPTY_DECAY_POLICY = DecayPolicy(source_type_rules={}, url_rules=())


async def load_decay_policy(session: AsyncSession) -> DecayPolicy:
    """Snapshot one store's decay policy: local config composed over adopted lenses.

    Called once per query — from the queried store in ``query``, from the
    **viewer's** store in ``query_federated``. The result is reused
    across every candidate, so per-particle scoring stays free of DB round trips.
    """
    from particles.config import get_config

    cfg = get_config().content_age_decay
    source_type_rules: dict[str, DecayParams] = {
        source_type: (c.half_life_days, c.floor) for source_type, c in cfg.sources.items()
    }

    # overlay adopted lenses under the local config.
    lens_source_type: dict[str, DecayParams] = {}
    lens_url: dict[str, DecayParams] = {}
    for lens in await get_adopted_lenses(session):
        for rule in lens.decay_rules:
            target = lens_source_type if rule.scope == "source_type" else lens_url
            current = target.get(rule.pattern)
            params = (rule.half_life_days, rule.floor)
            target[rule.pattern] = (
                params
                if current is None
                else (min(current[0], params[0]), min(current[1], params[1]))
            )

    # Local config wins per source_type; lenses fill the keys it does not set.
    for source_type, params in lens_source_type.items():
        source_type_rules.setdefault(source_type, params)

    url_rules: list[tuple[re.Pattern[str], float, float]] = []
    for pattern, (half_life, floor) in lens_url.items():
        with contextlib.suppress(re.error):
            url_rules.append((re.compile(pattern), half_life, floor))

    return DecayPolicy(source_type_rules=source_type_rules, url_rules=tuple(url_rules))
