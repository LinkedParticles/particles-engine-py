"""Tests for the per-observer decay policy (decay as a 4th lens layer)."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

import pytest

from particles.operations.query.decay_policy import (
    EMPTY_DECAY_POLICY,
    DecayPolicy,
    load_decay_policy,
)

# A fixed "now" and a publication date exactly one REDDIT_POST half-life (60d) old.
_NOW = datetime(2026, 6, 1, tzinfo=UTC)
_PUB_60D = datetime(2026, 4, 2, tzinfo=UTC)  # 60 days before _NOW


def _url_rule(pattern: str, half_life: float, floor: float) -> tuple[re.Pattern[str], float, float]:
    return (re.compile(pattern), half_life, floor)


class TestResolve:
    """The two-tier ladder: URL-most-specific, then source_type."""

    def test_empty_policy_never_decays(self) -> None:
        assert EMPTY_DECAY_POLICY.resolve("REDDIT_POST", None) is None
        assert EMPTY_DECAY_POLICY.recency_factor(_PUB_60D, "REDDIT_POST", None, now=_NOW) == 1.0

    def test_source_type_layer(self) -> None:
        policy = DecayPolicy(source_type_rules={"REDDIT_POST": (60.0, 0.10)}, url_rules=())
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/x") == (60.0, 0.10)
        # Unconfigured source type → no decay.
        assert policy.resolve("PDF", None) is None

    def test_url_rule_overrides_source_type_either_direction(self) -> None:
        # The owner's case: a subreddit MORE durable than the REDDIT_POST default.
        policy = DecayPolicy(
            source_type_rules={"REDDIT_POST": (60.0, 0.10)},
            url_rules=(
                _url_rule(r"reddit\.com/r/AskHistorians", 1825.0, 0.50),
                _url_rule(r"reddit\.com/r/wallstreetbets", 7.0, 0.05),
            ),
        )
        # URL is more specific than source_type → it wins, in either direction.
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/AskHistorians/x") == (
            1825.0,
            0.50,
        )
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/wallstreetbets/y") == (7.0, 0.05)
        # A REDDIT_POST whose URL matches no rule falls back to the source_type default.
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/pics/z") == (60.0, 0.10)

    def test_url_layer_takes_most_skeptical_across_matching_rules(self) -> None:
        policy = DecayPolicy(
            source_type_rules={},
            url_rules=(
                _url_rule(r"reddit\.com", 90.0, 0.30),
                _url_rule(r"/r/wallstreetbets", 7.0, 0.05),
            ),
        )
        # Both rules match this URI → shortest half-life + lowest floor win.
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/wallstreetbets/x") == (7.0, 0.05)

    def test_file_uri_skips_url_layer(self) -> None:
        policy = DecayPolicy(
            source_type_rules={"REDDIT_POST": (60.0, 0.10)},
            url_rules=(_url_rule(r"x", 5.0, 0.01),),
        )
        # file:// sources only consult the source_type layer.
        assert policy.resolve("REDDIT_POST", "file:///tmp/x") == (60.0, 0.10)


class TestRecencyFactor:
    def test_one_half_life_halves(self) -> None:
        policy = DecayPolicy(source_type_rules={"REDDIT_POST": (60.0, 0.10)}, url_rules=())
        rf = policy.recency_factor(_PUB_60D, "REDDIT_POST", None, now=_NOW)
        assert math.isclose(rf, 0.5, abs_tol=1e-9)

    def test_no_rule_is_no_decay(self) -> None:
        policy = DecayPolicy(source_type_rules={"REDDIT_POST": (60.0, 0.10)}, url_rules=())
        assert policy.recency_factor(_PUB_60D, "PDF", None, now=_NOW) == 1.0

    def test_floor_is_respected(self) -> None:
        # 10 half-lives → 0.5**10 ≈ 0.001, clamped up to the floor 0.25.
        policy = DecayPolicy(source_type_rules={"REDDIT_POST": (6.0, 0.25)}, url_rules=())
        rf = policy.recency_factor(_PUB_60D, "REDDIT_POST", None, now=_NOW)
        assert math.isclose(rf, 0.25, abs_tol=1e-9)


class TestLoadDecayPolicy:
    """Composition from local config + adopted lenses (DB-backed)."""

    @staticmethod
    async def _adopt_decay_lens(
        session: object, name: str, version: int, decay_rules: list[dict[str, object]]
    ) -> None:
        from particles.core.schema import TrustLensDecayRule, TrustLensDefinition
        from particles.store.lens_store import adopt_lens, materialise_lens

        lens = TrustLensDefinition(
            name=name,
            version=version,
            decay_rules=[TrustLensDecayRule(**r) for r in decay_rules],  # type: ignore[arg-type]
        )
        await materialise_lens(session, lens)  # type: ignore[arg-type]
        await adopt_lens(session, name)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_no_lens_matches_global_config(self, db_session: object) -> None:
        # Backward-compat: with no decay-bearing lens, the policy IS the config.
        policy = await load_decay_policy(db_session)  # type: ignore[arg-type]
        assert policy.resolve("REDDIT_POST", None) == (60.0, 0.10)  # config default
        assert policy.url_rules == ()

    @pytest.mark.asyncio
    async def test_local_config_wins_over_lens_for_source_type(self, db_session: object) -> None:
        # A lens trying to LENGTHEN REDDIT_POST is ignored — local config wins.
        await self._adopt_decay_lens(
            db_session,
            "slow-reddit",
            1,
            [
                {
                    "scope": "source_type",
                    "pattern": "REDDIT_POST",
                    "half_life_days": 999.0,
                    "floor": 0.9,
                }
            ],
        )
        policy = await load_decay_policy(db_session)  # type: ignore[arg-type]
        assert policy.resolve("REDDIT_POST", None) == (60.0, 0.10)  # local config, not the lens

    @pytest.mark.asyncio
    async def test_lens_fills_unconfigured_source_type(self, db_session: object) -> None:
        # WEB_PAGE has no local decay config → an adopted lens supplies it.
        await self._adopt_decay_lens(
            db_session,
            "web-decay",
            1,
            [
                {
                    "scope": "source_type",
                    "pattern": "WEB_PAGE",
                    "half_life_days": 180.0,
                    "floor": 0.10,
                }
            ],
        )
        policy = await load_decay_policy(db_session)  # type: ignore[arg-type]
        assert policy.resolve("WEB_PAGE", None) == (180.0, 0.10)

    @pytest.mark.asyncio
    async def test_lens_url_rule_overrides_source_type_default(self, db_session: object) -> None:
        # The flagship case: a per-subreddit lens rule, more durable than the
        # REDDIT_POST default, reaches the resolved policy via the URL layer.
        await self._adopt_decay_lens(
            db_session,
            "subreddit-decay",
            1,
            [
                {
                    "scope": "url_pattern",
                    "pattern": r"reddit\.com/r/AskHistorians",
                    "half_life_days": 1825.0,
                    "floor": 0.50,
                }
            ],
        )
        policy = await load_decay_policy(db_session)  # type: ignore[arg-type]
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/AskHistorians/abc") == (
            1825.0,
            0.50,
        )
        # A different subreddit still gets the source_type default.
        assert policy.resolve("REDDIT_POST", "https://reddit.com/r/pics/abc") == (60.0, 0.10)
