"""Tests: content age decay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.scoring.decay import recency_factor


class TestRecencyFactor:
    def _now(self) -> datetime:
        return datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)

    def test_none_published_at_returns_one(self) -> None:
        assert recency_factor(None, "REDDIT_POST") == 1.0

    def test_unknown_source_type_returns_one(self) -> None:
        now = self._now()
        pub_at = now - timedelta(days=120)
        assert recency_factor(pub_at, "NUMISTA_API_COIN", now=now) == 1.0

    def test_fresh_post_near_one(self) -> None:
        now = self._now()
        pub_at = now - timedelta(days=1)
        factor = recency_factor(pub_at, "REDDIT_POST", now=now)
        # 1 day old with 60-day half-life: 0.5^(1/60) ≈ 0.9885
        assert factor == pytest.approx(0.9885, abs=0.001)

    def test_one_half_life_gives_half(self) -> None:
        now = self._now()
        pub_at = now - timedelta(days=60)
        factor = recency_factor(pub_at, "REDDIT_POST", now=now)
        assert factor == pytest.approx(0.5, abs=0.001)

    def test_two_half_lives_gives_quarter(self) -> None:
        now = self._now()
        pub_at = now - timedelta(days=120)
        factor = recency_factor(pub_at, "REDDIT_POST", now=now)
        assert factor == pytest.approx(0.25, abs=0.001)

    def test_five_months_roughly_eighteen_percent(self) -> None:
        now = self._now()
        pub_at = now - timedelta(days=150)
        factor = recency_factor(pub_at, "REDDIT_POST", now=now)
        # 0.5^(150/60) ≈ 0.177
        assert factor == pytest.approx(0.177, abs=0.005)

    def test_floor_applied_for_very_old_content(self) -> None:
        now = self._now()
        pub_at = now - timedelta(days=3650)  # ~10 years
        factor = recency_factor(pub_at, "REDDIT_POST", now=now)
        assert factor == pytest.approx(0.10)  # floor

    def test_future_published_at_returns_one(self) -> None:
        now = self._now()
        pub_at = now + timedelta(days=1)
        assert recency_factor(pub_at, "REDDIT_POST", now=now) == 1.0

    def test_naive_datetime_treated_as_utc(self) -> None:
        now = self._now()
        naive_pub_at = (now - timedelta(days=60)).replace(tzinfo=None)
        factor = recency_factor(naive_pub_at, "REDDIT_POST", now=now)
        assert factor == pytest.approx(0.5, abs=0.001)


class TestRecencyFactorInEffectiveConfidence:
    def test_recency_multiplied_in(self) -> None:
        # confidence=0.8, trust=0.4, recency=0.5 → 0.8*0.4*0.5 = 0.16
        eff = compute_effective_confidence(0.8, extractor_trust_weight=0.4, recency_factor=0.5)
        assert eff == pytest.approx(0.16)

    def test_no_decay_unchanged(self) -> None:
        eff = compute_effective_confidence(0.8, extractor_trust_weight=0.4, recency_factor=1.0)
        assert eff == pytest.approx(0.32)

    def test_clamped_to_one(self) -> None:
        eff = compute_effective_confidence(1.0, extractor_trust_weight=1.0, recency_factor=1.0)
        assert eff == pytest.approx(1.0)


class TestRedditImporterPublishedAt:
    """Integration: RedditImporter extracts created_utc from JSON blob."""

    @pytest.mark.asyncio
    async def test_deposit_sets_content_published_at(self, db_session: object) -> None:
        import json
        from unittest.mock import AsyncMock, patch

        from particles.ingest.importers.reddit import RedditImporter

        created_utc = 1777836000  # 2026-05-03 (approx)
        blob = [
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "subreddit": "test",
                                "title": "Test post",
                                "author": "testuser",
                                "score": 10,
                                "selftext": "",
                                "created_utc": created_utc,
                            },
                        }
                    ]
                }
            },
            {"data": {"children": []}},
        ]

        with patch(
            "particles.ingest.importers.reddit._fetch_with_curl",
            AsyncMock(return_value=json.dumps(blob).encode()),
        ):
            d = RedditImporter()
            _, snapshot_id = await d.deposit(
                db_session,  # type: ignore[arg-type]
                "https://www.reddit.com/r/test/comments/abc123/test_post/",
                "test-operator",
                [],
            )

        from particles.corpus.store import get_snapshot

        snap = await get_snapshot(db_session, snapshot_id)  # type: ignore[arg-type]
        assert snap is not None
        assert snap.content_published_at is not None
        assert snap.content_published_at.year == 2026
        assert snap.content_published_at.month == 5
