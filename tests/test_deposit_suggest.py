"""Tests for citation-signal deposit-suggestion ranking + dismiss."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from particles.operations.deposit_suggest import (
    DepositSuggestReport,
    dismiss_suggestion,
    suggest_deposits,
)
from particles.store.url_mention_store import record_url_mentions

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _add_entry(session: AsyncSession, entry_id: str, uri_r: str | None) -> None:
    from particles.corpus.store import CorpusEntryRow

    session.add(
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri_r,
            source_type="WEB_PAGE",
            mutability="MUTABLE",
            fetch_policy="LAZY",
            created_at=datetime.now(UTC),
            deposited_by="test",
        )
    )
    await session.flush()


async def _add_domain_trust(session: AsyncSession, netloc: str, score: float) -> None:
    from particles.store.trust_store import SourceTrustRow

    session.add(
        SourceTrustRow(
            scope="domain",
            pattern=netloc,
            score=score,
            created_at=datetime.now(UTC),
            asserted_by="test",
        )
    )
    await session.flush()


class TestRankingGate:
    async def test_min_sources_gate_filters_single_source(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "s1", "https://s1.example/a")
        # Cited by only one distinct source — below the default min of 2.
        await record_url_mentions(
            db_session, source_entry_id="s1", canonical_urls=["https://press.example/x"]
        )
        report = await suggest_deposits(db_session)
        assert report.suggestions == []

    async def test_two_distinct_sources_surface(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "s1", "https://s1.example/a")
        await _add_entry(db_session, "s2", "https://s2.example/b")
        for s in ("s1", "s2"):
            await record_url_mentions(
                db_session, source_entry_id=s, canonical_urls=["https://press.example/x"]
            )
        report = await suggest_deposits(db_session)
        assert len(report.suggestions) == 1
        sug = report.suggestions[0]
        assert sug.canonical_url == "https://press.example/x"
        assert sug.distinct_sources == 2
        assert sorted(sug.citing_entry_ids) == ["s1", "s2"]

    async def test_diversity_beats_frequency(self, db_session: AsyncSession) -> None:
        # URL A: 3 distinct sources. URL B: one source repeating it (still 1).
        for s in ("s1", "s2", "s3"):
            await _add_entry(db_session, s, f"https://{s}.example/p")
        await record_url_mentions(
            db_session,
            source_entry_id="s1",
            canonical_urls=["https://a.example/x", "https://b.example/y"],
        )
        await record_url_mentions(
            db_session, source_entry_id="s2", canonical_urls=["https://a.example/x"]
        )
        await record_url_mentions(
            db_session, source_entry_id="s3", canonical_urls=["https://a.example/x"]
        )
        report = await suggest_deposits(db_session, min_sources=1)
        ranked = [s.canonical_url for s in report.suggestions]
        # a.example/x (3 sources) outranks b.example/y (1 source).
        assert ranked[0] == "https://a.example/x"
        assert ranked.index("https://a.example/x") < ranked.index("https://b.example/y")


class TestTrustWeighting:
    async def test_low_trust_sources_score_lower(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "hi", "https://trusted.example/a")
        await _add_entry(db_session, "lo", "https://spam.example/b")
        await _add_domain_trust(db_session, "trusted.example", 0.9)
        await _add_domain_trust(db_session, "spam.example", 0.1)
        await record_url_mentions(
            db_session, source_entry_id="hi", canonical_urls=["https://good.example/x"]
        )
        await record_url_mentions(
            db_session, source_entry_id="lo", canonical_urls=["https://meh.example/y"]
        )
        report = await suggest_deposits(db_session, min_sources=1)
        scores = {s.canonical_url: s.score for s in report.suggestions}
        assert scores["https://good.example/x"] > scores["https://meh.example/y"]


class TestSiteInternalFilter:
    async def test_same_host_only_is_filtered(self, db_session: AsyncSession) -> None:
        # news.example's own pages all link news.example/about — site chrome.
        await _add_entry(db_session, "p1", "https://news.example/article-1")
        await _add_entry(db_session, "p2", "https://news.example/article-2")
        for s in ("p1", "p2"):
            await record_url_mentions(
                db_session, source_entry_id=s, canonical_urls=["https://news.example/about"]
            )
        report = await suggest_deposits(db_session)
        assert report.suggestions == []

    async def test_cross_site_citation_survives(self, db_session: AsyncSession) -> None:
        # The same URL cited from a *different* domain is a real signal.
        await _add_entry(db_session, "p1", "https://news.example/article-1")
        await _add_entry(db_session, "p2", "https://reddit.example/thread")
        for s in ("p1", "p2"):
            await record_url_mentions(
                db_session, source_entry_id=s, canonical_urls=["https://news.example/about"]
            )
        report = await suggest_deposits(db_session)
        assert [s.canonical_url for s in report.suggestions] == ["https://news.example/about"]


class TestSuppressionAndCap:
    async def test_dismissed_url_excluded(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "s1", "https://s1.example/a")
        await _add_entry(db_session, "s2", "https://s2.example/b")
        for s in ("s1", "s2"):
            await record_url_mentions(
                db_session, source_entry_id=s, canonical_urls=["https://press.example/x"]
            )
        await dismiss_suggestion(db_session, canonical_url="https://press.example/x", actor="test")
        report = await suggest_deposits(db_session)
        assert report.suggestions == []

    async def test_rank_cap_sets_capped_flag(self, db_session: AsyncSession) -> None:
        await _add_entry(db_session, "s1", "https://s1.example/a")
        await _add_entry(db_session, "s2", "https://s2.example/b")
        for i in range(5):
            for s in ("s1", "s2"):
                await record_url_mentions(
                    db_session, source_entry_id=s, canonical_urls=[f"https://press.example/x{i}"]
                )
        report = await suggest_deposits(db_session, limit=3)
        assert isinstance(report, DepositSuggestReport)
        assert len(report.suggestions) == 3
        assert report.total_candidates == 5
        assert report.capped is True


class TestDismiss:
    async def test_permanent_dismiss_records_event(self, db_session: AsyncSession) -> None:
        from particles.store.event_store import OperatorEventType, list_events

        until = await dismiss_suggestion(
            db_session, canonical_url="https://a.example/x", actor="cli:corpus-links-dismiss"
        )
        assert until.year == 9999
        events = await list_events(
            db_session, event_type=OperatorEventType.DEPOSIT_SUGGESTION_DISMISSED
        )
        assert len(events) == 1
        assert events[0].payload is not None
        assert events[0].payload["canonical_url"] == "https://a.example/x"
        assert events[0].payload["snooze_days"] is None

    async def test_snooze_sets_near_window(self, db_session: AsyncSession) -> None:
        until = await dismiss_suggestion(
            db_session, canonical_url="https://a.example/x", actor="test", snooze_days=7
        )
        delta = until - datetime.now(UTC)
        assert timedelta(days=6) < delta <= timedelta(days=7)
