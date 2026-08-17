"""Tests for the §9.5 Reindex operation (particles/operations/reindex.py).

reindex.py had 0% coverage in the architecture-review baseline despite being
a destructive write path: it supersedes existing ACTIVE particles after
re-extracting. These tests cover scope discovery, the supersession contract
("re-extract first, then supersede"), and the failure/empty-scope envelopes.

extract_snapshot is mocked throughout — these are unit tests, not integration
tests against the LLM.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.operations.reindex import _identify_scope, _reindex_snapshot, reindex
from particles.store.particle_store import get_particle, insert_particle

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _add_entry(
    session: Any,
    *,
    source_type: str = "WEB_PAGE",
    uri_r: str | None = "https://example.com/x",
) -> CorpusEntry:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type=source_type,
        uri_r=uri_r,
        deposited_by="test",
    )
    session.add(CorpusEntryRow.from_model(entry))
    await session.flush()
    return entry


async def _add_snapshot(
    session: Any,
    entry: CorpusEntry,
    *,
    extraction_status: ExtractionStatus = ExtractionStatus.COMPLETE,
    captured_at: datetime | None = None,
    content_hash: str | None = None,
) -> Snapshot:
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=captured_at or datetime.now(UTC),
        content_hash=content_hash or f"hash-{uuid.uuid4().hex[:16]}",
        extraction_status=extraction_status,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    session.add(SnapshotRow.from_model(snap, entry.entry_id))
    await session.flush()
    return snap


async def _add_particle(
    session: Any,
    entry: CorpusEntry,
    snap: Snapshot,
    *,
    content: str = "Test claim.",
    schema_version: str = SCHEMA_VERSION,
    extractor_version: str = "0.3.0",
    extractor_id: str = "general-extractor",
    extraction_provider_model: str | None = None,
) -> Particle:
    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        schema_version=schema_version,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry.entry_id,
                snapshot_id=snap.snapshot_id,
            )
        ],
        extractor_ref={"name": extractor_id, "version": extractor_version},
        extraction_provider_model=extraction_provider_model,
    )
    await insert_particle(session, p)
    return p


def _patch_extract(
    monkeypatch: pytest.MonkeyPatch,
    return_value: list[Particle] | None = None,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Replace operations.reindex.extract_snapshot with a controllable mock."""
    mock = AsyncMock(return_value=return_value if return_value is not None else [])
    if side_effect is not None:
        mock.side_effect = side_effect
    monkeypatch.setattr("particles.operations.reindex.extract_snapshot", mock)
    return mock


# ---------------------------------------------------------------------------
# _identify_scope — explicit entry_ids
# ---------------------------------------------------------------------------


class TestIdentifyScopeExplicit:
    @pytest.mark.asyncio
    async def test_full_uuid_resolves_to_latest_complete_snapshot(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap_old = await _add_snapshot(
            db_session, entry, captured_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
        snap_new = await _add_snapshot(
            db_session, entry, captured_at=datetime(2026, 5, 1, tzinfo=UTC)
        )
        await db_session.commit()

        scope = await _identify_scope(db_session, [entry.entry_id], None, None, False)
        assert scope == [(entry.entry_id, snap_new.snapshot_id)]
        assert snap_old.snapshot_id != snap_new.snapshot_id

    @pytest.mark.asyncio
    async def test_prefix_unique_match(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await db_session.commit()

        prefix = entry.entry_id[:8]
        scope = await _identify_scope(db_session, [prefix], None, None, False)
        assert scope == [(entry.entry_id, snap.snapshot_id)]

    @pytest.mark.asyncio
    async def test_prefix_no_match_is_skipped(self, db_session: Any) -> None:
        scope = await _identify_scope(db_session, ["deadbeef"], None, None, False)
        assert scope == []

    @pytest.mark.asyncio
    async def test_entry_without_complete_snapshot_is_skipped(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.PENDING)
        await db_session.commit()

        scope = await _identify_scope(db_session, [entry.entry_id], None, None, False)
        assert scope == []  # no COMPLETE snapshot → not reindexable

    @pytest.mark.asyncio
    async def test_repeated_entry_id_yields_one_pair(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [entry.entry_id, entry.entry_id[:8]], None, None, False
        )
        assert scope == [(entry.entry_id, snap.snapshot_id)]


# ---------------------------------------------------------------------------
# _identify_scope — explicit entry_ids AND a particle-matching flag
# ---------------------------------------------------------------------------


class TestIdentifyScopeExplicitIntersection:
    """The named-entry scope intersects with the particle-matching flags.

    Before the explicit branch returned early and discarded
    ``--extractor-version`` / ``--extractor-id`` / ``--provider-model``
    silently, handing the operator a *wider* scope than they asked for on a
    verb that supersedes particles.
    """

    @pytest.mark.asyncio
    async def test_named_entry_not_matching_provider_model_is_dropped(
        self, db_session: Any
    ) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, extraction_provider_model=None)
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [entry.entry_id], None, None, False, "openai:gpt-5.6-luna"
        )
        assert scope == []
        # …and the unfiltered call still selects it, so the entry itself is fine.
        assert await _identify_scope(db_session, [entry.entry_id], None, None, False) == [
            (entry.entry_id, snap.snapshot_id)
        ]

    @pytest.mark.asyncio
    async def test_named_entry_matching_provider_model_is_kept(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(
            db_session, entry, snap, extraction_provider_model="openai:gpt-5.6-luna"
        )
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [entry.entry_id], None, None, False, "openai:gpt-5.6-luna"
        )
        assert scope == [(entry.entry_id, snap.snapshot_id)]

    @pytest.mark.asyncio
    async def test_intersection_selects_the_named_subset(self, db_session: Any) -> None:
        """Two entries share a pairing; only the named one is reindexed."""
        named = await _add_entry(db_session)
        named_snap = await _add_snapshot(db_session, named)
        await _add_particle(
            db_session, named, named_snap, extraction_provider_model="openai:gpt-5.6-luna"
        )
        other = await _add_entry(db_session)
        other_snap = await _add_snapshot(db_session, other)
        await _add_particle(
            db_session, other, other_snap, extraction_provider_model="openai:gpt-5.6-luna"
        )
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [named.entry_id], None, None, False, "openai:gpt-5.6-luna"
        )
        assert scope == [(named.entry_id, named_snap.snapshot_id)]

    @pytest.mark.asyncio
    async def test_extractor_version_narrows_named_entries(self, db_session: Any) -> None:
        match = await _add_entry(db_session)
        match_snap = await _add_snapshot(db_session, match)
        await _add_particle(db_session, match, match_snap, extractor_version="0.1.0")
        miss = await _add_entry(db_session)
        miss_snap = await _add_snapshot(db_session, miss)
        await _add_particle(db_session, miss, miss_snap, extractor_version="0.3.0")
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [match.entry_id, miss.entry_id], "0.1.0", None, False
        )
        assert scope == [(match.entry_id, match_snap.snapshot_id)]
        assert (miss.entry_id, miss_snap.snapshot_id) not in scope

    @pytest.mark.asyncio
    async def test_extractor_id_narrows_named_entries(self, db_session: Any) -> None:
        match = await _add_entry(db_session)
        match_snap = await _add_snapshot(db_session, match)
        await _add_particle(db_session, match, match_snap, extractor_id="github-repo-extractor")
        miss = await _add_entry(db_session)
        miss_snap = await _add_snapshot(db_session, miss)
        await _add_particle(db_session, miss, miss_snap, extractor_id="general-extractor")
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [match.entry_id, miss.entry_id], None, "github-repo-extractor", False
        )
        assert scope == [(match.entry_id, match_snap.snapshot_id)]

    @pytest.mark.asyncio
    async def test_multiple_flags_union_before_intersecting(self, db_session: Any) -> None:
        """The particle-matching flags OR with each other, then AND with the entries."""
        by_version = await _add_entry(db_session)
        version_snap = await _add_snapshot(db_session, by_version)
        await _add_particle(db_session, by_version, version_snap, extractor_version="0.1.0")
        by_model = await _add_entry(db_session)
        model_snap = await _add_snapshot(db_session, by_model)
        await _add_particle(
            db_session, by_model, model_snap, extraction_provider_model="openai:gpt-5.6-luna"
        )
        neither = await _add_entry(db_session)
        neither_snap = await _add_snapshot(db_session, neither)
        await _add_particle(db_session, neither, neither_snap)
        await db_session.commit()

        scope = await _identify_scope(
            db_session,
            [by_version.entry_id, by_model.entry_id, neither.entry_id],
            "0.1.0",
            None,
            False,
            "openai:gpt-5.6-luna",
        )
        assert set(scope) == {
            (by_version.entry_id, version_snap.snapshot_id),
            (by_model.entry_id, model_snap.snapshot_id),
        }
        assert (neither.entry_id, neither_snap.snapshot_id) not in scope

    @pytest.mark.asyncio
    async def test_intersection_is_over_pairs_not_entries(self, db_session: Any) -> None:
        """A match on an *older* snapshot does not put the entry's latest in scope.

        Supersession is computed per snapshot, so the intersection is over
        ``(entry_id, snapshot_id)`` pairs. The named entry resolves to its
        latest COMPLETE snapshot; the pairing only ever produced particles for
        the older one, so there is nothing here the flag asked for.
        """
        entry = await _add_entry(db_session)
        snap_old = await _add_snapshot(
            db_session, entry, captured_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
        await _add_snapshot(db_session, entry, captured_at=datetime(2026, 5, 1, tzinfo=UTC))
        await _add_particle(
            db_session, entry, snap_old, extraction_provider_model="openai:gpt-5.6-luna"
        )
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [entry.entry_id], None, None, False, "openai:gpt-5.6-luna"
        )
        assert scope == []

    @pytest.mark.asyncio
    async def test_stale_schema_does_not_re_widen_a_filtered_entry(self, db_session: Any) -> None:
        """The store-wide auto-discovery unions stay bypassed on the named path."""
        named = await _add_entry(db_session)
        named_snap = await _add_snapshot(db_session, named)
        # Stale schema AND no matching pairing: the flag must still exclude it.
        await _add_particle(db_session, named, named_snap, schema_version="0.2.0")
        unnamed = await _add_entry(db_session)
        unnamed_snap = await _add_snapshot(db_session, unnamed)
        await _add_particle(db_session, unnamed, unnamed_snap, schema_version="0.2.0")
        await db_session.commit()

        scope = await _identify_scope(
            db_session, [named.entry_id], None, None, True, "openai:gpt-5.6-luna"
        )
        assert scope == []
        assert (unnamed.entry_id, unnamed_snap.snapshot_id) not in scope

    @pytest.mark.asyncio
    async def test_narrowing_is_reported_to_the_progress_callback(self, db_session: Any) -> None:
        """Narrowing is safe, but it must not be silent either."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap)
        await db_session.commit()

        lines: list[str] = []
        scope = await _identify_scope(
            db_session,
            [entry.entry_id],
            None,
            None,
            False,
            "openai:gpt-5.6-luna",
            progress=lines.append,
        )
        assert scope == []
        assert any("0 of 1 named entries" in line for line in lines)
        assert any("openai:gpt-5.6-luna" in line for line in lines)

    @pytest.mark.asyncio
    async def test_no_narrowing_notice_when_every_entry_matches(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(
            db_session, entry, snap, extraction_provider_model="openai:gpt-5.6-luna"
        )
        await db_session.commit()

        lines: list[str] = []
        await _identify_scope(
            db_session,
            [entry.entry_id],
            None,
            None,
            False,
            "openai:gpt-5.6-luna",
            progress=lines.append,
        )
        assert lines == []


# ---------------------------------------------------------------------------
# _identify_scope — auto-discovery
# ---------------------------------------------------------------------------


class TestIdentifyScopeAuto:
    @pytest.mark.asyncio
    async def test_include_failed_picks_up_failed_and_pending(self, db_session: Any) -> None:
        entry_failed = await _add_entry(db_session)
        snap_failed = await _add_snapshot(
            db_session, entry_failed, extraction_status=ExtractionStatus.FAILED
        )
        entry_pending = await _add_entry(db_session)
        snap_pending = await _add_snapshot(
            db_session, entry_pending, extraction_status=ExtractionStatus.PENDING
        )
        entry_complete = await _add_entry(db_session)
        await _add_snapshot(db_session, entry_complete, extraction_status=ExtractionStatus.COMPLETE)
        await db_session.commit()

        scope = await _identify_scope(db_session, None, None, None, include_failed=True)
        assert (entry_failed.entry_id, snap_failed.snapshot_id) in scope
        assert (entry_pending.entry_id, snap_pending.snapshot_id) in scope
        # COMPLETE entries are NOT auto-included by include_failed
        assert not any(e == entry_complete.entry_id for e, _ in scope)

    @pytest.mark.asyncio
    async def test_include_failed_false_excludes_failed_and_pending(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        scope = await _identify_scope(db_session, None, None, None, include_failed=False)
        assert scope == []

    @pytest.mark.asyncio
    async def test_extractor_version_picks_up_matching_particles(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, extractor_version="0.1.0")
        await db_session.commit()

        scope = await _identify_scope(db_session, None, "0.1.0", None, include_failed=False)
        assert (entry.entry_id, snap.snapshot_id) in scope

    @pytest.mark.asyncio
    async def test_stale_schema_version_picks_up_old_particles(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, schema_version="0.2.0")
        await db_session.commit()

        scope = await _identify_scope(db_session, None, None, None, include_failed=False)
        assert (entry.entry_id, snap.snapshot_id) in scope

    @pytest.mark.asyncio
    async def test_extractor_id_picks_up_matching_particles(self, db_session: Any) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, extractor_id="github-repo-extractor")
        other_entry = await _add_entry(db_session)
        other_snap = await _add_snapshot(db_session, other_entry)
        await _add_particle(db_session, other_entry, other_snap, extractor_id="reddit-extractor")
        await db_session.commit()

        scope = await _identify_scope(
            db_session, None, None, "github-repo-extractor", include_failed=False
        )
        assert (entry.entry_id, snap.snapshot_id) in scope
        assert (other_entry.entry_id, other_snap.snapshot_id) not in scope

    @pytest.mark.asyncio
    async def test_scope_is_deduplicated(self, db_session: Any) -> None:
        """A snapshot that matches both extractor-version and stale-schema is included once."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(
            db_session, entry, snap, schema_version="0.2.0", extractor_version="0.1.0"
        )
        await db_session.commit()

        scope = await _identify_scope(db_session, None, "0.1.0", None, include_failed=False)
        assert scope.count((entry.entry_id, snap.snapshot_id)) == 1


# ---------------------------------------------------------------------------
# _reindex_snapshot — supersession contract
# ---------------------------------------------------------------------------


class TestReindexSnapshot:
    @pytest.mark.asyncio
    async def test_supersedes_only_particles_from_this_snapshot(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        snap_this = await _add_snapshot(db_session, entry, content_hash="this")
        snap_other = await _add_snapshot(db_session, entry, content_hash="other")
        p_this = await _add_particle(db_session, entry, snap_this, content="from this snap")
        p_other = await _add_particle(db_session, entry, snap_other, content="from other snap")
        await db_session.commit()

        _patch_extract(monkeypatch, return_value=[])
        await _reindex_snapshot(db_session, entry.entry_id, snap_this.snapshot_id, None)

        # The particle attached to snap_this is now SUPERSEDED; the other is untouched.
        after_this = await get_particle(db_session, p_this.id)
        after_other = await get_particle(db_session, p_other.id)
        assert after_this is not None and after_this.status == Status.SUPERSEDED
        assert after_other is not None and after_other.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_extract_failure_leaves_existing_active(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 'extract first, then supersede' contract: a failed extract must
        not leave the entry with no ACTIVE particles."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        p = await _add_particle(db_session, entry, snap)
        await db_session.commit()

        _patch_extract(monkeypatch, side_effect=RuntimeError("LLM API down"))
        with pytest.raises(RuntimeError):
            await _reindex_snapshot(db_session, entry.entry_id, snap.snapshot_id, None)

        # Old particle is still ACTIVE — supersession only happens on success.
        unchanged = await get_particle(db_session, p.id)
        assert unchanged is not None and unchanged.status == Status.ACTIVE


# ---------------------------------------------------------------------------
# reindex — end-to-end summary
# ---------------------------------------------------------------------------


class TestReindexSummary:
    @pytest.mark.asyncio
    async def test_empty_scope_returns_zeros(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_extract(monkeypatch)
        result = await reindex(db_session, run_post_lint=False)
        assert result == {
            "dry_run": False,
            "scope": 0,
            "succeeded": 0,
            "failed": 0,
            "failed_entries": [],
            "lint_summary": {},
            "plan": {
                "entries": 0,
                "snapshots": 0,
                "particles": 0,
                "missing_blobs": 0,
                "scope_description": "auto: stale schema + failed/pending",
                "snapshot_plans": [],
            },
        }

    @pytest.mark.asyncio
    async def test_successful_reindex_counts(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        mock = _patch_extract(monkeypatch, return_value=[])
        result = await reindex(db_session, run_post_lint=False, rate_limit_per_minute=0)
        assert result["scope"] == 1
        assert result["succeeded"] == 1
        assert result["failed"] == 0
        assert result["failed_entries"] == []
        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_extract_recorded_in_summary(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        _patch_extract(monkeypatch, side_effect=RuntimeError("boom"))
        result = await reindex(db_session, run_post_lint=False, rate_limit_per_minute=0)
        assert result["scope"] == 1
        assert result["succeeded"] == 0
        assert result["failed"] == 1
        assert result["failed_entries"] == [entry.entry_id]

    @pytest.mark.asyncio
    async def test_post_lint_skipped_when_disabled(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        _patch_extract(monkeypatch, return_value=[])
        lint_mock = AsyncMock()
        monkeypatch.setattr("particles.operations.reindex.run_lint", lint_mock)

        await reindex(db_session, run_post_lint=False, rate_limit_per_minute=0)
        lint_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_lint_skipped_when_nothing_succeeded(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        _patch_extract(monkeypatch, side_effect=RuntimeError("boom"))
        lint_mock = AsyncMock()
        monkeypatch.setattr("particles.operations.reindex.run_lint", lint_mock)

        await reindex(db_session, run_post_lint=True, rate_limit_per_minute=0)
        lint_mock.assert_not_called()  # nothing succeeded → no point linting

    @pytest.mark.asyncio
    async def test_entry_ids_plus_unmatched_filter_extracts_nothing(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """end to end: the filter reaches the destructive path.

        The named entry is perfectly reindexable, so before the fix this
        superseded its particles despite the pairing not matching.
        """
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, extraction_provider_model="anthropic:opus")
        await db_session.commit()

        mock = _patch_extract(monkeypatch, return_value=[])
        result = await reindex(
            db_session,
            entry_ids=[entry.entry_id],
            provider_model="openai:gpt-5.6-luna",
            run_post_lint=False,
            rate_limit_per_minute=0,
        )
        assert result["scope"] == 0
        mock.assert_not_called()
        assert snap.snapshot_id  # the entry was reindexable; only the filter excluded it

    @pytest.mark.asyncio
    async def test_progress_callback_reports_scope_and_per_entry(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verbose hook fires once for scope size and once per entry so
        operators can see a long-running reindex isn't stuck."""
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        _patch_extract(monkeypatch, return_value=[])
        lines: list[str] = []

        await reindex(
            db_session,
            run_post_lint=False,
            rate_limit_per_minute=0,
            progress=lines.append,
        )

        assert any(line.startswith("Reindex plan: 1 entries") for line in lines)
        assert any("[1/1] reindexing" in line for line in lines)
        assert any(entry.entry_id[:8] in line for line in lines)


# ---------------------------------------------------------------------------
# work plan + --dry-run (2026-08-02 full-store sweep incident)
# ---------------------------------------------------------------------------


class TestReindexPlanAndDryRun:
    """The upfront work plan and the dry-run contract.

    Motivating incident: a bare ``reindex --extractor-version`` silently swept
    the whole store — nothing reported how many entries / snapshots / particles
    were in scope before the first LLM call was spent.
    """

    @pytest.mark.asyncio
    async def test_dry_run_returns_plan_without_extracting(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, extractor_version="0.13.0")
        await db_session.commit()

        mock = _patch_extract(monkeypatch)
        result = await reindex(
            db_session,
            extractor_version="0.13.0",
            include_failed=False,
            run_post_lint=False,
            dry_run=True,
        )

        mock.assert_not_called()
        assert result["dry_run"] is True
        assert result["succeeded"] == 0
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["entries"] == 1
        assert plan["snapshots"] == 1
        assert plan["particles"] == 1
        assert plan["scope_description"] == "extractor-version 0.13.0; auto: stale schema"
        (sp,) = plan["snapshot_plans"]
        assert sp["entry_id"] == entry.entry_id
        assert sp["snapshot_id"] == snap.snapshot_id
        assert sp["particles"] == 1
        assert sp["uri"] == "https://example.com/x"

    @pytest.mark.asyncio
    async def test_dry_run_leaves_particles_active(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero writes: nothing is superseded on a dry run."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        p = await _add_particle(db_session, entry, snap)
        await db_session.commit()

        _patch_extract(monkeypatch)
        await reindex(db_session, entry_ids=[entry.entry_id], run_post_lint=False, dry_run=True)

        refreshed = await get_particle(db_session, p.id)
        assert refreshed is not None
        assert refreshed.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_plan_flags_missing_blob(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A snapshot whose blob is absent is flagged in the plan (the
        'Blob not found for hash' failures that previously surfaced only as
        mid-run ERROR logs). Test snapshots never write blobs, so the flag
        must be set; patching presence must clear it."""
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry)
        await db_session.commit()

        _patch_extract(monkeypatch)
        result = await reindex(
            db_session, entry_ids=[entry.entry_id], run_post_lint=False, dry_run=True
        )
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["missing_blobs"] == 1
        assert plan["snapshot_plans"][0]["blob_missing"] is True

        monkeypatch.setattr("particles.operations.reindex.blob_exists", lambda _h: True)
        result = await reindex(
            db_session, entry_ids=[entry.entry_id], run_post_lint=False, dry_run=True
        )
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["missing_blobs"] == 0
        assert plan["snapshot_plans"][0]["blob_missing"] is False

    @pytest.mark.asyncio
    async def test_on_plan_fires_before_extraction(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A live run emits the plan line (and missing-blob warnings) via
        ``on_plan`` before the first extract_snapshot call."""
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        events: list[str] = []

        async def record_extract(*args: Any, **kwargs: Any) -> list[Particle]:
            events.append("extract")
            return []

        mock = _patch_extract(monkeypatch)
        mock.side_effect = record_extract

        result = await reindex(
            db_session,
            run_post_lint=False,
            rate_limit_per_minute=0,
            on_plan=events.append,
        )

        assert result["succeeded"] == 1
        assert events[0].startswith("Reindex plan: 1 entries, 1 snapshots, 0 particles")
        assert any(e.startswith("  blob missing:") for e in events[1:])
        assert events[-1] == "extract"

    @pytest.mark.asyncio
    async def test_live_summary_carries_full_plan(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live envelope keeps the per-snapshot detail: the CLI's human
        rendering caps the missing-blob list and points at ``--format json``,
        so the envelope must actually carry the complete list."""
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        _patch_extract(monkeypatch)
        result = await reindex(db_session, run_post_lint=False, rate_limit_per_minute=0)

        assert result["dry_run"] is False
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["snapshots"] == 1
        assert len(plan["snapshot_plans"]) == 1

    @pytest.mark.asyncio
    async def test_missing_blob_lines_are_capped(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seven missing blobs emit five detail lines + one remainder line —
        a store with hundreds must not flood the terminal (the full list
        stays in the plan's ``snapshot_plans``)."""
        entry_ids = []
        for _ in range(7):
            entry = await _add_entry(db_session)
            await _add_snapshot(db_session, entry)
            entry_ids.append(entry.entry_id)
        await db_session.commit()

        _patch_extract(monkeypatch)
        events: list[str] = []
        result = await reindex(
            db_session,
            entry_ids=entry_ids,
            run_post_lint=False,
            dry_run=True,
            on_plan=events.append,
        )

        blob_lines = [e for e in events if e.startswith("  blob missing:")]
        assert len(blob_lines) == 5
        assert events[-1] == "  … and 2 more (see --format json)"
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert sum(1 for sp in plan["snapshot_plans"] if sp["blob_missing"]) == 7

    @pytest.mark.asyncio
    async def test_on_status_reports_position_after_each_snapshot(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry_a = await _add_entry(db_session)
        await _add_snapshot(db_session, entry_a, extraction_status=ExtractionStatus.FAILED)
        entry_b = await _add_entry(db_session)
        await _add_snapshot(db_session, entry_b, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        _patch_extract(monkeypatch, return_value=[])
        statuses: list[str] = []
        await reindex(
            db_session,
            run_post_lint=False,
            rate_limit_per_minute=0,
            on_status=statuses.append,
        )

        assert len(statuses) == 2
        assert statuses[0].startswith("snapshot 1/2 (entry ")
        assert statuses[1].startswith("snapshot 2/2 (entry ")
        assert not any("failed" in s for s in statuses)

    @pytest.mark.asyncio
    async def test_on_status_carries_running_failure_count(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = await _add_entry(db_session)
        await _add_snapshot(db_session, entry, extraction_status=ExtractionStatus.FAILED)
        await db_session.commit()

        mock = _patch_extract(monkeypatch)
        mock.side_effect = RuntimeError("boom")
        statuses: list[str] = []
        await reindex(
            db_session,
            run_post_lint=False,
            rate_limit_per_minute=0,
            on_status=statuses.append,
        )

        assert statuses == [f"snapshot 1/1 (entry {entry.entry_id[:8]}…) — 1 failed"]
