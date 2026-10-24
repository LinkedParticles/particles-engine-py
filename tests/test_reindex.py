# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
from particles.corpus.store import CorpusEntryRow, SnapshotRow, update_extraction_status
from particles.extraction.components import ComponentRecord
from particles.operations.reindex import (
    ReindexPlan,
    _collapse_for_auto_discovery,
    _identify_scope,
    _reindex_snapshot,
    current_component_tables,
    reindex,
)
from particles.operations.reindex_scope import ONLY_CHANGED_COMPONENTS_REFUSAL
from particles.store.particle_store import get_particle, insert_particle
from tests._late_generation import (
    ADDED,
    DROPPED,
    KEPT,
    Scripted,
    active_copies,
    deposit_version,
    extract,
    note_uri,
    quiet_probes,  # noqa: F401 (fixture)
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Stamps relative to the patched SDK version ``CURRENT`` (D1): tests
#: that classify stamps patch ``version_guard.SCHEMA_VERSION`` to ``CURRENT``.
CURRENT, OLDER, NEWER = "1.2.0", "1.1.0", "1.3.0"


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
    async def test_a_newer_revisit_is_not_the_snapshot_to_reindex(self, db_session: Any) -> None:
        """A REVISIT is COMPLETE, newer and blob-less; reindexing it extracts nothing."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry, captured_at=datetime(2026, 1, 1, tzinfo=UTC))
        revisit = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=datetime(2026, 5, 1, tzinfo=UTC),
            content_hash=snap.content_hash,
            extraction_status=ExtractionStatus.COMPLETE,
            warc_record_type=WarcRecordType.REVISIT,
            refers_to=snap.snapshot_id,
        )
        db_session.add(SnapshotRow.from_model(revisit, entry.entry_id))
        await db_session.commit()

        scope = await _identify_scope(db_session, [entry.entry_id], None, None, False)
        assert scope == [(entry.entry_id, snap.snapshot_id)]

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
        await _add_particle(db_session, named, named_snap, schema_version=OLDER)
        unnamed = await _add_entry(db_session)
        unnamed_snap = await _add_snapshot(db_session, unnamed)
        await _add_particle(db_session, unnamed, unnamed_snap, schema_version=OLDER)
        await db_session.commit()

        scope = await _identify_scope(
            db_session,
            [named.entry_id],
            None,
            None,
            True,
            "openai:gpt-5.6-luna",
            schema_stale_versions=[OLDER],
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
    async def test_auto_discovery_collapses_superseded_generations(self, db_session: Any) -> None:
        """a FAILED old generation of a MUTABLE entry is not worth a retry."""
        from particles.core.schema import Mutability
        from particles.corpus.deposit import save_blob, sha256

        entry = await _add_entry(db_session)
        row = await db_session.get(CorpusEntryRow, entry.entry_id)
        row.mutability = Mutability.MUTABLE.value
        stale = await _add_snapshot(
            db_session,
            entry,
            extraction_status=ExtractionStatus.FAILED,
            captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        content = b"the newest generation"
        newest = await _add_snapshot(
            db_session,
            entry,
            extraction_status=ExtractionStatus.PENDING,
            captured_at=datetime(2026, 5, 1, tzinfo=UTC),
            content_hash=sha256(content),
        )
        save_blob(content, newest.content_hash)
        await db_session.commit()

        # The collapse is its own step before scope gathering (D2);
        # --dry-run promises zero writes: it reports the narrowed scope, marks nothing.
        lines: list[str] = []
        planned = await _collapse_for_auto_discovery(
            db_session, None, True, progress=lines.append, dry_run=True
        )
        assert planned == {stale.snapshot_id}
        dry = await _identify_scope(
            db_session, None, None, None, include_failed=True, collapsed=planned
        )
        assert dry == [(entry.entry_id, newest.snapshot_id)]
        assert any("Skipped 1 superseded snapshot(s)" in line for line in lines)
        stale_row = await db_session.get(SnapshotRow, stale.snapshot_id)
        await db_session.refresh(stale_row)
        assert stale_row.extraction_status == ExtractionStatus.FAILED.value
        assert stale_row.superseded_by_snapshot_id is None

        # The real run marks it, so the scope gather no longer even lists it.
        applied = await _collapse_for_auto_discovery(db_session, None, True)
        assert applied == {stale.snapshot_id}
        scope = await _identify_scope(db_session, None, None, None, include_failed=True)
        assert scope == [(entry.entry_id, newest.snapshot_id)]
        await db_session.refresh(stale_row)
        assert stale_row.extraction_status == ExtractionStatus.COMPLETE.value
        assert stale_row.superseded_by_snapshot_id == newest.snapshot_id

    @pytest.mark.asyncio
    async def test_reindex_runs_the_collapse_before_scoping(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``reindex`` wires the collapse step: planned on --dry-run, applied live."""
        from particles.core.schema import Mutability
        from particles.corpus.deposit import save_blob, sha256

        entry = await _add_entry(db_session)
        row = await db_session.get(CorpusEntryRow, entry.entry_id)
        row.mutability = Mutability.MUTABLE.value
        stale = await _add_snapshot(
            db_session,
            entry,
            extraction_status=ExtractionStatus.FAILED,
            captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        content = b"the newest generation, reindexed"
        newest = await _add_snapshot(
            db_session,
            entry,
            extraction_status=ExtractionStatus.PENDING,
            captured_at=datetime(2026, 5, 1, tzinfo=UTC),
            content_hash=sha256(content),
        )
        save_blob(content, newest.content_hash)
        await db_session.commit()
        _patch_extract(monkeypatch)

        lines: list[str] = []
        dry = await reindex(db_session, run_post_lint=False, dry_run=True, progress=lines.append)
        assert dry["scope"] == 1
        assert any("Skipped 1 superseded snapshot(s)" in line for line in lines)
        stale_row = await db_session.get(SnapshotRow, stale.snapshot_id)
        await db_session.refresh(stale_row)
        assert stale_row.superseded_by_snapshot_id is None

        live = await reindex(db_session, run_post_lint=False)
        assert live["scope"] == 1
        await db_session.refresh(stale_row)
        assert stale_row.superseded_by_snapshot_id == newest.snapshot_id

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
    async def test_stale_schema_selects_only_the_named_stamps(self, db_session: Any) -> None:
        """Stale-schema discovery matches the stamps it is handed, by ``IN`` (D5)."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, schema_version=OLDER)
        other = await _add_entry(db_session)
        other_snap = await _add_snapshot(db_session, other)
        await _add_particle(db_session, other, other_snap, schema_version=NEWER)
        await db_session.commit()

        scope = await _identify_scope(
            db_session, None, None, None, include_failed=False, schema_stale_versions=[OLDER]
        )
        assert scope == [(entry.entry_id, snap.snapshot_id)]

    @pytest.mark.asyncio
    async def test_stale_schema_is_not_gathered_without_stamps(self, db_session: Any) -> None:
        """No stamps named, no stale-schema pairs: the default since D5."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, schema_version=OLDER)
        await db_session.commit()

        scope = await _identify_scope(db_session, None, None, None, include_failed=False)
        assert scope == []

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
            db_session, entry, snap, schema_version=OLDER, extractor_version="0.1.0"
        )
        await db_session.commit()

        scope = await _identify_scope(
            db_session, None, "0.1.0", None, include_failed=False, schema_stale_versions=[OLDER]
        )
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
            "skipped_newer_schema": 0,
            "plan": {
                "entries": 0,
                "snapshots": 0,
                "particles": 0,
                "missing_blobs": 0,
                "scope_description": "auto: failed/pending",
                "snapshot_plans": [],
                "absent_entries": [],
                "retargeted": 0,
                "schema_stale_particles": 0,
                "schema_stale_included": False,
                "skipped_newer_schema": 0,
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
        assert plan["scope_description"] == "extractor-version 0.13.0"
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

        monkeypatch.setattr("particles.operations.reindex.blob_size", lambda _h: 1234)
        result = await reindex(
            db_session, entry_ids=[entry.entry_id], run_post_lint=False, dry_run=True
        )
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["missing_blobs"] == 0
        assert plan["snapshot_plans"][0]["blob_missing"] is False
        # the plan carries each snapshot's size, what --estimate prices.
        assert plan["snapshot_plans"][0]["source_bytes"] == 1234

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


# ---------------------------------------------------------------------------
# --only-changed-components
# ---------------------------------------------------------------------------


def _current_record(extractor: str = "general-extractor") -> ComponentRecord:
    """A record whose every exercised component matches today's table."""
    table = current_component_tables()[extractor]
    return ComponentRecord(
        extractor=extractor,
        exercised={name: table.digests[name] for name in table.always},
        available=sorted(table.digests),
    )


class TestOnlyChangedComponents:
    @pytest.mark.asyncio
    async def test_refused_while_not_enabled(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _patch_extract(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            await reindex(
                db_session,
                extractor_version="0.3.0",
                only_changed_components=True,
                run_post_lint=False,
            )
        assert str(excinfo.value) == ONLY_CHANGED_COMPONENTS_REFUSAL
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_when_enabled_it_skips_unchanged_snapshots_and_keeps_their_stamp(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The design the trigger flips on, exercised end to end.

        The unchanged snapshot is skipped and its particle keeps ACTIVE with its
        old version stamp, so the next scope over that version reads it again;
        the snapshot with no record is re-extracted.
        """
        monkeypatch.setattr("particles.operations.reindex.ONLY_CHANGED_COMPONENTS_ENABLED", True)
        entry_same = await _add_entry(db_session)
        snap_same = await _add_snapshot(db_session, entry_same)
        kept_particle = await _add_particle(db_session, entry_same, snap_same)
        await update_extraction_status(
            db_session,
            snap_same.snapshot_id,
            ExtractionStatus.COMPLETE,
            components=_current_record(),
        )
        entry_legacy = await _add_entry(db_session, uri_r="https://example.com/y")
        snap_legacy = await _add_snapshot(db_session, entry_legacy)
        await _add_particle(db_session, entry_legacy, snap_legacy)
        await db_session.commit()

        mock = _patch_extract(monkeypatch)
        result = await reindex(
            db_session,
            extractor_version="0.3.0",
            only_changed_components=True,
            include_failed=False,
            run_post_lint=False,
        )
        reindexed = {call.args[2] for call in mock.await_args_list}
        assert reindexed == {snap_legacy.snapshot_id}
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert "1 snapshot(s) skipped, components unchanged" in plan["scope_description"]
        survivor = await get_particle(db_session, kept_particle.id)
        assert survivor is not None
        assert survivor.status is Status.ACTIVE
        assert survivor.extractor_ref is not None
        assert survivor.extractor_ref.version == "0.3.0"

    @pytest.mark.asyncio
    async def test_needs_a_version_scope(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("particles.operations.reindex.ONLY_CHANGED_COMPONENTS_ENABLED", True)
        _patch_extract(monkeypatch)
        with pytest.raises(ValueError, match="--extractor-version scope only"):
            await reindex(
                db_session, extractor_id="general-extractor", only_changed_components=True
            )


# ---------------------------------------------------------------------------
# A provenance scope re-extracts the current generation
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("quiet_probes")
class TestRetargetToCurrent:
    """``reindex --extractor-version`` over a note edited since that version.

    The note said "kept" and "dropped", then "kept" and "added". Its claims'
    refs name the day-1 snapshot, so the scope names it. Re-extracting the
    day-1 text would supersede "kept" and then retire its re-extraction,
    which the current text does not anchor; the scope is re-aimed at day 2.
    """

    @staticmethod
    async def _edited_note(session: Any, ex: Scripted) -> tuple[str, str, str]:
        uri = note_uri()
        entry_id, older = await deposit_version(session, ex, uri, [KEPT, DROPPED], day=1)
        await extract(session, ex, entry_id, older)
        _, newer = await deposit_version(session, ex, uri, [KEPT, ADDED], day=2)
        await extract(session, ex, entry_id, newer)
        return entry_id, older, newer

    @staticmethod
    async def _reindex(session: Any, ex: Scripted, monkeypatch: Any, version: str) -> Any:
        from functools import partial

        from particles.ingest.pipeline import extract_snapshot
        from particles.operations import reindex as reindex_mod

        old_version = ex.EXTRACTOR_VERSION
        ex.EXTRACTOR_VERSION = version
        monkeypatch.setattr(
            reindex_mod, "extract_snapshot", partial(extract_snapshot, extractor=ex)
        )
        return await reindex_mod.reindex(
            session,
            extractor_version=old_version,
            include_failed=False,
            rate_limit_per_minute=0,
            run_post_lint=False,
        )

    @pytest.mark.asyncio
    async def test_every_claim_the_current_text_states_keeps_one_active_copy(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        ex = Scripted()
        await self._edited_note(db_session, ex)

        result = await self._reindex(db_session, ex, monkeypatch, "2")

        assert result["plan"]["retargeted"] >= 1  # type: ignore[index]
        assert await active_copies(db_session, KEPT) == 1
        assert await active_copies(db_session, ADDED) == 1
        assert await active_copies(db_session, DROPPED) == 0

    @pytest.mark.asyncio
    async def test_a_claim_anchored_only_to_the_old_snapshot_converges_in_two_passes(
        self, db_session: Any, monkeypatch: Any
    ) -> None:
        """An old carry-forward, written before re-observation refs: only day 1."""
        import json

        from sqlalchemy import select

        from particles.store.particle_store import ParticleRow

        ex = Scripted()
        _, _, newer = await self._edited_note(db_session, ex)
        row = (
            await db_session.execute(select(ParticleRow).where(ParticleRow.content == KEPT))
        ).scalar_one()
        refs = json.loads(row.provenance_json)
        row.provenance_json = json.dumps([r for r in refs if r.get("snapshot_id") != newer])
        await db_session.commit()

        for version in ("2", "3"):
            await self._reindex(db_session, ex, monkeypatch, version)
            assert await active_copies(db_session, KEPT) == 1
            assert await active_copies(db_session, ADDED) == 1
            assert await active_copies(db_session, DROPPED) == 0


# ---------------------------------------------------------------------------
# D2 / D5: same-major schema minors
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk_at_current(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as an SDK at ``CURRENT``, so ``OLDER`` / ``NEWER`` are its minors."""
    monkeypatch.setattr("particles.operations.version_guard.SCHEMA_VERSION", CURRENT)


#: Per selector: the particle fields that make it match, and the reindex kwargs.
_SELECTORS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "named": ({}, {"include_failed": False}),
    "extractor_version": (
        {"extractor_version": "0.1.0"},
        {"extractor_version": "0.1.0", "include_failed": False},
    ),
    "extractor_id": (
        {"extractor_id": "old-extractor"},
        {"extractor_id": "old-extractor", "include_failed": False},
    ),
    "provider_model": (
        {"extraction_provider_model": "openai:luna"},
        {"provider_model": "openai:luna", "include_failed": False},
    ),
    "failed_pending": ({}, {"include_failed": True}),
    "stale_schema": ({}, {"include_failed": False, "include_schema_stale": True}),
}


@pytest.mark.usefixtures("sdk_at_current")
class TestSchemaMinors:
    @pytest.mark.asyncio
    async def test_bare_reindex_never_selects_older_minors(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opt-in only (D5); the plan still says what the flag would add."""
        entry = await _add_entry(db_session)
        snap = await _add_snapshot(db_session, entry)
        await _add_particle(db_session, entry, snap, schema_version=OLDER)
        await _add_particle(db_session, entry, snap, schema_version=OLDER, content="Another.")
        await db_session.commit()

        mock = _patch_extract(monkeypatch)
        result = await reindex(db_session, include_failed=False, run_post_lint=False)

        mock.assert_not_called()
        assert result["scope"] == 0
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["schema_stale_particles"] == 2
        assert plan["schema_stale_included"] is False
        assert plan["scope_description"] == "nothing selected"
        line = ReindexPlan.model_validate(plan).format_line()
        assert "2 particle(s) at an older schema minor not in scope" in line
        assert "--include-schema-stale" in line

    @pytest.mark.asyncio
    async def test_flag_selects_older_minors_only(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Equal and newer minors are never discovered by schema version (D5)."""
        snaps: dict[str, tuple[str, str]] = {}
        for stamp in (OLDER, CURRENT, NEWER):
            entry = await _add_entry(db_session)
            snap = await _add_snapshot(db_session, entry)
            await _add_particle(db_session, entry, snap, schema_version=stamp)
            snaps[stamp] = (entry.entry_id, snap.snapshot_id)
        await db_session.commit()

        mock = _patch_extract(monkeypatch)
        result = await reindex(
            db_session,
            include_failed=False,
            include_schema_stale=True,
            run_post_lint=False,
            dry_run=True,
        )

        mock.assert_not_called()
        plan = result["plan"]
        assert isinstance(plan, dict)
        selected = [(sp["entry_id"], sp["snapshot_id"]) for sp in plan["snapshot_plans"]]
        assert selected == [snaps[OLDER]]
        assert plan["scope_description"] == "auto: stale schema"
        assert plan["schema_stale_particles"] == 1
        assert plan["schema_stale_included"] is True
        # The NEWER snapshot was never in scope, so nothing was skipped for it.
        assert result["skipped_newer_schema"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("selector", sorted(_SELECTORS))
    async def test_snapshot_holding_a_newer_minor_is_skipped_under_every_selector(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch, selector: str
    ) -> None:
        """An older SDK never re-mints a newer-minor particle (D2), whatever reached it."""
        fields, kwargs = _SELECTORS[selector]
        status = (
            ExtractionStatus.FAILED if selector == "failed_pending" else ExtractionStatus.COMPLETE
        )
        stamp = OLDER if selector == "stale_schema" else CURRENT

        held = await _add_entry(db_session)
        held_snap = await _add_snapshot(db_session, held, extraction_status=status)
        await _add_particle(db_session, held, held_snap, schema_version=stamp, **fields)
        newer = await _add_particle(
            db_session, held, held_snap, schema_version=NEWER, content="Newer.", **fields
        )
        control = await _add_entry(db_session)
        control_snap = await _add_snapshot(db_session, control, extraction_status=status)
        await _add_particle(db_session, control, control_snap, schema_version=stamp, **fields)
        await db_session.commit()
        if selector == "named":
            kwargs = {**kwargs, "entry_ids": [held.entry_id, control.entry_id]}

        mock = _patch_extract(monkeypatch)
        result = await reindex(db_session, run_post_lint=False, rate_limit_per_minute=0, **kwargs)

        assert result["skipped_newer_schema"] == 1
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["skipped_newer_schema"] == 1
        assert "1 snapshot(s) skipped, they hold particles a newer SDK wrote" in (
            ReindexPlan.model_validate(plan).format_line()
        )
        assert result["scope"] == 1
        extracted = [call.args[2] for call in mock.call_args_list]
        assert extracted == [control_snap.snapshot_id]
        after = await get_particle(db_session, newer.id)
        assert after is not None and after.status == Status.ACTIVE

    @pytest.mark.asyncio
    async def test_no_particle_load_when_the_store_holds_no_newer_minor(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The normal case does no per-snapshot work: the newer-minor load never runs."""
        from particles.operations import reindex as reindex_module

        for _ in range(3):
            entry = await _add_entry(db_session)
            snap = await _add_snapshot(db_session, entry)
            await _add_particle(
                db_session, entry, snap, schema_version=CURRENT, extractor_version="0.1.0"
            )
        await db_session.commit()

        load = AsyncMock(wraps=reindex_module.get_active_particles_at_schema_versions)
        monkeypatch.setattr(
            "particles.operations.reindex.get_active_particles_at_schema_versions", load
        )
        _patch_extract(monkeypatch)
        result = await reindex(
            db_session,
            extractor_version="0.1.0",
            include_failed=False,
            run_post_lint=False,
            rate_limit_per_minute=0,
        )

        assert result["succeeded"] == 3
        load.assert_not_called()

    @pytest.mark.asyncio
    async def test_newer_minor_check_is_one_load_per_run(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With newer stamps present, the check is one bulk read, not one per snapshot."""
        from particles.operations import reindex as reindex_module

        for _ in range(3):
            entry = await _add_entry(db_session)
            snap = await _add_snapshot(db_session, entry)
            await _add_particle(
                db_session, entry, snap, schema_version=CURRENT, extractor_version="0.1.0"
            )
        elsewhere = await _add_entry(db_session)
        elsewhere_snap = await _add_snapshot(db_session, elsewhere)
        await _add_particle(db_session, elsewhere, elsewhere_snap, schema_version=NEWER)
        await db_session.commit()

        load = AsyncMock(wraps=reindex_module.get_active_particles_at_schema_versions)
        monkeypatch.setattr(
            "particles.operations.reindex.get_active_particles_at_schema_versions", load
        )
        _patch_extract(monkeypatch)
        result = await reindex(
            db_session,
            extractor_version="0.1.0",
            include_failed=False,
            run_post_lint=False,
            rate_limit_per_minute=0,
        )

        assert result["succeeded"] == 3
        assert result["skipped_newer_schema"] == 0
        load.assert_awaited_once()
        assert load.await_args is not None
        assert list(load.await_args.args[1]) == [NEWER]
