# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the refresh-outcome classification."""

from __future__ import annotations

from particles.core.schema import Snapshot, WarcRecordType
from particles.corpus.refresh_outcome import RefreshOutcome, classify_refresh


def _snap(snapshot_id: str, record: WarcRecordType = WarcRecordType.RESPONSE) -> Snapshot:
    return Snapshot(snapshot_id=snapshot_id, content_hash="0" * 64, warc_record_type=record)


class TestClassifyRefresh:
    def test_no_snapshot_is_missing(self) -> None:
        assert classify_refresh("s1", None) is RefreshOutcome.MISSING

    def test_no_snapshot_and_no_prior_is_missing(self) -> None:
        assert classify_refresh(None, None) is RefreshOutcome.MISSING

    def test_same_snapshot_back_is_unchanged_by_mtime(self) -> None:
        assert classify_refresh("s1", _snap("s1")) is RefreshOutcome.UNCHANGED_MTIME

    def test_same_revisit_back_is_unchanged_by_mtime(self) -> None:
        # The id match wins: no row was written, so no hash was compared.
        snap = _snap("s1", WarcRecordType.REVISIT)
        assert classify_refresh("s1", snap) is RefreshOutcome.UNCHANGED_MTIME

    def test_new_revisit_is_unchanged_by_hash(self) -> None:
        snap = _snap("s2", WarcRecordType.REVISIT)
        assert classify_refresh("s1", snap) is RefreshOutcome.UNCHANGED_HASH

    def test_new_response_is_changed(self) -> None:
        assert classify_refresh("s1", _snap("s2")) is RefreshOutcome.CHANGED

    def test_first_capture_is_changed(self) -> None:
        assert classify_refresh(None, _snap("s1")) is RefreshOutcome.CHANGED
