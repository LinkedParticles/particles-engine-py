# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Classify one :func:`~particles.corpus.fetch.maybe_refetch` result.

The decide half of a refresh, as a pure function over plain values (
D2). ``particles corpus refresh`` and consolidation pass 0.5 both derive their
counts from :func:`classify_refresh`. The outcomes are fine-grained, so a report
that wants one ``unchanged`` total sums the two unchanged outcomes.
"""

from __future__ import annotations

from enum import StrEnum

from particles.core.schema import Snapshot, WarcRecordType


class RefreshOutcome(StrEnum):
    """What one re-check of a corpus entry found."""

    MISSING = "missing"
    """No snapshot came back: the source is unavailable, or the entry is not refetchable."""
    UNCHANGED_MTIME = "unchanged_mtime"
    """The latest snapshot came back untouched: a tier-1 short-circuit, before any read."""
    UNCHANGED_HASH = "unchanged_hash"
    """A REVISIT was written: the body was read and its hash matched."""
    CHANGED = "changed"
    """A new RESPONSE snapshot landed PENDING."""


def classify_refresh(before_id: str | None, snap: Snapshot | None) -> RefreshOutcome:
    """Classify a refetch result against the entry's latest snapshot id before it.

    Args:
        before_id: The entry's newest snapshot id before the refetch, or None
            when it had none.
        snap: What ``maybe_refetch`` returned.

    Returns:
        The outcome. A returned snapshot whose id equals ``before_id`` is
        ``UNCHANGED_MTIME`` whatever its record type, since no new row was written.
    """
    if snap is None:
        return RefreshOutcome.MISSING
    if snap.snapshot_id == before_id:
        return RefreshOutcome.UNCHANGED_MTIME
    if snap.warc_record_type is WarcRecordType.REVISIT:
        return RefreshOutcome.UNCHANGED_HASH
    return RefreshOutcome.CHANGED
