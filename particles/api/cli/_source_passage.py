# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering for a hydrated source passage.

Used by ``particle source`` and ``query --show-source``. The match label is
spelled out every time: an ``EXACT`` passage is verified against the recorded
chunk hash, a ``LOCATED`` one is only the best term-overlap paragraph, and a
reader must never have to guess which they are looking at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particles.operations.source_passage import SourcePassage


def match_label(passage: SourcePassage) -> str:
    """One-line, human-readable statement of how the passage was found."""
    match passage.match.value:
        case "EXACT":
            return "exact: the chunk the extractor saw, verified by its recorded hash"
        case "LOCATED":
            pct = f"{(passage.locate_overlap or 0.0):.0%}"
            return f"located: best term overlap with the belief ({pct}); not hash-verified"
        case "WHOLE":
            return "whole source: no single passage stood out"
        case _:
            return "unavailable"


def header_lines(passage: SourcePassage) -> list[str]:
    """Identifying metadata for a passage, one string per output line."""
    lines: list[str] = []
    if passage.uri_r or passage.source_type:
        kind = f"  ({passage.source_type})" if passage.source_type else ""
        lines.append(f"Source:    {passage.uri_r or '(no URI)'}{kind}")
    if passage.corpus_entry_id:
        snap = f"  snapshot {passage.snapshot_id[:8]}…" if passage.snapshot_id else ""
        lines.append(f"Entry:     {passage.corpus_entry_id[:8]}…{snap}")
    lines.append(f"Match:     {match_label(passage)}")
    if passage.truncated:
        lines.append(
            f"           (passage cut to {len(passage.text)} characters; "
            f"`particles corpus cat {(passage.snapshot_id or '')[:8]}` shows the full text)"
        )
    if passage.note:
        lines.append(f"Note:      {passage.note}")
    return lines
