# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim: source-passage hydration moved to :mod:`particles.ingest.source_passage`.

Extraction's second reading of a contradiction reads a claim's
passage from below ``operations``, so the module moved down into ``ingest``.
It reads only ``config``, ``core``, ``corpus``, ``extraction`` and ``store``.
Importing from ``particles.operations.source_passage`` still works; new code
should import from ``particles.ingest.source_passage``.
"""

from __future__ import annotations

from particles.ingest.source_passage import (
    PassageMatch,
    SourcePassage,
    derive_passage,
    extractor_view_text,
    find_chunk,
    focus_window,
    hydrate_source_passage,
    locate_passage,
)

__all__ = [
    "PassageMatch",
    "SourcePassage",
    "derive_passage",
    "extractor_view_text",
    "find_chunk",
    "focus_window",
    "hydrate_source_passage",
    "locate_passage",
]
