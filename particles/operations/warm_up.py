# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Warm a long-lived engine's render path at start.

The first ``MEMORY.md`` render after an engine starts builds the candidate
cache and pays the render path's first queries; on a 61k-belief store that is
about 5 s, more than a SessionStart's freshness share. :func:`warm_render_path`
pays it once, in the background, before a session asks: it builds the store's
candidate cache without vectors and runs one confidence-only retrieval through
the same query path the region render takes. Nothing is written, and the
result is discarded.
"""

from __future__ import annotations

import logging
import time

from particles.core.schema import QueryRequest
from particles.db import StoreHandle, session_scope
from particles.operations.query import retrieve_ranked
from particles.store.candidate_cache import load_active_candidate_particles

log = logging.getLogger(__name__)


async def warm_render_path(store: StoreHandle) -> float:
    """Build ``store``'s candidate cache and run one render-path retrieval; seconds taken."""
    started = time.monotonic()
    async with session_scope(store) as session:
        await load_active_candidate_particles(session)
        await retrieve_ranked(
            session,
            QueryRequest(question="warm-up", top_k=40),
            use_embeddings=False,
            apply_utility=True,
            apply_owner=True,
        )
    seconds = time.monotonic() - started
    log.info("warmed the render path of store %r in %.1f s", store, seconds)
    return seconds
