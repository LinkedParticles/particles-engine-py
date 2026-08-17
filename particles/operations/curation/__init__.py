"""Curation surface — the bus-stop-editing queue + session model.

A finite, leverage-ranked worklist that **unions the existing read diagnostics**
(lint, ``links suggest``, ``corpus links suggest``, the contested digest,
``quality``) into one ``CurationCard`` list — composition only, **no new
detection logic**. The session is finite by design ("today's N") so curation
stays a habit, not an infinite backlog.

Since the expensive *collection* half is persisted (a
``curation_snapshots`` cache row, built nightly by the cycle or on a
cold read) and only the cheap *session* half runs per request, so the queue is
served in well under a second on a store where building it fresh took minutes.

Public surface:
    build_curation_queue   — the ranked, finite, snooze-filtered queue + staleness stamp
    apply_gesture          — dispatch a card's gesture onto an existing write op
    rebuild_curation_snapshot — force a store-wide rebuild of the collection
    CurationQueueResult    — the queue plus its staleness stamp
    QueueSource            — snapshot (default) vs live collection
    CurationCard, CardKind — the normalized card shape
    DuplicateVerdict       — the advisory LLM same-claim verdict on a duplicate card
    register_projection_hook — wire the projection-blocking signal
"""

from __future__ import annotations

from .cards import CardKind, CurationCard, DuplicateVerdict
from .leverage import register_projection_hook
from .session import apply_gesture, build_curation_queue, rebuild_curation_snapshot
from .snapshot import CurationQueueResult, QueueSource

__all__ = [
    "build_curation_queue",
    "rebuild_curation_snapshot",
    "apply_gesture",
    "CurationCard",
    "CardKind",
    "CurationQueueResult",
    "DuplicateVerdict",
    "QueueSource",
    "register_projection_hook",
]
