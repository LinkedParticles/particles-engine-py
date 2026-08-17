"""Explicit operator usefulness gesture — the second utility channel.

Utility is mined from the harvest transcript's *tool-call actions* —
"credit action, not attention". That signal is reliable precisely because it
needs no user reaction, and it structurally cannot reach one class of belief:
**prohibitions and design stances**. ``_NEGATION_RE`` deliberately routes a
prohibition out of the literal tier (if the belief is *"never prepend `export
PATH`"* and an action line contains ``export PATH``, the match is evidence of a
*violation*), and the behavioural tier reads tool-call lines only — so
compliance with "never do X" is the *absence* of an action, and a miner reading
what happened cannot observe what didn't. Measured across the whole utility-mining
projection-diff series, three such guidelines never once reached the projection
head.

This module is the admissible signal for that class: an explicit operator
gesture, recorded as an operator event and credited into the same
reinforcement count ``R`` the miner feeds.

**Two writes, one transaction**. The ``BELIEF_MARKED_USEFUL``
event is the durable, append-only **system of record**; the ``utility_events``
row is a **derived index** that exists so composition is reached
through the path it already uses. Because every explicit row is reconstructible
from the log, ``rebuild-utility`` can stay a blunt truncate-and-re-derive —
:func:`rederive_explicit_credits` is that reconstruction.

The gesture is **operator-only**: there is no MCP tool. An agent
crediting its own beliefs would close a self-reinforcement loop through the digest it is shown next session, and containment cannot
reach it — every §6 defence is confidence-shaped, while utility was made an
*additive* term on top of effective confidence, so no trust multiplier
attenuates a rank-lift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.status import Status
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_particle_events,
    record_event,
)
from particles.store.particle_store import get_particle
from particles.store.utility_store import (
    SOURCE_EXPLICIT,
    clear_utility_events,
    explicit_credit_key,
    record_explicit_utility_event,
)

logger = logging.getLogger(__name__)

CLI_ACTOR = "cli:memory-useful"
"""Actor stamped by ``particles memory useful`` (``iface:slug`` form)."""

HTTP_ACTOR = "http:/memory/useful"
"""Actor stamped by ``POST /memory/useful``."""


class BeliefNotCreditable(ValueError):
    """The target belief does not exist, or is not ACTIVE.

    Raised rather than silently no-op'ing: the explicit channel carries the
    store's heaviest single utility weight, so crediting the wrong id — or a
    retracted one that can never be projected — must be loud at every surface,
    not just the CLI.
    """


@dataclass(frozen=True)
class MarkUsefulResult:
    """Outcome of one gesture.

    Attributes:
        particle_id: The credited belief.
        credit_key: The daily natural key this press folded into.
        counted: ``True`` when a new utility credit was written, ``False`` when
            this principal had already credited this belief today. The audit
            event is recorded either way — the log records what the operator
            did, the index records what it was worth.
        event_id: The ``BELIEF_MARKED_USEFUL`` event's id.
    """

    particle_id: str
    credit_key: str
    counted: bool
    event_id: str


async def mark_belief_useful(
    session: AsyncSession,
    particle_id: str,
    *,
    actor: str = CLI_ACTOR,
    reason: str | None = None,
    now: datetime | None = None,
) -> MarkUsefulResult:
    """Credit one belief from the explicit operator channel.

    Writes the ``utility_events`` row (idempotent per the §7 daily key) and the
    ``BELIEF_MARKED_USEFUL`` operator event, flushing both. **The caller
    commits**, so the event and the credit share one transaction — there is
    never a credit without its event, nor an event without its credit
    (transaction rule).

    Args:
        session: Active store session.
        particle_id: Full id of an ACTIVE belief (callers resolve prefixes).
        actor: The pressing principal; also the identity the §7 rate bound is
            keyed on, so two interfaces are rate-limited independently.
        reason: Optional operator note, stored on the event.
        now: Override for current time (used in tests).

    Raises:
        BeliefNotCreditable: The belief is missing or not ACTIVE.
    """
    particle = await get_particle(session, particle_id)
    if particle is None:
        raise BeliefNotCreditable(f"no belief with id {particle_id!r}")
    if particle.status is not Status.ACTIVE:
        raise BeliefNotCreditable(
            f"belief {particle_id!r} is {particle.status.value}, not ACTIVE — only ACTIVE "
            "beliefs are projected, so crediting it could not surface anything"
        )

    observed_at = now or datetime.now(UTC)
    credit_key = explicit_credit_key(actor, observed_at)
    counted = await record_explicit_utility_event(session, particle_id, credit_key, observed_at)
    event = await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.BELIEF_MARKED_USEFUL,
        reason=reason,
        refs=[(EventRefKind.PARTICLE, particle_id)],
        payload={
            "credit_key": credit_key,
            "counted": counted,
            # The credit's own timestamp, not the event's. `record_event` stamps
            # `occurred_at` itself, so the two coincide in production but are
            # independently sourced — and the §2 invariant is that the row is
            # reconstructible *from the payload*, not from a coincidence.
            "observed_at": observed_at.isoformat(),
        },
    )
    return MarkUsefulResult(
        particle_id=particle_id,
        credit_key=credit_key,
        counted=counted,
        event_id=event.event_id,
    )


async def rederive_explicit_credits(session: AsyncSession) -> int:
    """Rebuild the explicit ``utility_events`` rows from the operator event log.

    The reconstruction half of "the log is the system of record;
    the table is a derived index". ``rebuild-utility`` truncates the table and
    re-mines the transcripts; this replays the append-only
    ``BELIEF_MARKED_USEFUL`` events so the explicit channel comes back with it,
    with each credit's **original** ``observed_at`` from the payload rather than
    stamped as now — otherwise a rebuild would silently reset the decay clock on
    every gesture the operator ever made.

    Replaying is naturally idempotent: several presses on one belief on one day
    share a ``credit_key``, so they collapse to one credit exactly as they did
    when first recorded. Returns the number of credits written.
    """
    await clear_utility_events(session, source=SOURCE_EXPLICIT)
    written = 0
    for particle_id, occurred_at, payload in await list_particle_events(
        session, OperatorEventType.BELIEF_MARKED_USEFUL
    ):
        key = (payload or {}).get("credit_key")
        if not isinstance(key, str) or not key:
            logger.warning(
                "skipping BELIEF_MARKED_USEFUL event for %s: no credit_key in payload",
                particle_id,
            )
            continue
        if await record_explicit_utility_event(
            session, particle_id, key, _credit_timestamp(payload, occurred_at)
        ):
            written += 1
    return written


def _credit_timestamp(payload: dict[str, Any] | None, occurred_at: datetime) -> datetime:
    """The credit's ``observed_at``, from the payload, falling back to the event's clock.

    Reading the payload rather than ``occurred_at`` is what stops a rebuild from
    resetting the decay clock on every gesture the operator ever made — the two
    are stamped independently, and only the payload records the value the
    original row was written with.
    """
    raw = (payload or {}).get("observed_at")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            logger.warning("BELIEF_MARKED_USEFUL payload has unparseable observed_at %r", raw)
        else:
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return occurred_at
