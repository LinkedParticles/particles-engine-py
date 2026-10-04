# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Consult and extend the probe-verdict ledger from a probing pass.

The two pairwise probing passes, the contradiction census
(:mod:`particles.operations.lint.contradictions`) and the update sweep
(:func:`particles.operations.reconcile.reconcile_updates`), share three steps
around their LLM calls: read what is already answered, record what was just
answered, and name the model that answered. This module is those three steps
over :mod:`particles.store.probe_verdict_store`; the decision of what a
remembered answer means stays with each pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.probe_verdict import ProbeKind, VerdictKey
from particles.db import write_transaction
from particles.store.probe_verdict_store import lookup_verdicts, record_verdicts

__all__ = ["model_label", "recall", "remember"]


def model_label(purpose: str) -> str:
    """``provider/model`` for the LLM purpose a probe runs on, as recorded on a verdict."""
    selection = get_config().llm.for_purpose(purpose)
    return f"{selection.provider}/{selection.model}"


async def recall(
    session: AsyncSession, kind: ProbeKind, prompt_hash: str, keys: Iterable[VerdictKey]
) -> dict[VerdictKey, bool]:
    """The verdicts already recorded for ``keys`` under ``prompt_hash``."""
    return await lookup_verdicts(session, kind, prompt_hash, keys)


async def remember(
    session: AsyncSession,
    kind: ProbeKind,
    prompt_hash: str,
    verdicts: Mapping[VerdictKey, bool],
    *,
    purpose: str,
) -> int:
    """Record fresh verdicts and commit them under the writer lock.

    Committed at once, so a pass that stops part-way (the breaker opened, the
    process died) keeps what it paid for. Anything the session holds
    uncommitted is committed with it, as with every ``write_transaction``.
    """
    if not verdicts:
        return 0
    async with write_transaction(session):
        return await record_verdicts(
            session, kind, prompt_hash, verdicts, model=model_label(purpose)
        )
