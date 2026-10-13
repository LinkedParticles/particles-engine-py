# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles subjects fix-labels`` repairs label-cache rows an outage poisoned.

Before the fix, a failed Wikidata label fetch persisted the raw id as the
entity's label, and later extractions dropped every statement pointing at it.
The verb re-fetches every cached row whose label equals its own id, item and
property alike: a fetched label replaces the row, and a failed fetch deletes it
so the next extraction asks again.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from particles.api.cli.subjects import _fix_subject_labels
from particles.store.wikidata_cache import get_label, set_label


@pytest.mark.asyncio
async def test_poisoned_rows_are_repaired_or_cleared(db_session: Any) -> None:
    await set_label(db_session, "P571", "P571")  # poisoned, Wikidata now answers
    await set_label(db_session, "Q42", "Q42")  # poisoned, Wikidata still unreachable
    await set_label(db_session, "Q99", "Q99")  # genuinely has no English label
    await set_label(db_session, "Q64", "Berlin")  # healthy, never re-fetched
    await db_session.commit()

    answers = {"P571": "inception", "Q42": None, "Q99": "Q99"}
    fetch = AsyncMock(side_effect=lambda entity_id: answers[entity_id])
    with patch("particles.extraction.wikidata.fetch_live_label", fetch):
        fixed, skipped, repaired, cleared = await _fix_subject_labels()

    assert (fixed, skipped, repaired, cleared) == (0, 0, 1, 1)
    assert sorted(c.args[0] for c in fetch.await_args_list) == ["P571", "Q42", "Q99"]

    db_session.expire_all()
    assert await get_label(db_session, "P571") == "inception"
    assert await get_label(db_session, "Q42") is None
    assert await get_label(db_session, "Q99") == "Q99"
    assert await get_label(db_session, "Q64") == "Berlin"
