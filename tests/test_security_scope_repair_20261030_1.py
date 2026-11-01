# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""One project is one key in the scope repair verbs (security review 2026-10-30-1, AB33, AB69).

Reads compare project keys in NFC. These tests drive the administrative verbs
that write and remove ``project:`` tags, and the rank-lift monitor's live
keys, against a scratch store (the suite's ``db_session``): no model call, no
outside connection.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from tests._observer_scope import belief, project_source_tags, source

NFC = unicodedata.normalize("NFC", "café")
NFD = unicodedata.normalize("NFD", "café")


async def test_a_withdrawn_key_is_out_of_view_whatever_form_the_entry_stored_it_in(
    db_session: Any,
) -> None:
    """When ``withdraw_key`` reports a key withdrawn, the entry's beliefs are out of its view."""
    from particles.operations.observer_scope import withdraw_key
    from particles.operations.query.observer_scope import load_scopes

    entry = await source(db_session, "t", [*project_source_tags("other"), f"project:{NFD}"])
    claim = await belief(db_session, "a claim", entry)
    assert NFC in (await load_scopes(db_session, [claim]))[claim.id].keys

    reported = await withdraw_key(db_session, entry[0], NFC, actor="op")

    still = NFC in (await load_scopes(db_session, [claim]))[claim.id].keys
    assert not (reported and still), (
        "withdraw reported success, yet the project still sees the belief"
    )


async def test_assign_key_stores_the_canonical_spelling(db_session: Any) -> None:
    """``assign_key`` stores a ``project:`` tag in the NFC spelling every binding compares."""
    from particles.corpus.store import get_entry
    from particles.operations.observer_scope import assign_key

    entry = await source(db_session, "u", ["claude-code"])
    assert await assign_key(db_session, entry[0], f" {NFD} ", actor="op")
    stored = await get_entry(db_session, entry[0])
    assert stored is not None
    project_tags = [t for t in stored.tags if t.startswith("project:")]
    assert project_tags == [f"project:{NFC}"], project_tags


async def test_an_observer_that_canonicalises_to_nothing_sees_only_what_every_project_sees(
    db_session: Any,
) -> None:
    """An observer of only whitespace is no project: keyed beliefs stay out of view (control)."""
    from particles.core.observer_scope import visible
    from particles.operations.query.observer_scope import load_scopes

    keyed = await belief(
        db_session, "keyed", await source(db_session, "k", project_source_tags("a"))
    )
    glob = await belief(db_session, "global", await source(db_session, "g", ["web"]))
    scopes = await load_scopes(db_session, [keyed, glob])
    assert not visible(scopes[keyed.id], "   ")
    assert visible(scopes[glob.id], "   ")


async def test_the_rank_lift_monitor_reads_one_live_key_per_project(db_session: Any) -> None:
    """Two trusted exposure rows naming one project in two normalisation forms are one live key."""
    from datetime import UTC, datetime, timedelta

    from particles.operations.rank_lift_monitor import live_project_keys
    from particles.store.session_exposure_store import SessionExposure, record_session_exposure

    now = datetime.now(UTC)
    for i, key in enumerate((NFD, NFC)):
        await record_session_exposure(
            db_session,
            SessionExposure(
                session_id=f"s-{i}",
                recorded_at=now,
                source="startup",
                action="full",
                project_key=key,
                observer_scope_applied=True,
                beliefs=[],
            ),
        )
    await db_session.flush()
    live = await live_project_keys(db_session, now - timedelta(minutes=1))
    assert len(live) == 1, sorted(live)
