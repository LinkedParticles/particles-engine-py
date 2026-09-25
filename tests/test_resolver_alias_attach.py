# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The resolver's alias attach goes through ``add_aliases``.

When the cascade attaches an extracted name to an existing Subject (a live
authority returning a stored Subject, or the insert-time dedup by external ref
or by the authority's rewritten name) the write is disclosed by one
``SUBJECT_ALIASED`` event under ``RESOLVER_ALIAS_ACTOR``, a repeat attach is a
no-op, and the name resolves to the attached Subject afterwards.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ExternalRef, Subject
from particles.ingest.authorities import AuthorityResolution
from particles.ingest.subject_resolver import (
    RESOLVER_ALIAS_ACTOR,
    _attach_alias,
    resolve_subject,
)
from particles.store import subject_cache
from particles.store.event_store import OperatorEvent, OperatorEventType, list_events
from particles.store.subject_store import get_subject, insert_subject

_REF = ExternalRef(namespace="fake", id="F1", uri=None, confidence=0.95)


class _FakeAuthority:
    """A live authority whose ``resolve`` answer is fixed per test."""

    NAMESPACE = "fake"
    PRIORITY = 0
    LIVE = True
    DEFAULT_LINK_CONFIDENCE = 0.95
    APPLICABILITY: list[Any] = []

    def __init__(self, answer: AuthorityResolution) -> None:
        self.answer = answer
        self.calls = 0

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> ExternalRef | None:
        return None

    async def resolve(
        self,
        session: AsyncSession,
        name: str,
        *,
        particle_content: str | None,
        domain: str | None,
    ) -> AuthorityResolution | None:
        self.calls += 1
        return self.answer

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        return None


async def _stored(session: AsyncSession, name: str, refs: list[ExternalRef]) -> Subject:
    subject = Subject(
        canonical_name=name,
        external_ids=refs,
        created_at=datetime.now(UTC),
        asserted_by="test",
    )
    await insert_subject(session, subject)
    return subject


async def _aliased_events(session: AsyncSession, subject_id: str) -> list[OperatorEvent]:
    return await list_events(
        session, ref_id=subject_id, event_type=OperatorEventType.SUBJECT_ALIASED
    )


async def _resolve_with(session: AsyncSession, auth: _FakeAuthority, name: str) -> Subject:
    with patch("particles.ingest.subject_resolver.get_authorities", return_value=[auth]):
        return await resolve_subject(session, name)


async def _attach_path(session: AsyncSession, path: str) -> tuple[Subject, _FakeAuthority]:
    """The stored Subject and an authority that drives ``path``'s attach."""
    match path:
        case "live-existing":
            target = await _stored(session, "Society of Mind", [_REF])
            answer = AuthorityResolution(existing=target)
        case "dedup-by-ref":
            target = await _stored(session, "Society of Mind", [_REF])
            answer = AuthorityResolution(external_ref=_REF, canonical_name="Society of Mind")
        case "dedup-by-rewritten-name":
            target = await _stored(session, "Society of Mind", [])
            answer = AuthorityResolution(external_ref=_REF, canonical_name="Society of Mind")
        case _:
            raise AssertionError(path)
    return target, _FakeAuthority(answer)


_PATHS = ["live-existing", "dedup-by-ref", "dedup-by-rewritten-name"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
async def test_attach_records_one_event_under_the_resolver(
    db_session: AsyncSession, path: str
) -> None:
    target, auth = await _attach_path(db_session, path)

    resolved = await _resolve_with(db_session, auth, "the society of mind (book)")

    assert resolved.id == target.id
    assert resolved.aliases == ["the society of mind (book)"]
    events = await _aliased_events(db_session, target.id)
    assert [(e.actor, e.payload) for e in events] == [
        (RESOLVER_ALIAS_ACTOR, {"added": ["the society of mind (book)"]})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
async def test_the_name_resolves_to_the_attached_subject_next_time(
    db_session: AsyncSession, path: str
) -> None:
    target, auth = await _attach_path(db_session, path)
    await _resolve_with(db_session, auth, "SoM book")

    # The cache entry written after the attach survived ``add_aliases``' clear.
    cached = subject_cache.cache_get(subject_cache.make_key(db_session, "SoM book"))
    assert isinstance(cached, subject_cache.CacheEntry)
    assert cached.subject is not None and cached.subject.id == target.id
    again = await _resolve_with(db_session, auth, "SoM book")
    assert again.id == target.id
    assert auth.calls == 1

    # Cold, it resolves through the recorded alias, still without the authority.
    subject_cache.clear()
    cold = await _resolve_with(db_session, auth, "SoM book")
    assert cold.id == target.id
    assert auth.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
async def test_a_repeat_attach_records_nothing(db_session: AsyncSession, path: str) -> None:
    target, auth = await _attach_path(db_session, path)
    for _ in range(3):
        subject_cache.clear()
        await _resolve_with(db_session, auth, "SoM book")

    assert len(await _aliased_events(db_session, target.id)) == 1
    stored = await get_subject(db_session, target.id)
    assert stored is not None and stored.aliases == ["SoM book"]


@pytest.mark.asyncio
async def test_attaching_a_name_the_subject_answers_to_is_a_no_op(
    db_session: AsyncSession,
) -> None:
    """The early return: no write, no event, and the resolution cache is kept."""
    target = await _stored(db_session, "Society of Mind", [])
    attached = await _attach_alias(db_session, target, "SoM")
    subject_cache.cache_set(subject_cache.make_key(db_session, "kept"), attached)

    for name in ("SoM", "som", "Society of Mind", "SOCIETY OF MIND"):
        assert await _attach_alias(db_session, attached, name) is attached

    assert len(await _aliased_events(db_session, target.id)) == 1
    assert subject_cache.cache_get(subject_cache.make_key(db_session, "kept")) is not None
