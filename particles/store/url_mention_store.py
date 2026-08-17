"""Undeposited URL-mention store — citation signal.

Tracks every external URL *mentioned* across the corpus — including ones the
corpus has **not** deposited — so frequently-cited primary sources can be
ranked as operator deposit suggestions. This is the substrate
``corpus_follow_edges`` cannot carry: a follow edge's target must
be a deposited entry, while a mention has no required corpus entry until (and
unless) the operator acts on the suggestion.

Two tables, sitting **beside** ``corpus_follow_edges`` (resolving the
"subsumes or sits beside" fork toward *beside*):

* ``url_mentions`` — one row per ``(source_entry_id, canonical_url)``,
  idempotent so re-extraction never inflates counts. ``target_entry_id`` is
  ``NULL`` while the URL is undeposited and is filled at deposit time by
  :func:`reconcile_url_to_entry`, which also lets the suggestion query exclude
  now-deposited URLs with a plain ``IS NULL`` filter.
* ``url_suggestion_state`` — one row per ``canonical_url`` carrying a
  ``suppressed_until`` timestamp, the queryable half of dismiss / snooze (the
  audited half rides the operator event log).

Canonicalization is the caller's job (``particles.url_canonical``); this module
stores whatever canonical string it is handed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base


class UrlMentionRow(Base):
    """One ``(source_entry_id, canonical_url)`` citation of a URL."""

    __tablename__ = "url_mentions"
    __table_args__ = (
        Index("ix_url_mentions_canonical", "canonical_url"),
        Index("ix_url_mentions_target", "target_entry_id"),
    )

    source_entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    canonical_url: Mapped[str] = mapped_column(String, primary_key=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # NULL while undeposited; set to the resulting entry when the URL is later
    # deposited (reconcile_url_to_entry). The suggestion query ranks NULL rows.
    target_entry_id: Mapped[str | None] = mapped_column(String, nullable=True)


class UrlSuggestionStateRow(Base):
    """Per-URL dismiss / snooze state — the queryable half."""

    __tablename__ = "url_suggestion_state"

    canonical_url: Mapped[str] = mapped_column(String, primary_key=True)
    # A suggestion is suppressed while ``now < suppressed_until``. A permanent
    # dismiss is a far-future timestamp; a snooze is ``now + duration``.
    suppressed_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


async def record_url_mentions(
    session: AsyncSession,
    *,
    source_entry_id: str,
    canonical_urls: Iterable[str],
    deposited_map: Mapping[str, str] | None = None,
) -> int:
    """Record URL mentions for one source, idempotently.

    A duplicate ``(source_entry_id, canonical_url)`` is a no-op — the mention
    already exists — so re-extracting the same source never inflates counts
    . ``deposited_map`` (``{canonical_url: entry_id}``, from
    :func:`build_deposited_url_map`) lets a URL that is *already* deposited be
    born with its ``target_entry_id`` set, so it never surfaces as a
    suggestion. Returns the number of new rows inserted.
    """
    urls = list(dict.fromkeys(canonical_urls))  # de-dupe, preserve order
    if not urls:
        return 0
    existing = set(
        (
            await session.execute(
                select(UrlMentionRow.canonical_url).where(
                    UrlMentionRow.source_entry_id == source_entry_id,
                    UrlMentionRow.canonical_url.in_(urls),
                )
            )
        )
        .scalars()
        .all()
    )
    now = datetime.now(UTC)
    deposited = deposited_map or {}
    inserted = 0
    for url in urls:
        if url in existing:
            continue
        session.add(
            UrlMentionRow(
                source_entry_id=source_entry_id,
                canonical_url=url,
                discovered_at=now,
                target_entry_id=deposited.get(url),
            )
        )
        inserted += 1
    if inserted:
        await session.flush()
    return inserted


async def reconcile_url_to_entry(
    session: AsyncSession,
    *,
    canonical_url: str,
    target_entry_id: str,
) -> list[str]:
    """Bind every undeposited mention of ``canonical_url`` to a deposited entry.

    Called from the deposit path when a previously-mentioned URL
    is finally deposited. Sets ``target_entry_id`` on all matching rows whose
    target is still ``NULL`` and returns the distinct ``source_entry_id`` values
    that were bound — the deposit path turns each into a ``COMMENT_LINK`` follow
    edge. Idempotent: a second call finds no ``NULL`` rows and
    returns ``[]``.
    """
    rows = list(
        (
            await session.execute(
                select(UrlMentionRow).where(
                    UrlMentionRow.canonical_url == canonical_url,
                    UrlMentionRow.target_entry_id.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    vias: list[str] = []
    for row in rows:
        row.target_entry_id = target_entry_id
        if row.source_entry_id not in vias:
            vias.append(row.source_entry_id)
    if rows:
        await session.flush()
    return vias


async def build_deposited_url_map(session: AsyncSession) -> dict[str, str]:
    """Return ``{canonical_url: entry_id}`` over every deposited corpus entry.

    Canonicalizes each entry's ``uri_r`` with the same rules mention capture
    uses, so a URL already in the corpus is recognized at capture time and its
    mention is born deposited (never surfaced as a suggestion). Entries with no
    URL or an uncanonicalizable one are skipped. When two entries canonicalize
    to the same URL the later row wins — harmless for the membership test the
    callers make.
    """
    from particles.corpus.store import CorpusEntryRow
    from particles.url_canonical import canonicalize_url

    rows = (await session.execute(select(CorpusEntryRow.entry_id, CorpusEntryRow.uri_r))).all()
    mapping: dict[str, str] = {}
    for entry_id, uri_r in rows:
        if not uri_r:
            continue
        canon = canonicalize_url(uri_r)
        if canon is not None:
            mapping[canon] = entry_id
    return mapping


async def list_undeposited_mentions(session: AsyncSession) -> list[UrlMentionRow]:
    """Every mention of a URL the corpus has not deposited (``target IS NULL``)."""
    return list(
        (
            await session.execute(
                select(UrlMentionRow).where(UrlMentionRow.target_entry_id.is_(None))
            )
        )
        .scalars()
        .all()
    )


async def suppress_suggestion(
    session: AsyncSession,
    *,
    canonical_url: str,
    until: datetime,
) -> None:
    """Suppress a URL suggestion until ``until`` (dismiss = far future, snooze).

    Upserts the ``url_suggestion_state`` row. The matching audit event is the
    caller's responsibility."""
    row = await session.get(UrlSuggestionStateRow, canonical_url)
    if row is None:
        session.add(UrlSuggestionStateRow(canonical_url=canonical_url, suppressed_until=until))
    else:
        row.suppressed_until = until
    await session.flush()


async def get_suppressed_urls(session: AsyncSession, *, now: datetime | None = None) -> set[str]:
    """Canonical URLs whose suppression is still in effect (``now < until``)."""
    cutoff = now or datetime.now(UTC)
    return set(
        (
            await session.execute(
                select(UrlSuggestionStateRow.canonical_url).where(
                    UrlSuggestionStateRow.suppressed_until > cutoff
                )
            )
        )
        .scalars()
        .all()
    )
