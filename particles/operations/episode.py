# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The episode reading, in shadow: gather, count, regenerate, sample (stage 1).

Four Engine halves around the pure :mod:`particles.core.episode`:

* :func:`asked_claims` gathers what the episode ladder reads about each claim
  every source of which is a session type (``curation.session_source_types``)
  and settles the ladder, reporting each run to the open rung tally so a
  ``CONSOLIDATION_RUN`` counts verdicts per rung.
* :func:`shadow_for_cards` counts what the fold would withhold on a card list.
  A curation build records it beside its cards; nothing is withheld.
* :func:`regenerate_episode` is ``particles episode``: it reads stored claims,
  several of one source per call, each beside its source passage, and writes
  the reading through :func:`~particles.store.episode_store.set_episode_reading`.
* :func:`write_sheet` writes the owner's labelling sheet: a seeded
  random sample of the cards each entry would withhold, both claims with their
  passages, plus a sample of ``STANDING`` readings for recall.

No reading feeds a write, a rank or a confidence: the ``episode`` verdict is a
premise of no other question.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.episode import (
    OPERATOR_RULE,
    WITHHELD_KINDS,
    CardSides,
    EpisodeClaim,
    EpisodeInput,
    EpisodeShadow,
    episode_shadow,
    fold_entry,
    stamp_for,
)
from particles.core.ladders import episode_ladder
from particles.core.rung_ladder import EpisodeVerdict, QuestionKind, settle
from particles.corpus.deposit import load_blob
from particles.corpus.store import get_snapshot
from particles.extraction.episode import (
    EpisodeItem,
    EpisodeReply,
    read_episodes,
    regen_instructions,
    regeneration_stamp,
)
from particles.extraction.general import episode_extractor_stamp
from particles.ingest.ladder_shell import report_settled
from particles.ingest.source_passage import extractor_view_text, locate_passage
from particles.llm import fenced_prompt
from particles.store.curation_snapshot_store import latest_snapshot
from particles.store.episode_store import (
    EpisodeRow,
    credited_ids,
    get_episode_rows,
    set_episode_reading,
)

log = logging.getLogger(__name__)

__all__ = [
    "EpisodeRegenSummary",
    "Reader",
    "SheetSummary",
    "asked_claims",
    "latest_cards",
    "regenerate_episode",
    "shadow_for_cards",
    "write_sheet",
]

#: The reader seam: several claims of one source in, one reply out.
Reader = Callable[[Sequence[EpisodeItem]], Awaitable[EpisodeReply | None]]


def _session_types() -> frozenset[str]:
    return frozenset(get_config().curation.session_source_types)


async def _asked_rows(
    session: AsyncSession, particle_ids: Iterable[str] | None
) -> tuple[list[EpisodeRow], set[str]]:
    types = _session_types()
    ids = None if particle_ids is None else list(particle_ids)
    rows = [r for r in await get_episode_rows(session, ids) if r.session_only(types)]
    credited = await credited_ids(session, [r.particle_id for r in rows])
    return rows, credited


async def asked_claims(
    session: AsyncSession, particle_ids: Iterable[str] | None = None
) -> dict[str, EpisodeClaim]:
    """The settled episode ladder of every asked claim among ``particle_ids``.

    A claim with any source outside ``curation.session_source_types`` is never
    asked and is absent from the result. ``None`` asks every ACTIVE claim.
    """
    claims, _ = await _asked(session, particle_ids)
    return claims


async def _asked(
    session: AsyncSession, particle_ids: Iterable[str] | None
) -> tuple[dict[str, EpisodeClaim], dict[str, EpisodeRow]]:
    rows, credited = await _asked_rows(session, particle_ids)
    out: dict[str, EpisodeClaim] = {}
    for row in rows:
        item = EpisodeInput(
            content=row.content,
            credited=row.particle_id in credited,
            asserted=row.asserted,
            valid_from_basis=row.valid_from_basis,
            reading=row.reading,
        )
        settled = settle(episode_ladder(row.reading), item)
        report_settled(QuestionKind.EPISODE, settled)
        out[row.particle_id] = EpisodeClaim(row.particle_id, row.entries, settled)
    return out, {r.particle_id: r for r in rows}


async def shadow_for_cards(
    session: AsyncSession, cards: Iterable[tuple[str, Sequence[str]]]
) -> EpisodeShadow:
    """What the fold would withhold among ``cards`` (``(kind, particle_ids)``), in shadow."""
    sides = [CardSides(kind, tuple(ids)) for kind, ids in cards if kind in WITHHELD_KINDS and ids]
    ids = {pid for card in sides for pid in card.particle_ids}
    claims = await asked_claims(session, ids) if ids else {}
    return episode_shadow(sides, claims, get_config().curation.episode.admitted)


async def latest_cards(session: AsyncSession) -> list[tuple[str, list[str]]]:
    """``(kind, particle_ids)`` of every pair card in the newest curation collection."""
    row = await latest_snapshot(session)
    if row is None:
        return []
    try:
        raw = json.loads(row.cards_json)
    except (TypeError, ValueError):
        return []
    items = raw.get("cards") if isinstance(raw, dict) else None
    out: list[tuple[str, list[str]]] = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        kind, ids = item.get("kind"), item.get("particle_ids")
        if kind in WITHHELD_KINDS and isinstance(ids, list):
            out.append((str(kind), [str(i) for i in ids]))
    return out


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------


class _PassageCache:
    """The extractor-view text of the current entry's snapshots.

    Callers walk claims grouped by entry, so the cache holds one entry's
    snapshots at a time and is emptied when the entry changes: a whole-store
    run never holds every transcript at once.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._entry: str | None = None
        self._texts: dict[str, str | None] = {}

    async def text(
        self, entry_id: str, snapshot_id: str | None, source_type: str | None
    ) -> str | None:
        if entry_id != self._entry:
            self._entry, self._texts = entry_id, {}
        if snapshot_id is None:
            return None
        if snapshot_id not in self._texts:
            text: str | None = None
            snapshot = await get_snapshot(self._session, snapshot_id)
            if snapshot is not None:
                try:
                    text = extractor_view_text(load_blob(snapshot.content_hash), source_type)
                except (OSError, ValueError) as exc:
                    # A passage is a reading aid: a claim whose source cannot
                    # be read is read without one, never the run's end.
                    log.info("episode: snapshot %s has no readable text: %s", snapshot_id, exc)
            self._texts[snapshot_id] = text
        return self._texts[snapshot_id]

    async def passage(self, row: EpisodeRow) -> str:
        if not row.sources:
            return ""
        source = row.sources[0]
        text = await self.text(source.corpus_entry_id, source.snapshot_id, source.source_type)
        if not text:
            return ""
        located = locate_passage(text, row.content, latest=True)
        return located[0] if located is not None else ""


# ---------------------------------------------------------------------------
# Regeneration (``particles episode``)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRegenSummary:
    """What one regeneration run did, or (dry run) would do."""

    dry_run: bool
    scope: str
    backlog: int
    read: int = 0
    written: int = 0
    session_bound: int = 0
    standing: int = 0
    skipped_operator: int = 0
    failed: int = 0
    calls: int = 0
    remaining: int = 0
    stamp: str = ""
    readings: dict[str, int] = field(default_factory=dict)
    #: The list-price estimate of the calls this invocation would make (its
    #: ``--limit``), and of the whole backlog; set on a dry run only.
    estimate: dict[str, object] | None = None
    backlog_estimate: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready form for the verb's report."""
        return dataclasses.asdict(self)


def _needs_reading(row: EpisodeRow, current: frozenset[str], regen: str, restamp: bool) -> bool:
    reading = row.reading
    if reading is None:
        return True
    if reading.rule == OPERATOR_RULE:
        return False
    if restamp:
        return str(reading.stamp) != regen
    return str(reading.stamp) not in current


def _serving_model() -> str:
    """The ``"<provider>:<model>"`` the reader's purpose resolves to (``extraction``)."""
    selection = get_config().llm.for_purpose("extraction")
    return f"{selection.provider}:{selection.model}"


#: The answer's own tokens per claim in a reply: ``{"claim": n, "episode": "…"}``.
_ANSWER_TOKENS_PER_CLAIM = 15


def _estimate(batches: Sequence[Sequence[EpisodeRow]]) -> dict[str, object]:
    """List-price estimate of reading ``batches`` (an upper bound; no call is made).

    Each passage is priced at its full ``curation.episode.passage_chars`` and
    each call's thinking at ``curation.episode.estimate_thinking_tokens``, so
    the figure leans high, as every spend estimate here does.
    """
    # Deferred import, a cycle break (AGENTS.md § Deferred imports case 1):
    # spend_estimate imports operations.audit, which imports the curation
    # package, whose snapshot module imports this one.
    from particles.operations.spend_estimate import chars_per_token

    cfg = get_config()
    episode = cfg.curation.episode
    selection = cfg.llm.for_purpose("extraction")
    system, empty_user = fenced_prompt(regen_instructions(), "", label="claims")
    frame = len(system) + len(empty_user)
    per_token = chars_per_token(selection)
    input_chars = output_tokens = 0
    for batch in batches:
        input_chars += frame
        for n, row in enumerate(batch, start=1):
            claim = len(f"Claim {n}: {row.content}") + 2
            passage = len("Passage:\n") + episode.passage_chars + 2 if episode.passage_chars else 0
            input_chars += claim + passage
        output_tokens += episode.estimate_thinking_tokens + _ANSWER_TOKENS_PER_CLAIM * len(batch)
    input_tokens = round(input_chars / per_token)
    price = cfg.llm.price_for(selection)
    usd = (
        None
        if price is None
        else round((input_tokens * price.input + output_tokens * price.output) / 1_000_000, 2)
    )
    return {
        "calls": len(batches),
        "model": f"{selection.provider}:{selection.model}",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": usd,
        "per_mtok": None if price is None else {"input": price.input, "output": price.output},
        "basis": "passages at curation.episode.passage_chars; thinking at "
        "curation.episode.estimate_thinking_tokens per call",
    }


def _admitted_stamps() -> frozenset[str]:
    out: set[str] = set()
    for key in get_config().curation.episode.admitted:
        _, _, stamps = key.partition(":")
        out.update(s for s in stamps.split("+") if s)
    return frozenset(out)


def _census(rows: Sequence[EpisodeRow]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.reading is None:
            counts["none"] += 1
        else:
            counts[f"{row.reading.stamp}:{row.reading.verdict.value}"] += 1
    return dict(sorted(counts.items()))


def _batches(rows: Sequence[EpisodeRow], size: int) -> list[list[EpisodeRow]]:
    by_entry: dict[str, list[EpisodeRow]] = defaultdict(list)
    for row in rows:
        by_entry[min(row.entries) if row.entries else ""].append(row)
    out: list[list[EpisodeRow]] = []
    for entry in sorted(by_entry):
        group = by_entry[entry]
        out.extend(group[i : i + size] for i in range(0, len(group), size))
    return out


async def regenerate_episode(
    session: AsyncSession,
    *,
    scope: str = "cards",
    limit: int | None = None,
    rate_limit_per_minute: int | None = None,
    restamp: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
    read: Reader | None = None,
) -> EpisodeRegenSummary:
    """Read the episode question over stored claims whose reading is missing or stale.

    ``scope`` is ``cards`` (the sides of the newest collection's pair cards,
    the population the fold acts on; the cheapest run that answers the
    admission question) or ``all`` (every asked ACTIVE claim). A reading is
    stale when its stamp is neither the current extractor's, the current
    regeneration's, nor an admitted one, so a prompt edit never resets what is
    admitted; ``restamp`` rewrites every reading but an
    operator's under the current regeneration stamp. A claim the reply gives
    no verdict for keeps what it had. Commits as it goes.

    Args:
        scope: ``cards`` or ``all``.
        limit: cap on claims read this run; defaults to
            ``curation.episode.batch_limit``, and 0 means the whole backlog.
        rate_limit_per_minute: max reader calls per minute; defaults to
            ``curation.episode.rate_limit_per_minute``, and 0 disables it.
        restamp: rewrite readings under any other stamp but an operator's.
        dry_run: report the backlog and the stored census; write nothing.
        progress: optional callback for human-readable progress lines.
        read: the reader seam, for tests; defaults to the LLM call.
    """
    if scope not in {"cards", "all"}:
        raise ValueError(f"unknown scope {scope!r}; use 'cards' or 'all'")
    cfg = get_config().curation.episode
    effective_limit = cfg.batch_limit if limit is None else limit
    rate = cfg.rate_limit_per_minute if rate_limit_per_minute is None else rate_limit_per_minute
    model = _serving_model()
    stamp = str(stamp_for(regeneration_stamp(), model))

    ids: set[str] | None = None
    if scope == "cards":
        ids = {pid for _, pids in await latest_cards(session) for pid in pids}
    rows, _ = await _asked_rows(session, ids)
    extracted = str(stamp_for(episode_extractor_stamp(), model))
    current = frozenset({stamp, extracted}) | _admitted_stamps()
    # Grouped by entry before any cap, so a capped run fills its calls with
    # claims of one session instead of scattering a few across many.
    backlog = sorted(
        (r for r in rows if _needs_reading(r, current, stamp, restamp)),
        key=lambda r: (min(r.entries) if r.entries else "", r.particle_id),
    )
    summary = EpisodeRegenSummary(
        dry_run=dry_run, scope=scope, backlog=len(backlog), stamp=stamp, readings=_census(rows)
    )
    planned = backlog if effective_limit <= 0 else backlog[:effective_limit]
    batches = _batches(planned, cfg.claims_per_call)
    if dry_run:
        whole = _batches(backlog, cfg.claims_per_call)
        summary.calls = len(whole)
        summary.remaining = len(backlog)
        summary.estimate = _estimate(batches)
        summary.backlog_estimate = _estimate(whole)
        if progress is not None:
            progress(f"Episode backlog: {len(backlog)} claims, {summary.calls} reader calls")
        return summary

    run_read: Reader = read or (lambda items: read_episodes(items, passage_chars=cfg.passage_chars))
    interval = 60.0 / rate if rate > 0 else 0.0
    passages = _PassageCache(session)
    since_commit = 0
    for n, batch in enumerate(batches, start=1):
        started = time.monotonic()
        items = [EpisodeItem(r.content, await passages.passage(r)) for r in batch]
        if progress is not None:
            progress(f"[{n}/{len(batches)}] {len(batch)} claims of {min(batch[0].entries)[:8]}…")
        reply: EpisodeReply | None
        try:
            reply = await run_read(items)
        except Exception as exc:  # noqa: BLE001 - one failed call never ends the run
            log.error("episode reader failed on %d claims: %s", len(batch), exc)
            reply = None
        summary.calls += 1
        summary.read += len(batch)
        verdicts = reply.verdicts if reply is not None else (None,) * len(batch)
        for row, verdict in zip(batch, verdicts, strict=False):
            if verdict is None or verdict is EpisodeVerdict.ABSTAIN or reply is None:
                summary.failed += 1
                continue
            written = await set_episode_reading(
                session,
                row.particle_id,
                verdict=verdict,
                rule=reply.stamp,
                model=reply.provider_model,
            )
            if not written:
                summary.skipped_operator += 1
                continue
            summary.written += 1
            if verdict is EpisodeVerdict.SESSION_BOUND:
                summary.session_bound += 1
            else:
                summary.standing += 1
        since_commit += len(batch)
        if since_commit >= cfg.commit_interval:
            await session.commit()
            since_commit = 0
        if interval > 0:
            delay = interval - (time.monotonic() - started)
            if delay > 0:
                await asyncio.sleep(delay)
    await session.commit()
    summary.remaining = max(0, len(backlog) - summary.written - summary.skipped_operator)
    return summary


# ---------------------------------------------------------------------------
# The labelling sheet
# ---------------------------------------------------------------------------


@dataclass
class SheetSummary:
    """What a sheet holds: per entry, the cards sampled of those it would withhold."""

    path: str
    entries: dict[str, dict[str, int]] = field(default_factory=dict)
    standing_sampled: int = 0

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready form for the verb's report."""
        return {
            "path": self.path,
            "entries": self.entries,
            "standing_sampled": self.standing_sampled,
        }


def _claim_view(row: EpisodeRow, claim: EpisodeClaim, passage: str) -> dict[str, object]:
    settled = claim.settled
    return {
        "id": row.particle_id,
        "text": row.content,
        "entry": min(row.entries) if row.entries else None,
        "passage": passage,
        "verdict": settled.verdict.value,
        "decided_by": str(settled.decided_by) if settled.decided_by is not None else None,
        "shadow": {str(s): v.value for s, v in settled.shadow},
    }


async def write_sheet(
    session: AsyncSession,
    path: Path,
    *,
    size: int | None = None,
    seed: int = 20261108,
    cards: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> SheetSummary:
    """Write the labelling sheet as JSON Lines: one line per sampled card or claim.

    For each ``(stratum, stamp combination)`` entry, a seeded random sample of
    ``size`` (default ``curation.episode.sheet_size``) of the cards it would
    withhold, each line carrying both claims, their passages and verdicts and
    an empty ``label`` for the owner: ``nothing`` (nothing to decide) or
    ``something`` (a side worth keeping as a standing fact). A last block
    samples ``size`` card sides read ``STANDING`` by a reading, labelled for
    recall only. Writes nothing to the store.
    """
    cfg = get_config().curation.episode
    n = cfg.sheet_size if size is None else size
    card_list = list(cards) if cards is not None else await latest_cards(session)
    ids = {pid for _, pids in card_list for pid in pids}
    claims, rows = await _asked(session, ids) if ids else ({}, {})

    by_entry: dict[str, list[tuple[str, Sequence[str]]]] = defaultdict(list)
    for kind, pids in card_list:
        sides = [claims.get(pid) for pid in pids]
        if kind not in WITHHELD_KINDS or any(s is None for s in sides):
            continue
        entry = fold_entry([s for s in sides if s is not None])
        if entry is not None:
            by_entry[entry.key].append((kind, pids))

    rng = random.Random(seed)
    passages = _PassageCache(session)
    summary = SheetSummary(path=str(path))
    lines: list[str] = []
    for key in sorted(by_entry):
        pool = sorted(by_entry[key], key=lambda c: (c[0], tuple(c[1])))
        sample = rng.sample(pool, min(n, len(pool)))
        summary.entries[key] = {"withheld": len(pool), "sampled": len(sample)}
        for kind, pids in sample:
            views = [
                _claim_view(rows[pid], claims[pid], await passages.passage(rows[pid]))
                for pid in pids
            ]
            lines.append(
                json.dumps(
                    {"sheet": "withheld", "entry": key, "kind": kind, "claims": views, "label": ""}
                )
            )
    standing = sorted(
        pid
        for pid, claim in claims.items()
        if claim.settled.verdict is EpisodeVerdict.STANDING
        and claim.settled.decided_by is not None
        and claim.settled.decided_by.name != "credit"
    )
    for pid in rng.sample(standing, min(n, len(standing))):
        view = _claim_view(rows[pid], claims[pid], await passages.passage(rows[pid]))
        lines.append(json.dumps({"sheet": "standing", "claims": [view], "label": ""}))
        summary.standing_sampled += 1
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return summary
