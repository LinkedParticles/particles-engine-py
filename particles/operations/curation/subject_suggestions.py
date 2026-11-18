# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Suggest the subjects a belief is about.

The assign-subject form helps only an operator who already knows a subject's
name. This module offers the likely subjects before anything is typed, from two
signals the store already holds, with no model call:

- **Named in the claim.** A subject whose canonical name or alias appears in the
  claim as a run of whole words (:func:`claim_tokens`, :func:`name_keys`).
- **Carried by similar beliefs.** The subjects linked to the claim's nearest
  ACTIVE neighbours by embedding (:func:`neighbour_support`).

:func:`rank_suggestions` merges the two over plain values; the shell,
:func:`suggest_subjects`, gathers them. Nothing is written: a suggestion is
applied only when the operator picks it on the form.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Particle, ProvenanceRefType, Subject
from particles.corpus.store import get_entry
from particles.embeddings import stacked_cosine_similarities
from particles.operations.query.observer_scope import filter_visible
from particles.store.candidate_cache import load_candidate_matrix, load_candidate_vectors
from particles.store.subject_store import (
    ParticleSubjectRow,
    SubjectAliasRow,
    SubjectRow,
    alias_key,
    persona_alias_guard,
)

#: Characters a token loses at either end, until nothing more comes off.
_STRIP = "\"'`()[]{},;:!?*“”‘’"
#: The longest name, in words, looked up; covers all but 16 of the 17,660
#: names on the owner's store (2026-11-08).
MAX_NAME_WORDS = 10
#: A run shorter than this many characters is never looked up.
MIN_NAME_CHARS = 3
#: Named subjects placed ahead of the neighbour signal (§2c).
NAMED_FIRST = 3
#: The most suggestions a request may ask for.
LIMIT_CLAMP = 20
#: Ids per ``IN`` list.
_IN_CHUNK = 500
#: Neighbours read per round while looking for ones that carry a subject.
_NEIGHBOUR_ROUND = 200


@dataclass(frozen=True, slots=True)
class NamedMatch:
    """A subject the claim names: the longest matched run, and its linked beliefs."""

    subject_id: str
    length: int
    degree: int


@dataclass(frozen=True, slots=True)
class NeighbourSupport:
    """A subject carried by similar beliefs: summed similarity, and how many carry it."""

    subject_id: str
    score: float
    count: int


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One suggested subject and why it was suggested."""

    subject_id: str
    named: bool
    neighbours: int


def claim_tokens(text: str) -> list[str]:
    """The claim's whitespace tokens, each stripped of surrounding punctuation.

    Quotes, brackets, commas, colons, semicolons, question and exclamation
    marks, asterisks and backticks come off either end, and full stops off the
    end, repeated until nothing more comes off: ``trust'.`` gives ``trust``.
    """
    out: list[str] = []
    for raw in text.split():
        token, previous = raw, None
        while token != previous:
            previous = token
            token = token.strip(_STRIP).rstrip(".")
        if token:
            out.append(token)
    return out


def name_keys(tokens: Sequence[str]) -> list[str]:
    """Every run of one to ten tokens, folded for lookup, longest runs first.

    A run shorter than three characters is left out, and each key appears once.
    """
    keys: dict[str, None] = {}
    for n in range(min(MAX_NAME_WORDS, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            key = alias_key(" ".join(tokens[i : i + n]))
            if len(key) >= MIN_NAME_CHARS:
                keys.setdefault(key, None)
    return list(keys)


def neighbour_support(
    neighbours: Iterable[tuple[float, Collection[str]]], min_support: int
) -> list[NeighbourSupport]:
    """Score the subjects the neighbouring beliefs carry, best first.

    Each neighbour is its similarity and its subjects. A subject scores the sum
    of the similarities of the neighbours carrying it and is kept only when at
    least ``min_support`` of them carry it. Ties go to the subject on more
    neighbours, then to its id, so the order is stable.
    """
    score: dict[str, float] = {}
    count: dict[str, int] = {}
    for similarity, subjects in neighbours:
        for sid in subjects:
            score[sid] = score.get(sid, 0.0) + similarity
            count[sid] = count.get(sid, 0) + 1
    kept = [
        NeighbourSupport(sid, score[sid], count[sid]) for sid in score if count[sid] >= min_support
    ]
    return sorted(kept, key=lambda s: (-s.score, -s.count, s.subject_id))


def rank_suggestions(
    named: Sequence[NamedMatch],
    neighbours: Sequence[NeighbourSupport],
    *,
    exclude: Collection[str],
    limit: int,
) -> list[Suggestion]:
    """Merge the two signals into the list the form shows.

    Up to three named subjects come first, longer names first and, among names
    of one length, the subject on fewer beliefs first; then the
    neighbour subjects by score; then the remaining named ones. A subject in
    ``exclude`` (one the belief already carries) never appears, and a subject
    both signals found appears once, with both its reasons.
    """
    by_name = sorted(named, key=lambda m: (-m.length, m.degree, m.subject_id))
    named_ids = [m.subject_id for m in by_name if m.subject_id not in exclude]
    support = {s.subject_id: s.count for s in neighbours if s.subject_id not in exclude}
    order = named_ids[:NAMED_FIRST] + list(support) + named_ids[NAMED_FIRST:]
    named_set = set(named_ids)
    out: list[Suggestion] = []
    seen: set[str] = set()
    for sid in order:
        if sid in seen:
            continue
        seen.add(sid)
        out.append(Suggestion(sid, sid in named_set, support.get(sid, 0)))
        if len(out) >= limit:
            break
    return out


# --- The shell ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubjectSuggestion:
    """A suggestion with its subject, as the route returns it."""

    subject: Subject
    named: bool
    neighbours: int


def _chunks(items: Sequence[str]) -> Iterable[list[str]]:
    for i in range(0, len(items), _IN_CHUNK):
        yield list(items[i : i + _IN_CHUNK])


async def _source_facts(session: AsyncSession, particle: Particle) -> tuple[str | None, list[str]]:
    """The type and tags of the belief's first SOURCE corpus entry, for the persona guard."""
    ref = next((r for r in particle.provenance if r.type == ProvenanceRefType.SOURCE), None)
    if ref is None or not ref.corpus_entry_id:
        return None, []
    entry = await get_entry(session, ref.corpus_entry_id)
    if entry is None:
        return None, []
    return entry.source_type, list(entry.tags)


async def _named_matches(session: AsyncSession, particle: Particle) -> list[NamedMatch]:
    """The subjects whose canonical name or alias the claim spells (§2a)."""
    keys = name_keys(claim_tokens(particle.content))
    if not keys:
        return []
    source_type, source_tags = await _source_facts(session, particle)
    longest: dict[str, int] = {}
    for chunk in _chunks(keys):
        # Canonical names compare as the resolver compares them (find_by_name).
        canonical = await session.execute(
            select(SubjectRow.id, func.lower(SubjectRow.canonical_name)).where(
                func.lower(SubjectRow.canonical_name).in_(chunk)
            )
        )
        for sid, key in canonical.all():
            longest[sid] = max(longest.get(sid, 0), len(key))
        aliased = await session.execute(
            select(SubjectAliasRow.subject_id, SubjectAliasRow.alias_key, SubjectRow.canonical_name)
            .join(SubjectRow, SubjectRow.id == SubjectAliasRow.subject_id)
            .where(SubjectAliasRow.alias_key.in_(chunk))
        )
        for sid, key, canonical_name in aliased.all():
            # A persona form ("I", "the user") answers through the persona
            # subject's aliases only inside a persona source, exactly where the
            # resolver would bind it.
            skip = {c.lower() for c in persona_alias_guard(key, source_type, source_tags)}
            if canonical_name.lower() in skip:
                continue
            longest[sid] = max(longest.get(sid, 0), len(key))
    if not longest:
        return []
    degree = await _link_degree(session, list(longest))
    return [NamedMatch(sid, length, degree.get(sid, 0)) for sid, length in longest.items()]


async def _link_degree(session: AsyncSession, subject_ids: Sequence[str]) -> dict[str, int]:
    """How many beliefs, of any status, each subject is linked to.

    The tie-break among names of one length. Counted from the link table's
    index alone: restricting it to ACTIVE beliefs joined every link to its
    belief, 130 to 160 ms for a subject on thousands of beliefs (2026-11-08).
    """
    out: dict[str, int] = {}
    for chunk in _chunks(subject_ids):
        rows = await session.execute(
            select(ParticleSubjectRow.subject_id, func.count())
            .where(ParticleSubjectRow.subject_id.in_(chunk))
            .group_by(ParticleSubjectRow.subject_id)
        )
        out.update({sid: int(n) for sid, n in rows.all()})
    return out


async def _links_of(session: AsyncSession, particle_ids: Sequence[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for chunk in _chunks(particle_ids):
        rows = await session.execute(
            select(ParticleSubjectRow.particle_id, ParticleSubjectRow.subject_id).where(
                ParticleSubjectRow.particle_id.in_(chunk)
            )
        )
        for pid, sid in rows.all():
            out.setdefault(pid, set()).add(sid)
    return out


async def _neighbour_support(
    session: AsyncSession,
    particle_id: str,
    *,
    observer_project: str | None,
    k: int,
    min_support: int,
) -> list[NeighbourSupport]:
    """The subjects the belief's nearest linked ACTIVE beliefs carry (§2b)."""
    own = (await load_candidate_vectors(session, [particle_id])).get(particle_id)
    if own is None:
        # No vector, or one of another model: named suggestions only.
        return []
    matrix = await load_candidate_matrix(session)
    sims = stacked_cosine_similarities(own, matrix.vectors, matrix.norms)
    eligible = np.ones(len(matrix.particles), dtype=bool)
    if observer_project is not None:
        scope = await filter_visible(session, matrix.particles, observer_project)
        eligible = np.fromiter(
            (p.id in scope.visible_ids for p in matrix.particles), dtype=bool, count=len(eligible)
        )
    sims = np.where(eligible & (sims > 0.0), sims, -1.0)
    order = np.argsort(-sims, kind="stable")
    found: list[tuple[float, Collection[str]]] = []
    for start in range(0, len(order), _NEIGHBOUR_ROUND):
        rows = [
            int(i)
            for i in order[start : start + _NEIGHBOUR_ROUND]
            if sims[i] > 0.0 and matrix.particles[i].id != particle_id
        ]
        if not rows:
            break
        links = await _links_of(session, [matrix.particles[i].id for i in rows])
        for i in rows:
            subjects = links.get(matrix.particles[i].id)
            if subjects:
                found.append((float(sims[i]), subjects))
                if len(found) >= k:
                    return neighbour_support(found, min_support)
    return neighbour_support(found, min_support)


async def suggest_subjects(
    session: AsyncSession,
    particle: Particle,
    *,
    limit: int | None = None,
    observer_project: str | None = None,
) -> list[SubjectSuggestion]:
    """The subjects most likely to be what ``particle`` is about, best first.

    ``limit`` defaults to ``curation.subject_suggestions.limit`` and is clamped
    to 20. A subject the belief already carries is never suggested. The
    neighbours are read through the project observer when ``observer_project``
    is given; the curation clients send none.
    """
    cfg = get_config().curation.subject_suggestions
    wanted = max(1, min(LIMIT_CLAMP, limit if limit is not None else cfg.limit))
    carried = set(
        (
            await session.execute(
                select(ParticleSubjectRow.subject_id).where(
                    ParticleSubjectRow.particle_id == particle.id
                )
            )
        ).scalars()
    )
    named = await _named_matches(session, particle)
    neighbours = await _neighbour_support(
        session,
        particle.id,
        observer_project=observer_project,
        k=cfg.neighbours,
        min_support=cfg.min_support,
    )
    ranked = rank_suggestions(named, neighbours, exclude=carried, limit=wanted)
    if not ranked:
        return []
    rows = await session.execute(
        select(SubjectRow).where(SubjectRow.id.in_([s.subject_id for s in ranked]))
    )
    subjects = {row.id: row.to_model() for row in rows.scalars()}
    return [
        SubjectSuggestion(subjects[s.subject_id], s.named, s.neighbours)
        for s in ranked
        if s.subject_id in subjects
    ]
