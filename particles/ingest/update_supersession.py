# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Same-subject update supersession — candidacy and the rung 2.5 inputs.

Two things the §6.6 ladder cannot do on its own:

* **Find the pair.** Extraction's own candidate search is per corpus entry and
  gated at ``extraction.similarity_threshold`` (0.80), so "the user lives in
  Denver" from today's session never meets "the user lives in Boston" from last
  week's: they are in different entries, and they score ~0.70 anyway. This
  module adds a second search, keyed on the claim's *about-subject*, across all
  entries, at a subject-scoped floor (:class:`SubjectIndex`,
  :func:`about_subject_name`).
* **Order the pair.** When the probe confirms the contradiction and trust
  cannot tell the sides apart, rung 2.5 prefers the strictly newer claim.
  :func:`update_order` decides whether the pair qualifies and in which
  direction; :func:`particles.core.conflict_resolution.resolve_conflict` applies
  it.

The about-subject is the entity a claim is *about*: the triple's
subject when the claim has one, else its sole subject. A particle's
``subject_ids`` also carry the entities it merely mentions ("moved from Boston
to Denver" names both cities), which must never become keys.

One subject is added to that key: the speaker's, for a claim that opens on a
speaker reference and carries the persona Subject (
:class:`SpeakerKey`). The extractor often leaves the speaker out of a
first-person claim's subjects, so "The user is not going back to Delhi" keyed
on Delhi alone and never met the speaker's earlier claims about Delhi.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import numpy as np

from particles.config import SubjectsConfig
from particles.core.generics import is_generic_claim
from particles.core.reanchor import is_reanchor_restatement
from particles.core.schema import (
    CorpusEntry,
    Particle,
    ProvenanceRefType,
    is_truth_apt,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import stance_holder
from particles.corpus.rule_sources import is_rule_source
from particles.embeddings import cosine_similarity
from particles.extraction.components import component_digest
from particles.extraction.general import CandidateParticle
from particles.extraction.journal import SOURCE_TYPE as JOURNAL_SOURCE_TYPE
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: The asserter-identity prefix of the agent write surface. A claim an
#: agent asserted is never retired by, nor retires, an extracted claim on the
#: strength of a timestamp (condition 4).
MCP_ASSERTER_PREFIX = "mcp:"


@dataclass(frozen=True)
class SpeakerKey:
    """The persona Subject a speaker-anchored claim is also keyed on.

    Built once per pass by the caller that builds the index
    (:func:`~particles.ingest.subject_resolver.load_speaker_key`) and passed in,
    so :func:`particle_subject_ids` stays a pure function of its arguments.
    """

    #: The persona Subject's id.
    subject_id: str
    #: Casefolded speaker references a claim may open on, longest first
    #: (:func:`speaker_forms`).
    forms: tuple[str, ...]
    #: Casefolded names that fold onto the persona in a persona source
    #: (:func:`persona_names`).
    names: frozenset[str]


def speaker_forms(cfg: SubjectsConfig) -> tuple[str, ...]:
    """The speaker references a first-person claim may open on.

    ``persona_canonical_name`` plus each ``persona_openers`` form that is also a
    persona alias, casefolded, longest first so "jeff gage" is tried before
    "jeff". An opener that is not an alias is ignored: the name §1 appends must
    fold onto the persona, or it would mint a Subject of its own. Empty when
    ``persona_openers`` is empty, which turns the rule off.
    """
    if not cfg.persona_openers:
        return ()
    aliases = {a.strip().casefold() for a in cfg.persona_aliases}
    forms = {cfg.persona_canonical_name.strip().casefold()}
    forms |= {o.strip().casefold() for o in cfg.persona_openers if o.strip().casefold() in aliases}
    return tuple(sorted((f for f in forms if f), key=lambda f: (-len(f), f)))


def persona_names(cfg: SubjectsConfig) -> frozenset[str]:
    """Casefolded names the persona fold maps onto the persona Subject."""
    return frozenset(
        {cfg.persona_canonical_name.strip().casefold()}
        | {a.strip().casefold() for a in cfg.persona_aliases}
    )


def speaker_openers_digest(forms: Collection[str]) -> str:
    """A digest of the speaker references in force, for run records.

    The key is derived at read time, so an edit to ``persona_openers`` rekeys
    every stored claim without writing anything. The sweep and the
    consolidation record carry this digest beside their pair counts, so a change
    in pairs can be traced to an opener edit.
    """
    return component_digest(*sorted(forms))


#: The component record name of the attach.
SPEAKER_COMPONENT = "attach.speaker"

#: Source types the attach never runs on. The journal extractor withholds the
#: author on purpose, behind the gate; attaching the
#: persona would cross that gate on its own.
SPEAKER_EXCLUDED_SOURCE_TYPES = frozenset({JOURNAL_SOURCE_TYPE})

_POSSESSIVES = ("'s", "\u2019s")


def speaker_opener(content: str, forms: Iterable[str]) -> str | None:
    """The speaker reference ``content`` opens on, spelled as in ``content``, or ``None``.

    ``forms`` are casefolded (:func:`speaker_forms`). A form matches at the
    start of the claim, case-insensitively, when whitespace or a possessive
    ``'s`` follows it: "The user's flat is in Bandra" opens on "The user",
    while "User accounts are locked" opens on nothing for the form "user", and
    "Sandeep's is near the user's flat" mentions the speaker without opening on
    them.
    """
    text = content.lstrip()
    for form in forms:
        head = text[: len(form)]
        if not form or head.casefold() != form:
            continue
        rest = text[len(form) :]
        if rest[:1].isspace() or (rest[:2] in _POSSESSIVES and not rest[2:3].isalnum()):
            return head
    return None


def attach_speaker(
    candidate: CandidateParticle, forms: Iterable[str], names: Collection[str]
) -> str | None:
    """Append the speaker reference a candidate opens on to its subjects.

    Returns the appended name, or ``None`` when the candidate opens on no
    speaker reference or already lists a name in ``names`` (one that folds onto
    the persona). The appended name is the surface form the claim used, so the
    persona fold, its alias record, ``bind_subject_id`` and the §6.6 precompute
    all see the name the extractor would have emitted. The caller decides that
    the source is a persona source and not a journal.
    """
    if any(n.strip().casefold() in names for n in candidate.subjects):
        return None
    spoken = speaker_opener(candidate.content, forms)
    if spoken is None:
        return None
    candidate.subjects.append(spoken)
    return spoken


def candidate_speaker_name(candidate: CandidateParticle, speaker: SpeakerKey | None) -> str | None:
    """The name a speaker-anchored candidate carries for the persona, or ``None``.

    Speaker-anchored: the claim opens on a speaker reference and one of its
    subject names folds onto the persona. The §6.6 precompute resolves the name
    and keys on it only when it resolves to ``speaker.subject_id``, so a
    candidate is keyed on the persona exactly when the stored claim it becomes
    would be (:func:`particle_subject_ids`).
    """
    if speaker is None or speaker_opener(candidate.content, speaker.forms) is None:
        return None
    return next((n for n in candidate.subjects if n.strip().casefold() in speaker.names), None)


def candidate_subject_names(candidate: CandidateParticle) -> list[str]:
    """The subject name a candidate is paired on — the *about*-subject, or none.

    The triple's subject when it names one of the candidate's subjects (the
    same case-insensitive match :func:`particles.extraction.structure.bind_subject_id`
    applies after resolution), else the candidate's sole subject.

    Candidacy spanned **every** subject a candidate named for one release.
    Its own live run withdrew it, and the correction there
    records the measurement: on a kept store the wider key produced 167
    qualifying pairs, spent the whole probe cap on them and demoted 3, because
    a restatement of the *same* value outranks a genuine old→new pair on
    cosine (0.99 against 0.6–0.8) and the probe answers NO to nearly all of
    them. The list shape is kept — it is the plural the callers were rebuilt
    around, and a re-ranked wider key is the open question — but it
    now carries at most one name.
    """
    names = [n.strip() for n in candidate.subjects if n.strip()]
    if candidate.structured_claim is not None:
        wanted = candidate.structured_claim.subject.value.strip().casefold()
        about = next((n for n in names if n.casefold() == wanted), None)
        if about is not None:
            return [about]
    return names if len(names) == 1 else []


def particle_subject_ids(particle: Particle, *, speaker: SpeakerKey | None) -> list[str]:
    """The subject ids a stored particle is paired on.

    Its about-subject, or none; plus the persona Subject when the claim is
    speaker-anchored, i.e. it opens on a speaker reference and carries the
    persona. ``speaker`` is required so every reader of the key
    decides it explicitly: a caller that forgot it would silently read the
    narrower key, and the write path and the sweep would disagree about a pair.
    Pass ``None`` only where the rule is off.
    """
    if particle.structured_claim is not None and particle.structured_claim.subject_id:
        keys = [particle.structured_claim.subject_id]
    else:
        keys = list(particle.subject_ids) if len(particle.subject_ids) == 1 else []
    if (
        speaker is not None
        and speaker.subject_id in particle.subject_ids
        and speaker.subject_id not in keys
        and speaker_opener(particle.content, speaker.forms) is not None
    ):
        keys.append(speaker.subject_id)
    return keys


def is_reconcilable(particle: Particle) -> bool:
    """Whether a stored particle may be a subject-scoped §6.6 candidate at all.

    The same exclusions the same-entry search applies: truth-apt only,
    no stances (a stance only ever pairs with a same-holder
    stance, which the same-entry search handles), and no DOCUMENT_META or
    non-asserted claims.
    """
    return (
        is_truth_apt(particle)
        and stance_holder(particle) is None
        and not is_excluded_document_meta(particle.properties)
        and not is_non_asserted(particle.properties)
    )


def is_extractor_asserted(particle: Particle) -> bool:
    """True for a machine-extracted claim — neither operator- nor agent-asserted.

    Operator beliefs carry ``HUMAN_REVIEW`` calibration (the guard
    reads the same field); agent writes carry ``AGENT_ASSERTED`` calibration and
    an ``mcp:`` asserter identity; an extracted claim has a SOURCE provenance ref.
    """
    if particle.confidence.calibration_source in (
        CalibrationSource.HUMAN_REVIEW,
        CalibrationSource.AGENT_ASSERTED,
    ):
        return False
    if particle.asserted_by.startswith(MCP_ASSERTER_PREFIX):
        return False
    return any(ref.type is ProvenanceRefType.SOURCE for ref in particle.provenance)


@dataclass
class SubjectIndex:
    """ACTIVE, reconcilable particles grouped by about-subject, with embeddings."""

    by_subject: dict[str, list[tuple[Particle, Embedding]]] = field(default_factory=dict)
    #: Particles retired earlier in the same pass — never offered again.
    retired: set[str] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        pairs: Iterable[tuple[Particle, Embedding | None]],
        *,
        speaker: SpeakerKey | None,
        exclude_ids: Iterable[str] = (),
    ) -> SubjectIndex:
        """Index ``pairs`` (typically every ACTIVE particle with an embedding).

        ``speaker`` is the persona key (see :func:`particle_subject_ids`).
        """
        excluded = set(exclude_ids)
        index = cls()
        for particle, emb in pairs:
            if emb is None or particle.id in excluded or not is_reconcilable(particle):
                continue
            for key in particle_subject_ids(particle, speaker=speaker):
                index.by_subject.setdefault(key, []).append((particle, emb))
        return index

    def candidates(
        self,
        subject_ids: Iterable[str],
        embedding: Embedding,
        *,
        floor: float,
        limit: int,
        skip_ids: Iterable[str] = (),
        generic: bool | None = None,
    ) -> list[Particle]:
        """Most similar particles sharing any of ``subject_ids``, best first.

        ``generic`` is whether the claim being matched is a generic claim
        (:func:`~particles.core.generics.is_generic_claim`), or ``None`` to skip
        the check. A particle of the other kind never enters the pool: a
        generic and an instance claim are not an adjudicable pair.
        """
        skip = set(skip_ids) | self.retired
        seen: dict[str, float] = {}
        pool: dict[str, Particle] = {}
        for subject_id in subject_ids:
            for particle, emb in self.by_subject.get(subject_id, []):
                if particle.id in skip or particle.id in seen:
                    continue
                score = cosine_similarity(embedding, emb)
                if score >= floor:
                    if generic is not None and is_generic_claim(particle.content) != generic:
                        skip.add(particle.id)
                        continue
                    seen[particle.id] = score
                    pool[particle.id] = particle
        best = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [pool[pid] for pid, _ in best]


def _snapshot_of(entry: CorpusEntry | None, snapshot_id: str | None) -> Any:
    if entry is None or not snapshot_id:
        return None
    return next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)


def _authority(uri: str | None) -> str | None:
    if not uri:
        return None
    return urlparse(uri).netloc or None


#: Tags that only the owner's own harvesters stamp (the Claude Code hooks and
#: the memory audit). Two local entries are one lineage only when both carry
#: one of these, or a recorded author (security review N6).
OWNER_HARVEST_TAGS = frozenset({"memory-file", "claude-code"})


def _local_lineage_shared(
    new_entry: CorpusEntry, new_snap: Any, existing_entry: CorpusEntry
) -> bool:
    """Whether two distinct entries already keyed as one lineage name one principal.

    An entry with a URL authority is identified by it, so the authority key is
    enough. A local entry (no authority) carries no principal of its own: it
    counts as the same lineage only when the snapshots record an author (the
    caller has checked the two match), the entries record contributors (likewise
    matched), or both carry the same owner-harvest tag.
    """
    if _authority(new_entry.uri_r) is not None:
        return True
    if new_snap.author_id or _contributors_key(new_entry):
        return True
    shared = set(new_entry.tags or []) & set(existing_entry.tags or [])
    return bool(shared & OWNER_HARVEST_TAGS)


def _contributors_key(entry: CorpusEntry) -> tuple[tuple[str, str], ...]:
    # Who and in what role — never *when*: the same contributor acting on two
    # dates is one lineage (ContributorRef.at defaults to the act's time).
    return tuple(sorted((c.id, c.role) for c in entry.contributors or []))


def _as_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def own_assertion_order(new_particle: Particle, existing: Particle) -> int | None:
    """Rung 2.5 input for the assertion pathway.

    ``+1`` when both claims were asserted by the **same agent identity** and
    the new one is strictly later, else ``None``. An agent revising its own
    earlier belief is what ``particle_supersede`` already lets it do
    explicitly (own-beliefs-only); this is the automatic form and
    grants no further reach. Every other pair — against an extracted claim, an
    operator's, or another agent's — returns ``None`` and keeps failing closed.
    Ordered by ``asserted_at``: an assertion has no source content date.
    """
    agent = CalibrationSource.AGENT_ASSERTED
    if new_particle.confidence.calibration_source is not agent:
        return None
    if existing.confidence.calibration_source is not agent:
        return None
    if not new_particle.asserted_by or new_particle.asserted_by != existing.asserted_by:
        return None
    new_at, old_at = _as_aware(new_particle.asserted_at), _as_aware(existing.asserted_at)
    return 1 if new_at > old_at else None


async def latest_source_date(
    session: Any,
    particle: Particle,
    cache: dict[str, CorpusEntry | None] | None = None,
) -> datetime | None:
    """The date of ``particle``'s **latest** observation, across every SOURCE ref.

    A claim restated later keeps its original ref and *gains* another, in that
    later session's corpus entry (folds a re-observation into the
    particle it matches rather than minting a copy). Dating such a particle by
    its first ref makes a value that is current again look stale: measured on a
    rot world, where the sweep retired a reverted value because its
    first observation predated the value it had replaced. Every ref is
    consulted, each in its own entry, and the newest wins.
    """
    from particles.corpus.store import get_entry

    store = cache if cache is not None else {}
    dates: list[datetime] = []
    for ref in particle.provenance:
        if ref.type is not ProvenanceRefType.SOURCE:
            continue
        if ref.corpus_entry_id not in store:
            store[ref.corpus_entry_id] = await get_entry(session, ref.corpus_entry_id)
        entry = store[ref.corpus_entry_id]
        if entry is None:
            continue
        snap = next((s for s in entry.snapshots if s.snapshot_id == ref.snapshot_id), None)
        if snap is not None and snap.content_published_at is not None:
            dates.append(_as_aware(snap.content_published_at))
    return max(dates) if dates else None


def update_order(
    new_particle: Particle,
    new_entry: CorpusEntry | None,
    new_snapshot_id: str | None,
    existing: Particle,
    existing_entry: CorpusEntry | None,
    existing_snapshot_id: str | None,
    *,
    require_attribution: bool = False,
    new_date: datetime | None = None,
    existing_date: datetime | None = None,
) -> int | None:
    """The rung 2.5 input for one confirmed pair: ``+1``, ``-1``, or ``None``.

    ``+1`` when ``new`` is the strictly newer claim, ``-1`` when ``existing`` is,
    and ``None`` whenever the pair does not qualify:

    * either side is operator- or agent-asserted (:func:`is_extractor_asserted`);
    * either side is a restatement the re-anchor pass wrote, which is dated
      history and never an old or a new value (guard 1);
    * the sides are distinguishable to trust — a different URL authority (the
      rule key), source type, snapshot author (the §6.4 AUTHOR tier),
      or contributor attribution. Only the entry itself may differ; a
      CORPUS_ENTRY-tier trust statement that separates two entries is rung 2's
      to act on, which runs first;
    * either side is a rule file (``rule-file``) and the two sides are
      different entries. Every rule file is a ``file://`` Markdown document, so
      the keys above cannot tell the owner's ``CLAUDE.md`` from a cloned
      repository's ``AGENTS.md``, whose authors are not the owner's lineage
      (security review D13). A rule file updates only its own earlier
      snapshots; a later claim from any other entry is held for review;
    * the sides are two different local entries (no URL authority, such as two
      ``file://`` Markdown files) that share neither an owner-harvest tag
      (:data:`OWNER_HARVEST_TAGS`) nor a recorded author. A local path names no
      principal, so a cloned vault or repository's notes would otherwise be one
      lineage with the owner's memory files, and a self-dated third-party file
      could retire the owner's claim (security review N6);
    * either side's source content date is unknown, or the dates are equal.
      ``new_date`` / ``existing_date`` override the snapshot's own date, which
      is how a caller supplies the **latest** observation of a re-stated claim
      (:func:`source_dates`) rather than its first;
    * ``require_attribution`` (the ``multi`` regime) and the
      lineage is anonymous — no snapshot author and no contributors on either
      side. In a shared store two unattributed claims cannot be assumed to
      share a principal, so the rung fails closed rather than letting one
      contributor's date retire another's belief.
    """
    if not (is_extractor_asserted(new_particle) and is_extractor_asserted(existing)):
        return None
    if is_reanchor_restatement(new_particle) or is_reanchor_restatement(existing):
        return None
    if new_entry is None or existing_entry is None:
        return None
    new_snap = _snapshot_of(new_entry, new_snapshot_id)
    old_snap = _snapshot_of(existing_entry, existing_snapshot_id)
    if new_snap is None or old_snap is None:
        return None
    same_lineage = (
        _authority(new_entry.uri_r) == _authority(existing_entry.uri_r)
        and new_entry.source_type == existing_entry.source_type
        and new_snap.author_id == old_snap.author_id
        and _contributors_key(new_entry) == _contributors_key(existing_entry)
    )
    if not same_lineage:
        return None
    if new_entry.entry_id != existing_entry.entry_id and (
        is_rule_source(new_entry) or is_rule_source(existing_entry)
    ):
        return None
    if new_entry.entry_id != existing_entry.entry_id and not _local_lineage_shared(
        new_entry, new_snap, existing_entry
    ):
        return None
    if require_attribution and not (new_snap.author_id or _contributors_key(new_entry)):
        return None
    new_at = new_date or new_snap.content_published_at
    old_at = existing_date or old_snap.content_published_at
    if new_at is None or old_at is None:
        return None
    new_at, old_at = _as_aware(new_at), _as_aware(old_at)
    if new_at > old_at:
        return 1
    if new_at < old_at:
        return -1
    return None
