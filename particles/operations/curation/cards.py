"""The normalized curation card shape (§2).

Every finder's native output projects into one ``CurationCard``; each
``CardKind`` maps 1:1 to one existing finder, so the queue is a *projection* of
work the store already knows about, not a new analysis. The card's ``key`` is a
stable, finder-output-derived identity used by snooze / affirm filtering
.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from particles.core.schema import JudgeVerdictKind


class CardKind(StrEnum):
    """One kind per card-producing finder."""

    STALE = "stale"  # lint STALENESS
    RETRACTION_CASCADE = "retraction_cascade"  # lint RETRACTION_CASCADE
    BROKEN_PROVENANCE = "broken_provenance"  # lint CORPUS_LINK_INTEGRITY
    CONFIDENCE_DECAY = "confidence_decay"  # lint CONFIDENCE_DECAY
    RECENCY_DECAY = "recency_decay"  # lint RECENCY_DECAY ()
    CONTRADICTION = "contradiction"  # lint CONTRADICTION (semantic)
    CONTESTED = "contested"  # lint CONTESTED (was get_inconsistency_backrefs)
    NO_SUBJECT = "no_subject"  # lint NO_SUBJECT
    DUPLICATE_PAIR = "duplicate_pair"  # links suggest (REPORT)
    UNCITED_URL = "uncited_url"  # corpus links suggest
    FAILED_SNAPSHOTS = "failed_snapshots"  # quality report (batch)
    PROPOSED_ABSTRACTION = "proposed_abstraction"  # candidate events


# The gestures each kind offers. The verbs that need operator
# content or judgment (supersede / edit / comment / reindex) are *surfaced* —
# the card shows the resolving command — while the safe, card-resolvable ones
# (affirm / snooze / dismiss / retract / merge / deposit) dispatch directly.
_GESTURES: dict[CardKind, tuple[str, ...]] = {
    CardKind.STALE: ("affirm", "supersede", "retract", "snooze"),
    CardKind.RETRACTION_CASCADE: ("supersede", "retract", "snooze"),
    CardKind.BROKEN_PROVENANCE: ("supersede", "retract", "snooze"),
    CardKind.CONFIDENCE_DECAY: ("affirm", "supersede", "snooze"),
    # shared-seam extension: the age-discount finding becomes
    # a card, mirroring confidence_decay (an aged belief is re-affirmed,
    # superseded by a fresher claim, or snoozed).
    CardKind.RECENCY_DECAY: ("affirm", "supersede", "snooze"),
    CardKind.CONTRADICTION: ("comment", "supersede", "retract", "snooze"),
    CardKind.CONTESTED: ("comment", "affirm", "snooze"),
    # the orphan-claim card. assign-subject (the provenance-preserving
    # operator-supersede) is the resolving write; supersede / retract / snooze
    # are the standard fallbacks for a spurious orphan.
    CardKind.NO_SUBJECT: ("assign-subject", "supersede", "retract", "snooze"),
    CardKind.DUPLICATE_PAIR: ("merge", "dismiss", "snooze"),
    CardKind.UNCITED_URL: ("deposit", "dismiss"),
    CardKind.FAILED_SNAPSHOTS: ("reindex", "snooze"),
    # propose mode: accept asserts the candidate abstraction with
    # its premise links; reject records the verdict (a labelled datapoint for
    # the §8 faithfulness evaluation). Both resolve via ABSTRACTION_RESOLVED.
    CardKind.PROPOSED_ABSTRACTION: ("accept", "reject", "snooze"),
}


def gestures_for(kind: CardKind) -> list[str]:
    """The gesture names a card of this kind offers."""
    return list(_GESTURES[kind])


def contested_gestures(bases: Sequence[str]) -> list[str]:
    """The gestures a CONTESTED card offers given the bases that fired.

    The curation rule offers ``comment`` only on cards "backed by an INCONSISTENCY
    §9.6 resolves", because ``review.resolve`` is the sole annotation primitive
    and it needs an INCONSISTENCY to act on. Once the class widened past that
    one basis, a stance- or divergence-only card has nothing for
    `review` to resolve — so ``comment`` is withheld there rather than routing
    the operator to a verb that cannot act. ``_GESTURES`` keeps the superset.
    """
    return [g for g in _GESTURES[CardKind.CONTESTED] if g != "comment" or "inconsistency" in bases]


class ParticleBrief(BaseModel):
    """A compact summary of one particle a card references.

    Just enough context to judge the card's gesture — the claim ``content``, the
    ``subject_labels`` it attaches to, its query-time ``effective_confidence``
    , and its ``status`` — so a client (the bus-stop PWA) can
    decide *which* of a duplicate pair to keep without a second
    ``particles particle show <id>`` round-trip. Populated server-side when the queue is
    built; never stored. Deliberately **not** the full ``Particle`` (the
    provenance/expand view is deferred).
    """

    particle_id: str
    content: str
    subject_labels: list[str] = Field(default_factory=list)
    effective_confidence: float
    status: str


class DuplicateVerdict(BaseModel):
    """The LLM judge's read on a duplicate-pair card.

    Carried on a ``DUPLICATE_PAIR`` card when the queue ran the duplicate finder
    in ``LLM_JUDGE`` mode (``semantic=True``): the per-pair same-claim
    ``verdict`` (``PARAPHRASE`` = same claim, safe to merge; ``DISTINCT`` = not a
    duplicate; ``UNSURE`` = ambiguous) plus its short ``rationale`` when the
    candidate exposes one. **Advisory** — it informs the operator's Merge /
    Dismiss gesture; it never mutates the store. ``None`` on
    every non-duplicate card and on duplicate cards built in ``REPORT`` mode
    (``semantic=False`` or the LLM unavailable).
    """

    verdict: JudgeVerdictKind
    rationale: str | None = None


class CurationCard(BaseModel):
    """One bite-sized curation task, normalized from a finder's output."""

    kind: CardKind
    # The belief(s) the card is about — a pair for duplicate_pair / contradiction,
    # the flagged belief for contested, empty for uncited_url / failed_snapshots.
    particle_ids: list[str] = Field(default_factory=list)
    # Subjects touched (set for duplicate_pair).
    subject_ids: list[str] = Field(default_factory=list)
    # Set for uncited_url.
    corpus_url: str | None = None
    # The finder's reason string + any evidence (e.g. the partner particle).
    diagnostic: str
    # The gesture names that resolve this card (§4).
    suggested_gestures: list[str] = Field(default_factory=list)
    # Per-particle briefs aligned to particle_ids, populated server-side when the
    # queue is built so a client can judge the card without a second
    # round-trip. Empty for cards with no particle (uncited_url / failed_snapshots).
    particles: list[ParticleBrief] = Field(default_factory=list)
    # The LLM judge's same-claim verdict for a DUPLICATE_PAIR card built in
    # LLM_JUDGE mode (semantic=True). None for every other kind, and
    # for duplicate cards built in REPORT mode / when the LLM was unavailable.
    # Advisory only — it informs the merge decision and demotes a DISTINCT card
    # in leverage; it never participates in key / from_key (snooze identity).
    verdict: DuplicateVerdict | None = None
    # the ABSTRACTION_CANDIDATE event a PROPOSED_ABSTRACTION card
    # fronts. The event payload is the candidate's persistence — the accept /
    # reject gestures re-read it by this id. None for every other kind.
    candidate_event_id: str | None = None
    # the bases that fired on a CONTESTED card, in the badge's
    # canonical order (stance, divergence, inconsistency). Carried structurally
    # rather than only in `diagnostic` so the census and the
    # run record can break the class down by basis instead of reporting one
    # unattributed total. None for every other kind. Never part of `key` — the
    # snooze identity stays basis-free so a suppression survives a basis change.
    contested_bases: list[str] | None = None
    # The full id of the open INCONSISTENCY behind a CONTESTED card whose
    # ``inconsistency`` basis fired (the ``diagnostic`` prose truncates it).
    # Lets a client link the card straight to the contradiction's evidence —
    # the ``scope=inconsistency`` graph render. None otherwise.
    # Never part of ``key`` (same rule as ``contested_bases``).
    inconsistency_id: str | None = None
    # The leverage score (§2), filled by leverage.score_cards.
    leverage: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> str:
        """Stable identity for snooze / affirm filtering.

        Finder-output-derived, so the same underlying problem yields the same
        key across sessions. Serialized into the queue response (a
        ``@computed_field``) so a client (PWA / Obsidian) can echo the exact
        ``card_key`` back to ``POST /curation/affirm`` / ``snooze`` —
        the snooze/affirm filter matches on it.
        """
        if self.kind is CardKind.UNCITED_URL:
            return f"uncited_url:{self.corpus_url}"
        if self.kind is CardKind.FAILED_SNAPSHOTS:
            return "failed_snapshots"
        if self.kind is CardKind.PROPOSED_ABSTRACTION:
            return f"proposed_abstraction:{self.candidate_event_id}"
        return f"{self.kind.value}:" + "|".join(sorted(self.particle_ids))

    @classmethod
    def from_key(cls, key: str) -> CurationCard:
        """Reconstruct a minimal card from its ``key`` (the apply-a-gesture path).

        The key round-trips the kind plus the particle ids / URL — enough for
        ``apply_gesture`` to dispatch (the underlying write op re-validates the
        target). ``diagnostic`` and ``leverage`` are not recoverable from the key
        and are left empty. Raises ``ValueError`` on an unparseable key.
        """
        if key == "failed_snapshots":
            return cls(
                kind=CardKind.FAILED_SNAPSHOTS,
                diagnostic="",
                suggested_gestures=gestures_for(CardKind.FAILED_SNAPSHOTS),
            )
        prefix, sep, rest = key.partition(":")
        if not sep:
            raise ValueError(f"Unrecognized card key {key!r}.")
        if prefix == CardKind.UNCITED_URL.value:
            return cls(
                kind=CardKind.UNCITED_URL,
                corpus_url=rest,
                diagnostic="",
                suggested_gestures=gestures_for(CardKind.UNCITED_URL),
            )
        if prefix == CardKind.PROPOSED_ABSTRACTION.value:
            return cls(
                kind=CardKind.PROPOSED_ABSTRACTION,
                candidate_event_id=rest,
                diagnostic="",
                suggested_gestures=gestures_for(CardKind.PROPOSED_ABSTRACTION),
            )
        try:
            kind = CardKind(prefix)
        except ValueError as exc:
            raise ValueError(f"Unrecognized card key {key!r}.") from exc
        return cls(
            kind=kind,
            particle_ids=[s for s in rest.split("|") if s],
            diagnostic="",
            suggested_gestures=gestures_for(kind),
        )
