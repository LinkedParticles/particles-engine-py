# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The normalized curation card shape (§2).

Every finder's native output projects into one ``CurationCard``; each
``CardKind`` maps 1:1 to one existing finder, so the queue is a *projection* of
work the store already knows about, not a new analysis. The card's ``key`` is a
stable, finder-output-derived identity used by snooze / affirm filtering.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from particles.config import get_config
from particles.core.conflict_review import is_quarantined
from particles.core.schema import JudgeVerdictKind, Particle
from particles.core.status import Status


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
    GATED_SUBJECTS = "gated_subjects"  # NO_SUBJECT orphans a relink recovers (batch)
    DUPLICATE_PAIR = "duplicate_pair"  # links suggest (REPORT)
    UNCITED_URL = "uncited_url"  # corpus links suggest
    FAILED_SNAPSHOTS = "failed_snapshots"  # quality report (batch)
    PROPOSED_ABSTRACTION = "proposed_abstraction"  # candidate events
    STALE_BASIS = "stale_basis"  # unrestated re-anchor events
    INCONSISTENCY = "inconsistency"  # lint OPEN_INCONSISTENCY, one per open record
    DEMOTION = "demotion"  # a claim retired as replaced, with its replacement


# The gestures each kind offers. The verbs another command
# resolves (edit / reindex) are *surfaced*: the card shows the resolving
# command. Every other gesture dispatches directly, supersede with the
# operator's replacement belief as flags, and resolve with the
# review action.
_GESTURES: dict[CardKind, tuple[str, ...]] = {
    CardKind.STALE: ("affirm", "supersede", "retract", "snooze"),
    CardKind.RETRACTION_CASCADE: ("supersede", "retract", "snooze"),
    CardKind.BROKEN_PROVENANCE: ("supersede", "retract", "snooze"),
    CardKind.CONFIDENCE_DECAY: ("affirm", "supersede", "snooze"),
    # shared-seam extension: the age-discount finding becomes
    # a card, mirroring confidence_decay (an aged belief is re-affirmed,
    # superseded by a fresher claim, or snoozed).
    CardKind.RECENCY_DECAY: ("affirm", "supersede", "snooze"),
    # `comment` left every kind. A contradiction card is an
    # unrecorded pair with no record for `review` to resolve, and a conflict
    # with a record has its own card, which resolves it directly.
    CardKind.CONTRADICTION: ("supersede", "retract", "snooze"),
    # a CONTESTED card now fires only on the observer-signal
    # bases (stance, divergence); its conflicts, if any, have their own cards.
    CardKind.CONTESTED: ("affirm", "snooze"),
    # the orphan-claim card. assign-subject (the provenance-preserving
    # operator-supersede) is the resolving write; supersede / retract / snooze
    # are the standard fallbacks for a spurious orphan.
    CardKind.NO_SUBJECT: ("assign-subject", "supersede", "retract", "snooze"),
    # every orphan whose gated subject names the relink can
    # recover, as one card. relink applies the accepted tiers in place.
    CardKind.GATED_SUBJECTS: ("relink", "snooze"),
    CardKind.DUPLICATE_PAIR: ("merge", "dismiss", "snooze"),
    CardKind.UNCITED_URL: ("deposit", "dismiss"),
    CardKind.FAILED_SNAPSHOTS: ("reindex", "snooze"),
    # propose mode: accept asserts the candidate abstraction with
    # its premise links; reject records the verdict (a labelled datapoint for
    # the §8 faithfulness evaluation). Both resolve via ABSTRACTION_RESOLVED.
    CardKind.PROPOSED_ABSTRACTION: ("accept", "reject", "snooze"),
    # a belief that relied on a state an update retired, which the
    # re-anchor pass could not restate safely. affirm keeps it as it stands;
    # supersede / retract replace or withdraw it.
    CardKind.STALE_BASIS: ("affirm", "supersede", "retract", "snooze"),
    # an open conflict is resolved or put off. affirm and dismiss
    # would hide the record while leaving it open; a real "not a conflict" or
    # "neither is worth keeping" is a resolution (BOTH_VALID / DISCARD).
    CardKind.INCONSISTENCY: ("resolve", "snooze"),
    # A claim a later one retired as its replacement. Affirm rules
    # the replacement correct, dismiss rules that both claims hold, and each
    # ruling is kept as a labelled benchmark pair. Neither gesture changes a
    # status, because a demotion encodes a judgment.
    CardKind.DEMOTION: ("affirm", "dismiss", "snooze"),
}

#: The review actions a conflict card can offer, in menu order.
#: DEFER is not among them: it leaves the record open, so the card offers
#: `snooze` for "decide later", and DEFER stays available to `curate apply
#: resolve --action DEFER` for recording a note.
RESOLVE_ACTIONS: tuple[str, ...] = ("PREFER_A", "PREFER_B", "BOTH_VALID", "DISCARD")


def gestures_for(kind: CardKind) -> list[str]:
    """The gesture names a card of this kind offers."""
    return list(_GESTURES[kind])


# A human title per kind, and the question each kind puts to the curator: what
# a listing leads with so the operator knows what is being asked before reading
# the evidence.
KIND_TITLES: dict[CardKind, tuple[str, str]] = {
    CardKind.STALE: (
        "Expired belief",
        "The belief's stated validity window (valid_until) has passed. Is it still true?",
    ),
    CardKind.RETRACTION_CASCADE: (
        "Rests on a retracted belief",
        "A belief this one depends on was retracted or superseded. Does it still stand?",
    ),
    CardKind.BROKEN_PROVENANCE: (
        "Broken provenance",
        "The source snapshot this belief cites no longer exists. Can it still be trusted?",
    ),
    CardKind.CONFIDENCE_DECAY: (
        "Uncertain confidence",
        "The confidence estimate for this belief has a high variance. "
        "Is it sound as stated, or should it be replaced with a firmer claim?",
    ),
    CardKind.RECENCY_DECAY: (
        "Aging belief",
        "The belief is old enough that its confidence is discounted. Is it still current?",
    ),
    CardKind.CONTRADICTION: (
        "Contradiction",
        "The belief contradicts another active belief. Which one is right?",
    ),
    CardKind.CONTESTED: (
        "Contested belief",
        "Sources or trust policies disagree about this belief. Does it stand?",
    ),
    CardKind.NO_SUBJECT: (
        "Belief with no subject",
        "The belief is attached to no subject, so subject-scoped reads never see it. "
        "Which subject is it about?",
    ),
    CardKind.GATED_SUBJECTS: (
        "Subjects the gate withheld",
        "These beliefs name files, records, identifiers or commands the extraction gate "
        "withheld as subjects. Link them to project-scoped subjects in one pass?",
    ),
    CardKind.DUPLICATE_PAIR: (
        "Possible duplicate",
        "Are these two beliefs the same claim?",
    ),
    CardKind.UNCITED_URL: (
        "Frequently cited, never deposited",
        "Several sources cite this URL, but it is not in the corpus. Deposit it?",
    ),
    CardKind.FAILED_SNAPSHOTS: (
        "Failed extractions",
        "Some corpus snapshots produced no beliefs because extraction failed.",
    ),
    CardKind.PROPOSED_ABSTRACTION: (
        "Proposed generalization",
        "The consolidation cycle proposes a general belief over the specifics below. "
        "Is it faithful to them?",
    ),
    CardKind.INCONSISTENCY: (
        "Open conflict",
        "These claims were recorded as conflicting. Which is right, or are both?",
    ),
    CardKind.DEMOTION: (
        "Retired as replaced",
        "A later claim retired this one as its replacement. Was the replacement "
        "right, or do both claims hold?",
    ),
    CardKind.STALE_BASIS: (
        "Relied on a changed circumstance",
        "A belief this one relied on was replaced by an update, and it could not be "
        "restated safely. Is it still true as stated?",
    ),
}

# What each gesture does, in the curator's terms. ``{snooze_days}`` and
# ``{inconsistency_id}`` are filled per card by ``describe_gesture``. A
# (kind, gesture) entry overrides the generic one where the same verb means
# something different on that kind.
_GESTURE_HELP: dict[str, str] = {
    "affirm": "The belief is still correct. Record that, and hide this card for good.",
    "snooze": "Decide later. Hide this card for {snooze_days} days (change with --days N).",
    "dismiss": "Not a real problem. Hide this card for good.",
    "retract": "The belief is wrong. Retract it (explain with --reason TEXT).",
    "merge": "Same claim, stated twice. Link the two as co-evidential; both stay active.",
    "deposit": "Fetch the URL and deposit it into the corpus for extraction.",
    "assign-subject": "Attach the belief to a subject with --subject ID-OR-NAME.",
    "relink": "Link every belief on this card to the subjects its own record, triple or "
    "backticks name, scoped by project. Preview with `particles subjects relink-gated`.",
    "accept": "Assert the generalization as a derived belief linked to its premises.",
    "reject": "The generalization is wrong or unhelpful. Discard it (--reason TEXT).",
    "supersede": "The belief is outdated or imprecise. Replace it with a corrected "
    "belief (--content TEXT --reason TEXT --confidence F; subjects are kept unless "
    "you pass --subject).",
    "resolve": "Settle the conflict with --action {resolve_actions} [--note TEXT]. "
    "PREFER_A keeps A and demotes B, PREFER_B the reverse, BOTH_VALID keeps both, "
    "DISCARD retracts both. Snooze to decide later.",
    # Surfaced gestures: the card names the resolving command.
    "reindex": "Re-extract the failed snapshots. Not applied by this command: run "
    "`particles reindex`.",
}

_KIND_GESTURE_HELP: dict[tuple[CardKind, str], str] = {
    (CardKind.CONTESTED, "affirm"): "The belief stands despite the disagreement. Hide this "
    "card for good (a conflict it is in keeps its own card).",
    (CardKind.CONFIDENCE_DECAY, "affirm"): "The belief is sound as stated. Record that, "
    "and hide this card for good.",
    (CardKind.DUPLICATE_PAIR, "dismiss"): "They are different claims. Hide this card for good.",
    (CardKind.UNCITED_URL, "dismiss"): "Not worth depositing. Stop suggesting this URL.",
    (CardKind.DEMOTION, "affirm"): "The replacement was right. Hide this card for good.",
    (CardKind.DEMOTION, "dismiss"): "Both claims hold. Hide this card for good. The "
    "retired claim stays retired, because a demotion cannot be reversed: assert it "
    "again if it is still needed.",
}

#: The gestures whose ruling on a ``demotion`` card is kept as a labelled pair,
#: and the ruling each records.
DEMOTION_RULINGS: dict[str, Literal["replacement", "coexist"]] = {
    "affirm": "replacement",
    "dismiss": "coexist",
}


def describe_gesture(card: CurationCard, gesture: str, *, snooze_days: int) -> str:
    """One line saying what ``gesture`` would do to ``card``, in the curator's terms.

    Surfaced gestures (reindex) name the command that actually resolves the
    card; ``resolve`` names the actions this card offers.
    """
    template = _KIND_GESTURE_HELP.get((card.kind, gesture)) or _GESTURE_HELP.get(gesture)
    if template is None:
        return ""
    actions = card.resolve_actions if card.resolve_actions is not None else RESOLVE_ACTIONS
    text = template.format(
        snooze_days=snooze_days,
        resolve_actions="|".join(actions) or "BOTH_VALID|DISCARD",
    )
    if (
        card.kind is CardKind.DEMOTION
        and gesture in DEMOTION_RULINGS
        and get_config().benchmark.record_demotion_rulings
    ):
        text += " The ruling is kept as a benchmark fixture."
    return text


def resolve_actions_for(a: Particle | None, b: Particle | None) -> list[str]:
    """The review actions a conflict card offers, given claims A and B.

    An action is offered only when it can do what it says:

    - ``PREFER_A`` needs A ACTIVE. PREFER_A demotes B and never re-promotes A,
      so on a record whose A was demoted it would close the conflict with
      neither claim ACTIVE (holds the review verb's own semantics).
    - ``PREFER_B`` needs B ACTIVE or quarantined, the only B it can restore.
    - ``BOTH_VALID`` and ``DISCARD`` are always offered.
    """
    offered: list[str] = []
    for action in RESOLVE_ACTIONS:
        if action == "PREFER_A" and (a is None or a.status is not Status.ACTIVE):
            continue
        if action == "PREFER_B" and (
            b is None or (b.status is not Status.ACTIVE and not is_quarantined(b))
        ):
            continue
        offered.append(action)
    return offered


class ParticleBrief(BaseModel):
    """A compact summary of one particle a card references.

    Just enough context to judge the card's gesture (the claim ``content``, the
    ``subject_labels`` it attaches to, its query-time ``effective_confidence``,
    and its ``status``) so a client (the curation PWA) can
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
    # Why the belief is in that status (e.g. ``CONFLICT_PENDING`` for a claim
    # quarantined until its conflict is reviewed), verbatim from the store.
    # ``None`` for an ACTIVE belief with no recorded reason.
    status_reason: str | None = None
    # Where the belief came from: the URI of its first SOURCE corpus entry
    # (a web URL, or e.g. ``claude-code://session/…`` for a harness
    # transcript) and when it was asserted. A belief with no subject is
    # otherwise unplaceable from the card alone.
    source_uri: str | None = None
    asserted_at: datetime | None = None


class ConflictBrief(BaseModel):
    """Both sides of an open INCONSISTENCY record.

    Carried by the record's own ``INCONSISTENCY`` card, and by a ``CONTESTED``
    card whose belief also sits in a conflict. ``a`` and ``b`` are the
    record's claims A and B in the order ``review`` names them, so a client
    can offer "keep A" / "keep B" as ``PREFER_A`` / ``PREFER_B`` without
    guessing which side is which. Either may be ``None`` when the record
    names a particle that no longer exists. ``further_a`` / ``further_b`` are
    a census record's further members by side, empty for every
    other record. Attached live to the served slice, never stored.
    """

    inconsistency_id: str
    a: ParticleBrief | None = None
    b: ParticleBrief | None = None
    further_a: list[ParticleBrief] = Field(default_factory=list)
    further_b: list[ParticleBrief] = Field(default_factory=list)


class DuplicateVerdict(BaseModel):
    """The LLM judge's read on a duplicate-pair card.

    Carried on a ``DUPLICATE_PAIR`` card when the queue ran the duplicate finder
    in ``LLM_JUDGE`` mode (``semantic=True``): the per-pair same-claim
    ``verdict`` (``PARAPHRASE`` = same claim, safe to merge; ``DISTINCT`` = not a
    duplicate; ``UNSURE`` = ambiguous) plus its short ``rationale`` when the
    candidate exposes one. **Advisory**: it informs the operator's Merge /
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
    # Both sides of that INCONSISTENCY, briefed like ``particles``. Filled
    # with the briefs on the served slice; None when no inconsistency backs
    # the card.
    conflict: ConflictBrief | None = None
    # the review actions an INCONSISTENCY card offers, computed
    # from its members' statuses when the slice is served. Every client
    # renders exactly this list. None for every other kind.
    resolve_actions: list[str] | None = None
    # The leverage score (§2), filled by leverage.score_cards.
    leverage: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def title(self) -> str:
        """The kind's human title, e.g. "Contested belief" (from ``KIND_TITLES``)."""
        return KIND_TITLES[self.kind][0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def question(self) -> str:
        """The question this kind puts to the curator (from ``KIND_TITLES``)."""
        return KIND_TITLES[self.kind][1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> str:
        """Stable identity for snooze / affirm filtering.

        Finder-output-derived, so the same underlying problem yields the same
        key across sessions. Serialized into the queue response (a
        ``@computed_field``) so a client (PWA / Obsidian) can echo the exact
        ``card_key`` back to ``POST /curation/affirm`` / ``snooze``;
        the snooze/affirm filter matches on it.
        """
        if self.kind is CardKind.UNCITED_URL:
            return f"uncited_url:{self.corpus_url}"
        if self.kind is CardKind.FAILED_SNAPSHOTS:
            return "failed_snapshots"
        if self.kind is CardKind.GATED_SUBJECTS:
            # A constant: the card is one batch, whatever beliefs it covers.
            return "gated_subjects"
        if self.kind is CardKind.PROPOSED_ABSTRACTION:
            return f"proposed_abstraction:{self.candidate_event_id}"
        if self.kind is CardKind.INCONSISTENCY:
            # one record, one key, whatever its members.
            return f"inconsistency:{self.inconsistency_id}"
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
        if key == "gated_subjects":
            return cls(
                kind=CardKind.GATED_SUBJECTS,
                diagnostic="",
                suggested_gestures=gestures_for(CardKind.GATED_SUBJECTS),
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
        if prefix == CardKind.INCONSISTENCY.value:
            if not rest:
                raise ValueError(f"Unrecognized card key {key!r}.")
            return cls(
                kind=CardKind.INCONSISTENCY,
                inconsistency_id=rest,
                diagnostic="",
                suggested_gestures=gestures_for(CardKind.INCONSISTENCY),
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
