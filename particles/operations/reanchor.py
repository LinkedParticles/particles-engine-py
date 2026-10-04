# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The re-anchor pass: restate claims that relied on a superseded state.

When rung 2.5 retires a state claim R ("The user lives in Delhi, in Lajpat
Nagar."), a claim cut from the same passage may have relied on R being current
("Sandeep's Curry House is a ten-minute walk from the user's flat."). Its
referent moved with the update, so it keeps reading as a fact about now and is
false. This pass finds such claims and replaces each with a dated restatement
anchored to R ("… from the flat in Lajpat Nagar, Delhi, where the user lived as
of 2026-09-04."), which stays true whichever way the update turns out.

Gather, decide, apply (D2): this module gathers (triggers, candidates,
passages, LLM replies) and applies (writes, events); every decision is a pure
function in :mod:`particles.core.reanchor`.

Per trigger R, oldest retirement first after the cursor:

1. **Candidates, zero-LLM** (§2): ACTIVE claims sharing a SOURCE snapshot and a
   subject with R, extractor-asserted and reconcilable, not observed after R,
   inside R's observer scope, and not a restatement this pass wrote.
2. **One probe** (the ``extraction`` purpose) returns HOLDS or DEPENDS with a
   restatement, per candidate.
3. **A second reading** (the ``verification`` purpose) re-asks the dependency
   and checks the restatement's faithfulness; both must hold.
4. **No side effects** (§3): a restatement an ACTIVE claim already states
   retires the original in its favour; one that contradicts a standing claim,
   or whose check could not run, writes nothing and is shown to the owner.
5. **The writes**: the restatement ACTIVE, the original ``SUPERSEDED`` /
   ``SUPERSEDED_BY_REANCHOR``; or a ``DEPENDENT_CLAIM_EXAMINED`` event.

The caller owns the transaction: nothing here commits.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.duplicate_key import content_hash
from particles.core.generics import is_generic_claim
from particles.core.observer_scope import PairPrecondition
from particles.core.reanchor import (
    Cursor,
    DependencyVerdict,
    Outcome,
    ProbeVerdict,
    Reading,
    anchor_date,
    build_restatement,
    decide_after_reading,
    decide_write,
    is_before_or_at,
    parse_probe_reply,
    parse_reading_reply,
)
from particles.core.schema import (
    JudgeVerdictKind,
    Particle,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.status import Status, StatusReason
from particles.corpus.deposit import load_blob
from particles.corpus.store import get_entry, get_snapshot
from particles.db import write_transaction
from particles.embeddings import cosine_similarity, get_embedding_model
from particles.ingest.observer_gate import ObserverGate
from particles.ingest.update_supersession import (
    is_extractor_asserted,
    is_reconcilable,
    latest_source_date,
)
from particles.llm import data_fence_instruction, fence, make_nonce
from particles.operations._llm import _llm_call, record_unusable_reply
from particles.operations.source_passage import derive_passage, extractor_view_text
from particles.store.event_store import (
    EventRefKind,
    OperatorEventType,
    list_events,
    record_event,
)
from particles.store.particle_store import (
    copy_modality_stamp,
    count_update_retirements_after,
    get_active_particles_by_content_hashes,
    get_active_particles_for_entry,
    get_active_particles_with_embeddings,
    get_particles_by_ids,
    get_update_retirements_after,
    insert_particle,
    update_particle_status,
)

log = logging.getLogger(__name__)

Embedding = np.ndarray[Any, np.dtype[np.float32]]

#: Output budgets. The probe and the reading run on purposes that may be routed
#: to an adaptive-thinking model, whose thinking counts against ``max_tokens``
#: (see :func:`particles.operations._llm._llm_call`), so both are generous.
_PROBE_MAX_TOKENS = 4096
_READING_MAX_TOKENS = 2048
#: The passage the probe sees is capped; the located paragraphs of R and each
#: candidate are what it needs, not the whole transcript.
_PASSAGE_MAX_CHARS = 8000
#: How many run records are searched for the last cursor.
_CURSOR_LOOKBACK = 100
#: Near-duplicates the paraphrase judge is asked about, at most.
_DEDUP_JUDGED = 3

_PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["HOLDS", "DEPENDS"]},
                    "restatement": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_READING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "depended": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["depended", "faithful", "reason"],
    "additionalProperties": False,
}


class ReanchorReport(BaseModel):
    """What one run of the pass did (the run record's ``reanchor`` block)."""

    enabled: bool = True
    dry_run: bool = False
    scope: Literal["cursor", "store"] = "cursor"
    #: Set when the pass did no work at all, with the reason.
    skipped_reason: str | None = None
    triggers: int = 0
    #: Update retirements past the cursor left for the next run.
    waiting: int = 0
    candidates: int = 0
    observer_declined: int = 0
    probes: int = 0
    readings: int = 0
    duplicate_checks: int = 0
    conflict_probes: int = 0
    holds: int = 0
    dependency_rejected: int = 0
    #: ``{"original", "restatement", "retired"}`` per claim restated.
    restated: list[dict[str, str]] = Field(default_factory=list)
    #: ``{"original", "existing", "retired", "event"}`` per claim matched.
    matched_existing: list[dict[str, str]] = Field(default_factory=list)
    #: ``{"original", "retired", "event", "why"}`` per claim kept for the owner.
    unrestated: list[dict[str, str]] = Field(default_factory=list)
    #: The cursor after this run; ``None`` until a first trigger is examined.
    cursor: dict[str, str] | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        return self.probes + self.readings + self.duplicate_checks + self.conflict_probes

    def payload(self) -> dict[str, Any]:
        """The compact block written to the ``CONSOLIDATION_RUN`` payload."""
        return {
            "enabled": self.enabled,
            "skipped_reason": self.skipped_reason,
            "triggers": self.triggers,
            "waiting": self.waiting,
            "candidates": self.candidates,
            "observer_declined": self.observer_declined,
            "probes": self.probes,
            "readings": self.readings,
            "holds": self.holds,
            "dependency_rejected": self.dependency_rejected,
            "restated": self.restated,
            "matched_existing": self.matched_existing,
            "unrestated": self.unrestated,
            "cursor": self.cursor,
        }


# ---------------------------------------------------------------------------
# Prompts (every claim and passage F3-fenced: they are extracted from untrusted
# sources and must not be able to coerce a verdict)
# ---------------------------------------------------------------------------


_PROBE_SYSTEM = (
    "You maintain an agent's memory of claims extracted from sources. One claim "
    "about the speaker's circumstances, the RETIRED claim, is no longer current: "
    "a later source reported that the circumstance changed. The user message "
    "shows the retired claim, the passage the numbered claims were extracted "
    "from, and the passage's date. For each numbered claim decide:\n"
    "- HOLDS: the claim is true as stated whether or not the retired claim is "
    "still current. A claim that only shares a subject with the retired claim "
    "holds. A lasting fact, preference or evaluation holds. A claim that names "
    "the place or thing the retired claim described, without relying on it being "
    "the speaker's current circumstance, holds (for example 'X is in <place>').\n"
    "- DEPENDS: read as a statement about now, the claim refers to the "
    "circumstance the retired claim described through a relative or possessive "
    "reference (such as 'my flat', 'the user's flat', 'near my office', 'my "
    "manager', 'nearby', 'here', 'local'), so it stopped being true when that "
    "circumstance changed. A claim that names the place or person but still "
    "presents it as the speaker's current circumstance ('the user's flat in "
    "Lajpat Nagar', 'their manager Priya') depends too.\n"
    "For each DEPENDS claim write a restatement: one sentence that (1) is "
    "entailed by the passage, its date and the retired claim, with nothing from "
    "anywhere else; (2) names the circumstance the retired claim described in "
    "place of the relative reference; (3) is in the past tense and dated with "
    "'as of <date>' ('… where the user lived as of <date>', '… took forty "
    "minutes as of <date>'), using the date the passage itself states it was "
    "written when it states one, else the date given. The date must scope what "
    "the claim itself says, not only the circumstance: write 'Acme Analytics, "
    "where the user worked, used Slack as of 2026-04-21', not 'Acme Analytics, "
    "where the user worked as of 2026-04-21, used Slack'; (4) keeps everything else "
    "the claim said, about the same thing it said it about; (5) is not a "
    "restatement of the retired claim itself.\n"
    "Example. Retired: 'The user lives in Delhi, in Lajpat Nagar.' Claim: "
    "'Sandeep's Curry House is a ten-minute walk from the user's flat.' "
    "Restatement: 'Sandeep's Curry House was a ten-minute walk from the flat in "
    "Lajpat Nagar, Delhi, where the user lived, as of 2026-09-04.'\n"
    'Return a JSON object: {"claims": [{"claim": "c1", "verdict": '
    '"HOLDS"|"DEPENDS", "restatement": "...", "reason": "one sentence"}]}, one '
    "entry per numbered claim; omit restatement for HOLDS.\n\n"
)

_READING_SYSTEM = (
    "You check one proposed rewrite in an agent's memory. A claim about the "
    "speaker's circumstances, the RETIRED claim, is no longer current: a later "
    "source reported that the circumstance changed. The ORIGINAL claim was "
    "extracted from the passage shown. An earlier reading judged that the "
    "original depended on the retired claim being current and wrote the "
    "RESTATEMENT. Answer two questions independently.\n"
    "1. depended: read as a statement about now, did the ORIGINAL claim depend "
    "on the retired claim's circumstance still being current? It did when it "
    "refers to that circumstance through a relative or possessive reference: "
    "'my flat', 'near my office', 'my manager', 'here', or 'my car' or 'my "
    "phone' when the retired claim said which car or phone that was. Such a "
    "reference now points at the new circumstance, so the claim stopped being "
    "true even though it names no model, place or person. A claim that is true "
    "regardless, such as a lasting fact, preference or evaluation about the "
    "speaker, did not depend.\n"
    "2. faithful: is the RESTATEMENT entailed by the passage, its date and the "
    "retired claim, with nothing from anywhere else; does it keep every other "
    "detail of the original; and does it no longer depend on the circumstance "
    "being current? The restatement is meant to name the circumstance the "
    "retired claim described in place of the relative reference (for example "
    "'the user's Skoda Octavia' for 'the user's car'): that detail comes from "
    "the retired claim and is not an addition. It is also meant to be dated and "
    "in the past tense ('… as of <date>'): that change of tense is not a change "
    "of detail. The date must scope what the claim itself says, not only the "
    "circumstance: in 'Acme Analytics, where the user worked as of 2026-04-21, "
    "used Slack' the use of Slack is left undated, which is not faithful. Its "
    "date should be the one the passage states it was written, "
    "when it states one, else the date of the source record. A restatement that "
    "adds anything else, or drops or changes any other detail, is not "
    "faithful.\n"
    'Return a JSON object: {"depended": true|false, "faithful": true|false, '
    '"reason": "one sentence"}.\n\n'
)


async def _probe(
    retired: Particle, passage: str, date: str | None, candidates: list[Particle]
) -> list[ProbeVerdict] | None:
    """One probe call for one trigger; ``None`` when no usable reply came back."""
    nonce = make_nonce()
    labels = {f"c{i + 1}": c.id for i, c in enumerate(candidates)}
    user = (
        f"Retired claim:\n{fence(retired.content, nonce, label='retired')}\n\n"
        f"Date of the source record: {date or 'unknown'}\n\n"
        f"Passage:\n{fence(passage, nonce, label='passage')}\n\n"
        + "\n\n".join(
            f"{label}:\n{fence(c.content, nonce, label=label)}"
            for label, c in zip(labels, candidates, strict=True)
        )
    )
    reply = await _llm_call(
        user,
        max_tokens=_PROBE_MAX_TOKENS,
        system=_PROBE_SYSTEM + data_fence_instruction(nonce),
        response_schema=_PROBE_SCHEMA,
        purpose="extraction",
        # A classification with a written answer: sampling spread only makes
        # the verdict and the wording vary between runs.
        temperature=0.0,
    )
    if reply is None:
        return None
    verdicts = parse_probe_reply(reply, labels)
    if verdicts is None:
        record_unusable_reply("re-anchor probe", reply)
    return verdicts


async def _read(
    retired: Particle, original: Particle, restatement: str, passage: str, date: str | None
) -> Reading | None:
    """The second reading of one restatement; ``None`` when unusable."""
    nonce = make_nonce()
    user = (
        f"Retired claim:\n{fence(retired.content, nonce, label='retired')}\n\n"
        f"Date of the source record: {date or 'unknown'}\n\n"
        f"Passage:\n{fence(passage, nonce, label='passage')}\n\n"
        f"Original claim:\n{fence(original.content, nonce, label='original')}\n\n"
        f"Restatement:\n{fence(restatement, nonce, label='restatement')}"
    )
    reply = await _llm_call(
        user,
        max_tokens=_READING_MAX_TOKENS,
        system=_READING_SYSTEM + data_fence_instruction(nonce),
        response_schema=_READING_SCHEMA,
        purpose="verification",
        temperature=0.0,
    )
    if reply is None:
        return None
    reading = parse_reading_reply(reply)
    if reading is None:
        record_unusable_reply("re-anchor second reading", reply)
    return reading


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


async def prior_cursor(session: AsyncSession, actor: str) -> Cursor | None:
    """The cursor the last run of ``actor`` recorded, or ``None`` before the first."""
    events = await list_events(
        session, event_type=OperatorEventType.CONSOLIDATION_RUN, limit=_CURSOR_LOOKBACK
    )
    for event in events:
        if event.actor != actor:
            continue
        census = (event.payload or {}).get("census")
        block = census.get("reanchor") if isinstance(census, dict) else None
        if isinstance(block, dict) and block.get("cursor") is not None:
            return Cursor.from_payload(block.get("cursor"))
    return None


def _source_snapshots(particle: Particle) -> set[str]:
    return {
        r.snapshot_id
        for r in particle.provenance
        if r.type is ProvenanceRefType.SOURCE and r.snapshot_id
    }


def _source_entries(particle: Particle) -> list[str]:
    return list(
        dict.fromkeys(
            r.corpus_entry_id for r in particle.provenance if r.type is ProvenanceRefType.SOURCE
        )
    )


async def _passage_for(
    session: AsyncSession, retired: Particle, candidates: list[Particle]
) -> tuple[str, str | None] | None:
    """The passage R and its candidates were cut from, and its date.

    The located paragraphs of R and of each candidate in R's snapshot, in the
    order they appear, capped. ``None`` when the snapshot's text is unavailable:
    a restatement must be written from the source, never from memory.
    """
    ref = next(
        (r for r in retired.provenance if r.type is ProvenanceRefType.SOURCE and r.snapshot_id),
        None,
    )
    if ref is None or ref.snapshot_id is None:
        return None
    snapshot = await get_snapshot(session, ref.snapshot_id)
    entry = await get_entry(session, ref.corpus_entry_id)
    if snapshot is None or entry is None:
        return None
    try:
        text = extractor_view_text(load_blob(snapshot.content_hash), entry.source_type)
    except (FileNotFoundError, ValueError):
        return None
    if not text.strip():
        return None
    found: dict[str, int] = {}
    for particle in [retired, *candidates]:
        chunk = next(
            (
                r.chunk_hash
                for r in particle.provenance
                if r.type is ProvenanceRefType.SOURCE and r.snapshot_id == ref.snapshot_id
            ),
            None,
        )
        _, passage, _ = derive_passage(text, content=particle.content, chunk_hash=chunk)
        found.setdefault(passage, text.find(passage))
    # The source's opening line (a title, a dated header) rides along: it is
    # where a note or transcript usually says when it was written, and the
    # restatement is dated from the passage before the record's dates.
    opening = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if opening and not any(opening in passage for passage in found):
        found.setdefault(opening, -1)
    ordered = sorted(found, key=lambda p: found[p])
    joined = "\n…\n".join(ordered)
    date = anchor_date(snapshot.content_published_at or snapshot.captured_at)
    return joined[:_PASSAGE_MAX_CHARS], date


async def _is_restatement(session: AsyncSession, particle: Particle) -> bool:
    """Candidacy condition 1: a claim this pass wrote is never re-examined."""
    if particle.supersedes is None:
        return False
    prior = (await get_particles_by_ids(session, [particle.supersedes])).get(particle.supersedes)
    return prior is not None and prior.status_reason is StatusReason.SUPERSEDED_BY_REANCHOR


async def _already_examined(session: AsyncSession, particle_id: str, retired_id: str) -> bool:
    """True when an event already records this claim against this retirement."""
    for event in await list_events(
        session,
        ref_kind=EventRefKind.PARTICLE,
        ref_id=particle_id,
        event_type=OperatorEventType.DEPENDENT_CLAIM_EXAMINED,
        limit=50,
    ):
        if (event.payload or {}).get("retired_id") == retired_id:
            return True
    return False


async def _candidates(
    session: AsyncSession,
    retired: Particle,
    gate: ObserverGate,
    report: ReanchorReport,
    dates: dict[str, Any],
) -> list[Particle]:
    """Candidacy conditions 1 to 6, before ranking."""
    snapshots = _source_snapshots(retired)
    subjects = set(retired.subject_ids)
    if not snapshots or not subjects:
        return []
    pool: dict[str, Particle] = {}
    for entry_id in _source_entries(retired):
        for particle in await get_active_particles_for_entry(session, entry_id):
            pool.setdefault(particle.id, particle)
    retired_date = await latest_source_date(session, retired, dates)
    out: list[Particle] = []
    retired_scope = await gate.scope_of(session, retired) if gate.engaged else None
    for particle in pool.values():
        if particle.id == retired.id or particle.status is not Status.ACTIVE:
            continue
        if not _source_snapshots(particle) & snapshots:
            continue
        if not set(particle.subject_ids) & subjects:
            continue
        if not (is_extractor_asserted(particle) and is_reconcilable(particle)):
            continue
        if particle.uncertainty_nature is UncertaintyNature.ALEATORY:
            continue
        if not is_before_or_at(await latest_source_date(session, particle, dates), retired_date):
            continue
        if await _is_restatement(session, particle):
            continue
        if await _already_examined(session, particle.id, retired.id):
            continue
        if retired_scope is not None and (
            await gate.verdict(session, retired_scope, particle) is not PairPrecondition.RECONCILE
        ):
            report.observer_declined += 1
            continue
        out.append(particle)
    return out


async def _encode(texts: list[str]) -> list[Embedding]:
    model = get_embedding_model()
    assert model is not None  # checked by the caller before any work
    encoded = await asyncio.to_thread(
        model.encode, texts, convert_to_numpy=True, normalize_embeddings=True
    )
    return [encoded[i] for i in range(len(texts))]


class _Pool:
    """ACTIVE reconcilable claims by every subject they name, with embeddings.

    The contradiction check's candidates. Unlike the update search's
    about-subject index, every subject counts: a restatement
    names the place and the speaker, and a claim about either may contradict it.
    """

    def __init__(self, pairs: Iterable[tuple[Particle, Embedding | None]]) -> None:
        self.by_subject: dict[str, list[tuple[Particle, Embedding]]] = {}
        #: Claims retired earlier in this run: never offered again.
        self.retired: set[str] = set()
        for particle, emb in pairs:
            if emb is None or not is_reconcilable(particle):
                continue
            for subject_id in particle.subject_ids:
                self.by_subject.setdefault(subject_id, []).append((particle, emb))

    def nearest(
        self,
        subject_ids: list[str],
        embedding: Embedding,
        *,
        floor: float,
        limit: int,
        skip: set[str],
    ) -> list[Particle]:
        scored: dict[str, tuple[float, Particle]] = {}
        for subject_id in subject_ids:
            for particle, emb in self.by_subject.get(subject_id, []):
                if particle.id in skip or particle.id in self.retired or particle.id in scored:
                    continue
                score = cosine_similarity(embedding, emb)
                if score >= floor:
                    scored[particle.id] = (score, particle)
        ranked = sorted(scored.values(), key=lambda t: -t[0])[:limit]
        return [p for _, p in ranked]


# ---------------------------------------------------------------------------
# Decide, per restatement, and apply
# ---------------------------------------------------------------------------


async def _duplicate_of(
    session: AsyncSession,
    restatement: str,
    embedding: Embedding,
    original: Particle,
    retired: Particle,
    report: ReanchorReport,
) -> str | None:
    """An ACTIVE claim that already states the restatement, or ``None``.

    Exact identity first (the key, store-wide), then the paraphrase
    judge over the nearest ACTIVE claims of the original's snapshot, the place
    a re-emitted twin or an earlier restatement of the same line sits.
    """
    # Deferred import: tests/test_reanchor.py patches the judge at its source
    # module; a module-top binding would freeze it (tests/AGENTS.md § Mocking
    # strategy).
    from particles.operations.abstraction import _paraphrase_verdict

    exclude = {original.id, retired.id}
    for match in await get_active_particles_by_content_hashes(session, [content_hash(restatement)]):
        if match.id not in exclude:
            return match.id
    snapshots = _source_snapshots(original)
    near: list[tuple[float, Particle]] = []
    floor = get_config().links_suggest.candidate_threshold
    siblings: dict[str, Particle] = {}
    for entry_id in _source_entries(original):
        for particle in await get_active_particles_for_entry(session, entry_id):
            if particle.id not in exclude and _source_snapshots(particle) & snapshots:
                siblings.setdefault(particle.id, particle)
    if siblings:
        sibling_list = list(siblings.values())
        embeddings = await _encode([p.content for p in sibling_list])
        for particle, emb in zip(sibling_list, embeddings, strict=True):
            score = cosine_similarity(embedding, emb)
            if score >= floor:
                near.append((score, particle))
    near.sort(key=lambda t: -t[0])
    for _score, particle in near[:_DEDUP_JUDGED]:
        report.duplicate_checks += 1
        verdict = await _paraphrase_verdict(restatement, particle.content, purpose="verification")
        if verdict is JudgeVerdictKind.PARAPHRASE:
            return particle.id
    return None


async def _conflicting(
    session: AsyncSession,
    restatement: str,
    embedding: Embedding,
    original: Particle,
    retired: Particle,
    index: _Pool,
    report: ReanchorReport,
) -> tuple[bool, str | None]:
    """Whether the restatement contradicts a standing claim: ``(conflict, with_id)``.

    The nearest ACTIVE claims sharing any subject with it, at the update
    search's subject floor and candidate limit, excluding the
    original and R, and excluding a claim of the other kind when one side is
    generic and the other an instance claim. A probe that fails is
    a conflict: a restatement is written only when the check ran clean.
    """
    # Deferred import: the breaker-seam probe, patched at its source module by
    # tests/test_reanchor.py (tests/AGENTS.md § Mocking strategy).
    from particles.operations.reconcile import _has_contradiction_signal

    cfg = get_config()
    update_cfg = cfg.reconciliation.update_supersession
    exclude = {original.id, retired.id}
    pool = index.nearest(
        list(dict.fromkeys([*original.subject_ids, *retired.subject_ids])),
        embedding,
        floor=update_cfg.subject_floor,
        limit=update_cfg.max_candidates,
        skip=exclude,
    )
    generic = is_generic_claim(restatement)
    for particle in pool:
        # A generic and an instance claim are not an adjudicable pair.
        if is_generic_claim(particle.content) != generic:
            continue
        report.conflict_probes += 1
        signal = await _has_contradiction_signal(particle.content, restatement)
        if signal is None:
            return True, None
        if signal:
            return True, particle.id
    return False, None


async def _record(
    session: AsyncSession,
    *,
    actor: str,
    outcome: Outcome,
    original: Particle,
    retired: Particle,
    other_id: str | None,
    reason: str,
    why: str,
    restatement: str | None,
) -> str:
    refs = [(EventRefKind.PARTICLE, original.id), (EventRefKind.PARTICLE, retired.id)]
    if other_id is not None:
        refs.append((EventRefKind.PARTICLE, other_id))
    event = await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.DEPENDENT_CLAIM_EXAMINED,
        reason=why,
        refs=refs,
        payload={
            "outcome": outcome.value,
            "particle_id": original.id,
            "retired_id": retired.id,
            "retired_content": retired.content,
            "other_id": other_id,
            "probe_reason": reason,
            "why": why,
            "restatement": restatement,
        },
    )
    return event.event_id


async def _settle(
    session: AsyncSession,
    *,
    verdict: ProbeVerdict,
    original: Particle,
    retired: Particle,
    passage: str,
    date: str | None,
    index: _Pool,
    report: ReanchorReport,
    actor: str,
    now: datetime,
) -> None:
    """Read, check and write one DEPENDS verdict.

    The reading and the checks run first; the verdict's writes then commit
    under the writer lock, so no write is held open across the next
    verdict's LLM calls.
    """
    restatement = verdict.restatement
    if restatement is None:
        async with write_transaction(session):
            event_id = await _record(
                session,
                actor=actor,
                outcome=Outcome.UNRESTATED,
                original=original,
                retired=retired,
                other_id=None,
                reason=verdict.reason,
                why="the probe judged it dependent but wrote no restatement",
                restatement=None,
            )
        report.unrestated.append(
            {"original": original.id, "retired": retired.id, "event": event_id, "why": "none"}
        )
        return

    report.readings += 1
    reading = await _read(retired, original, restatement, passage, date)
    settled = decide_after_reading(reading)
    if settled is Outcome.DEPENDENCY_REJECTED:
        report.dependency_rejected += 1
        return
    if settled is Outcome.UNRESTATED:
        why = (
            "the second reading could not be completed"
            if reading is None
            else f"the second reading found the restatement unfaithful: {reading.reason}"
        )
        async with write_transaction(session):
            event_id = await _record(
                session,
                actor=actor,
                outcome=Outcome.UNRESTATED,
                original=original,
                retired=retired,
                other_id=None,
                reason=verdict.reason,
                why=why,
                restatement=restatement,
            )
        report.unrestated.append(
            {"original": original.id, "retired": retired.id, "event": event_id, "why": why}
        )
        return

    [embedding] = await _encode([restatement])
    duplicate = await _duplicate_of(session, restatement, embedding, original, retired, report)
    conflict, conflict_id = (
        (False, None)
        if duplicate is not None
        else await _conflicting(session, restatement, embedding, original, retired, index, report)
    )
    outcome = decide_write(duplicate_of=duplicate, conflict=conflict)

    if outcome is Outcome.MATCHED_EXISTING:
        assert duplicate is not None
        async with write_transaction(session):
            await update_particle_status(
                session, original.id, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REANCHOR
            )
            event_id = await _record(
                session,
                actor=actor,
                outcome=outcome,
                original=original,
                retired=retired,
                other_id=duplicate,
                reason=verdict.reason,
                why="an ACTIVE claim already states the restatement",
                restatement=restatement,
            )
        index.retired.add(original.id)
        report.matched_existing.append(
            {
                "original": original.id,
                "existing": duplicate,
                "retired": retired.id,
                "event": event_id,
            }
        )
        return

    if outcome is Outcome.UNRESTATED:
        why = (
            f"the restatement contradicts {conflict_id}"
            if conflict_id is not None
            else "the contradiction check could not be completed"
        )
        async with write_transaction(session):
            event_id = await _record(
                session,
                actor=actor,
                outcome=outcome,
                original=original,
                retired=retired,
                other_id=conflict_id,
                reason=verdict.reason,
                why=why,
                restatement=restatement,
            )
        report.unrestated.append(
            {"original": original.id, "retired": retired.id, "event": event_id, "why": why}
        )
        return

    selection = get_config().llm.for_purpose("extraction")
    written = build_restatement(
        original,
        retired,
        restatement,
        provider_model=f"{selection.provider}:{selection.model}",
        now=now,
    )
    async with write_transaction(session):
        await insert_particle(session, written, embedding=[float(x) for x in embedding])
        # The restatement carries the original's modality value, so it carries
        # the stamp that says who set it: an operator verdict stays pinned.
        await copy_modality_stamp(session, original.id, written.id)
        await update_particle_status(
            session, original.id, Status.SUPERSEDED, StatusReason.SUPERSEDED_BY_REANCHOR
        )
    index.retired.add(original.id)
    report.restated.append(
        {"original": original.id, "restatement": written.id, "retired": retired.id}
    )


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


async def run_reanchor(
    session: AsyncSession,
    *,
    cursor: Cursor | None,
    scope: Literal["cursor", "store"] = "cursor",
    dry_run: bool = False,
    actor: str = "memory-consolidate",
    progress: Callable[[str], None] | None = None,
) -> ReanchorReport:
    """Run the re-anchor pass once.

    Commits each verdict's writes as it settles, under the writer lock,
    rather than leaving them to the caller: held open, they kept
    SQLite's write lock through every later probe. A run stopped
    part-way leaves its settled verdicts in place; the next run skips them,
    since a restated or matched claim is no longer ACTIVE and an examined one
    carries its ``DEPENDENT_CLAIM_EXAMINED`` event.

    Args:
        cursor: where the previous run stopped; ignored with ``scope="store"``,
            which examines every update retirement from the first.
        dry_run: report triggers, candidates and the calls a run would make,
            and make none.
        actor: the event actor for anything this run records.
        progress: optional human-readable progress callback.

    Returns:
        The run's report. ``report.cursor`` is where the next run resumes.
    """
    cfg = get_config().consolidation.reanchor
    report = ReanchorReport(enabled=cfg.enabled, dry_run=dry_run, scope=scope)
    start = None if scope == "store" else cursor
    report.cursor = start.payload() if start is not None else None
    if not cfg.enabled:
        report.skipped_reason = "consolidation.reanchor.enabled is false"
        return report
    if get_embedding_model() is None:
        report.skipped_reason = "no embedding model (candidate ranking and checks need one)"
        return report

    after = (start.retired_at, start.particle_id) if start is not None else None
    triggers = await get_update_retirements_after(session, after, limit=cfg.max_retirements_per_run)
    gate = await ObserverGate.open(session)
    dates: dict[str, Any] = {}
    index: _Pool | None = None
    now = datetime.now(UTC)

    for retired, retired_at in triggers:
        if not dry_run and report.readings >= cfg.max_restatements_per_run:
            break
        candidates = await _candidates(session, retired, gate, report, dates)
        report.triggers += 1
        position = Cursor(retired_at=retired_at, particle_id=retired.id)
        if not candidates:
            report.cursor = position.payload()
            continue
        [retired_emb] = await _encode([retired.content])
        embeddings = await _encode([c.content for c in candidates])
        ranked = sorted(
            zip(candidates, embeddings, strict=True),
            key=lambda pair: -cosine_similarity(retired_emb, pair[1]),
        )
        chosen = [c for c, _ in ranked[: cfg.max_candidates_per_retirement]]
        report.candidates += len(chosen)
        if dry_run:
            report.probes += 1
            report.cursor = position.payload()
            continue

        located = await _passage_for(session, retired, chosen)
        if located is None:
            report.warnings.append(
                f"no source passage for {retired.id[:8]}; its dependents were not examined"
            )
            report.cursor = position.payload()
            continue
        passage, date = located
        report.probes += 1
        verdicts = await _probe(retired, passage, date, chosen)
        if verdicts is None:
            # Keep the cursor before this trigger, so the next run asks again.
            report.warnings.append(f"the probe for {retired.id[:8]} failed; retried next run")
            break
        by_id = {c.id: c for c in chosen}
        if index is None:
            index = _Pool(await get_active_particles_with_embeddings(session))
        for verdict in verdicts:
            if verdict.verdict is DependencyVerdict.HOLDS:
                report.holds += 1
                continue
            await _settle(
                session,
                verdict=verdict,
                original=by_id[verdict.candidate_id],
                retired=retired,
                passage=passage,
                date=date,
                index=index,
                report=report,
                actor=actor,
                now=now,
            )
        report.cursor = position.payload()
        if progress is not None:
            progress(
                f"{retired.id[:8]}: {len(chosen)} candidate(s), "
                f"{sum(1 for v in verdicts if v.verdict is DependencyVerdict.DEPENDS)} dependent"
            )

    final = Cursor.from_payload(report.cursor) if report.cursor is not None else None
    report.waiting = await count_update_retirements_after(
        session, (final.retired_at, final.particle_id) if final is not None else None
    )
    if dry_run:
        report.cursor = start.payload() if start is not None else None
    log.info(
        "re-anchor: %d trigger(s), %d candidate(s), %d restated, %d matched, %d kept for "
        "the owner, %d waiting%s",
        report.triggers,
        report.candidates,
        len(report.restated),
        len(report.matched_existing),
        len(report.unrestated),
        report.waiting,
        " (dry run)" if dry_run else "",
    )
    return report


# ---------------------------------------------------------------------------
# The curation card's feed
# ---------------------------------------------------------------------------


async def pending_unrestated_events(session: AsyncSession) -> list[Any]:
    """Open ``unrestated`` events: the claim is still ACTIVE. Newest per claim, oldest first.

    The STALE_BASIS card fronts each. A claim superseded or
    retracted since has left ACTIVE, which closes its card; ``affirm`` and
    ``snooze`` act on the card key as for any card.
    """
    events = await list_events(
        session, event_type=OperatorEventType.DEPENDENT_CLAIM_EXAMINED, limit=1000
    )
    latest: dict[str, Any] = {}
    for event in events:  # newest first
        payload = event.payload or {}
        if payload.get("outcome") != Outcome.UNRESTATED.value:
            continue
        pid = payload.get("particle_id")
        if isinstance(pid, str) and pid not in latest:
            latest[pid] = event
    if not latest:
        return []
    live = await get_particles_by_ids(session, list(latest))
    open_events = [latest[pid] for pid, p in live.items() if p.status is Status.ACTIVE]
    open_events.sort(key=lambda e: e.occurred_at)
    return open_events
