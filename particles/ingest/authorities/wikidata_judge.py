# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The subject-link judge: which Wikidata candidate a claim means, or none.

An unqualified common-word name ("Harbor", an invented payroll company) has a
common noun as its top Wikidata hit, and no score of description against claim
separates the right link from the wrong one (the 2026-09-30 measurement: correct
links 0.16 to 0.58, wrong ones 0.20 to 0.33). This module asks a model instead,
for the names that need it, when ``subjects.wikidata_candidate_selection`` is
``llm_judge``.

Three parts, in the order a resolution uses them:

* :func:`is_ambiguous` is the pure test that decides whether a name reaches the
  model at all. Most names never do.
* :func:`judge_candidates` makes the one call per ambiguous name, never one per
  candidate. The model sees the claim and each candidate's id, label and
  description, all of which the search response already carries, and answers a
  QID or ``none``.
* Every usable answer is kept in the probe-verdict ledger
  (:mod:`particles.store.probe_verdict_store`) under the name and claim, the
  candidate set, the prompt version and the model, and read back before any
  call. A fixed input therefore resolves the same way (whitepaper § 3.3), and a
  re-extraction pays nothing for a name it has already judged.

The call runs on the ``subject_resolution`` purpose through the completion port,
so it is metered by whatever run scope is open: an extraction's
``EXTRACT_RUN`` event and usage line include it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from particles.config import get_config
from particles.core.probe_verdict import ProbeKind, prompt_hash, subject_link_key
from particles.llm.breaker import _llm_call, record_unusable_reply
from particles.llm.fencing import data_fence_instruction, fence, make_nonce
from particles.store.probe_verdict_store import lookup_answer, record_answer

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

__all__ = [
    "NONE_OF_THESE",
    "Judgement",
    "judge_candidates",
    "judge_prompt_hash",
    "judge_request",
    "judgeable_hits",
    "is_ambiguous",
    "offered_candidates",
    "parse_judge_reply",
]

#: The judge's answer when no candidate is the entity the claim names.
NONE_OF_THESE = "none"

#: A candidate as the judge is shown it: ``(id, label, description)``.
Candidate = tuple[str, str, str]

_INSTRUCTION = (
    "You decide which Wikidata entity a name in a claim refers to. You are given "
    "the name, the claim it appears in, and the candidates a Wikidata search "
    "returned for the name, each as one line: id | label | description.\n\n"
    "Answer with the id of the candidate the name refers to in this claim, or "
    f'"{NONE_OF_THESE}" when none of them is that entity. A name that is also a '
    "common word often belongs to a company, product, project or person that has "
    "no Wikidata item: when the claim treats the name as such an entity and no "
    f'candidate describes it, answer "{NONE_OF_THESE}", even if a candidate is '
    "the ordinary meaning of the word. Pick a candidate only when it is the thing "
    "the claim is about.\n\n"
    'Reply with JSON only, {"qid": "<id>"} or {"qid": "' + NONE_OF_THESE + '"}.'
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"qid": {"type": "string"}},
    "required": ["qid"],
}


def is_ambiguous(candidate_count: int, top_score: float | None, floor: float) -> bool:
    """Whether a name's Wikidata candidates need the judge (pure).

    ``candidate_count`` is the number of usable search hits, ``top_score`` the
    top hit's description scored against the claim (``None`` when it could not
    be scored), and ``floor`` is ``subjects.wikidata_link_suppress_threshold``,
    the line above which a link is shown.

    * **No candidate** is not ambiguous: there is nothing to choose, and the
      name becomes a bare local Subject. Most invented names end here.
    * **Several candidates** are ambiguous. Wikidata's search ranks by
      popularity, not by the claim, and the description score cannot pick
      among them (it measured worse than the top hit).
    * **One candidate** is ambiguous only when its description scored below
      ``floor``, the case of a lone common noun shown against a claim about
      something else. An unscored lone candidate is not ambiguous: with no
      description or no claim to compare, the judge has nothing to weigh, and
      the top hit is taken as before.
    """
    if candidate_count <= 0:
        return False
    if candidate_count > 1:
        return True
    return top_score is not None and top_score < floor


#: Descriptions Wikidata gives an item that stands for a name or a list of
#: senses, never for an entity a claim is about. Matched whole and
#: case-blind against the search hit's English description.
_NOT_A_REFERENT = re.compile(
    r"(?:(?:female|male|unisex) )?given name|family name|surname|name"
    r"|Wikimedia disambiguation page",
    re.IGNORECASE,
)


def judgeable_hits(hits: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """The hits the judge is shown: every one but a name or disambiguation item (pure).

    A given-name, family-name or disambiguation item cannot be the person,
    place or thing a claim names, yet shown a long list the judge sometimes
    links an invented person to one: shown seven hits for "Jordan, who joined
    from Salesforce", it chose "unisex given name" in 7 of 10 runs, and for
    "My neighbour Priya" "female given name" in 2 of 10, against none of 10 at
    five hits. The rule reads the description the search response
    already carries, so it costs no call. Rank order is kept.
    """
    return [
        hit
        for hit in hits
        if not _NOT_A_REFERENT.fullmatch(" ".join(str(hit.get("description") or "").split()))
    ]


def offered_candidates(hits: Sequence[dict[str, object]]) -> list[Candidate]:
    """Each search hit as the judge sees it, in rank order (pure)."""
    return [
        (
            str(hit.get("id", "")),
            " ".join(str(hit.get("label") or "").split()),
            " ".join(str(hit.get("description") or "").split()),
        )
        for hit in hits
    ]


def judge_request(
    name: str, claim: str, candidates: Sequence[Candidate], *, nonce: str
) -> tuple[str, str]:
    """``(system, user)`` for one judge call (pure).

    The claim and the candidates are untrusted (a deposited document, and a
    wiki anyone may edit), so each is fenced and the instructions stay in the
    system turn (security F3).
    """
    listing = "\n".join(
        f"{qid} | {label or '(no label)'} | {description or '(no description)'}"
        for qid, label, description in candidates
    )
    user = (
        f"Name:\n{fence(name, nonce, label='name')}\n\n"
        f"Claim:\n{fence(claim, nonce, label='claim')}\n\n"
        f"Candidates:\n{fence(listing, nonce, label='candidates')}"
    )
    return _INSTRUCTION + "\n" + data_fence_instruction(nonce), user


def judge_prompt_hash() -> str:
    """The judge's prompt version: the request rendered over placeholders.

    Built by :func:`judge_request` itself, so any change to the instruction or
    to the layout of the user turn changes the hash and misses old verdicts.
    """
    system, user = judge_request(
        "{name}", "{claim}", [("{id}", "{label}", "{description}")], nonce="{nonce}"
    )
    return prompt_hash(system, user)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_reply(reply: str, qids: Sequence[str]) -> str | None:
    """The judge's pick from its reply: a QID in ``qids``, ``"none"``, or ``None`` (pure).

    ``None`` means the reply is unusable: not JSON, or naming an id it was not
    offered. A reply is never read as a pick it did not make, so an off-protocol
    answer falls back to the top hit rather than to a guessed candidate.
    """
    text = reply.strip()
    match = _JSON_OBJECT.search(text)
    raw: object = None
    if match is not None:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            raw = data.get("qid")
    elif text:
        raw = text.strip("\"'` .")
    if not isinstance(raw, str):
        return None
    answer = raw.strip()
    if answer.lower() == NONE_OF_THESE:
        return NONE_OF_THESE
    by_upper = {qid.upper(): qid for qid in qids}
    return by_upper.get(answer.upper())


@dataclass(frozen=True)
class Judgement:
    """The judge's answer for one name: a candidate index, or ``None`` for none of them."""

    index: int | None
    #: True when the answer was read from the ledger, not asked for.
    recorded: bool = False


def _model_label() -> str:
    selection = get_config().llm.for_purpose("subject_resolution")
    return f"{selection.provider}/{selection.model}"


def _answer_json(qid: str | None, qids: Sequence[str]) -> str:
    return json.dumps({"qid": qid, "candidates": list(qids)}, sort_keys=True)


def _recorded_pick(answer: str, qids: Sequence[str]) -> str | None:
    """The pick a ledger answer holds, if it names one of ``qids`` or none."""
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    qid = data.get("qid")
    if qid is None:
        return NONE_OF_THESE
    return qid if isinstance(qid, str) and qid in qids else None


def _judgement(pick: str, qids: Sequence[str], *, recorded: bool) -> Judgement:
    return Judgement(
        index=None if pick == NONE_OF_THESE else list(qids).index(pick), recorded=recorded
    )


async def judge_candidates(
    session: AsyncSession,
    name: str,
    claim: str,
    hits: Sequence[dict[str, object]],
) -> Judgement | None:
    """Which of ``hits`` the claim's ``name`` refers to, asked once and remembered.

    Returns ``None`` when no usable answer came back (the call failed, the
    breaker is open, or the reply was off protocol); the caller then takes the
    top hit, the resolver's behaviour without the judge. A usable answer is
    recorded in the ledger before it is returned. Flushes; the caller commits.
    """
    offered = offered_candidates(hits)
    qids = [qid for qid, _, _ in offered]
    key = subject_link_key(name, claim, offered)
    version = judge_prompt_hash()
    model = _model_label()

    recorded = await lookup_answer(session, ProbeKind.SUBJECT_LINK, version, key, model=model)
    if recorded is not None:
        pick = _recorded_pick(recorded, qids)
        if pick is not None:
            log.debug("Wikidata judge for %r: %s (recorded)", name, pick)
            return _judgement(pick, qids, recorded=True)

    system, user = judge_request(name, claim, offered, nonce=make_nonce())
    reply = await _llm_call(
        user,
        max_tokens=get_config().subjects.wikidata_judge_max_tokens,
        system=system,
        response_schema=_RESPONSE_SCHEMA,
        purpose="subject_resolution",
        temperature=0.0,
    )
    if reply is None:
        return None
    pick = parse_judge_reply(reply, qids)
    if pick is None:
        record_unusable_reply("Wikidata subject-link judge", reply)
        return None
    await record_answer(
        session,
        ProbeKind.SUBJECT_LINK,
        version,
        key,
        verdict=pick != NONE_OF_THESE,
        answer=_answer_json(None if pick == NONE_OF_THESE else pick, qids),
        model=model,
    )
    log.info("Wikidata judge for %r: %s of %d candidate(s)", name, pick, len(qids))
    return _judgement(pick, qids, recorded=False)
