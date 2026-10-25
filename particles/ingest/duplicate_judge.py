# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The equivalence judge: are two claims one claim?.

One prompt, one reply format, one parser, shared by every caller that asks
the duplicate question of a model: the curation duplicate cards and the
``LLM_JUDGE`` link mode (:mod:`particles.operations.links_suggest`),
``reindex --estimate``, and the restatement match of a re-extracted
``MUTABLE`` source (:mod:`particles.ingest.restatement`). It lives
in ``ingest`` so the extraction pipeline can ask it without importing
``operations``.

The model call itself is the caller's: :func:`run_judge` takes it as a
parameter, so each caller keeps its own test seam and breaker.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from particles.core.probe_verdict import prompt_hash
from particles.core.rung_ladder import DuplicateVerdict, QuestionKind, RungStamp
from particles.core.schema import JudgeVerdictKind
from particles.ingest.ladder_shell import report
from particles.llm.breaker import PROBE_MAX_TOKENS
from particles.llm.fencing import data_fence_instruction, fence, make_nonce

__all__ = [
    "DUPLICATE_VERDICT",
    "JUDGE_TOKENS_PER_PAIR",
    "JudgeCall",
    "duplicate_judge_stamp",
    "judge_key",
    "judge_max_tokens",
    "judge_prompt",
    "judge_request",
    "key_index",
    "parse_verdicts",
    "run_judge",
    "tally_judge",
]

# The judge's reply budget, per pair and per call. A pair is keyed by its batch
# position (``p0``, ``p1``, …), mapped back locally, so a reply entry is
# ``"p24": "PARAPHRASE",`` whatever the tokenizer. The budget covers a reply
# that spends one token per byte, indented and fenced, with headroom; the
# test pins that bound. Keying by ``a8+b8`` particle-id prefixes cost
# ``claude-haiku-5-5`` about 57 tokens an entry, so a 25-pair batch ran out of
# a 40-per-pair budget, the truncated JSON parsed to nothing, and every pair
# read UNSURE.
JUDGE_TOKENS_PER_PAIR = 48
_JUDGE_TOKENS_BASE = 64
_JUDGE_MIN_TOKENS = 200


class JudgeCall(Protocol):
    """The model call a caller hands :func:`run_judge`: request in, reply text out.

    ``system`` carries the trusted instructions and ``prompt`` the fenced pairs
    (security F3), so a caller's call must forward ``system`` to the model.
    """

    def __call__(
        self, prompt: str, *, max_tokens: int, system: str | None = None
    ) -> Awaitable[str | None]: ...


def judge_max_tokens(pair_count: int) -> int:
    """The output budget for one judge call over ``pair_count`` pairs.

    The reply's own size, plus the room a model that thinks before it answers
    spends first (``PROBE_MAX_TOKENS``): its thinking counts against the same
    budget, so a budget sized to the reply alone can return no text.
    """
    reply = max(_JUDGE_MIN_TOKENS, _JUDGE_TOKENS_BASE + JUDGE_TOKENS_PER_PAIR * pair_count)
    return PROBE_MAX_TOKENS + reply


def judge_key(index: int) -> str:
    """The prompt key for the pair at ``index`` in its batch."""
    return f"p{index}"


def key_index(key: str) -> int | None:
    """The batch position a reply key names, tolerating case and brackets."""
    text = key.strip().strip("[]").strip().lower()
    if not text.startswith("p") or not text[1:].isdigit():
        return None
    return int(text[1:])


_JUDGE_INSTRUCTIONS = (
    "You are curating an epistemic knowledge base. The user message holds "
    "candidate pairs of claims that may assert the SAME underlying fact (a "
    "paraphrase) or DIFFERENT facts. Each pair has a key in brackets and its two "
    "claims, A and B. For each pair, choose exactly one verdict:\n"
    "- PARAPHRASE: the two claims assert the same underlying fact\n"
    "- DISTINCT: the two claims assert different facts\n"
    "- UNSURE: you cannot tell\n\n"
    "Return ONLY a JSON object mapping each bracketed pair key to its "
    'verdict, e.g. {"p0": "PARAPHRASE", "p1": "DISTINCT"}. No prose.'
)


def judge_request(pairs: Sequence[tuple[str, str]], *, nonce: str | None = None) -> tuple[str, str]:
    """``(system, user)`` for one judge call over ``(claim_a, claim_b)`` pairs.

    The claims are stored text, so the instructions stay in the system turn
    and every claim is fenced (security F3). One nonce covers the call, and
    each claim's label names its pair and side (``p0_a``, ``p0_b``, …), so a
    claim cannot pose as another pair. ``nonce`` is fixed only by
    :func:`duplicate_judge_stamp`; a real call mints its own.
    """
    nonce = nonce or make_nonce()
    blocks = []
    for i, (a, b) in enumerate(pairs):
        key = judge_key(i)
        blocks.append(
            f"[{key}]\nA:\n{fence(a, nonce, label=f'{key}_a')}\n"
            f"B:\n{fence(b, nonce, label=f'{key}_b')}"
        )
    system = f"{_JUDGE_INSTRUCTIONS}\n\n{data_fence_instruction(nonce)}"
    return system, "\n\n".join(blocks)


def judge_prompt(pairs: Sequence[tuple[str, str]]) -> str:
    """The judge's whole request over ``pairs`` as one string, for sizing a call."""
    system, user = judge_request(pairs)
    return f"{system}\n\n{user}"


def duplicate_judge_stamp() -> RungStamp:
    """The judge as the duplicate ladder's model rung, at its prompt's version."""
    return RungStamp(
        "duplicate_judge", prompt_hash(*judge_request([("{a}", "{b}")], nonce="{nonce}"))
    )


#: The judge's answer in the duplicate ladder's vocabulary; ``UNSURE`` abstains.
DUPLICATE_VERDICT = {
    JudgeVerdictKind.PARAPHRASE: DuplicateVerdict.SAME,
    JudgeVerdictKind.DISTINCT: DuplicateVerdict.DIFFERENT,
}


def tally_judge(verdicts: Sequence[JudgeVerdictKind | None]) -> None:
    """Count the judge's verdicts as the duplicate ladder's.

    ``UNSURE`` and an omitted pair are the rung's abstention; with one rung,
    an abstention leaves no rung deciding.
    """
    stamp = duplicate_judge_stamp()
    for verdict in verdicts:
        mapped = DUPLICATE_VERDICT.get(verdict) if verdict is not None else None
        value = (mapped or DuplicateVerdict.ABSTAIN).value
        report(QuestionKind.DUPLICATE, "rungs", stamp, value)
        report(QuestionKind.DUPLICATE, "settled", stamp if mapped else None, value)


def parse_verdicts(response: str | None) -> dict[str, JudgeVerdictKind]:
    """Parse the LLM's JSON verdict object, tolerating code fences / prose."""
    if not response:
        return {}
    text = response.strip()
    # Strip a ```json … ``` fence if present, then isolate the JSON object.
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        raw: Any = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, JudgeVerdictKind] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = JudgeVerdictKind(str(value).strip().upper())
        except ValueError:
            continue
    return out


async def run_judge(
    pairs: Sequence[tuple[str, str]],
    call: JudgeCall | Callable[..., Awaitable[str | None]],
) -> list[JudgeVerdictKind | None]:
    """One judge call over ``(claim_a, claim_b)`` pairs; a verdict per pair, in order.

    A pair the model omits or answers unparseably is ``None``.
    """
    system, prompt = judge_request(pairs)
    response = await call(prompt, max_tokens=judge_max_tokens(len(pairs)), system=system)
    verdicts: list[JudgeVerdictKind | None] = [None] * len(pairs)
    for key, verdict in parse_verdicts(response).items():
        index = key_index(key)
        if index is not None and index < len(pairs) and verdicts[index] is None:
            verdicts[index] = verdict
    tally_judge(verdicts)
    return verdicts
