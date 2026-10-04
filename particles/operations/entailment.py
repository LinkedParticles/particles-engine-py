# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The entailment judge: is one claim entailed by a set of premise claims?.

Abstraction promotion gates every synthesized general claim on it
(``consolidation.abstraction.require_entailment``), and its revalidation
ladder re-asks it when a premise changes. The model-prior leakage benchmark
asks it of every sentence of a query answer against the particles
the answer was composed from. One judge, two rubrics:

* the **prompt shape** is shared — trusted instructions in the system turn,
  the claim and each premise behind one per-call nonce fence in the user turn
  (claims are LLM-extracted from untrusted sources and must not be able to
  coerce a verdict), JSON out;
* the **rubric** is the caller's — what counts as entailed differs. Promotion
  fails a general claim that drops a date or a count, because a reader of the
  general claim alone could no longer recover it; an answer sentence that
  restates a dated premise without its date asserts nothing beyond it.

The call itself stays with the caller. Each caller owns its LLM seam — the
abstraction pass its circuit-breaker binding (which its tests patch), the
benchmark its retrying, failure-classifying scored call — and its purpose
routing, which is what lets the judge run on a different model
from the one whose output it judges. This module builds the prompt and reads
the reply; it makes no call and touches no store.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from particles.llm import data_fence_instruction, fence, make_nonce


class EntailmentVerdict(StrEnum):
    """What the judge decided about one claim."""

    ENTAILED = "entailed"
    NOT_ENTAILED = "not_entailed"
    #: The claim asserts nothing about the world (meta-commentary, a statement
    #: about what the premises do or do not say). Only a rubric with
    #: ``allow_no_claim`` can return it.
    NO_CLAIM = "no_claim"


@dataclass(frozen=True)
class EntailmentRubric:
    """One caller's definition of "entailed", and how its prompt is labelled.

    ``instructions`` is the whole trusted system text bar the fence
    instruction, including the JSON shape the judge must return. With
    ``allow_no_claim`` the reply also carries ``asserts_claim``, and a
    ``false`` there reads as :attr:`EntailmentVerdict.NO_CLAIM`.
    """

    instructions: str
    claim_heading: str
    claim_label: str
    premise_heading: str
    premise_label: str
    allow_no_claim: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        """The structured-output schema for the reply."""
        # ``reason`` comes first so the verdict is decoded after the reasoning:
        # a verdict-first reply commits and then argues itself out of it in
        # ``reason``. Strict-dialect providers emit properties in schema order;
        # the Anthropic adapter ignores the schema, so a rubric's instructions
        # name the order too.
        properties: dict[str, Any] = {"reason": {"type": "string"}}
        required: list[str] = ["reason"]
        if self.allow_no_claim:
            properties["asserts_claim"] = {"type": "boolean"}
            required.append("asserts_claim")
        properties["entailed"] = {"type": "boolean"}
        required.append("entailed")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def entailment_prompt(
    claim: str,
    premises: Sequence[str],
    *,
    rubric: EntailmentRubric,
    context: Sequence[tuple[str, str, str]] = (),
) -> tuple[str, str]:
    """``(system, user)`` for one judge call.

    ``context`` is ``(heading, label, text)`` blocks placed before the claim,
    fenced like everything else in the user turn — for a claim that cannot be
    read alone (an answer sentence whose "it" refers to the one before). The
    judge is told what they are by the rubric, never by the blocks.
    """
    nonce = make_nonce()
    system = rubric.instructions + "\n\n" + data_fence_instruction(nonce)
    lead = "".join(
        f"{heading}:\n{fence(text, nonce, label=label)}\n\n" for heading, label, text in context
    )
    user = (
        lead
        + f"{rubric.claim_heading}:\n{fence(claim, nonce, label=rubric.claim_label)}\n\n"
        + "\n\n".join(
            f"{rubric.premise_heading} {i + 1}:\n"
            f"{fence(content, nonce, label=f'{rubric.premise_label}_{i + 1}')}"
            for i, content in enumerate(premises)
        )
    )
    return system, user


def parse_json_object(response: str | None) -> dict[str, Any] | None:
    """Tolerantly isolate and parse one JSON object from an LLM reply."""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        raw: Any = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        raw = None
    if isinstance(raw, dict):
        return raw
    # A reason-first judge sometimes reasons in prose before the object, and
    # that prose can carry a brace of its own ("the set {dana, kofi, mei}"),
    # which defeats the outermost-brace slice. Decode from each brace in turn
    # and take the first object that parses.
    decoder = json.JSONDecoder()
    while start != -1:
        try:
            raw, _ = decoder.raw_decode(text, start)
        except ValueError:
            raw = None
        if isinstance(raw, dict):
            return raw
        start = text.find("{", start + 1)
    return None


def parse_entailment(
    response: str | None, *, rubric: EntailmentRubric
) -> tuple[EntailmentVerdict, str] | None:
    """``(verdict, reason)`` from a judge reply; ``None`` when it is unusable.

    An unusable reply — nothing, no JSON object, a non-boolean ``entailed`` —
    is never read as either verdict: the caller decides what an unrunnable
    check means (promotion discards the candidate; the benchmark excludes the
    sentence from the denominator).
    """
    data = parse_json_object(response)
    if data is None:
        return None
    reason = str(data.get("reason") or "").strip()
    if rubric.allow_no_claim:
        asserts = data.get("asserts_claim")
        if not isinstance(asserts, bool):
            return None
        if not asserts:
            return EntailmentVerdict.NO_CLAIM, reason
    entailed = data.get("entailed")
    if not isinstance(entailed, bool):
        return None
    return (EntailmentVerdict.ENTAILED if entailed else EntailmentVerdict.NOT_ENTAILED), reason
