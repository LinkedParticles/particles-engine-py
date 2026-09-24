# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted perception for the ``oracle`` and ``probe`` arms.

The Engine's *decision logic* (the §6.6 ladder, source trust, ranking) and its
*perception* (extraction, the contradiction probe) fail differently. These
three pieces replace perception with ground truth so the decision logic can be
measured on its own, for free, through the real pipeline:

* :class:`OracleExtractor` — an :class:`~particles.extraction.registry.ExtractorPlugin`
  that emits exactly what a perfect extractor would for each session;
* :class:`OracleProbeProvider` — a :class:`~particles.llm.CompletionProvider`
  answering the §6.6 contradiction probe from the world's value pools;
* :class:`RefusingProvider` — routed to every purpose an arm must not pay
  for, so an unanticipated call fails open (the caller's documented fallback)
  and is counted, rather than billing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from particles.benchmark.rot.generator import (
    SLOTS_BY_KEY,
    claim_history,
    claim_state,
    value_pattern,
)
from particles.benchmark.rot.schema import (
    EventKind,
    Phrasing,
    PoisonChannel,
    RotSession,
    RotWorld,
)
from particles.benchmark.rot.scoring import STRONG_HISTORY_MARKERS
from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.llm import CompletionError, VisionImage

#: The subject every scripted claim is about — the persona.
PERSONA_SUBJECT = "the user"


def oracle_claims(world: RotWorld, session: RotSession) -> list[str]:
    """What a perfect extractor records for ``session``.

    One state claim per stated value; for a transition, also the past-tense
    claim for the value it replaced (honest history the scorer must not count
    as stale); a past-tense claim for a history decoy; **nothing** for a
    ``relay`` or ``tool_turn`` poison (a perfect extractor does not adopt tool
    output as a fact about the user); and the asserted claim for a ``source``
    poison — the page does assert it, and keeping it out is the trust layer's
    job, which is what that channel tests.
    """
    if session.event_index is None:
        return [session.filler_claim] if session.filler_claim else []
    ev = world.events[session.event_index]
    noun = SLOTS_BY_KEY[ev.slot].noun
    if ev.kind in (EventKind.INITIAL, EventKind.UPDATE):
        claims = [claim_state(noun, ev.value)]
        if ev.phrasing is Phrasing.TRANSITION and ev.previous is not None:
            claims.append(claim_history(noun, ev.previous))
        return claims
    if ev.kind is EventKind.HISTORY:
        return [claim_history(noun, ev.value)]
    if ev.channel is PoisonChannel.SOURCE:
        return [claim_state(noun, ev.value)]
    return []


class OracleExtractor:
    """Emit the perfect extraction for each deposited session.

    Keyed by the exact deposited text, which the runner registers before each
    extract call; an unregistered text yields no candidates and a quality note
    rather than an exception (bad cases never abort a run).
    """

    EXTRACTOR_ID = "rot-oracle"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self._claims: dict[bytes, list[str]] = {}
        self.calls = 0

    def register(self, text: str, claims: list[str]) -> None:
        """Record the claims to emit for one deposited text."""
        self._claims[text.encode("utf-8")] = claims

    def accepts(self, source_type: str) -> bool:
        """Accept every source type — the arm owns which sessions reach it."""
        return True

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        """Return the registered claims for ``content`` as candidates."""
        self.calls += 1
        claims = self._claims.get(content)
        if claims is None:
            return ExtractionResult(quality_notes=["rot-oracle: unregistered session text"])
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=c,
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=[PERSONA_SUBJECT],
                )
                for c in claims
            ]
        )


_CLAIMS = re.compile(r"Claim A:\s*(?P<a>.*?)\n\s*\nClaim B:\s*(?P<b>.*)\Z", re.DOTALL)


def oracle_contradiction(world: RotWorld, claim_a: str, claim_b: str) -> bool:
    """Ground-truth §6.6 verdict: do the two claims give one slot different values?

    Two claims contradict iff some slot has a value in each, the value sets
    differ, and neither is framed as past (a past-tense mention of a
    superseded value does not disagree with the current one).
    """
    if STRONG_HISTORY_MARKERS.search(claim_a) or STRONG_HISTORY_MARKERS.search(claim_b):
        return False
    for pool in world.pools.values():
        in_a = {v for v in pool if value_pattern(v).search(claim_a)}
        in_b = {v for v in pool if value_pattern(v).search(claim_b)}
        if in_a and in_b and in_a != in_b:
            return True
    return False


class OracleProbeProvider:
    """Answer the §6.6 contradiction-probe prompt from the world."""

    def __init__(self, world: RotWorld) -> None:
        self._world = world
        self.calls = 0

    @property
    def provider_model(self) -> str:
        """Disclosure key recorded on the run tuple."""
        return "rot-oracle:contradiction-probe"

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        images: Sequence[VisionImage] | None = None,
        response_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        **opts: object,
    ) -> str:
        """``"YES: …"`` or ``"NO"`` in the probe's own reply format."""
        self.calls += 1
        match = _CLAIMS.search(prompt)
        if match is None:
            raise CompletionError("rot-oracle: not a contradiction-probe prompt")
        if oracle_contradiction(self._world, match["a"].strip(), match["b"].strip()):
            return "YES: the claims give one attribute different values"
        return "NO"


class RefusingProvider:
    """A provider that never answers — the zero-spend guarantee of a scripted arm.

    Every purpose an arm must not pay for routes here. The call raises
    :class:`~particles.llm.CompletionError`, which every call site already
    handles with its purpose-specific fallback, and is counted so
    the report can disclose a seam the arm did not anticipate.
    """

    def __init__(self, purpose: str, counts: dict[str, int]) -> None:
        self._purpose = purpose
        self._counts = counts

    @property
    def provider_model(self) -> str:
        """Disclosure key recorded on the run tuple."""
        return f"rot-refused:{self._purpose}"

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        images: Sequence[VisionImage] | None = None,
        response_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        **opts: object,
    ) -> str:
        """Count the call and refuse it."""
        self._counts[self._purpose] = self._counts.get(self._purpose, 0) + 1
        raise CompletionError(
            f"rot benchmark: purpose {self._purpose!r} is not paid for by this arm"
        )
