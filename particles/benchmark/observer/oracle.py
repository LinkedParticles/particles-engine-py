# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted perception for the observer fixture — zero LLM spend by construction.

The rot harness's split, reused: extraction and the §6.6
contradiction probe are replaced by ground truth so the *decision logic* —
rung 2.5, duplicate suppression, the generation cascade, and the observer
lens — is what gets measured. Every other purpose routes to the rot harness's
:class:`~particles.benchmark.rot.oracle.RefusingProvider` and is counted.

Two extractors, one per arm. :class:`LinesExtractor` re-emits every line on
every extraction, so an unchanged claim reaches the store through duplicate
suppression. :class:`ChunkedLinesExtractor` sends two lines per chunk through
the real chunk-hash carry-forward, so an unchanged chunk's claims are carried
forward rather than re-emitted — the other way a re-observation reaches the
store.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from particles.benchmark.observer.generator import GLOBAL_CONTESTS, SLOTS, value_pattern
from particles.benchmark.observer.schema import MemoryLine
from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward
from particles.llm import CompletionError, VisionImage

_CLAIMS = re.compile(r"Claim A:\s*(?P<a>.*?)\n\s*\nClaim B:\s*(?P<b>.*)\Z", re.DOTALL)


class LinesExtractor:
    """Emit one candidate per memory line, with the line's own subject.

    Keyed by the exact deposited bytes, registered by the runner before each
    extract; an unregistered text yields no candidates and a quality note
    rather than an exception.
    """

    EXTRACTOR_ID = "observer-oracle"
    EXTRACTOR_VERSION = "1"

    def __init__(self) -> None:
        self._lines: dict[bytes, list[MemoryLine]] = {}
        self.calls = 0

    def register(self, text: str, lines: list[MemoryLine]) -> None:
        self._lines[text.encode("utf-8")] = lines

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        self.calls += 1
        lines = self._lines.get(content)
        if lines is None:
            return ExtractionResult(quality_notes=["observer-oracle: unregistered text"])
        return ExtractionResult(candidates=[_candidate(line) for line in lines])


def _candidate(line: MemoryLine) -> CandidateParticle:
    return CandidateParticle(
        content=line.text,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[line.subject],
    )


class ChunkedLinesExtractor(LinesExtractor):
    """The ``chunked`` arm: two lines per chunk, through carry-forward.

    The chunk text is the lines' own bullets, so a chunk whose two lines are
    unchanged hashes as before and its particles are carried forward — no
    candidate is emitted for them at all. A line added or dropped shifts every
    later chunk, which is then re-emitted and reaches duplicate suppression, as
    in the other arm. ``chunks_carried`` / ``chunks_extracted`` say how much of
    each run went which way, so a report shows the arm exercised the mechanism.
    """

    CHUNK_LINES = 2

    def __init__(self) -> None:
        super().__init__()
        self.chunks_carried = 0
        self.chunks_extracted = 0

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        self.calls += 1
        lines = self._lines.get(content)
        if lines is None:
            return ExtractionResult(quality_notes=["observer-oracle: unregistered text"])
        by_bullet = {_bullet(line): line for line in lines}
        step = self.CHUNK_LINES
        chunks = [
            ChunkUnit(
                chunk_id=f"lines_{start}",
                chunk_text="".join(_bullet(line) for line in lines[start : start + step]),
            )
            for start in range(0, len(lines), step)
        ]

        async def per_chunk(chunk_text: str) -> tuple[list[CandidateParticle], list[str], bool]:
            self.chunks_extracted += 1
            return (
                [_candidate(by_bullet[bullet + "\n"]) for bullet in chunk_text.splitlines()],
                [],
                False,
            )

        session = kwargs.get("session")
        entry_id = kwargs.get("corpus_entry_id")
        result = await extract_with_carry_forward(
            session,  # type: ignore[arg-type]
            chunks,
            entry_id if isinstance(entry_id, str) else None,
            self.EXTRACTOR_ID,
            self.EXTRACTOR_VERSION,
            call_llm=per_chunk,
        )
        self.chunks_carried += _carried_chunks(result)
        return result


def _bullet(line: MemoryLine) -> str:
    return f"- {line.text}\n"


def _carried_chunks(result: ExtractionResult) -> int:
    return sum(1 for note in result.quality_notes if note.startswith("CHUNK_CARRY_FORWARD:"))


def contradicts(claim_a: str, claim_b: str) -> bool:
    """Ground truth: do the two claims give one slot different values?

    A global line and the project line scripted to contest it
    (``GLOBAL_CONTESTS``) contradict too.
    """
    for global_text, contest in GLOBAL_CONTESTS.items():
        if {claim_a, claim_b} == {global_text, contest.text}:
            return True
    for _, _, pool in SLOTS.values():
        in_a = {v for v in pool if value_pattern(v).search(claim_a)}
        in_b = {v for v in pool if value_pattern(v).search(claim_b)}
        if in_a and in_b and in_a != in_b:
            return True
    return False


class SlotProbeProvider:
    """Answer the §6.6 contradiction-probe prompt from the slot table."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_model(self) -> str:
        return "observer-oracle:contradiction-probe"

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
        self.calls += 1
        match = _CLAIMS.search(prompt)
        if match is None:
            raise CompletionError("observer-oracle: not a contradiction-probe prompt")
        if contradicts(match["a"].strip(), match["b"].strip()):
            return "YES: the claims give one attribute different values"
        return "NO"
