# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Match emitted particles to expected particles.

Two strategies:

* ``embedding`` — cosine similarity over the shared sentence-transformer
  embedding model. Fast, deterministic, no API key required.
  Default threshold 0.80 picks up paraphrase that the model considers
  close while rejecting unrelated claims.
* ``llm`` — first pre-filter pairs with cosine similarity ≥ a wider
  threshold (default 0.65), then ask an LLM judge whether each pair's
  meanings match. Higher fidelity at higher cost; reuses the wiki
  exporter's Layer-B JSON-verdict prompt for consistency.

The matching is greedy in similarity-descending order so a single
emitted particle can't satisfy more than one expected particle (and
vice versa). This is the assignment most operators expect when
eyeballing the report — strongest pairings first.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from particles.benchmark.schema import ExpectedParticle
from particles.core.schema import Particle

log = logging.getLogger(__name__)


class EquivalenceJudge(StrEnum):
    EMBEDDING = "embedding"
    LLM = "llm"


@dataclass
class MatchResult:
    """The output of :func:`match_emitted_to_expected`.

    * ``matched`` — ``(expected, emitted)`` pairs the judge accepted
    * ``missed_required`` — required expected particles with no matched
      emitted particle
    * ``spurious`` — emitted particles that matched no expected particle
    * ``under_confidence`` — ``(expected, emitted)`` pairs that matched
      *semantically* but whose stated confidence was below the
      expected ``confidence_min`` — neither precision nor recall credit,
      but a separate signal so operators can see the under-trusting
      failure mode (the extractor *got it right* but stated it too
      timidly to surface in operator workflows).
    """

    matched: list[tuple[ExpectedParticle, Particle]] = field(default_factory=list)
    missed_required: list[ExpectedParticle] = field(default_factory=list)
    spurious: list[Particle] = field(default_factory=list)
    under_confidence: list[tuple[ExpectedParticle, Particle]] = field(default_factory=list)

    @property
    def matched_ids(self) -> set[str]:
        """Set of emitted particle IDs that participated in a match.

        Excludes under-confidence partial matches (which by design do
        not count toward precision/recall but do feed the
        calibration-error histogram via :attr:`under_confidence`).
        """
        return {p.id for _, p in self.matched}


def subject_qualified(content: str, subjects: Sequence[str]) -> str:
    """Render one emitted claim with any subject its ``content`` omits.

    A particle's subject lives in a **field**, not in its ``content`` — the
    schema's own separation — so an extractor that obeys the data model emits
    "Operating costs for 2025 were $940,000." and links it to the subject
    *Halcyon Grid Cooperative*. Gold prose written by a human restates the
    subject inline. Comparing the two directly scores the pair on a difference
    the data model asked the extractor to make, and charges it twice: once to
    precision as a spurious claim, once to recall as a required miss.

    Only *absent* subjects are prepended. A claim that already names its
    subject is returned unchanged (identity, not a copy), because
    double-naming it — "Fernwood Systems: Fernwood Systems is a 40-person …"
    — measurably distorts the embedding.

    **This function is one of two renderings, never a replacement.** The caller
    scores bare ``content`` *and* this, and keeps the higher similarity per
    pair; see :func:`match_emitted_to_expected`. Using it alone is wrong and
    the seed suites prove it — a suite whose gold was copied verbatim from
    extractor output is already subject-elided, and unconditional
    qualification drove ``numismatic-seed-001`` from 1.00 to 0.57 precision.

    Measured on ``prose-article-seed-001`` / ``claude-haiku-4-5`` over one
    fixed emission set, best-of-both moved precision 0.800 -> 0.912 and recall
    0.714 -> 0.857, admitting nine pairs that are the same fact in different
    clothes and no pair that is not. It is **not** a loosened threshold: the
    0.80 equivalence floor and 0.65 LLM pre-filter are untouched. Lowering the
    threshold to 0.70 instead was measured on the same emissions and is worse
    (recall 0.800), because it admits unrelated claims rather than reading
    related ones correctly.
    """
    lowered = content.lower()
    absent = [name.strip() for name in subjects if name.strip()]
    absent = [name for name in absent if name.lower() not in lowered]
    if not absent:
        return content
    return f"{', '.join(absent)}: {content}"


async def match_emitted_to_expected(
    emitted: list[Particle],
    expected: list[ExpectedParticle],
    *,
    judge: EquivalenceJudge = EquivalenceJudge.EMBEDDING,
    threshold: float = 0.80,
    llm_prefilter: float = 0.65,
    subject_names: Mapping[str, Sequence[str]] | None = None,
) -> MatchResult:
    """Greedy-similarity assignment of emitted to expected.

    Returns a :class:`MatchResult` with every emitted/expected pair
    classified into matched / spurious / missed_required /
    under_confidence. The greedy assignment runs in descending order
    of pair similarity so the strongest pairings land first; subsequent
    iterations skip already-matched particles on either side.

    ``threshold`` is the minimum similarity for a pair to count as a
    match under the embedding judge. Under the LLM judge it is *also*
    consulted — the LLM is asked only about pairs whose similarity
    is in ``[llm_prefilter, threshold)``; pairs above ``threshold``
    are accepted on similarity alone, pairs below ``llm_prefilter``
    are rejected without an LLM call. This bounds the LLM token cost.

    ``subject_names`` maps an emitted particle's ``id`` to the subject names it
    is about, and turns on subject-aware matching: the emitted side
    is embedded as :func:`subject_qualified` renders it, so a claim whose
    subject lives in a field is not scored against gold prose as if it had
    omitted the subject. Passing ``None`` (the default) embeds bare ``content``
    — the pre-0262 semantics, kept reachable so historical runs stay
    reproducible.

    **Why names and not ``particle.subject_ids``.** In a store-backed particle
    those ids are UUIDs, and prefixing a claim with a UUID would silently
    poison every similarity in the matrix. The benchmark runner happens to
    carry unresolved subject *names* there, but relying on that would make a
    correct-looking call site produce garbage the day it is handed a persisted
    particle. The caller states the names explicitly instead.
    """
    if not emitted and not expected:
        return MatchResult()

    expected_texts = [e.content for e in expected]
    emitted_texts = [p.content for p in emitted]
    similarities = _similarity_matrix(emitted_texts, expected_texts)

    # Per (emitted, expected) pair, the rendering whose cosine won — consulted
    # only by the LLM judge, so it reads the exact string that was scored.
    judged_text: dict[tuple[int, int], str] = {}

    if subject_names:
        # Score each emitted claim under BOTH renderings and keep the better
        # one per pair. Qualifying *unconditionally* is wrong, and the seed
        # suites prove it: the authoring contract says to copy the extractor's
        # current output verbatim into the gold, so a structured suite's gold
        # is already subject-elided and prepending the subject destroys an
        # exact match (numismatic-seed-001 fell 1.00 -> 0.57 precision when
        # this was tried).
        #
        # Taking the max is monotone **per pair** — a pair's similarity can
        # only rise, so no pair that cleared the threshold stops clearing it.
        # The greedy assignment is *not* monotone in the same way: raising one
        # pair can reorder the greedy pass and strand a gold that a
        # lower-scoring pair used to match. See
        # ``test_raising_a_score_can_reshuffle_the_greedy_assignment``. The
        # guarantee this rests on is therefore the pair-level one — it is what
        # rules out the numismatic collapse, where substitution pushed scores
        # *below* the floor wholesale — and the suites are measured, not
        # assumed.
        qualified = [subject_qualified(p.content, subject_names.get(p.id, ())) for p in emitted]
        changed = [
            i
            for i, (bare, qual) in enumerate(zip(emitted_texts, qualified, strict=True))
            if bare != qual
        ]
        if changed:
            # Only the rewritten rows need a second embedding pass; on the
            # measured suite that is 21 of 80 claims.
            q_sims = _similarity_matrix([qualified[i] for i in changed], expected_texts)
            for row, i in enumerate(changed):
                for j in range(len(expected_texts)):
                    if q_sims[row][j] > similarities[i][j]:
                        similarities[i][j] = q_sims[row][j]
                        judged_text[(i, j)] = qualified[i]

    # Build all candidate pairs sorted by similarity (desc).
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(emitted)):
        for j in range(len(expected)):
            sim = similarities[i][j]
            if judge is EquivalenceJudge.EMBEDDING and sim < threshold:
                continue
            if judge is EquivalenceJudge.LLM and sim < llm_prefilter:
                continue
            candidates.append((sim, i, j))
    candidates.sort(reverse=True)

    used_emitted: set[int] = set()
    used_expected: set[int] = set()
    result = MatchResult()

    for sim, i, j in candidates:
        if i in used_emitted or j in used_expected:
            continue
        emitted_p = emitted[i]
        expected_p = expected[j]

        # LLM judge for the contested band [prefilter, threshold);
        # above-threshold pairs are accepted on cosine alone.
        if (
            judge is EquivalenceJudge.LLM
            and sim < threshold
            and not await _llm_pair_aligned(judged_text.get((i, j), emitted_texts[i]), expected_p)
        ):
            continue

        # Semantic match — but is the stated confidence high enough?
        if emitted_p.confidence.value < expected_p.confidence_min:
            result.under_confidence.append((expected_p, emitted_p))
        else:
            result.matched.append((expected_p, emitted_p))
        used_emitted.add(i)
        used_expected.add(j)

    # Anything left over.
    for i, emitted_p in enumerate(emitted):
        if i not in used_emitted:
            result.spurious.append(emitted_p)
    for j, expected_p in enumerate(expected):
        if j not in used_expected and expected_p.required:
            result.missed_required.append(expected_p)
    return result


# ---------------------------------------------------------------------------
# Internal helpers — embedding + LLM-judge
# ---------------------------------------------------------------------------


def _similarity_matrix(emitted_texts: list[str], expected_texts: list[str]) -> list[list[float]]:
    """Cosine similarity matrix between emitted and expected texts.

    Uses the shared :func:`particles.embeddings.get_embedding_model`.
    Embeddings come back normalised (per the encoder config), so the
    cosine reduces to a dot product. Returns a zero matrix if the
    embedding model is unavailable (e.g. test environment with the
    embedding model mocked to ``None``) — every pair scores 0 and no
    matches are found, which is the conservative fallback.
    """
    if not emitted_texts or not expected_texts:
        return [[0.0 for _ in expected_texts] for _ in emitted_texts]

    from particles.embeddings import get_embedding_model

    model = get_embedding_model()
    if model is None:
        log.warning("Embedding model unavailable; all similarities = 0")
        return [[0.0 for _ in expected_texts] for _ in emitted_texts]

    import numpy as np

    emit_emb = np.array(
        model.encode(emitted_texts, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    exp_emb = np.array(
        model.encode(expected_texts, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    # clamp negatives to 0 so the judge's similarities stay on the
    # normative [0, 1] scale (vectorized analogue of embeddings.cosine_similarity).
    sims = np.clip(emit_emb @ exp_emb.T, 0.0, 1.0)
    return sims.tolist()  # type: ignore[no-any-return]


_JUDGE_PROMPT = """\
You are auditing one (claim, gold-standard claim) pair. Does the claim's
meaning match the gold standard's meaning? Use ONLY the two strings
below — no external knowledge.

Respond with a single word: "aligned" or "misaligned".

CLAIM:        {emitted}
GOLD STANDARD: {expected}"""


async def _llm_pair_aligned(emitted_text: str, expected: ExpectedParticle) -> bool:
    """LLM-judge whether one emitted claim matches one expected particle.

    Takes the rendered emitted *text*, not the ``Particle``, so the judge reads
    exactly the string the embedding judge scored — subject-qualified when
    subject-aware matching is on. Handing the model bare ``content``
    while the cosine was computed on the qualified form would make the two
    judges disagree about what the claim even says, on precisely the claims
    (subject in a field, not in the prose) the qualification exists for.

    Routes through the ``benchmark`` completion purpose; the
    provider/model resolve from ``config.llm.benchmark`` (falling back to
    ``config.llm.default``). The hardcoded model default this helper used to
    carry was removed.

    Conservative on the judge being unavailable: an exception or an
    unparseable response returns ``False`` so the pair does *not*
    silently match. The operator sees this in the report's quality
    notes (the runner records judge-call failures separately).

    **That conservatism is the right default for §13.3 and the wrong one for
    calibration**: a failed judge call becomes a label saying the
    extractor was incorrect, so an unreachable judge does not degrade a
    calibration gracefully — it manufactures negatives and biases the fit.
    Which is why the temperature note below matters more than it looks.

    ``temperature=0.0`` asks for a deterministic verdict, which is the whole
    point of a judge whose output becomes a label. Models that have deprecated
    the parameter answer it with a 400 (``claude-sonnet-5``, measured
    2026-08-03, where every contested pair failed and silently returned
    ``False``); the Anthropic adapter now retries once without it and remembers,
    so the request no longer fails and determinism is preserved wherever the
    model still supports it.
    """
    from particles.llm import complete

    prompt = _JUDGE_PROMPT.format(emitted=emitted_text, expected=expected.content)
    try:
        verdict = (await complete("benchmark", prompt, max_tokens=16, temperature=0.0)).lower()
    except Exception as exc:
        log.warning("LLM judge call failed for pair (labelled not-aligned): %s", exc)
        return False
    return verdict.startswith("aligned")
