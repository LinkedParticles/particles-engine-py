"""Layer B — semantic-alignment LLM-judge.

Layer A only checks that every cited ID belongs to the allowed set;
the LLM can still cite a real particle whose *content* is unrelated to
the claim it backs. Layer B runs a second LLM call ("judge") over each
(claim_window, particle.content) pair and returns one of three
verdicts: ``supports`` / ``unrelated`` / ``contradicts``.

The pass rule lives in :func:`_decide_layer_b_pass`: any
``contradicts`` is a hard fail, and the ``unrelated`` fraction must be
within the operator-configured tolerance
(``config.wiki.layer_b_unrelated_tolerance``). The
:class:`LayerBResult` carries the verdict counts plus the
``misalignments`` list so the orchestrator's strict-retry prompt can
show the LLM exactly what it got wrong.

Also home to :func:`_call_synthesis_llm` — the single LLM-call seam
used by both the judge here and the orchestrator's first/second
synthesis call. Wrapped as its own helper so tests can mock at
``particles.llm.set_client`` without monkey-patching this module
directly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from particles.core.schema import Particle
from particles.render.article_synthesis.layer_a import _CITATION_RE, _short_id

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM call seam
# ---------------------------------------------------------------------------


async def _call_synthesis_llm(
    *, prompt: str, max_tokens: int, temperature: float, system: str | None = None
) -> str:
    """Send the synthesis prompt and return the raw response text.

    Routes through the ``synthesis`` completion purpose: the
    provider/model resolve from ``config.llm.synthesis`` (falling back to
    ``config.llm.default``). Tests still mock at the ``particles.llm.set_client``
    seam — that reaches the ``anthropic`` adapter this purpose resolves to.

    ``system`` carries the trusted instructions for the Layer B judge call (F11
    hardening); the orchestrator's synthesis calls leave it ``None`` and are
    unchanged.

    Any provider exception propagates — callers translate to a fallback
    decision. The adapter runs the synchronous SDK call in a worker thread, so
    ``KeyboardInterrupt`` stays responsive between subjects on a long
    multi-hundred-subject export (the property this seam guaranteed before the
    port consolidated it).
    """
    from particles.llm import complete

    return await complete(
        "synthesis", prompt, max_tokens=max_tokens, temperature=temperature, system=system
    )


# ---------------------------------------------------------------------------
# Layer B — semantic-alignment LLM-judge
# ---------------------------------------------------------------------------

# Window of characters either side of a citation marker we feed to the
# judge as the "claim under review". Wider than a single sentence to
# give the judge enough context without ballooning prompt size.
_LAYER_B_WINDOW = 200


# F11 hardening: the audit instructions live in the trusted ``system`` turn; the
# (claim, particle) pairs — both attacker-influenceable (synthesised body text +
# LLM-extracted particle content) — go in the user turn behind a per-call nonce
# fence. The Layer A citation-id membership gate stays the structural backstop.
_LAYER_B_INSTRUCTIONS = """\
You are auditing one synthesised article for citation accuracy. For each
(claim, particle) pair in the user message, decide which of three verdicts
best describes their relationship:

  - "supports" — the particle is a necessary input to the claim. The
    claim may paraphrase, summarise, or combine the particle with other
    sources, but the particle's content is consistent with what the
    claim asserts.
  - "unrelated" — the particle is real but is not a necessary input to
    the claim. The claim could be made without this citation; the
    citation looks ornamental.
  - "contradicts" — the particle asserts something incompatible with
    the claim. This is the hard failure case: the LLM has cited a
    particle whose content actively disagrees with the claim it backs.

Use ONLY the particle content — no external knowledge. Respond as a
JSON array, one object per pair, with keys:
  - id: integer pair id
  - verdict: "supports" | "unrelated" | "contradicts"
  - reason: under 20 words

Do not include any prose outside the JSON array."""


def extract_cited_segments(body: str, window: int = _LAYER_B_WINDOW) -> list[tuple[str, str]]:
    """Return ``(segment_text, short_id)`` for every citation in ``body``.

    The segment is a ±``window``-char span around the citation marker,
    snapped to the nearest sentence boundary inside the window. Used as
    the "claim" the Layer-B judge evaluates against the cited particle.
    """
    segments: list[tuple[str, str]] = []
    for match in _CITATION_RE.finditer(body):
        start = max(0, match.start() - window)
        end = min(len(body), match.end() + window)
        left_dot = body.rfind(".", start, match.start())
        right_dot = body.find(".", match.end(), end)
        seg_start = left_dot + 1 if left_dot != -1 else start
        seg_end = right_dot + 1 if right_dot != -1 else end
        segments.append((body[seg_start:seg_end].strip(), match.group(1).lower()))
    return segments


def _build_layer_b_prompt(
    segments: list[tuple[str, str]], particle_by_short: dict[str, Particle]
) -> tuple[str, str]:
    """Assemble the batched judge ``(system, user)`` over every cited segment.

    The audit instructions go in ``system``; the ``PAIRS:`` block goes in
    ``user`` behind a per-call nonce fence (F11), so neither the synthesised
    claim text nor the cited particle content can override the judge rules.
    """
    from particles.llm import data_fence_instruction, fence, make_nonce

    pairs: list[str] = []
    for idx, (seg, sid) in enumerate(segments):
        particle = particle_by_short.get(sid)
        particle_text = particle.content if particle else "(missing)"
        pairs.append(f"  id: {idx}\n  claim: {seg!r}\n  particle [{sid}]: {particle_text!r}\n")
    nonce = make_nonce()
    system = f"{_LAYER_B_INSTRUCTIONS}\n\n{data_fence_instruction(nonce)}"
    user = "PAIRS:\n" + fence("\n".join(pairs), nonce, label="pairs")
    return system, user


# Verdict normalisation. The trichotomy is supports / unrelated /
# contradicts. Any other string the LLM returns
# (e.g. "partially supports", legacy "aligned" / "misaligned") degrades
# to "unrelated" — counts toward the tolerance but doesn't hard-fail
# the article. The exception is "aligned" specifically, which is the
# pre-trichotomy positive verdict; legacy responses normalise to
# "supports" so a model still calibrated to the old prompt doesn't
# spuriously fail an article.
_VERDICT_ALIASES: dict[str, str] = {
    "supports": "supports",
    "support": "supports",
    "supported": "supports",
    "aligned": "supports",  # legacy pre-trichotomy verdict
    "unrelated": "unrelated",
    "irrelevant": "unrelated",
    "ornamental": "unrelated",
    "contradicts": "contradicts",
    "contradict": "contradicts",
    "contradicted": "contradicts",
    "conflict": "contradicts",
    "conflicts": "contradicts",
}


def _normalise_verdict(raw: str | None) -> str:
    """Map any judge-emitted verdict string to one of the three buckets.

    Unknown / malformed verdicts degrade to ``unrelated`` so a noisy
    judge counts toward the tolerance budget but doesn't hard-fail an
    article — the "default unknown verdicts to a conservative bucket"
    rule (§Consequences).
    """
    if raw is None:
        return "unrelated"
    key = raw.strip().lower()
    return _VERDICT_ALIASES.get(key, "unrelated")


def _parse_layer_b_verdicts(raw: str) -> list[dict[str, Any]] | None:
    """Parse the judge's JSON response; return None if unparseable."""
    cleaned = raw.strip()
    # Tolerate ```json … ``` fences the model sometimes emits despite the
    # explicit instruction not to wrap.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict)]


@dataclass(frozen=True)
class LayerBResult:
    """Layer B judge outcome.

    Carries the trichotomy verdict counts plus the pass/fail decision
    after applying the operator-configured tolerance. ``misalignments``
    bundles the ``unrelated`` and ``contradicts`` verdicts (with their
    judge-supplied reasons) so the strict-retry prompt can show the
    LLM exactly what it got wrong.
    """

    passed: bool | None
    supports_count: int = 0
    unrelated_count: int = 0
    contradicts_count: int = 0
    misalignments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.supports_count + self.unrelated_count + self.contradicts_count


def _decide_layer_b_pass(
    *,
    supports: int,
    unrelated: int,
    contradicts: int,
    tolerance: float,
) -> bool:
    """Apply pass rules to verdict counts.

    Passes when zero ``contradicts`` AND
    ``unrelated / total <= tolerance``. Any ``contradicts`` verdict
    hard-fails regardless of tolerance.
    """
    if contradicts > 0:
        return False
    total = supports + unrelated + contradicts
    if total == 0:
        # Should not happen — caller short-circuits on zero citations —
        # but a vacuous-pass is the conservative interpretation.
        return True
    return (unrelated / total) <= tolerance


async def layer_b_check(
    body: str,
    particles: list[Particle],
    *,
    max_tokens: int = 1024,
    unrelated_tolerance: float | None = None,
) -> LayerBResult:
    """Run the per-sentence semantic-alignment judge.

    Returns a :class:`LayerBResult` carrying counts of each verdict
    bucket and the tolerance-applied pass decision:

      * ``passed=True``  — zero ``contradicts`` AND the ``unrelated``
        fraction is within the operator-configured tolerance
      * ``passed=False`` — at least one ``contradicts``, OR the
        ``unrelated`` fraction exceeds tolerance
      * ``passed=None``  — the judge couldn't be run (no citations, the
        judge response was unparseable, or the call itself errored).
        Callers treat None as "Layer B not applicable" and persist
        ``layer_b_passed=None`` in the frontmatter so the gap is visible
        to the operator.

    ``unrelated_tolerance`` defaults to
    ``config.wiki.layer_b_unrelated_tolerance`` (resolved at call time,
    so test overrides via env vars take effect).

    Skipping Layer B entirely (``config.wiki.layer_b_enabled=False``)
    is the *caller's* decision — this function always tries when
    invoked.
    """
    if unrelated_tolerance is None:
        from particles.config import get_config

        unrelated_tolerance = get_config().wiki.layer_b_unrelated_tolerance

    segments = extract_cited_segments(body)
    if not segments:
        return LayerBResult(passed=None)
    particle_by_short = {_short_id(p.id): p for p in particles}
    system, user = _build_layer_b_prompt(segments, particle_by_short)
    try:
        raw = await _call_synthesis_llm(
            prompt=user, system=system, max_tokens=max_tokens, temperature=0.0
        )
    except Exception as exc:
        log.warning("Layer B judge call failed: %s", exc)
        return LayerBResult(passed=None)
    verdicts = _parse_layer_b_verdicts(raw)
    if verdicts is None:
        return LayerBResult(passed=None)

    supports = unrelated = contradicts = 0
    misalignments: list[dict[str, Any]] = []
    for v in verdicts:
        bucket = _normalise_verdict(str(v.get("verdict")) if v.get("verdict") else None)
        if bucket == "supports":
            supports += 1
        elif bucket == "unrelated":
            unrelated += 1
            misalignments.append({**v, "verdict": "unrelated"})
        else:  # contradicts
            contradicts += 1
            misalignments.append({**v, "verdict": "contradicts"})

    passed = _decide_layer_b_pass(
        supports=supports,
        unrelated=unrelated,
        contradicts=contradicts,
        tolerance=unrelated_tolerance,
    )
    return LayerBResult(
        passed=passed,
        supports_count=supports,
        unrelated_count=unrelated,
        contradicts_count=contradicts,
        misalignments=misalignments,
    )
