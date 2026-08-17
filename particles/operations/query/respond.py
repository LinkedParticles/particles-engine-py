"""Natural-language response generation for §9.3 Query.

Builds a per-audience prompt over the top-k particles and asks the shared
``particles.llm`` client for an answer. A failing LLM call **raises** — the
caller (``main.query``) catches it, renders the deterministic
:func:`fallback_listing` instead, and *discloses* the failure on the response
(``QueryResponse.answer_generation_error``). The old silent
concatenate-and-pretend fallback is deliberately gone: a billing or network
failure must never masquerade as an answer (the honesty posture —
"not probed this run", never a quiet degradation).
"""

from __future__ import annotations

import logging
from datetime import datetime

from particles.core.schema import AudienceHint, Particle, ParticleType

log = logging.getLogger(__name__)

# F9 hardening: the trusted instructions live in the ``system`` turn; the
# untrusted question and particle list (particle content is LLM-extracted from
# untrusted sources) go in the user turn behind per-call nonce fences, so a
# crafted question or a poisoned particle cannot override the answering rules.
_RESPONSE_SYSTEM_TEMPLATE = """\
You are answering a question using a curated knowledge base. The question and the
most relevant knowledge particles (individual verified claims, each with a
confidence score) are provided in the user message, wrapped in data fences.

Instructions:
- Answer the question using only the provided particles
- {audience_instructions}
- Render each particle in the voice of its modality marker: <EXPERIENTIAL> as
  the author's lived experience, <EVALUATIVE> as the author's opinion,
  <FALSIFIABLE> as fact. Never present a feeling or opinion as objective fact,
  and never truth-hedge a non-FALSIFIABLE particle on its confidence.
- A "Narrative —" line groups a memory's claims (the indented bullets beneath
  it, in order); answer from those claims, not the one-line narrative label.
- If particles conflict, acknowledge the uncertainty
- If particles are insufficient, say so clearly
- If the provided particles do not bear on the question, your reply must
  begin with the exact line NO_RELEVANT_KNOWLEDGE, followed by a brief
  statement that the knowledge base holds nothing relevant to this
  question — do not construct an answer from unrelated particles
- If a coverage note is present, include a caveat that your answer may be incomplete"""

# refusal protocol: the marker line the responder is instructed to
# lead with when the particles do not bear on the question. The caller strips
# it and records the machine-readable ``QueryResponse.answer_refused`` flag so
# consumers can relabel the hit list and drop k-widening advice. Worst case a
# crafted question coaxes the marker out: the only effect is a self-inflicted
# display change on the requester's own result.
REFUSAL_MARKER = "NO_RELEVANT_KNOWLEDGE"


def strip_refusal_marker(answer: str) -> tuple[str, bool]:
    """Split the §4 refusal marker off a generated answer.

    Returns ``(clean_answer, refused)``. The marker is honoured only at the
    very start of the reply (as instructed); a mention elsewhere is prose.
    """
    stripped = answer.lstrip()
    if not stripped.startswith(REFUSAL_MARKER):
        return answer, False
    rest = stripped[len(REFUSAL_MARKER) :].lstrip(" :\n")
    if not rest:
        rest = "The knowledge base holds nothing relevant to this question."
    return rest, True


_AUDIENCE_INSTRUCTIONS: dict[AudienceHint, str] = {
    AudienceHint.EXPERT: (
        "Include numeric confidence values and uncertainty classifications "
        "(ALEATORY/EPISTEMIC) in your response"
    ),
    AudienceHint.GENERAL: (
        "Use natural language hedging (e.g. 'likely', 'may', 'probably') "
        "to convey confidence — do not include raw numbers"
    ),
    AudienceHint.REGULATORY: (
        "Cite the source provenance (corpus_entry_id and snapshot_id) for every claim. "
        "Include full confidence metadata"
    ),
}


def _particle_line(p: Particle, ec: float, audience: AudienceHint, *, indent: bool = False) -> str:
    """One particle bullet: content + modality marker + audience-conditional tags.

    The ``<MODALITY>`` marker is always present — it drives the
    responder's rendering voice — while the nature / confidence / provenance tags
    stay audience-conditional. ``indent`` nests the line under a narrative header.
    """
    nature_tag = f"[{p.uncertainty_nature}]" if audience == AudienceHint.EXPERT else ""
    conf_tag = (
        f"(confidence: {ec:.2f})"
        if audience in (AudienceHint.EXPERT, AudienceHint.REGULATORY)
        else ""
    )
    prov_tag = ""
    if audience == AudienceHint.REGULATORY and p.provenance:
        pref = p.provenance[0]
        prov_tag = f" [source: {pref.corpus_entry_id}/{pref.snapshot_id}]"
    body = f"{p.content} <{p.assertion_modality.value}>{nature_tag}{conf_tag}{prov_tag}".strip()
    return ("    - " if indent else "- ") + body


async def _generate_response(
    question: str,
    particles: list[Particle],
    eff_confs: list[float],
    audience: AudienceHint,
    confidence_note: str,
    coverage_note: str = "",
    *,
    narrative_constituents: dict[str, list[Particle]] | None = None,
    as_of: datetime | None = None,
) -> str:
    if not particles:
        if as_of is not None:
            # a T before the store's first assertion (or with no
            # matching beliefs) is a legitimate question with an honest empty
            # answer, not an error.
            return f"The store held no beliefs matching this question as of {as_of.isoformat()}."
        return "No relevant particles found for this question in the current knowledge base."

    narrative_constituents = narrative_constituents or {}
    particle_lines: list[str] = []
    for p, ec in zip(particles, eff_confs, strict=False):
        constituents = (
            narrative_constituents.get(p.id) if p.particle_type == ParticleType.NARRATIVE else None
        )
        if constituents:
            # expand a narrative hit — its label as a header, then its
            # constituents in SEQUENCE_IN order, so the answer draws on the
            # memory's actual claims rather than the one-line label.
            particle_lines.append(f"- Narrative — {p.content}:")
            particle_lines.extend(
                _particle_line(c, c.confidence.value, audience, indent=True) for c in constituents
            )
        else:
            particle_lines.append(_particle_line(p, ec, audience))

    particle_list = "\n".join(particle_lines)
    audience_instructions = _AUDIENCE_INSTRUCTIONS[audience]

    from particles.config import get_config
    from particles.llm import complete, data_fence_instruction, fence, make_nonce

    # Trusted instructions → system; untrusted question + particle list →
    # user, each behind the same per-call nonce fence (F9).
    nonce = make_nonce()
    system_body = _RESPONSE_SYSTEM_TEMPLATE.format(audience_instructions=audience_instructions)
    if as_of is not None:
        # tell the responder the reference instant so the
        # prose answers in the right tense ("As of 2000-01-01 the store
        # believed …"). The as-of instruction is trusted (server-built),
        # so it belongs in the system turn, not the fenced user data.
        system_body += (
            f"\n- This is an as-of query: answer what was believed as of "
            f"{as_of.isoformat()} (the reference instant), in the "
            f"appropriate tense. The particles provided are exactly the "
            f"beliefs that held at that instant; some may have been "
            f"superseded or retracted since."
        )
    system = system_body + "\n\n" + data_fence_instruction(nonce)
    user = (
        f"Question:\n{fence(question, nonce, label='question')}\n\n"
        f"Relevant particles ({audience.value} audience):\n"
        f"{fence(particle_list, nonce, label='particles')}\n\n"
        f"{confidence_note}\n{coverage_note}"
    )

    max_tokens = get_config().extraction.query_max_tokens
    return await complete("query_response", user, max_tokens=max_tokens, system=system)


def fallback_listing(particles: list[Particle]) -> str:
    """Deterministic no-LLM rendering of the retrieved beliefs, one bullet each.

    Used by the caller when answer generation fails — always paired with the
    ``answer_generation_error`` disclosure, never served as if it were prose.
    """
    return "\n".join(f"• {p.content}" for p in particles)


def generation_error_reason(exc: Exception) -> str:
    """A short human-readable reason for a failed answer generation.

    The provider's message (e.g. Anthropic's "credit balance is too low")
    is the actionable part — keep it, bounded so a pathological exception
    cannot balloon the response envelope.
    """
    reason = str(exc).strip() or type(exc).__name__
    return reason[:300]
