"""Engine-side narrative-merge post-pass for chunked journal extraction.

When an over-length journal entry is extracted in multiple carry-forward passes
(`particles/extraction/journal.py`), each chunk emits its own NARRATIVE candidate
and its own ``narrative_index`` sequence restarting at 0. Left as-is, the entry
would fragment into N disconnected narratives — and the pipeline's NARRATIVE
edge-writer fires only when exactly one NARRATIVE candidate is present, so it
would write nothing.

:func:`collapse_chunk_narratives` is the bridge. It consumes the *accumulated
output of all passes* and arbitrates one canonical structure from it — choosing
(synthesizing) the single whole-entry label and re-deriving a global
``SEQUENCE_IN`` order no single pass could see. That is "reasoning over
accumulated state", so it is **Engine** and lives here in
``ingest/``, adjacent to the edge-writer it feeds. The Client extractor stays a
clean per-pass candidate producer.

The merge runs *before* embedding / §6.6 / the write loop, so the rest of the
pipeline sees output shaped exactly like single-pass journal output (one
NARRATIVE candidate + constituents in one global ``narrative_index`` sequence)
and runs unchanged. It is a **no-op for ≤ 1 NARRATIVE candidate** — single-pass
journals, every non-journal extractor, and the empty case — so behaviour is
byte-for-byte unchanged everywhere except the multi-chunk journal path.
"""

from __future__ import annotations

import logging

from particles.config import get_config
from particles.core.schema import ParticleType
from particles.extraction.general import CandidateParticle

log = logging.getLogger(__name__)

# Whole-entry label synthesis is one short sentence; a small cap suffices.
# Literal, mirroring the §6.6 contradiction probe's ``max_tokens=120`` in
# pipeline.py (small bounded LLM calls don't warrant a config knob).
_MERGE_LABEL_MAX_TOKENS = 256


async def collapse_chunk_narratives(
    candidates: list[CandidateParticle],
) -> tuple[list[CandidateParticle], list[str]]:
    """Collapse per-chunk NARRATIVE fragments into one whole-entry NARRATIVE.

    Returns ``(candidates, notes)``. For **≤ 1** NARRATIVE candidate the input
    list is returned unchanged (the no-op case). For **≥ 2** the NARRATIVE
    candidates are collapsed to one (the first), its ``content`` set to a
    synthesized whole-entry label (deterministic first-label fallback, §
    :func:`_merge_label`), the rest dropped, and every constituent's
    ``narrative_index`` reassigned to a global monotonic order following
    candidate-list position — which is whole-entry document order, because
    ``extract_with_carry_forward`` processes chunks in order and appends their
    candidates in order (3).
    """
    narr_positions = [
        i for i, c in enumerate(candidates) if c.particle_type == ParticleType.NARRATIVE
    ]
    if len(narr_positions) <= 1:
        return candidates, []

    notes: list[str] = []
    labels = [candidates[i].content for i in narr_positions]
    candidates[narr_positions[0]].content = await _merge_label(labels, notes)

    drop = set(narr_positions[1:])
    merged = [c for i, c in enumerate(candidates) if i not in drop]

    # Reassign a global, monotonic narrative_index over the constituents in list
    # (= document) order. The surviving NARRATIVE container is skipped; a
    # candidate that never carried an index (not a narrative constituent) is left
    # alone so the merge can't conscript stray candidates into the chain.
    global_index = 0
    for c in merged:
        if c.particle_type != ParticleType.NARRATIVE and c.narrative_index is not None:
            c.narrative_index = global_index
            global_index += 1

    notes.append(
        f"NARRATIVE_MERGE: collapsed {len(narr_positions)} per-chunk narrative "
        f"fragments into one whole-entry NARRATIVE over {global_index} constituents"
    )
    return merged, notes


async def _merge_label(labels: list[str], notes: list[str]) -> str:
    """Synthesize one whole-entry label from the ordered per-chunk labels.

    One LLM call. Falls back deterministically to the **first**
    chunk's label when synthesis is disabled
    (``journal_extractor.synthesize_merged_narrative = false``), the call raises,
    or it returns nothing usable — so the merge never fails extraction. Appends a
    quality note on every fallback path.
    """
    first = labels[0] if labels else ""
    if not get_config().journal_extractor.synthesize_merged_narrative:
        return first

    from particles.llm import complete

    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
    prompt = (
        "These one-sentence labels each describe a consecutive slice of ONE "
        "journal entry, in order. Write ONE sentence capturing the whole entry's "
        "overall arc or theme. Return only the sentence, with no preamble or "
        "quotation marks.\n\n" + numbered
    )
    try:
        raw = await complete("extraction", prompt, max_tokens=_MERGE_LABEL_MAX_TOKENS)
    except Exception as exc:
        log.warning("Narrative-label synthesis failed (%s); using first chunk's label", exc)
        notes.append("NARRATIVE_MERGE: label synthesis failed; used first chunk's label")
        return first

    merged = raw.strip().strip('"').strip()
    if not merged:
        notes.append("NARRATIVE_MERGE: label synthesis returned empty; used first chunk's label")
        return first
    return merged
