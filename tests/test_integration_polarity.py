"""Live claim-polarity acceptance check — integration tier (cap. 1).

Runs the **real** general extractor over a compact ADR-genre snippet that mirrors
the Rejected-Alternatives / Deferred / counterfactual structure of the project's
own ADRs (0061 / 0065 / 0072 / 0119 — the corpus the R1.6 trust checkpoint
flagged). Pins the *contract*: the document's rejected / deferred / counterfactual
prose lands non-asserted (``is_non_asserted`` True) and the real decision lands
asserted, so the default factual surface (the predicate the query / export
consumers share) excludes the former. Never pins exact counts or model wording
(tests/AGENTS.md § Integration tests).

Run with ``uv run pytest tests/`` (no ``-m`` filter) on a developer key; CI
excludes the tier with ``-m "not integration"``. Without ANTHROPIC_API_KEY the
module skips wholesale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.extraction.polarity import is_non_asserted
from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]

# A tiny ADR-genre document with one of each polarity, framed exactly as the
# named ADRs frame theirs (a Decision, a Rejected-Alternatives entry, a
# motivational counterfactual, a Deferred/out-of-scope item).
_ADR_SNIPPET = """\
# Structure-aware extraction

## Decision

The SDK adds a claim-polarity axis stored on properties["extraction:polarity"].

## Alternatives considered

Reusing assertion_modality for polarity was rejected: a rejected design is
perfectly falsifiable, so folding polarity into modality would conflate two
orthogonal axes.

## Context

Without a polarity signal, a README projection that queries the store will
return a contradictory, partly-retired answer with no way to prefer current
truth.

## Deferred

A first-class Core assertion_polarity field is out of scope for this increment;
v1 keeps polarity on the properties dict.
"""


@pytest.mark.asyncio
async def test_real_extractor_tags_rejected_and_counterfactual_non_asserted(
    db_session: object, tmp_path: Path
) -> None:
    from particles.corpus.deposit import deposit_file
    from particles.ingest.pipeline import extract_snapshot

    doc = tmp_path / "adr-0145.md"
    doc.write_text(_ADR_SNIPPET, encoding="utf-8")

    session = db_session  # type: ignore[assignment]
    entry_id, snapshot_id = await deposit_file(session, doc, deposited_by="acceptance")  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    particles = await extract_snapshot(session, entry_id, snapshot_id)  # type: ignore[arg-type]
    await session.commit()  # type: ignore[union-attr]

    assert particles, "extractor produced no particles"

    asserted = [p for p in particles if not is_non_asserted(p.properties)]
    non_asserted = [p for p in particles if is_non_asserted(p.properties)]

    # The rejected / deferred / counterfactual prose must produce at least one
    # non-asserted particle — the whole point of capability 1.
    assert non_asserted, (
        "expected the rejected / deferred / counterfactual prose to land "
        f"non-asserted; got polarities {[p.properties for p in particles]}"
    )
    # The real Decision must remain on the factual surface.
    assert asserted, "expected the Decision to land asserted (on the factual surface)"

    # The default factual surface (the predicate query / export share) excludes
    # exactly the non-asserted particles.
    surface = [p for p in particles if not is_non_asserted(p.properties)]
    assert len(surface) == len(asserted)
    assert all(not is_non_asserted(p.properties) for p in surface)
