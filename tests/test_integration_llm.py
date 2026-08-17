"""Live Anthropic round-trip tests — the opt-in integration tier (P3-7).

The unit suite mocks every LLM call at the ``particles.llm.set_client``
seam, so prompt drift that breaks a response *contract* — the extraction
JSON array, the Layer-B verdict JSON, the YES/NO contradiction protocol —
is invisible to it. Each test here makes one or two cheap live calls and
asserts only the contract: required fields populated, JSON parseable,
protocol token honoured. Never specific model wording, and never counts
beyond the minimum — model-behaviour assertions would make the tier flaky.

Run with ``uv run pytest tests/`` (no ``-m`` filter) on a developer key;
CI excludes the tier with ``-m "not integration"``. Without
``ANTHROPIC_API_KEY`` the module skips wholesale (``pytestmark`` below)
rather than erroring. Inputs are a few sentences each — these run on a
developer's key, so cost discipline matters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.status import Status
from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]


_SOURCE_TEXT = (
    "The Eiffel Tower is located in Paris, France. "
    "It was completed in 1889. "
    "The tower is approximately 330 metres tall."
)


async def test_live_extraction_round_trip(db_session: AsyncSession, tmp_path: Path) -> None:
    """Deposit a tiny text source and run real extraction end to end.

    Pins the extraction output contract — at least one particle, required
    fields populated, provenance pointing at the deposited snapshot, ACTIVE
    status — not model behaviour (no claim-text or exact-count assertions).

    The embedding model is mocked: it is local deterministic compute already
    covered by the unit suite, and mocking keeps this test pinned to the
    Anthropic round trip alone (no sentence-transformers download). The
    Subject Authority lookup stays live; it degrades to a bare local subject
    on network failure, which none of the assertions depend on.
    """
    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_file
    from particles.ingest.pipeline import extract_snapshot

    doc = tmp_path / "eiffel.txt"
    doc.write_text(_SOURCE_TEXT)

    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda texts, **_kw: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
    )
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        entry_id, snapshot_id = await deposit_file(db_session, doc, deposited_by="integration-test")
        await db_session.commit()

        particles = await extract_snapshot(db_session, entry_id, snapshot_id)
        await db_session.commit()
    finally:
        ep.set_embedding_model(original_model)

    assert len(particles) >= 1
    for p in particles:
        assert p.content.strip(), f"particle {p.id} has empty content"
        assert 0.0 <= p.confidence.value <= 1.0
        assert p.uncertainty_nature in (
            UncertaintyNature.EPISTEMIC,
            UncertaintyNature.ALEATORY,
        )
        # Fresh store, no conflicts possible — everything lands ACTIVE.
        assert p.status is Status.ACTIVE
        entry_refs = [r for r in p.provenance if r.corpus_entry_id == entry_id]
        assert entry_refs, f"particle {p.id} has no provenance ref at the deposited entry"
        assert any(r.snapshot_id == snapshot_id for r in entry_refs)


def _judge_particle(content: str) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=0.9),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="integration-test",
    )


async def test_live_layer_b_judge_round_trip() -> None:
    """Layer-B citation judge on one supported and one unsupported citation.

    Contract: the verdict JSON parses (``passed is not None``), one verdict
    per pair, and the deliberately unsupported pair is not classified
    ``supports``. Conservative on the unsupported side — either ``unrelated``
    or ``contradicts`` is acceptable; both land in ``misalignments``.
    """
    from particles.exporters.article_synthesis import layer_b_check

    supported = _judge_particle("The Eiffel Tower was completed in 1889.")
    unsupported = _judge_particle("The Great Wall of China is more than 13,000 miles long.")
    body = (
        f"The Eiffel Tower was completed in 1889 [^p-{supported.id[:8]}]. "
        f"The Eiffel Tower stands on the Champ de Mars in Paris "
        f"[^p-{unsupported.id[:8]}]."
    )

    result = await layer_b_check(body, [supported, unsupported])

    # passed=None means the judge call failed or its JSON was unparseable —
    # exactly the contract break this test exists to surface.
    assert result.passed is not None, "Layer-B judge response did not parse"
    assert result.total == 2, "judge must return one verdict object per (claim, particle) pair"
    # Pair ids follow citation order in the body, so the unsupported pair is
    # id 1. A "supports" verdict never enters misalignments.
    assert any(m.get("id") == 1 for m in result.misalignments), (
        f"the deliberately unsupported citation was classified 'supports': {result.misalignments!r}"
    )


async def test_live_contradiction_gate_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    """§6.6 contradiction-confirmation seam on one contradictory, one compatible pair.

    The seam swallows API errors into ``False``, which would make a False on
    the compatible pair indistinguishable from a failed call — so also assert
    the failure warning never fired. The contradictory pair is deliberately
    extreme: its True return pins the YES-protocol parse, not model judgment
    subtlety.
    """
    from particles.ingest.pipeline import _llm_confirms_contradiction

    with caplog.at_level(logging.WARNING, logger="particles.ingest.pipeline"):
        contradictory = await _llm_confirms_contradiction(
            "The Eiffel Tower is approximately 330 metres tall.",
            "The Eiffel Tower is approximately 95 metres tall.",
        )
        compatible = await _llm_confirms_contradiction(
            "The Eiffel Tower is located in Paris.",
            "The Eiffel Tower was completed in 1889.",
        )

    failures = [r for r in caplog.records if "Contradiction confirmation" in r.getMessage()]
    assert not failures, f"LLM contradiction call failed instead of answering: {failures!r}"
    assert contradictory is True, "clear-cut contradiction was not confirmed (YES path broken?)"
    assert compatible is False, "compatible pair was confirmed as contradictory"


# a / M4 — held-out stance-emission precision gate.
#
# A small labeled set: each case is a short source that either *should* yield a
# stance (the source explicitly positions itself toward a co-extracted claim) or
# *should not* (plain assertions, a question, a hedge). The test runs real
# extraction over each, measures stance-emission precision and the
# false-positive-stance rate, LOGS both (the M4 "report"), and asserts a
# conservative floor — loose enough to not flake on model variance, tight enough
# to catch a broken §5a rule. The numeric floor is the operator's activation gate.
_STANCE_HELD_OUT: list[tuple[str, str, bool]] = [
    # (source text, author id, expects a stance)
    (
        "Smith (2019) reported that the 1948 1-Pfennig coin was made of aluminium. "
        "I disagree that the 1948 1-Pfennig was aluminium — it was bronze.",
        "blog:numista_fan",
        True,
    ),
    (
        "An earlier note gives the mint date as 1948. The authors concur with that 1948 mint date.",
        "blog:historian",
        True,
    ),
    (
        "@bob stated that the Eiffel Tower is 300 metres tall. "
        "@bob is wrong about the tower height; it is about 330 metres.",
        "forum:alice",
        True,
    ),
    (
        "The Eiffel Tower is located in Paris. It was completed in 1889. "
        "The tower is approximately 330 metres tall.",
        "blog:facts",
        False,
    ),
    (
        "Was the 1948 coin made of aluminium? The historical record is unclear. "
        "The material might have been bronze.",
        "forum:curious",
        False,
    ),
    (
        "The coin was struck in Berlin. It weighed two grams. Its diameter was 17 millimetres.",
        "blog:catalogue",
        False,
    ),
]


async def test_live_stance_emission_precision(db_session: AsyncSession, tmp_path: Path) -> None:
    """M4: measure stance-emission precision + false-positive rate on a held-out set."""
    import sqlalchemy as sa

    from particles import embeddings as ep
    from particles.core.stance import stance_holder
    from particles.corpus.deposit import deposit_file
    from particles.corpus.store import SnapshotRow
    from particles.ingest.pipeline import extract_snapshot

    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda texts, **_kw: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
    )
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)

    true_pos = false_pos = expected = 0
    try:
        for idx, (text, author, expects_stance) in enumerate(_STANCE_HELD_OUT):
            doc = tmp_path / f"case_{idx}.txt"
            doc.write_text(text)
            entry_id, snapshot_id = await deposit_file(db_session, doc, deposited_by="it")
            await db_session.execute(
                sa.update(SnapshotRow)
                .where(SnapshotRow.snapshot_id == snapshot_id)
                .values(author_id=author)
            )
            await db_session.commit()
            written = await extract_snapshot(db_session, entry_id, snapshot_id)
            await db_session.commit()

            emitted = any(stance_holder(p) is not None for p in written)
            if expects_stance:
                expected += 1
                if emitted:
                    true_pos += 1
            elif emitted:
                false_pos += 1
    finally:
        ep.set_embedding_model(original_model)

    emitted_total = true_pos + false_pos
    precision = true_pos / emitted_total if emitted_total else 1.0
    recall = true_pos / expected if expected else 1.0
    negatives = len(_STANCE_HELD_OUT) - expected
    fp_rate = false_pos / negatives if negatives else 0.0
    logging.getLogger(__name__).warning(
        "stance held-out: precision=%.2f recall=%.2f fp_rate=%.2f "
        "(tp=%d fp=%d expected=%d)",
        precision,
        recall,
        fp_rate,
        true_pos,
        false_pos,
        expected,
    )
    # Conservative activation floor (the report above is the measured value).
    assert precision >= 0.6, f"stance-emission precision {precision:.2f} below floor 0.6"
    assert fp_rate <= 0.5, f"false-positive-stance rate {fp_rate:.2f} above ceiling 0.5"


async def test_live_rule_source_prescriptions_reach_the_default_surface(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """a registered rule file's prescriptions must be visible, whatever
    the classifier calls them.

    The assertion is deliberately **label-invariant**. The defect had the
    classifier routing a rules document's prescriptions to ``DOCUMENT_META``;
    the live store shows it doing so for a minority of them, and two captures
    on 2026-07-25 disagreed with each other on the same genre. So asserting
    ``scope == WORLD`` (or ``== DOCUMENT_META``) would be asserting a coin
    flip. What must hold either way is the outcome: after extraction from a
    ``rule-file``-tagged entry, no prescription is hidden from the default
    surface.

    The observed labels are logged, not asserted. A shift in that line is the
    evidence the prompt-side genre fix is waiting for, and it should
    be *visible* without turning this tier flaky. The verbatim-recording pins
    live in tests/test_scope_exemption.py; this one guards their premise
    against a model or prompt change that the recordings cannot see.
    """
    from particles import embeddings as ep
    from particles.corpus.rule_sources import sync_rule_sources
    from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry
    from particles.extraction.scope import SCOPE_KEY, is_excluded_document_meta
    from particles.ingest.pipeline import extract_snapshot

    doc = tmp_path / "AGENTS.md"
    doc.write_text(
        "# AGENTS.md\n\n"
        "Loaded by agents when working in this repository.\n\n"
        "## Committing\n\n"
        "Commits must be made with `uv run git commit -s`; a bare `git commit` "
        "fails because the pre-commit hook is not on PATH.\n\n"
        "`--no-gpg-sign` must never be passed.\n"
    )

    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda texts, **_kw: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
    )
    original_model = ep._embedding_model
    ep.set_embedding_model(mock_model)
    try:
        await sync_rule_sources(db_session, [str(doc)])
        await db_session.commit()
        entry = await get_entry_by_uri(db_session, doc.resolve().as_uri())
        assert entry is not None
        snapshot = (await list_snapshots_for_entry(db_session, entry.entry_id))[0]
        written = await extract_snapshot(db_session, entry.entry_id, snapshot.snapshot_id)
        await db_session.commit()
    finally:
        ep.set_embedding_model(original_model)

    logging.getLogger(__name__).warning(
        "live scope labels: %s",
        [
            f"{(p.properties or {}).get(SCOPE_KEY, 'WORLD')}/{p.assertion_modality.value}"
            for p in written
        ],
    )

    hidden = [p.content for p in written if is_excluded_document_meta(p.properties)]
    assert not hidden, f"a registered rule source still has hidden claims: {hidden}"
    visible = [p.content for p in written if p.status is Status.ACTIVE]
    assert any("--no-gpg-sign" in c for c in visible), (
        "the prohibition did not reach the default surface"
    )
