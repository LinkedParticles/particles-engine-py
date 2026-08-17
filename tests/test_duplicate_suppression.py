"""Tests for extract-time exact-duplicate suppression.

The leak this closes, measured on the dogfood store 2026-07-25: 3,411 of 21,650
particles minted over eight days were verbatim copies of claims the store
already held (15.8 %) — one fresh particle per re-extraction of re-deposited
harvest material, because ``extract_snapshot`` scopes §6.6 to a single
corpus entry and an exact duplicate that *does* reach the ladder returns
``CORROBORATES`` and is written anyway.

Three properties carry the design and are asserted directly:

* **exactness** — the predicate is content identity, never similarity. The false positives that a cosine tier would have merged (0.9951 for
  ``claude-opus-4-6`` / ``4-5``) must survive as distinct particles.
* **source-faithfulness** — a suppressed candidate's provenance lands on the
  surviving particle, appended after the decay anchor, idempotently.
* **no claim ever vanishes** — suppression must never fold a candidate into a
  particle that is about to be superseded (the reindex hazard).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from particles.core.duplicate_key import content_hash, duplicate_key, normalize_content
from particles.core.schema import (
    AssertionModality,
    Confidence,
    CorpusEntry,
    FetchPolicy,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.stance import STANCE_HOLDER_KEY
from particles.core.status import Status
from particles.corpus.store import CorpusEntryRow
from particles.extraction.polarity import POLARITY_DECLINED, POLARITY_KEY
from particles.extraction.scope import SCOPE_DOCUMENT_META, SCOPE_KEY
from particles.ingest.duplicate_suppression import (
    DuplicateIndex,
    is_suppression_eligible,
    particle_key,
)
from particles.store.particle_store import (
    append_provenance_ref,
    get_active_particles_by_content_hashes,
    get_particle,
    insert_particle,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _particle(
    content: str,
    *,
    subject_ids: list[str] | None = None,
    holder: str | None = None,
    modality: AssertionModality = AssertionModality.FALSIFIABLE,
    properties: dict[str, Any] | None = None,
    status: Status = Status.ACTIVE,
    entry_id: str = "entry-1",
    snapshot_id: str = "snap-1",
) -> Particle:
    props = dict(properties or {})
    if holder is not None:
        props[STANCE_HOLDER_KEY] = holder
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        status=status,
        assertion_modality=modality,
        subject_ids=subject_ids or [],
        properties=props or None,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        ],
    )


# ---------------------------------------------------------------------------
# The predicate — exactness is the whole safety argument
# ---------------------------------------------------------------------------


def test_normalization_collapses_only_whitespace_and_trailing_marks() -> None:
    """Whitespace runs and sentence-final punctuation are noise; wording is not."""
    assert normalize_content("The  live store is  named `particles.db`.") == normalize_content(
        "The live store is named `particles.db`"
    )
    # Case is preserved — this is exact-content identity, not fuzzy matching.
    assert normalize_content("The Live Store") != normalize_content("the live store")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # The measurement's worst false positive: cosine 0.9951, one
        # token apart. Any similarity-based tier would have merged these.
        ("Model `claude-opus-4-6` is current.", "Model `claude-opus-4-5` is current."),
        # An outright contradiction that cosine scored 0.9272.
        ("There are four commits on branch X.", "There are three commits on branch X."),
        # other measured near-misses.
        ("The OTLP port is 4318.", "The OTLP port is 4317."),
        ("Coverage comes from `audit.py`.", "Coverage comes from `test_audit.py`."),
    ],
)
def test_one_token_differences_are_never_the_same_claim(left: str, right: str) -> None:
    """Exactness: a load-bearing token difference must survive as two particles."""
    assert duplicate_key(left, [], None) != duplicate_key(right, [], None)
    assert content_hash(left) != content_hash(right)


def test_subject_set_separates_otherwise_identical_claims() -> None:
    """Same sentence about different subjects is not one claim (Tier A)."""
    index = DuplicateIndex([_particle("It ships on Tuesday.", subject_ids=["subj-a"])])
    assert index.find(_particle("It ships on Tuesday.", subject_ids=["subj-b"])) is None
    assert index.find(_particle("It ships on Tuesday.", subject_ids=["subj-a"])) is not None


def test_subjectless_duplicates_are_reachable() -> None:
    """The empty subject set matches the empty set — the mop's blind spot."""
    index = DuplicateIndex([_particle("An orphan claim.")])
    assert index.find(_particle("An orphan claim.")) is not None


def test_stance_holder_separates_identical_text() -> None:
    """Identical text held by different principals is not one claim."""
    index = DuplicateIndex([_particle("Rust is the better choice.", holder="alice")])
    assert index.find(_particle("Rust is the better choice.", holder="bob")) is None
    assert index.find(_particle("Rust is the better choice.", holder="alice")) is not None


def test_non_truth_apt_and_document_meta_are_ineligible() -> None:
    """Suppression inherits the modality / scope / polarity gates
    ."""
    evaluative = _particle("The API feels clunky.", modality=AssertionModality.EVALUATIVE)
    doc_meta = _particle(
        "Section 10.4 defines the exporter.", properties={SCOPE_KEY: SCOPE_DOCUMENT_META}
    )
    declined = _particle("The rejected alternative.", properties={POLARITY_KEY: POLARITY_DECLINED})

    assert not is_suppression_eligible(evaluative)
    assert not is_suppression_eligible(doc_meta)
    assert not is_suppression_eligible(declined)

    # Ineligible on both sides: never indexed, never matched.
    assert DuplicateIndex([evaluative]).find(evaluative) is None
    assert DuplicateIndex([doc_meta]).find(doc_meta) is None


def test_index_is_first_writer_wins() -> None:
    """Determinism: the earliest registration is the suppression target."""
    first = _particle("A repeated claim.")
    second = _particle("A repeated claim.")
    index = DuplicateIndex([first, second])
    assert index.find(_particle("A repeated claim.")) is not None
    assert index.find(_particle("A repeated claim.")).id == first.id  # type: ignore[union-attr]


def test_excluded_ids_are_never_suppression_targets() -> None:
    """The reindex hazard: folding into a to-be-superseded particle loses the claim."""
    doomed = _particle("A claim about to be superseded.")
    index = DuplicateIndex([doomed], exclude_ids=frozenset({doomed.id}))
    assert index.find(_particle("A claim about to be superseded.")) is None


def test_particle_key_matches_duplicate_key() -> None:
    """The stored-particle key and the pure key agree."""
    p = _particle("A claim.", subject_ids=["s1"], holder="alice")
    assert particle_key(p) == duplicate_key("A claim.", ["s1"], "alice")


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_hash_lookup_returns_only_active_matches(db_session: Any) -> None:
    active = _particle("A findable claim.")
    retired = _particle("A findable claim.", status=Status.SUPERSEDED)
    other = _particle("An unrelated claim.")
    for p in (active, retired, other):
        await insert_particle(db_session, p)
    await db_session.flush()

    found = await get_active_particles_by_content_hashes(
        db_session, [content_hash("A findable claim.")]
    )
    assert [p.id for p in found] == [active.id]


@pytest.mark.asyncio
async def test_lookup_matches_across_normalization(db_session: Any) -> None:
    """The stored hash is over normalized content, so spacing variants collide."""
    stored = _particle("The  live store is named `particles.db`.")
    await insert_particle(db_session, stored)
    await db_session.flush()

    found = await get_active_particles_by_content_hashes(
        db_session, [content_hash("The live store is named `particles.db`")]
    )
    assert [p.id for p in found] == [stored.id]


@pytest.mark.asyncio
async def test_append_provenance_is_idempotent_and_keeps_decay_anchor(db_session: Any) -> None:
    """Append-only, deduped, and never disturbs provenance[0] (the anchor)."""
    p = _particle("A corroborated claim.", entry_id="entry-1", snapshot_id="snap-1")
    await insert_particle(db_session, p)
    await db_session.flush()

    newer = ProvenanceRef(
        type=ProvenanceRefType.SOURCE,
        corpus_entry_id="entry-2",
        snapshot_id="snap-2",
    )
    assert await append_provenance_ref(db_session, p.id, newer) is True
    # Re-offering the same snapshot is a no-op — catch-up sweep
    # re-offers unchanged snapshots routinely.
    assert await append_provenance_ref(db_session, p.id, newer) is False

    reloaded = await get_particle(db_session, p.id)
    assert reloaded is not None
    assert len(reloaded.provenance) == 2
    # Earliest ref stays first: re-observation must not refresh the claim's age.
    assert reloaded.provenance[0].snapshot_id == "snap-1"
    assert reloaded.provenance[1].snapshot_id == "snap-2"


@pytest.mark.asyncio
async def test_append_provenance_leaves_confidence_and_identity_untouched(
    db_session: Any,
) -> None:
    """confidence.value is immutable; asserted_* record first mint."""
    p = _particle("An immutable claim.")
    await insert_particle(db_session, p)
    await db_session.flush()

    await append_provenance_ref(
        db_session,
        p.id,
        ProvenanceRef(
            type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-9", snapshot_id="snap-9"
        ),
    )
    reloaded = await get_particle(db_session, p.id)
    assert reloaded is not None
    assert reloaded.confidence.value == p.confidence.value
    # Compared tz-naive: in-memory SQLite round-trips DATETIME without the
    # tzinfo, which is orthogonal to the property under test (the instant did
    # not move).
    assert reloaded.asserted_at.replace(tzinfo=None) == p.asserted_at.replace(tzinfo=None)
    assert reloaded.asserted_by == p.asserted_by
    assert reloaded.status is Status.ACTIVE


@pytest.mark.asyncio
async def test_append_provenance_second_snapshot_of_same_entry_adds_no_edge(
    db_session: Any,
) -> None:
    """The edge index is keyed (particle, entry) — a second snapshot needs no row."""
    from sqlalchemy import select

    from particles.store.particle_store import ProvenanceEdgeRow

    p = _particle("A re-observed claim.", entry_id="entry-1", snapshot_id="snap-1")
    await insert_particle(db_session, p)
    await db_session.flush()

    await append_provenance_ref(
        db_session,
        p.id,
        ProvenanceRef(
            type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-2"
        ),
    )
    rows = (
        (
            await db_session.execute(
                select(ProvenanceEdgeRow).where(ProvenanceEdgeRow.particle_id == p.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    # Deliberately still the original snapshot: the cascade is told to
    # skip this particle via exclude_ids rather than by re-pointing the index.
    assert rows[0].snapshot_id == "snap-1"

    # A genuinely new entry does get its own edge.
    await append_provenance_ref(
        db_session,
        p.id,
        ProvenanceRef(
            type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-2", snapshot_id="snap-3"
        ),
    )
    rows = (
        (
            await db_session.execute(
                select(ProvenanceEdgeRow).where(ProvenanceEdgeRow.particle_id == p.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# End-to-end through the extraction pipeline
# ---------------------------------------------------------------------------


def _mock_llm(contents: list[str]) -> MagicMock:
    import anthropic

    payload = MagicMock()
    payload.text = json.dumps(
        [
            {
                "content": c,
                "confidence_value": 0.9,
                "uncertainty_nature": "EPISTEMIC",
            }
            for c in contents
        ]
    )
    resp = MagicMock()
    resp.content = [payload]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(return_value=resp)
    return client


async def _deposit(session: Any, tmp_path: Path, name: str, text: str) -> tuple[str, str]:
    from particles.corpus.deposit import deposit_file

    doc = tmp_path / name
    doc.write_text(text)
    entry_id, snapshot_id = await deposit_file(session, doc, deposited_by="test")
    await session.commit()
    return entry_id, snapshot_id


@pytest.mark.asyncio
async def test_second_entry_with_same_claim_is_suppressed(db_session: Any, tmp_path: Path) -> None:
    """The measured leak: the same claim harvested from two entries mints once."""
    from particles import embeddings as ep
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    claim = "The live store is named `particles.db`."
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm([claim]))
    try:
        e1, s1 = await _deposit(db_session, tmp_path, "a.md", "first source")
        first = await extract_snapshot(db_session, e1, s1)
        await db_session.commit()
        assert len(first) == 1

        e2, s2 = await _deposit(db_session, tmp_path, "b.md", "second source")
        suppressed: list[str] = []
        second = await extract_snapshot(db_session, e2, s2, suppressed_ids_out=suppressed)
        await db_session.commit()

        # Nothing minted; the existing particle absorbed the observation.
        assert second == []
        assert suppressed == [first[0].id]

        survivor = await get_particle(db_session, first[0].id)
        assert survivor is not None
        assert survivor.status is Status.ACTIVE
        # Source-faithfulness: the second source's evidence landed.
        entries = {ref.corpus_entry_id for ref in survivor.provenance}
        assert entries == {e1, e2}
        # Decay anchor unmoved.
        assert survivor.provenance[0].corpus_entry_id == e1
    finally:
        set_client(None)
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_suppression_disabled_restores_pre_0211_behaviour(
    db_session: Any, tmp_path: Path
) -> None:
    """The flag is a true off-switch: two entries, two particles, as before."""
    from particles import embeddings as ep
    from particles.config import get_config
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    claim = "A claim that will be duplicated."
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm([claim]))
    get_config().extraction.duplicate_suppression.enabled = False
    try:
        e1, s1 = await _deposit(db_session, tmp_path, "a.md", "first source")
        first = await extract_snapshot(db_session, e1, s1)
        await db_session.commit()

        e2, s2 = await _deposit(db_session, tmp_path, "b.md", "second source")
        suppressed: list[str] = []
        second = await extract_snapshot(db_session, e2, s2, suppressed_ids_out=suppressed)
        await db_session.commit()

        assert len(first) == 1
        assert len(second) == 1
        assert second[0].id != first[0].id
        assert suppressed == []
    finally:
        set_client(None)
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_distinct_claims_are_untouched(db_session: Any, tmp_path: Path) -> None:
    """No-op guarantee: nothing suppresses when nothing duplicates."""
    from particles import embeddings as ep
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm(["Model `claude-opus-4-6` is current."]))
    try:
        e1, s1 = await _deposit(db_session, tmp_path, "a.md", "first")
        first = await extract_snapshot(db_session, e1, s1)
        await db_session.commit()

        # One token apart — the 0.9951 false positive.
        set_client(_mock_llm(["Model `claude-opus-4-5` is current."]))
        e2, s2 = await _deposit(db_session, tmp_path, "b.md", "second")
        suppressed: list[str] = []
        second = await extract_snapshot(db_session, e2, s2, suppressed_ids_out=suppressed)
        await db_session.commit()

        assert len(first) == 1
        assert len(second) == 1
        assert suppressed == []
    finally:
        set_client(None)
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_reindex_does_not_lose_a_claim_to_suppression(
    db_session: Any, tmp_path: Path
) -> None:
    """The hazard: re-extracting an entry must never fold into its own doomed copy.

    A reindex passes the particles it is about to supersede as ``supersede_ids``.
    If the rung suppressed the fresh candidate into one of those, the claim would
    be superseded with no replacement — it would vanish from the store.
    """
    from particles import embeddings as ep
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client
    from particles.store.particle_store import get_active_particles_for_entry

    claim = "A claim that survives a reindex."
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm([claim]))
    try:
        e1, s1 = await _deposit(db_session, tmp_path, "a.md", "source")
        first = await extract_snapshot(db_session, e1, s1)
        await db_session.commit()
        assert len(first) == 1

        # Re-extract the same snapshot with the prior particle marked doomed.
        suppressed: list[str] = []
        again = await extract_snapshot(
            db_session,
            e1,
            s1,
            supersede_ids=frozenset({first[0].id}),
            suppressed_ids_out=suppressed,
        )
        await db_session.commit()

        # A fresh particle was minted rather than folded into the doomed one.
        assert suppressed == []
        assert len(again) == 1
        assert again[0].id != first[0].id
        active = await get_active_particles_for_entry(db_session, e1)
        assert claim in {p.content for p in active}
    finally:
        set_client(None)
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_suppression_is_disclosed_in_quality_notes(
    db_session: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """a suppressing pass must never look like a silent no-op."""
    import logging

    from particles import embeddings as ep
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    claim = "A disclosed claim."
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm([claim]))
    try:
        e1, s1 = await _deposit(db_session, tmp_path, "a.md", "first")
        await extract_snapshot(db_session, e1, s1)
        await db_session.commit()

        e2, s2 = await _deposit(db_session, tmp_path, "b.md", "second")
        with caplog.at_level(logging.INFO, logger="particles.ingest.pipeline"):
            await extract_snapshot(db_session, e2, s2)
        await db_session.commit()

        assert "DUPLICATE_SUPPRESSED" in caplog.text
    finally:
        set_client(None)
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_mutable_reharvest_keeps_the_survivor_active(db_session: Any, tmp_path: Path) -> None:
    """The interaction: the generation cascade must not demote the survivor.

    A MUTABLE memory file re-deposited with a new snapshot is the exact shape
    that produced the measured duplicate mass. The suppressed-into particle
    still points at the snapshot it was first extracted from, so without the
    ``exclude_ids`` hand-off the cascade would retire the very particle the
    suppression kept — and the claim would leave the ACTIVE surface entirely.
    """
    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_file
    from particles.ingest.pipeline import extract_snapshot
    from particles.llm import set_client

    claim = "A memory-file claim that persists across sessions."
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    set_client(_mock_llm([claim]))
    try:
        doc = tmp_path / "MEMORY.md"
        doc.write_text("- a memory line\n")
        e1, s1 = await deposit_file(
            db_session, doc, deposited_by="test", mutability=Mutability.MUTABLE
        )
        await db_session.commit()
        first = await extract_snapshot(db_session, e1, s1)
        await db_session.commit()
        assert len(first) == 1

        # The file changes (as MEMORY.md does most sessions) → a new snapshot.
        doc.write_text("- a memory line\n- another line\n")
        e2, s2 = await deposit_file(
            db_session, doc, deposited_by="test", mutability=Mutability.MUTABLE
        )
        await db_session.commit()
        assert e2 == e1 and s2 != s1

        suppressed: list[str] = []
        await extract_snapshot(db_session, e2, s2, suppressed_ids_out=suppressed)
        await db_session.commit()

        assert suppressed == [first[0].id]
        survivor = await get_particle(db_session, first[0].id)
        assert survivor is not None
        assert survivor.status is Status.ACTIVE, (
            "the generation cascade demoted the suppressed-into particle; "
            "the claim has no ACTIVE copy left"
        )
    finally:
        set_client(None)
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_repeated_assertion_is_idempotent(db_session: Any) -> None:
    """the rung runs on reconcile_and_insert too."""
    from particles import embeddings as ep
    from particles.ingest.pipeline import reconcile_and_insert

    model = MagicMock()
    model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        entry = CorpusEntry(
            entry_id="entry-assert",
            source_type="CONVERSATION",
            uri_r="claude-code://session/x",
            fetch_policy=FetchPolicy.NEVER,
            mutability=Mutability.APPEND_ONLY,
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(entry))
        await db_session.flush()

        first = await reconcile_and_insert(
            db_session, _particle("An asserted belief.", entry_id="entry-assert")
        )
        assert first is not None
        second = await reconcile_and_insert(
            db_session,
            _particle("An asserted belief.", entry_id="entry-assert", snapshot_id="snap-2"),
        )
        # The existing particle is returned rather than a second copy written.
        assert second is not None
        assert second.id == first.id

        found = await get_active_particles_by_content_hashes(
            db_session, [content_hash("An asserted belief.")]
        )
        assert len(found) == 1
    finally:
        ep.set_embedding_model(original)


@pytest.mark.asyncio
async def test_stale_rows_without_a_hash_are_simply_missed(db_session: Any) -> None:
    """A pre-migration row (NULL hash) is invisible to the lookup, never a crash."""
    from sqlalchemy import update

    from particles.store.particle_store import ParticleRow

    p = _particle("A legacy claim.")
    await insert_particle(db_session, p)
    await db_session.execute(
        update(ParticleRow).where(ParticleRow.id == p.id).values(content_norm_hash=None)
    )
    await db_session.flush()

    found = await get_active_particles_by_content_hashes(
        db_session, [content_hash("A legacy claim.")]
    )
    assert found == []


def test_empty_and_missing_hashes_short_circuit() -> None:
    """Guard: an empty candidate list must not build a degenerate query."""
    index = DuplicateIndex()
    assert index.find(_particle("anything")) is None


@pytest.mark.asyncio
async def test_lookup_with_no_hashes_returns_empty(db_session: Any) -> None:
    assert await get_active_particles_by_content_hashes(db_session, []) == []
    assert await get_active_particles_by_content_hashes(db_session, [""]) == []


def test_suppression_note_reports_the_count() -> None:
    from particles.ingest.duplicate_suppression import suppression_note

    note = suppression_note(12)
    assert note.startswith("DUPLICATE_SUPPRESSED:")
    assert "12" in note


def test_index_add_ignores_ineligible_particles() -> None:
    """An ineligible particle never becomes a suppression target."""
    index = DuplicateIndex()
    index.add(_particle("An opinion.", modality=AssertionModality.EVALUATIVE))
    assert index.find(_particle("An opinion.", modality=AssertionModality.FALSIFIABLE)) is None


@pytest.mark.asyncio
async def test_append_to_missing_particle_is_a_no_op(db_session: Any) -> None:
    assert (
        await append_provenance_ref(
            db_session,
            str(uuid.uuid4()),
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e", snapshot_id="s"),
        )
        is False
    )


def test_hash_is_stable_across_calls() -> None:
    """The hash is persisted, so it must not depend on process state."""
    assert content_hash("A claim.") == content_hash("A claim.")
    assert len(content_hash("A claim.")) == 64
