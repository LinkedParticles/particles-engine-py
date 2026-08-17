"""Tests for exact-duplicate auto-merge — `links dedup`.

The Tier-A predicate is **identical content under the §6.10 normalized key**
(the same key extract-time suppression uses), not a similarity
threshold, so these tests pin the boundary hard: a one-token difference is
never merged (the measured worst false positive, `claude-opus-4-6` vs
`claude-opus-4-5`, sits at cosine 0.9951), detection never touches an
embedding or the LLM, `--dry-run` writes nothing, and a merge links +
supersedes but never deletes and never mutates the survivor.

Covers `find_exact_duplicate_groups` / `auto_merge_exact_duplicates` in
`particles/operations/links_suggest.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select

from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.operations.links_suggest import (
    AutoMergeDisabled,
    auto_merge_exact_duplicates,
    find_exact_duplicate_groups,
)
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import (
    ParticleRow,
    get_particle,
    insert_particle,
)
from particles.store.relation_store import ParticleRelationRow, create_relation
from particles.store.subject_store import insert_subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_DUP = "uv parses pyproject.toml during settings discovery."
# One token apart — the false-positive shape. Never Tier A.
_NEAR = "uv parses pyproject.tomls during settings discovery."


def _pid(n: int) -> str:
    return f"00000000-0000-0000-0000-{n:012d}"


def _mk(
    content: str,
    particle_id: str,
    subject_ids: list[str],
    *,
    asserted_at: datetime | None = None,
    modality: AssertionModality = AssertionModality.FALSIFIABLE,
    properties: dict[str, object] | None = None,
) -> Particle:
    return Particle(
        id=particle_id,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=asserted_at or datetime(2026, 1, 1, tzinfo=UTC),
        subject_ids=subject_ids,
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-default")],
        assertion_modality=modality,
        properties=properties or {},
    )


async def _seed_subject(session: AsyncSession, name: str = "uv") -> Subject:
    subj = Subject(canonical_name=name, asserted_by="test")
    await insert_subject(session, subj)
    return subj


async def _seed_triplet(session: AsyncSession) -> tuple[Subject, list[str]]:
    """Three byte-identical copies (deliberately inserted newest-first) + one near-dup."""
    subj = await _seed_subject(session)
    ids = [_pid(1), _pid(2), _pid(3)]
    stamps = [
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
    ]
    for pid, stamp in zip(ids, stamps, strict=True):
        await insert_particle(session, _mk(_DUP, pid, [subj.id], asserted_at=stamp))
    await insert_particle(session, _mk(_NEAR, _pid(9), [subj.id]))
    await session.commit()
    return subj, ids


def _enable(monkeypatch: pytest.MonkeyPatch, *, max_per_run: int = 500) -> None:
    from particles.config import get_config

    cfg = get_config().links_suggest.auto_merge
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "max_per_run", max_per_run)


# ---------------------------------------------------------------------------
# Detection — exact identity only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groups_only_byte_identical_active_content(db_session: AsyncSession) -> None:
    """One group over the three identical copies; the near-duplicate is excluded."""
    _subj, ids = await _seed_triplet(db_session)

    groups = await find_exact_duplicate_groups(db_session)

    assert len(groups) == 1
    group = groups[0]
    assert {group.survivor_id, *group.redundant_ids} == set(ids)
    assert len(group.redundant_ids) == 2
    assert _pid(9) not in {group.survivor_id, *group.redundant_ids}
    assert group.merged is False


@pytest.mark.asyncio
async def test_trailing_punctuation_twins_are_one_group(db_session: AsyncSession) -> None:
    """the mop keys on the same normalized key suppression does.

    Before the swap the mop keyed on ``sha256(content)`` while the newer
    extract-time rung keyed on the normalized key, so this pair could never be
    minted twice yet — once minted — was permanently unreachable by cleanup.
    """
    subj = await _seed_subject(db_session)
    await insert_particle(db_session, _mk(_DUP, _pid(1), [subj.id]))
    await insert_particle(db_session, _mk(f"  {_DUP.rstrip('.')}  ", _pid(2), [subj.id]))
    await db_session.commit()

    groups = await find_exact_duplicate_groups(db_session)

    assert len(groups) == 1
    assert {groups[0].survivor_id, *groups[0].redundant_ids} == {_pid(1), _pid(2)}


@pytest.mark.asyncio
async def test_normalisation_never_reaches_a_wording_difference(
    db_session: AsyncSession,
) -> None:
    """Normalization absorbs whitespace and trailing marks — never a token."""
    subj = await _seed_subject(db_session)
    await insert_particle(db_session, _mk(_DUP, _pid(1), [subj.id]))
    await insert_particle(db_session, _mk(_NEAR.rstrip("."), _pid(2), [subj.id]))
    # Case is preserved too: "UV parses…" is a different claim key.
    await insert_particle(db_session, _mk(_DUP.upper(), _pid(3), [subj.id]))
    await db_session.commit()

    assert await find_exact_duplicate_groups(db_session) == []


@pytest.mark.asyncio
async def test_near_duplicate_alone_is_never_a_group(db_session: AsyncSession) -> None:
    """A one-token difference is Tier B forever — no threshold reaches it."""
    subj = await _seed_subject(db_session)
    await insert_particle(db_session, _mk(_DUP, _pid(1), [subj.id]))
    await insert_particle(db_session, _mk(_NEAR, _pid(2), [subj.id]))
    await db_session.commit()

    assert await find_exact_duplicate_groups(db_session) == []


@pytest.mark.asyncio
async def test_non_active_copies_are_not_grouped(db_session: AsyncSession) -> None:
    """Only ACTIVE content participates; a superseded twin is invisible."""
    subj = await _seed_subject(db_session)
    await insert_particle(db_session, _mk(_DUP, _pid(1), [subj.id]))
    await insert_particle(db_session, _mk(_DUP, _pid(2), [subj.id]))
    await db_session.commit()
    from particles.store.particle_store import update_particle_status

    await update_particle_status(
        db_session, _pid(2), Status.SUPERSEDED, StatusReason.EXPLICIT_SUPERSESSION
    )
    await db_session.commit()

    assert await find_exact_duplicate_groups(db_session) == []


@pytest.mark.asyncio
async def test_non_truth_apt_and_non_asserted_excluded(db_session: AsyncSession) -> None:
    """gates are inherited, not re-derived."""
    from particles.extraction.polarity import POLARITY_KEY

    subj = await _seed_subject(db_session)
    # Two identical EVALUATIVE copies …
    await insert_particle(
        db_session, _mk(_DUP, _pid(1), [subj.id], modality=AssertionModality.EVALUATIVE)
    )
    await insert_particle(
        db_session, _mk(_DUP, _pid(2), [subj.id], modality=AssertionModality.EVALUATIVE)
    )
    # … and two identical DECLINED ones.
    await insert_particle(
        db_session, _mk(_NEAR, _pid(3), [subj.id], properties={POLARITY_KEY: "DECLINED"})
    )
    await insert_particle(
        db_session, _mk(_NEAR, _pid(4), [subj.id], properties={POLARITY_KEY: "DECLINED"})
    )
    await db_session.commit()

    assert await find_exact_duplicate_groups(db_session) == []


@pytest.mark.asyncio
async def test_different_holder_stances_never_merged(db_session: AsyncSession) -> None:
    """identical text held by different principals is not one claim."""
    from particles.core.stance import STANCE_HOLDER_KEY

    subj = await _seed_subject(db_session)
    await insert_particle(
        db_session, _mk(_DUP, _pid(1), [subj.id], properties={STANCE_HOLDER_KEY: "x:alice"})
    )
    await insert_particle(
        db_session, _mk(_DUP, _pid(2), [subj.id], properties={STANCE_HOLDER_KEY: "x:bob"})
    )
    await db_session.commit()

    assert await find_exact_duplicate_groups(db_session) == []


@pytest.mark.asyncio
async def test_cross_subject_duplicates_stay_unreachable(db_session: AsyncSession) -> None:
    """Copies with *disagreeing* Subjects are still never merged.

    The finder was widened to subject-*less* copies; it did not widen it
    across Subjects. Two copies whose Subjects differ share no component, so
    each is a component of one and nothing merges.
    """
    subj_a = await _seed_subject(db_session, "uv")
    subj_b = await _seed_subject(db_session, "pip")
    await insert_particle(db_session, _mk(_DUP, _pid(3), [subj_a.id]))
    await insert_particle(db_session, _mk(_DUP, _pid(4), [subj_b.id]))
    await db_session.commit()

    assert await find_exact_duplicate_groups(db_session) == []


@pytest.mark.asyncio
async def test_subject_less_duplicates_are_merged(db_session: AsyncSession) -> None:
    """the Subject is no longer a membership gate."""
    await insert_particle(db_session, _mk(_DUP, _pid(1), []))
    await insert_particle(db_session, _mk(_DUP, _pid(2), []))
    await db_session.commit()

    (group,) = await find_exact_duplicate_groups(db_session)
    assert group.survivor_id == _pid(1)
    assert group.redundant_ids == [_pid(2)]
    assert group.subject_class == "orphan"
    assert group.subject_ids == []


@pytest.mark.asyncio
async def test_mixed_group_absorbs_orphans_and_elects_the_linked_survivor(
    db_session: AsyncSession,
) -> None:
    """§2 + §3 — the regression the naive widening would have shipped.

    The orphan here is the **earliest** copy, so Subject-blind
    election would have crowned it and superseded the subject-linked copy
    underneath — dropping that claim out of §6.7 subject-filtered query and
    manufacturing a NO_SUBJECT orphan. The leading election term is what
    prevents it, so this test fails if that term is removed.
    """
    subj = await _seed_subject(db_session)
    await insert_particle(
        db_session, _mk(_DUP, _pid(1), [], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    await insert_particle(
        db_session, _mk(_DUP, _pid(2), [], asserted_at=datetime(2026, 2, 1, tzinfo=UTC))
    )
    await insert_particle(
        db_session, _mk(_DUP, _pid(3), [subj.id], asserted_at=datetime(2026, 3, 1, tzinfo=UTC))
    )
    await db_session.commit()

    (group,) = await find_exact_duplicate_groups(db_session)
    assert group.survivor_id == _pid(3), "the subject-linked copy must survive"
    assert group.redundant_ids == [_pid(1), _pid(2)]
    assert group.subject_class == "mixed"
    assert group.subject_ids == [subj.id]


@pytest.mark.asyncio
async def test_orphans_stay_separate_when_the_home_is_ambiguous(
    db_session: AsyncSession,
) -> None:
    """two linked components ⇒ the pass declines to pick a referent.

    Absorbing the orphans into either Subject would be a guess about what the
    claim is *about*, so they merge only among themselves.
    """
    subj_a = await _seed_subject(db_session, "uv")
    subj_b = await _seed_subject(db_session, "pip")
    await insert_particle(db_session, _mk(_DUP, _pid(1), []))
    await insert_particle(db_session, _mk(_DUP, _pid(2), []))
    await insert_particle(db_session, _mk(_DUP, _pid(3), [subj_a.id]))
    await insert_particle(db_session, _mk(_DUP, _pid(4), [subj_b.id]))
    await db_session.commit()

    (group,) = await find_exact_duplicate_groups(db_session)
    assert group.subject_class == "orphan"
    assert group.survivor_id == _pid(1)
    assert group.redundant_ids == [_pid(2)]
    # Neither subject-linked copy is touched — no component reaches size 2.
    assert _pid(3) not in group.redundant_ids
    assert _pid(4) not in group.redundant_ids


@pytest.mark.asyncio
async def test_subject_linked_outranks_subject_less_regardless_of_age(
    db_session: AsyncSession,
) -> None:
    """the leading term dominates `asserted_at`, in both directions."""
    from particles.operations.links_suggest import _election_key

    subj = await _seed_subject(db_session)
    old_orphan = _mk(_DUP, _pid(1), [], asserted_at=datetime(2020, 1, 1, tzinfo=UTC))
    new_linked = _mk(_DUP, _pid(2), [subj.id], asserted_at=datetime(2030, 1, 1, tzinfo=UTC))
    assert _election_key(new_linked) < _election_key(old_orphan)

    # …and among equals the ordering is untouched.
    early = _mk(_DUP, _pid(3), [subj.id], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    late = _mk(_DUP, _pid(4), [subj.id], asserted_at=datetime(2026, 2, 1, tzinfo=UTC))
    assert _election_key(early) < _election_key(late)
    tie_hi = _mk(_DUP, _pid(6), [], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    tie_lo = _mk(_DUP, _pid(5), [], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert _election_key(tie_lo) < _election_key(tie_hi)


@pytest.mark.asyncio
async def test_merging_a_mixed_group_leaves_the_claim_subject_indexed(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The payoff: the absorbed orphan's claim keeps its Subject.

    End-to-end over the write path, because this is the property that makes the
    widening safe rather than merely wider.
    """
    _enable(monkeypatch)
    subj = await _seed_subject(db_session)
    await insert_particle(
        db_session, _mk(_DUP, _pid(1), [], asserted_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    await insert_particle(
        db_session, _mk(_DUP, _pid(2), [subj.id], asserted_at=datetime(2026, 2, 1, tzinfo=UTC))
    )
    await db_session.commit()

    report = await auto_merge_exact_duplicates(db_session, dry_run=False)
    assert report.merged_groups == 1

    survivor = await get_particle(db_session, _pid(2))
    assert survivor is not None
    assert survivor.status is Status.ACTIVE
    assert survivor.subject_ids == [subj.id]
    absorbed = await get_particle(db_session, _pid(1))
    assert absorbed is not None
    assert absorbed.status is Status.SUPERSEDED
    assert absorbed.status_reason is StatusReason.DUPLICATE_MERGED


@pytest.mark.asyncio
async def test_survivor_election_is_deterministic(db_session: AsyncSession) -> None:
    """Earliest asserted_at wins; a tie breaks on the smallest id."""
    _subj, _ids = await _seed_triplet(db_session)
    (group,) = await find_exact_duplicate_groups(db_session)
    # _pid(2) carries the earliest stamp although it was inserted second.
    assert group.survivor_id == _pid(2)
    assert group.redundant_ids == [_pid(3), _pid(1)]  # ordered by the same key

    # Tie on asserted_at → lexicographically smallest id.
    subj = await _seed_subject(db_session, "tie")
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    await insert_particle(db_session, _mk("tied claim.", _pid(21), [subj.id], asserted_at=stamp))
    await insert_particle(db_session, _mk("tied claim.", _pid(20), [subj.id], asserted_at=stamp))
    await db_session.commit()

    groups = await find_exact_duplicate_groups(db_session, subject_id=subj.id)
    assert [(g.survivor_id, g.redundant_ids) for g in groups] == [(_pid(20), [_pid(21)])]


# ---------------------------------------------------------------------------
# Dry run — read-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_reports_and_mutates_nothing(db_session: AsyncSession) -> None:
    _subj, ids = await _seed_triplet(db_session)

    report = await auto_merge_exact_duplicates(db_session)

    assert report.dry_run is True
    assert report.total_groups == 1
    assert report.total_redundant == 2
    assert report.merged_groups == 0
    assert report.merged_particles == 0
    assert report.links_created == 0
    assert all(not g.merged for g in report.groups)

    # Nothing changed: every copy is still ACTIVE, no relations, no events.
    for pid in ids:
        particle = await get_particle(db_session, pid)
        assert particle is not None
        assert particle.status is Status.ACTIVE
        assert particle.status_reason is None
    assert (
        await db_session.execute(select(func.count()).select_from(ParticleRelationRow))
    ).scalar_one() == 0
    assert await list_events(db_session) == []


@pytest.mark.asyncio
async def test_dry_run_works_with_auto_merge_disabled(db_session: AsyncSession) -> None:
    """Detection needs no opt-in — only writing does."""
    from particles.config import get_config

    assert get_config().links_suggest.auto_merge.enabled is False
    await _seed_triplet(db_session)

    report = await auto_merge_exact_duplicates(db_session, dry_run=True)
    assert report.total_groups == 1


# ---------------------------------------------------------------------------
# Apply — default OFF, then link + supersede, never delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_refused_when_disabled_and_writes_nothing(db_session: AsyncSession) -> None:
    """Default OFF is the gate; a refused run leaves the store untouched."""
    _subj, ids = await _seed_triplet(db_session)

    with pytest.raises(AutoMergeDisabled):
        await auto_merge_exact_duplicates(db_session, dry_run=False)

    for pid in ids:
        particle = await get_particle(db_session, pid)
        assert particle is not None
        assert particle.status is Status.ACTIVE
    assert await list_events(db_session) == []


@pytest.mark.asyncio
async def test_apply_links_supersedes_and_never_deletes(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch)
    _subj, ids = await _seed_triplet(db_session)
    survivor_before = await get_particle(db_session, _pid(2))
    assert survivor_before is not None

    report = await auto_merge_exact_duplicates(db_session, dry_run=False)

    assert report.dry_run is False
    assert (report.merged_groups, report.merged_particles, report.links_created) == (1, 2, 2)
    assert report.groups[0].merged is True

    # Survivor: still ACTIVE and byte-for-byte unmutated (§3 "never
    # mutate the survivor" — the write set is losers-only).
    survivor = await get_particle(db_session, _pid(2))
    assert survivor is not None
    assert survivor.status is Status.ACTIVE
    assert survivor.status_reason is None
    assert survivor.content == survivor_before.content
    assert survivor.confidence.value == survivor_before.confidence.value
    assert survivor.provenance == survivor_before.provenance

    # Losers: SUPERSEDED with the dedicated reason — and still readable. Nothing
    # is ever hard-deleted; the row count is unchanged.
    for pid in (_pid(1), _pid(3)):
        loser = await get_particle(db_session, pid)
        assert loser is not None, "auto-merge must never delete a particle"
        assert loser.status is Status.SUPERSEDED
        assert loser.status_reason is StatusReason.DUPLICATE_MERGED
        assert loser.content == _DUP
    total = (await db_session.execute(select(func.count()).select_from(ParticleRow))).scalar_one()
    assert total == 4  # 3 copies + the near-duplicate, all still present

    # CO_EVIDENTIAL edges, attributed to the deterministic path (never LLM_JUDGE).
    rows = (await db_session.execute(select(ParticleRelationRow))).scalars().all()
    assert len(rows) == 2
    for row in rows:
        assert row.relation_type == RelationType.CO_EVIDENTIAL.value
        assert row.created_by == RelationCreatedBy.EXACT_DUPLICATE.value
        assert _pid(2) in (row.particle_a, row.particle_b)


@pytest.mark.asyncio
async def test_apply_records_one_audited_event_per_group(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DUPLICATES_MERGED event is the audit trail and the revert input (§5)."""
    _enable(monkeypatch)
    await _seed_triplet(db_session)

    await auto_merge_exact_duplicates(db_session, dry_run=False)

    events = await list_events(db_session)
    assert len(events) == 1
    event = events[0]
    assert event.event_type is OperatorEventType.DUPLICATES_MERGED
    assert event.actor == "links-dedup"
    assert event.payload is not None
    assert event.payload["survivor"] == _pid(2)
    assert sorted(event.payload["superseded"]) == [_pid(1), _pid(3)]
    assert len(event.payload["content_hash"]) == 64
    assert event.payload["config"] == {"enabled": True, "max_per_run": 500}
    # Every touched particle is a queryable ref.
    assert {r.ref_id for r in event.refs} == {_pid(1), _pid(2), _pid(3)}


@pytest.mark.asyncio
async def test_apply_is_idempotent(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merged copy leaves ACTIVE, so the second run has nothing to find."""
    _enable(monkeypatch)
    await _seed_triplet(db_session)

    first = await auto_merge_exact_duplicates(db_session, dry_run=False)
    assert first.merged_groups == 1

    second = await auto_merge_exact_duplicates(db_session, dry_run=False)
    assert (second.total_groups, second.merged_groups, second.merged_particles) == (0, 0, 0)
    assert second.links_created == 0
    assert len(await list_events(db_session)) == 1  # no second event


@pytest.mark.asyncio
async def test_apply_skips_an_already_linked_pair(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing co-evidential edge is not duplicated; the copy still supersedes."""
    _enable(monkeypatch)
    await _seed_triplet(db_session)
    await create_relation(
        db_session,
        _pid(2),
        _pid(1),
        RelationType.CO_EVIDENTIAL,
        RelationCreatedBy.MANUAL_CLI,
    )
    await db_session.commit()

    report = await auto_merge_exact_duplicates(db_session, dry_run=False)

    assert report.links_created == 1  # only the un-linked copy got a new edge
    assert report.merged_particles == 2
    rows = (await db_session.execute(select(ParticleRelationRow))).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_max_per_run_caps_and_discloses_the_remainder(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capped run never reads as a complete cleanup."""
    _enable(monkeypatch, max_per_run=1)
    subj = await _seed_subject(db_session)
    for n, text in ((1, "claim one."), (2, "claim one."), (3, "claim two."), (4, "claim two.")):
        await insert_particle(db_session, _mk(text, _pid(n), [subj.id]))
    await db_session.commit()

    report = await auto_merge_exact_duplicates(db_session, dry_run=False)

    assert report.total_groups == 2
    assert report.merged_groups == 1
    assert (report.deferred_groups, report.deferred_redundant) == (1, 1)
    assert any("max_per_run=1" in w for w in report.warnings)


@pytest.mark.asyncio
async def test_no_llm_call_in_the_merge_path(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier A is a hash comparison — an exploding LLM seam must never be reached."""

    async def _boom(*args: object, **kwargs: object) -> str:
        raise AssertionError("auto-merge must not call the LLM")

    monkeypatch.setattr("particles.operations._llm._llm_call", _boom)
    _enable(monkeypatch)
    await _seed_triplet(db_session)

    report = await auto_merge_exact_duplicates(db_session, dry_run=False)
    assert report.merged_groups == 1
