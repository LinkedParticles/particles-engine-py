# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer-aware reconciliation, through the real extraction pipeline.

Two projects' memory files are deposited exactly as the SessionEnd harvest
deposits them (``LOCAL_MARKDOWN``, ``MUTABLE``, a ``project:`` key) and
extracted by a scripted extractor that sends one chunk per line through the
chunk carry-forward, so an unchanged line is carried rather than re-emitted.
The §6.6 probe is scripted too: two lines contradict when they give one slot
different values.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    Mutability,
    RelationCreatedBy,
    RelationType,
    Snapshot,
    UncertaintyNature,
)
from particles.corpus.deposit import deposit_text_versioned
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward
from particles.ingest.observer_gate import DivergenceTally
from particles.ingest.pipeline import extract_snapshot
from particles.llm import CompletionError, VisionImage, override_providers
from particles.store.observer_scope_join import load_scopes
from particles.store.particle_store import ParticleRow, ProvenanceEdgeRow
from particles.store.relation_store import ParticleRelationRow
from tests._observer_scope import rescoped
from tests.test_benchmark_rot import bow_encoder  # noqa: F401 — the stand-in encoder fixture

pytestmark = pytest.mark.usefixtures("bow_encoder")

#: ``line → (subject, slot value)``; a line with no value is a slot-less rule.
SUBJECT = "the repository"
VALUES = ("main", "master", "trunk")


def branch(value: str) -> str:
    return f"The default branch of the repository is {value}."


class _LineChunkExtractor:
    """One chunk per line, through the real carry-forward."""

    EXTRACTOR_ID = "observer-test"
    EXTRACTOR_VERSION = "1"

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        lines = [ln for ln in content.decode().splitlines() if ln.strip()]

        async def per_chunk(text: str) -> tuple[list[CandidateParticle], list[str], bool]:
            return (
                [
                    CandidateParticle(
                        content=text.strip(),
                        confidence_value=0.9,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        subjects=[SUBJECT],
                    )
                ],
                [],
                False,
            )

        entry_id = kwargs.get("corpus_entry_id")
        return await extract_with_carry_forward(
            kwargs.get("session"),  # type: ignore[arg-type]
            [ChunkUnit(chunk_id=f"line_{i}", chunk_text=ln) for i, ln in enumerate(lines)],
            entry_id if isinstance(entry_id, str) else None,
            self.EXTRACTOR_ID,
            self.EXTRACTOR_VERSION,
            call_llm=per_chunk,
        )


class _SlotProbe:
    """The §6.6 probe: two branch lines contradict when their values differ."""

    @property
    def provider_model(self) -> str:
        return "test:slot-probe"

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
        found = {v for v in VALUES if f" {v}." in prompt}
        if not found:
            raise CompletionError("not a probe prompt")
        return "YES: different values" if len(found) > 1 else "NO"


@pytest.fixture
def scripted_probe() -> Generator[None, None, None]:
    with override_providers({"semantic_lint": _SlotProbe()}):
        yield


_CLOCK = [datetime(2026, 1, 1, tzinfo=UTC)]


async def harvest(
    session: AsyncSession, project: str, lines: list[str], extractor: Any
) -> tuple[str, str]:
    """Deposit ``project``'s memory file and extract it; returns ``(entry, snapshot)``.

    Each harvest is dated a day after the last, as rung 2.5 needs.
    """
    _CLOCK[0] += timedelta(days=1)
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        session,
        text="\n".join(lines) + "\n",
        uri_r=f"file:///home/me/.claude/projects/{project}/memory/MEMORY.md",
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.MUTABLE,
        tags=["claude-code", "memory-file", f"project:{project}"],
        deposited_by="claude-code-hook",
        content_published_at=_CLOCK[0],
    )
    await session.commit()
    if not unchanged:
        await extract_snapshot(session, entry_id, snapshot_id, extractor=extractor)
        await session.commit()
    return entry_id, snapshot_id


async def row_for(session: AsyncSession, content: str) -> ParticleRow:
    rows = (
        (await session.execute(select(ParticleRow).where(ParticleRow.content == content)))
        .scalars()
        .all()
    )
    assert len(rows) == 1, f"{len(rows)} rows hold {content!r}"
    return rows[0]


class TestCarryForwardRecordsReobservation:
    """§1: a carried-forward particle gains a ref naming the re-observing snapshot."""

    async def test_the_ref_is_appended_and_nothing_is_re_pointed(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        rule = "Every commit to the repository needs a sign-off."
        _, first = await harvest(db_session, "alpha", [rule, branch("main")], extractor)
        _, second = await harvest(db_session, "alpha", [rule, branch("trunk")], extractor)

        row = await row_for(db_session, rule)
        refs = json.loads(row.provenance_json)
        assert row.status == "ACTIVE"
        # The earliest ref is still the decay anchor; the new one names snapshot 2.
        assert [r["snapshot_id"] for r in refs] == [first, second]
        assert refs[0]["chunk_hash"] == refs[1]["chunk_hash"] is not None
        edges = (
            await db_session.execute(
                select(ProvenanceEdgeRow.snapshot_id).where(ProvenanceEdgeRow.particle_id == row.id)
            )
        ).scalars()
        assert list(edges) == [first]  # one edge per entry, never re-pointed

    async def test_re_extracting_the_same_snapshot_adds_nothing(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        rule = "The test suite must pass before a merge."
        entry_id, first = await harvest(db_session, "alpha", [rule], extractor)
        # A retry of the same generation carries the chunk forward onto itself.
        await extract_snapshot(db_session, entry_id, first, extractor=extractor)
        await db_session.commit()

        refs = json.loads((await row_for(db_session, rule)).provenance_json)
        assert [r["snapshot_id"] for r in refs] == [first]


async def relations(session: AsyncSession) -> list[tuple[str, str, str, str]]:
    rows = await session.execute(
        select(
            ParticleRelationRow.particle_a,
            ParticleRelationRow.particle_b,
            ParticleRelationRow.relation_type,
            ParticleRelationRow.created_by,
        )
    )
    return [tuple(r) for r in rows.all()]  # type: ignore[misc]


async def hand_note(session: AsyncSession, lines: list[str], extractor: Any) -> None:
    """The operator's own keyless deposit: a global source."""
    _CLOCK[0] += timedelta(days=1)
    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text="\n".join(lines) + "\n",
        uri_r="file:///home/me/notes/how-i-work.md",
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.STABLE,
        tags=[],
        content_published_at=_CLOCK[0],
    )
    await session.commit()
    await extract_snapshot(session, entry_id, snapshot_id, extractor=extractor)
    await session.commit()


class TestPrecondition:
    """§2: a pair is reconciled only when ``scope(existing) ⊆ scope(candidate)``."""

    async def test_two_projects_divergent_values_both_stand_joined_by_a_contradiction(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        tally = DivergenceTally()
        entry_id, snapshot_id, _ = await deposit_text_versioned(
            db_session,
            text=branch("master") + "\n",
            uri_r="file:///home/me/.claude/projects/beta/memory/MEMORY.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            tags=["claude-code", "memory-file", "project:beta"],
            content_published_at=_CLOCK[0] + timedelta(days=1),
        )
        await db_session.commit()
        await extract_snapshot(
            db_session, entry_id, snapshot_id, extractor=extractor, divergences_out=tally
        )
        await db_session.commit()

        main, master = (
            await row_for(db_session, branch("main")),
            await row_for(db_session, branch("master")),
        )
        assert (main.status, master.status) == ("ACTIVE", "ACTIVE")
        assert (tally.declined, tally.recorded) == (1, 1)
        assert await relations(db_session) == [
            (
                *sorted((main.id, master.id)),
                RelationType.CONTRADICTS.value,
                RelationCreatedBy.OBSERVER_DIVERGENCE.value,
            )
        ]

    async def test_a_projects_own_update_still_supersedes_its_own_value(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        await harvest(db_session, "beta", [branch("master")], extractor)
        await harvest(db_session, "alpha", [branch("trunk")], extractor)

        main = await row_for(db_session, branch("main"))
        assert (main.status, main.status_reason) == ("PROVENANCE_STALE", "SUPERSEDED_BY_UPDATE")
        assert (await row_for(db_session, branch("master"))).status == "ACTIVE"
        assert (await row_for(db_session, branch("trunk"))).status == "ACTIVE"

    async def test_a_claim_both_projects_state_survives_one_projects_update(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        await harvest(db_session, "beta", [branch("main")], extractor)  # folds
        await harvest(db_session, "alpha", [branch("trunk")], extractor)

        shared = await row_for(db_session, branch("main"))
        assert shared.status == "ACTIVE"  # beta's file still says main
        scope = (await load_scopes(db_session, [shared.to_model()]))[shared.id]
        assert scope.keys == {"beta"}  # alpha's file no longer does (§3)
        assert (await row_for(db_session, branch("trunk"))).status == "ACTIVE"

    async def test_a_global_claim_contested_by_a_project_goes_to_review(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await hand_note(db_session, [branch("main")], extractor)
        await harvest(db_session, "alpha", [branch("trunk")], extractor)

        assert (await row_for(db_session, branch("main"))).status == "ACTIVE"
        trunk = await row_for(db_session, branch("trunk"))
        assert (trunk.status, trunk.status_reason) == ("PROVENANCE_STALE", "CONFLICT_PENDING")
        inconsistencies = (
            await db_session.execute(
                select(ParticleRow.id).where(ParticleRow.status == "INCONSISTENCY")
            )
        ).all()
        assert len(inconsistencies) == 1

    async def test_a_global_candidate_pairs_as_today(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        await hand_note(db_session, [branch("trunk")], extractor)

        assert await relations(db_session) == []
        main = await row_for(db_session, branch("main"))
        assert (main.status, main.status_reason) == ("PROVENANCE_STALE", "SUPERSEDED_BY_UPDATE")

    async def test_an_unrescoped_store_reconciles_exactly_as_before(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        await harvest(db_session, "alpha", [branch("main")], extractor)
        await harvest(db_session, "beta", [branch("master")], extractor)

        main = await row_for(db_session, branch("main"))
        assert (main.status, main.status_reason) == ("PROVENANCE_STALE", "SUPERSEDED_BY_UPDATE")
        assert await relations(db_session) == []


class TestCarryForwardLookup:
    """The lookup reads the refs, so a folded or moved line is found."""

    async def test_a_claim_folded_from_another_file_is_carried_forward(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        rule = "Every commit to the repository needs a sign-off."
        await harvest(db_session, "beta", [rule], extractor)
        _, a1 = await harvest(db_session, "alpha", [rule], extractor)  # folds into beta's
        await harvest(db_session, "beta", [branch("main")], extractor)  # beta drops the rule
        _, a2 = await harvest(db_session, "alpha", [rule, branch("main")], extractor)

        row = await row_for(db_session, rule)
        assert row.status == "ACTIVE"
        assert a2 in {r["snapshot_id"] for r in json.loads(row.provenance_json)}

    async def test_a_chunk_whose_line_came_back_is_re_extracted_once(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        extractor = _LineChunkExtractor()
        rule = "The test suite must pass before a merge."
        await harvest(db_session, "alpha", [rule], extractor)
        await harvest(db_session, "alpha", [branch("main")], extractor)  # the rule retires
        await harvest(db_session, "alpha", [rule, branch("main")], extractor)  # and is back

        rows = (
            (await db_session.execute(select(ParticleRow).where(ParticleRow.content == rule)))
            .scalars()
            .all()
        )
        assert sorted(r.status for r in rows) == ["ACTIVE", "PROVENANCE_STALE"]
        # The restated claim is ACTIVE now, so the chunk is a hit again.
        entry_id, _ = await harvest(db_session, "alpha", [rule, branch("trunk")], extractor)
        from particles.store.particle_store import get_active_particles_for_chunk_hash

        hits = await get_active_particles_for_chunk_hash(
            db_session,
            entry_id,
            json.loads(next(r for r in rows if r.status == "ACTIVE").provenance_json)[0][
                "chunk_hash"
            ],
            _LineChunkExtractor.EXTRACTOR_ID,
            _LineChunkExtractor.EXTRACTOR_VERSION,
        )
        assert [p.content for p in hits] == [rule]


class TestOtherRoutes:
    """The precondition is a property of the pair, whichever route found it (§2)."""

    async def test_the_backlog_sweep_skips_a_declined_pair_before_probing(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        from particles.operations.reconcile import reconcile_updates

        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        await harvest(db_session, "beta", [branch("master")], extractor)

        summary = await reconcile_updates(db_session)

        assert summary["demoted"] == 0
        assert summary["observer_declined"] == 1
        assert (await row_for(db_session, branch("main"))).status == "ACTIVE"

    async def test_an_assertion_in_another_project_is_left_standing(
        self, db_session: AsyncSession, scripted_probe: None
    ) -> None:
        from particles.core.schema import Confidence, Particle, ProvenanceRef, ProvenanceRefType
        from particles.ingest.pipeline import reconcile_and_insert
        from particles.store.subject_store import find_by_name

        extractor = _LineChunkExtractor()
        await rescoped(db_session)
        await harvest(db_session, "alpha", [branch("main")], extractor)
        beta_entry, beta_snapshot, _ = await deposit_text_versioned(
            db_session,
            text="an agent's note in project beta",
            uri_r="mcp://beta/assert/1",
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            tags=["claude-code", "project:beta"],
        )
        subject = await find_by_name(db_session, SUBJECT)
        assert subject is not None
        asserted = Particle(
            content=branch("master"),
            confidence=Confidence(value=0.9),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            asserted_by="agent",
            subject_ids=[subject.id],
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=beta_entry,
                    snapshot_id=beta_snapshot,
                )
            ],
        )

        written = await reconcile_and_insert(db_session, asserted, fail_closed=True)
        await db_session.commit()

        assert written is not None and written.id == asserted.id
        assert (await row_for(db_session, branch("main"))).status == "ACTIVE"
        assert (await row_for(db_session, branch("master"))).status == "ACTIVE"
        assert [r[2:] for r in await relations(db_session)] == [
            (RelationType.CONTRADICTS.value, RelationCreatedBy.OBSERVER_DIVERGENCE.value)
        ]
