"""CLI + retraction-handling tests (`particles links`)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.db import session_scope
from particles.store.particle_store import insert_particle, update_particle_status
from particles.store.relation_store import (
    create_relation,
    get_co_evidential_group,
    get_relations_for_particle,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _mk_particle(content: str, particle_id: str) -> Particle:
    return Particle(
        id=particle_id,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-x"),
        ],
    )


# ---------------------------------------------------------------------------
# Retraction handling: RETRACTED particles drop out of their co-evidential group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retraction_removes_particle_from_relations(db_session: AsyncSession) -> None:
    """update_particle_status to RETRACTED must drop the particle from particle_relations."""
    p1 = _mk_particle("first", "00000000-0000-0000-0000-000000000001")
    p2 = _mk_particle("second", "00000000-0000-0000-0000-000000000002")
    p3 = _mk_particle("third", "00000000-0000-0000-0000-000000000003")
    for p in (p1, p2, p3):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, p1.id, p2.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await create_relation(
        db_session, p2.id, p3.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    # Before retraction: full triangle group.
    assert await get_co_evidential_group(db_session, p1.id) == {p1.id, p2.id, p3.id}

    # Retract p2 — the hub. p1 and p3 should now be singleton groups.
    await update_particle_status(
        db_session, p2.id, Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION
    )
    await db_session.commit()

    assert await get_co_evidential_group(db_session, p1.id) == {p1.id}
    assert await get_co_evidential_group(db_session, p3.id) == {p3.id}
    # p2 has no relations left at all.
    assert await get_relations_for_particle(db_session, p2.id) == []


@pytest.mark.asyncio
async def test_non_retraction_status_changes_preserve_relations(
    db_session: AsyncSession,
) -> None:
    """Status transitions other than RETRACTED must not touch particle_relations."""
    p1 = _mk_particle("first", "00000000-0000-0000-0000-000000000001")
    p2 = _mk_particle("second", "00000000-0000-0000-0000-000000000002")
    for p in (p1, p2):
        await insert_particle(db_session, p)
    await create_relation(
        db_session, p1.id, p2.id, RelationType.CO_EVIDENTIAL, RelationCreatedBy.HUMAN_REVIEW
    )
    await db_session.commit()

    # Transition p1 to PROVENANCE_STALE (a non-RETRACTED status change). The
    # relation should survive — a stale particle is still part of its group.
    await update_particle_status(
        db_session, p1.id, Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY
    )
    await db_session.commit()

    rels = await get_relations_for_particle(db_session, p1.id)
    assert len(rels) == 1


# ---------------------------------------------------------------------------
# CLI: particles links add / remove / list
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _setup_two_particles(db_path: Path, ids: list[str]) -> None:
    async def _impl() -> None:
        async with session_scope() as session:
            for pid in ids:
                p = _mk_particle(content=f"claim {pid[:4]}", particle_id=pid)
                await insert_particle(session, p)
            await session.commit()

    asyncio.run(_impl())


def test_links_add_creates_relation(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    result = runner.invoke(app, ["links", "add", p_a, p_b], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Linked" in result.output
    assert "MANUAL_CLI" in result.output


def test_links_add_rejects_self_link(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    _setup_two_particles(cli_db, [p_a])

    result = runner.invoke(app, ["links", "add", p_a, p_a], catch_exceptions=False)
    assert result.exit_code == 1
    assert "itself" in result.output


def test_links_add_resolves_prefix(runner: CliRunner, cli_db: Path) -> None:
    p_a = "aaaaaaaa-1111-1111-1111-111111111111"
    p_b = "bbbbbbbb-2222-2222-2222-222222222222"
    _setup_two_particles(cli_db, [p_a, p_b])

    # Use only the first 8 chars of each ID — should still resolve uniquely
    # because the prefixes are distinct.
    result = runner.invoke(app, ["links", "add", p_a[:8], p_b[:8]], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_links_add_rejects_short_prefix(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    result = runner.invoke(app, ["links", "add", "0000", p_b], catch_exceptions=False)
    assert result.exit_code == 1
    assert "at least 8" in result.output


def test_links_add_rejects_unknown_relation_type(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    result = runner.invoke(
        app, ["links", "add", p_a, p_b, "--type", "nonsense"], catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "Unknown relation type" in result.output


def test_links_remove(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    runner.invoke(app, ["links", "add", p_a, p_b], catch_exceptions=False)
    result = runner.invoke(app, ["links", "remove", p_a, p_b], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Removed" in result.output


def test_links_remove_returns_error_when_missing(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    # No relation exists yet
    result = runner.invoke(app, ["links", "remove", p_a, p_b], catch_exceptions=False)
    assert result.exit_code == 1
    assert "No CO_EVIDENTIAL relation found" in result.output


def test_links_list_shows_singleton_for_unlinked_particle(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    _setup_two_particles(cli_db, [p_a])

    result = runner.invoke(app, ["links", "list", p_a], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No direct relations" in result.output
    assert "singleton" in result.output


def test_links_list_shows_group_after_add(runner: CliRunner, cli_db: Path) -> None:
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    runner.invoke(app, ["links", "add", p_a, p_b], catch_exceptions=False)
    result = runner.invoke(app, ["links", "list", p_a], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Direct relations (1)" in result.output
    assert "Co-evidential group (2 members" in result.output
    assert "← this particle" in result.output


# ---------------------------------------------------------------------------
# CLI: particles links list --kind
# ---------------------------------------------------------------------------


def _setup_mixed_relations(runner: CliRunner, p_a: str, p_b: str, p_c: str) -> None:
    """Link A↔B CO_EVIDENTIAL and A→C PART_OF via the CLI."""
    r1 = runner.invoke(app, ["links", "add", p_a, p_b], catch_exceptions=False)
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["links", "add", p_a, p_c, "--type", "part-of"], catch_exceptions=False)
    assert r2.exit_code == 0, r2.output


def test_links_list_kind_filter_matches(runner: CliRunner, cli_db: Path) -> None:
    """--kind restricts the direct-relations listing to the requested kind."""
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    p_c = "00000000-0000-0000-0000-00000000cccc"
    _setup_two_particles(cli_db, [p_a, p_b, p_c])
    _setup_mixed_relations(runner, p_a, p_b, p_c)

    result = runner.invoke(app, ["links", "list", p_a, "--kind", "part-of"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Direct relations of kind PART_OF (1)" in result.output
    assert "type=PART_OF" in result.output
    assert "type=CO_EVIDENTIAL" not in result.output
    # Filtering to a non-co-evidential kind skips the group section.
    assert "Co-evidential group" not in result.output


def test_links_list_kind_filter_co_evidential_keeps_group(runner: CliRunner, cli_db: Path) -> None:
    """--kind co-evidential filters the edges but still shows the group."""
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    p_c = "00000000-0000-0000-0000-00000000cccc"
    _setup_two_particles(cli_db, [p_a, p_b, p_c])
    _setup_mixed_relations(runner, p_a, p_b, p_c)

    result = runner.invoke(
        app, ["links", "list", p_a, "--kind", "CO_EVIDENTIAL"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "Direct relations of kind CO_EVIDENTIAL (1)" in result.output
    assert "type=PART_OF" not in result.output
    assert "Co-evidential group (2 members" in result.output


def test_links_list_kind_filter_excludes(runner: CliRunner, cli_db: Path) -> None:
    """--kind with no matching edges reports the kind-scoped empty message."""
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    _setup_two_particles(cli_db, [p_a, p_b])

    r = runner.invoke(app, ["links", "add", p_a, p_b], catch_exceptions=False)
    assert r.exit_code == 0, r.output

    result = runner.invoke(
        app, ["links", "list", p_a, "--kind", "endorses"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "No direct relations of kind ENDORSES." in result.output
    assert "Co-evidential group" not in result.output


def test_links_list_kind_accepts_reserved_kind(runner: CliRunner, cli_db: Path) -> None:
    """RESERVED kinds are defined in the registry and are valid filter values."""
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    _setup_two_particles(cli_db, [p_a])

    result = runner.invoke(
        app, ["links", "list", p_a, "--kind", "mentions"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "No direct relations of kind MENTIONS." in result.output


def test_links_list_rejects_unknown_kind(runner: CliRunner, cli_db: Path) -> None:
    """An unknown --kind errors out listing the valid choices."""
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    _setup_two_particles(cli_db, [p_a])

    result = runner.invoke(
        app, ["links", "list", p_a, "--kind", "nonsense"], catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "Unknown relation type 'nonsense'" in result.output
    for valid in ("CO_EVIDENTIAL", "PART_OF", "SEQUENCE_IN", "ENDORSES", "DISPUTES"):
        assert valid in result.output


def test_links_list_without_kind_unchanged(runner: CliRunner, cli_db: Path) -> None:
    """No --kind: all kinds listed and the group section renders as before."""
    p_a = "00000000-0000-0000-0000-00000000aaaa"
    p_b = "00000000-0000-0000-0000-00000000bbbb"
    p_c = "00000000-0000-0000-0000-00000000cccc"
    _setup_two_particles(cli_db, [p_a, p_b, p_c])
    _setup_mixed_relations(runner, p_a, p_b, p_c)

    result = runner.invoke(app, ["links", "list", p_a], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Direct relations (2):" in result.output
    assert "type=CO_EVIDENTIAL" in result.output
    assert "type=PART_OF" in result.output
    assert "Co-evidential group (2 members" in result.output
