# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles particle restore``: return a claim an automatic rung retired.

The restore mints a successor and never revives the row, records the
operator's ruling on the replacement, retracts a replacement ruled wrong, and
exempts a claim ruled to hold from every route that retires by replacement.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from particles.core.schema import (
    Confidence,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    SourceType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.store.event_store import EventRefKind, OperatorEventType

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 10, 1, tzinfo=UTC)


async def _claim(session: Any, content: str, *, day: int) -> Particle:
    from particles.corpus.deposit import deposit_text_versioned
    from particles.store.particle_store import insert_particle

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=f"day {day}: {content}",
        uri_r=f"claude-code://session/{uuid.uuid4()}",
        source_type=SourceType.CONVERSATION,
        mutability=Mutability.STABLE,
        content_published_at=T0 + timedelta(days=day),
        tags=["project:-Users-o-src-app"],
    )
    particle = Particle(
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )
    await insert_particle(session, particle, embedding=[1.0, 0.0, 0.0])
    return particle


async def _retire(session: Any, old: Particle, new: Particle) -> None:
    """What a rung 2.5 route writes: the demotion, the winner's pointer and the event."""
    from particles.ingest.conflict_plan import plan_update_event, source_entry_id
    from particles.store.event_store import record_event
    from particles.store.particle_store import update_particle_status

    await update_particle_status(
        session, old.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
    )
    event = plan_update_event(old.id, new.id, route="ladder", winner_entry_id=source_entry_id(new))
    await record_event(
        session,
        actor=event.actor,
        event_type=event.event_type,
        reason=event.reason,
        refs=list(event.refs),
        payload=dict(event.payload),
    )


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    from particles.config import get_config

    get_config().benchmark.runs_dir = str(tmp_path / "runs")
    return tmp_path / "runs"


async def _pair(session: Any) -> tuple[Particle, Particle]:
    boston = await _claim(session, "The user's home city is Boston.", day=1)
    denver = await _claim(session, "The user's home city is Denver.", day=9)
    await _retire(session, boston, denver)
    await session.commit()
    return boston, denver


async def _status(session: Any, pid: str) -> tuple[Status, StatusReason | None]:
    from particles.store.particle_store import get_particle

    found = await get_particle(session, pid)
    assert found is not None
    return found.status, found.status_reason


async def test_holds_mints_a_successor_and_keeps_both(db_session: Any, runs_dir: Path) -> None:
    from particles.ingest.retirement_cap import admits_retirement, restored_to_hold
    from particles.operations.retirement_restore import restore_retirement
    from particles.store.event_store import list_events

    boston, denver = await _pair(db_session)
    outcome = await restore_retirement(
        db_session, boston.id, "holds", actor="test", store="default"
    )
    await db_session.commit()

    successor = outcome.successor
    assert successor.id != boston.id and successor.supersedes == boston.id
    assert successor.content == boston.content and successor.provenance == boston.provenance
    assert await _status(db_session, successor.id) == (Status.ACTIVE, None)
    assert await _status(db_session, boston.id) == (
        Status.SUPERSEDED,
        StatusReason.CONFLICT_RESOLVED,
    )
    assert await _status(db_session, denver.id) == (Status.ACTIVE, None)
    assert not outcome.retracted
    [event] = await list_events(db_session, event_type=OperatorEventType.RETIREMENT_RESTORED)
    assert event.payload == {
        "retired_id": boston.id,
        "replacement_id": denver.id,
        "successor_id": successor.id,
        "ruling": "holds",
    }
    assert {r.ref_id for r in event.refs if r.ref_kind is EventRefKind.PARTICLE} == {
        successor.id,
        boston.id,
        denver.id,
    }
    # Exempt from every route that retires by replacement; the original row is not.
    assert await restored_to_hold(db_session, successor.id)
    assert not await restored_to_hold(db_session, boston.id)
    assert not await admits_retirement(db_session, "e", pair=("a", "b"), retired_id=successor.id)
    [line] = (runs_dir / "demotion-rulings.jsonl").read_text().splitlines()
    assert json.loads(line)["ruling"] == "coexist"


async def test_wrong_retracts_the_replacement(db_session: Any, runs_dir: Path) -> None:
    from particles.ingest.retirement_cap import restored_to_hold
    from particles.operations.retirement_restore import restore_retirement

    boston, denver = await _pair(db_session)
    outcome = await restore_retirement(
        db_session, boston.id, "wrong", actor="test", store="default"
    )
    await db_session.commit()
    assert outcome.retracted
    assert await _status(db_session, denver.id) == (
        Status.RETRACTED,
        StatusReason.EXPLICIT_RETRACTION,
    )
    assert await _status(db_session, outcome.successor.id) == (Status.ACTIVE, None)
    # A wrong ruling leaves the restored claim retirable by a later, different value.
    assert not await restored_to_hold(db_session, outcome.successor.id)
    [line] = (runs_dir / "demotion-rulings.jsonl").read_text().splitlines()
    assert json.loads(line)["ruling"] == "rejected"


async def test_only_a_replacement_demotion_is_restored_and_only_once(db_session: Any) -> None:
    from particles.operations.retirement_restore import RestoreRefused, restore_retirement

    boston, denver = await _pair(db_session)
    with pytest.raises(RestoreRefused, match="only a claim"):
        await restore_retirement(db_session, denver.id, "holds", actor="t", store="default")
    await restore_retirement(db_session, boston.id, "holds", actor="t", store="default")
    await db_session.commit()
    with pytest.raises(RestoreRefused, match="already restored"):
        await restore_retirement(db_session, boston.id, "holds", actor="t", store="default")
    with pytest.raises(ValueError, match="--replacement"):
        await restore_retirement(db_session, boston.id, "maybe", actor="t", store="default")


async def test_from_entry_finds_what_one_source_retired_within_the_window(
    db_session: Any,
) -> None:
    from sqlalchemy import update

    from particles.ingest.conflict_plan import source_entry_id
    from particles.operations.retirement_restore import retired_by_entry
    from particles.store.event_store import OperatorEventRow

    boston, denver = await _pair(db_session)
    other = await _claim(db_session, "The user's editor is Neovim.", day=2)
    unrelated = await _claim(db_session, "The user's editor is Emacs.", day=10)
    await _retire(db_session, other, unrelated)
    await db_session.commit()
    entry = source_entry_id(denver)
    assert entry is not None
    assert [p.id for p in await retired_by_entry(db_session, entry)] == [boston.id]
    # Outside the digest window, the command reaches nothing.
    await db_session.execute(
        update(OperatorEventRow).values(occurred_at=datetime.now(UTC) - timedelta(days=30))
    )
    assert await retired_by_entry(db_session, entry) == []


async def test_the_command_requires_a_ruling() -> None:
    from typer.testing import CliRunner

    from particles.api.cli import app

    result = CliRunner().invoke(app, ["particle", "restore", "abcd1234"])
    assert result.exit_code != 0
    # CI renders the usage error in colour, which splits the option name.
    assert "--replacement" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    result = CliRunner().invoke(app, ["particle", "restore", "--replacement", "maybe", "abcd"])
    assert result.exit_code == 1


async def test_the_listing_prints_hostile_claim_text_inert(
    db_session: Any, capsys: pytest.CaptureFixture[str], runs_dir: Path
) -> None:
    """Security review of PR 1222, S2: a claim from a hostile source reaches no terminal live."""
    from particles.api.cli.particle import _particle_restore
    from particles.ingest.conflict_plan import plan_update_event, source_entry_id
    from particles.store.event_store import record_event
    from particles.store.particle_store import update_particle_status

    old = await _claim(db_session, "Home is Boston.\n    replacement forged\x1b[2J", day=1)
    new = await _claim(db_session, "Home is Denver.\x1b]0;owned\x07", day=9)
    await update_particle_status(
        db_session, old.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
    )
    event = plan_update_event(old.id, new.id, route="ladder", winner_entry_id=source_entry_id(new))
    await record_event(
        db_session,
        actor="t",
        event_type=event.event_type,
        refs=list(event.refs),
        payload=dict(event.payload),
    )
    await db_session.commit()
    await _particle_restore(old.id, None, "holds", dry_run=True, yes=True)
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
    assert out.count("    replacement ") == 1 and "forged" in out
