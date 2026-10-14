# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The verbs: ``particles validity`` and ``particles particle date``.

The predicate and the parser are covered by ``tests/test_validity.py`` and the
route by ``tests/test_window_query.py``; this file pins the regeneration pass
over a file-backed store (what it sets, what it skips, what it preserves, the
rule-digest staleness), the operator verdict and its pin, the remote refusal,
the argument validation, and the dry-run envelope.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
    ValidKind,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.validity import (
    VALID_FROM_BASIS_KEY,
    VALID_FROM_RULE_KEY,
    ValidFromBasis,
)

runner = CliRunner()


def _seed(*contents: str, **kwargs: object) -> list[str]:
    async def _insert() -> list[str]:
        from particles.db import session_scope
        from particles.store.particle_store import insert_particle

        ids = []
        async with session_scope() as session:
            for content in contents:
                particle = Particle(
                    content=content,
                    confidence=Confidence(
                        value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                    ),
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    asserted_by="test",
                    provenance=[
                        ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-1")
                    ],
                    **kwargs,  # type: ignore[arg-type]
                )
                await insert_particle(session, particle)
                ids.append(particle.id)
            await session.commit()
        return ids

    return asyncio.run(_insert())


def _read(particle_id: str) -> tuple[datetime | None, str | None, dict[str, object]]:
    async def _get() -> tuple[datetime | None, str | None, dict[str, object]]:
        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async with session_scope() as session:
            row = await session.get(ParticleRow, particle_id)
            assert row is not None
            props = json.loads(row.properties_json) if row.properties_json else {}
            return row.valid_from, row.valid_kind, props

    return asyncio.run(_get())


# ---------------------------------------------------------------------------
# particles validity
# ---------------------------------------------------------------------------


def test_validity_dry_run_reports_scope_and_census(cli_db: Path) -> None:
    _seed("Picked up the keys on 2026-10-10.", "The user likes thali.")
    result = runner.invoke(app, ["validity", "--dry-run"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dry_run"] is True
    assert report["scope"] == 2
    assert report["set"] == 1 and report["skipped"] == {"no_date": 1}
    assert report["census"]["total"] == 2 and report["census"]["dated"] == 0


def test_validity_sets_and_skips(cli_db: Path) -> None:
    keys, trip, powai, none, two = _seed(
        "Picked up the keys on 2026-10-10.",
        "The user is in Mumbai for a week, as of 2026-10-08.",
        "Spent six months in Powai in 2019.",
        "The user likes thali.",
        "From 2026-10-08 the user was away, returning on 2026-10-15.",
    )
    result = runner.invoke(app, ["validity"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["set"] == 3
    assert report["skipped"] == {"no_date": 1, "several_dates": 1}
    assert report["by_kind"] == {"EVENT": 2, "STATE": 1}
    assert report["census"]["dated"] == 3 and report["census"]["total"] == 5

    vf, kind, props = _read(keys)
    assert vf is not None and vf.date().isoformat() == "2026-10-10" and kind == "EVENT"
    assert props[VALID_FROM_BASIS_KEY] == ValidFromBasis.PROSE and props[VALID_FROM_RULE_KEY]
    _, kind, props = _read(trip)
    assert kind == "STATE" and props[VALID_FROM_BASIS_KEY] == ValidFromBasis.AS_OF
    _, kind, props = _read(powai)
    assert kind == "EVENT" and props[VALID_FROM_BASIS_KEY] == ValidFromBasis.COARSE
    assert _read(none)[0] is None and _read(two)[0] is None

    # A second run finds nothing in scope: the rule digest is current.
    again = json.loads(runner.invoke(app, ["validity"]).output)
    assert again["scope"] == 2 and again["set"] == 0  # the two undated remain in scope


def test_validity_regenerates_a_stale_default_and_preserves_verdicts(cli_db: Path) -> None:
    (pid,) = _seed("Picked up the keys on 2026-10-10.")
    stated = _seed(
        "Met her on 2026-10-11.",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_kind=ValidKind.EVENT,
        properties={VALID_FROM_BASIS_KEY: ValidFromBasis.STATED},
    )[0]
    assert runner.invoke(app, ["validity"]).exit_code == 0

    # Backdate the rule digest to an old value: the row is stale and regenerates.
    async def _stale() -> None:
        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async with session_scope() as session:
            row = await session.get(ParticleRow, pid)
            assert row is not None
            props = json.loads(row.properties_json or "{}")
            props[VALID_FROM_RULE_KEY] = "old"
            row.properties_json = json.dumps(props)
            await session.commit()

    asyncio.run(_stale())
    report = json.loads(runner.invoke(app, ["validity"]).output)
    assert report["scope"] == 1 and report["set"] == 1
    assert _read(pid)[2][VALID_FROM_RULE_KEY] != "old"
    # The stated value was never in scope and is untouched.
    vf, _, props = _read(stated)
    assert vf is not None and vf.date().isoformat() == "2026-01-01"
    assert props[VALID_FROM_BASIS_KEY] == ValidFromBasis.STATED


def test_validity_refuses_a_remote_engine(cli_db: Path) -> None:
    with patch("particles.api.client.get_backend", return_value=MagicMock(remote=True)):
        result = runner.invoke(app, ["validity"])
    assert result.exit_code == 2
    assert "one local store" in result.output


# ---------------------------------------------------------------------------
# particles particle date
# ---------------------------------------------------------------------------


def test_date_requires_from_and_kind(cli_db: Path) -> None:
    (pid,) = _seed("x")
    result = runner.invoke(app, ["particle", "date", pid[:8], "--reason", "r"])
    assert result.exit_code == 1
    assert "--from and --kind" in result.output


def test_date_rejects_bad_values(cli_db: Path) -> None:
    (pid,) = _seed("x")
    bad_date = runner.invoke(
        app, ["particle", "date", pid[:8], "--from", "nope", "--kind", "EVENT", "--reason", "r"]
    )
    assert bad_date.exit_code == 1 and "Invalid --from" in bad_date.output
    bad_kind = runner.invoke(
        app,
        ["particle", "date", pid[:8], "--from", "2026-10-10", "--kind", "SPAN", "--reason", "r"],
    )
    assert bad_kind.exit_code == 1 and "Unknown kind" in bad_kind.output
    blank = runner.invoke(
        app,
        ["particle", "date", pid[:8], "--from", "2026-10-10", "--kind", "EVENT", "--reason", " "],
    )
    assert blank.exit_code == 1


def test_date_writes_an_operator_verdict_the_pass_preserves(cli_db: Path) -> None:
    (pid,) = _seed("Picked up the keys on 2026-10-10.")
    result = runner.invoke(
        app,
        [
            "particle",
            "date",
            pid[:8],
            "--from",
            "2026-10-09",
            "--kind",
            "STATE",
            "--reason",
            "the keys came a day earlier",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "by operator verdict" in result.output
    vf, kind, props = _read(pid)
    assert vf is not None and vf.date().isoformat() == "2026-10-09" and kind == "STATE"
    assert props[VALID_FROM_BASIS_KEY] == ValidFromBasis.OPERATOR
    # The prose default would have said 2026-10-10 / EVENT; the verdict is pinned.
    report = json.loads(runner.invoke(app, ["validity"]).output)
    assert report["scope"] == 0
    assert _read(pid)[0] is not None and _read(pid)[0].date().isoformat() == "2026-10-09"  # type: ignore[union-attr]

    # The event log carries it.
    async def _events() -> list[str]:
        from particles.db import session_scope
        from particles.store.event_store import list_events

        async with session_scope() as session:
            return [e.event_type for e in await list_events(session, limit=10)]

    assert "PARTICLE_DATED" in asyncio.run(_events())


def test_date_clear(cli_db: Path) -> None:
    (pid,) = _seed("x", valid_from=datetime(2026, 10, 10, tzinfo=UTC), valid_kind=ValidKind.EVENT)
    result = runner.invoke(app, ["particle", "date", pid[:8], "--clear", "--reason", "wrong"])
    assert result.exit_code == 0, result.output
    assert _read(pid)[0] is None and _read(pid)[1] is None


def test_date_event_refused_on_a_bounded_claim(cli_db: Path) -> None:
    (pid,) = _seed("Bounded.", valid_until=datetime(2027, 1, 1, tzinfo=UTC))
    result = runner.invoke(
        app,
        ["particle", "date", pid[:8], "--from", "2026-10-10", "--kind", "EVENT", "--reason", "r"],
    )
    assert result.exit_code == 1
    assert "valid_until" in result.output
