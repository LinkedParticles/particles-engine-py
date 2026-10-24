# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Same-major ``schema_version`` compatibility.

The pure algebra (D1, D6's lower-of rule), the guard in both directions with
its index-only probe (D3), the L-STR-07 classification (D4), and the two
mechanical halves of the release ordering (D9).

An SDK behind or ahead of its store is simulated by patching the reader's
``SCHEMA_VERSION`` on the module that reads it, and inserting rows at real
stamps: this SDK is at ``1.0.0``, so no other minor exists outside a test.
"""

from __future__ import annotations

import re
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    SCHEMA_VERSION,
    TOLERANT_GUARD_SINCE,
    Confidence,
    Particle,
    SchemaVersionMismatchError,
    UncertaintyNature,
)
from particles.core.schema_version import (
    SchemaCompatibility,
    lower_schema_version,
    major_prefix_range,
    parse_schema_version,
    schema_compatibility,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.lint.coverage import _report_schema_versions
from particles.operations.version_guard import (
    active_stamps,
    assert_store_schema_current,
    written_by_newer_minor,
)
from particles.store.particle_store import has_active_schema_mismatch, insert_particle
from tests._query_plan import plans_of
from tests._upstream import REPO_ROOT, upstream_only

C = SchemaCompatibility


def _particle(stamp: str) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content="A test claim about a thing.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        schema_version=stamp,
    )


async def _store(session: AsyncSession, *stamps: str) -> None:
    for stamp in stamps:
        await insert_particle(session, _particle(stamp))
    await session.commit()


@pytest.fixture
def sdk_at_1_2(monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the guard, lint and helpers as an SDK at ``1.2.0``."""
    monkeypatch.setattr("particles.operations.version_guard.SCHEMA_VERSION", "1.2.0")
    monkeypatch.setattr("particles.operations.lint.coverage.SCHEMA_VERSION", "1.2.0")
    return "1.2.0"


# ---------------------------------------------------------------------------
# D1: the algebra
# ---------------------------------------------------------------------------


class TestCompatibility:
    @pytest.mark.parametrize(
        ("stored", "current", "expected"),
        [
            ("1.2.0", "1.2.0", C.EQUAL),
            ("1.2.7", "1.2.0", C.EQUAL),  # PATCH is never read
            ("1.1.0", "1.2.0", C.OLDER_MINOR),
            ("1.3.0", "1.2.0", C.NEWER_MINOR),
            # Ordered as integers, never as text.
            ("1.10.0", "1.9.0", C.NEWER_MINOR),
            ("1.9.0", "1.10.0", C.OLDER_MINOR),
            ("10.0.0", "1.0.0", C.NEWER_MAJOR),
            ("2.0.0", "1.4.0", C.NEWER_MAJOR),
            ("0.3.0", "1.0.0", C.OLDER_MAJOR),  # refusal is kept
            ("1.x", "1.0.0", C.MALFORMED),
            ("1", "1.0.0", C.MALFORMED),
            ("1.0.0-rc1", "1.0.0", C.MALFORMED),
            ("", "1.0.0", C.MALFORMED),
        ],
    )
    def test_table(self, stored: str, current: str, expected: SchemaCompatibility) -> None:
        assert schema_compatibility(stored, current) is expected

    def test_only_same_major_kinds_are_compatible(self) -> None:
        assert {k for k in C if k.compatible} == {C.EQUAL, C.OLDER_MINOR, C.NEWER_MINOR}

    @pytest.mark.parametrize("current", ["0.3.0", "1.x", ""])
    def test_the_readers_own_stamp_must_be_a_real_one(self, current: str) -> None:
        with pytest.raises(ValueError):
            schema_compatibility("1.0.0", current)

    def test_parse(self) -> None:
        assert parse_schema_version("1.10.3") == (1, 10, 3)
        assert parse_schema_version("1.10") is None

    def test_major_prefix_range_holds_exactly_the_major(self) -> None:
        lower, upper = major_prefix_range("1.4.2")
        assert (lower, upper) == ("1.", "1/")
        inside = ["1.0.0", "1.9.0", "1.10.0", "1.x"]
        outside = ["0.3.0", "1", "10.0.0", "2.0.0"]
        assert all(lower <= s < upper for s in inside)
        assert not any(lower <= s < upper for s in outside)


class TestLowerOf:
    @pytest.mark.parametrize(
        ("record", "reader", "stamped"),
        [
            ("1.3.0", "1.2.0", "1.2.0"),  # an older reader cannot carry 1.3's fields
            ("1.1.0", "1.2.0", "1.1.0"),  # a newer reader keeps the record's own
            ("1.2.4", "1.2.0", "1.2.4"),  # a tie on MAJOR.MINOR keeps the record's PATCH
            ("1.2.0", "1.2.9", "1.2.0"),
        ],
    )
    def test_lower_compares_major_minor_and_keeps_the_record_on_a_tie(
        self, record: str, reader: str, stamped: str
    ) -> None:
        assert lower_schema_version(record, reader) == stamped


class TestStampForm:
    @pytest.mark.parametrize("stamp", ["1.x", "1", "1.0", "1.0.0-rc1", "v1.0.0"])
    def test_a_malformed_stamp_cannot_be_written(self, stamp: str) -> None:
        # The guard's probe sees only the major prefix, so the model closes
        # the one malformed shape it cannot see (D3).
        with pytest.raises(ValidationError):
            _particle(stamp)


# ---------------------------------------------------------------------------
# D3: the guard and its message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_a_store_behind_is_told_to_rebuild(self) -> None:
        msg = str(
            SchemaVersionMismatchError(
                current_version="1.0.0", found_versions={"1.0.0": 3, "0.3.0": 2}
            )
        )
        assert "particles db init --force" in msg
        assert "2 at v0.3.0" in msg

    def test_a_store_ahead_is_never_told_to_rebuild(self) -> None:
        msg = str(
            SchemaVersionMismatchError(
                current_version="1.0.0", found_versions={"1.0.0": 3, "2.1.0": 4}
            )
        )
        assert "db init --force" not in msg
        assert "Upgrade this SDK" in msg
        assert "v2.x" in msg

    def test_both_directions_get_both_remedies(self) -> None:
        msg = str(
            SchemaVersionMismatchError(
                current_version="1.0.0", found_versions={"0.3.0": 1, "2.0.0": 1}
            )
        )
        assert "db init --force" in msg
        assert "Upgrade this SDK" in msg

    def test_compatible_stamps_are_not_listed(self) -> None:
        msg = str(
            SchemaVersionMismatchError(
                current_version="1.2.0", found_versions={"1.1.0": 5, "1.3.0": 6, "2.0.0": 1}
            )
        )
        assert "v1.1.0" not in msg
        assert "v1.3.0" not in msg
        assert "1 ACTIVE particles" in msg


class TestGuard:
    @pytest.mark.asyncio
    async def test_other_minors_in_either_direction_do_not_refuse(
        self, db_session: AsyncSession, sdk_at_1_2: str
    ) -> None:
        await _store(db_session, "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.10.0")
        await assert_store_schema_current(db_session)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stamp", ["0.3.0", "2.0.0", "10.0.0"])
    async def test_another_major_refuses(
        self, db_session: AsyncSession, sdk_at_1_2: str, stamp: str
    ) -> None:
        await _store(db_session, "1.3.0", stamp)
        with pytest.raises(SchemaVersionMismatchError) as exc_info:
            await assert_store_schema_current(db_session)
        assert exc_info.value.current_version == "1.2.0"
        assert exc_info.value.found_versions == {"1.3.0": 1, stamp: 1}

    @pytest.mark.asyncio
    async def test_the_probe_is_two_covering_index_seeks(self, db_session: AsyncSession) -> None:
        await _store(db_session, SCHEMA_VERSION)
        ((_, plan),) = await plans_of(
            db_session, lambda: has_active_schema_mismatch(db_session, SCHEMA_VERSION)
        )
        searches = [d for d in plan if d.startswith("SEARCH")]
        assert len(searches) == 2
        assert all("COVERING INDEX ix_particles_status_schema_version" in d for d in searches)
        assert not any(d.startswith("SCAN particles") for d in plan)

    @pytest.mark.asyncio
    async def test_a_compatible_store_costs_one_probe(
        self, db_session: AsyncSession, sdk_at_1_2: str
    ) -> None:
        # No GROUP BY on the happy path: the census runs only to refuse.
        await _store(db_session, "1.1.0", "1.3.0")
        plans = await plans_of(db_session, lambda: assert_store_schema_current(db_session))
        assert len(plans) == 1

    @pytest.mark.asyncio
    async def test_minor_helpers(self, db_session: AsyncSession, sdk_at_1_2: str) -> None:
        await _store(db_session, "1.1.0", "1.1.0", "1.2.0", "1.3.0")
        assert await active_stamps(db_session, C.OLDER_MINOR) == {"1.1.0": 2}
        assert await active_stamps(db_session, C.NEWER_MINOR) == {"1.3.0": 1}
        assert written_by_newer_minor("1.3.0")
        assert not written_by_newer_minor("1.2.5")
        assert not written_by_newer_minor("2.0.0")


# ---------------------------------------------------------------------------
# D4: L-STR-07
# ---------------------------------------------------------------------------


class TestLint:
    @pytest.mark.asyncio
    async def test_each_stamp_is_classified(
        self, db_session: AsyncSession, sdk_at_1_2: str
    ) -> None:
        await _store(db_session, "1.2.0", "1.2.3", "1.1.0", "1.3.0", "0.3.0", "2.0.0")
        findings = await _report_schema_versions(db_session)
        by_type = [(f.finding_type, f.severity, f.detail.split("'")[1]) for f in findings]
        assert by_type == [
            ("SCHEMA_VERSION_MISMATCH", "WARNING", "0.3.0"),
            ("SCHEMA_VERSION_OLDER_MINOR", "INFO", "1.1.0"),
            ("SCHEMA_VERSION_NEWER_MINOR", "INFO", "1.3.0"),
            ("SCHEMA_VERSION_MISMATCH", "WARNING", "2.0.0"),
        ]

    @pytest.mark.asyncio
    async def test_the_two_mismatches_name_opposite_remedies(
        self, db_session: AsyncSession, sdk_at_1_2: str
    ) -> None:
        await _store(db_session, "0.3.0", "2.0.0")
        behind, ahead = await _report_schema_versions(db_session)
        assert "db init --force" in (behind.recommended_action or "")
        assert "db init --force" not in (ahead.recommended_action or "")


# ---------------------------------------------------------------------------
# D9: the first bump waits for the tolerant guard
# ---------------------------------------------------------------------------


def _project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def _release_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class TestReleaseOrdering:
    def test_a_bump_never_rides_the_guards_own_release(self) -> None:
        # Any SDK published before TOLERANT_GUARD_SINCE refuses a store holding
        # a stamp other than 1.0.0, so a bump past it must come strictly later.
        assert parse_schema_version(TOLERANT_GUARD_SINCE) is not None
        if SCHEMA_VERSION != "1.0.0":
            assert _release_tuple(TOLERANT_GUARD_SINCE) < _release_tuple(_project_version())

    @upstream_only
    def test_the_constant_names_the_release_that_ships_the_guard(self) -> None:
        # The version is minted before merge and may be re-bumped then, so the
        # constant is tied to the changelog section that announces the guard.
        changelog = Path(REPO_ROOT / "CHANGELOG.md").read_text()
        sections = re.split(r"^## \[", changelog, flags=re.MULTILINE)
        heads = {s.split("]", 1)[0]: s for s in sections[1:]}
        if TOLERANT_GUARD_SINCE in heads:
            body = heads[TOLERANT_GUARD_SINCE]
        else:
            # Before the cut-changelog hook runs, the line is still Unreleased.
            assert _project_version() == TOLERANT_GUARD_SINCE
            body = heads["Unreleased"]
        assert "same-major `schema_version` compatibility" in body.lower()
