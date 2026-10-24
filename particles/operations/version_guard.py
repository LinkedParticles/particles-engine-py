# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""SCHEMA_VERSION compatibility guard (amended).

Single guard helper called at the entry point of every operation that would
produce wrong results on a store it cannot read: ``query``, ``extract``,
``review``, ``reindex``, ``reconcile``. It refuses only on an *incompatible*
stamp, another major or one that does not parse; a different minor of the
same major is read and written normally (D2). Lint does NOT call
this: it reports every stamp other than the SDK's via
:func:`particles.operations.lint.coverage._report_schema_versions` so the
operator can diagnose before being forced to rebuild.

The guard asks one indexed question, whether any ACTIVE row lies outside the
SDK's major (:func:`particles.store.particle_store.has_active_schema_mismatch`),
which costs the same at any store size. Only when the answer is yes does it
count every version
(:func:`particles.store.particle_store.count_active_particles_by_schema_version`,
a GROUP BY over the ACTIVE rows). The count used to run on every call, and it
read the whole table: 0.47 s of every query at 1M rows (the query-latency
baseline, ``docs/benchmarks/query-latency.md``). A stamp of the SDK's major
that does not parse (``"1.x"``) passes the probe; it cannot be written,
because :class:`particles.core.schema.Particle` validates the stamp's form,
and lint reports one that arrived by direct SQL.

The other helpers here answer the minor-level questions the rest of the
Engine asks (D2, D5): which stamps in the store are older or newer
minors of the SDK's major, and whether one particle was written by a newer
minor. Each reads :data:`SCHEMA_VERSION` from this module at call time, which
is the seam tests patch to simulate an SDK behind or ahead of its store.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import SCHEMA_VERSION, SchemaVersionMismatchError
from particles.core.schema_version import SchemaCompatibility, schema_compatibility
from particles.store.particle_store import (
    count_active_particles_by_schema_version,
    has_active_schema_mismatch,
)


async def assert_store_schema_current(session: AsyncSession) -> None:
    """Raise :class:`SchemaVersionMismatchError` if the store holds an
    ACTIVE particle whose ``schema_version`` is incompatible with the SDK's
    :data:`particles.core.schema.SCHEMA_VERSION` (D1, D3).

    No-op when the store is empty or every ACTIVE stamp shares the SDK's major
    and parses. The raised exception carries the operator message, which
    differs for a store behind the SDK (rebuild) and one ahead of
    it (upgrade the SDK).
    """
    if not await has_active_schema_mismatch(session, SCHEMA_VERSION):
        return
    counts = await count_active_particles_by_schema_version(session)
    if any(not schema_compatibility(v, SCHEMA_VERSION).compatible for v in counts):
        raise SchemaVersionMismatchError(
            current_version=SCHEMA_VERSION,
            found_versions=counts,
        )


async def active_stamps(session: AsyncSession, kind: SchemaCompatibility) -> dict[str, int]:
    """The ACTIVE ``{stamp: count}`` entries that relate to the SDK as ``kind``."""
    counts = await count_active_particles_by_schema_version(session)
    return {v: c for v, c in counts.items() if schema_compatibility(v, SCHEMA_VERSION) is kind}


def written_by_newer_minor(schema_version: str) -> bool:
    """True when a record at this stamp comes from a newer minor of the SDK's major.

    The test applied before a housekeeping replacement: an SDK older than a
    record never retires it in favour of a poorer copy of itself.
    """
    return schema_compatibility(schema_version, SCHEMA_VERSION) is SchemaCompatibility.NEWER_MINOR
