# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""SCHEMA_VERSION mismatch guard.

Single guard helper called at the entry point of every operation that would
produce wrong results on a store with mixed-schema particles: ``query``,
``extract``, ``review``, ``reindex``. Lint does NOT call this — it reports
mismatches as `SCHEMA_VERSION_MISMATCH` findings via
:func:`particles.operations.lint.coverage._report_schema_versions` so the
operator can diagnose before being forced to rebuild.

The guard asks one indexed question, whether any ACTIVE row is on another
version (:func:`particles.store.particle_store.has_active_schema_mismatch`),
which costs the same at any store size. Only when the answer is yes does it
count every version for the error message
(:func:`particles.store.particle_store.count_active_particles_by_schema_version`,
a GROUP BY over the ACTIVE rows). The count used to run on every call, and it
read the whole table: 0.47 s of every query at 1M rows (the query-latency
baseline, ``docs/benchmarks/query-latency.md``).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import SCHEMA_VERSION, SchemaVersionMismatchError
from particles.store.particle_store import (
    count_active_particles_by_schema_version,
    has_active_schema_mismatch,
)


async def assert_store_schema_current(session: AsyncSession) -> None:
    """Raise :class:`SchemaVersionMismatchError` if any ACTIVE particle in the
    store carries a ``schema_version`` other than the SDK's current
    :data:`particles.core.schema.SCHEMA_VERSION`.

    No-op when the store is empty or all ACTIVE particles match.
    the operator's remediation is `particles db init --force` + re-extract;
    the raised exception carries the canonical operator message.
    """
    if not await has_active_schema_mismatch(session, SCHEMA_VERSION):
        return
    counts = await count_active_particles_by_schema_version(session)
    mismatched = {v: c for v, c in counts.items() if v != SCHEMA_VERSION}
    if mismatched:
        raise SchemaVersionMismatchError(
            current_version=SCHEMA_VERSION,
            found_versions=counts,
        )
