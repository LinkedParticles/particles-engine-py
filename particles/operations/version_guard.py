"""SCHEMA_VERSION mismatch guard.

Single guard helper called at the entry point of every operation that would
produce wrong results on a store with mixed-schema particles: ``query``,
``extract``, ``review``, ``reindex``. Lint does NOT call this — it reports
mismatches as `SCHEMA_VERSION_MISMATCH` findings via
:func:`particles.operations.lint.coverage._report_schema_versions` so the
operator can diagnose before being forced to rebuild.

The guard reuses the existing
:func:`particles.store.particle_store.count_active_particles_by_schema_version`
helper — a single GROUP BY SELECT, microseconds on SQLite stores of typical
single-user PKM scale.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import SCHEMA_VERSION, SchemaVersionMismatchError
from particles.store.particle_store import count_active_particles_by_schema_version


async def assert_store_schema_current(session: AsyncSession) -> None:
    """Raise :class:`SchemaVersionMismatchError` if any ACTIVE particle in the
    store carries a ``schema_version`` other than the SDK's current
    :data:`particles.core.schema.SCHEMA_VERSION`.

    No-op when the store is empty or all ACTIVE particles match.
    the operator's remediation is `particles db init --force` + re-extract;
    the raised exception carries the canonical operator message.
    """
    counts = await count_active_particles_by_schema_version(session)
    mismatched = {v: c for v, c in counts.items() if v != SCHEMA_VERSION}
    if mismatched:
        raise SchemaVersionMismatchError(
            current_version=SCHEMA_VERSION,
            found_versions=counts,
        )
