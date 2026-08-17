"""SQLAlchemy ORM and repository for extractor records (Extension A)."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Boolean, Float, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.config import get_config, register_reset_hook
from particles.core.schema import ApplicabilityClause, ExtractorCalibration, ExtractorRecord
from particles.db import Base
from particles.store.lens_store import get_adopted_lens_extractor_weights


class ExtractorRow(Base):
    __tablename__ = "extractor_records"

    extractor_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    applicability_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    trust_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.7)
    # last persisted conformance verdict — True iff the last
    # `extractor conform` run produced particles AND a REQUIRED field genuinely
    # fell short (an *evaluable* failure). None = never evaluated / no fixture
    # (unknown, never a failure); False = evaluated and passed. Read-side only:
    # the trust cap clamps the *effective* weight when this is True, never the
    # stored ``trust_weight``. Not part of ``ExtractorRecord`` (store-internal).
    conformance_required_failure: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=None
    )
    registered_by: Mapped[str] = mapped_column(Text, nullable=False)
    registered_at: Mapped[datetime] = mapped_column(
        String,
        nullable=False,  # stored as ISO string for SQLite compat
    )
    # Calibration moved out of this row to ``extractor_calibrations``:
    # several models' calibrations coexist there, keyed by provider_model. This
    # registry row no longer carries one. ``ExtractorRecord.calibration`` stays a
    # schema field but is no longer sourced here — read it via ``get_calibration``.

    def to_model(self) -> ExtractorRecord:
        clauses = [ApplicabilityClause(**c) for c in json.loads(self.applicability_json)]
        return ExtractorRecord(
            extractor_id=self.extractor_id,
            name=self.name,
            version=self.version,
            applicability=clauses,
            trust_weight=self.trust_weight,
            registered_by=self.registered_by,
            registered_at=datetime.fromisoformat(str(self.registered_at)),
            calibration=None,
        )

    @classmethod
    def from_model(cls, r: ExtractorRecord) -> ExtractorRow:
        return cls(
            extractor_id=r.extractor_id,
            name=r.name,
            version=r.version,
            applicability_json=json.dumps([c.model_dump() for c in r.applicability]),
            trust_weight=r.trust_weight,
            registered_by=r.registered_by,
            registered_at=r.registered_at.isoformat(),
        )


# A calibration fitted before the provider_model key existed is keyed
# under the historical default model so it still applies under that pairing and
# is correctly missed by any swap away from it.
LEGACY_PROVIDER_MODEL = "anthropic:claude-sonnet-4-6"


class ExtractorCalibrationRow(Base):
    """One :class:`ExtractorCalibration` per ``(extractor_id, provider_model)``.

    Replaces the single ``ExtractorRow.calibration_json`` column so several
    models' calibrations coexist. The pipeline selects the row matching the
    configured extraction model, so a provider swap keeps
    ``CALIBRATED_BENCHMARK`` without re-benchmarking on every switch. One active
    record per pairing (the composite PK enforces it); a re-fit replaces only
    that pairing's row.
    """

    __tablename__ = "extractor_calibrations"

    extractor_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider_model: Mapped[str] = mapped_column(String, primary_key=True)
    calibration_json: Mapped[str] = mapped_column(Text, nullable=False)


async def upsert_extractor_record(session: AsyncSession, record: ExtractorRecord) -> bool:
    """Insert or update an extractor registry record. Returns True if written.

    trust_weight is only written on INSERT — operator overrides are preserved.
    version and applicability_json are updated when the version changes.
    Calibration is **not** stored here since — it lives per
    provider_model in ``extractor_calibrations``; use :func:`upsert_calibration`.
    """
    row = await session.get(ExtractorRow, record.extractor_id)
    if row is None:
        session.add(ExtractorRow.from_model(record))
        await session.flush()
        return True
    wrote = False
    if row.version != record.version:
        row.version = record.version
        row.applicability_json = json.dumps([c.model_dump() for c in record.applicability])
        wrote = True
    if wrote:
        await session.flush()
    return wrote


async def get_extractor_record(session: AsyncSession, extractor_id: str) -> ExtractorRecord | None:
    """Return the registered extractor registry record, or None if not found.

    The returned record's ``calibration`` is always ``None`` since —
    calibrations are per-provider_model; fetch one with :func:`get_calibration`.
    """
    row = await session.get(ExtractorRow, extractor_id)
    return row.to_model() if row is not None else None


async def upsert_calibration(
    session: AsyncSession, extractor_id: str, calibration: ExtractorCalibration
) -> None:
    """Store ``calibration`` for ``(extractor_id, its provider_model)``.

    Keyed by ``calibration.provider_model`` (falling back to
    :data:`LEGACY_PROVIDER_MODEL` when unset), so re-fitting one pairing leaves
    other models' calibrations intact. A re-fit of the same pairing replaces its
    one row.
    """
    key = calibration.provider_model or LEGACY_PROVIDER_MODEL
    payload = calibration.model_dump_json()
    row = await session.get(ExtractorCalibrationRow, (extractor_id, key))
    if row is None:
        session.add(
            ExtractorCalibrationRow(
                extractor_id=extractor_id, provider_model=key, calibration_json=payload
            )
        )
    else:
        row.calibration_json = payload
    await session.flush()


async def get_calibration(
    session: AsyncSession, extractor_id: str, provider_model: str
) -> ExtractorCalibration | None:
    """Return the calibration for ``(extractor_id, provider_model)``, or None.

    The pipeline calls this with the currently-configured extraction
    provider_model, so a loaded record matches by construction (no separate
    match guard needed).
    """
    row = await session.get(ExtractorCalibrationRow, (extractor_id, provider_model))
    return ExtractorCalibration.model_validate_json(row.calibration_json) if row else None


async def delete_calibration(
    session: AsyncSession, extractor_id: str, provider_model: str
) -> ExtractorCalibration | None:
    """Remove one ``(extractor_id, provider_model)`` calibration.

    Returns the record that was removed, or ``None`` if the pairing had none.
    The counterpart to :func:`upsert_calibration`: before a stored
    calibration could only be *replaced*, by re-fitting under the same pairing,
    so retiring one fitted against a model no longer reachable meant
    re-provisioning that model. Other pairings are untouched.

    Removal returns the pairing to ``calibration_source=EXTRACTOR_DIRECT`` —
    the documented fallback the pipeline already uses for an uncalibrated
    pairing — for particles minted *after* it. Particles already stored keep
    the confidence they were minted with, so this is never a
    retroactive edit; an operator who wants the change applied to existing
    particles runs ``particles reindex --extractor-id <id>``.
    """
    row = await session.get(ExtractorCalibrationRow, (extractor_id, provider_model))
    if row is None:
        return None
    record = ExtractorCalibration.model_validate_json(row.calibration_json)
    await session.delete(row)
    await session.flush()
    return record


async def get_calibrations(session: AsyncSession, extractor_id: str) -> list[ExtractorCalibration]:
    """Return every stored calibration for ``extractor_id``, one per pairing."""
    rows = (
        (
            await session.execute(
                select(ExtractorCalibrationRow).where(
                    ExtractorCalibrationRow.extractor_id == extractor_id
                )
            )
        )
        .scalars()
        .all()
    )
    return [ExtractorCalibration.model_validate_json(r.calibration_json) for r in rows]


async def set_trust_weight(session: AsyncSession, extractor_id: str, trust_weight: float) -> bool:
    """Update trust_weight for an extractor. Returns False if not found."""
    row = await session.get(ExtractorRow, extractor_id)
    if row is None:
        return False
    row.trust_weight = trust_weight
    await session.flush()
    return True


async def set_conformance_status(
    session: AsyncSession, extractor_id: str, has_evaluable_failure: bool
) -> bool:
    """Persist the extractor's evaluable-REQUIRED-failure verdict.

    The read-side trust-cap input: ``True`` when the last conformance run
    produced particles **and** a REQUIRED field genuinely fell short. The
    stored ``trust_weight`` is never touched — the cap clamps the *effective*
    weight at read time. Returns ``False`` if the extractor has no registry row
    (register it via the extraction pipeline first); the cap only ever applies
    to registered extractors, so an unregistered one is correctly a no-op.
    """
    row = await session.get(ExtractorRow, extractor_id)
    if row is None:
        return False
    row.conformance_required_failure = has_evaluable_failure
    await session.flush()
    return True


async def get_all_records(session: AsyncSession) -> list[ExtractorRecord]:
    rows = (await session.execute(select(ExtractorRow))).scalars().all()
    return [r.to_model() for r in rows]


async def get_trust_weight_map(session: AsyncSession) -> dict[str, float]:
    """Return {extractor_id: trust_weight} for all registered extractors.

    Adopted trust-lens extractor-weight overrides compose in by
    **minimum** — most-skeptical-wins. Every call site of this function seeds
    the per-run trust cache, so the lens overlay applies wherever effective
    confidence is computed (query, federation under the viewer's session,
    exporters); the raw stored weights remain visible via ``get_all_records``.

    When the conformance trust cap is enabled, an extractor whose last
    conformance run showed an *evaluable* REQUIRED failure
    (``conformance_required_failure is True``) and is not exempt has its
    effective weight clamped to ``cap_value`` — another ``min`` demotion, so it
    composes order-independently with the lens overlay. ``None`` / ``False``
    (unknown / passed) never clamps. Disabled by default: the map is unchanged.
    """
    rows = (await session.execute(select(ExtractorRow))).scalars().all()
    weights = {r.extractor_id: r.trust_weight for r in rows}
    for extractor_id, lens_weight in (await get_adopted_lens_extractor_weights(session)).items():
        current = weights.get(extractor_id)
        weights[extractor_id] = lens_weight if current is None else min(current, lens_weight)

    cap_cfg = get_config().conformance.trust_cap
    if cap_cfg.enabled:
        exempt = set(cap_cfg.exempt)
        for r in rows:
            if r.conformance_required_failure and r.extractor_id not in exempt:
                current = weights.get(r.extractor_id, r.trust_weight)
                weights[r.extractor_id] = min(current, cap_cfg.cap_value)
    return weights


#: Process-global snapshot of the store's extractor trust weights, warmed by
#: ``populate_trust_cache`` at each read-surface entry point (query, digest,
#: every exporter) and read per-particle during scoring. It is *derived from a
#: store*, so it must not outlive the store it was read from — see the reset
#: hook at the bottom of this module.
_trust_cache: dict[str, float] | None = None


def get_cached_trust_weight(extractor_id: str, default: float = 1.0) -> float:
    """Return trust_weight from the in-process cache (1.0 if cache not loaded)."""
    if _trust_cache is None:
        return default
    return _trust_cache.get(extractor_id, default)


def populate_trust_cache(trust_map: dict[str, float]) -> None:
    """Load the in-process trust cache from a DB-fetched map."""
    global _trust_cache
    _trust_cache = dict(trust_map)


def invalidate_trust_cache() -> None:
    global _trust_cache
    _trust_cache = None


# Same inverted config↔Engine coupling ``db.reset_engine`` uses:
# the trust cache is a snapshot of *one store's* extractor weights, and
# ``reset_config()`` is exactly the point at which the cached engines — hence
# the store behind that snapshot — are discarded. Without this, a stale cache
# survives the store swap and silently rescales every effective confidence
# computed on a path that does not re-warm it (``score_effective_confidence``
# with ``populate_cache=False``, which is the documented in-query default).
#
# Concretely, this was a flaky pre-commit gate: under ``-n auto`` xdist could
# schedule ``test_extractor_registry.py`` (which populates the cache with
# ``general-extractor: 0.70``) into the same worker, ahead of
# ``test_utility_policy.py``, and three utility tests then scored 0.99 × 0.70 =
# 0.693 instead of the untouched 0.99. Re-running turned it green, which is the
# worst possible failure mode for a gate.
register_reset_hook(invalidate_trust_cache)
