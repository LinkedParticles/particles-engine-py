"""SQLAlchemy ORM and repository for trust lenses.

A ``TrustLensDefinition`` arrives as a deposited corpus artefact (the taxonomy pattern); the ``TrustLensExtractor`` hands the parsed model
to :func:`materialise_lens` via the sink seam registered at the
bottom of this module. Adoption is **store state** — one row per adopted lens
name — so a store's viewpoint survives config reloads and is visible to
federation. Composition into the query-time ``TrustPolicy`` lives in
``particles/operations/query/source_trust.py``; the extractor-weight overlay
lives in ``extractor_store.get_trust_weight_map``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.core.schema import (
    TrustLensDecayRule,
    TrustLensDefinition,
    TrustLensStatement,
    TrustLensUrlRule,
    TrustLensUtilityRule,
)
from particles.db import Base
from particles.store.event_store import OperatorEventType, record_event

log = logging.getLogger(__name__)


class TrustLensRow(Base):
    """One materialised lens (latest version per name)."""

    __tablename__ = "trust_lenses"

    lens_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    publisher: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    corpus_entry_id: Mapped[str | None] = mapped_column(String, nullable=True)
    materialised_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TrustLensEntryRow(Base):
    """One policy entry of a lens — statement, URL rule, or extractor weight.

    Polymorphic by ``entry_kind`` with nullable per-kind columns, mirroring
    the ``SourceTrustRow`` style.
    """

    __tablename__ = "trust_lens_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    lens_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    entry_kind: Mapped[str] = mapped_column(
        String, nullable=False
    )  # "statement" | "domain" | "url_pattern" | "extractor_weight"
    #    | "decay_source_type" | "decay_url_pattern"
    # statement fields
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    source_type: Mapped[str | None] = mapped_column(String, nullable=True)
    trust_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    basis: Mapped[str | None] = mapped_column(Text, nullable=True)
    # URL-rule fields (shape); ``pattern`` is reused by decay url rules
    pattern: Mapped[str | None] = mapped_column(String, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    modifier: Mapped[float | None] = mapped_column(Float, nullable=True)
    # extractor-weight fields
    extractor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    # decay-rule fields; ``pattern`` holds the source_type / url regex
    half_life_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    floor: Mapped[float | None] = mapped_column(Float, nullable=True)
    # utility-rule fields (composition). Reuses
    # ``pattern`` (None for scope=default); ``rank_lift`` is the λ that
    # replaced the superseded ``weight`` / ``floor`` / ``cap`` triple.
    half_life_uses_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_lift: Mapped[float | None] = mapped_column(Float, nullable=True)


class LensAdoptionRow(Base):
    """One adopted lens, keyed by name (stable across lens versions)."""

    __tablename__ = "trust_lens_adoptions"

    lens_name: Mapped[str] = mapped_column(String, primary_key=True)
    adopted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    adopted_by: Mapped[str] = mapped_column(String, nullable=False)


def _entry_rows(lens: TrustLensDefinition) -> list[TrustLensEntryRow]:
    rows: list[TrustLensEntryRow] = []
    for s in lens.statements:
        rows.append(
            TrustLensEntryRow(
                lens_id=lens.lens_id,
                entry_kind="statement",
                domain=s.domain,
                source_type=s.source_type,
                trust_rank=s.trust_rank,
                basis=s.basis,
            )
        )
    for r in lens.url_rules:
        rows.append(
            TrustLensEntryRow(
                lens_id=lens.lens_id,
                entry_kind=r.scope,
                pattern=r.pattern,
                score=r.score,
                modifier=r.modifier,
            )
        )
    for extractor_id, weight in lens.extractor_weights.items():
        rows.append(
            TrustLensEntryRow(
                lens_id=lens.lens_id,
                entry_kind="extractor_weight",
                extractor_id=extractor_id,
                weight=weight,
            )
        )
    for d in lens.decay_rules:
        rows.append(
            TrustLensEntryRow(
                lens_id=lens.lens_id,
                entry_kind=f"decay_{d.scope}",
                pattern=d.pattern,
                half_life_days=d.half_life_days,
                floor=d.floor,
            )
        )
    for u in lens.utility_rules:
        rows.append(
            TrustLensEntryRow(
                lens_id=lens.lens_id,
                entry_kind=f"utility_{u.scope}",
                pattern=u.pattern,
                half_life_uses_days=u.half_life_uses_days,
                rank_lift=u.rank_lift,
            )
        )
    return rows


def _to_definition(row: TrustLensRow, entries: list[TrustLensEntryRow]) -> TrustLensDefinition:
    statements: list[TrustLensStatement] = []
    url_rules: list[TrustLensUrlRule] = []
    extractor_weights: dict[str, float] = {}
    decay_rules: list[TrustLensDecayRule] = []
    utility_rules: list[TrustLensUtilityRule] = []
    for e in entries:
        if e.entry_kind == "statement" and e.domain and e.source_type and e.trust_rank is not None:
            statements.append(
                TrustLensStatement(
                    domain=e.domain,
                    source_type=e.source_type,
                    trust_rank=e.trust_rank,
                    basis=e.basis,
                )
            )
        elif e.entry_kind in ("domain", "url_pattern") and e.pattern:
            url_rules.append(
                TrustLensUrlRule(
                    scope=e.entry_kind,  # type: ignore[arg-type]
                    pattern=e.pattern,
                    score=e.score,
                    modifier=e.modifier,
                )
            )
        elif e.entry_kind == "extractor_weight" and e.extractor_id and e.weight is not None:
            extractor_weights[e.extractor_id] = e.weight
        elif (
            e.entry_kind in ("decay_source_type", "decay_url_pattern")
            and e.pattern
            and e.half_life_days is not None
            and e.floor is not None
        ):
            decay_rules.append(
                TrustLensDecayRule(
                    scope="source_type" if e.entry_kind == "decay_source_type" else "url_pattern",
                    pattern=e.pattern,
                    half_life_days=e.half_life_days,
                    floor=e.floor,
                )
            )
        elif (
            e.entry_kind in ("utility_default", "utility_source_type", "utility_url_pattern")
            and e.half_life_uses_days is not None
            # compat: a row materialised under the superseded
            # weight/floor/cap vocabulary carries no `rank_lift`, so it is
            # skipped — the lens reads as silent about utility and the store's
            # local `utility` config applies. Re-deposit the lens to restore it.
            and e.rank_lift is not None
        ):
            utility_rules.append(
                TrustLensUtilityRule(
                    scope=e.entry_kind[len("utility_") :],  # type: ignore[arg-type]
                    pattern=e.pattern,
                    half_life_uses_days=e.half_life_uses_days,
                    rank_lift=e.rank_lift,
                )
            )
    return TrustLensDefinition(
        lens_id=row.lens_id,
        name=row.name,
        version=row.version,
        publisher=row.publisher,
        description=row.description,
        statements=statements,
        url_rules=url_rules,
        extractor_weights=extractor_weights,
        decay_rules=decay_rules,
        utility_rules=utility_rules,
        corpus_entry_id=row.corpus_entry_id,
    )


async def materialise_lens(session: AsyncSession, lens: TrustLensDefinition) -> str | None:
    """Materialise a parsed lens; replace a lower-versioned one of the same name.

    Returns a human-readable rejection reason for a lower-or-equal version
    (the extractor surfaces it as a quality note), or ``None`` on success.
    The corpus entry remains the immutable record of every version deposited.
    """
    existing = (
        await session.execute(select(TrustLensRow).where(TrustLensRow.name == lens.name))
    ).scalar_one_or_none()
    if existing is not None:
        if lens.version <= existing.version:
            return (
                f"TrustLensDefinition {lens.name!r} v{lens.version} not materialised: "
                f"v{existing.version} is already current (versions are monotonic)."
            )
        await session.execute(
            delete(TrustLensEntryRow).where(TrustLensEntryRow.lens_id == existing.lens_id)
        )
        await session.delete(existing)
        await session.flush()

    session.add(
        TrustLensRow(
            lens_id=lens.lens_id,
            name=lens.name,
            version=lens.version,
            publisher=lens.publisher,
            description=lens.description,
            corpus_entry_id=lens.corpus_entry_id,
            materialised_at=datetime.now(UTC),
        )
    )
    session.add_all(_entry_rows(lens))
    await session.flush()
    # A re-deposit of an *adopted* lens changes effective policy, so the
    # materialisation itself is event-worthy.
    await record_event(
        session,
        actor="trust-lens-extractor",
        event_type=OperatorEventType.TRUST_CHANGED,
        reason=None,
        payload={
            "kind": "lens_materialised",
            "name": lens.name,
            "version": lens.version,
            "replaced_version": existing.version if existing is not None else None,
        },
    )
    return None


async def list_lenses(session: AsyncSession) -> list[tuple[TrustLensRow, bool]]:
    """Return every materialised lens row with its adoption flag."""
    rows = (await session.execute(select(TrustLensRow).order_by(TrustLensRow.name))).scalars().all()
    adopted = {
        r.lens_name for r in (await session.execute(select(LensAdoptionRow))).scalars().all()
    }
    return [(row, row.name in adopted) for row in rows]


async def get_lens(session: AsyncSession, name: str) -> TrustLensDefinition | None:
    """Return the materialised lens as its Pydantic model, or None."""
    row = (
        await session.execute(select(TrustLensRow).where(TrustLensRow.name == name))
    ).scalar_one_or_none()
    if row is None:
        return None
    entries = (
        (
            await session.execute(
                select(TrustLensEntryRow).where(TrustLensEntryRow.lens_id == row.lens_id)
            )
        )
        .scalars()
        .all()
    )
    return _to_definition(row, list(entries))


async def adopt_lens(session: AsyncSession, name: str, actor: str = "operator") -> None:
    """Adopt a materialised lens by name. Raises ValueError if unknown / already adopted."""
    lens = (
        await session.execute(select(TrustLensRow).where(TrustLensRow.name == name))
    ).scalar_one_or_none()
    if lens is None:
        raise ValueError(f"No materialised lens named {name!r} — deposit its definition first.")
    existing = await session.get(LensAdoptionRow, name)
    if existing is not None:
        raise ValueError(f"Lens {name!r} is already adopted.")
    session.add(LensAdoptionRow(lens_name=name, adopted_at=datetime.now(UTC), adopted_by=actor))
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.TRUST_CHANGED,
        reason=None,
        payload={"kind": "lens_adopt", "name": name, "version": lens.version},
    )


async def unadopt_lens(session: AsyncSession, name: str, actor: str = "operator") -> None:
    """Remove a lens adoption. Raises ValueError if not adopted."""
    existing = await session.get(LensAdoptionRow, name)
    if existing is None:
        raise ValueError(f"Lens {name!r} is not adopted.")
    await session.delete(existing)
    await session.flush()
    await record_event(
        session,
        actor=actor,
        event_type=OperatorEventType.TRUST_CHANGED,
        reason=None,
        payload={"kind": "lens_unadopt", "name": name},
    )


async def get_adopted_lenses(session: AsyncSession) -> list[TrustLensDefinition]:
    """Return the full definitions of every adopted lens (for policy composition)."""
    adopted_names = [
        r.lens_name for r in (await session.execute(select(LensAdoptionRow))).scalars().all()
    ]
    lenses: list[TrustLensDefinition] = []
    for name in adopted_names:
        lens = await get_lens(session, name)
        if lens is not None:
            lenses.append(lens)
        else:  # adoption row outlived its lens (shouldn't happen; be loud, not fatal)
            log.warning("Adopted lens %r has no materialised definition; ignoring.", name)
    return lenses


async def get_adopted_lens_extractor_weights(session: AsyncSession) -> dict[str, float]:
    """Min-composed extractor-weight overrides across all adopted lenses."""
    weights: dict[str, float] = {}
    for lens in await get_adopted_lenses(session):
        for extractor_id, weight in lens.extractor_weights.items():
            current = weights.get(extractor_id)
            weights[extractor_id] = weight if current is None else min(current, weight)
    return weights


# ---------------------------------------------------------------------------
# register the Engine-side persistence sink with the Client-side
# extractor (mirrors taxonomy_store / extraction.taxonomy).
# ---------------------------------------------------------------------------
from particles.extraction.trust_lens import register_lens_sink  # noqa: E402

register_lens_sink(materialise_lens)
