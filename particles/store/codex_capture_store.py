# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic capture provenance, retained independently of rollout lifetime.

Corpus snapshots remain extraction progress. These primary records prove
source continuity and verified fork boundaries, without storing raw dialogue.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from particles.db import Base

PRESERVED_TABLES = frozenset(
    {"codex_capture_heads", "codex_capture_provenance", "codex_prefix_proofs"}
)


class MessageProof(BaseModel):
    """Accepted redacted body identity; never raw text or a secret."""

    id: str
    body_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class CaptureProvenance(BaseModel):
    """Source continuity metadata independent of corpus snapshot IDs."""

    session_id: str
    home_path: str
    home_id: str
    contract: str
    launch_root: str | None
    project_key: str | None
    raw_size: int
    raw_hash: str
    text_size: int
    text_hash: str
    messages: list[MessageProof]
    parent: dict[str, str | int] | None = None


class CaptureManifest(CaptureProvenance):
    """One verified capture attempt and its committed corpus snapshot."""

    capture_id: str = Field(default_factory=lambda: str(uuid4()))
    entry_id: str
    snapshot_id: str


class CodexCaptureRow(Base):
    __tablename__ = "codex_capture_provenance"
    capture_id: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"), nullable=False
    )
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CodexCaptureHeadRow(Base):
    __tablename__ = "codex_capture_heads"
    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("codex_capture_provenance.capture_id", ondelete="CASCADE"), nullable=False
    )


class CodexPrefixRow(Base):
    __tablename__ = "codex_prefix_proofs"
    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    byte_offset: Mapped[int] = mapped_column(Integer, primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("codex_capture_provenance.capture_id", ondelete="CASCADE"), nullable=False
    )
    digest: Mapped[str] = mapped_column(String, nullable=False)


async def get_capture(session: AsyncSession, identity: str) -> CaptureManifest | None:
    """Return the current source manifest, never an ORM row."""
    head = await session.get(CodexCaptureHeadRow, identity)
    if head is None:
        return None
    row = await session.get(CodexCaptureRow, head.capture_id)
    if row is None:
        raise ValueError("codex_capture_head_missing_provenance")
    return CaptureManifest.model_validate_json(row.payload_json)


class CaptureStateConflict(ValueError):
    """A concurrent writer changed the committed head or prefix."""


async def record_capture(
    session: AsyncSession, manifest: CaptureManifest, *, expected_capture_id: str | None
) -> None:
    """Append provenance and advance its head only from the expected state."""
    session.add(
        CodexCaptureRow(
            capture_id=manifest.capture_id,
            snapshot_id=manifest.snapshot_id,
            payload_json=manifest.model_dump_json(),
            captured_at=datetime.now(UTC),
        )
    )
    await session.flush()
    if expected_capture_id is None:
        session.add(
            CodexCaptureHeadRow(session_id=manifest.session_id, capture_id=manifest.capture_id)
        )
        try:
            await session.flush()
        except IntegrityError:
            # Another transaction established the same session. The caller
            # rolls back this transaction, including its attempted snapshot.
            raise CaptureStateConflict("codex_capture_head_changed") from None
    else:
        result = await session.execute(
            update(CodexCaptureHeadRow)
            .where(
                CodexCaptureHeadRow.session_id == manifest.session_id,
                CodexCaptureHeadRow.capture_id == expected_capture_id,
            )
            .values(capture_id=manifest.capture_id)
            .returning(CodexCaptureHeadRow.capture_id)
        )
        if result.scalar_one_or_none() is None:
            raise CaptureStateConflict("codex_capture_head_changed")


async def record_prefix(
    session: AsyncSession, manifest: CaptureManifest, offset: int, ordinal: int, digest: str
) -> None:
    """Record a checked boundary, hashed by the process that read the rollout."""
    key = (manifest.session_id, offset, ordinal)
    old = await session.get(CodexPrefixRow, key)
    if old is not None:
        if old.digest != digest:
            raise CaptureStateConflict("codex_conflicting_parent_prefix")
        return
    session.add(
        CodexPrefixRow(
            session_id=manifest.session_id,
            byte_offset=offset,
            ordinal=ordinal,
            capture_id=manifest.capture_id,
            digest=digest,
        )
    )
    try:
        await session.flush()
    except IntegrityError:
        raise CaptureStateConflict("codex_prefix_changed") from None


async def verified_prefix(
    session: AsyncSession, identity: str, offset: int, ordinal: int, home_path: str, home_id: str
) -> str | None:
    """Only an already-proved prefix from this bound home can stand in for a lost ancestor."""
    row = (
        await session.execute(
            select(CodexPrefixRow, CodexCaptureRow)
            .join(CodexCaptureRow, CodexPrefixRow.capture_id == CodexCaptureRow.capture_id)
            .where(
                CodexPrefixRow.session_id == identity,
                CodexPrefixRow.byte_offset == offset,
                CodexPrefixRow.ordinal == ordinal,
            )
        )
    ).first()
    if row is None:
        return None
    proof, capture = row
    payload = json.loads(capture.payload_json)
    if payload["home_path"] != home_path or payload["home_id"] != home_id:
        raise ValueError("codex_home_moved")
    return str(proof.digest)
