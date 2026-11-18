# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Codex capture continuity and URI-owned writes for local and engine intake."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import UNATTRIBUTED_MARKER
from particles.core.schema import CorpusEntry, FetchPolicy, Mutability, Snapshot, WarcRecordType
from particles.corpus.deposit import (
    _refuse_foreign_entry,
    own_entries_only_for,
    save_blob,
    write_entry_and_snapshot,
)
from particles.corpus.harvest_marks import HARNESS_SOURCES
from particles.corpus.redaction import redact_secrets
from particles.corpus.store import get_entry_by_uri, get_latest_snapshot
from particles.extraction.codex_transcript import (
    CONTRACT,
    Ancestry,
    UnsupportedTranscript,
    parse_transcript,
    session_uuid,
)
from particles.extraction.day_marks import TURN_HEADER, escape_mark_lines, unescape_mark_lines
from particles.store.codex_capture_store import (
    CaptureManifest,
    CaptureProvenance,
    MessageProof,
    get_capture,
    record_capture,
    record_prefix,
    verified_prefix,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prefix(raw: bytes, boundary: Ancestry) -> str:
    end = boundary.end_byte_offset
    if end > len(raw) or end < 0 or (end and raw[end - 1 : end] != b"\n"):
        raise UnsupportedTranscript("invalid_fork_byte_boundary")
    records = raw[:end].splitlines()
    if len(records) != boundary.end_ordinal or any(
        json.loads(line).get("ordinal") != index for index, line in enumerate(records)
    ):
        raise UnsupportedTranscript("invalid_fork_ordinal_boundary")
    return _digest(raw[:end])


async def capture_codex(
    session: AsyncSession,
    *,
    raw: bytes,
    home: Path,
    home_id: str,
    project_key: str | None,
    launch_root: Path | None,
    emitted_hashes: frozenset[str] = frozenset(),
    requested_prefix: Ancestry | None = None,
) -> tuple[CaptureManifest, bool]:
    """Deposit a validated conversation and provenance atomically; caller commits.

    No HTTP principal or caller-selected URI/depositor is accepted. The
    Surface supplies a contained input; this gate independently parses it.
    """
    if len(raw) > get_config().codex.harvest.max_rollout_bytes:
        raise UnsupportedTranscript("rollout_oversized")
    transcript = parse_transcript(raw.decode("utf-8"), emitted_hashes=emitted_hashes)
    session_uuid(home_id)
    home_path = str(home.resolve())
    text, identities = transcript.render(redact_secrets)
    content = text.encode()
    proofs = [MessageProof.model_validate(item) for item in identities]
    parent: dict[str, str | int] | None = None
    with own_entries_only_for(_CODEX.depositor):
        latest, previous = await _capture_head(session, transcript.session_id)
        if previous is not None:
            if previous.home_path != home_path or previous.home_id != home_id:
                raise UnsupportedTranscript("codex_home_moved")
            if (
                previous.contract != CONTRACT
                or previous.project_key != project_key
                or previous.launch_root != (str(launch_root) if launch_root else None)
            ):
                raise UnsupportedTranscript("capture_binding_changed")
            if (
                len(raw) < previous.raw_size
                or _digest(raw[: previous.raw_size]) != previous.raw_hash
            ):
                raise UnsupportedTranscript("non_append_history")
            if proofs[: len(previous.messages)] != previous.messages:
                raise UnsupportedTranscript("non_append_history")
            if (
                len(content) < previous.text_size
                or _digest(content[: previous.text_size]) != previous.text_hash
            ):
                raise UnsupportedTranscript("non_append_history")
        if transcript.ancestry is not None:
            ancestor = transcript.ancestry
            proof = await verified_prefix(
                session,
                ancestor.session_id,
                ancestor.end_byte_offset,
                ancestor.end_ordinal,
                home_path,
                home_id,
            )
            if proof is None:
                raise UnsupportedTranscript("unverified_fork_ancestor")
            parent = {
                "session_id": ancestor.session_id,
                "byte_offset": ancestor.end_byte_offset,
                "ordinal": ancestor.end_ordinal,
                "sha256": proof,
            }
        boundary_hash = None
        if requested_prefix is not None:
            if requested_prefix.session_id != transcript.session_id:
                raise UnsupportedTranscript("wrong_parent_identity")
            boundary_hash = _prefix(raw, requested_prefix)
        provenance = CaptureProvenance(
            session_id=transcript.session_id,
            home_path=home_path,
            home_id=home_id,
            contract=CONTRACT,
            launch_root=str(launch_root) if launch_root else None,
            project_key=project_key,
            raw_size=len(raw),
            raw_hash=_digest(raw),
            text_size=len(content),
            text_hash=_digest(content),
            messages=proofs,
            parent=parent,
        )
        manifest, unchanged = await _persist_capture(
            session, provenance, content, transcript.started_at, latest, previous
        )
        if requested_prefix is not None and boundary_hash is not None:
            await record_prefix(
                session,
                manifest,
                requested_prefix.end_byte_offset,
                requested_prefix.end_ordinal,
                boundary_hash,
            )
        return manifest, unchanged


_CODEX = HARNESS_SOURCES["codex"]
CODEX_CAPTURE_PROTOCOL = 1
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
BoundaryValue = Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]


class CodexBoundary(BaseModel):
    """An exact raw prefix attested by the client that read the rollout."""

    model_config = ConfigDict(extra="forbid")
    byte_offset: BoundaryValue
    ordinal: BoundaryValue
    sha256: Sha256


class CodexParent(CodexBoundary):
    """The already-recorded parent boundary inherited by a new fork."""

    session_id: str


class CodexCaptureMessage(BaseModel):
    """A bounded message identity and Unicode length of its escaped body."""

    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1, max_length=1024)]
    body_length: BoundaryValue


class CodexSubmission(BaseModel):
    """Versioned wire payload with distilled text and client-attested evidence."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    session_id: str
    text: str
    home_path: str
    home_id: str
    launch_root: str | None
    project_key: str | None
    contract: Literal["codex-conversation/1"]
    started_at: AwareDatetime
    raw_size: BoundaryValue
    raw_hash: Sha256
    messages: list[CodexCaptureMessage]
    expected_capture_id: str | None
    prior_raw_hash: Sha256 | None = None
    parent: CodexParent | None = None
    prefix: CodexBoundary | None = None


async def _capture_head(
    session: AsyncSession, identity: str
) -> tuple[Snapshot | None, CaptureManifest | None]:
    """Check corpus ownership and the committed head before using any proof."""
    uri = _CODEX.session_prefix + identity
    existing: CorpusEntry | None = await get_entry_by_uri(session, uri)
    if existing is not None:
        _refuse_foreign_entry(existing, uri)
    previous = await get_capture(session, identity)
    if (existing is None) != (previous is None):
        raise UnsupportedTranscript("missing_capture_manifest")
    if existing is None or previous is None:
        return None, None
    latest = await get_latest_snapshot(session, existing.entry_id)
    if (
        latest is None
        or latest.snapshot_id != previous.snapshot_id
        or existing.entry_id != previous.entry_id
        or previous.session_id != identity
    ):
        raise UnsupportedTranscript("external_source_change")
    if existing.mutability != Mutability.APPEND_ONLY or existing.source_type != "CONVERSATION":
        raise UnsupportedTranscript("source_shape_changed")
    return latest, previous


async def _persist_capture(
    session: AsyncSession,
    provenance: CaptureProvenance,
    content: bytes,
    started_at: datetime,
    latest: Snapshot | None,
    previous: CaptureManifest | None,
    *,
    mark_unattributed: bool = False,
) -> tuple[CaptureManifest, bool]:
    unchanged = latest is not None and latest.content_hash == provenance.text_hash
    if unchanged and previous is not None:
        entry_id, snapshot_id = previous.entry_id, previous.snapshot_id
    else:
        archive = save_blob(content, provenance.text_hash)
        tags = [_CODEX.tag]
        if provenance.project_key:
            tags.append(f"project:{provenance.project_key}")
        elif mark_unattributed:
            tags.append(UNATTRIBUTED_MARKER)
        entry_id, snapshot_id = await write_entry_and_snapshot(
            session=session,
            uri_r=_CODEX.session_prefix + provenance.session_id,
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            fetch_policy=FetchPolicy.NEVER,
            content=content,
            content_hash=provenance.text_hash,
            archive_path=archive,
            deposited_by=_CODEX.depositor,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
            content_published_at=started_at,
            identified_by_uri=True,
            own_entries_only=True,
        )
    # A no-op snapshot can still have new continuity metadata.
    manifest = CaptureManifest(
        **provenance.model_dump(), entry_id=entry_id, snapshot_id=snapshot_id
    )
    if previous is not None and previous.model_dump(exclude={"capture_id"}) == manifest.model_dump(
        exclude={"capture_id"}
    ):
        return previous, unchanged
    await record_capture(
        session, manifest, expected_capture_id=previous.capture_id if previous else None
    )
    return manifest, unchanged


async def read_codex_capture(
    session: AsyncSession,
    identity: str,
    *,
    home_path: str,
    home_id: str,
    project_binding: str | None,
) -> CaptureManifest | None:
    """Read only the committed head for the caller's project and bound home."""
    with own_entries_only_for(_CODEX.depositor):
        _, previous = await _capture_head(session, identity)
    if previous is not None and (
        previous.home_id != home_id
        or previous.home_path != home_path
        or (project_binding is not None and previous.project_key != project_binding)
    ):
        return None
    return previous


def _mask_remote_turns(submission: CodexSubmission) -> tuple[bytes, list[MessageProof]]:
    """Rebuild canonical text and proofs from engine-masked bodies."""
    text = submission.text
    offset = 0
    turns: list[str] = []
    proofs: list[MessageProof] = []
    for message in submission.messages:
        heading = TURN_HEADER.match(text, offset)
        if heading is None:
            raise UnsupportedTranscript("invalid_capture_turns")
        end = heading.end() + message.body_length
        if text[end : end + 2] != "\n\n":
            raise UnsupportedTranscript("invalid_capture_turns")
        escaped = text[heading.end() : end]
        body = unescape_mark_lines(escaped)
        if escape_mark_lines(body) != escaped:
            raise UnsupportedTranscript("invalid_capture_turns")
        masked = redact_secrets(body)
        turns.append(f"{heading[0]}{escape_mark_lines(masked)}\n\n")
        proofs.append(MessageProof(id=message.id, body_hash=_digest(masked.encode())))
        offset = end + 2
    rendered = "".join(turns)
    if offset != len(text):
        raise UnsupportedTranscript("invalid_capture_turns")
    # Fail closed if the masker recognizes something across turn boundaries.
    if redact_secrets(rendered) != rendered:
        raise UnsupportedTranscript("unsafe_capture_turn_boundaries")
    content = rendered.encode()
    if len(content) > get_config().codex.harvest.max_rollout_bytes:
        raise UnsupportedTranscript("capture_text_oversized")
    return content, proofs


def _check_remote_binding(
    previous: CaptureManifest,
    submission: CodexSubmission,
    parent: dict[str, str | int] | None,
) -> None:
    if (previous.home_path, previous.home_id) != (submission.home_path, submission.home_id):
        raise UnsupportedTranscript("codex_home_moved")
    if (
        previous.contract != submission.contract
        or previous.project_key != submission.project_key
        or previous.launch_root != submission.launch_root
        or previous.parent != parent
    ):
        raise UnsupportedTranscript("capture_binding_changed")


def _check_remote_append(
    previous: CaptureManifest,
    submission: CodexSubmission,
    content: bytes,
    messages: list[MessageProof],
) -> None:
    if (
        submission.raw_size < previous.raw_size
        or submission.prior_raw_hash != previous.raw_hash
        or len(content) < previous.text_size
        or _digest(content[: previous.text_size]) != previous.text_hash
        or messages[: len(previous.messages)] != previous.messages
        or (
            submission.raw_size == previous.raw_size
            and (
                submission.raw_hash != previous.raw_hash
                or len(content) != previous.text_size
                or messages != previous.messages
            )
        )
    ):
        raise UnsupportedTranscript("non_append_history")


async def capture_remote_codex(
    session: AsyncSession,
    submission: CodexSubmission,
    *,
    project_binding: str | None,
) -> tuple[CaptureManifest, bool]:
    """Verify distilled continuity and the raw attestation under the writer lock."""
    with own_entries_only_for(_CODEX.depositor):
        latest, previous = await _capture_head(session, submission.session_id)
        parent = submission.parent.model_dump() if submission.parent else None
        if previous is not None:
            _check_remote_binding(previous, submission, parent)
        if submission.expected_capture_id != (previous.capture_id if previous else None):
            raise UnsupportedTranscript("codex_capture_head_changed")
        content, messages = _mask_remote_turns(submission)
        if previous is not None:
            _check_remote_append(previous, submission, content, messages)
        if submission.parent is not None:
            ancestor = submission.parent
            parent_head = await read_codex_capture(
                session,
                ancestor.session_id,
                home_path=submission.home_path,
                home_id=submission.home_id,
                project_binding=project_binding,
            )
            if (
                parent_head is None
                or await verified_prefix(
                    session,
                    ancestor.session_id,
                    ancestor.byte_offset,
                    ancestor.ordinal,
                    submission.home_path,
                    submission.home_id,
                )
                != ancestor.sha256
            ):
                raise UnsupportedTranscript("unverified_fork_ancestor")
        prefix = submission.prefix
        if prefix is not None:
            existing_prefix = await verified_prefix(
                session,
                submission.session_id,
                prefix.byte_offset,
                prefix.ordinal,
                submission.home_path,
                submission.home_id,
            )
            if existing_prefix is not None and existing_prefix != prefix.sha256:
                raise UnsupportedTranscript("codex_conflicting_parent_prefix")
        provenance = CaptureProvenance(
            session_id=submission.session_id,
            home_path=submission.home_path,
            home_id=submission.home_id,
            contract=submission.contract,
            launch_root=submission.launch_root,
            project_key=submission.project_key,
            raw_size=submission.raw_size,
            raw_hash=submission.raw_hash,
            text_size=len(content),
            text_hash=_digest(content),
            messages=messages,
            parent=parent,
        )
        started_at = latest.content_published_at if latest is not None else None
        manifest, unchanged = await _persist_capture(
            session,
            provenance,
            content,
            started_at or submission.started_at,
            latest,
            previous,
            mark_unattributed=True,
        )
        if prefix is not None:
            await record_prefix(
                session, manifest, prefix.byte_offset, prefix.ordinal, prefix.sha256
            )
        return manifest, unchanged
