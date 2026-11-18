# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Codex thin-client pins, key checks and remote continuity."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from particles.api._keys import Role, role_from_prefix
from particles.api.cli._hook_state import read_json
from particles.api.cli._thin_client import validate_engine_url
from particles.api.client.base import CodexCaptureEvidence
from particles.api.client.http import EngineHttpError, EngineUnreachableError, HttpBackend
from particles.config import environment_engine_url, get_config, loaded_config_file
from particles.corpus.harvest_marks import HARNESS_SOURCES
from particles.corpus.redaction import redact_secrets
from particles.extraction.codex_transcript import (
    CONTRACT,
    Ancestry,
    Transcript,
    UnsupportedTranscript,
)
from particles.operations.codex_capture import CODEX_CAPTURE_PROTOCOL, _prefix
from particles.secrets import EngineTokenFileError, check_engine_token_file
from particles.store.codex_capture_store import CaptureManifest

ROLES = {"recall": Role.READER, "capture": Role.CODEX_CAPTURE, "mcp": Role.AGENT}
_SOURCE = HARNESS_SOURCES["codex"]


class CodexKeyRefused(ValueError):
    """A stable diagnostic about the enrolled key, never its value."""


def checked_endpoint(raw: str) -> str:
    """Select the first-release loopback transport without claiming host attestation."""
    try:
        url = validate_engine_url(raw, allow_plaintext=False)
        host = urlsplit(url).hostname
        port = urlsplit(url).port
        loopback = host == "localhost" or (
            host is not None and ipaddress.ip_address(host).is_loopback
        )
        if not loopback or port == 0 or any(ch.isspace() or ord(ch) < 32 for ch in raw):
            raise ValueError
    except ValueError:
        raise ValueError(
            "Codex thin mode requires a valid same-host loopback engine URL."
        ) from None
    return url


def checked_config(path: Path) -> None:
    """Require a private, stable, absolute config pin before selecting an engine."""
    import os

    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("Codex thin mode requires an absolute config pin without links.")
    try:
        info = path.stat()
        parent = path.parent.stat()
    except OSError:
        raise ValueError("Codex thin mode requires an existing config pin.") from None
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o022
    ):
        raise ValueError("Codex thin mode requires a config pin writable only by this user.")


def destination_id(url: str, store: str) -> str:
    """A local state scope for a remote destination, disjoint from physical local stores."""
    return hashlib.sha256(json.dumps(["codex-engine", url, store]).encode()).hexdigest()


def installed_destination(
    store: str, expected: str | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Verify config and ownership before any bearer can reach an engine."""
    cfg = get_config()
    if environment_engine_url() or not cfg.engine.base_url:
        raise ValueError("Codex engine pin changed or is missing. Re-run particles init codex.")
    url = checked_endpoint(cfg.engine.base_url)
    identity = destination_id(url, store)
    if expected is not None and expected != identity:
        raise ValueError("Codex engine destination changed. Re-run particles init codex.")
    loaded = loaded_config_file()
    if loaded is None:
        raise ValueError("Codex engine config pin is missing. Re-run particles init codex.")
    checked_config(loaded.path)
    info = loaded.path.stat()
    if (
        (info.st_dev, info.st_ino, info.st_ctime_ns)
        != (loaded.device, loaded.inode, loaded.ctime_ns)
        or loaded.mode & 0o022
        or loaded.dir_mode is None
        or loaded.dir_mode & 0o022
        or loaded.uid != info.st_uid
        or loaded.dir_uid != info.st_uid
    ):
        raise ValueError("Codex engine config changed while loading. Re-run the command.")
    state = Path(cfg.codex.state_dir).expanduser().resolve()
    manifest = read_json(state / "installations" / f"{identity}.json", {})
    home = Path(cfg.codex.home_dir).expanduser().resolve()
    if (
        manifest.get("engine") != url
        or manifest.get("store") != store
        or manifest.get("config") != str(loaded.path)
        or manifest.get("home") != str(home)
    ):
        raise ValueError("Codex engine destination is not installed. Re-run particles init codex.")
    return url, identity, manifest


def capture_enabled() -> bool:
    """Whether the pinned configuration currently opts in to remote capture."""
    harvest = get_config().codex.harvest
    return harvest.enabled and harvest.allow_remote


def token_path(state: Path, client: str) -> Path:
    """The private owned credential path for one Codex consumer."""
    return state / "engine-tokens" / client


def read_client_key(path: Path, client: str) -> str:
    """Use the shared checked-file reader and refuse wrong roles without exposing key bytes."""
    token = check_engine_token_file(str(path))
    if role_from_prefix(token) is not ROLES[client]:
        raise CodexKeyRefused(f"{client} key must have role {ROLES[client].value}.")
    return token


async def check_key(
    backend: HttpBackend,
    client: str,
    *,
    asserter: str,
    expected_project: str | None = None,
    check_project: bool = False,
) -> str | None:
    """Prove the resolved role, principal and capture protocol on the engine."""
    who = await backend.whoami()
    if who.get("anonymous") or who.get("role") != ROLES[client].value:
        raise CodexKeyRefused(f"{client} key resolved to the wrong role.")
    principal = {"capture": _SOURCE.depositor, "mcp": asserter}.get(client)
    if principal is not None and who.get("principal") != principal:
        raise CodexKeyRefused(f"{client} key resolved to the wrong principal.")
    project = who.get("project")
    if project is not None and not isinstance(project, str):
        raise CodexKeyRefused(f"{client} key returned an invalid project binding.")
    if check_project and project != expected_project:
        raise CodexKeyRefused(
            f"{client} key has a different project binding. Re-run particles init codex."
        )
    if client == "capture" and who.get("codex_capture_protocol") != CODEX_CAPTURE_PROTOCOL:
        raise CodexKeyRefused("engine does not support this Codex capture protocol.")
    return project


def installed_backend(
    store: str,
    client: str,
    expected: str | None = None,
) -> tuple[HttpBackend, dict[str, Any]]:
    """Read only the installed credential for the named consumer."""
    url, _, manifest = installed_destination(store, expected)
    path = token_path(Path(get_config().codex.state_dir).expanduser().resolve(), client)
    if manifest.get("tokens", {}).get(client) != str(path):
        raise CodexKeyRefused(
            f"Codex {client} credential is not enrolled. Re-run particles init codex."
        )
    return HttpBackend(token=read_client_key(path, client), base_url=url), manifest


async def prove_installed_key(
    backend: HttpBackend,
    client: str,
    manifest: dict[str, Any],
) -> None:
    await check_key(
        backend,
        client,
        asserter=get_config().mcp.write.asserter_identity,
        expected_project=manifest.get("key_projects", {}).get(client),
        check_project=True,
    )


def failure_reason(exc: Exception) -> str:
    """Classify a remote failure without persisting server messages or request data."""
    if isinstance(exc, EngineUnreachableError):
        return "engine_unreachable"
    if isinstance(exc, EngineTokenFileError):
        return "capture_key_file_refused"
    if isinstance(exc, CodexKeyRefused):
        return "capture_key_or_protocol_refused"
    if isinstance(exc, EngineHttpError):
        if exc.status_code in {401, 403}:
            return "capture_key_or_namespace_refused"
        if exc.status_code == 429:
            return "engine_rate_limited"
        if exc.status_code is not None and exc.status_code >= 500:
            return "engine_unavailable"
        if exc.status_code == 409:
            for reason in ("codex_home_moved", "capture_binding_changed"):
                if str(exc).endswith(f": {reason}"):
                    return reason
            return "capture_continuity_refused"
    if isinstance(exc, TimeoutError):
        return "engine_timeout"
    return "capture_failed"


async def parent_proof(
    backend: HttpBackend,
    store: str,
    home: Path,
    home_id: str,
    boundary: Ancestry,
) -> str | None:
    response = await backend.codex_capture_proof(
        store,
        boundary.session_id,
        home_path=str(home),
        home_id=home_id,
        boundary=(boundary.end_byte_offset, boundary.end_ordinal),
    )
    proof = response.get("prefix") if response else None
    if isinstance(proof, dict):
        digest = proof.get("sha256")
        if (
            proof.get("byte_offset") == boundary.end_byte_offset
            and proof.get("ordinal") == boundary.end_ordinal
            and isinstance(digest, str)
            and len(digest) == 64
            and all(ch in "0123456789abcdef" for ch in digest)
        ):
            return digest
    return None


async def push_capture(
    backend: HttpBackend,
    store: str,
    *,
    home: Path,
    home_id: str,
    raw: bytes,
    transcript: Transcript,
    root: Path | None,
    project: str | None,
    boundary: Ancestry | None,
) -> None:
    """Validate the remote head locally and let the engine atomically compare it again."""
    body_lengths: list[int] = []
    text, messages = transcript.render(redact_secrets, body_lengths=body_lengths)
    content = text.encode()
    parent: dict[str, str | int] | None = None
    if transcript.ancestry is not None:
        ancestor = transcript.ancestry
        proof = await parent_proof(backend, store, home, home_id, ancestor)
        if proof is None:
            raise UnsupportedTranscript("unverified_fork_ancestor")
        parent = {
            "session_id": ancestor.session_id,
            "byte_offset": ancestor.end_byte_offset,
            "ordinal": ancestor.end_ordinal,
            "sha256": proof,
        }
    prefix: dict[str, str | int] | None = None
    if boundary is not None:
        prefix = {
            "byte_offset": boundary.end_byte_offset,
            "ordinal": boundary.end_ordinal,
            "sha256": _prefix(raw, boundary),
        }
    launch_root = str(root) if root else None
    for attempt in range(2):
        response = await backend.codex_capture_proof(
            store, transcript.session_id, home_path=str(home), home_id=home_id
        )
        previous = CaptureManifest.model_validate(response["manifest"]) if response else None
        prior_hash = None
        if previous is not None:
            if (previous.session_id, previous.home_path, previous.home_id) != (
                transcript.session_id,
                str(home),
                home_id,
            ):
                raise UnsupportedTranscript("codex_home_moved")
            if (previous.project_key, previous.launch_root, previous.contract, previous.parent) != (
                project,
                launch_root,
                CONTRACT,
                parent,
            ):
                raise UnsupportedTranscript("capture_binding_changed")
            prior_hash = hashlib.sha256(raw[: previous.raw_size]).hexdigest()
            if (
                len(raw) < previous.raw_size
                or prior_hash != previous.raw_hash
                or len(content) < previous.text_size
                or hashlib.sha256(content[: previous.text_size]).hexdigest() != previous.text_hash
                or messages[: len(previous.messages)] != [m.model_dump() for m in previous.messages]
            ):
                raise UnsupportedTranscript("non_append_history")
        evidence: CodexCaptureEvidence = {
            "version": 1,
            "session_id": transcript.session_id,
            "home_path": str(home),
            "home_id": home_id,
            "launch_root": launch_root,
            "project_key": project,
            "contract": CONTRACT,
            "started_at": transcript.started_at.isoformat(),
            "raw_size": len(raw),
            "raw_hash": hashlib.sha256(raw).hexdigest(),
            "messages": [
                {"id": message["id"], "body_length": length}
                for message, length in zip(messages, body_lengths, strict=True)
            ],
            "expected_capture_id": previous.capture_id if previous else None,
            "prior_raw_hash": prior_hash,
            "parent": parent,
            "prefix": prefix,
        }
        try:
            await backend.deposit_text_at_uri(
                text=text,
                uri_r=_SOURCE.session_prefix + transcript.session_id,
                source_type="CONVERSATION",
                mutability="APPEND_ONLY",
                fetch_policy="NEVER",
                tags=[],
                deposited_by=_SOURCE.depositor,
                content_published_at=transcript.started_at,
                store=store,
                codex_capture=evidence,
            )
        except EngineHttpError as exc:
            if attempt or exc.status_code != 409 or "codex_capture_head_changed" not in str(exc):
                raise
        else:
            return
