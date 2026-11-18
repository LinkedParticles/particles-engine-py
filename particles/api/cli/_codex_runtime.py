# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Local Codex capture composition, fingerprint ledger and operator disclosure."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from particles.api.client.http import HttpBackend

from sqlalchemy.engine import URL, make_url

from particles.api.cli._hook_state import advisory_lock, read_json, write_json
from particles.config import get_config
from particles.extraction.codex_transcript import (
    Ancestry,
    UnsupportedTranscript,
    parse_transcript,
    session_uuid,
)

HOME_MOVED = (
    "codex_home_moved: This session is bound to a different Codex home; capture "
    "from a home restored or moved to a new path is a known first-release limit. "
    "Existing captured data is retained; operator rebind is deferred."
)


def physical_store(dsn: str) -> str:
    """Hash a canonical database destination; handles/harnesses never define it."""
    url = make_url(dsn)
    if url.drivername.startswith("sqlite"):
        if not url.database or url.database == ":memory:" or not Path(url.database).is_absolute():
            raise ValueError("Codex requires an absolute persistent database location")
        destination = "sqlite:" + str(Path(url.database).resolve())
    else:
        destination = URL.create(
            drivername=url.drivername.split("+")[0],
            host=url.host.lower() if url.host else None,
            port=url.port or (5432 if url.drivername.startswith("postgresql") else None),
            database=url.database,
            query=url.query,
        ).render_as_string(hide_password=False)
    return hashlib.sha256(destination.encode()).hexdigest()


def store_dsn(store: str) -> str:
    """Resolve only the configured local handle, without touching a database."""
    cfg = get_config()
    if cfg.engine.base_url:
        raise ValueError("Codex integration supports local stores only; remote harvest is deferred")
    dsn = cfg.storage.database_url if store == "default" else cfg.storage.stores.get(store)
    if not dsn:
        raise ValueError("Codex store is not configured")
    return dsn


def paths(store: str) -> tuple[Path, Path, str]:
    """Resolve configured paths at call time, including test isolation overrides."""
    cfg = get_config().codex
    if get_config().engine.base_url:
        from particles.api.cli._codex_remote import installed_destination

        _, identity, _ = installed_destination(store)
    else:
        identity = physical_store(store_dsn(store))
    return (
        Path(cfg.home_dir).expanduser().resolve(),
        Path(cfg.state_dir).expanduser().resolve() / "stores" / identity,
        identity,
    )


def home_binding(home: Path) -> str:
    """Require the installer-generated binding; workers never silently mint it."""
    binding = read_json(Path(get_config().codex.state_dir).expanduser() / "home.json", {})
    if binding.get("path") != str(home):
        raise UnsupportedTranscript("codex_home_moved")
    return session_uuid(binding.get("id"))


def fingerprints(state: Path, identity: str) -> frozenset[str]:
    """Load exact-output tripwires; absence is different from unreadable state."""
    value = read_json(state / "emitted" / f"{session_uuid(identity)}.json", [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or len(item) != 64 for item in value
    ):
        raise ValueError("invalid_emitted_fingerprints")
    return frozenset(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def health_summary(store: str) -> str:
    """One disclosure for doctor AND every scheduled run, including not-due runs."""
    _home, state, _store_id = paths(store)
    health = read_json(state / "health.json", {})
    failures = health.get("quarantined", {})
    attempts = health.get("attempts", 0)
    count = len(failures)
    if not attempts:
        detail = "no capture attempts since installation/last success"
    elif count and health.get("completed_quarantines", 0) == attempts:
        detail = "EVERY recent completed capture attempt was quarantined"
    else:
        detail = f"{attempts} completed attempts since last successful capture"
    return (
        f"codex capture: {count} quarantined sessions; {detail}; "
        f"last success: {health.get('last_success', 'none')}"
    )


def _record_result(state: Path, identity: str, reason: str | None, *, success: bool) -> None:
    health = read_json(state / "health.json", {})
    if success:
        # Successful capture resets the disclosure window; individual unresolved
        # pending hints survive and remain visible independently.
        health = {
            "last_success": _now(),
            "attempts": 0,
            "completed_quarantines": 0,
            "quarantined": {},
        }
    else:
        health["attempts"] = health.get("attempts", 0) + 1
        if reason is not None:
            health["completed_quarantines"] = health.get("completed_quarantines", 0) + 1
            health.setdefault("quarantined", {})[identity] = {"reason": reason, "at": _now()}
    write_json(state / "health.json", health)


async def _capture_one(
    store: str,
    identity: str,
    cwd: str | None,
    visiting: set[str],
    boundary: Ancestry | None = None,
    excluded: str | None = None,
    remote: HttpBackend | None = None,
) -> bool:
    from particles.api.cli._codex_files import resolve_rollout
    from particles.api.cli._project_identity import claude_project_slug, find_repository_root
    from particles.db import session_scope
    from particles.operations.codex_capture import capture_codex
    from particles.store.codex_capture_store import verified_prefix

    home, state, _store_id = paths(store)
    home_id = home_binding(home)
    if identity in visiting:
        raise UnsupportedTranscript("fork_ancestry_cycle")

    async def have_boundary() -> bool:
        if boundary is None:
            return False
        if remote is not None:
            from particles.api.cli._codex_remote import parent_proof

            return await parent_proof(remote, store, home, home_id, boundary) is not None
        async with session_scope(store) as session:
            return (
                await verified_prefix(
                    session,
                    identity,
                    boundary.end_byte_offset,
                    boundary.end_ordinal,
                    str(home),
                    home_id,
                )
                is not None
            )

    if identity == excluded:
        if await have_boundary():
            return True
        raise UnsupportedTranscript("active_fork_ancestor")
    visiting.add(identity)
    try:
        try:
            rollout = resolve_rollout(
                home, identity, get_config().codex.harvest.max_rollout_bytes, cwd
            )
        except FileNotFoundError:
            if await have_boundary():
                return True
            raise
        if boundary is None:
            hint = state / "pending" / f"{identity}.json"
            previous = read_json(hint, {})
            metadata = {
                "session_id": identity,
                "store_id": _store_id,
                "home": str(home),
                "first_attempt": previous.get("first_attempt", _now()),
                "last_attempt": _now(),
                "failure": "attempting_capture",
                "relative_path": str(rollout.path.relative_to(home)),
                "raw_sha256": hashlib.sha256(rollout.content).hexdigest(),
            }
            if len(json.dumps(metadata).encode()) > 4096:
                raise ValueError("capture_hint_oversized")
            write_json(hint, metadata)
        emitted = fingerprints(state, identity)
        transcript = parse_transcript(rollout.content.decode(), emitted_hashes=emitted)
        if transcript.ancestry:
            ancestor = transcript.ancestry
            await _capture_one(
                store, ancestor.session_id, None, visiting, ancestor, excluded, remote
            )
        root = find_repository_root(Path(transcript.cwd))
        project = claude_project_slug(root) if root else None
        if remote is not None:
            from particles.api.cli._codex_remote import push_capture

            await push_capture(
                remote,
                store,
                home=home,
                home_id=home_id,
                raw=rollout.content,
                transcript=transcript,
                root=root,
                project=project,
                boundary=boundary,
            )
        else:
            async with session_scope(store, write=True) as session:
                await capture_codex(
                    session,
                    raw=rollout.content,
                    home=home,
                    home_id=home_id,
                    project_key=project,
                    launch_root=root,
                    emitted_hashes=emitted,
                    requested_prefix=boundary,
                )
                await session.commit()
        return True
    finally:
        visiting.remove(identity)


async def capture_worker(store: str, identity: str, cwd: str, catch_up: bool) -> None:
    """Bounded serialized repair; cursors and hints never mean extracted/captured."""
    from particles.api.cli._codex_files import inventory
    from particles.api.cli._codex_remote import (
        capture_enabled,
        failure_reason,
        installed_backend,
        prove_installed_key,
    )

    identity = session_uuid(identity)
    home, state, store_id = paths(store)
    if not get_config().codex.harvest.enabled:
        return
    remote_mode = bool(get_config().engine.base_url)
    if remote_mode and not capture_enabled():
        return
    with advisory_lock(state / "capture.lock") as acquired:
        if not acquired:
            return
        if catch_up:
            candidates: set[str] = set()
            for path in inventory(home):
                try:
                    candidates.add(session_uuid(path.name[-42:-6]))
                except UnsupportedTranscript:
                    continue
            pending = state / "pending"
            hints: set[str] = set()
            if pending.exists():
                for hint in pending.glob("*.json"):
                    try:
                        hinted = session_uuid(hint.stem)
                        metadata = read_json(hint, {})
                        if (
                            metadata.get("session_id") == hinted
                            and metadata.get("store_id") == store_id
                            and metadata.get("home") == str(home)
                        ):
                            hints.add(hinted)
                    except (ValueError, OSError):
                        continue
            candidates.discard(identity)
            hints.discard(identity)
            cursor = read_json(state / "cursor.json", {})

            def rotate(values: set[str], after: str) -> list[str]:
                ordered = sorted(values)
                return [i for i in ordered if i > after] + [i for i in ordered if i <= after]

            pools = {
                "pending": rotate(hints, cursor.get("pending_after", "")),
                "scan": rotate(candidates, cursor.get("after", "")),
            }
            selected: list[tuple[str, str]] = []
            seen: set[str] = set()
            prefer = "pending" if cursor.get("prefer_pending", True) else "scan"
            for _ in range(get_config().codex.harvest.catchup_limit):
                other = "scan" if prefer == "pending" else "pending"
                for origin in (prefer, other):
                    pools[origin] = [i for i in pools[origin] if i not in seen]
                    if pools[origin]:
                        candidate = pools[origin].pop(0)
                        selected.append((candidate, origin))
                        seen.add(candidate)
                        prefer = "scan" if origin == "pending" else "pending"
                        break
                else:
                    break
        else:
            selected = [(identity, "")]
            cursor = {}

        async def run() -> None:
            remote: HttpBackend | None = None
            key_failure = None
            if remote_mode and selected:
                try:
                    backend, manifest = installed_backend(store, "capture")
                    await prove_installed_key(backend, "capture", manifest)
                    remote = backend
                except Exception as exc:  # record only a stable, non-sensitive reason
                    key_failure = failure_reason(exc)
            for candidate, origin in selected:
                reason = key_failure
                success = False
                if reason is None:
                    try:
                        success = await _capture_one(
                            store,
                            candidate,
                            cwd if not catch_up else None,
                            set(),
                            excluded=identity if catch_up else None,
                            remote=remote,
                        )
                    except FileNotFoundError:
                        reason = "source_lost_before_capture"
                    except UnsupportedTranscript as exc:
                        reason = str(exc)
                    except Exception as exc:  # never log raw rollout, credentials or SQL errors
                        reason = failure_reason(exc) if remote_mode else "capture_failed"
                _record_result(state, candidate, reason, success=success)
                hint = state / "pending" / f"{candidate}.json"
                if success:
                    hint.unlink(missing_ok=True)  # ONLY after the corpus transaction committed
                elif reason:
                    previous = read_json(hint, {})
                    write_json(
                        hint,
                        {
                            "session_id": candidate,
                            "store_id": store_id,
                            "home": str(home),
                            "first_attempt": previous.get("first_attempt", _now()),
                            "last_attempt": _now(),
                            "failure": reason,
                            **{
                                key: previous[key]
                                for key in ("relative_path", "raw_sha256")
                                if key in previous
                            },
                        },
                    )
                if catch_up:
                    cursor["pending_after" if origin == "pending" else "after"] = candidate
                    cursor["prefer_pending"] = origin != "pending"
                    write_json(state / "cursor.json", cursor)

        await asyncio.wait_for(run(), get_config().codex.harvest.worker_deadline_seconds)


async def recall(store: str, identity: str, cwd: str, reason: str = "startup") -> str | None:
    """Recall is best effort, provenance-checked, with exposure ALWAYS unknown."""
    from particles.api.cli._claude_code import truncate_on_line_boundary
    from particles.api.cli._codex_files import resolve_rollout
    from particles.api.cli._project_identity import (
        claude_project_slug,
        find_repository_root,
        observer_for_policy,
        record_identity_root,
    )
    from particles.api.client import get_backend

    warning = capture_warning(store)
    warning_output = json.dumps({"systemMessage": warning}) if warning else None
    if reason not in {"startup", "clear", "compact", "fork"}:
        return warning_output  # resume still discloses failures without another digest
    home, state, _store_id = paths(store)
    home_binding(home)
    backend = get_backend()
    rollout = resolve_rollout(
        home, session_uuid(identity), get_config().codex.harvest.max_rollout_bytes, cwd
    )
    root = find_repository_root(Path(rollout.metadata["cwd"]))
    project = claude_project_slug(root) if root else None
    cfg = get_config().codex
    if project is None and cfg.observer_scope == "project":
        return warning_output
    if root and project:
        roots = record_identity_root(project, root, codex=True)
        if len(roots) > 1:
            import logging

            logging.getLogger(__name__).warning(
                "project_key_collision: %s shares scope across repositories/harnesses", project
            )
    observer = observer_for_policy(project, cfg.observer_scope) if project else None
    if observer is not None and not backend.remote:
        from particles.db import session_scope
        from particles.store.observer_scope_join import lens_may_engage

        async with session_scope(store) as session:
            if not await lens_may_engage(session):
                return warning_output  # never silently downgrade project recall to store scope
    if backend.remote:
        from particles.api.cli._codex_remote import installed_backend, prove_installed_key

        try:
            remote, manifest = installed_backend(store, "recall")
            await prove_installed_key(remote, "recall", manifest)
            rendered = await remote.digest_located(
                store, observer, strict_scope=observer is not None
            )
        except (ValueError, RuntimeError):
            return warning_output
        if observer is not None and rendered.observer_engaged is not True:
            return warning_output
    else:
        rendered = await backend.digest_located(store, observer, None)
    digest = truncate_on_line_boundary(rendered.markdown, cfg.digest_max_bytes)
    if not digest:
        return warning_output
    with advisory_lock(state / "emitted.lock") as acquired:
        if not acquired:
            return warning_output
        hashes = fingerprints(state, identity)
        write_json(
            state / "emitted" / f"{identity}.json",
            sorted(hashes | {hashlib.sha256(digest.encode()).hexdigest()}),
        )
    result: dict[str, Any] = {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": digest},
    }
    if warning:
        result["systemMessage"] = warning
    return json.dumps(result)


async def initialize_empty_scope(store: str) -> bool:
    """Mark only an empty new corpus scoped; leave existing attribution untouched."""
    from particles.corpus.store import list_entry_tag_rows
    from particles.db import session_scope
    from particles.operations.observer_scope import rescope
    from particles.store.observer_scope_join import lens_may_engage

    async with session_scope(store, write=True) as session:
        if await lens_may_engage(session):
            return True
        if await list_entry_tag_rows(session):
            return False
        await rescope(session, key_for=lambda *_args: None, actor="init:codex")
        await session.commit()
        return True


def _hint_age_seconds(value: object) -> float | None:
    """Read either launcher epoch or worker UTC time, without trusting a path."""
    try:
        if isinstance(value, (int, float)):
            then = datetime.fromtimestamp(value, UTC)
        elif isinstance(value, str):
            then = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if then.tzinfo is None:
                then = then.replace(tzinfo=UTC)
        else:
            return None
        return max(0.0, (datetime.now(UTC) - then).total_seconds())
    except (ValueError, OverflowError, OSError):
        return None


def warning_details(store: str) -> tuple[str, str] | None:
    """A metadata-only failure signature and bounded UI warning."""
    import shlex
    from collections import Counter

    _home, state, store_id = paths(store)
    failures: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    for hint in sorted((state / "pending").glob("*.json")):
        try:
            value = read_json(hint, {})
            reason = value.get("failure", "pending")
            if reason in {"pending", "attempting_capture"}:
                age = _hint_age_seconds(value.get("first_attempt"))
                if age is not None and age <= get_config().codex.harvest.worker_deadline_seconds:
                    continue
                kind = "stale"
            elif reason == "source_lost_before_capture":
                kind = "lost"
            else:
                kind = "quarantined"
        except (OSError, ValueError, AttributeError):
            reason, kind = "unreadable_state", "storage"
        counts[kind] += 1
        failures.append((hint.stem, str(reason)))
    learning = read_json(state / "learning.json", {})
    status = learning.get("credential", "unknown")
    installed = (state.parent.parent / "installations" / f"{store_id}.json").exists()
    if installed and status not in {
        "environment",
        "Keychain",
        "keyless local route",
        "job environment",
    }:
        counts["learning blocked"] += 1
        failures.append(("scheduled_credential", str(status)))
    if not failures:
        return None
    signature = hashlib.sha256(json.dumps(failures, sort_keys=True).encode()).hexdigest()
    summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    message = (
        f"Particles needs attention: {summary}. Delivery remains unknown. "
        f"Run particles hook doctor --harness codex --store {shlex.quote(store)}."
    )
    return signature, message


def capture_warning(store: str) -> str | None:
    """Repeat until resolved or this exact signature is explicitly acknowledged."""
    details = warning_details(store)
    if details is None:
        return None
    _home, state, _store_id = paths(store)
    if read_json(state / "warning-ack.json", {}).get("signature") == details[0]:
        return None
    return details[1]


def acknowledge_warning(store: str) -> None:
    """Acknowledge only the current UI signature; retain hints and corpus data."""
    details = warning_details(store)
    if details is None:
        raise ValueError("no active capture/learning failure signature to acknowledge")
    _home, state, _store_id = paths(store)
    write_json(state / "warning-ack.json", {"signature": details[0], "acknowledged_at": _now()})


def record_learning_status(store: str, status: str) -> None:
    """Persist non-secret scheduled readiness, without creating a capture queue."""
    _home, state, store_id = paths(store)
    if not (state.parent.parent / "installations" / f"{store_id}.json").exists():
        return
    write_json(state / "learning.json", {"credential": status, "checked_at": _now()})
