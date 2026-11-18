# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Failure, hand-off and diagnostic contracts without running an owner's job."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from particles.api.cli import _codex_launcher as launcher
from particles.api.cli import _codex_runtime as runtime
from particles.api.cli._hook_state import read_json, write_json


def arguments(tmp_path: Path, event: str) -> argparse.Namespace:
    return argparse.Namespace(
        state=str(tmp_path),
        home=str(tmp_path / "codex"),
        store_id="a" * 64,
        store="default",
        command="/example/bin/particles",
        event=event,
        deadline=1,
    )


def test_handoff_durable_hint_precedes_detach_and_ignores_transcript_path(tmp_path, monkeypatch):
    identity = str(uuid4())
    args = arguments(tmp_path, "SessionEnd")
    hint = tmp_path / "stores" / args.store_id / "pending" / f"{identity}.json"

    def spawn(argv, **kw):
        assert read_json(hint)["session_id"] == identity
        assert kw["start_new_session"] and kw["close_fds"]
        assert "/arbitrary/secret" not in argv

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    launcher.handoff(
        args,
        {
            "session_id": identity,
            "cwd": str(tmp_path),
            "hook_event_name": "SessionEnd",
            "transcript_path": "/arbitrary/secret",
        },
    )
    assert hint.stat().st_mode & 0o777 == 0o600
    assert hint.stat().st_size < 4096


def test_hint_failure_never_claims_launch(tmp_path, monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher, "atomic_metadata", Mock(side_effect=OSError("full")))
    with pytest.raises(OSError):
        launcher.handoff(
            arguments(tmp_path, "SessionEnd"),
            {"session_id": str(uuid4()), "cwd": str(tmp_path), "hook_event_name": "SessionEnd"},
        )
    spawn.assert_not_called()


def test_every_start_detaches_then_requests_warning_only_on_resume(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: calls.append("detach"))

    def run(*args, **kw):
        calls.append("recall")
        assert calls == ["detach", "recall"]
        assert "--reason" in args[0] and "resume" in args[0]
        return Mock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher.sys, "stdout", Mock(buffer=io.BytesIO()))
    launcher.handoff(
        arguments(tmp_path, "SessionStart"),
        {
            "session_id": str(uuid4()),
            "cwd": str(tmp_path),
            "hook_event_name": "SessionStart",
            "source": "resume",
        },
    )
    assert calls == ["detach", "recall"]
    assert not list(tmp_path.rglob("pending"))


def test_fingerprints_missing_is_empty_but_unsafe_state_refused(tmp_path):
    identity = str(uuid4())
    assert runtime.fingerprints(tmp_path, identity) == frozenset()
    target = tmp_path / "emitted" / f"{identity}.json"
    write_json(target, ["a" * 64])
    assert runtime.fingerprints(tmp_path, identity) == {"a" * 64}
    target.chmod(0o644)
    with pytest.raises(ValueError):
        runtime.fingerprints(tmp_path, identity)


def test_health_discloses_all_quarantines_without_a_session_start(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, "a" * 64))
    assert "no capture attempts" in runtime.health_summary("default")
    identity = str(uuid4())
    runtime._record_result(tmp_path, identity, "unknown_record", success=False)
    assert "EVERY" in runtime.health_summary("default")
    assert "1 quarantined sessions" in runtime.health_summary("default")
    runtime._record_result(tmp_path, identity, "unknown_record", success=False)
    assert "1 quarantined sessions" in runtime.health_summary("default")
    runtime._record_result(tmp_path, identity, None, success=True)
    assert "0 quarantined" in runtime.health_summary("default")


@pytest.mark.asyncio
async def test_catchup_fairness_excludes_current_and_deposit_before_clear(tmp_path, monkeypatch):
    from particles.api.cli import _codex_files
    from particles.config import get_config

    ids = sorted(str(uuid4()) for _ in range(4))
    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, "a" * 64))
    monkeypatch.setattr(
        _codex_files, "inventory", lambda home: [tmp_path / f"rollout-2026-{i}.jsonl" for i in ids]
    )
    get_config().codex.harvest.catchup_limit = 1
    recorded = []

    async def capture(store, identity, cwd, visiting, **kw):
        recorded.append(identity)
        assert identity != ids[0]
        return True

    monkeypatch.setattr(runtime, "_capture_one", capture)
    for identity in ids:
        write_json(
            tmp_path / "pending" / f"{identity}.json",
            {"session_id": identity, "store_id": "a" * 64, "home": str(tmp_path)},
        )
    for _ in range(6):
        await runtime.capture_worker("default", ids[0], str(tmp_path), True)
    assert set(recorded) == set(ids[1:])
    assert (tmp_path / "pending" / f"{ids[0]}.json").exists()
    assert not (tmp_path / "pending" / f"{ids[-1]}.json").exists()


@pytest.mark.asyncio
async def test_missing_source_retains_hint_and_failed_deposit_never_clears(tmp_path, monkeypatch):
    identity = str(uuid4())
    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, "a" * 64))

    async def missing(*a, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(runtime, "_capture_one", missing)
    await runtime.capture_worker("default", identity, str(tmp_path), False)
    hint = tmp_path / "pending" / f"{identity}.json"
    assert read_json(hint)["failure"] == "source_lost_before_capture"

    async def failed(*a, **kw):
        raise RuntimeError("private sql/raw content never printed")

    monkeypatch.setattr(runtime, "_capture_one", failed)
    await runtime.capture_worker("default", identity, str(tmp_path), False)
    assert read_json(hint)["failure"] == "capture_failed"
    assert "sql/raw content" not in hint.read_text()


def test_physical_store_excludes_database_credentials_and_handle_aliases(tmp_path):
    first = runtime.physical_store("postgresql+asyncpg://one:secret@LOCALHOST/test")
    second = runtime.physical_store("postgresql://two:changed@localhost:5432/test")
    assert first == second
    assert runtime.physical_store(
        f"sqlite+aiosqlite:///{tmp_path}/a/../b.db"
    ) == runtime.physical_store(f"sqlite:///{tmp_path}/b.db")


def test_operator_dismissal_acknowledges_only_validated_hint(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from particles.api.cli import app

    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, "a" * 64))
    identity = str(uuid4())
    hint = tmp_path / "pending" / f"{identity}.json"
    write_json(hint, {"failure": "source_lost_before_capture"})
    corpus = tmp_path / "retained-corpus.txt"
    corpus.write_text("previous capture")
    runner = CliRunner()
    result = runner.invoke(app, ["hook", "codex-dismiss", "--session", "../retained-corpus.txt"])
    assert result.exit_code == 1 and corpus.exists() and hint.exists()
    result = runner.invoke(app, ["hook", "codex-dismiss", "--session", identity])
    assert result.exit_code == 0 and not hint.exists() and corpus.read_text() == "previous capture"
    assert runner.invoke(app, ["hook", "codex-dismiss", "--session", identity]).exit_code == 1


@pytest.mark.asyncio
async def test_real_worker_commits_contained_source_before_clearing_hint(cli_db, tmp_path):
    from particles.config import get_config
    from particles.db import session_scope
    from particles.store.codex_capture_store import get_capture
    from tests.test_codex_transcript import SID, blob, message, metadata

    cfg = get_config().codex
    cfg.home_dir = str(tmp_path / "home")
    cfg.state_dir = str(tmp_path / "state")
    home, state, store_id = runtime.paths("default")
    home.mkdir(mode=0o700)
    (home / "sessions").mkdir(mode=0o700)
    source = home / "sessions" / f"rollout-synthetic-{SID}.jsonl"
    source.write_text(blob(metadata(), message("Synthetic human preference")))
    write_json(Path(cfg.state_dir) / "home.json", {"path": str(home), "id": str(uuid4())})
    hint = state / "pending" / f"{SID}.json"
    write_json(hint, {"session_id": SID, "home": str(home), "store_id": store_id})
    await runtime.capture_worker("default", SID, "/tmp/repository", False)
    async with session_scope() as session:
        captured = await get_capture(session, SID)
    assert captured is not None, read_json(state / "health.json")
    assert not hint.exists()
    assert captured.project_key is None  # unresolved is UNATTRIBUTED, never GLOBAL
    # A disk/SQL failure after an earlier commit retains the diagnostic hint.
    source.write_text(source.read_text().replace("Synthetic human", "Changed human"))
    await runtime.capture_worker("default", SID, "/tmp/repository", False)
    assert read_json(hint)["failure"] == "non_append_history"


@pytest.mark.asyncio
async def test_scope_initialization_changes_only_an_empty_store(cli_db):
    from particles.core.schema import Mutability
    from particles.corpus.deposit import deposit_text_versioned
    from particles.corpus.store import get_entry
    from particles.db import session_scope
    from particles.store.observer_scope_join import lens_may_engage

    async with session_scope(write=True) as session:
        entry, _snapshot, _ = await deposit_text_versioned(
            session,
            text="pre-existing operator corpus",
            uri_r="file:///tmp/operator.txt",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.MUTABLE,
            deposited_by="operator",
            tags=["original"],
        )
        await session.commit()
    assert not await runtime.initialize_empty_scope("default")
    async with session_scope() as session:
        assert not await lens_may_engage(session)
        assert (await get_entry(session, entry)).tags == ["original"]


@pytest.mark.asyncio
async def test_failed_fingerprint_write_prevents_recall_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from particles.api import client
    from particles.api.cli import _codex_files, _project_identity
    from particles.config import get_config

    get_config().codex.observer_scope = "store"
    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, "a" * 64))
    monkeypatch.setattr(runtime, "home_binding", lambda home: str(uuid4()))
    monkeypatch.setattr(
        _codex_files, "resolve_rollout", lambda *a: SimpleNamespace(metadata={"cwd": str(tmp_path)})
    )
    monkeypatch.setattr(_project_identity, "find_repository_root", lambda cwd: None)
    backend = SimpleNamespace(
        remote=False,
        digest_located=AsyncMock(return_value=SimpleNamespace(markdown="Synthetic digest\n")),
    )
    monkeypatch.setattr(client, "get_backend", lambda: backend)
    monkeypatch.setattr(runtime, "write_json", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        await runtime.recall("default", str(uuid4()), str(tmp_path))


@pytest.mark.asyncio
async def test_resume_discloses_failures_without_reading_a_rollout_and_ack_is_signature_bound(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, "a" * 64))
    monkeypatch.setattr(
        runtime, "home_binding", Mock(side_effect=AssertionError("resume reads no rollout"))
    )
    identity = str(uuid4())
    hint = tmp_path / "pending" / f"{identity}.json"
    write_json(hint, {"failure": "source_lost_before_capture"})
    result = json.loads(await runtime.recall("default", identity, str(tmp_path), "resume"))
    assert "systemMessage" in result and "1 lost" in result["systemMessage"]
    assert "additionalContext" not in json.dumps(result)
    assert "particles hook doctor --harness codex --store default" in result["systemMessage"]
    runtime.acknowledge_warning("default")
    assert hint.exists() and runtime.capture_warning("default") is None
    write_json(hint, {"failure": "unknown_record_type"})
    assert "1 quarantined" in runtime.capture_warning("default")
    runtime.acknowledge_warning("default")
    write_json(tmp_path / "pending" / f"{uuid4()}.json", {"failure": "unknown_record_type"})
    assert "2 quarantined" in runtime.capture_warning("default")


def test_thin_launcher_separates_pins_and_capture_opt_in(tmp_path, monkeypatch):
    args = arguments(tmp_path, "SessionStart")
    args.thin = True
    args.recall_token_file = "/checked/recall"
    args.capture_token_file = "/checked/capture"
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "inherited token")
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN_FILE", "/inherited")
    monkeypatch.setenv("PARTICLES_API_KEY", "inherited operator")
    popen = Mock()
    recall = Mock(return_value=Mock(returncode=0, stdout=b"", stderr=b""))
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.subprocess, "run", recall)
    monkeypatch.setattr(launcher.sys, "stdout", Mock(buffer=io.BytesIO()))
    event = {
        "session_id": str(uuid4()),
        "cwd": str(tmp_path),
        "hook_event_name": "SessionStart",
    }
    launcher.handoff(args, event)
    for consumer, mock in [("capture", popen), ("recall", recall)]:
        assert mock.call_args.kwargs["env"]["PARTICLES_ENGINE_TOKEN_FILE"] == f"/checked/{consumer}"
        assert not mock.call_args.kwargs["env"]["PARTICLES_ENGINE_TOKEN"]
        assert "PARTICLES_API_KEY" not in mock.call_args.kwargs["env"]
        assert mock.call_args.args[0][-2:] == ["--destination", args.store_id]

    popen.reset_mock()
    args.capture_token_file = None
    launcher.handoff(args, event)
    popen.assert_not_called()
    assert recall.call_count == 2
    assert not list(tmp_path.rglob("pending"))


@pytest.mark.asyncio
async def test_remote_capture_rechecks_head_after_race_and_reuses_committed_snapshot(tmp_path):
    import hashlib
    from unittest.mock import AsyncMock

    from particles.api.cli._codex_remote import push_capture
    from particles.api.client.http import EngineHttpError, HttpBackend
    from particles.extraction.codex_transcript import CONTRACT, parse_transcript
    from tests.test_codex_capture import HOME_ID, raw_source

    identity = str(uuid4())
    raw = raw_source(identity)
    transcript = parse_transcript(raw.decode())
    text, messages = transcript.render(lambda body: body)
    capture_id = str(uuid4())
    head = {
        "session_id": identity,
        "home_path": str(tmp_path),
        "home_id": HOME_ID,
        "launch_root": None,
        "project_key": None,
        "contract": CONTRACT,
        "raw_size": len(raw),
        "raw_hash": hashlib.sha256(raw).hexdigest(),
        "text_size": len(text.encode()),
        "text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "messages": messages,
        "entry_id": str(uuid4()),
        "snapshot_id": str(uuid4()),
        "capture_id": capture_id,
    }
    remote = AsyncMock(spec=HttpBackend)
    remote.codex_capture_proof.side_effect = [None, {"manifest": head}]
    stale = EngineHttpError("codex_capture_head_changed")
    stale.status_code = 409
    remote.deposit_text_at_uri.side_effect = [stale, None]
    await push_capture(
        remote,
        "default",
        home=tmp_path,
        home_id=HOME_ID,
        raw=raw,
        transcript=transcript,
        root=None,
        project=None,
        boundary=None,
    )
    first, retry = remote.deposit_text_at_uri.call_args_list
    assert first.kwargs["codex_capture"]["expected_capture_id"] is None
    assert retry.kwargs["codex_capture"]["expected_capture_id"] == capture_id
    assert retry.kwargs["codex_capture"]["prior_raw_hash"] == head["raw_hash"]
    assert retry.kwargs["text"] == text
    assert retry.kwargs["codex_capture"]["messages"] == [
        {"id": turn.message_id, "body_length": len(turn.text)} for turn in transcript.turns
    ]
    assert raw not in str(remote.mock_calls).encode()


def test_codex_hook_log_handles_missing_engine_install(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from particles.api.cli import app
    from particles.config import reset_config

    config = tmp_path.resolve() / "config.yaml"
    config.write_text(
        f"engine:\n  base_url: http://127.0.0.1:8000\ncodex:\n  state_dir: {tmp_path.resolve()}\n"
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    reset_config()
    result = CliRunner().invoke(app, ["hook", "log", "--harness", "codex"])
    assert result.exit_code == 1
    assert "Codex doctor:" in result.stderr
    assert "init codex" in result.stderr
    assert isinstance(result.exception, SystemExit)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_load", [True, False])
async def test_remote_worker_records_key_failure_once_per_candidate(
    tmp_path, monkeypatch, fail_load
):
    from unittest.mock import AsyncMock

    from particles.api.cli import _codex_remote
    from particles.config import get_config
    from particles.secrets import EngineTokenFileError

    cfg = get_config()
    monkeypatch.setattr(cfg.engine, "base_url", "http://127.0.0.1:8000")
    monkeypatch.setattr(cfg.codex.harvest, "enabled", True)
    monkeypatch.setattr(cfg.codex.harvest, "allow_remote", True)
    store_id = "a" * 64
    monkeypatch.setattr(runtime, "paths", lambda store: (tmp_path, tmp_path, store_id))
    hints = {}
    for _ in range(2):
        candidate = str(uuid4())
        hint = tmp_path / "pending" / f"{candidate}.json"
        write_json(hint, {"session_id": candidate, "store_id": store_id, "home": str(tmp_path)})
        hints[candidate] = hint
    backend = object()
    load = Mock(return_value=(backend, {}))
    prove = AsyncMock(side_effect=_codex_remote.CodexKeyRefused("synthetic invalid proof"))
    capture = AsyncMock()
    if fail_load:
        load.side_effect = EngineTokenFileError("/synthetic", "missing", "not found")
    monkeypatch.setattr(_codex_remote, "installed_backend", load)
    monkeypatch.setattr(_codex_remote, "prove_installed_key", prove)
    monkeypatch.setattr(runtime, "_capture_one", capture)

    await runtime.capture_worker("default", str(uuid4()), str(tmp_path), catch_up=True)

    load.assert_called_once_with("default", "capture")
    if fail_load:
        prove.assert_not_awaited()
        reason = "capture_key_file_refused"
    else:
        prove.assert_awaited_once_with(backend, "capture", {})
        reason = "capture_key_or_protocol_refused"
    capture.assert_not_awaited()
    health = read_json(tmp_path / "health.json")
    assert health["completed_quarantines"] == len(hints)
    for candidate, hint in hints.items():
        assert read_json(hint)["failure"] == reason
        assert health["quarantined"][candidate]["reason"] == reason
