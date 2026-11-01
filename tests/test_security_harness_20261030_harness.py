# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Security review 2026-10-30-1: the installer's output, the SessionEnd harvest and the masks.

Regression tests for AB20, AB21, AB25, AB26, AB27, AB49, AB65, AB66 and AB67.
Every test runs under a temporary ``HOME`` with the configuration read through
the loader; no model call and no outside connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

_KEY = "ptk_agent_0123456789ab_" + "A" * 20 + "b" * 23  # a well-formed, fake agent key
_SECRET = _KEY.split("_", 3)[3]
_TOKEN = "ptk_agent_0123456789ab_" + "Q" * 43  # another fake agent key


def _user_text(text: str) -> str:
    return (
        json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": text}]}})
        + "\n"
    )


def _harvested(jsonl: str) -> str:
    """The text a SessionEnd harvest deposits for a transcript."""
    from particles.api.cli.hook import _distilled_session

    return _distilled_session(jsonl, "s1")


def _archived(text: str) -> str:
    """The memory archive's mask: the file's raw text through ``redact_secrets``."""
    from particles.corpus.redaction import redact_secrets

    return redact_secrets(text)


# ---------------------------------------------------------------------------
# AB20, AB27 (E33): the harvest mask
# ---------------------------------------------------------------------------


def test_the_fake_key_parses_and_written_whole_is_masked() -> None:
    from particles.api._keys import parse_token

    assert parse_token(_KEY) is not None
    assert _SECRET not in _harvested(_user_text(f"my token is {_KEY}"))


def test_a_six_or_seven_character_bearer_value_is_masked() -> None:
    """Only a Bearer value under six characters is left as written (AB20)."""
    text = _harvested(_user_text('headers = {"X-Auth": "Bearer Zq7xK2p"}'))
    assert "Zq7xK2p" not in text
    assert "Bearer [REDACTED]" in _archived("Bearer abc123")
    assert _archived("Bearer abc12") == "Bearer abc12"


def test_a_six_character_basic_credential_is_masked() -> None:
    """A ``Basic`` value of six base64 characters (``a:bc``) is masked too (AB20)."""
    assert (
        _archived("Authorization was Basic YTpiYw== here")
        == "Authorization was Basic [REDACTED] here"
    )
    assert _archived("Basic config") == "Basic config"


@pytest.mark.parametrize(
    "noise",
    ["​", "­", "⁠", "﻿", "‍"],
    ids=["zero-width-space", "soft-hyphen", "word-joiner", "bom", "zero-width-joiner"],
)
def test_a_key_split_by_an_invisible_character_is_masked(noise: str) -> None:
    """A per-client key split by an invisible character is masked in a transcript (AB27)."""
    split = _KEY[:30] + noise + _KEY[30:]
    text = _harvested(_user_text(f"token {split} done"))
    assert _KEY[30:] not in text
    assert _KEY[:30] not in text
    assert text.endswith("done") or "done" in text


def test_a_key_whose_separator_is_a_json_escape_is_masked_in_a_memory_archive() -> None:
    escaped = _KEY.replace("_", "\\u005f", 1)
    assert _SECRET not in _archived(f'{{"note": "{escaped}"}}')


def test_a_percent_encoded_key_is_masked_in_a_memory_archive() -> None:
    encoded = _KEY.replace("_", "%5F")
    masked = _archived(f"curl -d a=1&b={encoded} https://engine.invalid")
    assert _SECRET not in masked
    assert masked == "curl -d a=1&b=[REDACTED ENGINE KEY] https://engine.invalid"


def test_the_known_secret_mask_also_reads_a_split_key() -> None:
    from particles.secrets import redact_known_secrets

    split = _KEY[:30] + "​" + _KEY[30:]
    assert redact_known_secrets(f"error: {split}!") == "error: [REDACTED]!"


def test_text_with_invisible_characters_and_no_key_is_unchanged() -> None:
    text = "a​b %5F c \\u005f d &#95; e ­"
    assert _archived(text) == text


# ---------------------------------------------------------------------------
# AB21, AB25, AB26: what the install and the remove print
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.config import reset_config

    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    os.chmod(home, 0o700)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir(mode=0o700)
    cfg = cfgdir / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\n"
        "engine:\n  base_url: http://127.0.0.1:8000\n"
        f"claude_code:\n  state_dir: {home / '.particles' / 'claude-code'}\n",
        encoding="utf-8",
    )
    os.chmod(cfg, 0o600)
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    reset_config()
    yield home
    reset_config()


def _settings_with_pinned_token(home: Path) -> Path:
    settings = home / ".claude" / "settings.json"
    command = (
        f"env PARTICLES_ENGINE_TOKEN={_TOKEN} PARTICLES_ENGINE_BASE_URL= "
        "/opt/particles/bin/particles hook session-end --store default"
    )
    settings.write_text(
        json.dumps(
            {"hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": command}]}]}}
        ),
        encoding="utf-8",
    )
    return settings


def test_the_remove_result_shows_no_hand_pinned_engine_token(home: Path) -> None:
    """The --remove result, which --json prints, never carries a pinned engine token (AB26)."""
    from particles.api.cli import init

    settings = _settings_with_pinned_token(home)
    result: dict[str, Any] = init._remove(settings, dry_run=True)
    assert _TOKEN not in json.dumps(result)
    (shown,) = result["hooks_removed"]
    assert "PARTICLES_ENGINE_TOKEN='<hidden>'" in shown


def test_the_remove_dry_run_prints_no_pinned_engine_token(home: Path) -> None:
    """The --remove --dry-run narration never prints a token another hook pins (AB25)."""
    from particles.api.cli import app

    settings = _settings_with_pinned_token(home)
    data = json.loads(settings.read_text())
    data["hooks"]["Stop"] = [
        {"hooks": [{"type": "command", "command": f"env OTHER_TOOL_TOKEN={_TOKEN} other-tool"}]}
    ]
    settings.write_text(json.dumps(data), encoding="utf-8")
    out = CliRunner().invoke(
        app, ["init", "claude-code", "--remove", "--dry-run"], catch_exceptions=False
    )
    assert out.exit_code == 0, out.output
    assert _TOKEN not in out.output
    assert "other-tool" in out.output


def test_the_install_hook_map_hides_a_pinned_token() -> None:
    from particles.api.cli.init import _shown_command

    command = f"env PARTICLES_ENGINE_TOKEN={_TOKEN} /opt/particles/bin/particles hook session-end"
    assert _TOKEN not in _shown_command(command)


def test_a_human_install_run_shows_the_hook_key_mint_command(home: Path) -> None:
    """With engine.base_url set, a run without --json shows the keys create command (AB21)."""
    from particles.api.cli import app

    out = CliRunner().invoke(
        app,
        ["init", "claude-code", "--dry-run", "--no-audit", "--no-skills"],
        catch_exceptions=False,
    )
    assert out.exit_code == 0, out.output
    assert "keys create" in out.output


def test_the_json_result_names_the_mint_command(home: Path) -> None:
    from particles.api.cli import app

    out = CliRunner().invoke(
        app, ["init", "claude-code", "--dry-run", "--json", "--no-skills"], catch_exceptions=False
    )
    assert out.exit_code == 0, out.output
    assert "keys create" in out.stdout
    assert any("keys create" in step for step in json.loads(out.stdout)["next_steps"])


def test_a_dry_run_install_prints_no_engine_token_from_the_settings_env_block(home: Path) -> None:
    """The --dry-run narration never prints a token the settings file's env block holds (AB25)."""
    from particles.api.cli import app

    settings = home / ".claude" / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "env": {"PARTICLES_ENGINE_TOKEN": _TOKEN, "SOME_API_KEY": "sk-not-a-real-key-000"},
                "statusLine": {"type": "command", "command": f"env T={_TOKEN} status"},
            }
        ),
        encoding="utf-8",
    )
    out = CliRunner().invoke(
        app,
        ["init", "claude-code", "--dry-run", "--no-audit", "--no-skills"],
        catch_exceptions=False,
    )
    assert out.exit_code == 0, out.output
    assert _TOKEN not in out.output
    assert "sk-not-a-real-key-000" not in out.output
    assert '"PARTICLES_ENGINE_TOKEN": "<hidden>"' in out.output
    # The file itself is not touched by a dry run.
    assert _TOKEN in settings.read_text()


# ---------------------------------------------------------------------------
# AB49: one refused memory file does not end the harvest
# ---------------------------------------------------------------------------


def test_one_refused_memory_file_does_not_stop_the_rest_of_the_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 403 on one memory file still lets the harvest send the project's other files (AB49)."""
    import particles.api.client as client_mod
    from particles.api.cli import hook
    from particles.api.client.http import EngineHttpError

    memory_dir = tmp_path / "projects" / "repo" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "a.md").write_text("first file\n", encoding="utf-8")
    (memory_dir / "b.md").write_text("second file\n", encoding="utf-8")
    sent: list[str] = []

    class _Outcome:
        unchanged = False
        entry_id = "e"
        snapshot_id = "s"

    class _Backend:
        remote = True

        async def deposit_text_at_uri(self, **kw: Any) -> _Outcome:
            sent.append(kw["uri_r"])
            if len(sent) == 1:
                raise EngineHttpError(
                    "403 from /corpus/deposit/text: lies under none of its memory roots"
                )
            return _Outcome()

    monkeypatch.setattr(client_mod, "get_backend", lambda *a, **k: _Backend())
    refused: list[dict[str, str]] = []
    with contextlib.suppress(EngineHttpError):
        harvested, _, _ = asyncio.run(
            hook._harvest_memory_files("default", memory_dir, "repo", refused=refused)
        )
    assert len(sent) == 2, sent
    assert len(harvested) == 1
    assert [r["uri"] for r in refused] == [sent[0]]
    assert refused[0]["error"].startswith("403")


def test_an_unreachable_engine_still_ends_the_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a per-file refusal is recorded; an unreachable engine still stops the harvest."""
    import particles.api.client as client_mod
    from particles.api.cli import hook
    from particles.api.client.http import EngineUnreachableError

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "a.md").write_text("first file\n", encoding="utf-8")
    (memory_dir / "b.md").write_text("second file\n", encoding="utf-8")
    sent: list[str] = []

    class _Backend:
        remote = True

        async def deposit_text_at_uri(self, **kw: Any) -> None:
            sent.append(kw["uri_r"])
            raise EngineUnreachableError("could not reach the engine")

    monkeypatch.setattr(client_mod, "get_backend", lambda *a, **k: _Backend())
    with pytest.raises(EngineUnreachableError):
        asyncio.run(hook._harvest_memory_files("default", memory_dir, "repo"))
    assert len(sent) == 1


def test_the_doctor_counts_the_last_session_ends_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    import particles.api.cli._claude_code as cc
    from particles.api.cli import hook

    log = [
        json.dumps(
            {"event": "session-end", "refused": [{"uri": "file:///a"}, {"uri": "file:///b"}]}
        ),
        json.dumps({"event": "session-start"}),
    ]
    monkeypatch.setattr(cc, "read_hook_log_tail", lambda n: log)
    assert hook._last_harvest_refusals() == 2
    log.append(json.dumps({"event": "session-end", "deposited": 3}))
    assert hook._last_harvest_refusals() == 0


# ---------------------------------------------------------------------------
# AB65, AB66: the claude stand-ins refuse what the real CLI refuses
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    return home


def test_a_registration_the_recorder_accepts_is_seen_by_the_next_install(bare_home: Path) -> None:
    """After the autouse recorder accepts an mcp add, the installer's reader sees it (AB65)."""
    import particles.api.cli.init as init_mod
    from particles.api.cli._claude_code import particles_mcp_registrations

    assert init_mod._claude_mcp(
        [
            "mcp",
            "add",
            "--scope",
            "user",
            "particles",
            "-e",
            "A=1",
            "--",
            "/opt/bin/particles",
            "mcp",
            "serve",
        ]
    )
    (registration,) = particles_mcp_registrations()
    assert registration.name == "particles"
    assert registration.env == {"A": "1"}
    assert init_mod._claude_mcp(["mcp", "remove", "--scope", "user", "particles"])
    assert particles_mcp_registrations() == []


def _run_stub(stub: Path, home: Path, *argv: str) -> int:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    return subprocess.run([str(stub), *argv], env=env, capture_output=True, timeout=60).returncode


@pytest.fixture
def stub(tmp_path: Path) -> Path:
    from tests.test_security_harness_20261027 import _STUB

    path = tmp_path / "claude"
    path.write_text(_STUB.format(python=sys.executable))
    path.chmod(0o755)
    return path


def test_the_installer_stub_refuses_a_remove_under_a_scope_the_add_did_not_use(
    bare_home: Path, stub: Path
) -> None:
    """The subprocess stand-in refuses a remove under a scope the add did not use (AB66)."""
    assert (
        _run_stub(
            stub, bare_home, "mcp", "add", "--scope", "user", "particles", "/opt/bin/particles"
        )
        == 0
    )
    assert _run_stub(stub, bare_home, "mcp", "remove", "--scope", "local", "particles") != 0
    assert _run_stub(stub, bare_home, "mcp", "remove", "--scope", "user", "particles") == 0


def test_the_installer_stub_keeps_a_local_scope_add_out_of_the_user_servers(
    bare_home: Path, stub: Path
) -> None:
    """A local-scope add lands under its project, not among the user-scope servers (AB66)."""
    assert (
        _run_stub(
            stub, bare_home, "mcp", "add", "--scope", "local", "particles", "/opt/bin/particles"
        )
        == 0
    )
    data = json.loads((bare_home / ".claude.json").read_text())
    assert "particles" not in data.get("mcpServers", {})
    assert _run_stub(stub, bare_home, "mcp", "remove", "--scope", "user", "particles") != 0


def test_the_installer_tests_run_the_real_trusted_path_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stub runner restores the installer's own ``_claude_mcp`` body (AB66)."""
    import particles.api.cli.init as init_mod
    from tests.test_security_harness_20261027 import _install_stub

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    _install_stub(tmp_path, monkeypatch)
    assert init_mod._claude_mcp.__name__ == "_claude_mcp"
    assert init_mod._claude_mcp(
        ["mcp", "add", "--scope", "user", "particles", "/opt/bin/particles"]
    )
    assert "particles" in json.loads((home / ".claude.json").read_text())["mcpServers"]


# ---------------------------------------------------------------------------
# AB67: the project settings write lands only in the directory the install made
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_checks() -> Generator[Any, None, None]:
    from particles.api.cli import init

    init._CHECKED_SETTINGS_DIRECTORIES.clear()
    init._CHECKED_PROJECT_SETTINGS.clear()
    yield init
    init._CHECKED_SETTINGS_DIRECTORIES.clear()
    init._CHECKED_PROJECT_SETTINGS.clear()


def test_the_settings_write_lands_only_in_the_directory_the_install_created(
    tmp_path: Path, clean_checks: Any
) -> None:
    """A .claude swapped in between the install's mkdir and its write is refused (AB67)."""
    init = clean_checks
    directory = tmp_path / "repo" / ".claude"
    init._refuse_foreign_settings_directory(directory)  # absent at the check: recorded as None
    directory.mkdir(parents=True)  # the install's mkdir
    directory.rename(tmp_path / "repo" / "moved")  # the swap
    directory.mkdir()
    with pytest.raises(typer.Exit):
        init._write_settings(directory / "settings.local.json", "{}\n")
    assert not (directory / "settings.local.json").exists()


def test_the_install_writes_into_the_directory_it_created(
    tmp_path: Path, clean_checks: Any
) -> None:
    init = clean_checks
    directory = tmp_path / "repo" / ".claude"
    init._refuse_foreign_settings_directory(directory)
    init._make_settings_directory(directory / "settings.local.json")
    init._write_settings(directory / "settings.local.json", "{}\n")
    assert (directory / "settings.local.json").read_text() == "{}\n"


def test_a_settings_directory_that_appeared_after_the_check_is_refused(
    tmp_path: Path, clean_checks: Any
) -> None:
    init = clean_checks
    directory = tmp_path / "repo" / ".claude"
    init._refuse_foreign_settings_directory(directory)
    directory.mkdir(parents=True)
    with pytest.raises(typer.Exit):
        init._make_settings_directory(directory / "settings.local.json")
