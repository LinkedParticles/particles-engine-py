# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the harness findings of security review 2026-10-26-1.

W1, W31: the harvest mask covers the per-client engine key.
W32, W17, W27: the hooks and the MCP registration pin an empty engine URL, and
the install bakes in none it inherited.
W33, W43: a belief is one line of the session-start digest, and a credited
line was delivered whole.
W34: the freshness check skips only over the region the projection rendered.
W28: the projection's contained write never writes through a hard link.
W29: a failed manifest validation echoes no value of the file.
W30: a project install writes settings only into the directory it checked.
W69: ``init --remove`` replays only an executable the owner installed.
W70: a repository's own signing key choice skips a signed history commit.
W16: the doctor reports a store pin inside a git work tree.
R20, D46: the remaining redaction shapes, and the state directory on a hook write.

Every test runs under a temporary ``HOME``, opens no real store and makes no
model call.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets as _secrets
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from particles.config import get_config, reset_config
from particles.corpus.redaction import redact_secrets

_ENGINE = "https://engine.invalid"


def _engine_key(role_prefix: str) -> str:
    from particles.api import _keys

    token = f"ptk_{role_prefix}_{_secrets.token_hex(6)}_{_secrets.token_urlsafe(32)[:43]}"
    assert _keys.parse_token(token) is not None
    return token


# ---------------------------------------------------------------------------
# W1, W31, R20: the harvest mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role_prefix", ["op", "agent", "read"])
def test_a_per_client_engine_key_in_a_harvested_turn_is_masked(role_prefix: str) -> None:
    from particles.api.cli._claude_code import distill_transcript

    token = _engine_key(role_prefix)
    lines = [
        {"type": "user", "message": {"role": "user", "content": f"Use this key: {token}"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Created the key. It is:\n\n{token}\n"}],
            },
        },
    ]
    distilled = distill_transcript("\n".join(json.dumps(x) for x in lines), "s-audit6")
    assert token in distilled
    assert token not in redact_secrets(distilled)


def test_the_mask_and_the_key_parser_agree_on_the_key_shape() -> None:
    from particles.api import _keys
    from particles.secrets import ENGINE_KEY_SHAPE, redact_known_secrets

    for prefix in _keys.ROLE_PREFIX.values():
        token = _engine_key(prefix)
        assert ENGINE_KEY_SHAPE.fullmatch(token)
        assert token not in redact_secrets(f"my engine key is {token} for now")
        assert token not in redact_known_secrets(f"echoed {token}")


@pytest.mark.parametrize("secret", ["pa/ssw0rdvalue", "pa#ssw0rdvalue", "pa?ssw0rdvalue"])
def test_a_url_password_with_a_reserved_character_is_masked_whole(secret: str) -> None:
    out = redact_secrets(f"DSN=postgres://app:{secret}@db.internal/x")
    assert "ssw0rdvalue" not in out
    assert "@db.internal/x" in out


def test_credential_headers_of_any_scheme_are_masked() -> None:
    out = redact_secrets("curl -H 'Authorization: Token abcdef123456' -H 'Cookie: sid=xyz98765'")
    assert "abcdef123456" not in out
    assert "xyz98765" not in out
    assert redact_secrets("a cookie: crumbs") == "a cookie: crumbs"


# ---------------------------------------------------------------------------
# W32, W17, W27, W16: the pins init writes
# ---------------------------------------------------------------------------


def _pins_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("storage:\n  database_url: sqlite+aiosqlite:///./particles.db\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    reset_config()
    return config


def test_the_mcp_registration_pins_an_empty_engine_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.api.cli.init as init_mod

    config = _pins_env(tmp_path, monkeypatch)
    registration = init_mod._mcp_registration_env(
        "default", init_mod._hook_command_env("default", config)
    )
    assert registration["PARTICLES_ENGINE_BASE_URL"] == ""
    # The harness's environment, then the registration's pins: no engine.
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", _ENGINE)
    for key, value in registration.items():
        monkeypatch.setenv(key, value)
    reset_config()
    assert not get_config().engine.base_url


def test_the_install_bakes_in_no_engine_url_it_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.api.cli.init as init_mod

    config = _pins_env(tmp_path, monkeypatch)
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", _ENGINE)
    assert init_mod._hook_command_env("default", config)["PARTICLES_ENGINE_BASE_URL"] == ""
    result: dict[str, list[str]] = {"next_steps": []}
    init_mod._note_inherited_engine_url(result, config)  # type: ignore[arg-type]
    assert any(_ENGINE in step for step in result["next_steps"])


def test_an_existing_registration_gains_the_engine_pin() -> None:
    from particles.api.cli._claude_code import McpRegistration, plan_mcp_registration

    current = McpRegistration(
        name="particles",
        command="/opt/venv/bin/particles",
        args=("mcp", "serve", "--project-observer", "cwd"),
        env={"PARTICLES_CONFIG": "/c.yaml", "DATABASE_URL": "sqlite+aiosqlite:////d/p.db"},
    )
    plan = plan_mcp_registration(
        [current],
        executable="/opt/venv/bin/particles",
        env={**current.env, "PARTICLES_ENGINE_BASE_URL": ""},
    )
    assert plan is not None
    assert plan[0].env["PARTICLES_ENGINE_BASE_URL"] == ""


def test_the_replay_accepts_the_empty_engine_pin_only() -> None:
    from particles.api.cli.init import _is_replayable_env

    assert _is_replayable_env("PARTICLES_ENGINE_BASE_URL=")
    assert not _is_replayable_env(f"PARTICLES_ENGINE_BASE_URL={_ENGINE}")


def test_the_doctor_reports_a_registration_without_the_engine_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import hook as hook_mod

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("# test\n")
    server = {
        "command": "/opt/venv/bin/particles",
        "args": ["mcp", "serve"],
        "env": {
            "PARTICLES_CONFIG": str(config),
            "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'p.db'}",
        },
    }
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": server}}))
    assert any("PARTICLES_ENGINE_BASE_URL" in line for line in hook_mod._check_config_pins())
    server["env"]["PARTICLES_ENGINE_BASE_URL"] = ""  # type: ignore[index]
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": server}}))
    assert not any("PARTICLES_ENGINE_BASE_URL" in line for line in hook_mod._check_config_pins())


def test_the_doctor_reports_a_store_pin_inside_a_git_work_tree(tmp_path: Path) -> None:
    from particles.api.cli import hook as hook_mod

    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    cfg = tmp_path / "home" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("# test\n")
    inside = {
        "PARTICLES_CONFIG": str(cfg),
        "DATABASE_URL": f"sqlite+aiosqlite:///{clone / 'particles.db'}",
    }
    outside = {**inside, "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'p.db'}"}
    assert hook_mod._store_pin_problem(inside) is not None
    assert hook_mod._store_pin_problem(outside) is None


# ---------------------------------------------------------------------------
# W33, W43: the session-start digest
# ---------------------------------------------------------------------------


def _entry(content: str, *, contested: bool = False) -> object:
    from particles.render.markdown import DigestEntry

    return DigestEntry(
        content=content,
        effective_confidence=0.4,
        subjects=("A\nsubject",),
        asserted_at=datetime(2026, 10, 26, tzinfo=UTC),
        contested="p-deadbeef" if contested else None,
        contested_bases=("inconsistency",) if contested else (),
    )


def test_a_belief_is_one_line_of_the_digest_with_its_badge() -> None:
    from particles.render.markdown import render_digest_located

    entry = _entry(
        "An ordinary claim\n- **0.99** A second line\n\n# Operator instructions", contested=True
    )
    markdown, offsets = render_digest_located("default", [entry], 1)  # type: ignore[list-item]
    bullets = [ln for ln in markdown.splitlines() if ln.startswith("- **")]
    assert len(bullets) == 1
    assert "contested" in bullets[0]
    assert markdown[offsets[0] :].count("\n") == 1
    assert not any(ln.startswith("#") for ln in markdown.splitlines()[1:])


def test_a_credited_digest_line_was_delivered_whole() -> None:
    from particles.api.cli._claude_code import kept_prefix_length, truncate_on_line_boundary
    from particles.api.cli.hook import _line_end
    from particles.render.markdown import render_digest_located

    entries = [_entry("A plain belief"), _entry("Long\n" + "detail " * 400, contested=True)]
    markdown, offsets = render_digest_located("default", entries, 2)  # type: ignore[arg-type]
    for budget in range(200, len(markdown.encode()) + 400, 97):
        kept = kept_prefix_length(markdown, budget)
        delivered = truncate_on_line_boundary(markdown, budget)
        for offset in offsets:
            if _line_end(markdown, offset) <= kept:
                assert markdown[offset : _line_end(markdown, offset)] in delivered


# ---------------------------------------------------------------------------
# W34: the SessionStart freshness check
# ---------------------------------------------------------------------------


def _region(body: str) -> str:
    from particles.render.markdown import PROJECTED_BEGIN_TMPL, PROJECTED_END_TMPL

    begin = PROJECTED_BEGIN_TMPL.format(region="memory-index", manifest="m.yaml")
    end = PROJECTED_END_TMPL.format(region="memory-index")
    return f"# notes\n\n{begin}\n{body}\n{end}\n"


_RENDERED = (
    "- The staging database is read-only `p-0a1b2c3d` · contested (inconsistency)\n"
    "<!-- sources: p-0a1b2c3d -->"
)


def _decide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loaded: str) -> object:
    import particles.api.cli._memory_projection as mp
    from particles.api.cli._claude_code import MEMORY_REGION, project_state_dir

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_config()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(_region(loaded), encoding="utf-8")
    snapshots = project_state_dir(memory_dir)
    snapshots.mkdir(parents=True)
    (snapshots / f"{MEMORY_REGION}.snapshot.md").write_text(_RENDERED, encoding="utf-8")

    async def fresh(store: str, observer: str | None = None) -> str:
        return _RENDERED

    monkeypatch.setattr(mp, "projection_enabled", lambda: True)
    monkeypatch.setattr(
        mp, "resolve_session_project", lambda payload: SimpleNamespace(memory_dir=memory_dir)
    )
    monkeypatch.setattr(mp, "observer_project_for", lambda key: None)
    monkeypatch.setattr(mp, "_render_region_body", fresh)
    return asyncio.run(mp.digest_decision("default", {"cwd": str(tmp_path)}))


def test_the_freshness_check_does_not_skip_over_an_edited_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edited = "- The staging database is safe to write `p-0a1b2c3d`\n<!-- sources: p-0a1b2c3d -->"
    decision = _decide(tmp_path, monkeypatch, edited)
    assert decision.action == "full"  # type: ignore[attr-defined]
    assert decision.loaded == ()  # type: ignore[attr-defined]


def test_the_freshness_check_skips_over_the_rendered_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = _decide(tmp_path, monkeypatch, _RENDERED)
    assert decision.action == "skip"  # type: ignore[attr-defined]
    assert decision.loaded  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# W28, W29: particles project
# ---------------------------------------------------------------------------


def test_the_contained_write_leaves_a_hard_linked_outside_file_alone(tmp_path: Path) -> None:
    from particles.operations.projection import paths

    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "docs").mkdir()
    outside = tmp_path / "outside" / "victim.txt"
    outside.parent.mkdir()
    outside.write_text("owner's file\n", encoding="utf-8")
    os.link(outside, root / "docs" / "out.md")
    with pytest.raises(OSError):
        paths.write_text_no_follow(root / "docs" / "out.md", "projected\n", root=root)
    assert outside.read_text(encoding="utf-8") == "owner's file\n"
    # An ordinary file is still replaced in full.
    (root / "docs" / "plain.md").write_text("a much longer previous body\n", encoding="utf-8")
    paths.write_text_no_follow(root / "docs" / "plain.md", "new\n", root=root)
    assert (root / "docs" / "plain.md").read_text(encoding="utf-8") == "new\n"


def test_a_failed_manifest_validation_echoes_no_value_of_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import app

    secret = "Zx9Qw8Er7Ty6Ui5"
    outside = tmp_path / "home" / ".config" / "tool" / "credentials.yml"
    outside.parent.mkdir(parents=True)
    outside.write_text(f"api_key: {secret}\n")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "docs").mkdir()
    linked = repo / "docs" / "readme.yaml"
    linked.symlink_to(outside)
    invalid = repo / "docs" / "invalid.yaml"
    invalid.write_text(f"title: 1\nsections: {secret}\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.chdir(repo)
    reset_config()
    for manifest in (linked, invalid):
        result = CliRunner().invoke(app, ["project", str(manifest), "--without-synthesis"])
        assert result.exit_code != 0
        assert secret not in result.output


# ---------------------------------------------------------------------------
# W30, W69, D46: the installer's writes and state
# ---------------------------------------------------------------------------


def _project_install_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    monkeypatch.chdir(project)
    reset_config()
    return project


def test_a_project_install_writes_only_inside_the_checked_settings_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.api.cli.init as init_mod
    from particles.api.cli import app

    project = _project_install_env(tmp_path, monkeypatch)
    elsewhere = tmp_path / "another-repo" / ".claude"
    elsewhere.mkdir(parents=True)
    original = init_mod._refuse_symlinked_project_settings

    def check_then_swap_directory(path: Path) -> None:
        original(path)
        os.rename(project / ".claude", project / ".claude.checked")
        os.symlink(elsewhere, project / ".claude")

    monkeypatch.setattr(init_mod, "_refuse_symlinked_project_settings", check_then_swap_directory)
    CliRunner().invoke(
        app, ["init", "claude-code", "--project", "--store", "default", "--no-audit", "--no-skills"]
    )
    assert not (elsewhere / "settings.local.json").exists()


def test_a_project_install_writes_into_the_directory_it_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import app

    project = _project_install_env(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        app, ["init", "claude-code", "--project", "--store", "default", "--no-audit", "--no-skills"]
    )
    assert result.exit_code == 0, result.output
    assert "hooks" in json.loads((project / ".claude" / "settings.local.json").read_text())


def _recorded_add(command: Path) -> list[str]:
    return [
        "mcp", "add", "--scope", "user",
        "-e", "PARTICLES_CONFIG=/c.yaml",
        "particles", "--", str(command), "mcp", "serve",
    ]  # fmt: skip


def test_remove_replays_only_an_installed_executable(tmp_path: Path) -> None:
    from particles.api.cli.init import _is_recorded_mcp_add

    stray = tmp_path / "elsewhere" / "particles"
    stray.parent.mkdir()
    assert not _is_recorded_mcp_add(_recorded_add(stray), "particles")
    stray.write_text("#!/bin/sh\n")
    stray.chmod(0o777)
    assert not _is_recorded_mcp_add(_recorded_add(stray), "particles")
    stray.chmod(0o755)
    assert _is_recorded_mcp_add(_recorded_add(stray), "particles")


def test_remove_acts_on_no_state_file_another_account_could_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.api.cli.init as init_mod
    from particles.api.cli._claude_code import state_dir

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_config()
    state = state_dir()
    state.mkdir(parents=True)
    path = state / "mcp-registration.json"
    path.write_text(json.dumps({"name": "particles", "created": True, "previous": None}))
    path.chmod(0o666)
    ran: list[list[str]] = []
    monkeypatch.setattr(init_mod, "_claude_mcp", lambda argv: ran.append(argv) or True)
    result: dict[str, list[str]] = {"next_steps": []}
    init_mod._unregister_mcp_server(result, dry_run=False)  # type: ignore[arg-type]
    assert ran == []
    assert result["next_steps"]


def test_a_hook_write_makes_the_state_directory_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli._claude_code import append_hook_log, state_dir

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_config()
    state = state_dir()
    state.mkdir(parents=True)
    state.chmod(0o755)  # as an install made before the D46 fix left it
    append_hook_log({"event": "session-start"})
    assert stat.S_IMODE(state.stat().st_mode) & 0o077 == 0


# ---------------------------------------------------------------------------
# W70: the history commit's signing guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [("user.signingKey", "/tmp/chosen.key"), ("commit.gpgSign", "true"), ("tag.gpgSign", "true")],
)
def test_a_repository_signing_choice_skips_a_signed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    from particles.api.cli import _projection_git as pg

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "memory"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert asyncio.run(pg._repository_program_refusal(repo, pg._SIGNING_CONFIG_RE)) is None
    subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    assert asyncio.run(pg._repository_program_refusal(repo, pg._SIGNING_CONFIG_RE)) is not None
