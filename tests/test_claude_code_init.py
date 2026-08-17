"""Tests for ``particles init claude-code`` + the settings/config surgeons.

Covers the ADR's installer checklist: settings-merge idempotence, ``--remove``
surgicality, foreign-entry preservation, unparseable-settings refusal,
``--dry-run``, and the fresh-install store auto-create including config.yaml
preservation (parse-preserve-append, never clobber).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import (
    ConfigEditError,
    absolutize_sqlite_dsn,
    build_hook_commands,
    disable_memory_store_text,
    enable_memory_store_text,
    merge_particles_hook_entries,
    render_env_prefix,
    strip_particles_hook_entries,
)

DSN = "sqlite+aiosqlite:////tmp/x/memory.db"
COMMANDS = build_hook_commands("/opt/venv/bin/particles", "memory")

FOREIGN_SETTINGS: dict[str, Any] = {
    "model": "opus",
    "env": {"FOO": "bar"},
    "hooks": {
        "SessionStart": [
            {"hooks": [{"type": "command", "command": "echo my-own-hook"}]},
        ],
        "PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "my-guard"}]},
        ],
    },
}


# ---------------------------------------------------------------------------
# Settings merge (pure helpers)
# ---------------------------------------------------------------------------


class TestSettingsMerge:
    def test_merge_into_empty(self) -> None:
        merged = merge_particles_hook_entries({}, COMMANDS)
        assert [h["command"] for g in merged["hooks"]["SessionStart"] for h in g["hooks"]] == [
            COMMANDS["SessionStart"]
        ]
        assert [h["command"] for g in merged["hooks"]["SessionEnd"] for h in g["hooks"]] == [
            COMMANDS["SessionEnd"]
        ]

    def test_merge_is_idempotent(self) -> None:
        once = merge_particles_hook_entries(FOREIGN_SETTINGS, COMMANDS)
        twice = merge_particles_hook_entries(once, COMMANDS)
        assert once == twice

    def test_rerun_replaces_our_entries(self) -> None:
        old = merge_particles_hook_entries({}, build_hook_commands("/old/particles", "memory"))
        new = merge_particles_hook_entries(old, COMMANDS)
        commands = [
            h["command"] for groups in new["hooks"].values() for g in groups for h in g["hooks"]
        ]
        assert COMMANDS["SessionStart"] in commands
        assert not any("/old/particles" in c for c in commands)

    def test_foreign_entries_and_keys_preserved(self) -> None:
        merged = merge_particles_hook_entries(FOREIGN_SETTINGS, COMMANDS)
        assert merged["model"] == "opus"
        assert merged["env"] == {"FOO": "bar"}
        assert merged["hooks"]["PreToolUse"] == FOREIGN_SETTINGS["hooks"]["PreToolUse"]
        starts = [h["command"] for g in merged["hooks"]["SessionStart"] for h in g["hooks"]]
        assert "echo my-own-hook" in starts
        # And the original input was not mutated.
        assert "SessionEnd" not in FOREIGN_SETTINGS["hooks"]

    def test_strip_is_surgical(self) -> None:
        merged = merge_particles_hook_entries(FOREIGN_SETTINGS, COMMANDS)
        stripped = strip_particles_hook_entries(merged)
        assert stripped == FOREIGN_SETTINGS

    def test_strip_removes_our_hook_from_shared_group(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo foreign"},
                            {"type": "command", "command": COMMANDS["SessionStart"]},
                        ]
                    }
                ]
            }
        }
        stripped = strip_particles_hook_entries(settings)
        assert stripped["hooks"]["SessionStart"] == [
            {"hooks": [{"type": "command", "command": "echo foreign"}]}
        ]

    def test_strip_on_clean_settings_is_identity(self) -> None:
        assert strip_particles_hook_entries(FOREIGN_SETTINGS) == FOREIGN_SETTINGS

    def test_merge_refuses_non_object_hooks(self) -> None:
        with pytest.raises(ValueError, match="refusing"):
            merge_particles_hook_entries({"hooks": "corrupt"}, COMMANDS)


# ---------------------------------------------------------------------------
# config.yaml surgeon (pure helpers)
# ---------------------------------------------------------------------------


class TestConfigSurgeon:
    def test_enable_on_empty_and_idempotent(self) -> None:
        out = enable_memory_store_text("", "memory", DSN)
        tree = yaml.safe_load(out)
        assert tree["storage"]["stores"]["memory"] == DSN
        assert tree["mcp"]["write"]["enabled_stores"] == ["memory"]
        assert enable_memory_store_text(out, "memory", DSN) == out

    def test_enable_preserves_comments_and_other_keys(self) -> None:
        original = (
            "# my precious comment\n"
            "extraction:\n"
            "  html_chunk_size: 9000  # tuned by hand\n"
            "mcp:\n"
            "  write:\n"
            "    agent_trust_rank: 0.7\n"
        )
        out = enable_memory_store_text(original, "memory", DSN)
        assert "# my precious comment" in out
        assert "html_chunk_size: 9000  # tuned by hand" in out
        tree = yaml.safe_load(out)
        assert tree["mcp"]["write"]["agent_trust_rank"] == 0.7
        assert tree["mcp"]["write"]["enabled_stores"] == ["memory"]
        assert tree["storage"]["stores"]["memory"] == DSN

    def test_enable_appends_to_existing_lists(self) -> None:
        original = (
            "storage:\n"
            "  stores:\n"
            "    research: sqlite+aiosqlite:///./r.db\n"
            "mcp:\n"
            "  write:\n"
            "    enabled_stores:\n"
            "      - research\n"
        )
        out = enable_memory_store_text(original, "memory", DSN)
        tree = yaml.safe_load(out)
        assert tree["storage"]["stores"] == {
            "research": "sqlite+aiosqlite:///./r.db",
            "memory": DSN,
        }
        assert tree["mcp"]["write"]["enabled_stores"] == ["research", "memory"]

    def test_enable_refuses_conflicting_dsn(self) -> None:
        original = "storage:\n  stores:\n    memory: sqlite+aiosqlite:///other.db\n"
        with pytest.raises(ConfigEditError, match="different DSN"):
            enable_memory_store_text(original, "memory", DSN)

    def test_enable_refuses_unparseable(self) -> None:
        with pytest.raises(ConfigEditError):
            enable_memory_store_text("mcp: [unclosed\n", "memory", DSN)
        with pytest.raises(ConfigEditError, match="not a mapping"):
            enable_memory_store_text("- just\n- a list\n", "memory", DSN)

    def test_disable_reverts_enable(self) -> None:
        original = "# keep me\nextraction:\n  html_chunk_size: 9000\n"
        enabled = enable_memory_store_text(original, "memory", DSN)
        assert disable_memory_store_text(enabled, "memory") == original

    def test_disable_keeps_other_stores(self) -> None:
        original = (
            "storage:\n"
            "  stores:\n"
            "    research: sqlite+aiosqlite:///./r.db\n"
            "mcp:\n"
            "  write:\n"
            "    enabled_stores:\n"
            "      - research\n"
        )
        enabled = enable_memory_store_text(original, "memory", DSN)
        assert disable_memory_store_text(enabled, "memory") == original

    def test_disable_is_noop_when_absent(self) -> None:
        original = "extraction:\n  html_chunk_size: 9000\n"
        assert disable_memory_store_text(original, "memory") == original


# ---------------------------------------------------------------------------
# The installer CLI
# ---------------------------------------------------------------------------


class TestHookCommandEnv:
    """The absolute env pins that make the installed hook CWD-independent (§1)."""

    def test_absolutize_relative_sqlite_dsn(self) -> None:
        out = absolutize_sqlite_dsn(
            "sqlite+aiosqlite:///./particles.db", base_dir=Path("/data/store")
        )
        assert out == "sqlite+aiosqlite:////data/store/particles.db"

    def test_absolute_and_network_dsn_unchanged(self) -> None:
        abs_dsn = "sqlite+aiosqlite:////var/particles.db"
        pg_dsn = "postgresql+asyncpg://host/db"
        assert absolutize_sqlite_dsn(abs_dsn) == abs_dsn
        assert absolutize_sqlite_dsn(pg_dsn, base_dir=Path("/anywhere")) == pg_dsn

    def test_render_env_prefix_quotes_and_trails(self) -> None:
        prefix = render_env_prefix({"PARTICLES_CONFIG": "/a b/config.yaml", "DATABASE_URL": "x"})
        assert prefix == "env PARTICLES_CONFIG='/a b/config.yaml' DATABASE_URL=x "

    def test_empty_env_is_no_prefix(self) -> None:
        assert render_env_prefix(None) == ""
        assert render_env_prefix({}) == ""
        # And build_hook_commands with no env reproduces the bare command exactly.
        bare = build_hook_commands("/venv/bin/particles", "memory")
        assert bare["SessionStart"] == "/venv/bin/particles hook session-start --store memory"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def init_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_db: Path) -> dict[str, Path]:
    """Isolated HOME + config.yaml for installer runs.

    ``cli_db`` provides the default store; the auto-created ``memory`` store
    gets its own SQLite file under the fake HOME. ``PARTICLES_CONFIG`` points
    at a tmp config.yaml so the store auto-create edits a sandboxed file.
    """
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "config.yaml"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    from particles.config import reset_config

    reset_config()
    return {
        "home": home,
        "config": config,
        "settings": home / ".claude" / "settings.json",
    }


def _settings(env: dict[str, Path]) -> dict[str, Any]:
    return json.loads(env["settings"].read_text())  # type: ignore[no-any-return]


def _commands(settings: dict[str, Any], event: str) -> list[str]:
    return [h["command"] for g in settings["hooks"][event] for h in g["hooks"]]


def _env_from_command(command: str) -> dict[str, str]:
    """Parse the leading ``env K=V …`` prefix of an installed hook command."""
    import shlex

    tokens = shlex.split(command)
    assert tokens and tokens[0] == "env", command
    env: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            break  # reached the console-script path
        key, value = token.split("=", 1)
        env[key] = value
    return env


class TestInitClaudeCode:
    def test_fresh_install_creates_everything(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 0, result.output

        # Settings: one Particles-owned entry per lifecycle event, each carrying
        # the absolute PARTICLES_CONFIG pin so harvest resolves the same store
        # from any working directory (CWD-independence).
        settings = _settings(init_env)
        config_abs = str(init_env["config"].resolve())
        for event, verb in (("SessionStart", "session-start"), ("SessionEnd", "session-end")):
            commands = [h["command"] for g in settings["hooks"][event] for h in g["hooks"]]
            assert len(commands) == 1
            assert f"particles hook {verb} --store memory" in commands[0]
            assert commands[0].startswith(f"env PARTICLES_CONFIG={config_abs} ")
            # The console-script path is still absolute, after the env prefix.
            assert Path(commands[0].split(" hook ")[0].split()[-1]).is_absolute()
            # A named store carries no DATABASE_URL pin — its DSN lives in the
            # pinned config (the override maps to `default` only).
            assert "DATABASE_URL" not in commands[0]

        # Store auto-create: config.yaml written, DB initialised, state dir made.
        tree = yaml.safe_load(init_env["config"].read_text())
        assert "memory" in tree["storage"]["stores"]
        assert tree["mcp"]["write"]["enabled_stores"] == ["memory"]
        db_file = init_env["home"] / ".particles" / "memory.db"
        assert db_file.exists()
        assert (init_env["home"] / ".particles" / "claude-code").is_dir()

        # The audit hand-off was skipped.
        assert "First-run memory audit" not in result.output

    def test_fresh_install_ships_the_skill_files(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        """init installs the onboarding skills by default."""
        from particles.skills import SKILL_FILENAMES

        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 0, result.output

        owned = init_env["home"] / ".claude" / "skills" / "particles"
        assert {p.name for p in owned.iterdir()} == set(SKILL_FILENAMES)
        assert "agent-onboarding skill file(s)" in result.output

    def test_no_skills_opts_out(self, runner: CliRunner, init_env: dict[str, Path]) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--no-audit", "--no-skills"])
        assert result.exit_code == 0, result.output
        assert not (init_env["home"] / ".claude" / "skills").exists()

    def test_remove_deletes_only_the_owned_skills_subdir(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        """--remove is surgical here too: a bystander skill survives."""
        runner.invoke(app, ["init", "claude-code", "--no-audit"])
        bystander = init_env["home"] / ".claude" / "skills" / "someone-else.md"
        bystander.write_text("not ours")

        result = runner.invoke(app, ["init", "claude-code", "--remove"])

        assert result.exit_code == 0, result.output
        assert not (init_env["home"] / ".claude" / "skills" / "particles").exists()
        assert bystander.read_text() == "not ours"

    def test_dry_run_writes_no_skill_files(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--no-audit", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "would write skill file" in result.output
        assert not (init_env["home"] / ".claude" / "skills").exists()

    def test_json_emits_a_machine_readable_result(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        """an agent runs the installer and parses the outcome.

        stdout must be the object and nothing else — the point of the flag is
        that no scraping of human-formatted narration is required.
        """
        result = runner.invoke(app, ["init", "claude-code", "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.stdout)
        assert payload["adapter"] == "claude-code"
        assert payload["action"] == "install"
        assert payload["dry_run"] is False
        assert payload["scope"] == "user"
        assert payload["store"] == "memory"
        assert payload["settings_path"] == str(init_env["settings"])
        assert payload["created"]["config_store_enabled"] is True
        assert payload["created"]["state_dir"].endswith("claude-code")
        assert len(payload["created"]["skills"]) == 3
        assert set(payload["hooks"]) == {"SessionStart", "SessionEnd"}

        # --json is non-interactive, so the audit becomes a named next step
        # rather than a prompt nobody is there to answer.
        assert any("particles audit" in step for step in payload["next_steps"])

    def test_json_suppresses_human_narration_on_stdout(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--json"])
        assert result.exit_code == 0, result.output
        assert "Installed Claude Code hooks into" not in result.stdout
        assert result.stdout.lstrip().startswith("{")

    def test_json_dry_run_reports_the_plan_without_writing(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--json", "--dry-run"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True
        assert len(payload["created"]["skills"]) == 3
        assert not init_env["settings"].exists()
        assert not (init_env["home"] / ".claude" / "skills").exists()

    def test_json_remove_reports_what_it_took_away(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        runner.invoke(app, ["init", "claude-code", "--no-audit"])
        result = runner.invoke(app, ["init", "claude-code", "--remove", "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.stdout)
        assert payload["action"] == "remove"
        assert payload["store"] == "memory"
        assert len(payload["hooks_removed"]) == 2
        assert payload["skills_removed"].endswith("skills/particles")
        assert any("State directory" in step for step in payload["next_steps"])

    def test_json_project_scope_is_reported(
        self, runner: CliRunner, init_env: dict[str, Path], tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        import os

        cwd = os.getcwd()
        os.chdir(repo)
        try:
            result = runner.invoke(app, ["init", "claude-code", "--project", "--json"])
        finally:
            os.chdir(cwd)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["scope"] == "project"

    def test_audit_handoff_runs_first_run_audit(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        # The hand-off: with no memory directories under the (fake)
        # HOME, the audit reports there is nothing to audit yet — and init
        # still succeeds.
        result = runner.invoke(app, ["init", "claude-code"])
        assert result.exit_code == 0, result.output
        assert "First-run memory audit" in result.output
        assert "no memory directories found" in result.output

    def test_rerun_is_idempotent(self, runner: CliRunner, init_env: dict[str, Path]) -> None:
        assert runner.invoke(app, ["init", "claude-code", "--no-audit"]).exit_code == 0
        first_settings = init_env["settings"].read_text()
        first_config = init_env["config"].read_text()
        assert runner.invoke(app, ["init", "claude-code", "--no-audit"]).exit_code == 0
        assert init_env["settings"].read_text() == first_settings
        assert init_env["config"].read_text() == first_config

    def test_dry_run_writes_nothing(self, runner: CliRunner, init_env: dict[str, Path]) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "would write" in result.output
        assert "store auto-create" in result.output
        assert not init_env["settings"].exists()
        assert not init_env["config"].exists()
        assert not (init_env["home"] / ".particles").exists()

    def test_command_override(self, runner: CliRunner, init_env: dict[str, Path]) -> None:
        result = runner.invoke(
            app,
            ["init", "claude-code", "--no-audit", "--command", "/custom/bin/particles"],
        )
        assert result.exit_code == 0, result.output
        settings = _settings(init_env)
        commands = [h["command"] for g in settings["hooks"]["SessionStart"] for h in g["hooks"]]
        config_abs = str(init_env["config"].resolve())
        assert commands == [
            f"env PARTICLES_CONFIG={config_abs} "
            "/custom/bin/particles hook session-start --store memory"
        ]

    def test_default_store_bakes_absolute_database_url(
        self,
        runner: CliRunner,
        init_env: dict[str, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The `default` store resolves to storage.database_url (CWD-relative by
        # default), so init pins its absolute DSN via DATABASE_URL.
        dsn = f"sqlite+aiosqlite:///{tmp_path / 'default-store' / 'particles.db'}"
        (tmp_path / "default-store").mkdir()
        monkeypatch.setenv("DATABASE_URL", dsn)
        from particles.config import reset_config

        reset_config()
        result = runner.invoke(app, ["init", "claude-code", "--no-audit", "--store", "default"])
        assert result.exit_code == 0, result.output
        cmd = _commands(_settings(init_env), "SessionEnd")[0]
        env = _env_from_command(cmd)
        assert env["DATABASE_URL"] == dsn
        assert " hook session-end --store default" in cmd

    def test_hook_env_resolves_named_store_from_foreign_cwd(
        self,
        runner: CliRunner,
        init_env: dict[str, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The core regression: a fresh install auto-creates the `memory` store,
        # and its hook command must resolve that store from ANY directory, not
        # just the one init ran in. Replay only the baked env pins from a foreign
        # cwd and confirm the store is still declared with its absolute DSN.
        assert runner.invoke(app, ["init", "claude-code", "--no-audit"]).exit_code == 0
        env = _env_from_command(_commands(_settings(init_env), "SessionEnd")[0])
        assert env["PARTICLES_CONFIG"] == str(init_env["config"].resolve())

        foreign = tmp_path / "some-worktree"
        foreign.mkdir()
        monkeypatch.chdir(foreign)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        from particles.config import get_config, reset_config

        reset_config()
        memory_dsn = get_config().storage.stores["memory"]
        assert memory_dsn == f"sqlite+aiosqlite:///{init_env['home'] / '.particles' / 'memory.db'}"

    def test_foreign_settings_preserved(self, runner: CliRunner, init_env: dict[str, Path]) -> None:
        init_env["settings"].parent.mkdir(parents=True)
        init_env["settings"].write_text(json.dumps(FOREIGN_SETTINGS))
        assert runner.invoke(app, ["init", "claude-code", "--no-audit"]).exit_code == 0
        settings = _settings(init_env)
        assert settings["model"] == "opus"
        assert settings["hooks"]["PreToolUse"] == FOREIGN_SETTINGS["hooks"]["PreToolUse"]
        starts = [h["command"] for g in settings["hooks"]["SessionStart"] for h in g["hooks"]]
        assert "echo my-own-hook" in starts
        assert len(starts) == 2

    def test_unparseable_settings_is_error_never_rewrite(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        init_env["settings"].parent.mkdir(parents=True)
        init_env["settings"].write_text("{ not json !!!")
        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 1
        assert "not parseable JSON" in result.output
        assert init_env["settings"].read_text() == "{ not json !!!"

    def test_project_flag_targets_settings_local(
        self,
        runner: CliRunner,
        init_env: dict[str, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.chdir(repo)
        result = runner.invoke(app, ["init", "claude-code", "--no-audit", "--project"])
        assert result.exit_code == 0, result.output
        assert (repo / ".claude" / "settings.local.json").exists()
        assert not init_env["settings"].exists()

    def test_multiple_enabled_stores_requires_flag(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        init_env["config"].write_text(
            "storage:\n"
            "  stores:\n"
            "    alpha: sqlite+aiosqlite:///./a.db\n"
            "    beta: sqlite+aiosqlite:///./b.db\n"
            "mcp:\n"
            "  write:\n"
            "    enabled_stores: [alpha, beta]\n"
        )
        from particles.config import reset_config

        reset_config()
        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 1
        assert "alpha" in result.output and "beta" in result.output
        assert "--store" in result.output

    def test_remove_reverts_fresh_install(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        init_env["settings"].parent.mkdir(parents=True)
        init_env["settings"].write_text(json.dumps(FOREIGN_SETTINGS))
        assert runner.invoke(app, ["init", "claude-code", "--no-audit"]).exit_code == 0

        result = runner.invoke(app, ["init", "claude-code", "--remove"])
        assert result.exit_code == 0, result.output

        # Settings: exactly the foreign content again.
        assert _settings(init_env) == FOREIGN_SETTINGS
        # Config: the (still-empty) store auto-create was reverted, DB deleted.
        assert yaml.safe_load(init_env["config"].read_text() or "null") in (None, {})
        assert not (init_env["home"] / ".particles" / "memory.db").exists()
        # The state dir (hook log history) is kept.
        assert (init_env["home"] / ".particles" / "claude-code").is_dir()

    def test_remove_keeps_store_holding_data(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        assert runner.invoke(app, ["init", "claude-code", "--no-audit"]).exit_code == 0

        # Put a corpus entry into the memory store — now it "holds data".
        import asyncio

        async def _seed() -> None:
            from particles.core.schema import Mutability
            from particles.corpus.deposit import deposit_text_versioned
            from particles.db import session_scope

            async with session_scope("memory") as session:
                await deposit_text_versioned(
                    session,
                    text="a belief source",
                    uri_r="claude-code://session/seed",
                    source_type="CONVERSATION",
                    mutability=Mutability.APPEND_ONLY,
                )
                await session.commit()

        asyncio.run(_seed())

        result = runner.invoke(app, ["init", "claude-code", "--remove"])
        assert result.exit_code == 0, result.output
        assert "never deleted" in result.output
        tree = yaml.safe_load(init_env["config"].read_text())
        assert "memory" in tree["storage"]["stores"]
        assert (init_env["home"] / ".particles" / "memory.db").exists()

    def test_remove_when_nothing_installed(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--remove"])
        assert result.exit_code == 0, result.output
        assert "No Particles-owned hook entries" in result.output


# ---------------------------------------------------------------------------
# projection provisioning (memory.yaml + region seeding)
# ---------------------------------------------------------------------------


class TestInitProjectionProvisioning:
    def test_install_writes_default_manifest(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 0, result.output
        manifest = init_env["home"] / ".particles" / "claude-code" / "memory.yaml"
        assert manifest.is_file()
        text = manifest.read_text()
        assert "name: memory-index" in text
        assert "render: bullets" in text
        assert "max_lines: 120" in text

    def test_rerun_never_clobbers_an_edited_manifest(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        manifest = init_env["home"] / ".particles" / "claude-code" / "memory.yaml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("# operator-tuned\nname: memory-index\n")
        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 0, result.output
        assert manifest.read_text() == "# operator-tuned\nname: memory-index\n"

    def test_install_seeds_region_into_existing_memory_md(
        self, runner: CliRunner, init_env: dict[str, Path]
    ) -> None:
        from particles.render.markdown import find_projected_regions

        memory_md = init_env["home"] / ".claude" / "projects" / "-p" / "memory" / "MEMORY.md"
        memory_md.parent.mkdir(parents=True)
        memory_md.write_text("# Memory\n- an existing note\n")

        result = runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert result.exit_code == 0, result.output
        text = memory_md.read_text()
        regions = find_projected_regions(text)
        assert len(regions) == 1 and regions[0].region == "memory-index"
        assert regions[0].body == ""  # empty until the first harvest cycle
        assert text.endswith("# Memory\n- an existing note\n")

        # Re-run does not double-insert.
        runner.invoke(app, ["init", "claude-code", "--no-audit"])
        assert len(find_projected_regions(memory_md.read_text())) == 1

    def test_dry_run_provisions_nothing(self, runner: CliRunner, init_env: dict[str, Path]) -> None:
        memory_md = init_env["home"] / ".claude" / "projects" / "-p" / "memory" / "MEMORY.md"
        memory_md.parent.mkdir(parents=True)
        memory_md.write_text("- a note\n")
        result = runner.invoke(app, ["init", "claude-code", "--no-audit", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "would write" in result.output
        assert not (init_env["home"] / ".particles" / "claude-code" / "memory.yaml").exists()
        assert memory_md.read_text() == "- a note\n"
        assert "would insert the projected region" in result.output
