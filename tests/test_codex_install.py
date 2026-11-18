# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Pure plans and transactional rollback using an injected user scheduler."""

from __future__ import annotations

import json
import plistlib
import subprocess
import tomllib
from pathlib import Path

import pytest

from particles.api.cli import _codex_install as install
from particles.api.cli import _codex_schedule as schedule
from particles.api.cli._codex_runtime import physical_store
from particles.api.cli._hook_state import write_json


class Scheduler:
    def __init__(self):
        self.loaded = set()
        self.calls = []
        self.fail_bootstrap = False
        self.cron = ""
        self.disabled = ""
        self.paths = {}
        self.definitions = {}

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[0].endswith("crontab"):
            return subprocess.CompletedProcess(argv, 0, self.cron, "")
        if argv[1] == "list":
            return subprocess.CompletedProcess(
                argv,
                0,
                "PID Status Label\n" + "\n".join(f"- 0 {label}" for label in self.loaded),
                "",
            )
        if argv[1] == "print-disabled":
            return subprocess.CompletedProcess(argv, 0, self.disabled, "")
        if argv[1] == "print":
            label = argv[-1].rsplit("/", 1)[1]
            path = self.paths.get(label)
            value = self.definitions.get(label, {})
            output = f"path = {path}\nprogram = {value.get('ProgramArguments', [''])[0]}\n"
            output += "arguments = {\n" + "\n".join(value.get("ProgramArguments", [])) + "\n}\n"
            output += (
                "environment = {\n"
                + "\n".join(
                    f"{key} => {val}" for key, val in value.get("EnvironmentVariables", {}).items()
                )
                + "\n}\n"
            )
            return subprocess.CompletedProcess(argv, 0, output, "")
        if argv[1] == "bootstrap":
            if self.fail_bootstrap:
                return subprocess.CompletedProcess(argv, 1, "", "")
            path = Path(argv[-1])
            value = plistlib.loads(path.read_bytes())
            label = value["Label"]
            self.loaded.add(label)
            self.paths[label] = path
            self.definitions[label] = value
        elif argv[1] == "bootout":
            self.loaded.discard(argv[-1].split("/")[-1])
        return subprocess.CompletedProcess(argv, 0, "", "")


def setup(tmp_path):
    config = tmp_path / "particles.yaml"
    dsn = f"sqlite+aiosqlite:///{tmp_path}/store.db"
    text = "# preserved operator comment\nstorage:\n  database_url: " + dsn + "\n"
    config.write_text(text)
    command = tmp_path / "particles"
    command.write_text("test executable; never executed")
    runner = Scheduler()
    env = {"PARTICLES_CONFIG": str(config), "DATABASE_URL": dsn}
    args = dict(
        home=tmp_path / "codex",
        state=tmp_path / "state",
        config=config,
        config_text=text,
        store="default",
        dsn=dsn,
        command=str(command),
        env=env,
        mcp_env=env,
        agents=tmp_path / "agents",
        runner=runner,
        platform="darwin",
        validator=lambda content: True,
    )
    return args, runner


def test_plan_is_no_write_and_platform_refusal_precedes_discovery(tmp_path):
    args, runner = setup(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    plan = install.plan_install(**args)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert not any("bootstrap" in argv or "bootout" in argv for argv in runner.calls)
    assert plan.report()["status"] == "configured, learning blocked"
    runner.calls.clear()
    with pytest.raises(ValueError, match="Linux lifecycle"):
        install.plan_install(**{**args, "platform": "linux"})
    assert not runner.calls


def test_apply_idempotent_and_remove_preserves_unrelated_files_and_store(tmp_path):
    args, runner = setup(tmp_path)
    home = args["home"]
    home.mkdir()
    (home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [{"hooks": [{"type": "command", "command": "foreign --keep"}]}]
                }
            }
        )
    )
    (home / "config.toml").write_text('# keep comment\nmodel = "operator"\n')
    plan = install.plan_install(**args)
    initialized = []
    install.apply_plan(plan, runner, lambda: initialized.append(True))
    assert initialized == [True]
    assert runner.loaded
    assert "# keep comment" in (home / "config.toml").read_text()
    assert tomllib.loads((home / "config.toml").read_text())["model"] == "operator"
    second = install.plan_install(**args)
    assert not second.report()["files"]
    install.apply_plan(second, runner)
    database = tmp_path / "store.db"
    database.write_text("existing corpus retained")
    remove = install.plan_install(**{**args, "remove": True})
    install.apply_plan(remove, runner)
    assert not runner.loaded
    assert database.read_text() == "existing corpus retained"
    assert "foreign --keep" in (home / "hooks.json").read_text()
    assert "mcp_servers" not in tomllib.loads((home / "config.toml").read_text())
    assert "operator" in (home / "config.toml").read_text()


def test_bootstrap_failure_rolls_back_owned_config_but_never_corpus(tmp_path):
    args, runner = setup(tmp_path)
    config_before = args["config"].read_bytes()
    plan = install.plan_install(**args)
    runner.fail_bootstrap = True
    database = tmp_path / "store.db"
    with pytest.raises(ValueError, match="bootstrap"):
        install.apply_plan(plan, runner, lambda: database.write_text("initialized data"))
    assert database.read_text() == "initialized data"
    assert args["config"].read_bytes() == config_before
    assert not (args["home"] / "hooks.json").exists()
    assert not list(args["agents"].glob("*.plist"))
    assert not runner.loaded
    assert not (args["state"] / "transaction.json").exists()


def test_input_race_and_changed_job_refused(tmp_path):
    args, runner = setup(tmp_path)
    plan = install.plan_install(**args)
    args["config"].write_text("# another writer\n")
    with pytest.raises(ValueError, match="changed since planning"):
        install.apply_plan(plan, runner)
    assert not runner.loaded
    args["config"].write_text(args["config_text"])
    plan = install.plan_install(**args)
    install.apply_plan(plan, runner)
    plan.owned_job.write_bytes(b"changed by operator")
    with pytest.raises(ValueError, match="job changed|unreadable launch agent"):
        install.plan_install(**{**args, "remove": True})


def test_existing_legacy_or_claude_job_reused_never_acquired_or_removed(tmp_path):
    args, runner = setup(tmp_path)
    args["agents"].mkdir()
    legacy = args["agents"] / "dev.particles.consolidate.plist"
    value = plistlib.loads(
        schedule.job_definition(
            physical_store(args["dsn"]), args["command"], "default", args["env"], tmp_path
        )
    )
    value["Label"] = "dev.particles.consolidate"
    value.pop("ParticlesOwner")
    legacy.write_bytes(plistlib.dumps(value))
    runner.loaded.add(value["Label"])
    runner.paths[value["Label"]] = legacy
    runner.definitions[value["Label"]] = value
    before = legacy.read_bytes()
    plan = install.plan_install(**args)
    assert plan.schedule.path == legacy
    assert plan.owned_job is None
    install.apply_plan(plan, runner)
    assert legacy.read_bytes() == before
    install.apply_plan(install.plan_install(**{**args, "remove": True}), runner)
    assert legacy.read_bytes() == before
    assert runner.loaded == {value["Label"]}
    assert not any(argv[1] in {"bootstrap", "bootout"} for argv in runner.calls)


def test_same_database_alias_and_disabled_schedule_count_as_existing(tmp_path):
    args, runner = setup(tmp_path)
    config = args["config"]
    config.write_text(args["config_text"] + "  stores:\n    other: " + args["dsn"] + "\n")
    args["config_text"] = config.read_text()
    args["agents"].mkdir()
    value = plistlib.loads(
        schedule.job_definition(
            physical_store(args["dsn"]),
            args["command"],
            "other",
            {"PARTICLES_CONFIG": str(config)},
            tmp_path,
        )
    )
    value["Disabled"] = True
    path = args["agents"] / "manual.plist"
    path.write_bytes(plistlib.dumps(value))
    plan = install.plan_install(**args)
    assert plan.owned_job is None
    assert not plan.schedule.enabled
    assert any("disabled/not loaded" in note for note in plan.notes)
    second = args["agents"] / "second.plist"
    value["Label"] = "second"
    second.write_bytes(plistlib.dumps(value))
    with pytest.raises(ValueError, match="multiple consolidation"):
        install.plan_install(**args)


def test_opaque_cron_and_foreign_mcp_inline_hooks_and_skills_refused(tmp_path):
    args, runner = setup(tmp_path)
    runner.cron = "0 3 * * * /bin/sh ~/nightly.sh\n"
    with pytest.raises(ValueError, match="unresolved"):
        install.plan_install(**args)
    runner.cron = ""
    args["home"].mkdir()
    config = args["home"] / "config.toml"
    config.write_text('[mcp_servers.particles]\ncommand = "foreign"\n')
    with pytest.raises(ValueError, match="foreign mcp"):
        install.plan_install(**args)
    config.write_text('[hooks]\nSessionEnd = "particles hook codex-session-end"\n')
    with pytest.raises(ValueError, match="inline TOML"):
        install.plan_install(**args)
    config.write_text("")
    path = args["home"] / "skills" / "particles-asking-the-store" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("operator authored")
    with pytest.raises(ValueError, match="skill"):
        install.plan_install(**args)
    assert path.read_text() == "operator authored"


def test_home_move_and_explicit_harness_tag_list_refused(tmp_path):
    args, runner = setup(tmp_path)
    write_json(args["state"] / "home.json", {"path": "/another/home", "id": "value"})
    with pytest.raises(ValueError, match="codex_home_moved"):
        install.plan_install(**args)
    (args["state"] / "home.json").unlink()
    with pytest.raises(ValueError, match="harness_tags"):
        install.plan_install(
            **{
                **args,
                "config_text": args["config_text"]
                + 'observer_scope:\n  harness_tags: ["claude-code"]\n',
            }
        )


def test_recovery_never_overwrites_another_writer(tmp_path):
    state = tmp_path / "state"
    target = tmp_path / "target"
    target.write_bytes(b"another writer")
    write_json(
        state / "transaction.json",
        {
            "files": [
                {
                    "path": str(target),
                    "before": install.digest(b"before"),
                    "after": install.digest(b"ours"),
                    "backup": str(tmp_path / "backup"),
                }
            ],
            "job": None,
        },
    )
    with pytest.raises(ValueError, match="another writer"):
        install.recover(state, Scheduler())
    assert target.read_bytes() == b"another writer"


def test_profile_surgeon_preserves_comments_and_refuses_unsafe_shapes():
    original = "# keep\nconsolidation:\n  # budget\n  min_interval_hours: 20\n"
    updated = install.set_profile(original, "codex")
    assert "# keep" in updated and "# budget" in updated
    assert install.set_profile(updated, "codex") == updated
    with pytest.raises(ValueError, match="block-style"):
        install.set_profile("consolidation: {}\n", "codex")


def test_secret_pins_and_parser_failures_never_create_files(tmp_path):
    args, runner = setup(tmp_path)
    with pytest.raises(ValueError, match="non-secret"):
        install.plan_install(**{**args, "env": {**args["env"], "ANTHROPIC_API_KEY": "never-write"}})
    assert not args["home"].exists()
    args["home"].mkdir()
    (args["home"] / "hooks.json").write_text('{"hooks":')
    with pytest.raises(ValueError):
        install.plan_install(**args)
    assert not args["agents"].exists()


def test_cli_dry_run_is_no_write_and_apply_initializes_before_bootstrap(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from particles.api.cli import _scheduled_credentials, app
    from particles.api.cli import codex as codex_cli
    from particles.config import reset_config

    # Replace this module's view only; global sys.platform and asyncio's
    # native event-loop selection must retain the actual CI operating system.
    monkeypatch.setattr(codex_cli, "sys", SimpleNamespace(platform="darwin", stdin=sys.stdin))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/fresh.db")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("PARTICLES_CODEX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PARTICLES_PROJECT_IDENTITY_LOCK", str(tmp_path / "identity.lock"))
    config = tmp_path / "operator.yaml"
    config.write_text("# operator config\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    command = tmp_path / "particles"
    command.write_text("not executed")
    monkeypatch.setattr(install, "discover", lambda *a: [])
    original_plan = install.plan_install

    def macos_plan(**kwargs):
        # Exercise the macOS CLI path on every CI OS without modifying global
        # sys.platform or invoking a host scheduler/plutil. Linux refusal is
        # separately tested before discovery or writes.
        return original_plan(**kwargs, platform="darwin", validator=lambda _content: True)

    monkeypatch.setattr(install, "plan_install", macos_plan)
    monkeypatch.setattr(_scheduled_credentials, "credential_status", lambda *a, **kw: "missing")
    runner = CliRunner()
    args = [
        "init",
        "codex",
        "--store",
        "default",
        "--config",
        str(config),
        "--command",
        str(command),
        "--json",
    ]
    result = runner.invoke(app, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"]
    assert not (tmp_path / "state").exists() and not (tmp_path / "codex").exists()
    assert config.read_text() == "# operator config\n"
    original = install.apply_plan
    scheduler = Scheduler()

    def apply(plan, initialize):
        original(plan, scheduler, initialize)

    monkeypatch.setattr(install, "apply_plan", apply)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    report = json.loads(result.stdout)
    assert report["applied"] and report["credential"] == "missing"
    assert "learning blocked" in report["status"]
    assert (tmp_path / "codex" / "hooks.json").exists()
    assert scheduler.loaded


def test_rerun_does_not_mint_replacement_home_identity(tmp_path):
    args, runner = setup(tmp_path)
    install.apply_plan(install.plan_install(**args), runner)
    (args["state"] / "home.json").unlink()
    with pytest.raises(ValueError, match="binding is missing or changed"):
        install.plan_install(**args)
    assert not (args["state"] / "home.json").exists()


def test_both_harness_installers_honor_the_same_lock(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from particles.api.cli import app
    from particles.api.cli._hook_state import advisory_lock, installer_lock_path

    args, runner = setup(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    plan = install.plan_install(**args)
    with advisory_lock(installer_lock_path()) as acquired:
        assert acquired
        with pytest.raises(ValueError, match="another harness installer"):
            install.apply_plan(plan, runner)
        result = CliRunner().invoke(
            app, ["init", "claude-code", "--store", "default", "--no-audit"]
        )
        assert result.exit_code == 1 and "Another harness installer" in result.output
    assert not args["home"].exists()
    assert not (tmp_path / "user" / ".claude").exists()


def test_loaded_job_cannot_be_replaced_under_an_owned_label(tmp_path):
    args, runner = setup(tmp_path)
    plan = install.plan_install(**args)
    install.apply_plan(plan, runner)
    label = f"dev.particles.consolidate.{plan.store_id}"
    runner.definitions[label]["EnvironmentVariables"]["DATABASE_URL"] = (
        f"sqlite+aiosqlite:///{tmp_path}/different.db"
    )
    before = len([argv for argv in runner.calls if argv[1] == "bootout"])
    with pytest.raises(ValueError, match="loaded consolidation"):
        install.plan_install(**{**args, "remove": True})
    assert len([argv for argv in runner.calls if argv[1] == "bootout"]) == before
    assert plan.owned_job.exists()


def test_remove_absent_installation_is_a_no_write_noop(tmp_path):
    args, runner = setup(tmp_path)
    plan = install.plan_install(**{**args, "remove": True})
    assert plan.report()["files"] == []
    install.apply_plan(plan, runner)
    assert not args["home"].exists() and not args["state"].exists()


def test_remove_without_manifest_refuses_unowned_particles_hook():
    raw = json.dumps(
        {"hooks": {"SessionStart": [{"hooks": [{"command": "/tmp/_codex_launcher.py"}]}]}}
    ).encode()
    with pytest.raises(ValueError, match="unowned/changed"):
        install._merge_hooks(raw, {}, {})


def test_removal_plan_discloses_credential_retention(tmp_path):
    args, _ = setup(tmp_path)
    plan = install.plan_install(**args, remove=True)
    assert plan.report()["status"] == "removal planned"
    assert any("Keychain Access" in note and "retained" in note for note in plan.notes)


def test_cli_linux_refusal_precedes_planning_and_writes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from typer.testing import CliRunner

    from particles.api.cli import app
    from particles.api.cli import codex as codex_cli

    plan = Mock(side_effect=AssertionError("Linux must refuse before planning"))
    monkeypatch.setattr(codex_cli, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(install, "plan_install", plan)
    target = tmp_path / "untouched.yaml"
    result = CliRunner().invoke(app, ["init", "codex", "--config", str(target), "--json"])
    assert result.exit_code == 1
    assert "Linux lifecycle and scheduling have not been measured" in result.output
    plan.assert_not_called()
    assert not target.exists()


def test_cli_dry_run_refuses_a_shared_installer_lock_directory(tmp_path, monkeypatch):
    """The dry run never reaches apply_plan's lock, so it asks the lock's check itself."""
    import sys
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from particles.api.cli import _codex_install as install
    from particles.api.cli import _scheduled_credentials, app
    from particles.api.cli import codex as codex_cli
    from particles.config import reset_config

    monkeypatch.setattr(codex_cli, "sys", SimpleNamespace(platform="darwin", stdin=sys.stdin))
    # The check runs once the plan is known; plan without the host scheduler.
    monkeypatch.setattr(install, "discover", lambda *a: [])
    original_plan = install.plan_install
    monkeypatch.setattr(
        install,
        "plan_install",
        lambda **kw: original_plan(**kw, platform="darwin", validator=lambda _c: True),
    )
    monkeypatch.setattr(_scheduled_credentials, "credential_status", lambda *a, **kw: "missing")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/fresh.db")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("PARTICLES_CODEX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PARTICLES_PROJECT_IDENTITY_LOCK", str(shared / "identity.lock"))
    config = tmp_path / "operator.yaml"
    config.write_text("# operator config\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    command = tmp_path / "particles"
    command.write_text("not executed")
    result = CliRunner().invoke(
        app,
        [
            "init",
            "codex",
            "--store",
            "default",
            "--config",
            str(config),
            "--command",
            str(command),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert f"chmod 700 {shared}" in result.output
    assert config.read_text() == "# operator config\n"
    assert not (tmp_path / "state").exists() and not (shared / "harness-installer.lock").exists()


def test_a_no_op_removal_takes_no_lock_and_so_is_not_refused():
    """apply_plan returns before the lock on a removal that changes nothing."""
    from types import SimpleNamespace

    from particles.api.cli._codex_install import plan_needs_lock

    same = SimpleNamespace(before=b"x", after=b"x")
    changed = SimpleNamespace(before=b"x", after=None)
    assert not plan_needs_lock(SimpleNamespace(remove=True, files=[same]))
    assert plan_needs_lock(SimpleNamespace(remove=True, files=[same, changed]))
    assert plan_needs_lock(SimpleNamespace(remove=False, files=[same]))


def thin_args(args):
    url = "http://127.0.0.1:18462"
    config = args["config"]
    config.write_text(args["config_text"] + f"engine:\n  base_url: {url}\n")
    tokens = {client: str(args["state"] / "engine-tokens" / client) for client in ("recall", "mcp")}
    return {
        **args,
        "config_text": config.read_text(),
        "dsn": "",
        "engine": url,
        "tokens": tokens,
        "env": {"PARTICLES_CONFIG": str(config), "PARTICLES_ENGINE_BASE_URL": ""},
        "mcp_env": {"PARTICLES_ENGINE_TOKEN": "", "PARTICLES_ENGINE_TOKEN_FILE": tokens["mcp"]},
    }


def test_thin_plan_skips_local_schedule_and_uses_separate_file_pins(tmp_path):
    args, runner = setup(tmp_path)
    remote = thin_args(args)
    before = sorted(tmp_path.rglob("*"))
    plan = install.plan_install(**remote)
    assert sorted(tmp_path.rglob("*")) == before
    assert not runner.calls
    report = plan.report()
    assert report["mode"] == "thin-client"
    assert report["schedule_action"] == "engine-managed"
    assert "--recall-token-file" in report["hooks"]["SessionStart"]
    assert "--capture-token-file" not in report["hooks"]["SessionEnd"]
    install.apply_plan(plan, runner)
    assert not runner.calls
    assert not (tmp_path / "store.db").exists()
    assert not args["agents"].exists()
    assert not install.plan_install(**remote).report()["files"]


def test_disabling_remote_capture_warns_about_retained_credential(tmp_path):
    from particles.api.cli._codex_remote import token_path

    args, runner = setup(tmp_path)
    remote = thin_args(args)
    plan = install.plan_install(**remote)
    install.apply_plan(plan, runner)
    retained = token_path(plan.state, "capture")
    install.write_file(retained, b"synthetic credential")
    plan = install.plan_install(**remote)
    assert (
        "Capture credential file retained. Revoke it on the engine." in plan.report()["next_steps"]
    )
    assert "--capture-token-file" not in plan.report()["hooks"]["SessionEnd"]
    install.apply_plan(plan, runner)
    assert retained.read_bytes() == b"synthetic credential"


def test_thin_cutover_rolls_back_then_preserves_corpus_hints_and_fingerprints(tmp_path):
    args, runner = setup(tmp_path)
    local = install.plan_install(**args)
    install.apply_plan(local, runner)
    before = (local.home / "hooks.json").read_bytes()
    before_binding = (local.state / "home.json").read_bytes()
    old = local.state / "stores" / local.store_id
    identity = "00000000-0000-4000-8000-000000000001"
    write_json(old / "emitted" / f"{identity}.json", ["a" * 64])
    write_json(old / "pending" / f"{identity}.json", {"session_id": identity})
    database = tmp_path / "store.db"
    database.write_bytes(b"local corpus")
    remote = thin_args(args)
    plan = install.plan_install(**remote)
    assert plan.owned_job == local.owned_job
    assert plan.report()["schedule_action"] == "retire owned local job"
    assert plan.previous_scope == local.store_id
    emitted = local.state / "stores" / plan.store_id / "emitted" / f"{identity}.json"

    def interrupted():
        assert emitted.exists()
        raise RuntimeError("injected cutover failure")

    with pytest.raises(RuntimeError, match="injected cutover"):
        install.apply_plan(plan, runner, interrupted)
    assert not emitted.exists()
    assert (local.home / "hooks.json").read_bytes() == before
    assert runner.loaded and local.owned_job.is_file()
    install.apply_plan(install.plan_install(**remote), runner)
    assert not local.owned_job.exists() and not runner.loaded
    assert json.loads(emitted.read_text()) == ["a" * 64]
    assert (old / "pending" / f"{identity}.json").exists()
    assert (old / "emitted" / f"{identity}.json").exists()
    assert (local.state / "home.json").read_bytes() == before_binding
    assert database.read_bytes() == b"local corpus"
