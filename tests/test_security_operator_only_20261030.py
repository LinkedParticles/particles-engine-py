# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the tenth delta security review's broad-root, config and key-file fixes.

Security review 2026-10-30-1: AB1 (a planted link never makes a directory a
broad root), AB3, AB14 / AB50, AB43 / AB51 and AB44 / AB53 (the key file's
place), AB11 (the suite's own config directory), AB12 and AB13 (the loaded
config record), AB19 (the MCP registration under ``--config``, and the
report of a kept store that differs) and AB59 (the counts with no effective
uid). Every layout is real, under ``tmp_path`` with ``HOME`` a closed
directory there, unless a test says which one fact it stands in for another
account.
"""

from __future__ import annotations

import dataclasses
import grp
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from particles.api import _keys
from particles.api.cli import hook as hook_mod
from particles.api.cli._claude_code import enclosing_work_tree
from particles.config import (
    get_config,
    is_broad_root,
    loaded_config_file,
    reset_config,
    resolve_config_relative_path,
    resolve_store_adjacent_path,
)
from particles.corpus import fetch, operator_only, rule_roots
from particles.corpus.operator_only import LinkFacts


def _closed_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _closed_file(path: Path, text: str = "") -> Path:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)
    return path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = _closed_dir(tmp_path / "home")
    monkeypatch.setenv("HOME", str(h))
    return h


# --- AB1: a planted link never makes a directory a broad root ------------------


def _plant_home_link(directory: Path, home: Path) -> Path:
    """A symlink named as home's last component, pointing at home, in ``directory``."""
    link = directory / home.name
    link.symlink_to(home, target_is_directory=True)
    return link


def _one_row_store(path: Path, value: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def test_a_planted_home_link_does_not_make_a_directory_broad(tmp_path: Path, home: Path) -> None:
    """A directory others may write stays judged after someone plants a link to home in it."""
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o1777)
    try:
        assert not is_broad_root(shared)
        _plant_home_link(shared, home)
        assert not is_broad_root(shared), "a planted link made a sticky directory a broad root"
    finally:
        os.chmod(shared, 0o700)


def test_a_link_to_homes_top_level_directory_does_not_make_a_clone_broad(
    tmp_path: Path, home: Path
) -> None:
    """A link named after home's first path component, pointing at it, makes nothing broad."""
    top = Path(os.path.realpath(home)).parts[1]
    clone = _closed_dir(tmp_path / "clone")
    (clone / ".git").mkdir()
    store = _closed_file(clone / "particles.db")
    assert fetch.store_file_judgement(store) is not None
    (clone / top).symlink_to("/" + top, target_is_directory=True)
    assert not is_broad_root(clone), f"a link named {top!r} made the clone a broad root"
    assert fetch.store_file_judgement(store) is not None


def test_a_store_link_in_a_sticky_directory_stays_refused_beside_a_planted_home_link(
    tmp_path: Path, home: Path
) -> None:
    """The store named as a link in a sticky directory is refused whatever else it holds."""
    real = _closed_file(_closed_dir(tmp_path / "closed") / "p.db")
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    os.chmod(sticky, 0o1777)
    name = sticky / "p.db"
    try:
        name.symlink_to(real)
        assert fetch.store_file_judgement(name) is not None
        _plant_home_link(sticky, home)
        verdict = fetch.store_file_judgement(name)
    finally:
        os.chmod(sticky, 0o700)
    assert verdict is not None, "a store link in a sticky directory was trusted"


def test_the_open_connection_swap_stays_refused_beside_a_planted_home_link(
    tmp_path: Path, home: Path
) -> None:
    """A name whose open connection serves another file than the one judged is refused."""
    operators = _closed_dir(tmp_path / "closed") / "p.db"
    _one_row_store(operators, "operator")
    os.chmod(operators, 0o600)
    planted = _closed_dir(tmp_path / "elsewhere") / "p.db"
    _one_row_store(planted, "planted")
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    os.chmod(sticky, 0o1777)
    name = sticky / "p.db"
    try:
        _plant_home_link(sticky, home)
        name.symlink_to(planted)
        conn = sqlite3.connect(name)
        try:
            name.unlink()
            name.symlink_to(operators)
            verdict = fetch.store_file_judgement(name)
            rows = conn.execute("SELECT x FROM t").fetchall()
        finally:
            conn.close()
    finally:
        os.chmod(sticky, 0o700)
    assert rows == [("planted",)]
    assert verdict is not None, "the gate trusted a name whose open connection serves another file"


def test_a_store_on_an_open_team_directory_stays_refused_beside_a_planted_home_link(
    tmp_path: Path, home: Path
) -> None:
    """A store under a parent others may write is refused, whatever links that parent holds."""
    team = tmp_path / "team"
    team.mkdir()
    store = _closed_file(_closed_dir(team / "op") / "p.db")
    os.chmod(team, 0o777)
    try:
        assert fetch.store_file_judgement(store) is not None
        _plant_home_link(team, home)
        verdict = fetch.store_file_judgement(store)
    finally:
        os.chmod(team, 0o700)
    assert verdict is not None, "a parent others may write stopped being judged"


def test_a_store_inside_a_clone_stays_refused_when_the_clone_ships_a_link_to_home(
    tmp_path: Path, home: Path
) -> None:
    """A store inside a git work tree is refused even when the clone holds a link to home."""
    clone = _closed_dir(tmp_path / "clone")
    (clone / ".git").mkdir()
    store = _closed_file(clone / "particles.db")
    assert fetch.store_file_judgement(store) is not None
    _plant_home_link(clone, home)
    assert fetch.store_file_judgement(store) is not None, (
        "a clone's store was trusted once the clone held a link to home"
    )


def test_a_config_inside_a_clone_stays_refused_when_the_clone_ships_a_link_to_home(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh's config gate, the doctor and the install all keep the work-tree rule."""
    clone = _closed_dir(tmp_path / "clone")
    (clone / ".git").mkdir()
    config = _closed_file(clone / "config.yaml")
    _plant_home_link(clone, home)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    assert fetch.config_file_judgement() is not None, "the refresh trusted a clone's config"
    assert hook_mod._config_pin_problem(str(config)) is not None
    assert enclosing_work_tree(config) == clone.resolve()


# --- AB59: with no effective uid the counts refuse what the read refuses ---------


_ORIGINAL_EFFECTIVE_UID = rule_roots.effective_uid


@contextmanager
def _no_uid() -> Iterator[None]:
    targets = [
        m
        for n, m in list(sys.modules.items())
        if n.startswith("particles.")
        and getattr(m, "effective_uid", None) is _ORIGINAL_EFFECTIVE_UID
    ]
    patches = [mock.patch.object(m, "effective_uid", lambda: None) for m in targets]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def test_with_no_uid_the_counts_refuse_what_the_read_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no effective uid a ~/.claude rule file is counted as refused, as the read refuses it."""
    home = _closed_dir(tmp_path / "home")
    _closed_dir(home / ".claude")
    monkeypatch.setenv("HOME", str(home))
    reset_config()
    rule = _closed_file(home / ".claude" / "CLAUDE.md", "# rules\n")
    assert fetch.admission_route(rule) is not None  # control: admitted with a uid
    with _no_uid():
        reason = fetch.local_refusal(rule)
        route = fetch.admission_route(rule)
    assert reason is not None, "the counts call a file admitted that the read refuses"
    assert route is None


# --- AB12 and AB13: the loaded config record -----------------------------------


def _config(directory: Path, root: Path, mode: int = 0o600) -> Path:
    config = directory / "config.yaml"
    config.write_text(f"local_refresh:\n  allowed_roots:\n    - {root}\n", encoding="utf-8")
    os.chmod(config, mode)
    return config


def test_a_config_others_could_write_when_loaded_is_refused_after_it_is_closed(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings read from a file others could write keep the gate refusing after it is closed."""
    root = _closed_dir(tmp_path / "chosen-root")
    config = _config(_closed_dir(tmp_path / "cfg"), root, mode=0o660)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    assert get_config().local_refresh.allowed_roots == [str(root)]
    assert fetch.config_file_judgement() is not None
    os.chmod(config, 0o600)
    assert get_config().local_refresh.allowed_roots == [str(root)]
    assert fetch.config_file_judgement() is not None, (
        "the gate trusted settings that were read from a file other users could write"
    )


def test_a_config_whose_directory_was_open_when_loaded_is_refused_after_it_is_closed(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings read from a file in an open directory keep the gate refusing after it is closed."""
    root = _closed_dir(tmp_path / "chosen-root")
    shared = tmp_path / "shared"
    shared.mkdir()
    config = _config(shared, root)
    os.chmod(shared, 0o777)
    try:
        monkeypatch.setenv("PARTICLES_CONFIG", str(config))
        reset_config()
        get_config()
        assert fetch.config_file_judgement() is not None
        os.chmod(shared, 0o700)
        verdict = fetch.config_file_judgement()
    finally:
        os.chmod(shared, 0o700)
    assert verdict is not None, "the gate trusted settings loaded from a file in an open directory"


def test_a_config_changed_after_it_was_loaded_is_refused(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config edited in place after the load no longer says what the settings in force do."""
    config = _config(_closed_dir(tmp_path / "cfg"), _closed_dir(tmp_path / "root"))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    get_config()
    assert fetch.config_file_judgement() is None  # control: closed, unchanged
    before = os.stat(config).st_ctime_ns
    with open(config, "a", encoding="utf-8") as handle:
        handle.write("# edited\n")
    if os.stat(config).st_ctime_ns == before:  # pragma: no cover - coarse clock
        pytest.skip("this filesystem's ctime did not move")
    assert fetch.config_file_judgement() is not None
    reset_config()
    assert fetch.config_file_judgement() is None  # a reload trusts the file again


def test_a_relative_config_value_resolves_beside_the_loaded_file_after_it_is_removed(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative value the loaded config names stays anchored to that file's directory."""
    cfg_dir = _closed_dir(tmp_path / "cfg")
    config = _config(cfg_dir, _closed_dir(tmp_path / "root"))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    get_config()
    assert resolve_config_relative_path("manifest.yaml") == cfg_dir.resolve() / "manifest.yaml"
    config.unlink()
    assert loaded_config_file() is not None
    assert resolve_config_relative_path("manifest.yaml") == cfg_dir.resolve() / "manifest.yaml"


def test_a_relative_config_value_follows_the_loaded_file_not_a_changed_pin(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed PARTICLES_CONFIG with no reload moves neither the settings nor their anchor."""
    cfg_dir = _closed_dir(tmp_path / "cfg")
    config = _config(cfg_dir, _closed_dir(tmp_path / "root"))
    other = _config(_closed_dir(tmp_path / "elsewhere"), _closed_dir(tmp_path / "root2"))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("DATABASE_URL", "postgresql://h/db")  # a store with no directory
    reset_config()
    get_config()
    monkeypatch.setenv("PARTICLES_CONFIG", str(other))
    loaded = loaded_config_file()
    assert loaded is not None and loaded.path == config
    assert resolve_store_adjacent_path("corpus_blobs") == cfg_dir.resolve() / "corpus_blobs"


# --- AB3, AB14 / AB50, AB43 / AB51, AB44 / AB53: the key file's place -----------


@pytest.fixture
def keydir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    yield keydir
    os.chmod(keydir, 0o700)
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


@pytest.mark.parametrize("mode", [0o000, 0o300])
def test_an_existing_key_file_behind_an_unsearchable_directory_never_makes_the_engine_keyless(
    keydir: Path, mode: int
) -> None:
    """A key file holding a live key that cannot be reached leaves the engine keyed and refused."""
    from particles.api import auth
    from particles.api._keys import Role

    _keys.mint(Role.READER, "viewer")
    os.chmod(keydir, mode)
    _keys.reset_key_cache()
    _keys.forget_key_files()  # a fresh engine process
    try:
        keyless = auth.engine_is_keyless()
        with pytest.raises(RuntimeError):
            auth.enforce_key_material_on_startup()
    finally:
        os.chmod(keydir, 0o700)
    assert not keyless, "an existing key file the engine cannot read made the engine keyless"


def test_a_key_file_under_a_missing_directory_is_no_keys(keydir: Path) -> None:
    """Control: a key path whose directory does not exist holds no keys."""
    from particles.api import auth

    os.rmdir(keydir)
    try:
        assert not _keys.live_keys()
        assert auth.engine_is_keyless()
    finally:
        keydir.mkdir(mode=0o700)


def test_a_key_directory_on_a_volume_that_ignores_ownership_is_refused(
    tmp_path: Path, home: Path
) -> None:
    """A key directory on a volume that ignores ownership is unusable, as the store's is.

    The volume flag the gather reads stands in for a mounted volume.
    """
    keys = _closed_file(_closed_dir(tmp_path / "keys") / "engine-keys.json", "{}")
    _keys._check_dir_safe(keys)  # control: a volume that keeps ownership
    with mock.patch.object(operator_only, "ownership_ignored", lambda fd: True):
        assert fetch.store_file_judgement(keys, sidecars=()) is not None
        with pytest.raises(_keys.KeyFileUnusable):
            _keys._check_dir_safe(keys)


def test_a_key_file_on_a_volume_that_ignores_ownership_is_unusable(keydir: Path) -> None:
    """The key file's own descriptor on an ownership-ignoring volume is never trusted."""
    from particles.api._keys import Role

    _keys.mint(Role.READER, "viewer")
    _keys.reset_key_cache()
    with (
        mock.patch.object(_keys, "_check_dir_safe", lambda path: None),
        mock.patch.object(_keys, "ownership_ignored", lambda fd: True),
        pytest.raises(_keys.KeyFileUnusable),
    ):
        _keys.live_keys()


def test_a_link_another_user_owns_on_the_key_files_path_makes_it_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real link on the key file's path, its owner the one fact rewritten, is refused."""
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o1777)
    real = _closed_dir(tmp_path / "realkeys")
    (shared / "keys").symlink_to(real)
    _closed_file(real / "engine-keys.json", '{"version": 1, "keys": []}\n')
    path = shared / "keys" / "engine-keys.json"
    real_gather = operator_only.gather_write_facts
    _keys._check_dir_safe(path)  # control: the link is this account's own

    def _foreign_links(p: Path, **kw: Any) -> Any:
        facts = real_gather(p, **kw)
        assert facts.links, "the layout must put a symlink on the path"
        return dataclasses.replace(
            facts,
            links=tuple(dataclasses.replace(link, uid=os.geteuid() + 4242) for link in facts.links),
        )

    monkeypatch.setattr(_keys, "gather_write_facts", _foreign_links)
    try:
        assert (
            operator_only.judge_write_facts(
                _foreign_links(path), operator_uid=os.geteuid(), named_ahead=False
            )
            is not None
        )
        with pytest.raises(_keys.KeyFileUnusable):
            _keys._check_dir_safe(path)
    finally:
        os.chmod(shared, 0o700)


def test_a_foreign_link_fact_on_the_key_files_path_makes_it_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any link fact another account owns refuses the key file, wherever on the path."""
    keydir = _closed_dir(tmp_path / "keys")
    real = _keys.gather_write_facts

    def with_a_foreign_link(path: Path) -> Any:
        facts = real(path)
        foreign = LinkFacts(keydir / "via", os.geteuid() + 4242)
        return dataclasses.replace(facts, links=(*facts.links, foreign))

    monkeypatch.setattr(_keys, "gather_write_facts", with_a_foreign_link)
    with pytest.raises(_keys.KeyFileUnusable):
        _keys._check_dir_safe(keydir / "engine-keys.json")


def test_a_gid_the_account_databases_cannot_enumerate_counts_as_shared() -> None:
    """A gid neither the group nor the account database lists is never read as unshared."""
    gid = 987_653
    try:
        grp.getgrgid(gid)
    except KeyError:
        pass
    else:  # pragma: no cover - a host that happens to know the gid
        pytest.skip("the gid is known on this host")
    assert _keys._group_shared_with_others(gid)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS nested groups")
def test_a_macos_group_every_local_account_belongs_to_is_shared() -> None:
    """A group every local account holds by nesting counts as shared with others."""
    try:
        gid = grp.getgrnam("localaccounts").gr_gid
    except KeyError:
        pytest.skip("no localaccounts group on this host")
    probe = subprocess.run(
        ["dseditgroup", "-o", "checkmember", "-m", "nobody", "localaccounts"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "yes" not in probe.stdout:
        pytest.skip("nobody is not reported as a localaccounts member here")
    assert _keys._group_shared_with_others(gid)


def test_a_setgid_key_directory_is_admitted_only_for_the_named_group(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fsGroup shape is admitted only when api.keys_dir_group names its group."""
    keydir = _closed_dir(tmp_path / "keys")
    os.chown(keydir, -1, os.getegid())
    os.chmod(keydir, 0o2770)
    path = keydir / "engine-keys.json"
    monkeypatch.setattr(_keys, "_group_shared_with_others", lambda gid: False)
    try:
        with pytest.raises(_keys.KeyFileUnusable):
            _keys._check_dir_safe(path)
        monkeypatch.setenv("PARTICLES_API_KEYS_DIR_GROUP", str(os.getegid() + 1))
        reset_config()
        with pytest.raises(_keys.KeyFileUnusable):
            _keys._check_dir_safe(path)
        monkeypatch.setenv("PARTICLES_API_KEYS_DIR_GROUP", str(os.getegid()))
        reset_config()
        _keys._check_dir_safe(path)
    finally:
        os.chmod(keydir, 0o700)


# --- AB11: the suite's own config directory --------------------------------------


_SUITE_ENV = (
    "DATABASE_URL",
    "PARTICLES_BLOB_DIR",
    "PARTICLES_CONFIG",
    "BENCHMARK_RUNS_DIR",
    "PARTICLES_API_KEYS_FILE",
    "PYTEST_XDIST_WORKER",
)


def test_the_suite_does_not_write_through_a_planted_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name another account planted in the shared temp directory never reaches a real config."""
    from tests import _client_fixtures

    for name in _SUITE_ENV:
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    shared_tmp = tmp_path / "shared-tmp"
    shared_tmp.mkdir()
    os.chmod(shared_tmp, 0o1777)
    developer = tmp_path / "dot-particles"
    developer.mkdir()
    os.chmod(developer, 0o755)
    real_config = developer / "config.yaml"
    real_config.write_text("storage:\n  database_url: kept\n", encoding="utf-8")
    (shared_tmp / f"particles_test_config_{os.getuid()}").symlink_to(developer)
    try:
        with (
            mock.patch.object(tempfile, "gettempdir", lambda: str(shared_tmp)),
            mock.patch.object(tempfile, "tempdir", str(shared_tmp)),
        ):
            _client_fixtures.pytest_configure(mock.Mock())
        text = real_config.read_text(encoding="utf-8")
        mode = stat.S_IMODE(os.stat(developer).st_mode)
        made = Path(os.environ["PARTICLES_CONFIG"])
    finally:
        os.chmod(shared_tmp, 0o700)
    assert text == "storage:\n  database_url: kept\n"
    assert mode == 0o755
    assert made.parent.parent == shared_tmp and not made.parent.is_symlink()


# --- AB19: --config replaces a registration that pinned no config ----------------


def test_with_config_a_registration_with_no_config_pin_serves_the_wanted_store() -> None:
    """After --config a registration pinning no config but another store serves the install's."""
    from particles.api.cli._claude_code import (
        PROJECT_OBSERVER_ARGS,
        McpRegistration,
        plan_mcp_registration,
    )

    wanted_env = {
        "PARTICLES_CONFIG": "/home/me/.particles/config.yaml",
        "DATABASE_URL": "sqlite+aiosqlite:////home/me/.particles/particles.db",
        "PARTICLES_ENGINE_BASE_URL": "",
    }
    existing = McpRegistration(
        name="particles",
        command="/opt/bin/particles",
        args=("mcp", "serve", *PROJECT_OBSERVER_ARGS),
        env={"DATABASE_URL": "sqlite+aiosqlite:////elsewhere/other.db"},
    )
    plan = plan_mcp_registration(
        [existing], executable="/opt/bin/particles", env=wanted_env, replace_config_pin=True
    )
    assert plan is not None
    wanted, _replaced = plan
    assert wanted.env == wanted_env


_AB19_WANTED_ENV = {
    "PARTICLES_CONFIG": "/home/me/.particles/config.yaml",
    "DATABASE_URL": "sqlite+aiosqlite:////home/me/.particles/particles.db",
    "PARTICLES_ENGINE_BASE_URL": "",
}
_AB19_OTHER_STORE = "sqlite+aiosqlite:////elsewhere/other.db"


def _register_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_env: dict[str, str],
    *,
    replace_config_pin: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    """Run the installer's MCP step over one registration, ``HOME`` a closed temporary directory."""
    import json

    import particles.api.cli.init as init_mod

    home = _closed_dir(tmp_path / "home")
    monkeypatch.setenv("HOME", str(home))
    server = {"command": "/opt/bin/particles", "args": ["mcp", "serve"], "env": registered_env}
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"particles": server}}))
    result: dict[str, Any] = {"next_steps": []}
    printed: list[str] = []
    monkeypatch.setattr(init_mod, "_say", printed.append)
    init_mod._register_mcp_server(
        result,
        executable="/opt/bin/particles",
        env=dict(_AB19_WANTED_ENV),
        dry_run=dry_run,
        replace_config_pin=replace_config_pin,
    )
    return result, "\n".join(printed)


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("replace_config_pin", [False, True])
def test_a_kept_registration_store_that_differs_is_reported_naming_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_config_pin: bool, dry_run: bool
) -> None:
    """A registration that keeps a DATABASE_URL other than the hooks' says so, naming both."""
    result, printed = _register_over(
        tmp_path,
        monkeypatch,
        {
            "PARTICLES_CONFIG": _AB19_WANTED_ENV["PARTICLES_CONFIG"],
            "DATABASE_URL": _AB19_OTHER_STORE,
        },
        replace_config_pin=replace_config_pin,
        dry_run=dry_run,
    )
    notes = [step for step in result["next_steps"] if "keeps DATABASE_URL" in step]
    assert len(notes) == 1
    assert _AB19_OTHER_STORE in notes[0]
    assert _AB19_WANTED_ENV["DATABASE_URL"] in notes[0]
    assert _AB19_OTHER_STORE in printed and _AB19_WANTED_ENV["DATABASE_URL"] in printed


@pytest.mark.parametrize(
    ("registered_env", "replace_config_pin"),
    [
        # The same store: nothing differs.
        (
            {
                "PARTICLES_CONFIG": _AB19_WANTED_ENV["PARTICLES_CONFIG"],
                "DATABASE_URL": _AB19_WANTED_ENV["DATABASE_URL"],
            },
            False,
        ),
        # No config pin under --config: the env is replaced, so nothing is kept.
        ({"DATABASE_URL": _AB19_OTHER_STORE}, True),
    ],
)
def test_a_registration_serving_the_hooks_store_gets_no_kept_store_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_env: dict[str, str],
    replace_config_pin: bool,
) -> None:
    """No note when the registration ends up serving the store the hooks pin."""
    result, printed = _register_over(
        tmp_path,
        monkeypatch,
        registered_env,
        replace_config_pin=replace_config_pin,
        dry_run=True,
    )
    assert not any("keeps DATABASE_URL" in step for step in result["next_steps"])
    assert "keeps its own DATABASE_URL" not in printed


# --- Z64, S3 attack 1: which store each stdio server and hook opens -------------


@pytest.fixture
def clone_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """HOME a temporary directory, the working directory a clone shipping a store and config."""
    home = tmp_path / "home"
    (home / ".particles").mkdir(parents=True)
    os.chmod(home, 0o700)
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "particles.db").write_bytes(b"")
    (clone / "config.yaml").write_text(
        "storage:\n  database_url: sqlite+aiosqlite:///./particles.db\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(clone)
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
    reset_config()
    return {"home": home, "clone": clone}


def _opened_store() -> Path:
    from particles.config import sqlite_file_path

    db = sqlite_file_path(get_config().storage.database_url)
    assert db is not None
    return Path(os.path.abspath(db))


def test_an_unpinned_server_serves_the_home_store_not_the_clones(
    clone_cwd: dict[str, Path],
) -> None:
    from particles.api.cli import prepare_stdio_server_store

    prepare_stdio_server_store("default")
    assert _opened_store() == clone_cwd["home"] / ".particles" / "particles.db"


def test_an_unpinned_server_anchors_a_relative_user_store_under_dot_particles(
    clone_cwd: dict[str, Path],
) -> None:
    from particles.api.cli import prepare_stdio_server_store

    (clone_cwd["home"] / ".particles" / "config.yaml").write_text(
        "storage:\n  database_url: sqlite+aiosqlite:///./mine.db\n", encoding="utf-8"
    )
    reset_config()
    prepare_stdio_server_store("default")
    assert _opened_store() == clone_cwd["home"] / ".particles" / "mine.db"


def test_a_pinned_config_naming_a_relative_store_is_refused(
    clone_cwd: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import typer

    from particles.api.cli import prepare_stdio_server_store

    pinned = tmp_path / "pinned.yaml"
    pinned.write_text("storage:\n  database_url: sqlite+aiosqlite:///./particles.db\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(pinned))
    reset_config()
    with pytest.raises(typer.Exit):
        prepare_stdio_server_store("default")


def test_a_relative_config_pin_is_refused_by_the_server(
    clone_cwd: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    from particles.api.cli import prepare_stdio_server_store

    monkeypatch.setenv("PARTICLES_CONFIG", "config.yaml")
    reset_config()
    with pytest.raises(typer.Exit):
        prepare_stdio_server_store("default")


def test_an_absolute_database_url_is_the_store_served(
    clone_cwd: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from particles.api.cli import prepare_stdio_server_store

    target = tmp_path / "chosen.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{target}")
    reset_config()
    prepare_stdio_server_store("default")
    assert _opened_store() == target


def test_the_hook_refuses_a_relative_store_and_a_relative_config(
    clone_cwd: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.config import relative_config_pin

    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./particles.db")
    reset_config()
    assert hook_mod._working_directory_store_refusal("default") is not None
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("PARTICLES_CONFIG", "config.yaml")
    assert relative_config_pin() == "config.yaml"
