# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The key file is trusted only from a place no other user may change (2026-10-28-2 review).

AA8: every directory the key file's name resolves through is judged, so a key
directory reached through a link in a directory others write is unusable, and
the file must be mode 0600. AA42: a sticky shared directory is accepted only as
the engine's own, and the lock ``keys create`` and ``keys revoke`` take is
checked and waited on for a bounded time. AA49: a setgid, group-writable key
directory is accepted only when no other account holds its group. AA65: an
operator record under a principal ``keys create`` refuses never authenticates.

Each test builds a scratch store and key file under ``tmp_path`` and sets
configuration only through the loader. No second local account is used; the
directory-only operations any writer could take run as the engine's own user.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import stat
import threading
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "") -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n" + body, encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))


@pytest.fixture
def engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, Path], None, None]:
    """An engine on per-client keys only, with reads requiring a key."""
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    _write_config(tmp_path, monkeypatch, "api:\n  require_auth_for_reads: true\n")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'k.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    from particles.api.app import app

    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    try:
        yield client, keydir
    finally:
        os.chmod(keydir, 0o700)
        _keys.reset_key_cache()
        _keys.forget_key_files()
        reset_config()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# AA8: the whole chain of directories, and the file's mode.
# ---------------------------------------------------------------------------


@pytest.fixture
def linked_key_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    """``<tmp>/shared/keys`` is a link, in a world-writable directory, to the operator's keys."""
    from particles.api import _keys
    from particles.api._keys import Role
    from particles.config import reset_config

    _write_config(tmp_path, monkeypatch)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(elsewhere / "engine-keys.json"))
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    token, _ = _keys.mint(Role.AGENT, "agent:someone")

    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o777)
    (shared / "keys").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(shared / "keys" / "engine-keys.json"))
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    yield token
    os.chmod(shared, 0o755)
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def test_a_key_file_reached_through_a_link_in_a_directory_others_write_is_unusable(
    linked_key_dir: str,
) -> None:
    """A key file whose directory others can swap authenticates nothing."""
    from particles.api import _keys

    with pytest.raises(_keys.KeyFileUnusable):
        _keys.lookup(linked_key_dir)


def test_a_key_file_reached_through_a_link_in_a_private_directory_is_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a link in a directory only the engine writes is judged, not refused as a link."""
    from particles.api import _keys
    from particles.api._keys import Role
    from particles.config import reset_config

    _write_config(tmp_path, monkeypatch)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "via").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(tmp_path / "via" / "engine-keys.json"))
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    try:
        token, record = _keys.mint(Role.READER, "viewer")
        found = _keys.lookup(token)
        assert found is not None and found.key_id == record.key_id
    finally:
        _keys.reset_key_cache()
        _keys.forget_key_files()


def test_a_key_file_other_users_may_read_is_unusable(engine: tuple[TestClient, Path]) -> None:
    """Only the 0600 file keys create writes is read."""
    from particles.api import _keys
    from particles.api._keys import Role

    client, keydir = engine
    token, _ = _keys.mint(Role.READER, "viewer")
    os.chmod(keydir / "engine-keys.json", 0o644)
    _keys.reset_key_cache()
    assert client.get("/events", headers=_bearer(token)).status_code == 503
    os.chmod(keydir / "engine-keys.json", 0o600)
    _keys.reset_key_cache()
    assert client.get("/events", headers=_bearer(token)).status_code == 200


# ---------------------------------------------------------------------------
# AA42: a sticky shared directory and the lock file.
# ---------------------------------------------------------------------------


def test_keys_revoke_never_waits_without_bound_on_a_lock_another_writer_holds(
    engine: tuple[TestClient, Path],
) -> None:
    """In a sticky shared key directory, a held lock file never stops a revocation forever."""
    from particles.api import _keys
    from particles.api._keys import Role

    client, keydir = engine
    _token, record = _keys.mint(Role.AGENT, "agent:phone")
    os.chmod(keydir, 0o1777)
    _keys.reset_key_cache()
    assert client.get("/auth/whoami", headers=_bearer(_token)).status_code == 200

    lock = keydir / ".engine-keys.json.lock"
    lock.unlink(missing_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    os.fchmod(fd, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)

    outcome: dict[str, object] = {}

    def _revoke() -> None:
        try:
            outcome["record"] = _keys.revoke(record.key_id)
        except Exception as exc:  # noqa: BLE001 - a refusal is an acceptable answer
            outcome["error"] = exc

    worker = threading.Thread(target=_revoke, daemon=True)
    worker.start()
    worker.join(timeout=3.0)
    finished = not worker.is_alive()
    os.close(fd)
    worker.join(timeout=5.0)
    assert finished, "keys revoke waited without bound on a lock file another writer holds"
    assert isinstance(outcome.get("error"), _keys.KeyFileUnusable)


def test_a_private_lock_another_process_holds_is_waited_on_for_a_bounded_time(
    engine: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock held past the bound refuses with a message naming the lock file."""
    from particles.api import _keys
    from particles.api._keys import Role

    _client, keydir = engine
    _token, record = _keys.mint(Role.AGENT, "agent:phone")
    monkeypatch.setattr(_keys, "_LOCK_WAIT_SECONDS", 0.2)
    lock = keydir / ".engine-keys.json.lock"
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(_keys.KeyRuleError, match="lock"):
            _keys.revoke(record.key_id)
    finally:
        os.close(fd)
    assert not _keys.revoke(record.key_id).live


def test_the_directory_holding_the_key_file_is_sticky_only_as_the_engines_own() -> None:
    """A sticky directory every user writes, owned by root (``/tmp``), cannot hold the key file."""
    from particles.api import _keys
    from particles.corpus.operator_only import NodeFacts

    tmp_like = NodeFacts(Path("/tmp"), True, True, 0, stat.S_IFDIR | 0o1777)
    assert _keys._directory_refusal(tmp_like, holding=True) is not None
    # Above the key file the sticky bit closes renaming, so the same directory passes.
    assert _keys._directory_refusal(tmp_like, holding=False) is None
    own = NodeFacts(Path("/x"), True, True, os.geteuid(), stat.S_IFDIR | 0o1777)
    assert _keys._directory_refusal(own, holding=True) is None


# ---------------------------------------------------------------------------
# AA49: a setgid, group-writable directory of a group other accounts share.
# ---------------------------------------------------------------------------


#: macOS puts every local account in ``everyone`` (12) and, as its primary
#: group, ``staff`` (20).
_EVERY_ACCOUNT_GROUPS = (12, 20)


def _a_group_every_account_shares() -> int:
    groups = {os.getegid(), *os.getgroups()}
    for gid in _EVERY_ACCOUNT_GROUPS:
        if gid in groups:
            return gid
    pytest.skip("this host has no group every local account belongs to")


def test_a_key_directory_every_local_account_may_write_into_is_refused(
    engine: tuple[TestClient, Path],
) -> None:
    """A setgid, group-writable key directory whose group is every account's is refused."""
    from particles.api import _keys
    from particles.api._keys import Role

    client, keydir = engine
    token, _ = _keys.mint(Role.READER, "viewer")
    gid = _a_group_every_account_shares()
    os.chown(keydir, -1, gid)
    os.chmod(keydir, 0o2770)
    st = keydir.stat()
    assert st.st_gid == gid and st.st_mode & stat.S_IWGRP and st.st_mode & stat.S_ISGID
    _keys.reset_key_cache()
    assert client.get("/events", headers=_bearer(token)).status_code == 503


def _a_group_the_databases_show_shared() -> int:
    """A group this account holds that the group or account database shows another account in.

    The Linux path of ``_group_shared_with_others`` reads those databases, so a
    real group they list another member of exercises it on CI's runner too,
    where the macOS-only groups above are absent.
    """
    import grp
    import pwd

    euid = os.geteuid()
    me = pwd.getpwuid(euid).pw_name
    accounts = pwd.getpwall()
    for gid in sorted({os.getegid(), *os.getgroups()}):
        try:
            members = set(grp.getgrgid(gid).gr_mem)
        except KeyError:
            continue
        if members - {me, "root"} or any(
            a.pw_gid == gid and a.pw_uid not in {euid, 0} for a in accounts
        ):
            return gid
    pytest.skip("this account holds no group the databases show another account in")


def test_a_key_directory_another_listed_account_may_write_into_is_refused(
    engine: tuple[TestClient, Path],
) -> None:
    """A setgid, group-writable key directory whose group lists another account is refused."""
    from particles.api import _keys
    from particles.api._keys import Role

    client, keydir = engine
    token, _ = _keys.mint(Role.READER, "viewer")
    gid = _a_group_the_databases_show_shared()
    os.chown(keydir, -1, gid)
    os.chmod(keydir, 0o2770)
    st = keydir.stat()
    assert st.st_gid == gid and st.st_mode & stat.S_IWGRP and st.st_mode & stat.S_ISGID
    _keys.reset_key_cache()
    assert client.get("/events", headers=_bearer(token)).status_code == 503


def test_a_revoked_key_stays_revoked_when_a_directory_writer_restores_the_old_file(
    engine: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A revocation is never undone by a link and a rename inside a group-shared key directory."""
    from particles.api import _keys
    from particles.api._keys import Role

    client, keydir = engine
    leaked, record = _keys.mint(Role.AGENT, "agent:phone")
    path = keydir / "engine-keys.json"
    kept = keydir / ".kept"
    os.link(path, kept)  # a directory writer's link to the live inode
    _keys.revoke(record.key_id)
    os.chown(keydir, -1, os.getegid())
    os.chmod(keydir, 0o2770)
    monkeypatch.setattr(_keys, "_group_shared_with_others", lambda gid: True)
    os.rename(kept, path)  # a directory writer's rename over the new file
    _keys.reset_key_cache()
    resp = client.get("/auth/whoami", headers=_bearer(leaked))
    assert resp.status_code != 200, resp.text


def test_an_unknown_gid_counts_as_shared() -> None:
    """A gid neither the group nor the account database lists counts as possibly shared.

    Inside a container the databases are the image's own, so a host group on a
    bind-mounted directory is one they never list, whoever holds it on the host
    (security review 2026-10-30-1, AB53).
    """
    import grp
    import sys

    from particles.api import _keys

    gid = 987_654
    try:
        grp.getgrgid(gid)
    except KeyError:
        pass
    else:
        pytest.skip("this host knows the test gid")
    if sys.platform == "darwin":
        assert _keys._group_shared_with_others(20)
    assert _keys._group_shared_with_others(gid)


def test_control_the_engine_owned_private_directory_is_accepted(
    engine: tuple[TestClient, Path],
) -> None:
    """Control: the default shape (owner-only directory) keeps working."""
    from particles.api import _keys
    from particles.api._keys import Role

    client, keydir = engine
    token, _ = _keys.mint(Role.READER, "viewer")
    _keys.reset_key_cache()
    assert client.get("/events", headers=_bearer(token)).status_code == 200
    assert json.loads((keydir / "engine-keys.json").read_text(encoding="utf-8"))["version"] == 1


# ---------------------------------------------------------------------------
# AA65: the record rule repeats check_mint's operator-principal rule.
# ---------------------------------------------------------------------------


def _rewrite(path: Path, key_id: str, **fields: object) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["keys"]:
        if item["key_id"] == key_id:
            item.update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)


@pytest.mark.parametrize("principal", ["claude-code-hook", "mcp:claude-code", "viewer"])
def test_an_operator_record_under_a_principal_minting_refuses_never_authenticates(
    engine: tuple[TestClient, Path], principal: str
) -> None:
    """A hand-written operator record whose principal is not the operator's never authenticates."""
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role
    from particles.config import get_config

    _client, keydir = engine
    with pytest.raises(KeyRuleError):
        _keys.check_mint(Role.OPERATOR, principal, None, reserved_ok=True)
    token, record = _keys.mint(Role.OPERATOR, get_config().curation.operator_identity)
    _rewrite(keydir / "engine-keys.json", record.key_id, principal=principal)
    _keys.reset_key_cache()
    assert _keys.lookup(token) is None
