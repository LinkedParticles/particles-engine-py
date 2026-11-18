# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""This installation's host id: minted once, never believed elsewhere."""

from __future__ import annotations

import json
import stat
from collections.abc import Generator
from pathlib import Path

import pytest

from particles import host_identity
from particles.config import reset_config


@pytest.fixture
def id_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    path = tmp_path / "state" / "host-id"
    monkeypatch.setenv("PARTICLES_HOST_ID_FILE", str(path))
    monkeypatch.setattr(host_identity, "machine_identifier", lambda: "machine-a")
    reset_config()
    host_identity.reset_host_id_cache()
    yield path
    host_identity.reset_host_id_cache()


def test_the_first_read_mints_a_private_id_and_later_reads_keep_it(id_file: Path) -> None:
    first = host_identity.host_identity()
    assert first.minted
    assert stat.S_IMODE(id_file.stat().st_mode) == 0o600
    host_identity.reset_host_id_cache()
    again = host_identity.host_identity()
    assert again.host_id == first.host_id and not again.minted
    assert host_identity.local_host_id() == first.host_id


def test_a_file_copied_from_another_machine_is_not_believed(
    id_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    laptop = host_identity.host_identity().host_id
    host_identity.reset_host_id_cache()
    monkeypatch.setattr(host_identity, "machine_identifier", lambda: "machine-b")
    mini = host_identity.host_identity()
    assert mini.minted and mini.replaced == laptop and mini.host_id != laptop
    assert json.loads(id_file.read_text())["replaced"] == laptop


def test_a_relative_id_file_is_refused_and_this_host_claims_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARTICLES_HOST_ID_FILE", "host-id")
    reset_config()
    host_identity.reset_host_id_cache()
    with pytest.raises(host_identity.HostIdUnusable):
        host_identity.host_identity()
    assert host_identity.local_host_id() is None
    host_identity.reset_host_id_cache()


def test_an_id_file_others_may_write_is_unusable(id_file: Path) -> None:
    host_identity.host_identity()
    id_file.chmod(0o666)
    host_identity.reset_host_id_cache()
    with pytest.raises(host_identity.HostIdUnusable):
        host_identity.host_identity()


def test_a_malformed_file_is_replaced_by_a_fresh_id(id_file: Path) -> None:
    id_file.parent.mkdir(parents=True)
    id_file.write_text("not json")
    id_file.chmod(0o600)
    identity = host_identity.host_identity()
    assert identity.minted
    assert json.loads(id_file.read_text())["host_id"] == identity.host_id


def test_two_processes_minting_at_once_agree_on_one_id(
    id_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loser of the exclusive link reads the winner's id rather than its own."""
    real_write = host_identity._write

    def _racing(path: Path, machine_id: str, *, replaced: str | None, exclusive: bool) -> str:
        real_write(path, machine_id, replaced=None, exclusive=False)  # the other process
        return real_write(path, machine_id, replaced=replaced, exclusive=exclusive)

    monkeypatch.setattr(host_identity, "_write", _racing)
    identity = host_identity.host_identity()
    assert not identity.minted
    assert identity.host_id == json.loads(id_file.read_text())["host_id"]


def test_an_empty_machine_read_never_rotates_the_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ioreg that timed out under load reads as unknown, not as another machine."""
    monkeypatch.setenv("PARTICLES_HOST_ID_FILE", str(tmp_path / "host-id"))
    reset_config()
    host_identity.reset_host_id_cache()
    monkeypatch.setattr(host_identity, "machine_identifier", lambda: "machine-a")
    first = host_identity.host_identity().host_id
    host_identity.reset_host_id_cache()
    monkeypatch.setattr(host_identity, "machine_identifier", lambda: "")
    assert host_identity.host_identity().host_id == first
    host_identity.reset_host_id_cache()
