# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The read-only lock checks the installers ask before a dry run or a write."""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.api.cli._hook_state import (
    installer_lock_refusal,
    private_dir,
    private_dir_refusal,
)
from particles.config import reset_config


def _lock_in(monkeypatch: pytest.MonkeyPatch, directory: Path) -> Path:
    monkeypatch.setenv("PARTICLES_PROJECT_IDENTITY_LOCK", str(directory / "identity.lock"))
    reset_config()
    return directory / "harness-installer.lock"


class TestPrivateDirRefusal:
    def test_a_missing_directory_is_not_refused_and_not_created(self, tmp_path: Path) -> None:
        target = tmp_path / "absent" / "state"
        assert private_dir_refusal(target) is None
        assert not (tmp_path / "absent").exists()

    def test_a_private_directory_passes(self, tmp_path: Path) -> None:
        target = tmp_path / "state"
        target.mkdir(mode=0o700)
        target.chmod(0o700)
        assert private_dir_refusal(target) is None

    def test_a_shared_directory_names_its_mode_and_the_fix(self, tmp_path: Path) -> None:
        target = tmp_path / "state"
        target.mkdir()
        target.chmod(0o755)
        refusal = private_dir_refusal(target)
        assert refusal is not None
        assert "0755" in refusal and f"chmod 700 {target}" in refusal

    def test_a_link_is_refused(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir(mode=0o700)
        link = tmp_path / "link"
        link.symlink_to(real)
        assert private_dir_refusal(link) is not None

    def test_private_dir_refuses_what_the_check_refuses(self, tmp_path: Path) -> None:
        target = tmp_path / "state"
        target.mkdir()
        target.chmod(0o755)
        with pytest.raises(ValueError, match="hook_state_directory_not_private"):
            private_dir(target)


class TestInstallerLockRefusal:
    def test_a_private_lock_directory_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "particles"
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        _lock_in(monkeypatch, directory)
        assert installer_lock_refusal() is None

    def test_a_shared_lock_directory_is_refused_without_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "particles"
        directory.mkdir()
        directory.chmod(0o755)
        lock = _lock_in(monkeypatch, directory)
        refusal = installer_lock_refusal()
        assert refusal is not None and "chmod 700" in refusal
        assert not lock.exists()

    def test_a_shared_lock_file_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "particles"
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        lock = _lock_in(monkeypatch, directory)
        lock.write_text("")
        lock.chmod(0o644)
        refusal = installer_lock_refusal()
        assert refusal is not None and str(lock) in refusal
