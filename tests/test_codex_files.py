# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Resolver containment, archive movement and conflicting-copy refusals."""

from pathlib import Path

import pytest

from particles.api.cli._codex_files import inventory, read_contained, resolve_rollout
from particles.api.cli._hook_state import advisory_lock
from particles.api.cli._project_identity import record_identity_root, recorded_identity_roots
from particles.config import get_config
from particles.extraction.codex_transcript import UnsupportedTranscript
from tests.test_codex_transcript import SID, blob, message, metadata


def rollout(root: Path) -> Path:
    path = root / "sessions" / "2026" / f"rollout-2026-11-01-{SID}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(blob(metadata(), message("synthetic")))
    return path


def test_archive_resolution_and_event_path_is_irrelevant(tmp_path: Path) -> None:
    path = rollout(tmp_path)
    archive = tmp_path / "archived_sessions" / path.name
    archive.parent.mkdir()
    path.rename(archive)
    result = resolve_rollout(tmp_path, SID, 10000, "/tmp/repository")
    assert result.path == archive
    with pytest.raises(UnsupportedTranscript, match="identity_or_cwd"):
        resolve_rollout(tmp_path, SID, 10000, "/another/root")


def test_symlink_components_traversal_and_oversize_refuse(tmp_path: Path) -> None:
    home, other = tmp_path / "home", tmp_path / "other"
    home.mkdir()
    path = rollout(other)
    (home / "sessions").symlink_to(other / "sessions", target_is_directory=True)
    assert inventory(home) == []
    with pytest.raises(OSError):
        read_contained(home, home / "sessions" / "2026" / path.name, 10000)
    with pytest.raises(UnsupportedTranscript, match="invalid_rollout_path"):
        read_contained(home, home / "sessions" / ".." / "outside.jsonl", 10000)
    with pytest.raises(UnsupportedTranscript, match="oversized"):
        read_contained(other, path, 5)


def test_conflicting_archive_copy_and_hardlink_refuse(tmp_path: Path) -> None:
    path = rollout(tmp_path)
    archive = tmp_path / "archived_sessions" / path.name
    archive.parent.mkdir()
    archive.write_text(blob(metadata(), message("different")))
    with pytest.raises(UnsupportedTranscript, match="ambiguous_session_copies"):
        resolve_rollout(tmp_path, SID, 10000)
    archive.unlink()
    archive.hardlink_to(path)
    with pytest.raises(UnsupportedTranscript, match="not_regular"):
        resolve_rollout(tmp_path, SID, 10000)


def test_diagnostic_union_and_lock_contention_are_not_identity_authority(tmp_path: Path) -> None:
    config = get_config()
    config.claude_code.state_dir = str(tmp_path / "claude")
    config.codex.state_dir = str(tmp_path / "codex")
    config.codex.identity_lock_path = str(tmp_path / "identity.lock")
    key = "same-key"
    assert record_identity_root(key, Path("/first")) == ["/first"]
    assert record_identity_root(key, Path("/second"), codex=True) == ["/first", "/second"]
    with advisory_lock(Path(config.codex.identity_lock_path)) as acquired:
        assert acquired
        assert record_identity_root(key, Path("/blocked")) == ["/first", "/second"]
    assert recorded_identity_roots(key) == ["/first", "/second"]
