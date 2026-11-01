# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0
"""What the local refresh tier admits and what it stores (security review 2026-10-24, third delta).

Safe properties:

* a recorded root's ``.git`` file cannot widen the allowlist to a path outside
  every recorded root (K6 through ``_main_checkout``; N38);
* discovery never records the home directory as a project root (K9);
* a hard link inside a root does not deliver the bytes of a file outside it
  (K10, D23, N24);
* a parent directory swapped for a symlink is refused at the read (N22, fixed);
* a memory-file snapshot the refresh captures goes through the same redaction
  the harvest applies (the refresh path stores raw bytes).
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from particles.config import get_config
from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
from particles.corpus.fetch import is_refresh_allowed, maybe_refetch, refresh_refusal
from particles.corpus.local_read import LocalReadRefused, read_local_file, within
from particles.corpus.store import CorpusEntryRow, get_latest_snapshot


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_recorded_root_git_file_cannot_admit_an_outside_file(tmp_path: Path) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    target = (tmp_path / "elsewhere" / "private.key").resolve()
    target.parent.mkdir()
    target.write_text("not a rule file")
    (project / ".git").write_text(f"gitdir: {target}/.git/worktrees/x\n")

    assert refresh_refusal(target, recorded_roots=[project], deposited_by="rule-sync") is not None


def test_recorded_root_git_file_cannot_admit_the_filesystem_root(tmp_path: Path) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    (project / ".git").write_text("gitdir: /.git/worktrees/x\n")
    elsewhere = (tmp_path / "unrelated" / "AGENTS.md").resolve()

    assert not is_refresh_allowed(elsewhere, recorded_roots=[project], deposited_by="rule-sync")


def test_discovery_never_records_the_home_directory(_isolated_home: Path) -> None:
    from particles.corpus.rule_sources import discover_default_roots

    (_isolated_home / ".git").mkdir()  # a dotfiles repository
    (_isolated_home / ".claude").mkdir()
    work = _isolated_home / "notes"
    work.mkdir()

    assert _isolated_home.resolve() not in [r.resolve() for r in discover_default_roots(work)]


def test_hard_link_in_a_walked_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes")
    link = root / "note.md"
    os.link(outside, link)

    with pytest.raises(LocalReadRefused):
        read_local_file(link, max_bytes=1 << 20, allowed=within(root))


def test_hard_link_in_a_refresh_root_is_refused(tmp_path: Path) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes")
    link = project / "AGENTS.md"
    os.link(outside, link)

    def _allowed(p: Path) -> bool:
        return is_refresh_allowed(p, recorded_roots=[project], deposited_by="rule-sync")

    assert _allowed(link)
    with pytest.raises(LocalReadRefused):
        read_local_file(link, max_bytes=1 << 20, allowed=_allowed)


def test_parent_swapped_for_a_symlink_is_refused_at_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed_root = (tmp_path / "notes").resolve()
    allowed_root.mkdir()
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (outside / "a.md").write_text("outside bytes")
    (allowed_root / "sub").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(allowed_root)])

    def _allowed(p: Path) -> bool:
        return is_refresh_allowed(p, deposited_by="operator")

    with pytest.raises(LocalReadRefused):
        read_local_file(allowed_root / "sub" / "a.md", max_bytes=1 << 20, allowed=_allowed)


async def test_refreshed_memory_file_snapshot_is_redacted(
    db_session: Any, _isolated_home: Path
) -> None:
    memory = _isolated_home / ".claude" / "projects" / "-repo" / "memory"
    memory.mkdir(parents=True)
    note = memory / "note.md"
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    note.write_text(f"the deploy key is {secret}\n", encoding="utf-8")
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=note.resolve().as_uri(),
        fetch_policy=FetchPolicy.LAZY,
        mutability=Mutability.MUTABLE,
        deposited_by="claude-code-hook",
        tags=["claude-code", "memory-file", "project:-repo"],
    )
    db_session.add(CorpusEntryRow.from_model(entry))
    await db_session.commit()

    snap = await maybe_refetch(db_session, entry.entry_id)
    await db_session.commit()
    assert snap is not None
    latest = await get_latest_snapshot(db_session, entry.entry_id)
    assert latest is not None and latest.archive_path is not None
    from particles.corpus.deposit import load_blob

    stored = load_blob(latest.content_hash)
    assert secret.encode() not in stored


def test_a_recorded_worktree_file_root_is_not_admitted_through_its_derived_main(
    tmp_path: Path,
) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    main = (tmp_path / "main").resolve()
    (main / ".git" / "worktrees" / "x").mkdir(parents=True)
    (project / ".git").write_text(f"gitdir: {main}/.git/worktrees/x\n")
    # The back-link git writes, without which the main checkout is not derived
    # (security review K6, P12; corpus/worktree.py).
    (main / ".git" / "worktrees" / "x" / "gitdir").write_text(f"{project / '.git'}\n")
    (main / "AGENTS.md").write_text("main checkout rule\n")
    (main / "notes.txt").write_text("not a rule file\n")

    assert is_refresh_allowed(
        main / "AGENTS.md", recorded_roots=[project], deposited_by="rule-sync"
    )
    assert not is_refresh_allowed(main, recorded_roots=[project], deposited_by="rule-sync")
    assert not is_refresh_allowed(
        main / "notes.txt", recorded_roots=[project], deposited_by="rule-sync"
    )


def test_a_recorded_home_row_admits_nothing(_isolated_home: Path) -> None:
    rule = _isolated_home / "notes" / "AGENTS.md"
    rule.parent.mkdir()
    rule.write_text("a rule under home\n")

    assert not is_refresh_allowed(
        rule, recorded_roots=[_isolated_home.resolve()], deposited_by="rule-sync"
    )
    assert not is_refresh_allowed(rule, recorded_roots=[Path("/")], deposited_by="rule-sync")


def test_dot_dot_in_a_file_uri_is_judged_after_resolution(tmp_path: Path) -> None:
    from particles.corpus.fetch import path_from_file_uri

    project = (tmp_path / "project").resolve()
    project.mkdir()
    (tmp_path / "AGENTS.md").write_text("outside the project\n")
    uri = f"file://{project}/../AGENTS.md"

    assert not is_refresh_allowed(
        path_from_file_uri(uri), recorded_roots=[project], deposited_by="rule-sync"
    )


def test_an_unresolved_alias_resolves_to_the_same_root(tmp_path: Path) -> None:
    # The shape of macOS ``/tmp`` → ``/private/tmp``: the entry's path goes
    # through a symlinked parent, the recorded root is the resolved path.
    real = (tmp_path / "real").resolve()
    project = real / "project"
    project.mkdir(parents=True)
    (project / "AGENTS.md").write_text("rule\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    assert is_refresh_allowed(
        alias / "project" / "AGENTS.md", recorded_roots=[project], deposited_by="rule-sync"
    )


async def test_record_skips_the_home_directory(db_session: Any, _isolated_home: Path) -> None:
    from particles.corpus.rule_roots import list_rule_source_roots, record_rule_source_roots

    project = _isolated_home / "src" / "repo"
    project.mkdir(parents=True)
    await record_rule_source_roots(db_session, [_isolated_home, Path("/"), project], by="rule-sync")

    assert await list_rule_source_roots(db_session) == [project.resolve()]


def test_resolution_refuses_an_explicit_home_root(_isolated_home: Path) -> None:
    from particles.corpus.rule_sources import resolve_rule_sources

    (_isolated_home / "AGENTS.md").write_text("a rule\n")
    resolution = resolve_rule_sources([str(_isolated_home)])

    assert resolution.files == []
    assert resolution.roots == []
    assert resolution.broad == [_isolated_home.resolve()]


async def test_rule_sync_refuses_a_hard_linked_rule_file(db_session: Any, tmp_path: Path) -> None:
    from particles.corpus.rule_sources import sync_rule_sources

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes\n")
    os.link(outside, project / "AGENTS.md")

    report = await sync_rule_sources(db_session, [str(project)])

    assert report.deposited == []
    assert [p.name for p, _why in report.failed] == ["AGENTS.md"]


async def test_a_masked_refresh_leaves_the_byte_identity_tier(
    db_session: Any, _isolated_home: Path
) -> None:
    memory = _isolated_home / ".claude" / "projects" / "-repo" / "memory"
    memory.mkdir(parents=True)
    note = memory / "note.md"
    note.write_text("token: " + "A1b2C3d4E5f6G7h8" + "\n", encoding="utf-8")
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=note.resolve().as_uri(),
        fetch_policy=FetchPolicy.LAZY,
        mutability=Mutability.MUTABLE,
        deposited_by="claude-code-hook",
        tags=["claude-code", "memory-file"],
    )
    db_session.add(CorpusEntryRow.from_model(entry))
    await db_session.commit()

    assert await maybe_refetch(db_session, entry.entry_id) is not None
    await db_session.commit()
    row = await db_session.get(CorpusEntryRow, entry.entry_id)
    assert row.fetch_policy == FetchPolicy.NEVER.value


async def test_an_operator_lazy_file_is_stored_as_its_bytes(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "notes").resolve()
    root.mkdir()
    note = root / "a.md"
    body = "the deploy key is sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2" + "\n"
    note.write_text(body, encoding="utf-8")
    monkeypatch.setattr(get_config().local_refresh, "allowed_roots", [str(root)])
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=note.as_uri(),
        fetch_policy=FetchPolicy.LAZY,
        mutability=Mutability.MUTABLE,
        deposited_by="operator",
        tags=[],
    )
    db_session.add(CorpusEntryRow.from_model(entry))
    await db_session.commit()

    snap = await maybe_refetch(db_session, entry.entry_id)
    assert snap is not None
    from particles.corpus.deposit import load_blob

    assert load_blob(snap.content_hash) == body.encode()


macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="APFS firmlinks and case folding")


def _firmlink_alias(path: Path) -> Path:
    """The same directory named through the Data volume (``/System/Volumes/Data``)."""
    return Path("/System/Volumes/Data") / str(path.resolve()).lstrip("/")


@macos_only
def test_is_broad_root_sees_home_through_the_data_volume_firmlink(_isolated_home: Path) -> None:
    """The home directory named through its firmlink is broad (security review Q50)."""
    from particles.config import is_broad_root

    alias = _firmlink_alias(_isolated_home)
    assert alias.is_dir()
    assert is_broad_root(alias)
    assert is_broad_root(alias.parent)
    assert not is_broad_root(alias / "nonexistent")


@macos_only
def test_is_broad_root_sees_home_under_another_case(_isolated_home: Path) -> None:
    """The home directory in another letter case on a case-insensitive volume is broad (Q50)."""
    from particles.config import is_broad_root

    alias = Path(str(_isolated_home.resolve()).swapcase())
    if not alias.is_dir():
        pytest.skip("case-sensitive volume")
    assert is_broad_root(alias)


@macos_only
async def test_record_skips_a_firmlink_alias_of_home(db_session: Any, _isolated_home: Path) -> None:
    """record_rule_source_roots never records the home directory under another name (Q50)."""
    from particles.corpus.rule_roots import list_rule_source_roots, record_rule_source_roots

    project = _isolated_home / "src" / "repo"
    project.mkdir(parents=True)
    await record_rule_source_roots(
        db_session, [_firmlink_alias(_isolated_home), project], by="rule-sync"
    )
    assert await list_rule_source_roots(db_session) == [project.resolve()]


@macos_only
def test_the_volume_mount_that_holds_home_is_broad(_isolated_home: Path) -> None:
    """The Data volume's mount point holds the home directory under another name (R16)."""
    from particles.config import is_broad_root

    volume = Path("/System/Volumes/Data")
    held = _firmlink_alias(_isolated_home)
    if not (volume.is_dir() and held.exists()):
        pytest.skip("no data volume holding home")
    assert os.path.samestat(os.stat(held), os.stat(_isolated_home))
    assert is_broad_root(volume)


def test_a_directory_holding_home_under_its_trailing_names_is_broad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bind-mount style parent that reaches home by home's own tail is broad (R16).

    A bind mount puts home's own directory at ``holder/users/me``, so an
    ``lstat`` of that real directory reports home's identity; the test stands
    the mount in by answering that one ``lstat`` with home's. A symlink there
    is not a mount, and anyone can plant one, so it makes nothing broad
    (security review 2026-10-30-1, AB1).
    """
    import particles.config as config_mod
    from particles.config import is_broad_root

    home = (tmp_path / "real" / "users" / "me").resolve()
    home.mkdir(parents=True)
    holder = (tmp_path / "mount").resolve()
    (holder / "users" / "me").mkdir(parents=True)
    linked = (tmp_path / "linked").resolve()
    (linked / "users").mkdir(parents=True)
    os.symlink(home, linked / "users" / "me")
    project = (tmp_path / "project").resolve()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    real_lstat = os.lstat
    mount_point = holder / "users" / "me"

    def _bind_mounted(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if Path(path) == mount_point:
            return real_lstat(home)
        return real_lstat(path, *args, **kwargs)

    with mock.patch.object(config_mod.os, "lstat", _bind_mounted):
        assert is_broad_root(holder)
        assert is_broad_root(holder / "users")
    assert not is_broad_root(holder)  # a plain directory without the mount
    assert not is_broad_root(linked)
    assert not is_broad_root(linked / "users")
    assert not is_broad_root(project)
