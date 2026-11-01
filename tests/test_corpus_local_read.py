# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``read_local_file``: one descriptor, judged by what it opened (security review N22, N24).

Every hostile tree is built under ``tmp_path``.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest

from particles.corpus.local_read import LocalReadRefused, read_local_file, within


def test_reads_a_regular_file(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_bytes(b"hello")
    read = read_local_file(f, allowed=within(tmp_path))
    assert read.content == b"hello"
    assert read.real_path == f.resolve()
    assert read.stat.st_size == 5


def test_a_symlink_is_refused_unless_followed(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_bytes(b"t")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    with pytest.raises(LocalReadRefused) as info:
        read_local_file(link)
    assert info.value.errno == errno.ELOOP
    assert read_local_file(link, follow_symlinks=True).content == b"t"


def test_a_followed_link_is_still_judged_by_its_target(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"secret")
    (root / "note.md").symlink_to(outside)
    with pytest.raises(LocalReadRefused) as info:
        read_local_file(root / "note.md", follow_symlinks=True, allowed=within(root))
    assert info.value.errno == errno.EACCES


def test_a_swapped_parent_is_judged_on_the_descriptor(tmp_path: Path) -> None:
    """A directory replaced by a symlink after any name check reads nothing outside."""
    root = tmp_path / "root"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "note.md").write_bytes(b"outside")
    (root / "sub").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(LocalReadRefused):
        read_local_file(root / "sub" / "note.md", allowed=within(root))


def test_a_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.md"
    os.mkfifo(fifo)
    with pytest.raises(LocalReadRefused) as info:
        read_local_file(fifo)
    assert info.value.errno == errno.EINVAL


def test_a_file_over_the_cap_is_refused_before_reading(tmp_path: Path) -> None:
    f = tmp_path / "big.md"
    f.write_bytes(b"x" * 11)
    with pytest.raises(LocalReadRefused) as info:
        read_local_file(f, max_bytes=10)
    assert info.value.errno == errno.EFBIG
    assert read_local_file(f, max_bytes=11).content == b"x" * 11


def test_a_missing_file_is_an_ordinary_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_local_file(tmp_path / "absent.md")


def test_a_file_another_uid_owns_is_refused_when_an_owner_is_named(tmp_path: Path) -> None:
    """``owner=`` refuses the descriptor of a file another uid owns."""
    f = tmp_path / "AGENTS.md"
    f.write_bytes(b"a rule")
    mine = os.stat(f).st_uid
    assert read_local_file(f, owner=mine).content == b"a rule"
    with pytest.raises(LocalReadRefused) as info:
        read_local_file(f, owner=mine + 1)
    assert info.value.errno == errno.EPERM
    assert read_local_file(f).content == b"a rule"


def test_an_owner_function_is_judged_on_the_opened_files_path(tmp_path: Path) -> None:
    """A callable ``owner`` gets the descriptor's resolved path, never the name (R15)."""
    real = tmp_path / "real"
    real.mkdir()
    f = real / "AGENTS.md"
    f.write_bytes(b"a rule")
    link = tmp_path / "via"
    os.symlink(real, link)
    mine = os.stat(f).st_uid
    seen: list[Path] = []

    def _owner(path: Path) -> int | None:
        seen.append(path)
        return mine + 1 if path.parent == real.resolve() else None

    with pytest.raises(LocalReadRefused) as info:
        read_local_file(link / "AGENTS.md", owner=_owner)
    assert info.value.errno == errno.EPERM
    assert seen == [f.resolve()]
    assert read_local_file(link / "AGENTS.md", owner=lambda _p: None).content == b"a rule"


@pytest.mark.asyncio
async def test_split_by_date_holds_a_named_file_to_the_local_files_cap(
    db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposit FILE --split-by-date`` reads its file under ``local_files.max_bytes``
    like every other local read (security review 2026-10-26-1, W63)."""
    from particles.config import get_config
    from particles.corpus.deposit import split_file_by_date

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setattr(get_config().local_files, "max_bytes", 64)
    big = tmp_path / "journal.md"
    big.write_text("2026-01-01\n" + "x" * 200 + "\n2026-01-02\n" + "y" * 200 + "\n")
    with pytest.raises(LocalReadRefused):
        await split_file_by_date(db_session, big)  # type: ignore[arg-type]


def test_a_read_with_an_owner_refuses_a_file_others_can_write(tmp_path: Path) -> None:
    """The owner check also refuses a file its group or other users may write (W23)."""
    path = tmp_path / "CLAUDE.md"
    path.write_text("# rules\n")
    os.chmod(path, 0o620)
    with pytest.raises(LocalReadRefused) as caught:
        read_local_file(path, owner=os.geteuid())
    assert caught.value.errno == errno.EPERM
    os.chmod(path, 0o644)
    assert read_local_file(path, owner=os.geteuid()).content == b"# rules\n"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACLs")
def test_an_owner_read_refuses_a_file_an_acl_lets_others_write(tmp_path: Path) -> None:
    """A read with an owner refuses a file an ACL entry lets every other user write (G20)."""
    import subprocess

    path = tmp_path / "CLAUDE.md"
    path.write_text("# rules\n")
    os.chmod(path, 0o644)
    assert read_local_file(path, owner=os.geteuid()).content == b"# rules\n"
    subprocess.run(["chmod", "+a", "everyone allow write", str(path)], check=True)
    with pytest.raises(LocalReadRefused):
        read_local_file(path, owner=os.geteuid())
    assert read_local_file(path).content == b"# rules\n"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACLs")
def test_an_acl_that_only_denies_or_reads_is_not_a_write_grant(tmp_path: Path) -> None:
    import subprocess

    path = tmp_path / "CLAUDE.md"
    path.write_text("# rules\n")
    os.chmod(path, 0o644)
    subprocess.run(["chmod", "+a", "everyone deny write", str(path)], check=True)
    subprocess.run(["chmod", "+a", "everyone allow read", str(path)], check=True)
    assert read_local_file(path, owner=os.geteuid()).content == b"# rules\n"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACLs")
def test_an_owner_read_refuses_a_file_whose_inherited_acl_lets_others_write(
    tmp_path: Path,
) -> None:
    """An inherited allow-write entry counts as a direct one does (O13, O22)."""
    import subprocess

    folder = tmp_path / "repo"
    folder.mkdir()
    subprocess.run(
        ["chmod", "+a", "everyone allow write,append,file_inherit,directory_inherit", str(folder)],
        check=True,
    )
    path = folder / "CLAUDE.md"
    path.write_text("# rules\n")
    os.chmod(path, 0o644)
    listing = subprocess.run(["ls", "-le", str(path)], capture_output=True, text=True).stdout
    assert "inherited allow write" in listing
    with pytest.raises(LocalReadRefused):
        read_local_file(path, owner=os.geteuid())


def test_an_acl_entry_counts_by_its_kind_whatever_flags_follow() -> None:
    """``allow`` with flags grants; ``deny`` with flags and ``only_inherit`` do not (O13, O22)."""
    from particles.corpus.local_read import _acl_line_grants_write

    uuid = "ABCDEFAB-CDEF-ABCD-EFAB-CDEF0000000C"
    owner = os.geteuid()

    def entry(kind: str) -> str:
        return f"group:{uuid}:everyone:12:{kind}:write,append"

    assert _acl_line_grants_write(entry("allow"), owner)
    assert _acl_line_grants_write(entry("allow,inherited"), owner)
    assert _acl_line_grants_write(entry("allow,file_inherit,directory_inherit"), owner)
    assert not _acl_line_grants_write(entry("allow,file_inherit,only_inherit"), owner)
    assert not _acl_line_grants_write(entry("deny,inherited"), owner)
