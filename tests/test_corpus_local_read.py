# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``read_local_file``: one descriptor, judged by what it opened (security review N22, N24).

Every hostile tree is built under ``tmp_path``.
"""

from __future__ import annotations

import errno
import os
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
