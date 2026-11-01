# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Blob reads and writes stay inside the blob directory (security review 2026-10-25, Q4).

The first four tests are the audit's proofs, which failed at 342ede50: a
symlink planted at a blob's place in the blob directory (the compiled
``./corpus_blobs`` of a cloned working directory, or any configured one) sent
the write to a file outside it and returned a host file to the read.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from particles.config import reset_config
from particles.corpus.blob_io import BlobRefused, check_blob_root, read_blob, write_blob
from particles.corpus.deposit import sha256


def _compiled_defaults_in(clone: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No DATABASE_URL, no PARTICLES_CONFIG, no user config file, cwd = ``clone``."""
    for name in ("DATABASE_URL", "PARTICLES_CONFIG", "PARTICLES_BLOB_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(clone)
    reset_config()


def test_default_cwd_blob_read_stays_in_working_directory_blob_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With compiled defaults, a blob read returns no bytes of a file outside ./corpus_blobs."""
    from particles.corpus.deposit import blob_path, load_blob

    clone, home = tmp_path / "clone", tmp_path / "home"
    clone.mkdir()
    home.mkdir()
    host_file = tmp_path / "host.txt"
    host_file.write_bytes(b"host file")
    digest = "b" * 64
    (clone / "corpus_blobs" / digest[:2]).mkdir(parents=True)
    (clone / "corpus_blobs" / digest[:2] / digest).symlink_to(host_file)

    _compiled_defaults_in(clone, home, monkeypatch)
    try:
        assert not blob_path(digest).is_absolute()
        try:
            data: bytes | None = load_blob(digest)
        except OSError:
            data = None
    finally:
        reset_config()
    assert data != b"host file"


def test_default_cwd_blob_write_creates_nothing_outside_blob_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With compiled defaults, a blob write creates no file outside ./corpus_blobs."""
    from particles.corpus.deposit import save_blob

    clone, home = tmp_path / "clone", tmp_path / "home"
    clone.mkdir()
    home.mkdir()
    target = tmp_path / "elsewhere" / "created"
    target.parent.mkdir()
    content = b"bytes the depositor chose"
    digest = sha256(content)
    (clone / "corpus_blobs" / digest[:2]).mkdir(parents=True)
    (clone / "corpus_blobs" / digest[:2] / digest).symlink_to(target)

    _compiled_defaults_in(clone, home, monkeypatch)
    try:
        with contextlib.suppress(OSError):
            save_blob(content, digest)
    finally:
        reset_config()
    assert not target.exists()


def test_save_blob_never_writes_outside_blob_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blob write creates no file outside a configured blob directory either."""
    from particles.corpus.deposit import save_blob

    blobs = tmp_path / "corpus_blobs"
    outside = tmp_path / "outside" / "target"
    outside.parent.mkdir()
    content = b"predictable bytes"
    digest = sha256(content)
    (blobs / digest[:2]).mkdir(parents=True)
    (blobs / digest[:2] / digest).symlink_to(outside)

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(blobs))
    reset_config()
    with pytest.raises(BlobRefused):
        save_blob(content, digest)
    assert not outside.exists(), "the blob write followed a symlink out of the blob directory"


def test_load_blob_never_reads_outside_blob_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blob read returns no bytes from a file outside the blob directory."""
    from particles.corpus.deposit import load_blob

    blobs = tmp_path / "corpus_blobs"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"host file")
    digest = sha256(b"host file")  # even a link to bytes that match the hash
    (blobs / digest[:2]).mkdir(parents=True)
    (blobs / digest[:2] / digest).symlink_to(outside)

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(blobs))
    reset_config()
    with pytest.raises(FileNotFoundError):
        load_blob(digest)


def test_round_trip_and_an_existing_blob_is_kept(tmp_path: Path) -> None:
    content = b"a blob"
    digest = sha256(content)
    path = write_blob(tmp_path / "blobs", digest, content)
    assert path == tmp_path / "blobs" / digest[:2] / digest
    assert read_blob(tmp_path / "blobs", digest) == content
    # Content-addressed: a second write of the same blob leaves the file alone.
    assert write_blob(tmp_path / "blobs", digest, content) == path
    assert path.read_bytes() == content


def test_missing_blob_is_file_not_found(tmp_path: Path) -> None:
    (tmp_path / "blobs").mkdir()
    with pytest.raises(FileNotFoundError):
        read_blob(tmp_path / "blobs", "c" * 64)


def test_symlinked_shard_directory_is_refused(tmp_path: Path) -> None:
    content = b"sharded"
    digest = sha256(content)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / digest).write_bytes(content)
    root = tmp_path / "blobs"
    root.mkdir()
    (root / digest[:2]).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(BlobRefused):
        read_blob(root, digest)
    with pytest.raises(BlobRefused):
        write_blob(root, digest, b"new")


def test_symlinked_root_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "blobs"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(BlobRefused):
        write_blob(link, sha256(b"x"), b"x")
    assert not any(real.iterdir())
    with pytest.raises(BlobRefused):
        check_blob_root(link)


def test_relative_root_refuses_a_symlinked_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A working-directory root is judged component by component."""
    (tmp_path / "real").mkdir()
    (tmp_path / "data").symlink_to(tmp_path / "real", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BlobRefused):
        check_blob_root(Path("data") / "corpus_blobs")
    check_blob_root(Path("corpus_blobs"))  # absent, so nothing to refuse


def test_bytes_that_miss_their_hash_are_refused(tmp_path: Path) -> None:
    """A regular file planted at a blob's place (a hard link included) is not that blob."""
    digest = sha256(b"the real bytes")
    (tmp_path / "blobs" / digest[:2]).mkdir(parents=True)
    (tmp_path / "blobs" / digest[:2] / digest).write_bytes(b"planted bytes")
    with pytest.raises(BlobRefused):
        read_blob(tmp_path / "blobs", digest)


def test_blob_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    """The owner's tripwire shape: a file where the blob directory should be."""
    (tmp_path / "blobs").write_text("tripwire")
    with pytest.raises(BlobRefused):
        read_blob(tmp_path / "blobs", sha256(b"x"))


def test_a_blob_write_over_a_planted_regular_file_leaves_the_deposit_readable(
    tmp_path: Path,
) -> None:
    """After a blob write returns, reading that blob returns the bytes written.

    Security review 2026-10-25-2, R40 (the audit's proof,
    test_s3_blob_dir.py, failed at 9a774070): a write took any regular file
    already at the blob's name as the blob, so a file a repository ships at
    the name of content the owner later deposits turned the deposit into a
    missing blob.
    """
    content = b"# Project rules the owner will deposit\n"
    digest = sha256(content)
    shard = tmp_path / "blobs" / digest[:2]
    shard.mkdir(parents=True)
    (shard / digest).write_bytes(b"planted placeholder")
    write_blob(tmp_path / "blobs", digest, content)
    assert read_blob(tmp_path / "blobs", digest) == content


def test_a_blob_write_over_a_hard_link_replaces_only_the_name(tmp_path: Path) -> None:
    """The planted name is unlinked, so a hard-linked host file keeps its bytes."""
    host = tmp_path / "host.txt"
    host.write_bytes(b"host bytes")
    content = b"the deposit"
    digest = sha256(content)
    shard = tmp_path / "blobs" / digest[:2]
    shard.mkdir(parents=True)
    os.link(host, shard / digest)
    write_blob(tmp_path / "blobs", digest, content)
    assert host.read_bytes() == b"host bytes"
    assert read_blob(tmp_path / "blobs", digest) == content


def test_a_blob_write_refuses_a_fifo_at_its_place_without_blocking(tmp_path: Path) -> None:
    content = b"fifo"
    digest = sha256(content)
    shard = tmp_path / "blobs" / digest[:2]
    shard.mkdir(parents=True)
    os.mkfifo(shard / digest)
    with pytest.raises(BlobRefused):
        write_blob(tmp_path / "blobs", digest, content)


def test_a_failed_write_never_removes_another_writers_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer whose own write fails never unlinks the complete blob a concurrent writer
    put at the name (security review 2026-10-26-1, W61)."""
    from particles.corpus import blob_io

    content = b"the deposit's bytes\n"
    content_hash = sha256(content)
    root = tmp_path / "corpus_blobs"
    real_fdopen = os.fdopen
    calls = {"n": 0}

    def interleaved_fdopen(fd: int, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            # Another writer runs to completion between this create and this write.
            monkeypatch.setattr(blob_io.os, "fdopen", real_fdopen)
            write_blob(root, content_hash, content)
            os.close(fd)
            raise OSError(28, "No space left on device")
        return real_fdopen(fd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(blob_io.os, "fdopen", interleaved_fdopen)
    with pytest.raises(OSError):
        write_blob(root, content_hash, content)
    monkeypatch.setattr(blob_io.os, "fdopen", real_fdopen)
    assert read_blob(root, content_hash) == content


def test_a_failed_write_removes_its_own_partial_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.corpus import blob_io

    content = b"bytes\n"
    content_hash = sha256(content)
    root = tmp_path / "corpus_blobs"

    def failing_fdopen(fd: int, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        os.close(fd)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(blob_io.os, "fdopen", failing_fdopen)
    with pytest.raises(OSError):
        write_blob(root, content_hash, content)
    assert not (root / content_hash[:2] / content_hash).exists()


def test_a_successful_write_leaves_its_blob_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``write_blob`` that returns leaves its blob readable at its name, although a
    concurrent writer of the same bytes failed its own write meanwhile (G53)."""
    import hashlib
    from typing import Any

    from particles.corpus import blob_io

    content = b"bytes of one deposit\n"
    digest = hashlib.sha256(content).hexdigest()
    root = tmp_path / "blobs"
    real = os.fdopen
    calls = {"n": 0}

    def failing(fd: int, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        if "w" not in mode:
            return real(fd, *args, **kwargs)
        os.close(fd)
        raise OSError(28, "No space left on device")

    def first(fd: int, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            monkeypatch.setattr(blob_io.os, "fdopen", failing)
            with pytest.raises(OSError):
                blob_io.write_blob(root, digest, content)
            monkeypatch.setattr(blob_io.os, "fdopen", real)
        return real(fd, *args, **kwargs)

    monkeypatch.setattr(blob_io.os, "fdopen", first)
    returned = blob_io.write_blob(root, digest, content)
    monkeypatch.setattr(blob_io.os, "fdopen", real)
    assert returned.exists()
    assert blob_io.read_blob(root, digest) == content
    assert sorted(p.name for p in returned.parent.iterdir()) == [digest]


def test_a_failed_write_leaves_no_name_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that fails part-way leaves neither a partial blob nor its temporary file."""
    import hashlib
    from typing import Any

    from particles.corpus import blob_io

    content = b"bytes\n"
    digest = hashlib.sha256(content).hexdigest()
    root = tmp_path / "blobs"

    def failing(fd: int, *args: Any, **kwargs: Any) -> Any:
        os.close(fd)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(blob_io.os, "fdopen", failing)
    with pytest.raises(OSError):
        blob_io.write_blob(root, digest, content)
    assert list((root / digest[:2]).iterdir()) == []


def test_a_file_at_a_blobs_place_is_refused_without_being_read_whole(tmp_path: Path) -> None:
    """A file at a blob's name that cannot be the blob is refused without buffering it.

    A store directory a repository ships names the blob and supplies the file,
    so the read's memory would be the planter's choice; the hash is taken in
    blocks before any buffering (security review 2026-10-30, AB15).
    """
    import hashlib
    import tracemalloc

    root = tmp_path / "corpus_blobs"
    name = hashlib.sha256(b"the real blob").hexdigest()
    shard = root / name[:2]
    shard.mkdir(parents=True)
    with open(shard / name, "wb") as handle:
        handle.truncate(256 * 1024 * 1024)  # sparse: costs no disk
    tracemalloc.start()
    try:
        with pytest.raises(BlobRefused):
            read_blob(root, name)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024, peak


def test_a_genuine_blob_still_reads_after_the_streamed_hash(tmp_path: Path) -> None:
    """The streamed hash check returns the blob's bytes unchanged when they match."""
    import hashlib

    content = b"x" * (3 * 1024 * 1024 + 17)
    digest = hashlib.sha256(content).hexdigest()
    root = tmp_path / "blobs"
    write_blob(root, digest, content)
    assert read_blob(root, digest) == content
