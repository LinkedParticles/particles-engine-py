# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A projection manifest's own paths stay inside its tree (security review 2026-10-24-2, P3).

A manifest a repository ships is data: neither its ``output:`` field nor a
mechanical block's path may reach a file outside the repository that holds it,
and nothing the projection writes goes through a symlink. Every CLI run uses a
scratch store and a temporary HOME.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.operations.projection.paths import (
    ManifestPathError,
    containment_root,
    read_block_text,
    resolve_manifest_path,
    write_text_no_follow,
)


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_db: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(repo)
    return repo, outside


def _project(*args: str) -> str:
    result = CliRunner().invoke(app, ["project", *args, "--without-synthesis"])
    return result.output


# ---------------------------------------------------------------------------
# The audit's proofs, end to end through the CLI
# ---------------------------------------------------------------------------


def test_manifest_output_outside_its_tree_is_refused(layout: tuple[Path, Path]) -> None:
    """A manifest's own ``output:`` never makes ``particles project`` write outside its tree."""
    repo, outside = layout
    victim = outside / "victim.md"
    victim.write_text("owner content\n", encoding="utf-8")
    (repo / "frag.md").write_text("fragment text\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text(
        f"name: doc\noutput: {victim}\nsections:\n  - block: frag.md\n", encoding="utf-8"
    )
    output = _project(str(manifest))
    assert "is absolute" in output
    assert victim.read_text(encoding="utf-8") == "owner content\n"


def test_manifest_block_outside_its_tree_is_not_read(layout: tuple[Path, Path]) -> None:
    """A mechanical block path outside the manifest's tree is never spliced into the output."""
    repo, outside = layout
    private = outside / "private.txt"
    private.write_text("PRIVATE-HOST-FILE-MARKER\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text(
        f"name: doc\noutput: out.md\nsections:\n  - block: {private}\n", encoding="utf-8"
    )
    _project(str(manifest))
    for written in repo.rglob("*"):
        if written.is_file() and written != manifest:
            assert "PRIVATE-HOST-FILE-MARKER" not in written.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The rest of the rule, through the CLI
# ---------------------------------------------------------------------------


def test_a_block_inside_the_tree_renders(layout: tuple[Path, Path]) -> None:
    repo, _outside = layout
    (repo / "frag.md").write_text("fragment text\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\noutput: out.md\nsections:\n  - block: frag.md\n")
    _project(str(manifest))
    assert "fragment text" in (repo / "out.md").read_text(encoding="utf-8")


def test_a_dotdot_output_is_refused(layout: tuple[Path, Path]) -> None:
    repo, outside = layout
    (repo / "frag.md").write_text("x\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\noutput: ../outside/out.md\nsections:\n  - block: frag.md\n")
    assert "contains '..'" in _project(str(manifest))
    assert not (outside / "out.md").exists()


def test_a_symlinked_block_leading_out_of_the_tree_is_refused(layout: tuple[Path, Path]) -> None:
    repo, outside = layout
    private = outside / "private.txt"
    private.write_text("PRIVATE-HOST-FILE-MARKER\n", encoding="utf-8")
    (repo / "frag.md").symlink_to(private)
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\noutput: out.md\nsections:\n  - block: frag.md\n")
    assert "Refused" in _project(str(manifest))
    assert not (repo / "out.md").exists()


def test_a_symlinked_output_is_never_written_through(layout: tuple[Path, Path]) -> None:
    """Even an OUTPUT the operator names is not written through a committed symlink."""
    repo, outside = layout
    victim = outside / "CLAUDE.md"
    victim.write_text("owner content\n", encoding="utf-8")
    (repo / "README.md").symlink_to(victim)
    (repo / "frag.md").write_text("x\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\nsections:\n  - block: frag.md\n")
    assert "symlink" in _project(str(manifest), "README.md")
    assert victim.read_text(encoding="utf-8") == "owner content\n"


def test_a_symlinked_snapshot_is_never_written_through(layout: tuple[Path, Path]) -> None:
    repo, outside = layout
    victim = outside / "victim.md"
    victim.write_text("owner content\n", encoding="utf-8")
    (repo / "doc.snapshot.md").symlink_to(victim)
    (repo / "frag.md").write_text("x\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\noutput: out.md\nsections:\n  - block: frag.md\n")
    assert "symlink" in _project(str(manifest))
    assert victim.read_text(encoding="utf-8") == "owner content\n"


def test_an_operator_output_outside_the_tree_is_honoured(layout: tuple[Path, Path]) -> None:
    """OUTPUT on the command line is the operator's choice, not the manifest's."""
    repo, outside = layout
    (repo / "frag.md").write_text("fragment text\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\nsections:\n  - block: frag.md\n")
    _project(str(manifest), str(outside / "out.md"))
    assert "fragment text" in (outside / "out.md").read_text(encoding="utf-8")


def test_allow_outside_paths_lifts_the_containment(layout: tuple[Path, Path]) -> None:
    repo, outside = layout
    (outside / "frag.md").write_text("shared fragment\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text(
        f"name: doc\noutput: {outside / 'out.md'}\nsections:\n  - block: ../outside/frag.md\n"
    )
    _project(str(manifest), "--allow-outside-paths")
    assert "shared fragment" in (outside / "out.md").read_text(encoding="utf-8")


def test_check_refuses_a_block_outside_the_tree(layout: tuple[Path, Path]) -> None:
    repo, outside = layout
    (outside / "private.txt").write_text("PRIVATE\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text("name: doc\nsections:\n  - block: ../outside/private.txt\n")
    result = CliRunner().invoke(app, ["project", str(manifest), "--check"])
    assert result.exit_code == 2
    assert "Refused" in result.output


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


class TestContainmentRoot:
    def test_the_repository_holding_the_manifest(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "docs" / "projection").mkdir(parents=True)
        assert containment_root(repo / "docs" / "projection") == repo.resolve()

    def test_the_manifest_directory_when_no_repository(self, tmp_path: Path) -> None:
        loose = tmp_path / "unpacked"
        loose.mkdir()
        assert containment_root(loose) == loose.resolve()

    def test_a_repository_at_home_is_never_the_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dotfiles repository at ``~`` does not widen the root to all of home."""
        home = tmp_path / "home"
        (home / ".git").mkdir(parents=True)
        unpacked = home / "Downloads" / "evil"
        unpacked.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        assert containment_root(unpacked) == unpacked.resolve()


class TestResolveManifestPath:
    @pytest.mark.parametrize("raw", ["/etc/passwd", "~/.claude/CLAUDE.md", "a/../../b"])
    def test_refused_lexically(self, tmp_path: Path, raw: str) -> None:
        with pytest.raises(ManifestPathError):
            resolve_manifest_path(
                raw, against=tmp_path, root=tmp_path, allow_outside=False, what="output"
            )

    def test_a_symlinked_directory_leading_out_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "elsewhere").mkdir()
        (root / "out").symlink_to(tmp_path / "elsewhere")
        with pytest.raises(ManifestPathError, match="outside"):
            resolve_manifest_path(
                "out/doc.md", against=root, root=root, allow_outside=False, what="output"
            )

    def test_inside_the_root_is_joined(self, tmp_path: Path) -> None:
        got = resolve_manifest_path(
            "docs/x.md", against=tmp_path, root=tmp_path, allow_outside=False, what="output"
        )
        assert got == tmp_path / "docs" / "x.md"


def test_read_block_text_refuses_a_fifo(tmp_path: Path) -> None:
    import os

    from particles.corpus.local_read import LocalReadRefused

    os.mkfifo(tmp_path / "pipe.md")
    with pytest.raises(LocalReadRefused):
        read_block_text("pipe.md", base_dir=tmp_path, allow_outside=False)


def test_write_text_no_follow_refuses_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("keep\n", encoding="utf-8")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(OSError, match="symlink"):
        write_text_no_follow(tmp_path / "link", "overwritten\n")
    assert target.read_text(encoding="utf-8") == "keep\n"


# ---------------------------------------------------------------------------
# Security review 2026-10-25-1: Q29 (a swapped parent), Q51 (refusal text),
# Q52 (a manifest in the home directory)
# ---------------------------------------------------------------------------


def test_an_output_directory_swapped_after_the_check_is_not_written_through(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document goes where the containment check judged, not where its parent
    directory points by the time the render finishes (Q29, the audit's proof)."""
    import particles.operations.projection as projection

    repo, outside = layout
    (repo / "out").mkdir()
    (repo / "frag.md").write_text("fragment text\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text(
        "name: doc\noutput: out/doc.md\nsections:\n  - block: frag.md\n", encoding="utf-8"
    )
    original = projection.project_document

    async def render_then_swap(*args: object, **kwargs: object) -> object:
        result = await original(*args, **kwargs)  # type: ignore[arg-type]
        out_dir = repo / "out"
        if out_dir.is_dir() and not out_dir.is_symlink():
            out_dir.rmdir()
            out_dir.symlink_to(outside)
        return result

    monkeypatch.setattr(projection, "project_document", render_then_swap)
    result = CliRunner().invoke(app, ["project", str(manifest), "--without-synthesis"])
    assert result.exit_code == 2
    assert not (outside / "doc.md").exists()


def test_a_manifest_output_in_a_new_directory_is_written(layout: tuple[Path, Path]) -> None:
    """The contained write creates the missing directories on the way down."""
    repo, _outside = layout
    (repo / "frag.md").write_text("fragment text\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    manifest.write_text(
        "name: doc\noutput: a/b/doc.md\nsections:\n  - block: frag.md\n", encoding="utf-8"
    )
    _project(str(manifest))
    assert "fragment text" in (repo / "a" / "b" / "doc.md").read_text(encoding="utf-8")


def test_a_refusal_does_not_relay_manifest_text_or_offer_the_bypass(
    layout: tuple[Path, Path],
) -> None:
    """A containment refusal names the refused field only: it neither echoes the
    manifest's own text nor names the flag that lifts the check (Q51)."""
    repo, _outside = layout
    marker = "MANIFEST-AUTHORED-TEXT"
    (repo / "frag.md").write_text("fragment text\n", encoding="utf-8")
    manifest = repo / "doc.yaml"
    for output in (f"/{marker}/out.md", f"../{marker}/out.md"):
        manifest.write_text(
            f"name: doc\noutput: {output}\nsections:\n  - block: frag.md\n", encoding="utf-8"
        )
        result = CliRunner().invoke(app, ["project", str(manifest), "--without-synthesis"])
        assert result.exit_code == 2
        assert "output" in result.output
        assert marker not in result.output
        assert "--allow-outside-paths" not in result.output


def test_a_resolved_refusal_does_not_name_the_resolved_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "MANIFEST-AUTHORED-TEXT").mkdir()
    (root / "out").symlink_to(tmp_path / "MANIFEST-AUTHORED-TEXT")
    with pytest.raises(ManifestPathError) as excinfo:
        resolve_manifest_path(
            "out/doc.md", against=root, root=root, allow_outside=False, what="output"
        )
    assert "MANIFEST-AUTHORED-TEXT" not in str(excinfo.value)
    assert "--allow-outside-paths" not in str(excinfo.value)


def test_a_manifest_in_the_home_directory_does_not_get_home_as_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The home directory is never a containment root, even for a manifest kept in it (Q52)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    assert containment_root(home) is None
    assert containment_root(tmp_path) is None
    with pytest.raises(ManifestPathError):
        resolve_manifest_path(
            "doc.md", against=home, root=containment_root(home), allow_outside=False, what="output"
        )
    (home / "frag.md").write_text("fragment\n", encoding="utf-8")
    with pytest.raises(ManifestPathError):
        read_block_text("frag.md", base_dir=home, allow_outside=False)
    assert read_block_text("frag.md", base_dir=home, allow_outside=True) == "fragment\n"


def test_a_manifest_in_home_still_writes_its_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_db: Path
) -> None:
    """The snapshot is named by the projection, not the manifest, so it is written beside it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)
    manifest = home / "doc.yaml"
    manifest.write_text(
        "name: doc\nsections:\n  - title: T\n    tags: [none-such]\n", encoding="utf-8"
    )
    result = CliRunner().invoke(
        app, ["project", str(manifest), str(tmp_path / "out.md"), "--without-synthesis"]
    )
    assert result.exit_code == 0, result.output
    assert (home / "doc.snapshot.md").exists()


class TestContainedWrite:
    def test_writes_inside_the_root(self, tmp_path: Path) -> None:
        write_text_no_follow(tmp_path / "d" / "x.md", "body\n", root=tmp_path)
        assert (tmp_path / "d" / "x.md").read_text(encoding="utf-8") == "body\n"

    def test_a_symlinked_parent_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (root / "d").symlink_to(elsewhere)
        with pytest.raises(OSError, match="symlink"):
            write_text_no_follow(root / "d" / "x.md", "body\n", root=root)
        assert not (elsewhere / "x.md").exists()

    def test_a_symlinked_parent_inside_the_root_is_refused_on_the_walk(
        self, tmp_path: Path
    ) -> None:
        # A link that stays inside the root resolves inside, so the walk follows
        # the real components and never the link itself.
        root = tmp_path / "root"
        (root / "real").mkdir(parents=True)
        (root / "alias").symlink_to(root / "real")
        write_text_no_follow(root / "alias" / "x.md", "body\n", root=root)
        assert (root / "real" / "x.md").read_text(encoding="utf-8") == "body\n"

    def test_a_symlinked_final_component_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.write_text("keep\n", encoding="utf-8")
        (tmp_path / "root").mkdir()
        (tmp_path / "root" / "link").symlink_to(target)
        with pytest.raises(OSError, match="symlink"):
            write_text_no_follow(tmp_path / "root" / "link", "x\n", root=tmp_path / "root")
        assert target.read_text(encoding="utf-8") == "keep\n"
