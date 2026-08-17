"""Tests for the shipped agent-onboarding skill files and `particles skills`.

The install is a file copy, so the assertions worth making are about the
*discipline* around it: that the owned subdirectory is the only thing written,
that removal is surgical, and that a re-run repairs rather than duplicates.
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli.skills import (
    OWNED_SUBDIR,
    default_skills_dir,
    install_skills,
    remove_skills,
)
from particles.skills import SKILL_FILENAMES, skill_files


def _clean(output: str) -> str:
    """Strip the ANSI escapes Typer/Rich injects around option tokens in CI."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


class TestSkillPackage:
    def test_ships_three_named_files(self) -> None:
        assert len(SKILL_FILENAMES) == 3
        assert [p.name for p in skill_files()] == list(SKILL_FILENAMES)


class TestInstallSkills:
    def test_writes_into_an_owned_subdirectory(self, tmp_path: Path) -> None:
        written = install_skills(tmp_path)
        assert {p.name for p in written} == set(SKILL_FILENAMES)
        assert all(p.parent == tmp_path / OWNED_SUBDIR for p in written)
        # The only thing created directly under the target is the owned subdir —
        # nothing else in a real ~/.claude/skills is touched.
        assert [p.name for p in tmp_path.iterdir()] == [OWNED_SUBDIR]

    def test_creates_missing_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "skills"
        install_skills(target)
        assert (target / OWNED_SUBDIR / SKILL_FILENAMES[0]).is_file()

    def test_rerun_repairs_rather_than_duplicating(self, tmp_path: Path) -> None:
        install_skills(tmp_path)
        damaged = tmp_path / OWNED_SUBDIR / SKILL_FILENAMES[0]
        damaged.write_text("clobbered", encoding="utf-8")
        install_skills(tmp_path)
        assert damaged.read_text(encoding="utf-8") != "clobbered"
        assert len(list((tmp_path / OWNED_SUBDIR).iterdir())) == len(SKILL_FILENAMES)

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        planned = install_skills(tmp_path, dry_run=True)
        assert len(planned) == len(SKILL_FILENAMES)
        assert not (tmp_path / OWNED_SUBDIR).exists()


class TestRemoveSkills:
    def test_removes_only_the_owned_subdirectory(self, tmp_path: Path) -> None:
        install_skills(tmp_path)
        bystander = tmp_path / "someone-elses-skill.md"
        bystander.write_text("not ours", encoding="utf-8")

        removed = remove_skills(tmp_path)

        assert removed == tmp_path / OWNED_SUBDIR
        assert not (tmp_path / OWNED_SUBDIR).exists()
        assert bystander.read_text(encoding="utf-8") == "not ours"

    def test_absent_is_not_an_error(self, tmp_path: Path) -> None:
        assert remove_skills(tmp_path) is None

    def test_dry_run_removes_nothing(self, tmp_path: Path) -> None:
        install_skills(tmp_path)
        assert remove_skills(tmp_path, dry_run=True) == tmp_path / OWNED_SUBDIR
        assert (tmp_path / OWNED_SUBDIR).is_dir()


class TestDefaultSkillsDir:
    def test_project_scope_is_repo_local(self) -> None:
        assert default_skills_dir(project=True) == Path.cwd() / ".claude" / "skills"

    def test_user_scope_is_home(self) -> None:
        assert default_skills_dir() == Path.home() / ".claude" / "skills"


class TestSkillsCli:
    def test_list_names_every_shipped_file(self) -> None:
        result = CliRunner().invoke(app, ["skills", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        for name in SKILL_FILENAMES:
            assert name in result.output

    def test_install_reports_what_it_wrote(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app, ["skills", "install", "--dir", str(tmp_path)], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Installed 3 skill file(s)" in _clean(result.output)
        assert (tmp_path / OWNED_SUBDIR / SKILL_FILENAMES[0]).is_file()

    def test_dry_run_reports_without_writing(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            ["skills", "install", "--dir", str(tmp_path), "--dry-run"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Would write" in _clean(result.output)
        assert not (tmp_path / OWNED_SUBDIR).exists()

    def test_remove_on_a_clean_dir_says_so(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            ["skills", "install", "--dir", str(tmp_path), "--remove"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Nothing to remove" in _clean(result.output)
