# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""skills sub-Typer — install the shipped agent-onboarding skill files.

The SDK ships three short, tool-agnostic Markdown files (``particles/skills/``)
that tell an agent how to use the store: which write verb to reach for, how to
read confidence and contestedness, and that ruling on a contradiction is the
operator's job. This group copies them somewhere a harness will read them.

**Namespaced, so removal is surgical.** Files land in a ``particles/``
subdirectory of the target skills directory, and ``--remove`` deletes exactly
that subdirectory — the same marker-owned discipline the settings
merge uses. Nothing outside it is ever touched.

**Never through a link.** A repository can ship ``.claude/skills`` itself, so
with ``--project`` any part of the destination may be a symlink a clone
planted. Install and remove refuse a symlinked owned directory or skill file,
and, below ``root`` (the working directory for a project install), a symlinked
parent; each file is written with ``O_NOFOLLOW`` (security review N27).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer

from particles.api.cli import app
from particles.skills import skill_files

skills_app = typer.Typer(
    help="Install the agent-onboarding skill files shipped with the SDK.",
    no_args_is_help=True,
)
app.add_typer(skills_app, name="skills")

#: The subdirectory this SDK owns inside a harness's skills directory.
OWNED_SUBDIR = "particles"


def default_skills_dir(project: bool = False) -> Path:
    """The Claude Code skills directory — user-level by default, repo-local with ``project``.

    Mirrors the ``init`` scope choice: memory is operator
    infrastructure, so one install should cover every project.
    """
    if project:
        return Path.cwd() / ".claude" / "skills"
    return Path.home() / ".claude" / "skills"


class SkillsPathError(OSError):
    """A skills destination runs through a symlink, so nothing was written or removed."""


def _refuse_symlinks(owned: Path, root: Path | None) -> None:
    """Raise :class:`SkillsPathError` if the owned dir or a parent below ``root`` is a link."""
    chain = [owned]
    if root is not None:
        parent = owned.parent
        while parent != root and root in parent.parents:
            chain.append(parent)
            parent = parent.parent
    for path in chain:
        if path.is_symlink():
            raise SkillsPathError(f"{path} is a symlink; refusing to write or delete through it")


def _write_no_follow(source: Path, destination: Path) -> None:
    """Copy ``source`` onto ``destination`` without following a link planted there."""
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as out:
        out.write(source.read_bytes())


def install_skills(
    target_dir: Path, *, dry_run: bool = False, root: Path | None = None
) -> list[Path]:
    """Copy the shipped skill files into ``target_dir/particles/``; return the paths.

    Overwrites its own files unconditionally — they are SDK-owned content, not
    operator-edited config, so a re-run is a repair/upgrade (the
    idempotence contract). ``dry_run`` computes the paths without writing.
    ``root`` is the directory a clone controls nothing above (the working
    directory for a project install): every directory between it and the owned
    subdirectory must be a real directory. A symlinked owned directory or skill
    file is refused at any scope, with :class:`SkillsPathError`.
    """
    owned = target_dir / OWNED_SUBDIR
    written = []
    if not dry_run:
        _refuse_symlinks(owned, root)
    for source in skill_files():
        destination = owned / source.name
        if not dry_run:
            owned.mkdir(parents=True, exist_ok=True)
            _refuse_symlinks(owned, root)
            _write_no_follow(source, destination)
        written.append(destination)
    return written


def remove_skills(
    target_dir: Path, *, dry_run: bool = False, root: Path | None = None
) -> Path | None:
    """Delete exactly the owned subdirectory; return it, or None if absent.

    Refuses (:class:`SkillsPathError`) when the owned subdirectory, or a parent
    below ``root``, is a symlink, so a link cannot aim the delete at a real
    directory elsewhere.
    """
    owned = target_dir / OWNED_SUBDIR
    if not owned.is_dir() and not owned.is_symlink():
        return None
    _refuse_symlinks(owned, root)
    if not dry_run:
        shutil.rmtree(owned)
    return owned


@skills_app.command("install")
def skills_install_cmd(
    directory: Path | None = typer.Option(
        None,
        "--dir",
        help=(
            "Skills directory to install into. Default: ~/.claude/skills "
            "(or ./.claude/skills with --project). Files land in a 'particles' "
            "subdirectory of it."
        ),
    ),
    project: bool = typer.Option(
        False, "--project", help="Install into ./.claude/skills instead of the user-level dir."
    ),
    remove: bool = typer.Option(
        False, "--remove", help="Delete exactly the Particles-owned skills subdirectory."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be written without writing it."
    ),
) -> None:
    """Install (or remove) the shipped agent-onboarding skill files."""
    target = directory if directory is not None else default_skills_dir(project)
    root = Path.cwd() if directory is None and project else None
    try:
        if remove:
            removed = remove_skills(target, dry_run=dry_run, root=root)
            if removed is None:
                typer.echo(f"Nothing to remove: {target / OWNED_SUBDIR} does not exist.")
                return
            verb = "Would remove" if dry_run else "Removed"
            typer.echo(f"{verb} {removed}")
            return
        written = install_skills(target, dry_run=dry_run, root=root)
    except SkillsPathError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    verb = "Would write" if dry_run else "Installed"
    typer.echo(f"{verb} {len(written)} skill file(s):")
    for path in written:
        typer.echo(f"  {path}")


@skills_app.command("list")
def skills_list_cmd() -> None:
    """List the skill files this SDK ships, with their first heading."""
    for path in skill_files():
        heading = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                heading = line[2:].strip()
                break
        typer.echo(f"{path.name:<26} {heading}")
