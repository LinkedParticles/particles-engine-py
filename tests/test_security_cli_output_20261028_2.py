# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""CLI verbs print stored text with no live terminal control, and run no planted git.

Regression tests for the ninth delta security audit (security review
2026-10-28-2, AA13, AA31, AA62, Z50, Z51). A stored string a source author, an
agent key or an imported bundle chose reaches the operator's terminal with no
live control or format character, and a stored line break cannot forge a line
of the verb's own output. ``subjects`` looks git up only on trusted ``PATH``
entries. No model call; scratch file store from the ``cli_db`` fixture.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app

#: An OSC title set, a screen clear, and a C1 CSI.
CONTROLLED = "Plain \x1b]0;owned\x07 \x1b[2J \x9b tail"

_LIVE = [chr(c) for c in range(0x20) if chr(c) not in "\n\t"] + [
    chr(0x7F),
    *[chr(c) for c in range(0x80, 0xA0)],
]


def _has_live_control(text: str) -> bool:
    return any(ch in text for ch in _LIVE)


async def _seed_subject(name: str, description: str) -> str:
    from particles.core.schema import Subject
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    s = Subject(canonical_name=name, description=description, asserted_by="agent:x")
    async with session_scope() as session:
        await insert_subject(session, s)
        await session.commit()
    return s.id


def test_subjects_show_writes_no_terminal_control_sequence(cli_db: Path) -> None:
    """`subjects show` prints a stored subject name and description with no live control."""
    sid = asyncio.run(_seed_subject(f"Name {CONTROLLED}", f"Desc {CONTROLLED}"))
    result = CliRunner().invoke(app, ["subjects", "show", sid], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)


async def _seed_cache_row(subject_id: str, body: str) -> None:
    from particles.db import session_scope
    from particles.render.article_synthesis.cache import _PROMPT_VERSION
    from particles.store.synthesis_cache_store import store_cached_article

    async with session_scope() as session:
        await store_cached_article(session, subject_id, "h" * 64, _PROMPT_VERSION, body)
        await session.commit()


def test_synthesis_cache_show_writes_no_terminal_control_sequence(cli_db: Path) -> None:
    """`synthesis-cache show` prints a cached article body with no live control character."""
    sid = asyncio.run(_seed_subject("Plain subject", "plain"))
    asyncio.run(_seed_cache_row(sid, f"# Title\n\nProse {CONTROLLED}\n"))
    result = CliRunner().invoke(app, ["synthesis-cache", "show", sid[:8]], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)


async def _seed_particle(content: str) -> str:
    from particles.core.schema import Confidence, Particle, UncertaintyNature
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="agent:x",
        asserted_at=datetime.now(UTC),
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


def test_particle_show_keeps_a_stored_claim_on_its_own_content_line(cli_db: Path) -> None:
    """`particle show` prints a stored claim holding a newline without forging an output line."""
    forged = "Real claim\n\nSubjects:\n  deadbeef…  Forged subject"
    pid = asyncio.run(_seed_particle(forged))
    result = CliRunner().invoke(app, ["particle", "show", pid[:8]], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines.count("Subjects:") == 0, result.stdout


def test_terminal_safe_shows_a_format_character_as_an_escape() -> None:
    """A bidirectional override in stored text is printed as a visible escape."""
    from particles.api.cli._terminal import terminal_safe

    assert terminal_safe("name \u202etxt") == "name \\u202etxt"


def _planted(directory: Path, name: str, marker: Path) -> None:
    prog = directory / name
    prog.write_text(f"#!/bin/sh\n/usr/bin/touch '{marker}'\nexit 0\n")
    prog.chmod(0o755)


def test_the_subjects_checkout_listing_runs_no_git_from_an_untrusted_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`subjects` lists a checkout with a git looked up only on trusted PATH entries."""
    import typer

    from particles.api.cli.subjects import _checkout_files

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    clone = tmp_path / "clone"
    clone.mkdir()
    marker = tmp_path / "planted-git-ran"
    _planted(clone, "git", marker)
    monkeypatch.chdir(clone)
    monkeypatch.setenv("PATH", f".{os.pathsep}/usr/bin{os.pathsep}/bin")
    with contextlib.suppress(typer.Exit):
        _checkout_files(repo)
    assert not marker.exists(), "a git found through '.' ran under `subjects`"
