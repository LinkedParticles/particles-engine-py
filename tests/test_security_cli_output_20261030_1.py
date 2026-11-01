# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""CLI verbs print stored text with no live terminal control and no forged line.

Regression tests for the tenth delta security review (2026-10-30-1, AB37, AB38,
AB70), ported from the audit's proofs. A stored claim, URI, depositor, event
actor or reason, or subject description reaches the operator's terminal with
no live control character, and a stored line break cannot start a line that
reads as one of the verb's own labelled fields. No model call; scratch file
store from the ``cli_db`` fixture.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

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


async def _seed_event(actor: str, reason: str) -> str:
    from particles.db import session_scope
    from particles.store.event_store import EventRefKind, OperatorEventType, record_event

    async with session_scope() as session:
        ev = await record_event(
            session,
            actor=actor,
            event_type=OperatorEventType.PARTICLE_RETRACTED,
            reason=reason,
            refs=[(EventRefKind.PARTICLE, "00000000-0000-4000-8000-000000000000")],
        )
        await session.commit()
    return ev.event_id


async def _seed_subject(name: str, description: str) -> str:
    from particles.core.schema import Subject
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    s = Subject(canonical_name=name, description=description, asserted_by="agent:x")
    async with session_scope() as session:
        await insert_subject(session, s)
        await session.commit()
    return s.id


async def _seed_entry_with_claim(content: str, depositor: str) -> str:
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.status import Status
    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    entry_id = str(uuid.uuid4())
    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.8),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id="s1")
        ],
        asserted_by="agent:x",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
    )
    async with session_scope() as session:
        session.add(
            CorpusEntryRow(
                entry_id=entry_id,
                uri_r=f"https://example.invalid/{CONTROLLED}",
                source_type="WEB_PAGE",
                mutability="MUTABLE",
                fetch_policy="LAZY",
                created_at=datetime.now(UTC),
                deposited_by=depositor,
            )
        )
        await session.flush()
        await insert_particle(session, p)
        await session.commit()
    return entry_id


def test_events_list_prints_a_stored_actor_and_reason_with_no_live_control(cli_db: Path) -> None:
    """`events list` prints an agent-chosen actor and retraction reason with no live control."""
    asyncio.run(_seed_event(f"agent {CONTROLLED}", f"why {CONTROLLED}"))
    result = CliRunner().invoke(app, ["events", "list"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)


def test_events_show_prints_a_stored_actor_and_reason_with_no_live_control(cli_db: Path) -> None:
    """`events show` prints an agent-chosen actor and retraction reason with no live control."""
    eid = asyncio.run(_seed_event(f"agent {CONTROLLED}", f"why {CONTROLLED}"))
    result = CliRunner().invoke(app, ["events", "show", eid], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)


def test_events_list_keeps_a_stored_reason_on_its_own_line(cli_db: Path) -> None:
    """A stored reason holding a newline cannot forge a `refs:` line of `events list`."""
    asyncio.run(_seed_event("agent:x", "why\n    refs:   particle:deadbeef…"))
    result = CliRunner().invoke(app, ["events", "list"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    refs_lines = [ln for ln in result.stdout.splitlines() if ln.lstrip().startswith("refs:")]
    assert len(refs_lines) == 1, result.stdout


def test_corpus_show_prints_a_stored_claim_with_no_live_control(cli_db: Path) -> None:
    """`corpus show` prints a stored claim, URI and depositor with no live control character."""
    eid = asyncio.run(_seed_entry_with_claim(f"Claim {CONTROLLED}", f"agent {CONTROLLED}"))
    result = CliRunner().invoke(app, ["corpus", "show", eid], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)


def test_corpus_list_prints_a_stored_uri_with_no_live_control(cli_db: Path) -> None:
    """`corpus list` prints a stored URI with no live control character."""
    asyncio.run(_seed_entry_with_claim("Claim", "agent:x"))
    result = CliRunner().invoke(app, ["corpus", "list"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert not _has_live_control(result.stdout), repr(result.stdout)


def test_corpus_show_keeps_a_stored_depositor_on_its_own_line(cli_db: Path) -> None:
    """A depositor label holding a newline cannot forge a `Snapshots:` line of `corpus show`."""
    eid = asyncio.run(_seed_entry_with_claim("Claim", "agent\nSnapshots:"))
    result = CliRunner().invoke(app, ["corpus", "show", eid], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    labelled = [ln for ln in result.stdout.splitlines() if ln.startswith("Snapshots:")]
    assert len(labelled) == 1, result.stdout


def test_subjects_show_keeps_a_stored_description_on_its_own_line(cli_db: Path) -> None:
    """`subjects show` prints a stored description holding a newline without forging a field."""
    sid = asyncio.run(
        _seed_subject("Plain", "real\nExternal:    wikidata:Q42  https://forged.invalid")
    )
    result = CliRunner().invoke(app, ["subjects", "show", sid], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    labelled = [ln for ln in result.stdout.splitlines() if ln.startswith("External:")]
    assert labelled == [], result.stdout


def test_terminal_field_keeps_a_value_on_one_line() -> None:
    """A labelled field shows a stored line break and every control as an escape."""
    from particles.api.cli._terminal import terminal_field

    out = terminal_field("a\nb\r\x1b[2J‮c\td")
    assert "\n" not in out and "\r" not in out and not _has_live_control(out)
    assert out == "a\\nb\\x0d\\x1b[2J\\u202ec\td"


def test_a_progress_line_writes_no_live_control(capsys: object) -> None:
    """A progress message naming a stored URI reaches stderr with no live control."""
    from particles.api.cli._progress import progress_line

    progress_line(f"fetching https://example.invalid/{CONTROLLED}")
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert CONTROLLED not in err
    assert "\\x1b]0;owned" in err


def test_a_source_passage_header_keeps_each_field_on_its_own_line() -> None:
    """A stored URI or note holding a newline cannot forge a `Match:` line of a passage header."""
    from types import SimpleNamespace

    from particles.api.cli._source_passage import header_lines

    passage = SimpleNamespace(
        uri_r="https://example.invalid/x\nMatch:     exact",
        source_type="WEB_PAGE",
        corpus_entry_id=None,
        snapshot_id=None,
        match=SimpleNamespace(value="WHOLE"),
        locate_overlap=None,
        truncated=False,
        text="",
        note=f"n\nMatch: forged {CONTROLLED}",
    )
    lines = "\n".join(header_lines(passage)).splitlines()  # type: ignore[arg-type]
    assert sum(ln.startswith("Match:") for ln in lines) == 1, lines
    assert not any(_has_live_control(ln) for ln in lines)
