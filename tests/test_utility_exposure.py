# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""what a session was shown, read as of its start, and the re-mine.

Acceptance unit tests 1, 2, 6, 7 and 8 of the record, over a file-backed store
so the entry points that open their own ``session_scope`` see the same rows.
Tests 3, 4 and 5 (the ruling and its evidence) live in
``tests/test_utility_mining.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.operations.utility_exposure import (
    ExposureReader,
    RuleFile,
    read_file_paths,
    rule_files_in_view,
    session_frame,
)
from particles.store.particle_store import insert_particle
from particles.store.session_exposure_store import (
    SessionExposure,
    ShownBelief,
    record_session_exposure,
)
from particles.store.utility_store import SOURCE_EXPLICIT, UtilityEventRow

ROOT = "/repo"
KEY = "-repo"
START = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# store fixtures
# ---------------------------------------------------------------------------


async def _entry(
    session: AsyncSession,
    entry_id: str,
    uri: str,
    tags: list[str],
    *,
    mutability: Mutability = Mutability.MUTABLE,
    source_type: str = "LOCAL_MARKDOWN",
) -> None:
    session.add(
        CorpusEntryRow(
            entry_id=entry_id,
            uri_r=uri,
            source_type=source_type,
            mutability=mutability.value,
            fetch_policy=FetchPolicy.NEVER.value,
            created_at=START - timedelta(days=30),
            deposited_by="test",
            tags_json=json.dumps(tags),
        )
    )


async def _snapshot(
    session: AsyncSession, snapshot_id: str, entry_id: str, captured_at: datetime
) -> None:
    session.add(
        SnapshotRow(
            snapshot_id=snapshot_id,
            entry_id=entry_id,
            captured_at=captured_at,
            content_hash=f"hash-{snapshot_id}",
            warc_record_type=WarcRecordType.RESPONSE.value,
            archive_path="a",
            extraction_status=ExtractionStatus.COMPLETE.value,
        )
    )


async def _belief(
    session: AsyncSession,
    pid: str,
    content: str,
    entry_id: str,
    snapshot_id: str,
    *,
    asserted_at: datetime = START - timedelta(days=20),
) -> Particle:
    particle = Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=asserted_at,
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
            )
        ],
    )
    await insert_particle(session, particle)
    return particle


async def _rule_file(
    session: AsyncSession, entry_id: str, path: str, *, key: str | None = KEY
) -> None:
    tags = ["rule-file", *([f"project:{key}"] if key else [])]
    await _entry(session, entry_id, f"file://{path}", tags)
    await _snapshot(session, f"{entry_id}-s1", entry_id, START - timedelta(days=25))


async def _conversation(session: AsyncSession, session_id: str) -> None:
    entry_id = f"conv-{session_id}"
    await _entry(
        session,
        entry_id,
        f"claude-code://session/{session_id}",
        ["claude-code", f"session:{session_id}", f"project:{KEY}"],
        mutability=Mutability.APPEND_ONLY,
        source_type="CONVERSATION",
    )
    await _snapshot(session, f"{entry_id}-s1", entry_id, START + timedelta(hours=1))


async def _shown(session: AsyncSession, session_id: str, action_lines: list[str]) -> set[str]:
    frame = await session_frame(
        session, session_id, entry_tags=[f"project:{KEY}"], first_captured_at=START
    )
    assert frame is not None
    reader = await ExposureReader.load(session)
    return set((await reader.exposure(session, frame, action_lines)).shown)


# ---------------------------------------------------------------------------
# pure path rules
# ---------------------------------------------------------------------------


class TestPathRules:
    def test_read_paths_skip_truncated_and_relative(self) -> None:
        lines = [
            "[tool: Read — /home/u/.claude/projects/-repo/memory/signoff.md]",
            "[tool: Read — /a/very/long/path/that/was/cut…]",
            "[tool: Read — relative/file.md]",
            "[tool: Bash — cat /etc/hosts]",
            "[tool: Read — /home/u/.claude/projects/-repo/memory/signoff.md]",
        ]
        assert read_file_paths(lines) == ["/home/u/.claude/projects/-repo/memory/signoff.md"]

    def test_root_rule_files_are_always_in_view(self) -> None:
        files = [
            RuleFile("root-agents", f"{ROOT}/AGENTS.md"),
            RuleFile("root-claude", f"{ROOT}/CLAUDE.md"),
            RuleFile("cli", f"{ROOT}/particles/api/cli/AGENTS.md"),
        ]
        assert rule_files_in_view(files, []) == ["root-agents", "root-claude"]

    def test_a_subdirectory_file_needs_a_path_under_its_directory(self) -> None:
        # Unit test 6.
        files = [
            RuleFile("root", f"{ROOT}/AGENTS.md"),
            RuleFile("cli", f"{ROOT}/particles/api/cli/AGENTS.md"),
            RuleFile("tests", f"{ROOT}/tests/AGENTS.md"),
        ]
        absolute = [f"[tool: Edit — {ROOT}/particles/api/cli/hook.py]"]
        relative = ["[tool: Bash — uv run pytest tests/test_x.py]"]
        worktree = ["[tool: Read — /repo/.claude/worktrees/w/particles/api/cli/memory.py]"]
        sibling = [f"[tool: Edit — {ROOT}/particles/api/app.py]"]
        assert rule_files_in_view(files, absolute) == ["root", "cli"]
        assert rule_files_in_view(files, relative) == ["root", "tests"]
        assert rule_files_in_view(files, worktree) == ["root", "cli"]
        assert rule_files_in_view(files, sibling) == ["root"]


# ---------------------------------------------------------------------------
# the three sets, as of the session's start
# ---------------------------------------------------------------------------


class TestExposure:
    @pytest.mark.asyncio
    async def test_no_recorded_row_means_rule_and_read_files_only(
        self, file_db_session: AsyncSession
    ) -> None:
        """Unit test 1: with no recorded exposure row, sets 2 and 3 are the whole exposure."""
        s = file_db_session
        await _rule_file(s, "agents", f"{ROOT}/AGENTS.md")
        topic = "/home/u/.claude/projects/-repo/memory/signoff.md"
        await _entry(s, "topic", f"file://{topic}", ["claude-code", f"project:{KEY}"])
        await _snapshot(s, "topic-s1", "topic", START - timedelta(days=25))
        await _entry(s, "conv-old", "claude-code://session/old", ["claude-code"])
        await _snapshot(s, "conv-old-s1", "conv-old", START - timedelta(days=25))
        await _belief(s, "p-rule", "Commit with `git commit -s`", "agents", "agents-s1")
        await _belief(s, "p-topic", "Sign every commit", "topic", "topic-s1")
        await _belief(s, "b-talk", "We discussed `git commit -s`", "conv-old", "conv-old-s1")
        await s.commit()

        assert await _shown(s, "fresh", []) == {"p-rule"}
        assert await _shown(s, "fresh", [f"[tool: Read — {topic}]"]) == {"p-rule", "p-topic"}

        # A recorded row adds set 1.
        await record_session_exposure(
            s,
            SessionExposure(
                session_id="recorded",
                recorded_at=START,
                source="startup",
                action="full",
                project_key=KEY,
                observer_scope_applied=False,
                beliefs=[ShownBelief(particle_id="b-talk", shown_as="digest")],
            ),
        )
        await s.commit()
        assert await _shown(s, "recorded", []) == {"p-rule", "b-talk"}

    @pytest.mark.asyncio
    async def test_a_subdirectory_rule_file_counts_only_when_worked_under(
        self, file_db_session: AsyncSession
    ) -> None:
        """Unit test 6, through the store."""
        s = file_db_session
        await _rule_file(s, "agents", f"{ROOT}/AGENTS.md")
        await _rule_file(s, "cli", f"{ROOT}/particles/api/cli/AGENTS.md")
        await _belief(s, "p-root", "Commit with `git commit -s`", "agents", "agents-s1")
        await _belief(s, "p-cli", "One file per verb", "cli", "cli-s1")
        await s.commit()

        assert await _shown(s, "x", ["[tool: Bash — git status]"]) == {"p-root"}
        under = [f"[tool: Edit — {ROOT}/particles/api/cli/hook.py]"]
        assert await _shown(s, "x", under) == {"p-root", "p-cli"}

    @pytest.mark.asyncio
    async def test_exposure_is_read_as_of_the_session_start(
        self, file_db_session: AsyncSession
    ) -> None:
        """Unit test 7: a later belief, or one only a later snapshot states, is not shown."""
        s = file_db_session
        await _rule_file(s, "agents", f"{ROOT}/AGENTS.md")
        # A September generation of the same MUTABLE rule file.
        await _snapshot(s, "agents-s2", "agents", START + timedelta(days=60))
        await _belief(s, "p-july", "Commit with `git commit -s`", "agents", "agents-s1")
        await _belief(
            s,
            "p-september",
            "Run `uv run pregen` first",
            "agents",
            "agents-s2",
            asserted_at=START + timedelta(days=60),
        )
        await _belief(
            s,
            "p-late",
            "Asserted after the session began",
            "agents",
            "agents-s1",
            asserted_at=START + timedelta(hours=2),
        )
        await s.commit()

        assert await _shown(s, "july", []) == {"p-july"}

    @pytest.mark.asyncio
    async def test_an_unshown_belief_is_not_a_candidate(
        self, file_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unit test 2: a token hit on a belief the session was not shown nominates nothing."""
        import particles.operations.utility_mining as um

        s = file_db_session
        await _rule_file(s, "agents", f"{ROOT}/AGENTS.md")
        await _entry(s, "conv-old", "claude-code://session/old", ["claude-code"])
        await _snapshot(s, "conv-old-s1", "conv-old", START - timedelta(days=25))
        await _belief(s, "p-rule", "Commit with `git commit -s`", "agents", "agents-s1")
        await _belief(s, "p-mention", "It ran `git commit -s` 8 times", "conv-old", "conv-old-s1")
        await _conversation(s, "now")
        await s.commit()

        prompts: list[str] = []

        async def judge(prompt: str, **_k: Any) -> str:
            prompts.append(prompt)
            return "[1]"

        monkeypatch.setattr(um, "complete", lambda _p, prompt, **k: judge(prompt, **k))
        get_config().utility.mining.behavioural_matching = True
        result = await um.mine_session_from_transcript(
            "default", "[tool: Bash — git commit -s -m x]\n", "now"
        )

        assert result.literal_nominated == 1
        assert result.candidates == 1  # only the rule file's belief was shown
        assert all("8 times" not in p for p in prompts)
        assert await _events(s) == {("p-rule", "now", "literal", "mined")}


async def _events(session: AsyncSession) -> set[tuple[str, str, str | None, str]]:
    rows = await session.execute(
        select(
            UtilityEventRow.particle_id,
            UtilityEventRow.session_id,
            UtilityEventRow.match_basis,
            UtilityEventRow.source,
        )
    )
    return {tuple(r) for r in rows.all()}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. the re-mine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_clears_the_mined_channel_and_rebuilds_the_explicit_one(
    file_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit test 8: every session re-mined under the new rule; explicit credits replayed."""
    import particles.operations.utility_mining as um
    from particles.operations.utility_feedback import mark_belief_useful
    from particles.store.utility_store import record_utility_events

    s = file_db_session
    await _rule_file(s, "agents", f"{ROOT}/AGENTS.md")
    await _entry(s, "conv-old", "claude-code://session/old", ["claude-code"])
    await _snapshot(s, "conv-old-s1", "conv-old", START - timedelta(days=25))
    await _belief(s, "p-rule", "Commit with `git commit -s`", "agents", "agents-s1")
    await _belief(s, "p-mention", "It ran `git commit -s` 8 times", "conv-old", "conv-old-s1")
    await _conversation(s, "a")
    await _conversation(s, "b")
    # The legacy channel: an ungated literal credit on the mention-only belief.
    await record_utility_events(s, "legacy", {"p-mention": "literal"})
    await mark_belief_useful(s, "p-mention", now=START)
    await s.commit()
    explicit_before = {e for e in await _events(s) if e[3] == SOURCE_EXPLICIT}
    assert explicit_before

    transcripts = {
        "hash-conv-a-s1": "[tool: Bash — git commit -s -m x]\n",
        "hash-conv-b-s1": "[tool: Bash — git log]\n",
    }
    monkeypatch.setattr(
        "particles.corpus.deposit.load_blob", lambda h: transcripts[h].encode("utf-8")
    )

    async def judge_many(_purpose: str, requests: list[Any], **_k: Any) -> list[str]:
        # Session a's only literal candidate is the rule; rule it applied.
        return ["[1]" if "git commit -s -m x" in r.prompt else "[]" for r in requests]

    monkeypatch.setattr("particles.llm.complete_many", judge_many)
    get_config().utility.mining.behavioural_matching = True

    plan = await um.plan_store_utility("default")
    assert plan.calls >= 1
    assert plan.estimate().calls == plan.calls
    result = await um.rebuild_store_utility("default", plan)

    events = await _events(s)
    assert ("p-mention", "legacy", "literal", "mined") not in events
    assert ("p-rule", "a", "literal", "mined") in events
    assert not any(e[1] == "b" and e[3] == "mined" for e in events)
    assert {e for e in events if e[3] == SOURCE_EXPLICIT} == explicit_before
    assert result.literal == 1
    assert result.explicit == 1


@pytest.mark.asyncio
async def test_a_rebuild_with_the_judge_off_records_no_mined_events(
    file_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consequences: a literal-only rebuild now records nothing."""
    import particles.operations.utility_mining as um

    s = file_db_session
    await _rule_file(s, "agents", f"{ROOT}/AGENTS.md")
    await _belief(s, "p-rule", "Commit with `git commit -s`", "agents", "agents-s1")
    await _conversation(s, "a")
    await s.commit()
    monkeypatch.setattr(
        "particles.corpus.deposit.load_blob", lambda _h: b"[tool: Bash - git commit -s]\n"
    )
    get_config().utility.mining.behavioural_matching = False

    result = await um.rebuild_store_utility("default")
    assert (result.literal, result.behavioural, result.behavioural_calls) == (0, 0, 0)
    assert await _events(s) == set()
