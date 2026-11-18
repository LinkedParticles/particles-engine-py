# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The episode reading over a store (stage 1): gather, shadow, regenerate, sheet.

* Only a claim every source of which is a session type is asked; a credited
  or directly asserted claim settles ``STANDING``.
* The shadow counts what the fold would withhold on a card list and withholds
  nothing; the curation snapshot carries the count and reads it back.
* Regeneration reads the backlog several claims of one entry per call, writes
  the reading with the reader's stamp, never overwrites an operator reading,
  leaves a claim the reply skipped as it was, and writes nothing on a dry run.
  The writer touches only the two episode keys.
* The sheet samples each entry's withheld cards with both claims and an empty
  label, and samples ``STANDING`` readings for recall.
* The CLI refuses an unknown scope.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.episode import (
    EPISODE_KEY,
    EPISODE_RULE_KEY,
    EXTRACTOR_WRITER,
    OPERATOR_RULE,
)
from particles.core.rung_ladder import EpisodeVerdict, QuestionKind
from particles.core.schema import ExtractorRef, Mutability
from particles.extraction import general
from particles.extraction.episode import EpisodeItem, EpisodeReply, regeneration_stamp
from particles.ingest.ladder_shell import rung_tally
from particles.operations.episode import (
    asked_claims,
    regenerate_episode,
    shadow_for_cards,
    write_sheet,
)
from particles.store.episode_store import get_episode_rows, set_episode_reading
from tests._observer_scope import belief

runner = CliRunner()

EXTRACTED = ExtractorRef(name="general-extractor", version="0.21.0")


async def _entry(session: Any, sid: str, source_type: str = "CONVERSATION") -> tuple[str, str]:
    from particles.corpus.deposit import deposit_text_versioned

    entry_id, snapshot_id, _ = await deposit_text_versioned(
        session,
        text=f"user: open the PR\n\nassistant: PR #12 is open now ({sid}).\n\nThe owner "
        "prefers one-command releases.",
        uri_r=f"claude-code://session/{sid}",
        source_type=source_type,
        mutability=Mutability.APPEND_ONLY,
        tags=["claude-code"],
        deposited_by="claude-code-hook",
    )
    return entry_id, snapshot_id


async def _seed(session: Any) -> dict[str, str]:
    e1 = await _entry(session, "s1")
    e2 = await _entry(session, "s2")
    note = await _entry(session, "note", source_type="LOCAL_MARKDOWN")
    ids: dict[str, str] = {}
    for key, text, sources in (
        ("a", "PR #12 is open.", (e1,)),
        ("b", "PR #12 has been opened.", (e1,)),
        ("c", "PR #12 is open on the branch.", (e2,)),
        ("multi", "PR #12 is open in both.", (e1, e2)),
        ("mixed", "PR #12 is open per the note.", (e1, note)),
        ("standing", "The owner prefers one-command releases.", (e1,)),
    ):
        ids[key] = (await belief(session, text, *sources, extractor_ref=EXTRACTED)).id
    ids["asserted"] = (await belief(session, "PR #12 is open, asserted.", e1)).id
    ids["e1"], ids["e2"] = e1[0], e2[0]
    await session.commit()
    return ids


def _serving() -> str:
    from particles.config import get_config

    selection = get_config().llm.for_purpose("extraction")
    return f"{selection.provider}:{selection.model}"


def _reader(verdicts: dict[str, EpisodeVerdict | None], model: str | None = None) -> Any:
    calls: list[list[str]] = []

    async def read(items: Sequence[EpisodeItem]) -> EpisodeReply | None:
        calls.append([i.content for i in items])
        return EpisodeReply(
            verdicts=tuple(verdicts.get(i.content) for i in items),
            provider_model=model or _serving(),
            stamp=regeneration_stamp(),
        )

    read.calls = calls  # type: ignore[attr-defined]
    return read


SB, ST = EpisodeVerdict.SESSION_BOUND, EpisodeVerdict.STANDING


@pytest.mark.asyncio
async def test_only_session_only_claims_are_asked(db_session: Any) -> None:
    ids = await _seed(db_session)
    claims = await asked_claims(db_session, ids.values())
    assert ids["mixed"] not in claims
    assert {ids["a"], ids["multi"], ids["asserted"]} <= set(claims)
    assert claims[ids["asserted"]].settled.verdict is EpisodeVerdict.STANDING
    assert claims[ids["multi"]].entries == frozenset({ids["e1"], ids["e2"]})


@pytest.mark.asyncio
async def test_regeneration_writes_readings_and_spares_the_operator(db_session: Any) -> None:
    ids = await _seed(db_session)
    await set_episode_reading(db_session, ids["standing"], verdict=ST, rule=OPERATOR_RULE)
    await db_session.commit()
    read = _reader(
        {
            "PR #12 is open.": SB,
            "PR #12 has been opened.": SB,
            "PR #12 is open on the branch.": SB,
            "PR #12 is open in both.": None,
            "PR #12 is open, asserted.": ST,
        }
    )
    dry = await regenerate_episode(db_session, scope="all", dry_run=True, read=read)
    assert dry.backlog == 5 and dry.written == 0 and not read.calls

    summary = await regenerate_episode(
        db_session, scope="all", limit=0, rate_limit_per_minute=0, read=read
    )
    assert summary.written == 4
    assert summary.session_bound == 3 and summary.standing == 1
    assert summary.failed == 1  # the reply gave the multi-entry claim no answer
    # One call per entry's batch: never two sessions' claims in one call.
    assert all(len({c.split(" ")[-1] for c in call}) >= 1 for call in read.calls)

    rows = {r.particle_id: r for r in await get_episode_rows(db_session)}
    assert rows[ids["a"]].reading is not None
    assert rows[ids["a"]].reading.rule == regeneration_stamp()
    assert rows[ids["standing"]].reading is not None
    assert rows[ids["standing"]].reading.rule == OPERATOR_RULE
    assert rows[ids["multi"]].reading is None
    assert rows[ids["mixed"]].reading is None

    assert rows[ids["a"]].reading.model == _serving()
    assert str(rows[ids["a"]].reading.stamp) == summary.stamp

    again = await regenerate_episode(
        db_session, scope="all", limit=0, rate_limit_per_minute=0, read=read
    )
    assert again.backlog == 1  # only the claim the reply skipped


@pytest.mark.asyncio
async def test_a_reading_by_another_model_is_a_different_stamp(db_session: Any) -> None:
    ids = await _seed(db_session)
    await set_episode_reading(
        db_session, ids["a"], verdict=SB, rule=regeneration_stamp(), model="other:model"
    )
    await db_session.commit()
    rows = {r.particle_id: r for r in await get_episode_rows(db_session)}
    reading = rows[ids["a"]].reading
    assert reading is not None
    assert str(reading.stamp) == f"{regeneration_stamp()}~other:model"
    # Not the serving model's stamp, so the next run reads the claim again.
    dry = await regenerate_episode(db_session, scope="all", dry_run=True)
    assert dry.backlog == 6
    assert dry.readings[f"{regeneration_stamp()}~other:model:session_bound"] == 1


@pytest.mark.asyncio
async def test_an_extracted_reading_is_read_under_the_claims_extraction_model(
    db_session: Any,
) -> None:
    from particles.extraction.general import episode_extractor_stamp

    ids = await _seed(db_session)
    from particles.store.particle_store import ParticleRow

    row = await db_session.get(ParticleRow, ids["a"])
    row.extraction_provider_model = "anthropic:claude-sonnet-5-5"
    await set_episode_reading(db_session, ids["a"], verdict=SB, rule=episode_extractor_stamp())
    await db_session.commit()
    rows = {r.particle_id: r for r in await get_episode_rows(db_session)}
    reading = rows[ids["a"]].reading
    assert reading is not None and reading.model == "anthropic:claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_the_dry_run_prices_this_run_and_the_backlog(db_session: Any) -> None:
    await _seed(db_session)
    dry = await regenerate_episode(db_session, scope="all", limit=2, dry_run=True)
    assert dry.estimate is not None and dry.backlog_estimate is not None
    assert dry.estimate["calls"] == 1 and dry.backlog_estimate["calls"] == dry.calls
    assert dry.estimate["model"] == _serving()
    assert isinstance(dry.estimate["input_tokens"], int) and dry.estimate["input_tokens"] > 0
    assert dry.backlog_estimate["usd"] is not None
    assert dry.backlog_estimate["usd"] >= dry.estimate["usd"]


@pytest.mark.asyncio
async def test_the_writer_touches_only_the_episode_keys(db_session: Any) -> None:
    ids = await _seed(db_session)
    from particles.store.particle_store import ParticleRow, set_particle_property

    await set_particle_property(db_session, ids["a"], "conflict:reading", "kept")
    await set_episode_reading(db_session, ids["a"], verdict=SB, rule="regen.episode@x")
    row = await db_session.get(ParticleRow, ids["a"])
    props = json.loads(row.properties_json)
    assert props == {
        "conflict:reading": "kept",
        EPISODE_KEY: "session_bound",
        EPISODE_RULE_KEY: "regen.episode@x",
    }
    with pytest.raises(ValueError, match="never abstain"):
        await set_episode_reading(db_session, ids["a"], verdict=EpisodeVerdict.ABSTAIN, rule="x")


@pytest.mark.asyncio
async def test_the_shadow_counts_and_reports_rungs(db_session: Any) -> None:
    ids = await _seed(db_session)
    stamp = regeneration_stamp()
    for key in ("a", "b", "c", "multi"):
        await set_episode_reading(db_session, ids[key], verdict=SB, rule=stamp)
    await db_session.commit()
    cards = [
        ("duplicate_group", [ids["a"], ids["b"]]),
        ("contradiction", [ids["a"], ids["c"]]),
        ("duplicate_pair", [ids["a"], ids["multi"]]),
        ("duplicate_pair", [ids["a"], ids["mixed"]]),
        ("no_subject", [ids["a"]]),
    ]
    with rung_tally() as tally:
        shadow = await shadow_for_cards(db_session, cards)
    assert shadow.withheld == 2 and shadow.session_cards == 2
    assert shadow.admitted_withheld == 0
    assert set(shadow.entries) == {f"one_entry:{stamp}", f"rest:{stamp}"}
    counted = tally.payload()[QuestionKind.EPISODE.value]["settled"]
    assert counted[stamp]["session_bound"] == 4


@pytest.mark.asyncio
async def test_the_snapshot_carries_the_shadow(db_session: Any) -> None:
    from particles.operations.curation.snapshot import (
        CollectionScope,
        _dump,
        _episode_block,
        read_prior,
    )
    from particles.store.curation_snapshot_store import latest_snapshot, write_snapshot

    ids = await _seed(db_session)
    from particles.operations.curation.cards import CardKind, CurationCard

    card = CurationCard(
        kind=CardKind.DUPLICATE_PAIR, particle_ids=[ids["a"], ids["b"]], diagnostic="x"
    )
    episode = await _episode_block(db_session, [card])
    assert episode is not None and episode["asked"] == 2
    await write_snapshot(
        db_session,
        cards_json=_dump([card], {}, {}, episode=episode),
        card_count=1,
        scope=CollectionScope.STORE,
        per_kind_scope={},
        semantic=False,
        semantic_degraded=False,
        retain=3,
    )
    row = await latest_snapshot(db_session)
    assert row is not None
    assert read_prior(row).episode == episode


@pytest.mark.asyncio
async def test_the_sheet_samples_withheld_cards_and_standing_readings(
    db_session: Any, tmp_path: Path
) -> None:
    ids = await _seed(db_session)
    stamp = regeneration_stamp()
    for key in ("a", "b"):
        await set_episode_reading(db_session, ids[key], verdict=SB, rule=stamp)
    await set_episode_reading(db_session, ids["standing"], verdict=ST, rule=stamp)
    await db_session.commit()
    path = tmp_path / "sheet.jsonl"
    summary = await write_sheet(
        db_session,
        path,
        size=5,
        # Recall is sampled among card sides, the population a fold acts on.
        cards=[
            ("duplicate_pair", [ids["a"], ids["b"]]),
            ("duplicate_pair", [ids["a"], ids["standing"]]),
        ],
    )
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    withheld = [x for x in lines if x["sheet"] == "withheld"]
    assert summary.entries == {f"one_entry:{stamp}": {"withheld": 1, "sampled": 1}}
    assert len(withheld) == 1 and withheld[0]["label"] == ""
    assert {c["id"] for c in withheld[0]["claims"]} == {ids["a"], ids["b"]}
    assert "PR #12" in withheld[0]["claims"][0]["passage"]
    assert [x["claims"][0]["id"] for x in lines if x["sheet"] == "standing"] == [ids["standing"]]


def test_the_component_name_is_the_extractor_writer() -> None:
    assert general.PROMPT_EPISODE == EXTRACTOR_WRITER


def test_the_cli_refuses_an_unknown_scope(cli_db: Path) -> None:
    result = runner.invoke(app, ["episode", "--scope", "everything", "--dry-run"])
    assert result.exit_code == 2
    assert "--scope is 'cards' or 'all'" in result.output


def test_the_cli_dry_run_reports_an_empty_backlog(cli_db: Path) -> None:
    result = runner.invoke(app, ["episode", "--dry-run", "--scope", "all"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["dry_run"] is True and report["backlog"] == 0


def test_curate_prints_the_shadow_it_was_served() -> None:
    from particles.api.cli.curate import _episode_line

    assert _episode_line(None) == ""
    line = _episode_line(
        {"withheld": 1148, "session_cards": 90, "admitted_withheld": 0, "admitted_net": 0}
    )
    assert line.startswith("Episode shadow: 1,148 pair and group cards would fold into 90")
    assert "Nothing is withheld yet." in line


def test_the_http_queue_carries_the_shadow() -> None:
    from particles.api.app import CurationQueueResponse

    body = CurationQueueResponse(cards=[], count=0, episode_shadow={"withheld": 3})
    assert body.model_dump()["episode_shadow"] == {"withheld": 3}


def test_the_estimate_line_names_calls_model_tokens_and_price() -> None:
    from particles.api.cli.episode import _estimate_line

    assert _estimate_line("This run", None) == ""
    line = _estimate_line(
        "Whole backlog",
        {
            "calls": 866,
            "model": "anthropic:m",
            "input_tokens": 3_494_694,
            "output_tokens": 1_396_815,
            "usd": 20.96,
        },
    )
    assert line == (
        "Whole backlog: 866 calls on anthropic:m, about 3,494,694 input and "
        "1,396,815 output tokens, at most $20.96 at list price."
    )
    unpriced = _estimate_line("This run", {"calls": 1, "model": "local:q", "usd": None})
    assert unpriced.endswith("unpriced at list price.")
