"""CLI tests for `particles corpus links suggest` / `dismiss`."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.db import session_scope


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_citation(db_path: Path, url: str, sources: dict[str, str]) -> None:
    """Seed ``sources`` ({entry_id: uri_r}) each citing ``url``."""

    async def _impl() -> None:
        from particles.corpus.store import CorpusEntryRow
        from particles.store.url_mention_store import record_url_mentions

        async with session_scope() as session:
            for entry_id, uri_r in sources.items():
                session.add(
                    CorpusEntryRow(
                        entry_id=entry_id,
                        uri_r=uri_r,
                        source_type="WEB_PAGE",
                        mutability="MUTABLE",
                        fetch_policy="LAZY",
                        created_at=datetime.now(UTC),
                        deposited_by="test",
                    )
                )
            await session.flush()
            for entry_id in sources:
                await record_url_mentions(session, source_entry_id=entry_id, canonical_urls=[url])
            await session.commit()

    asyncio.run(_impl())


def test_suggest_empty(runner: CliRunner, cli_db: Path) -> None:
    result = runner.invoke(app, ["corpus", "links", "suggest"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "No deposit suggestions" in result.output


def test_suggest_shows_cited_url(runner: CliRunner, cli_db: Path) -> None:
    _seed_citation(
        cli_db,
        "https://press.example/release",
        {"s1": "https://a.example/1", "s2": "https://b.example/2"},
    )
    result = runner.invoke(app, ["corpus", "links", "suggest"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "https://press.example/release" in result.output


def test_suggest_json_output(runner: CliRunner, cli_db: Path) -> None:
    _seed_citation(
        cli_db,
        "https://press.example/release",
        {"s1": "https://a.example/1", "s2": "https://b.example/2"},
    )
    result = runner.invoke(
        app, ["corpus", "links", "suggest", "-o", "json"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["suggestions"][0]["canonical_url"] == "https://press.example/release"
    assert payload["suggestions"][0]["distinct_sources"] == 2


def test_suggest_rejects_bad_format(runner: CliRunner, cli_db: Path) -> None:
    result = runner.invoke(
        app, ["corpus", "links", "suggest", "-o", "yaml"], catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "Invalid --output-format" in result.output


def test_dismiss_then_suggest_hides_url(runner: CliRunner, cli_db: Path) -> None:
    url = "https://press.example/release"
    _seed_citation(cli_db, url, {"s1": "https://a.example/1", "s2": "https://b.example/2"})

    dismissed = runner.invoke(app, ["corpus", "links", "dismiss", url], catch_exceptions=False)
    assert dismissed.exit_code == 0, dismissed.output
    assert "Dismissed" in dismissed.output

    after = runner.invoke(app, ["corpus", "links", "suggest"], catch_exceptions=False)
    assert url not in after.output


def test_dismiss_rejects_bad_url(runner: CliRunner, cli_db: Path) -> None:
    result = runner.invoke(app, ["corpus", "links", "dismiss", "not-a-url"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "Not a usable" in result.output
