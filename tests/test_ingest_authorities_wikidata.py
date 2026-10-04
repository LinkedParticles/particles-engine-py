# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Wikidata API helpers in ``particles.ingest.authorities.wikidata``.

Covers the response handling of ``_wikidata_candidates`` and ``_wikidata_aliases``
against a mocked ``particles_client`` (never the live API): hit selection, the
prefix-expansion skip and its alias rescue over recorded search responses,
alias collection and de-duplication, and the error-returns-nothing
branches. ``_is_prefix_expansion`` on its own, the
link-confidence scorer, and the resolver cascade are covered in
``tests/test_subjects.py`` and ``tests/test_wikidata_link_confidence.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from particles.config import get_config
from particles.ingest.authorities import wikidata
from particles.ingest.authorities._shared import reset_limiters

# Captured at import, before any autouse fixture swaps the module attribute.
from particles.ingest.authorities.wikidata import _wikidata_limiter as _real_wikidata_limiter
from tests._capped_http import set_capped_responses

_API = "https://www.wikidata.org/w/api.php"


class _NoWaitLimiter:
    async def acquire(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the 2 rps throttle out of these tests; its spacing is tested in
    ``tests/test_ingest_authorities_shared.py``."""
    monkeypatch.setattr(wikidata, "_wikidata_limiter", _NoWaitLimiter)


def _json_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(*responses: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    set_capped_responses(client, side_effect=list(responses))
    return client


def _stream_params(client: AsyncMock) -> dict[str, str]:
    call = client.stream.call_args
    assert call.args[0] == "GET"
    assert call.args[1] == _API
    params: dict[str, str] = call.kwargs["params"]
    return params


# ---------------------------------------------------------------------------
# _wikidata_candidates
# ---------------------------------------------------------------------------


async def test_search_returns_every_hit_in_rank_order_and_sends_wbsearchentities() -> None:
    hit = {"id": "Q28865", "label": "Python", "description": "programming language"}
    genus = {"id": "Q271218", "label": "Python (genus)"}
    client = _mock_client(_json_response({"search": [hit, genus]}))

    with patch(
        "particles.ingest.authorities.wikidata.particles_client", return_value=client
    ) as factory:
        result = await wikidata._wikidata_candidates("Python")

    # Each kept hit carries its rank in the response; the response's
    # own dicts are left alone.
    assert result == [{**hit, "rank": 1}, {**genus, "rank": 2}]
    assert "rank" not in hit
    factory.assert_called_once_with(timeout=10.0)
    assert _stream_params(client) == {
        "action": "wbsearchentities",
        "search": "Python",
        "language": "en",
        "format": "json",
        "limit": "5",
    }


async def test_search_skips_prefix_expansions_and_logs_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _mock_client(
        _json_response(
            {
                "search": [
                    {"id": "Q1", "label": "micrograd-like Methods", "description": "article"},
                    {"id": "Q2", "label": "micrograd: A Tiny Autograd", "description": "paper"},
                    {"id": "Q3", "label": "micrograd", "description": "autograd engine"},
                ]
            }
        )
    )

    with (
        patch("particles.ingest.authorities.wikidata.particles_client", return_value=client),
        caplog.at_level(logging.INFO, logger=wikidata.__name__),
    ):
        result = await wikidata._wikidata_candidates("micrograd")

    assert [h["id"] for h in result] == ["Q3"]
    skipped = [r.getMessage() for r in caplog.records if "prefix-expansion" in r.getMessage()]
    assert len(skipped) == 2
    assert "Q1" in skipped[0]
    assert "Q2" in skipped[1]


async def test_search_returns_nothing_when_every_hit_is_a_prefix_expansion() -> None:
    client = _mock_client(
        _json_response(
            {"search": [{"id": "Q1", "label": "FlashAttention: Fast and Memory-Efficient"}]}
        )
    )

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._wikidata_candidates("FlashAttention") == []


# Recorded live responses, 2026-10-01, trimmed to the fields the code reads.
# "Postgres": Wikidata's top hit is PostgreSQL, which continues the
# name with a letter, and its "Postgres" alias is held under the
# language-neutral ``mul`` code, not ``en``.
_POSTGRES_SEARCH = {
    "search": [
        {
            "id": "Q192490",
            "label": "PostgreSQL",
            "description": "free and open-source relational database management system",
            "match": {"type": "label", "language": "en", "text": "PostgreSQL"},
        },
        {
            "id": "Q28975208",
            "label": "POSTGRES",
            "description": "discontinued database software, predecessor to PostgreSQL",
            "match": {"type": "label", "language": "en", "text": "POSTGRES"},
        },
        {
            "id": "Q18563589",
            "label": "PostgreSQL License",
            "description": "permissive non-copyleft free software license",
            "match": {"type": "label", "language": "en", "text": "PostgreSQL License"},
        },
    ]
}
_POSTGRES_ALIASES = {
    "entities": {
        "Q192490": {
            "id": "Q192490",
            "aliases": {"mul": [{"language": "mul", "value": "Postgres"}]},
        },
        "Q18563589": {"id": "Q18563589", "aliases": {}},
    }
}

# "Go": the top hit is the Brazilian state Goiás, whose aliases hold "GO" but
# not "Go"; the programming language is the second hit.
_GO_SEARCH = {
    "search": [
        {"id": "Q41587", "label": "Goiás", "description": "state of Brazil"},
        {"id": "Q37227", "label": "Go", "description": "programming language"},
        {"id": "Q12013", "label": "Google Maps", "description": "web mapping service"},
    ]
}
_GO_ALIASES = {
    "entities": {
        "Q41587": {
            "aliases": {
                "en": [
                    {"language": "en", "value": "Goias state"},
                    {"language": "en", "value": "GO"},
                    {"language": "en", "value": "BR-GO"},
                ],
                "mul": [{"language": "mul", "value": "GO"}],
            }
        },
        "Q12013": {"aliases": {"en": [{"language": "en", "value": "maps.google.com"}]}},
    }
}


def _all_stream_params(client: AsyncMock) -> list[dict[str, str]]:
    return [call.kwargs["params"] for call in client.stream.call_args_list]


async def test_search_keeps_a_continuation_whose_aliases_name_the_query() -> None:
    # "PostgreSQL" continues "Postgres" with a letter, but the item
    # lists "Postgres" as one of its names, so it is the same entity and stays
    # the top hit. "PostgreSQL License" continues it too and does not.
    client = _mock_client(_json_response(_POSTGRES_SEARCH), _json_response(_POSTGRES_ALIASES))

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        result = await wikidata._wikidata_candidates("Postgres")

    assert [h["id"] for h in result] == ["Q192490", "Q28975208"]
    search, aliases = _all_stream_params(client)
    assert search["action"] == "wbsearchentities"
    # One batched read, over the continuation candidates only, in both
    # languages that can hold the alias.
    assert aliases == {
        "action": "wbgetentities",
        "ids": "Q192490|Q18563589",
        "props": "aliases",
        "languages": "en|mul",
        "format": "json",
    }


async def test_search_alias_rescue_is_case_sensitive() -> None:
    # Goiás lists "GO", not "Go": a case-insensitive match would make the
    # state the top hit for the programming language.
    client = _mock_client(_json_response(_GO_SEARCH), _json_response(_GO_ALIASES))

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        result = await wikidata._wikidata_candidates("Go")

    assert [h["id"] for h in result] == ["Q37227"]


async def test_search_makes_no_alias_read_without_a_continuation() -> None:
    hit = {"id": "Q28865", "label": "Python", "description": "programming language"}
    genus = {"id": "Q271218", "label": "Python (genus)"}
    client = _mock_client(_json_response({"search": [hit, genus]}))

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._wikidata_candidates("Python") == [
            {**hit, "rank": 1},
            {**genus, "rank": 2},
        ]

    assert client.stream.call_count == 1


async def test_search_failed_alias_read_rejects_every_continuation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The alias read failing leaves the filter as it was before the rescue:
    # continuations dropped, every other hit kept, and the failure disclosed.
    failing = _json_response({})
    failing.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
    )
    client = _mock_client(_json_response(_POSTGRES_SEARCH), failing)

    with (
        patch("particles.ingest.authorities.wikidata.particles_client", return_value=client),
        caplog.at_level(logging.WARNING, logger=wikidata.__name__),
    ):
        result = await wikidata._wikidata_candidates("Postgres")

    assert [h["id"] for h in result] == ["Q28975208"]
    assert any("continuation alias fetch failed" in r.getMessage() for r in caplog.records)


# A deeper search: the recorded "Postgres" response extended past
# five hits. Rank 6 continues the name with a letter and lists no "Postgres"
# alias, so the filter must drop it exactly as it drops rank 3; rank 7 breaks
# after the name with a space and is kept.
_POSTGRES_SEARCH_DEEP = {
    "search": [
        *_POSTGRES_SEARCH["search"],
        {"id": "Q4", "label": "Postgres (film)", "description": "short film"},
        {"id": "Q5", "label": "Postgres: The Definitive Guide", "description": "book"},
        {"id": "Q6", "label": "Postgresql Conference Europe", "description": "conference"},
        {"id": "Q7", "label": "Postgres Plus", "description": "database distribution"},
    ]
}
_POSTGRES_ALIASES_DEEP = {
    "entities": {**_POSTGRES_ALIASES["entities"], "Q6": {"id": "Q6", "aliases": {}}}
}


async def test_search_at_a_deeper_limit_sends_it_and_ranks_every_kept_hit() -> None:
    client = _mock_client(
        _json_response(_POSTGRES_SEARCH_DEEP), _json_response(_POSTGRES_ALIASES_DEEP)
    )

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        result = await wikidata._wikidata_candidates("Postgres", limit=7)

    search, aliases = _all_stream_params(client)
    assert search["limit"] == "7"
    # The filter ran over all seven: the paper title at rank 5 and the
    # continuation at rank 6 are gone, and the alias read covered rank 6 too.
    assert [(h["id"], h["rank"]) for h in result] == [
        ("Q192490", 1),
        ("Q28975208", 2),
        ("Q4", 4),
        ("Q7", 7),
    ]
    assert aliases["ids"] == "Q192490|Q18563589|Q6"


async def test_gate_hits_is_the_first_five_raw_hits_view() -> None:
    client = _mock_client(
        _json_response(_POSTGRES_SEARCH_DEEP), _json_response(_POSTGRES_ALIASES_DEEP)
    )
    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        deep = await wikidata._wikidata_candidates("Postgres", limit=7)

    five = {"search": _POSTGRES_SEARCH_DEEP["search"][:5]}
    client = _mock_client(_json_response(five), _json_response(_POSTGRES_ALIASES))
    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        shallow = await wikidata._wikidata_candidates("Postgres")

    # What the gate reads of a deep search is what a five-hit search returns.
    assert [h["id"] for h in wikidata.gate_hits(deep)] == [h["id"] for h in shallow]
    assert wikidata.gate_hits(deep) == deep[: len(shallow)]


def test_gate_hits_keeps_an_unranked_hit() -> None:
    hits: list[dict[str, object]] = [{"id": "Q1"}, {"id": "Q2", "rank": 5}, {"id": "Q3", "rank": 6}]
    assert wikidata.gate_hits(hits) == hits[:2]


async def test_continuation_aliases_reads_english_and_language_neutral() -> None:
    client = _mock_client(
        _json_response(
            {
                "entities": {
                    "Q1": {
                        "aliases": {
                            "en": [{"value": "One"}, {"value": ""}],
                            "mul": [{"value": "Uno"}],
                            "de": [{"value": "Eins"}],
                        }
                    }
                }
            }
        )
    )

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        aliases = await wikidata._continuation_aliases(["Q1", "Q2"])

    assert aliases == {"Q1": frozenset({"One", "Uno"}), "Q2": frozenset()}


async def test_continuation_aliases_of_nothing_makes_no_call() -> None:
    client = _mock_client()

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._continuation_aliases([]) == {}

    assert client.stream.call_count == 0


async def test_resolve_links_postgres_to_postgresql() -> None:
    # The baseline failure end to end: the top hit is adopted, not
    # the discontinued predecessor the filter used to fall through to.
    client = _mock_client(
        _json_response(_POSTGRES_SEARCH),
        _json_response(_POSTGRES_ALIASES),
        _json_response(
            {"entities": {"Q192490": {"labels": {"en": {"value": "PostgreSQL"}}, "aliases": {}}}}
        ),
    )

    with (
        patch("particles.ingest.authorities.wikidata.particles_client", return_value=client),
        patch(
            "particles.ingest.authorities.wikidata.find_by_external_ref",
            AsyncMock(return_value=None),
        ),
    ):
        resolution = await wikidata.WikidataAuthority().resolve(
            MagicMock(), "Postgres", particle_content=None, domain=None
        )

    assert resolution is not None
    assert resolution.external_ref is not None
    assert resolution.external_ref.id == "Q192490"
    assert resolution.canonical_name == "PostgreSQL"
    assert resolution.aliases == ["Postgres"]


@pytest.mark.parametrize("payload", [{"search": []}, {}])
async def test_search_returns_nothing_without_results(payload: dict[str, Any]) -> None:
    client = _mock_client(_json_response(payload))

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._wikidata_candidates("Nonexistent Thing") == []


async def test_search_http_error_returns_nothing_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resp = _json_response({"search": [{"id": "Q1", "label": "Python"}]})
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
    )
    client = _mock_client(resp)

    with (
        patch("particles.ingest.authorities.wikidata.particles_client", return_value=client),
        caplog.at_level(logging.WARNING, logger=wikidata.__name__),
    ):
        assert await wikidata._wikidata_candidates("Python") == []

    resp.json.assert_not_called()
    assert any("Wikidata search failed for 'Python'" in r.getMessage() for r in caplog.records)


async def test_search_connect_error_returns_nothing() -> None:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._wikidata_candidates("Python") == []


# ---------------------------------------------------------------------------
# _wikidata_aliases
# ---------------------------------------------------------------------------


async def test_aliases_puts_label_first_then_unique_aliases() -> None:
    client = _mock_client(
        _json_response(
            {
                "entities": {
                    "Q28865": {
                        "labels": {"en": {"language": "en", "value": "Python"}},
                        "aliases": {
                            "en": [
                                {"language": "en", "value": "Python language"},
                                {"language": "en", "value": "Python"},  # repeats the label
                                {"language": "en", "value": "py"},
                                {"language": "en", "value": "Python language"},  # duplicate
                                {"language": "en", "value": ""},  # empty
                                {"language": "en"},  # no value
                            ]
                        },
                    }
                }
            }
        )
    )

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        aliases = await wikidata._wikidata_aliases("Q28865")

    assert aliases == ["Python", "Python language", "py"]
    assert _stream_params(client) == {
        "action": "wbgetentities",
        "ids": "Q28865",
        "props": "labels|aliases",
        "languages": "en",
        "format": "json",
    }


async def test_aliases_without_english_label_keeps_aliases() -> None:
    client = _mock_client(
        _json_response(
            {"entities": {"Q1": {"labels": {}, "aliases": {"en": [{"value": "universe"}]}}}}
        )
    )

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._wikidata_aliases("Q1") == ["universe"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"entities": {}}, {"entities": {"Q999": {}}}, {"entities": {"Q1": {"labels": {}}}}],
)
async def test_aliases_missing_entity_returns_empty(payload: dict[str, Any]) -> None:
    client = _mock_client(_json_response(payload))

    with patch("particles.ingest.authorities.wikidata.particles_client", return_value=client):
        assert await wikidata._wikidata_aliases("Q1") == []


async def test_aliases_error_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    resp = _json_response({})
    resp.json = MagicMock(side_effect=ValueError("not JSON"))
    client = _mock_client(resp)

    with (
        patch("particles.ingest.authorities.wikidata.particles_client", return_value=client),
        caplog.at_level(logging.WARNING, logger=wikidata.__name__),
    ):
        assert await wikidata._wikidata_aliases("Q42") == []

    assert any("Wikidata alias fetch failed for Q42" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# WikidataAuthority.resolve over the mocked API
# ---------------------------------------------------------------------------


async def test_resolve_builds_resolution_from_search_and_aliases() -> None:
    client = _mock_client(
        _json_response(
            {"search": [{"id": "Q28865", "label": "Python", "description": "programming language"}]}
        ),
        _json_response(
            {
                "entities": {
                    "Q28865": {
                        "labels": {"en": {"value": "Python"}},
                        "aliases": {"en": [{"value": "Python language"}]},
                    }
                }
            }
        ),
    )

    with (
        patch("particles.ingest.authorities.wikidata.particles_client", return_value=client),
        patch(
            "particles.ingest.authorities.wikidata.find_by_external_ref",
            AsyncMock(return_value=None),
        ),
    ):
        resolution = await wikidata.WikidataAuthority().resolve(
            MagicMock(), "python3", particle_content=None, domain=None
        )

    assert resolution is not None
    assert resolution.canonical_name == "Python"
    # The queried name is kept as an alias when Wikidata does not list it.
    assert resolution.aliases == ["Python language", "python3"]
    assert resolution.description == "programming language"
    assert resolution.external_ref is not None
    assert resolution.external_ref.id == "Q28865"
    assert resolution.external_ref.uri == "https://www.wikidata.org/wiki/Q28865"
    assert client.stream.call_count == 2


# ---------------------------------------------------------------------------
# _wikidata_limiter
# ---------------------------------------------------------------------------


def test_wikidata_limiter_uses_configured_rate() -> None:
    reset_limiters()
    try:
        lim = _real_wikidata_limiter()
        assert _real_wikidata_limiter() is lim
        assert lim.requests_per_second == get_config().subjects.wikidata_rate_limit_rps
    finally:
        reset_limiters()
