# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Third-party JSON nested past the decoder's depth is refused, never raised past a reader.

Each test drives a body nested past the JSON decoder's depth (a few hundred
KiB, far under every byte cap) and asserts the function's documented refusal
(a ``SourceFetchError``, ``None``, or a quality note) rather than an escaping
``RecursionError`` (security review 2026-10-30, AB17, Z67).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

# Deep enough to pass the C decoder's stack check on any thread here
# (the measured limit is about 37k on the main thread, 75k on a worker).
_DEPTH = 200_000


def _deep_list() -> bytes:
    return ("[" * _DEPTH + "]" * _DEPTH).encode()


def _deep_in_dict(key: str) -> bytes:
    return ('{"' + key + '":' + "[" * _DEPTH + "]" * _DEPTH + "}").encode()


def _snapshot():
    from particles.core.schema import ExtractionStatus, Snapshot, WarcRecordType

    return Snapshot(
        snapshot_id="s4-snap",
        captured_at=datetime.now(UTC),
        content_hash="abc",
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
    )


def test_decoder_depth_is_reachable_under_the_byte_cap() -> None:
    """Sanity: the body is a few hundred KiB and the real decoder refuses it."""
    body = _deep_list()
    assert len(body) < 1_000_000
    with pytest.raises(RecursionError):
        json.loads(body)


async def test_reddit_arctic_shift_deep_body_is_a_source_fetch_error() -> None:
    """The archive answering with JSON nested past the decoder's depth is a SourceFetchError."""
    from particles.ingest.importers import reddit as imp

    body = _deep_in_dict("data")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(imp.SourceFetchError):
            await imp._arctic_shift_get(
                client, "https://arctic.example/api/posts/ids", {"ids": "x"}
            )


def test_reddit_importer_primary_url_deep_blob_reads_as_none() -> None:
    from particles.ingest.importers.reddit import RedditImporter

    assert RedditImporter().primary_url(_deep_list()) is None


def test_mastodon_importer_primary_url_deep_blob_reads_as_none() -> None:
    from particles.ingest.importers.mastodon import MastodonImporter

    assert MastodonImporter().primary_url(_deep_list()) is None


def test_hackernews_importer_primary_url_deep_blob_reads_as_none() -> None:
    from particles.ingest.importers.hackernews import HackerNewsImporter

    assert HackerNewsImporter().primary_url(_deep_list()) is None


def test_reddit_extractor_normalise_deep_blob_is_a_quality_note() -> None:
    from particles.extraction.reddit import RedditExtractor

    doc = RedditExtractor()._normalise(_deep_list(), _snapshot())
    assert doc.quality_notes


def test_mastodon_extractor_normalise_deep_blob_is_a_quality_note() -> None:
    from particles.extraction.mastodon import MastodonExtractor

    doc = MastodonExtractor()._normalise(_deep_list(), _snapshot())[0]
    assert doc.quality_notes


def test_hackernews_extractor_normalise_deep_blob_is_a_quality_note() -> None:
    from particles.extraction.hackernews import HackerNewsExtractor

    doc = HackerNewsExtractor()._normalise(_deep_list(), _snapshot())[0]
    assert doc.quality_notes


@pytest.mark.parametrize(
    "module, cls",
    [
        ("particles.extraction.numista.coin", None),
        ("particles.extraction.numista.issuer", None),
        ("particles.extraction.wikidata", None),
        ("particles.extraction.nomisma", None),
    ],
)
async def test_structured_extractors_refuse_a_deep_blob_with_a_quality_note(
    module: str, cls: str | None
) -> None:
    """A stored Numista, Wikidata or Nomisma blob nested past the decoder reads as a note."""
    import importlib
    import inspect

    mod = importlib.import_module(module)
    extractors = [
        obj
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == module and hasattr(obj, "extract") and hasattr(obj, "accepts")
    ]
    assert extractors, module
    result = await extractors[0]().extract(_snapshot(), _deep_list())
    assert result.quality_notes


async def test_gist_and_rdf_extractors_refuse_a_deep_blob_with_a_quality_note() -> None:
    """The gist and RDF extractors read a stored blob nested past the decoder as a note."""
    from particles.extraction.github.gist import GitHubGistExtractor
    from particles.extraction.rdf import RdfExtractor

    for extractor in (GitHubGistExtractor(), RdfExtractor()):
        result = await extractor.extract(_snapshot(), _deep_list())
        assert result.quality_notes, type(extractor).__name__


def test_nomisma_context_that_is_not_an_object_is_ignored() -> None:
    from particles.extraction.nomisma import _parse_graph

    nodes, prefixes = _parse_graph(b'{"@context": [1], "@graph": [{"@id": "x"}]}')
    assert nodes == [{"@id": "x"}]
    assert prefixes == {}
