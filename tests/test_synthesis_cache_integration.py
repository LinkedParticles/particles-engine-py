"""Cross-exporter synthesis cache.

End-to-end behaviour: when one exporter's ``render_article`` call
populates the ``synthesis_cache`` table, a subsequent call against
the same ``(subject, input_hash, prompt_version)`` triple — even
through a different exporter — returns the cached body without
hitting the LLM.

This is the promise made enforceable. The unit tests in
``test_synthesis_cache_store.py`` cover the storage layer; this
file exercises the orchestrator integration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from particles.core.schema import (
    CalibrationSource,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Status,
    Subject,
    UncertaintyNature,
)
from particles.db import session_scope
from particles.exporters.article_synthesis import compute_input_hash, render_article


def _make_particle(content: str) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=0.9,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="entry-1",
                snapshot_id="snap-1",
            )
        ],
        asserted_by="stub-extractor",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": "stub-extractor", "version": "0.1.0"},
        subject_ids=[],
    )


def _mock_client(text_responses: list[str]) -> MagicMock:
    """Stub Anthropic client whose ``messages.create`` returns texts in order."""
    import anthropic

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()

    iterator = iter(text_responses)

    def _create(**kwargs: Any) -> MagicMock:
        content = MagicMock()
        content.text = next(iterator)
        resp = MagicMock()
        resp.content = [content]
        return resp

    client.messages.create = MagicMock(side_effect=_create)
    return client


@pytest.mark.asyncio
async def test_render_article_caches_and_reuses_across_calls(
    db_session: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First render hits the LLM; second render against the same key
    returns the cached body without any further LLM call.

    Simulates the wiki → obsidian → logseq cascade — same subject,
    same input_hash, three exporters — by invoking render_article
    twice with the same arguments. Verifies the mock LLM gets called
    only during the first pass and the returned body is identical."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    subject = Subject(
        id=str(uuid.uuid4()),
        canonical_name="Test Subject",
        asserted_by="test",
    )
    particles = [_make_particle(f"claim {i}") for i in range(3)]
    cited = [p.id[:8] for p in particles]
    llm_body = (
        f"# Test Subject\n\n"
        f"First claim [^p-{cited[0]}]. "
        f"Second claim [^p-{cited[1]}]. "
        f"Third claim [^p-{cited[2]}].\n"
    )
    input_hash = compute_input_hash(particles, subject)
    eff = {p.id: 0.85 for p in particles}

    from particles.llm import set_client

    # Two responses for pass 1 (synthesis + LayerB returns []), then
    # an explicit raise on any further LLM call — second pass must
    # short-circuit on the cache hit before reaching the mock.
    client = _mock_client([llm_body, "[]"])
    set_client(client)
    try:
        async with session_scope() as session:
            # Pass 1 — cache miss → LLM call → store.
            body1, used1 = await render_article(
                subject=subject,
                particles=particles,
                eff=eff,
                input_hash=input_hash,
                corpus_uris={"entry-1": "https://example.com/source"},
                max_tokens=2048,
                layer_b_enabled=True,
                session=session,
            )
            await session.commit()

        assert used1 is True
        # 2 LLM calls in pass 1: synthesis + LayerB judge.
        assert client.messages.create.call_count == 2

        async with session_scope() as session:
            # Pass 2 — cache hit → no LLM call → returns stored body.
            body2, used2 = await render_article(
                subject=subject,
                particles=particles,
                eff=eff,
                input_hash=input_hash,
                corpus_uris={"entry-1": "https://example.com/source"},
                max_tokens=2048,
                layer_b_enabled=True,
                session=session,
            )

        assert used2 is True
        # The mock's call count must NOT have increased — pass 2 hit
        # the cache and returned before reaching the LLM.
        assert client.messages.create.call_count == 2
        # Same body returned both passes — the cache is round-trip stable.
        assert body1 == body2
    finally:
        set_client(None)


@pytest.mark.asyncio
async def test_render_article_no_session_does_not_use_cache(
    db_session: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``session=None``, the cache is bypassed entirely — each call
    re-hits the LLM. This is the opt-out path; callers without a
    session don't touch the DB at all."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    subject = Subject(
        id=str(uuid.uuid4()),
        canonical_name="Solo Subject",
        asserted_by="test",
    )
    particles = [_make_particle("only claim")]
    cited = particles[0].id[:8]
    llm_body = f"# Solo Subject\n\nThe only claim [^p-{cited}].\n"
    input_hash = compute_input_hash(particles, subject)
    eff = {particles[0].id: 0.85}

    from particles.llm import set_client

    # Four LLM-call slots: two passes × (synthesis + LayerB).
    client = _mock_client([llm_body, "[]", llm_body, "[]"])
    set_client(client)
    try:
        body1, _ = await render_article(
            subject=subject,
            particles=particles,
            eff=eff,
            input_hash=input_hash,
            corpus_uris={"entry-1": "https://example.com/source"},
            max_tokens=2048,
            layer_b_enabled=True,
            session=None,
        )
        body2, _ = await render_article(
            subject=subject,
            particles=particles,
            eff=eff,
            input_hash=input_hash,
            corpus_uris={"entry-1": "https://example.com/source"},
            max_tokens=2048,
            layer_b_enabled=True,
            session=None,
        )
        # Both passes called the LLM — no caching, no short-circuit.
        assert client.messages.create.call_count == 4
        # Bodies are independently rendered each pass; LLM body is the
        # same (deterministic mock) but rendered frontmatter carries a
        # per-call generated_at timestamp. Assert the LLM-side content
        # made it through both renders, not byte-equality.
        assert "only claim" in body1
        assert "only claim" in body2
    finally:
        set_client(None)


@pytest.mark.asyncio
async def test_obsidian_short_circuit_backfills_db_cache(db_session: object) -> None:
    """0.42.1 regression fix: when Obsidian's on-disk hash matches and the
    exporter takes its per-note shortcut (skipping render_article entirely),
    the DB cache also gets populated from the on-disk body. Without
    this backfill, vaults synthesised pre-0.41.0 force the next cross-
    exporter run to re-pay full LLM cost — the shortcut never reaches the
    cache-write path in ``render_article``.
    """
    from particles.exporters.obsidian.synthesis import (
        _backfill_synthesis_cache_if_absent,
    )
    from particles.render.article_synthesis.cache import _PROMPT_VERSION
    from particles.store.synthesis_cache_store import lookup_cached_article

    subject_id = str(uuid.uuid4())
    input_hash = "deadbeef" * 8
    prior_body = (
        "---\narticle_input_hash: " + input_hash + "\n---\n# A Subject\n\nSynthesised prose body.\n"
    )

    # Pre-condition: DB has no row for this key.
    async with session_scope() as session:
        assert await lookup_cached_article(session, subject_id, input_hash, _PROMPT_VERSION) is None

        # Backfill from the on-disk body.
        await _backfill_synthesis_cache_if_absent(
            session,
            subject_id=subject_id,
            input_hash=input_hash,
            prior_body=prior_body,
        )
        await session.commit()

    # Post-condition: DB now has the body.
    async with session_scope() as session:
        cached = await lookup_cached_article(session, subject_id, input_hash, _PROMPT_VERSION)
    assert cached == prior_body


@pytest.mark.asyncio
async def test_backfill_does_not_overwrite_existing_db_row(db_session: object) -> None:
    """If the DB cache already has a row for the key, the backfill is a
    no-op — we don't clobber a freshly-rendered body with stale on-disk
    content."""
    from particles.exporters.obsidian.synthesis import (
        _backfill_synthesis_cache_if_absent,
    )
    from particles.render.article_synthesis.cache import _PROMPT_VERSION
    from particles.store.synthesis_cache_store import (
        lookup_cached_article,
        store_cached_article,
    )

    subject_id = str(uuid.uuid4())
    input_hash = "0" * 64
    canonical_body = "canonical body from render_article"
    stale_body = "stale body from old on-disk vault"

    async with session_scope() as session:
        await store_cached_article(session, subject_id, input_hash, _PROMPT_VERSION, canonical_body)
        await _backfill_synthesis_cache_if_absent(
            session, subject_id=subject_id, input_hash=input_hash, prior_body=stale_body
        )
        await session.commit()

    async with session_scope() as session:
        cached = await lookup_cached_article(session, subject_id, input_hash, _PROMPT_VERSION)
    assert cached == canonical_body
