"""Tests for the transcript-mining pass (the reliable action signal)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from particles.config import get_config
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.utility_mining import (
    extract_action_lines,
    literal_tokens,
    match_literal,
    mine_session,
    session_id_from_uri,
)

_TRANSCRIPT = """\
**user**: please commit this
**assistant**: running the commit
[tool: Bash — git commit -s -m "fix"]
[tool: Bash — uv run pytest]
**assistant**: done
"""


def _belief(pid: str, content: str) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="e1", snapshot_id="s1")
        ],
    )


class TestLiteralMatching:
    def test_extract_action_lines_only_tool_calls(self) -> None:
        lines = extract_action_lines(_TRANSCRIPT)
        assert len(lines) == 2
        assert all(line.startswith("[tool:") for line in lines)

    def test_backtick_token_extraction(self) -> None:
        toks = literal_tokens("Every commit needs `git commit -s`")
        assert "git commit -s" in toks

    def test_literal_match_credits_followed_command(self) -> None:
        actives = [
            _belief("p-commit", "Every commit needs `git commit -s`"),
            _belief("p-unrelated", "The sky is `blue`"),
        ]
        matched = match_literal(actives, extract_action_lines(_TRANSCRIPT))
        assert "p-commit" in matched
        assert "p-unrelated" not in matched

    def test_credits_command_with_interposed_args(self) -> None:
        # Real invocations interpose flags between the belief's tokens, so a
        # contiguous-substring test silently under-credits a belief that was
        # applied correctly. Traced from the projection-diff report
        # (2026-07-19): `git commit -s` never climbed despite constant use.
        actives = [_belief("p-commit", "Every commit needs `git commit -s`")]
        transcript = "[tool: Bash — uv run git -C /repo commit -s -F -]\n"
        matched = match_literal(actives, extract_action_lines(transcript))
        assert matched.get("p-commit") == "git commit -s"

    def test_token_boundary_prevents_partial_flag_match(self) -> None:
        # `-s` must not be satisfied by `-short`: ordered matching stays tight.
        actives = [_belief("p-commit", "Every commit needs `git commit -s`")]
        transcript = "[tool: Bash — git commit -short]\n"
        assert match_literal(actives, extract_action_lines(transcript)) == {}

    def test_negation_belief_skipped_by_literal(self) -> None:
        # "never prepend `export PATH`" — a token match would be a VIOLATION, not
        # compliance, so the literal tier skips it (routed to behavioural).
        actives = [_belief("p-neg", "Never prepend `export PATH` to a command")]
        transcript = "[tool: Bash — export PATH=/x:$PATH && make]"
        matched = match_literal(actives, extract_action_lines(transcript))
        assert matched == {}

    def test_no_action_lines_no_matches(self) -> None:
        actives = [_belief("p-commit", "Every commit needs `git commit -s`")]
        assert match_literal(actives, []) == {}

    def test_action_not_attention(self) -> None:
        # The command appears only in PROSE, never a tool call → not credited.
        actives = [_belief("p-commit", "Every commit needs `git commit -s`")]
        prose_only = "**assistant**: you should run git commit -s next time\n"
        assert match_literal(actives, extract_action_lines(prose_only)) == {}


class TestSessionIdFromUri:
    def test_extracts_session(self) -> None:
        assert session_id_from_uri("claude-code://session/abc-123") == "abc-123"

    def test_none_for_other_uri(self) -> None:
        assert session_id_from_uri("https://example.com") is None
        assert session_id_from_uri(None) is None


@pytest.mark.asyncio
async def test_behavioural_matcher_degrades_on_provider_error(db_session: object) -> None:
    # A raw provider error (e.g. exhausted credit balance, a 4xx/5xx SDK error —
    # not just CompletionError) must degrade to literal-only, never crash mining.
    import particles.operations.utility_mining as um

    async def boom(*a: object, **k: object) -> str:
        raise RuntimeError("credit balance too low")  # a raw SDK-style error

    orig = um.complete
    um.complete = boom  # type: ignore[assignment]
    try:
        get_config().utility.mining.behavioural_matching = True
        actives = [_belief("p-soft", "Prefer general mechanisms over per-genre defaults")]
        # A guideline with no literal token → only the behavioural tier could match it.
        result = await mine_session(db_session, "sess-e", _TRANSCRIPT, actives)  # type: ignore[arg-type]
        assert result.literal == 0
        assert result.behavioural == 0
        assert result.behavioural_calls == 0  # the failed call is not counted
    finally:
        um.complete = orig  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_mine_session_records_literal_events(db_session: object) -> None:
    # Behavioural tier off → deterministic literal-only, no LLM needed. The
    # autouse fixture resets config before the next test (reset_config() mid-test
    # would dispose the db_session engine).
    get_config().utility.mining.behavioural_matching = False
    actives = [
        _belief("p-commit", "Every commit needs `git commit -s`"),
        _belief("p-none", "Unrelated trivia about `xyzzy-plugh`"),
    ]
    result = await mine_session(db_session, "sess-1", _TRANSCRIPT, actives)  # type: ignore[arg-type]
    assert result.literal == 1
    assert result.behavioural == 0
    assert result.candidates == 2

    from particles.store.utility_store import get_reinforcement_scores

    scores = await get_reinforcement_scores(
        db_session,  # type: ignore[arg-type]
        ["p-commit", "p-none"],
        30.0,
        now=datetime.now(UTC),
    )
    assert "p-commit" in scores
    assert "p-none" not in scores


@pytest.mark.asyncio
async def test_behavioural_matcher_threads_response_schema(db_session: object) -> None:
    # the matcher passes its verdict array schema through the port.
    import particles.operations.utility_mining as um

    captured: dict[str, object] = {}

    async def fake_complete(*a: object, **k: object) -> str:
        captured.update(k)
        return "[1]"

    orig = um.complete
    um.complete = fake_complete  # type: ignore[assignment]
    try:
        get_config().utility.mining.behavioural_matching = True
        actives = [_belief("p-soft", "Prefer general mechanisms over per-genre defaults")]
        result = await mine_session(db_session, "sess-s", _TRANSCRIPT, actives)  # type: ignore[arg-type]
        assert result.behavioural == 1
        assert captured["response_schema"] == {"type": "array", "items": {"type": "integer"}}
    finally:
        um.complete = orig  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_budget_override_caps_calls_and_reports_truncation(db_session: object) -> None:
    # correction (v1.74.1): a multi-session caller threads its
    # remaining per-run budget via max_behavioural_calls; the config cap is
    # NOT the binding one, and stopping short reports behavioural_truncated.
    import particles.operations.utility_mining as um

    calls_made = 0

    async def fake_complete(*a: object, **k: object) -> str:
        nonlocal calls_made
        calls_made += 1
        return "[]"

    orig = um.complete
    orig_batch = um._BEHAVIOURAL_BATCH
    um.complete = fake_complete  # type: ignore[assignment]
    um._BEHAVIOURAL_BATCH = 1  # one belief per call → two unmatched want 2 calls
    try:
        get_config().utility.mining.behavioural_matching = True
        get_config().utility.mining.max_behavioural_calls = 50
        actives = [
            _belief("p-soft1", "Prefer general mechanisms over per-genre defaults"),
            _belief("p-soft2", "Keep the surface thin over the operation"),
        ]
        result = await mine_session(
            db_session,  # type: ignore[arg-type]
            "sess-budget",
            _TRANSCRIPT,
            actives,
            max_behavioural_calls=1,
        )
        assert calls_made == 1
        assert result.behavioural_calls == 1
        assert result.behavioural_truncated is True
    finally:
        um.complete = orig  # type: ignore[assignment]
        um._BEHAVIOURAL_BATCH = orig_batch


@pytest.mark.asyncio
async def test_zero_budget_makes_no_calls_but_literal_tier_still_mines(
    db_session: object,
) -> None:
    # Budget exhausted upstream → zero behavioural calls, truncation flagged,
    # and the LLM-free literal tier keeps mining (correction v1.74.1).
    import particles.operations.utility_mining as um

    async def boom(*a: object, **k: object) -> str:
        raise AssertionError("behavioural matcher must not be called on a zero budget")

    orig = um.complete
    um.complete = boom  # type: ignore[assignment]
    try:
        get_config().utility.mining.behavioural_matching = True
        actives = [
            _belief("p-commit", "Every commit needs `git commit -s`"),
            _belief("p-soft", "Prefer general mechanisms over per-genre defaults"),
        ]
        result = await mine_session(
            db_session,  # type: ignore[arg-type]
            "sess-zero",
            _TRANSCRIPT,
            actives,
            max_behavioural_calls=0,
        )
        assert result.behavioural_calls == 0
        assert result.behavioural_truncated is True  # judgement was wanted
        assert result.literal == 1  # the LLM-free tier still ran
    finally:
        um.complete = orig  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_behavioural_matching_override_forces_literal_only(db_session: object) -> None:
    # a degraded consolidation pass forces the literal tier even
    # when the config knob is on — no LLM call may be attempted.
    import particles.operations.utility_mining as um

    async def boom(*a: object, **k: object) -> str:
        raise AssertionError("behavioural matcher must not be called")

    orig = um.complete
    um.complete = boom  # type: ignore[assignment]
    try:
        get_config().utility.mining.behavioural_matching = True
        actives = [_belief("p-soft", "Prefer general mechanisms over per-genre defaults")]
        result = await mine_session(
            db_session,  # type: ignore[arg-type]
            "sess-o",
            _TRANSCRIPT,
            actives,
            behavioural_matching=False,
        )
        assert result.behavioural == 0
        assert result.behavioural_calls == 0
    finally:
        um.complete = orig  # type: ignore[assignment]


class _FakeEmbeddingModel:
    """Deterministic encoder: every text embeds to the unit x-axis vector."""

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> list[object]:
        import numpy as np

        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]


class TestBehaviouralPrefilter:
    """The relevance pre-filter for the behavioural tier's candidate set."""

    @pytest.mark.asyncio
    async def test_keeps_most_action_similar_beliefs(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        import particles.operations.utility_mining as um
        from particles.operations.utility_mining import _prefilter_behavioural_candidates

        monkeypatch.setattr("particles.embeddings.get_embedding_model", _FakeEmbeddingModel)

        beliefs = [_belief("p-a", "A"), _belief("p-b", "B"), _belief("p-c", "C")]
        vecs = {
            "p-a": np.array([0.0, 1.0], dtype=np.float32),  # orthogonal to actions
            "p-b": np.array([1.0, 0.0], dtype=np.float32),  # aligned with actions
        }

        async def fake_embeddings(*a: object, **k: object) -> list[object]:
            return [(b, vecs[b.id]) for b in beliefs if b.id in vecs]

        monkeypatch.setattr(um, "get_active_particles_with_embeddings", fake_embeddings)

        kept, excluded, ranked = await _prefilter_behavioural_candidates(
            db_session,  # type: ignore[arg-type]
            beliefs,
            ["[tool: Bash — x]"],
            1,
        )
        assert [p.id for p in kept] == ["p-b"]
        assert excluded == 2
        assert ranked is True

    @pytest.mark.asyncio
    async def test_beliefs_without_stored_embeddings_rank_last(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        import particles.operations.utility_mining as um
        from particles.operations.utility_mining import _prefilter_behavioural_candidates

        monkeypatch.setattr("particles.embeddings.get_embedding_model", _FakeEmbeddingModel)
        beliefs = [_belief("p-none", "no embedding"), _belief("p-vec", "embedded")]

        async def fake_embeddings(*a: object, **k: object) -> list[object]:
            return [(beliefs[1], np.array([1.0, 0.0], dtype=np.float32))]

        monkeypatch.setattr(um, "get_active_particles_with_embeddings", fake_embeddings)

        kept, excluded, ranked = await _prefilter_behavioural_candidates(
            db_session,  # type: ignore[arg-type]
            beliefs,
            ["[tool: Bash — x]"],
            1,
        )
        assert [p.id for p in kept] == ["p-vec"]
        assert excluded == 1
        assert ranked is True

    @pytest.mark.asyncio
    async def test_disabled_or_small_set_passes_through(self, db_session: object) -> None:
        from particles.operations.utility_mining import _prefilter_behavioural_candidates

        beliefs = [_belief("p-a", "A"), _belief("p-b", "B")]
        # limit 0 disables the filter entirely
        kept, excluded, ranked = await _prefilter_behavioural_candidates(
            db_session,  # type: ignore[arg-type]
            beliefs,
            ["[tool: x]"],
            0,
        )
        assert kept == beliefs and excluded == 0 and ranked is True
        # candidate set already within the limit
        kept, excluded, ranked = await _prefilter_behavioural_candidates(
            db_session,  # type: ignore[arg-type]
            beliefs,
            ["[tool: x]"],
            2,
        )
        assert kept == beliefs and excluded == 0 and ranked is True

    @pytest.mark.asyncio
    async def test_no_model_degrades_to_first_n(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.operations.utility_mining import _prefilter_behavioural_candidates

        monkeypatch.setattr("particles.embeddings.get_embedding_model", lambda: None)
        beliefs = [_belief("p-a", "A"), _belief("p-b", "B"), _belief("p-c", "C")]
        kept, excluded, ranked = await _prefilter_behavioural_candidates(
            db_session,  # type: ignore[arg-type]
            beliefs,
            ["[tool: x]"],
            2,
        )
        assert [p.id for p in kept] == ["p-a", "p-b"]
        assert excluded == 1
        # the cut was arbitrary list order, and the third element is
        # how the caller learns not to claim "by action similarity".
        assert ranked is False

    @pytest.mark.asyncio
    async def test_no_model_log_does_not_claim_a_similarity_ranking(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """the count was always honest; the basis was not.

        The old line said the pre-filter "kept top N ... by action similarity"
        whether or not any similarity was computed, so an encoder-free run
        misreported how the LLM budget was spent.
        """
        import particles.operations.utility_mining as um

        async def fake_complete(*a: object, **k: object) -> str:
            return "[]"

        monkeypatch.setattr(um, "complete", fake_complete)
        monkeypatch.setattr("particles.embeddings.get_embedding_model", lambda: None)
        get_config().utility.mining.behavioural_matching = True
        get_config().utility.mining.behavioural_candidate_limit = 1

        actives = [
            _belief("p-far", "Prefer general mechanisms over per-genre defaults"),
            _belief("p-near", "Keep the surface thin over the operation"),
        ]
        with caplog.at_level("INFO", logger="particles.operations.utility_mining"):
            result = await mine_session(
                db_session,  # type: ignore[arg-type]
                "sess-nomodel",
                _TRANSCRIPT,
                actives,
            )

        assert result.behavioural_prefiltered == 1
        assert result.behavioural_prefilter_ranked is False
        assert "by action similarity" not in caplog.text
        assert "arbitrary order" in caplog.text

    @pytest.mark.asyncio
    async def test_mine_session_discloses_prefiltered_count(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        import particles.operations.utility_mining as um

        async def fake_complete(*a: object, **k: object) -> str:
            return "[1]"

        monkeypatch.setattr(um, "complete", fake_complete)
        monkeypatch.setattr("particles.embeddings.get_embedding_model", _FakeEmbeddingModel)

        actives = [
            _belief("p-far", "Prefer general mechanisms over per-genre defaults"),
            _belief("p-near", "Keep the surface thin over the operation"),
        ]
        vecs = {"p-near": np.array([1.0, 0.0], dtype=np.float32)}

        async def fake_embeddings(*a: object, **k: object) -> list[object]:
            return [(b, vecs[b.id]) for b in actives if b.id in vecs]

        monkeypatch.setattr(um, "get_active_particles_with_embeddings", fake_embeddings)
        get_config().utility.mining.behavioural_matching = True
        get_config().utility.mining.behavioural_candidate_limit = 1

        result = await mine_session(
            db_session,  # type: ignore[arg-type]
            "sess-pf",
            _TRANSCRIPT,
            actives,
        )
        assert result.behavioural_prefiltered == 1
        assert result.behavioural == 1  # only the similarity-ranked survivor was judged
        assert result.behavioural_calls == 1


@pytest.mark.asyncio
async def test_behavioural_matcher_batches_when_latency_tolerant(db_session: object) -> None:
    """an unattended mine sends the matcher's groups as one batch.

    ``_BEHAVIOURAL_BATCH`` is 15, so 32 unmatched beliefs make three groups —
    three sequential calls before, one ``complete_many`` submission now, with
    the same credited beliefs either way.
    """
    import particles.operations.utility_mining as um

    captured: dict[str, object] = {}

    async def fake_complete_many(
        purpose: str, requests: list[object], **kwargs: object
    ) -> list[str | None]:
        captured["purpose"] = purpose
        captured["groups"] = len(requests)
        captured.update(kwargs)
        # Credit guideline 1 of every group.
        return ["[1]"] * len(requests)

    async def boom(*a: object, **k: object) -> str:
        raise AssertionError("sequential complete() must not run when batching")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        monkeypatch.setattr(um, "complete", boom)
        get_config().utility.mining.behavioural_matching = True
        get_config().utility.mining.behavioural_candidate_limit = 0  # no pre-filter
        actives = [
            _belief(f"p-soft-{i}", f"Prefer general mechanism number {i}") for i in range(32)
        ]
        result = await mine_session(
            db_session,  # type: ignore[arg-type]
            "sess-batch",
            _TRANSCRIPT,
            actives,
            latency_tolerant=True,
        )
    finally:
        monkeypatch.undo()

    assert captured["groups"] == 3  # ceil(32 / 15)
    assert captured["latency_tolerant"] is True
    assert captured["purpose"] == "semantic_lint"
    assert result.behavioural_calls == 3
    assert result.behavioural == 3  # guideline 1 of each of the three groups


@pytest.mark.asyncio
async def test_behavioural_batch_dead_request_does_not_spend_budget(db_session: object) -> None:
    """A request the batch could not answer costs nothing usable, so it costs no budget."""

    async def fake_complete_many(*a: object, **k: object) -> list[str | None]:
        return ["[1]", None, None]

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("particles.llm.complete_many", fake_complete_many)
        get_config().utility.mining.behavioural_matching = True
        get_config().utility.mining.behavioural_candidate_limit = 0
        actives = [
            _belief(f"p-soft-{i}", f"Prefer general mechanism number {i}") for i in range(32)
        ]
        result = await mine_session(
            db_session,  # type: ignore[arg-type]
            "sess-batch-partial",
            _TRANSCRIPT,
            actives,
            latency_tolerant=True,
        )
    finally:
        monkeypatch.undo()

    assert result.behavioural_calls == 1  # only the answered group drew down the budget
    assert result.behavioural == 1
