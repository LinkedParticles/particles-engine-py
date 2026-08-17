"""Tests for the batch-completion surface of ``particles/llm``.

Covers the port (``complete_many`` and its sequential fallback), the Anthropic
Message Batches adapter, and the operations-layer ``_llm_call_many`` seam. The
Anthropic SDK is mocked through the standard ``particles.llm.set_client`` seam
(tests/AGENTS.md § Mocking strategy) — no network, no API key.

The invariant every test here defends: **batching is a price optimisation that
cannot change what a call site computes.** Same prompts, same parsing, same
results in the same order — only the billing and the wall clock differ.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest

from particles import llm
from particles.config import reset_config
from particles.llm import CompletionRequest


@pytest.fixture(autouse=True)
def _reset_client_around_each_test() -> Generator[None, None, None]:
    llm.set_client(None)
    yield
    llm.set_client(None)


def _batch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **knobs: object) -> None:
    """Point ``PARTICLES_CONFIG`` at an ``llm.batch`` block built from ``knobs``.

    The batch knobs have no env-var override (registers those only for
    backward compatibility), so a file is the supported way to set them.
    """
    body = "\n".join(f"    {key}: {value!r}".replace("'", "") for key, value in knobs.items())
    config = tmp_path / "config.yaml"
    config.write_text(f"llm:\n  batch:\n{body}\n" if knobs else "llm:\n  batch: {}\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()


def _text_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason="end_turn")


def _succeeded(custom_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=_text_message(text)),
    )


def _batches_client(
    results: list[SimpleNamespace],
    *,
    statuses: list[str] | None = None,
) -> MagicMock:
    """A mocked Anthropic client whose ``messages.batches`` returns ``results``.

    ``statuses`` is the sequence ``processing_status`` cycles through; it
    defaults to an immediately-finished batch.
    """
    client = MagicMock(spec=anthropic.Anthropic)
    pending = list(statuses or ["ended"])

    def _retrieve(_batch_id: str) -> SimpleNamespace:
        status = pending.pop(0) if len(pending) > 1 else pending[0]
        return SimpleNamespace(processing_status=status)

    client.messages.batches.create.return_value = SimpleNamespace(id="msgbatch_test")
    client.messages.batches.retrieve.side_effect = _retrieve
    client.messages.batches.results.return_value = iter(results)
    return client


def _sequential_client(texts: list[str]) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [_text_message(t) for t in texts]
    return client


_REQUESTS = [CompletionRequest(prompt=f"probe {i}", system=f"sys {i}") for i in range(4)]


# ---------------------------------------------------------------------------
# complete_many — routing between the batch and sequential paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_many_empty_returns_empty_without_a_provider_call() -> None:
    """No requests is not a batch of zero — it is no work at all."""
    client = MagicMock(spec=anthropic.Anthropic)
    llm.set_client(client)
    assert await llm.complete_many("semantic_lint", [], max_tokens=10, latency_tolerant=True) == []
    client.messages.batches.create.assert_not_called()


@pytest.mark.asyncio
async def test_complete_many_batches_when_latency_tolerant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A latency-tolerant caller gets ONE batch submission, not N calls."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    client = _batches_client([_succeeded(str(i), f"reply {i}") for i in range(4)])
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.batches.create.call_count == 1
    client.messages.create.assert_not_called()
    # Each request keeps its own system turn — the F3 per-probe fence nonce
    # cannot be shared across a batch.
    submitted = client.messages.batches.create.call_args.kwargs["requests"]
    assert [r["params"]["system"] for r in submitted] == ["sys 0", "sys 1", "sys 2", "sys 3"]
    assert [r["custom_id"] for r in submitted] == ["0", "1", "2", "3"]


@pytest.mark.asyncio
async def test_complete_many_realigns_out_of_order_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch results arrive in ANY order; they are keyed back by custom_id.

    The whole batch path would silently scramble every verdict onto the wrong
    candidate pair if this mapping were positional.
    """
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    shuffled = [_succeeded(cid, f"reply {cid}") for cid in ("2", "0", "3", "1")]
    llm.set_client(_batches_client(shuffled))

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]


@pytest.mark.asyncio
async def test_complete_many_stays_sequential_when_not_latency_tolerant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the caller's opt-in, nothing is batched however cheap it would be."""
    _batch_config(tmp_path, monkeypatch, min_requests=1)
    client = _sequential_client([f"reply {i}" for i in range(4)])
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.create.call_count == 4
    client.messages.batches.create.assert_not_called()


@pytest.mark.asyncio
async def test_complete_many_honours_the_enabled_kill_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``llm.batch.enabled: false`` restores the pre-ADR-0218 behaviour exactly."""
    _batch_config(tmp_path, monkeypatch, enabled=False, min_requests=1)
    client = _sequential_client([f"reply {i}" for i in range(4)])
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    client.messages.batches.create.assert_not_called()


@pytest.mark.asyncio
async def test_complete_many_below_min_requests_stays_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handful of probes is not worth a submit-and-poll round trip."""
    _batch_config(tmp_path, monkeypatch, min_requests=10)
    client = _sequential_client([f"reply {i}" for i in range(4)])
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    client.messages.batches.create.assert_not_called()


@pytest.mark.asyncio
async def test_complete_many_falls_back_when_submission_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that cannot batch loses the discount, never the capability."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    client = _sequential_client([f"reply {i}" for i in range(4)])
    client.messages.batches.create.side_effect = RuntimeError("batches unavailable")
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.create.call_count == 4


@pytest.mark.asyncio
async def test_complete_many_sequential_returns_none_for_one_bad_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failed prompt must not discard its siblings' answers."""
    _batch_config(tmp_path, monkeypatch, enabled=False)
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [
        _text_message("reply 0"),
        RuntimeError("one bad prompt"),
        _text_message("reply 2"),
        _text_message("reply 3"),
    ]
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10)

    assert out == ["reply 0", None, "reply 2", "reply 3"]


@pytest.mark.asyncio
async def test_complete_many_sequential_reraises_account_level_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired key fails every remaining call — raise once, don't log N times."""
    _batch_config(tmp_path, monkeypatch, enabled=False)
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid api key",
        response=MagicMock(status_code=401, headers={}),
        body=None,
    )
    llm.set_client(client)

    with pytest.raises(anthropic.AuthenticationError):
        await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10)
    assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# The Anthropic adapter — per-request failure, timeout, chunking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_reports_dead_requests_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """errored / expired / text-less requests degrade individually, not as a set."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    results = [
        _succeeded("0", "reply 0"),
        SimpleNamespace(custom_id="1", result=SimpleNamespace(type="errored")),
        SimpleNamespace(custom_id="2", result=SimpleNamespace(type="expired")),
        SimpleNamespace(
            custom_id="3",
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(content=[], stop_reason="refusal"),
            ),
        ),
    ]
    llm.set_client(_batches_client(results))

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", None, None, None]


@pytest.mark.asyncio
async def test_batch_truncated_request_warns_with_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A batch reply that hit the ceiling names the budget, not just the request.

    Truncation reaches the call site either as unparseable text or as no text
    block at all; both read as a broken model unless the budget is named.
    """
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    truncated = SimpleNamespace(
        custom_id="1",
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(text='[{"claim": "half a jso')],
                stop_reason="max_tokens",
            ),
        ),
    )
    results = [_succeeded("0", "reply 0"), truncated, _succeeded("2", "r2"), _succeeded("3", "r3")]
    llm.set_client(_batches_client(results))

    with caplog.at_level("WARNING", logger="particles.llm.adapters.anthropic"):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=4096, latency_tolerant=True
        )

    # The truncated text is still returned — the call site's parser is the
    # backstop, as before; what changes is that the cause is now legible.
    assert out == ["reply 0", '[{"claim": "half a jso', "r2", "r3"]
    warnings = [r.getMessage() for r in caplog.records if "truncated" in r.getMessage()]
    assert len(warnings) == 1
    assert "max_tokens=4096" in warnings[0]
    assert "request 1" in warnings[0]


@pytest.mark.asyncio
async def test_batch_missing_result_becomes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short result set is padded positionally, never silently truncated."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    llm.set_client(_batches_client([_succeeded("0", "reply 0"), _succeeded("2", "reply 2")]))

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", None, "reply 2", None]


@pytest.mark.asyncio
async def test_batch_that_outlives_its_deadline_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stuck batch must not hold the nightly run open until the API's 24h expiry."""
    _batch_config(
        tmp_path,
        monkeypatch,
        min_requests=2,
        poll_interval_seconds=0.01,
        max_wait_seconds=0.02,
    )
    client = _batches_client([], statuses=["in_progress"])
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == [None, None, None, None]
    client.messages.batches.cancel.assert_called_once_with("msgbatch_test")
    client.messages.batches.results.assert_not_called()


@pytest.mark.asyncio
async def test_batch_polls_until_ended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch still processing is polled, not abandoned."""
    _batch_config(tmp_path, monkeypatch, min_requests=2, poll_interval_seconds=0.01)
    client = _batches_client(
        [_succeeded(str(i), f"reply {i}") for i in range(4)],
        statuses=["in_progress", "in_progress", "ended"],
    )
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.batches.retrieve.call_count == 3


@pytest.mark.asyncio
async def test_batch_chunks_above_max_requests_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests beyond the per-batch ceiling become further batches, in order."""
    _batch_config(tmp_path, monkeypatch, min_requests=2, max_requests_per_batch=2)
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.batches.create.side_effect = [
        SimpleNamespace(id="msgbatch_a"),
        SimpleNamespace(id="msgbatch_b"),
    ]
    client.messages.batches.retrieve.return_value = SimpleNamespace(processing_status="ended")
    client.messages.batches.results.side_effect = [
        iter([_succeeded("0", "reply 0"), _succeeded("1", "reply 1")]),
        iter([_succeeded("0", "reply 2"), _succeeded("1", "reply 3")]),
    ]
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.batches.create.call_count == 2


# ---------------------------------------------------------------------------
# The operations seam — _llm_call_many and the breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_call_many_short_circuits_an_open_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open circuit breaker suppresses the whole set without touching the API."""
    from particles.operations import _llm as ops_llm

    _batch_config(tmp_path, monkeypatch, min_requests=2)
    client = MagicMock(spec=anthropic.Anthropic)
    llm.set_client(client)
    monkeypatch.setattr(ops_llm, "_tripped_until", ops_llm.time.monotonic() + 60)

    out = await ops_llm._llm_call_many(_REQUESTS, latency_tolerant=True)

    assert out == [None, None, None, None]
    client.messages.batches.create.assert_not_called()
    client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_llm_call_many_trips_the_breaker_on_account_level_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad key raised from the set trips the breaker once, as a single call would."""
    from particles.operations import _llm as ops_llm

    _batch_config(tmp_path, monkeypatch, enabled=False)
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid api key",
        response=MagicMock(status_code=401, headers={}),
        body=None,
    )
    llm.set_client(client)

    out = await ops_llm._llm_call_many(_REQUESTS, latency_tolerant=True)

    assert out == [None, None, None, None]
    assert ops_llm.llm_circuit_open() is True


@pytest.mark.asyncio
async def test_llm_call_many_counts_dead_requests_as_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unanswered probes feed the disclosure counter, so they cannot silently vanish."""
    from particles.operations import _llm as ops_llm

    _batch_config(tmp_path, monkeypatch, min_requests=2)
    results: list[Any] = [
        _succeeded("0", "YES: conflict"),
        SimpleNamespace(custom_id="1", result=SimpleNamespace(type="errored")),
        SimpleNamespace(custom_id="2", result=SimpleNamespace(type="errored")),
        _succeeded("3", "NO"),
    ]
    llm.set_client(_batches_client(results))
    before = ops_llm.llm_failure_count()

    out = await ops_llm._llm_call_many(_REQUESTS, latency_tolerant=True)

    assert out == ["YES: conflict", None, None, "NO"]
    assert ops_llm.llm_failure_count() - before == 2
    assert ops_llm.llm_circuit_open() is False


# ---------------------------------------------------------------------------
# complete_many_with_provider_model — the stamp seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_many_with_provider_model_reports_the_serving_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The many-request twin of complete_with_provider_model."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    llm.set_client(_batches_client([_succeeded(str(i), f"reply {i}") for i in range(4)]))

    out, pairing = await llm.complete_many_with_provider_model(
        "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
    )

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert pairing.startswith("anthropic:")


@pytest.mark.asyncio
async def test_complete_many_with_provider_model_empty_set_resolves_no_provider() -> None:
    out, pairing = await llm.complete_many_with_provider_model("semantic_lint", [], max_tokens=10)
    assert (out, pairing) == ([], "")


@pytest.mark.asyncio
async def test_batch_honours_a_request_cache_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a request's cache_prefix caches across batch entries too."""
    _batch_config(tmp_path, monkeypatch, min_requests=1)
    client = _batches_client([_succeeded("0", "r0")])
    llm.set_client(client)

    reqs = [CompletionRequest(prompt="p", system="VARIABLE", cache_prefix="FIXED RULES")]
    out = await llm.complete_many("extraction", reqs, max_tokens=8, latency_tolerant=True)

    assert out == ["r0"]
    submitted = client.messages.batches.create.call_args.kwargs["requests"]
    system = submitted[0]["params"]["system"]
    assert system[0]["text"] == "FIXED RULES"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[1]["text"] == "VARIABLE"
    assert "cache_control" not in system[1]
