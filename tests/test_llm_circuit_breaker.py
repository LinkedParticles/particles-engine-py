# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Circuit-breaker on the LLM seam for account-level failures.

An account-level failure (bad/missing key, no permission, out-of-credits) fails
*every* call, so `_llm_call` trips a process-wide breaker: subsequent calls
short-circuit to ``None`` without touching the API until the cool-off, and the
condition is logged once. A per-call/transient failure must NOT trip. The seam's
``_tripped_until`` is module-global, so this file resets it around every test.
"""

from __future__ import annotations

import pytest

import particles.operations._llm as seam


class _FakeStatusError(Exception):
    """Mimics a provider SDK error: carries ``status_code`` like anthropic's."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _reset_breaker() -> None:
    seam._tripped_until = 0.0
    yield
    seam._tripped_until = 0.0


async def test_out_of_credits_trips_breaker_and_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_complete(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        calls.append(prompt)
        raise _FakeStatusError("Your credit balance is too low to access the Anthropic API.", 400)

    monkeypatch.setattr("particles.llm.complete", fake_complete)

    # First call hits the API, fails account-level → trips the breaker.
    assert await seam._llm_call("first") is None
    assert seam.llm_circuit_open() is True

    # Second call short-circuits: no further API call while the breaker is open.
    assert await seam._llm_call("second") is None
    assert calls == ["first"]


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_and_permission_errors_trip(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    async def fake_complete(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        raise _FakeStatusError("nope", status)

    monkeypatch.setattr("particles.llm.complete", fake_complete)
    assert await seam._llm_call("x") is None
    assert seam.llm_circuit_open() is True


async def test_per_call_error_does_not_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_complete(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        calls.append(prompt)
        raise RuntimeError("transient blip")  # no status_code → per-call

    monkeypatch.setattr("particles.llm.complete", fake_complete)

    assert await seam._llm_call("a") is None
    assert seam.llm_circuit_open() is False
    # Not tripped → the next call still reaches the API.
    assert await seam._llm_call("b") is None
    assert calls == ["a", "b"]


async def test_generic_400_is_per_call_not_account_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_complete(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        raise _FakeStatusError("invalid request: prompt too long", 400)

    monkeypatch.setattr("particles.llm.complete", fake_complete)
    assert await seam._llm_call("x") is None
    # A 400 without a billing message is a one-off bad request, not account-level.
    assert seam.llm_circuit_open() is False


async def test_success_clears_a_stale_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ok(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        return "ANSWER"

    monkeypatch.setattr("particles.llm.complete", fake_ok)

    # Closed/half-open (deadline in the past) → the probe proceeds, succeeds, and
    # the breaker stays closed.
    assert await seam._llm_call("probe") == "ANSWER"
    assert seam.llm_circuit_open() is False


async def test_trip_is_counted_with_its_cause_class(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        raise _FakeStatusError("Your credit balance is too low to access the Anthropic API.", 400)

    monkeypatch.setattr("particles.llm.complete", fake_complete)
    before = seam.llm_trip_count()
    assert await seam._llm_call("first") is None
    assert await seam._llm_call("second") is None  # short-circuited: no second trip
    assert seam.llm_trip_count() == before + 1
    assert seam.llm_unavailable_cause() == "credit balance too low"


@pytest.mark.parametrize(
    ("message", "status", "cause"),
    [
        ("Your credit balance is too low.", 400, "credit balance too low"),
        ("Billing hard limit reached.", 400, "billing error"),
        ("invalid x-api-key", 401, "API key rejected"),
        ("forbidden", 403, "API key lacks permission"),
    ],
)
def test_account_level_cause_class(message: str, status: int, cause: str) -> None:
    from particles.llm.errors import describe_account_level_failure

    assert describe_account_level_failure(_FakeStatusError(message, status)) == cause


async def test_empty_reply_as_text_returns_empty_string_uncounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opted in, a no-text reply (budget spent on thinking) is ``""``, not a failed call."""
    from particles.llm import EmptyCompletionError

    async def fake_complete(purpose: str, prompt: str, max_tokens: int = 200, **_: object) -> str:
        raise EmptyCompletionError("no text block (stop_reason=max_tokens)")

    monkeypatch.setattr("particles.llm.complete", fake_complete)
    before = seam.llm_failure_count()

    assert await seam._llm_call("x", empty_reply_as_text=True) == ""
    assert seam.llm_failure_count() == before
    # Not opted in: the historical contract, a counted failure returning None.
    assert await seam._llm_call("x") is None
    assert seam.llm_failure_count() == before + 1
    assert seam.llm_circuit_open() is False
