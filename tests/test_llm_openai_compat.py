"""Tests for the generic OpenAI-compatible completion adapter (né 0172).

Covers the deterministic seams: request-body / header construction (including
the dialect knobs), response parsing, the bounded retry loop,
fail-fast on non-retryable status, the ``CompletionError`` contract, registry
resolution of named providers, the strict-dialect schema transform, and the breaker's duck-typing of a 401/403. The endpoint itself is never
hit — ``httpx.AsyncClient`` is mocked per ``tests/AGENTS.md``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from particles.llm.adapters import openai_compat as oc_mod
from particles.llm.adapters.openai_compat import (
    OpenAICompatCompletionError,
    OpenAICompatProvider,
    _extract_text,
    _post_with_retry,
    _scrub_response_body,
    _to_strict_schema,
)
from particles.llm.registry import CompletionError


def _ok_response(content: str, *, finish_reason: str | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    resp.json.return_value = {"choices": [choice]}
    resp.text = content
    return resp


def _status_response(status: int, body: str = "error") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    resp.json.return_value = {}
    return resp


def _mock_client(side_effect: Sequence[object]) -> MagicMock:
    """An async-context-manager mock whose ``.post`` yields ``side_effect`` in order.

    Each element is either a mock response (returned) or an ``Exception``
    instance (raised), letting one helper drive both the happy path and the
    transient-failure paths.
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(side_effect=list(side_effect))
    return client


def _install_client(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> None:
    monkeypatch.setattr(oc_mod.httpx, "AsyncClient", lambda **_kw: client)
    # Skip real backoff sleeps.
    monkeypatch.setattr(oc_mod.asyncio, "sleep", AsyncMock())


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "local",
    **entry_kwargs: object,
) -> None:
    """Install a config whose ``llm.providers[name]`` is built from ``entry_kwargs``."""
    from particles.config import (
        LLMConfig,
        OpenAICompatProviderConfig,
        ParticlesConfig,
    )

    entry_kwargs.setdefault("max_retries", 0)
    entry_kwargs.setdefault("retry_backoff_seconds", 0.0)
    cfg = ParticlesConfig(
        llm=LLMConfig(providers={name: OpenAICompatProviderConfig(**entry_kwargs)})  # type: ignore[arg-type]
    )
    monkeypatch.setattr("particles.config.get_config", lambda: cfg)
    monkeypatch.delenv("PARTICLES_LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv(f"PARTICLES_LLM_API_KEY_{name.upper()}", raising=False)


def test_provider_model_key_carries_the_provider_name() -> None:
    assert (
        OpenAICompatProvider(name="local", model="llama3.1:8b").provider_model
        == "local:llama3.1:8b"
    )
    assert (
        OpenAICompatProvider(name="openai", model="gpt-5.6-luna").provider_model
        == "openai:gpt-5.6-luna"
    )


def test_get_provider_resolves_named_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from particles.config import (
        LLMConfig,
        OpenAICompatProviderConfig,
        ParticlesConfig,
        ProviderSelection,
    )
    from particles.llm import get_provider

    cfg = ParticlesConfig(
        llm=LLMConfig(
            extraction=ProviderSelection(provider="openai", model="gpt-5.6-luna"),
            providers={"openai": OpenAICompatProviderConfig(base_url="https://api.openai.com/v1")},
        )
    )
    monkeypatch.setattr("particles.config.get_config", lambda: cfg)
    provider = get_provider("extraction")
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.name == "openai"
    assert provider.provider_model == "openai:gpt-5.6-luna"
    # An unrouted purpose still falls back to the native default.
    assert get_provider("synthesis").provider_model.startswith("anthropic:")


def test_get_provider_resolves_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """``provider: local`` keeps working with zero config (the contract)."""
    from particles.config import LLMConfig, ParticlesConfig, ProviderSelection
    from particles.llm import get_provider

    cfg = ParticlesConfig(
        llm=LLMConfig(extraction=ProviderSelection(provider="local", model="llama3.1:8b"))
    )
    monkeypatch.setattr("particles.config.get_config", lambda: cfg)
    provider = get_provider("extraction")
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.provider_model == "local:llama3.1:8b"


def test_local_provider_shim_is_the_named_local_entry() -> None:
    from particles.llm.adapters.local import LocalCompletionError, LocalProvider

    shim = LocalProvider(model="llama3.1:8b")
    assert isinstance(shim, OpenAICompatProvider)
    assert shim.provider_model == "local:llama3.1:8b"
    assert LocalCompletionError is OpenAICompatCompletionError


def test_complete_builds_request_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_client([_ok_response("hello from the model")])
    _install_client(monkeypatch, client)
    monkeypatch.delenv("PARTICLES_LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("PARTICLES_LLM_API_KEY_LOCAL", raising=False)

    out = asyncio.run(
        OpenAICompatProvider(name="local", model="llama3.1:8b").complete(
            "the prompt", max_tokens=64, system="be terse", temperature=0.0
        )
    )
    assert out == "hello from the model"

    kwargs = client.post.call_args.kwargs
    body = kwargs["json"]
    assert body["model"] == "llama3.1:8b"
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "the prompt"},
    ]
    # No key set → no Authorization header.
    assert "Authorization" not in kwargs["headers"]
    # base_url + /chat/completions, default Ollama endpoint.
    assert client.post.call_args.args[0].endswith("/chat/completions")


def test_unknown_provider_name_fails_legibly(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_client([_ok_response("never reached")])
    _install_client(monkeypatch, client)
    with pytest.raises(CompletionError, match="no llm.providers entry"):
        asyncio.run(OpenAICompatProvider(name="nope", model="m").complete("p", max_tokens=8))
    client.post.assert_not_called()


def test_complete_omits_optional_fields_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_client([_ok_response("ok")])
    _install_client(monkeypatch, client)
    asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
    body = client.post.call_args.kwargs["json"]
    assert "temperature" not in body  # unset → omitted
    # system unset → only the user message
    assert body["messages"] == [{"role": "user", "content": "p"}]


def test_complete_sends_auth_header_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _mock_client([_ok_response("ok")])
    _install_client(monkeypatch, client)
    monkeypatch.setenv("PARTICLES_LOCAL_LLM_API_KEY", "secret-token")
    asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"


def test_named_key_convention_reaches_the_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PARTICLES_LLM_API_KEY_<NAME>`` is the bearer for a named provider."""
    _configure(monkeypatch, name="openai", base_url="https://api.openai.com/v1")
    client = _mock_client([_ok_response("ok")])
    _install_client(monkeypatch, client)
    monkeypatch.setenv("PARTICLES_LLM_API_KEY_OPENAI", "sk-luna")
    asyncio.run(
        OpenAICompatProvider(name="openai", model="gpt-5.6-luna").complete("p", max_tokens=8)
    )
    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-luna"


# ---------------------------------------------------------------------------
# Dialect knobs
# ---------------------------------------------------------------------------


def test_max_tokens_param_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, name="openai", max_tokens_param="max_completion_tokens")
    client = _mock_client([_ok_response("ok")])
    _install_client(monkeypatch, client)
    asyncio.run(OpenAICompatProvider(name="openai", model="m").complete("p", max_tokens=64))
    body = client.post.call_args.kwargs["json"]
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body


def test_send_temperature_false_drops_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, name="openai", send_temperature=False)
    client = _mock_client([_ok_response("ok")])
    _install_client(monkeypatch, client)
    asyncio.run(
        OpenAICompatProvider(name="openai", model="m").complete("p", max_tokens=8, temperature=0.0)
    )
    assert "temperature" not in client.post.call_args.kwargs["json"]


# ---------------------------------------------------------------------------
# Retry / failure contract (unchanged)
# ---------------------------------------------------------------------------


def test_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_client([_status_response(503), _ok_response("recovered")])
    _install_client(monkeypatch, client)
    out = asyncio.run(
        _post_with_retry(
            "http://x/chat/completions",
            {"model": "m"},
            {},
            max_retries=2,
            backoff=0.0,
            timeout=1.0,
        )
    )
    assert out == "recovered"
    assert client.post.call_count == 2


def test_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_client([_status_response(503), _status_response(503), _status_response(503)])
    _install_client(monkeypatch, client)
    with pytest.raises(OpenAICompatCompletionError) as exc_info:
        asyncio.run(
            _post_with_retry(
                "http://x/chat/completions",
                {"model": "m"},
                {},
                max_retries=2,
                backoff=0.0,
                timeout=1.0,
            )
        )
    assert exc_info.value.status_code == 503
    assert client.post.call_count == 3  # initial + 2 retries


def test_fail_fast_on_non_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_client([_status_response(401, "unauthorized")])
    _install_client(monkeypatch, client)
    with pytest.raises(OpenAICompatCompletionError) as exc_info:
        asyncio.run(
            _post_with_retry(
                "http://x/chat/completions",
                {"model": "m"},
                {},
                max_retries=3,
                backoff=0.0,
                timeout=1.0,
            )
        )
    assert exc_info.value.status_code == 401
    assert client.post.call_count == 1  # no retry on a 4xx auth error


def test_connection_error_is_retried_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_client([httpx.ConnectError("refused"), httpx.ConnectError("refused")])
    _install_client(monkeypatch, client)
    with pytest.raises(OpenAICompatCompletionError) as exc_info:
        asyncio.run(
            _post_with_retry(
                "http://x/chat/completions",
                {"model": "m"},
                {},
                max_retries=1,
                backoff=0.0,
                timeout=1.0,
            )
        )
    # Connection errors have no HTTP status → None (breaker treats as transient).
    assert exc_info.value.status_code is None
    assert client.post.call_count == 2


def test_extract_text_raises_when_no_content() -> None:
    with pytest.raises(CompletionError):
        _extract_text({"choices": [{"message": {}}]})
    with pytest.raises(CompletionError):
        _extract_text({"choices": []})
    with pytest.raises(CompletionError):
        _extract_text({"choices": [{"message": {"content": "   "}}]})  # whitespace only


# ---------------------------------------------------------------------------
# Truncation — finish_reason == "length" (the budget-exhaustion masquerade)
# ---------------------------------------------------------------------------


def _payload(content: str, *, finish_reason: str | None = None) -> dict[str, object]:
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"choices": [choice]}


def test_truncated_reply_returns_its_text_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A budget-exhausted reply is still returned — but no longer silently.

    The text stops mid-token, so the call site's parser reports a malformed
    reply; without this warning the operator sees only "Unterminated string"
    and never learns the budget was the cause.
    """
    with caplog.at_level("WARNING", logger="particles.llm.adapters.openai_compat"):
        out = _extract_text(
            _payload('[{"claim": "half a jso', finish_reason="length"),
            provider_model="fireworks:kimi-k3",
            max_tokens=8192,
        )
    assert out == '[{"claim": "half a jso'
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "truncated" in message
    assert "fireworks:kimi-k3" in message
    assert "max_tokens=8192" in message
    assert "extraction.max_tokens" in message


def test_truncated_empty_reply_raises_a_budget_error() -> None:
    """Empty AND truncated is a budget problem, not a model failure — say so.

    A reasoning model can spend the whole budget on thinking and emit no
    answer at all; the old message ("carried no text content") read as a
    broken endpoint.
    """
    with pytest.raises(CompletionError) as exc_info:
        _extract_text(
            _payload("", finish_reason="length"),
            provider_model="fireworks:deepseek-v4-pro",
            max_tokens=8192,
        )
    message = str(exc_info.value)
    assert "truncated" in message
    assert "fireworks:deepseek-v4-pro" in message
    assert "max_tokens=8192" in message
    assert "extraction.max_tokens" in message


def test_normal_finish_reason_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """finish_reason "stop" — and an endpoint that omits it — behave as before."""
    with caplog.at_level("WARNING", logger="particles.llm.adapters.openai_compat"):
        assert _extract_text(_payload("full reply", finish_reason="stop")) == "full reply"
        assert _extract_text(_payload("full reply")) == "full reply"
    assert caplog.records == []


def test_complete_warns_with_the_configured_pairing_and_budget(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """End to end: the warning names the provider entry and the call's budget."""
    _configure(monkeypatch, name="fireworks")
    _install_client(monkeypatch, _mock_client([_ok_response("cut off", finish_reason="length")]))
    provider = OpenAICompatProvider(name="fireworks", model="kimi-k3")

    with caplog.at_level("WARNING", logger="particles.llm.adapters.openai_compat"):
        out = asyncio.run(provider.complete("prompt", max_tokens=16384))

    assert out == "cut off"
    message = caplog.records[0].getMessage()
    assert "fireworks:kimi-k3" in message
    assert "max_tokens=16384" in message


def test_scrub_response_body_redacts_bearer_token() -> None:
    """F33: a bearer token echoed in a gateway body must never reach the message."""
    body = 'Bad gateway. Got headers: {"Authorization": "Bearer SUPER-SECRET-TOKEN"}'
    scrubbed = _scrub_response_body(body)
    assert "SUPER-SECRET-TOKEN" not in scrubbed
    assert "[REDACTED]" in scrubbed
    # A bare "Bearer <tok>" (no Authorization: prefix) is scrubbed too.
    assert "SECRET2" not in _scrub_response_body("error: bearer SECRET2")
    # Truncation still applies (cap at 500 chars).
    assert len(_scrub_response_body("x" * 1000)) == 500


def test_error_does_not_surface_echoed_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """F33: a gateway echoing the request bearer must not leak it via the error.

    A misbehaving gateway returns a 401 whose body echoes the request's
    ``Authorization: Bearer SECRET`` header; the raised exception text must not
    contain the secret.
    """
    leaky = _status_response(401, "Unauthorized. Authorization: Bearer SECRET-LEAKED-TOKEN")
    client = _mock_client([leaky])
    _install_client(monkeypatch, client)
    with pytest.raises(OpenAICompatCompletionError) as exc_info:
        asyncio.run(
            _post_with_retry(
                "http://x/chat/completions",
                {"model": "m"},
                {},
                max_retries=2,
                backoff=0.0,
                timeout=1.0,
            )
        )
    assert "SECRET-LEAKED-TOKEN" not in str(exc_info.value)
    assert exc_info.value.status_code == 401


def test_breaker_duck_types_auth_failure() -> None:
    """A 401/403 must trip the account-level breaker."""
    from particles.operations._llm import _is_account_level

    assert _is_account_level(OpenAICompatCompletionError("nope", status_code=401)) is True
    assert _is_account_level(OpenAICompatCompletionError("nope", status_code=403)) is True
    # A connection failure (no status) is per-call, not account-level.
    assert _is_account_level(OpenAICompatCompletionError("down", status_code=None)) is False


# ---------------------------------------------------------------------------
# Structured output — strict dialect
# ---------------------------------------------------------------------------


_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"contradicts": {"type": "boolean"}},
    "required": ["contradicts"],
}
_ARRAY_SCHEMA = {"type": "array", "items": {"type": "integer"}}


class TestStructuredOutput:
    def test_object_schema_sent_as_json_schema_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure(monkeypatch)
        client = _mock_client([_ok_response('{"contradicts": false}')])
        _install_client(monkeypatch, client)
        out = asyncio.run(
            OpenAICompatProvider(name="local", model="m").complete(
                "p", max_tokens=8, response_schema=_OBJECT_SCHEMA
            )
        )
        assert out == '{"contradicts": false}'
        rf = client.post.call_args.kwargs["json"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == _OBJECT_SCHEMA

    def test_array_schema_is_wrapped_and_reply_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure(monkeypatch)
        client = _mock_client([_ok_response('{"items": [1, 4]}')])
        _install_client(monkeypatch, client)
        out = asyncio.run(
            OpenAICompatProvider(name="local", model="m").complete(
                "p", max_tokens=8, response_schema=_ARRAY_SCHEMA
            )
        )
        # Enforcement is invisible to the caller: the array parses as sent.
        assert json.loads(out) == [1, 4]
        schema = client.post.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert schema["properties"]["items"] == _ARRAY_SCHEMA
        assert schema["required"] == ["items"]

    def test_array_root_reply_passes_through_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An endpoint (e.g. Ollama) that honoured the array root anyway.
        _configure(monkeypatch)
        client = _mock_client([_ok_response("[2, 3]")])
        _install_client(monkeypatch, client)
        out = asyncio.run(
            OpenAICompatProvider(name="local", model="m").complete(
                "p", max_tokens=8, response_schema=_ARRAY_SCHEMA
            )
        )
        assert json.loads(out) == [2, 3]

    def test_unsupported_response_format_downgrades_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure(monkeypatch)
        rejected = _status_response(400, "Unknown parameter: 'response_format'")
        client = _mock_client([rejected, _ok_response("plain text reply")])
        _install_client(monkeypatch, client)
        out = asyncio.run(
            OpenAICompatProvider(name="local", model="m").complete(
                "p", max_tokens=8, response_schema=_ARRAY_SCHEMA
            )
        )
        assert out == "plain text reply"
        assert client.post.call_count == 2
        first = client.post.call_args_list[0].kwargs["json"]
        second = client.post.call_args_list[1].kwargs["json"]
        assert "response_format" in first
        assert "response_format" not in second

    def test_downgrade_retries_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure(monkeypatch)
        rejected = _status_response(400, "response_format is not supported")
        client = _mock_client([rejected, _status_response(400, "still bad request")])
        _install_client(monkeypatch, client)
        with pytest.raises(OpenAICompatCompletionError):
            asyncio.run(
                OpenAICompatProvider(name="local", model="m").complete(
                    "p", max_tokens=8, response_schema=_ARRAY_SCHEMA
                )
            )
        assert client.post.call_count == 2  # one downgrade retry, never a loop

    def test_unrelated_400_is_not_downgraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure(monkeypatch)
        client = _mock_client([_status_response(400, "context length exceeded")])
        _install_client(monkeypatch, client)
        with pytest.raises(OpenAICompatCompletionError):
            asyncio.run(
                OpenAICompatProvider(name="local", model="m").complete(
                    "p", max_tokens=8, response_schema=_ARRAY_SCHEMA
                )
            )
        assert client.post.call_count == 1  # a real request failure is not masked

    def test_off_disables_enforcement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure(monkeypatch, structured_output="off")
        client = _mock_client([_ok_response("ok")])
        _install_client(monkeypatch, client)
        asyncio.run(
            OpenAICompatProvider(name="local", model="m").complete(
                "p", max_tokens=8, response_schema=_ARRAY_SCHEMA
            )
        )
        assert "response_format" not in client.post.call_args.kwargs["json"]

    def test_no_schema_sends_no_response_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure(monkeypatch)
        client = _mock_client([_ok_response("ok")])
        _install_client(monkeypatch, client)
        asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
        assert "response_format" not in client.post.call_args.kwargs["json"]

    def test_json_object_mode_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure(monkeypatch)
        client = _mock_client([_ok_response("never reached")])
        _install_client(monkeypatch, client)
        with pytest.raises(CompletionError, match="json_object"):
            asyncio.run(
                OpenAICompatProvider(name="local", model="m").complete(
                    "p", max_tokens=8, response_schema={"type": "json_object"}
                )
            )
        client.post.assert_not_called()

    def test_strict_mode_transforms_the_wire_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under ``structured_output: strict`` optional keys go all-required + nullable."""
        _configure(monkeypatch, name="openai", structured_output="strict")
        client = _mock_client([_ok_response('{"a": 1, "b": null}')])
        _install_client(monkeypatch, client)
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        asyncio.run(
            OpenAICompatProvider(name="openai", model="m").complete(
                "p", max_tokens=8, response_schema=schema
            )
        )
        sent = client.post.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]
        assert sorted(sent["required"]) == ["a", "b"]
        assert sent["properties"]["a"] == {"type": "integer"}
        assert sent["properties"]["b"] == {"type": ["string", "null"]}
        assert sent["additionalProperties"] is False
        # The caller's schema object is untouched.
        assert schema["required"] == ["a"]


class TestToStrictSchema:
    def test_optional_keys_become_required_nullable(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "must": {"type": "string"},
                "may": {"type": "integer"},
            },
            "required": ["must"],
        }
        strict = _to_strict_schema(schema)
        assert sorted(strict["required"]) == ["may", "must"]
        assert strict["properties"]["must"] == {"type": "string"}
        assert strict["properties"]["may"] == {"type": ["integer", "null"]}
        assert strict["additionalProperties"] is False

    def test_recurses_through_arrays_and_nested_objects(self) -> None:
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": [],
            },
        }
        strict = _to_strict_schema(schema)
        inner = strict["items"]
        assert inner["required"] == ["x"]
        assert inner["properties"]["x"] == {"type": ["number", "null"]}
        assert inner["additionalProperties"] is False

    def test_already_strict_schema_is_unchanged_in_meaning(self) -> None:
        schema = {
            "type": "object",
            "properties": {"only": {"type": "boolean"}},
            "required": ["only"],
            "additionalProperties": False,
        }
        assert _to_strict_schema(schema) == schema

    def test_nullable_extends_type_lists_and_combinators(self) -> None:
        assert _to_strict_schema(
            {
                "type": "object",
                "properties": {"u": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                "required": [],
            }
        )["properties"]["u"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
            {"type": "null"},
        ]


def test_cache_prefix_is_folded_into_system_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """this adapter ignores the cache marker but preserves content."""
    client = _mock_client([_ok_response("ok")])
    _install_client(monkeypatch, client)
    monkeypatch.delenv("PARTICLES_LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("PARTICLES_LLM_API_KEY_LOCAL", raising=False)

    asyncio.run(
        OpenAICompatProvider(name="local", model="llama3.1:8b").complete(
            "the prompt", max_tokens=64, system="VARIABLE", cache_prefix="FIXED RULES "
        )
    )
    body = client.post.call_args.kwargs["json"]
    # The prefix is prepended to the system message; no cache_control leaks out.
    assert body["messages"][0] == {"role": "system", "content": "FIXED RULES VARIABLE"}
    assert "cache_control" not in json.dumps(body)
