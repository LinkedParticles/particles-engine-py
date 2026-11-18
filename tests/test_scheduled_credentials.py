# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduled credential isolation; no test opens a real Keychain."""

import pytest

from particles import keychain
from particles.config import ParticlesConfig
from particles.secrets import (
    ScheduledCredential,
    get_anthropic_api_key_optional,
    get_llm_api_key_optional,
    scheduled_credential_scope,
)


def test_only_explicit_matching_scope_reads_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def read(service: str, account: str) -> str:
        calls.append(service)
        assert account == "extraction"
        return "dummy-from-keychain"

    monkeypatch.setattr(keychain, "read_generic_password", read)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_anthropic_api_key_optional() is None
    binding = ScheduledCredential("nightly", "/tmp/a.db", "anthropic", "https://api.anthropic.com")
    with pytest.raises(RuntimeError), scheduled_credential_scope(binding):
        assert get_anthropic_api_key_optional() == "dummy-from-keychain"
        assert get_llm_api_key_optional("other-provider") is None
        raise RuntimeError("exercise finally")
    assert get_anthropic_api_key_optional() is None
    assert calls == [binding.service]


def test_environment_precedence_and_nested_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(service: str, account: str) -> str:
        raise AssertionError("Keychain must not be opened")

    monkeypatch.setattr(keychain, "read_generic_password", refuse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-env")
    with scheduled_credential_scope(ScheduledCredential("p", "s", "anthropic", "https://a")):
        assert get_anthropic_api_key_optional() == "dummy-env"
        with scheduled_credential_scope(None):
            assert get_anthropic_api_key_optional() == "dummy-env"


def test_locator_binds_all_four_identity_fields() -> None:
    original = ScheduledCredential("p", "store", "provider", "https://a")
    for other in [
        ScheduledCredential("q", "store", "provider", "https://a"),
        ScheduledCredential("p", "other", "provider", "https://a"),
        ScheduledCredential("p", "store", "other", "https://a"),
        ScheduledCredential("p", "store", "provider", "https://b"),
    ]:
        assert other.service != original.service
    assert original.service == ScheduledCredential("p", "store", "provider", "https://a").service


def test_denial_is_not_silently_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://a")

    def denied(service: str, account: str) -> str:
        raise keychain.KeychainError(-25308)

    monkeypatch.setattr(keychain, "read_generic_password", denied)
    with (
        scheduled_credential_scope(ScheduledCredential("p", "s", "anthropic", "https://a")),
        pytest.raises(keychain.KeychainError, match="locked/denied"),
    ):
        get_anthropic_api_key_optional()


def test_codex_policy_is_independent_and_explicit_tag_lists_stay_explicit() -> None:
    config = ParticlesConfig.model_validate({"claude_code": {"observer_scope": "store"}})
    assert config.codex.observer_scope == "project"
    assert config.codex.harvest.max_rollout_bytes == 64 * 1024 * 1024
    assert "codex" in config.observer_scope.harness_tags
    explicit = ParticlesConfig.model_validate({"observer_scope": {"harness_tags": ["claude-code"]}})
    assert explicit.observer_scope.harness_tags == ["claude-code"]


@pytest.mark.asyncio
async def test_consolidation_context_preflights_before_paid_passes_and_resets(
    monkeypatch, tmp_path
):
    import typer

    from particles.api.cli import _scheduled_credentials as scheduled
    from particles.api.cli import memory
    from particles.secrets import ScheduledCredential, _scheduled_credential

    binding = ScheduledCredential("profile", "store", "anthropic", "https://api.anthropic.com")
    monkeypatch.setattr(scheduled, "credential_binding", lambda store: binding)
    monkeypatch.setattr(scheduled, "credential_status", lambda selected: "missing")
    calls = []

    async def paid(**kw):
        calls.append(True)
        assert _scheduled_credential.get() == binding

    monkeypatch.setattr(memory, "_consolidate_impl_scoped", paid)
    kwargs = dict(
        store="default", if_due=True, structural_only=False, scope="delta", output=None, fmt="text"
    )
    with pytest.raises(typer.Exit) as failure:
        await memory._consolidate_impl(**kwargs)
    assert failure.value.exit_code == 2
    assert not calls and _scheduled_credential.get() is None
    monkeypatch.setattr(scheduled, "credential_status", lambda selected: "Keychain")
    await memory._consolidate_impl(**kwargs)
    assert calls == [True] and _scheduled_credential.get() is None


def test_keyless_readiness_requires_an_explicit_loopback_route(monkeypatch):
    from particles.api.cli._scheduled_credentials import keyless_route
    from particles.config import get_config

    local = ScheduledCredential("p", "s", "local", "http://127.0.0.1:8000/v1")
    assert keyless_route(local)
    assert not keyless_route(ScheduledCredential("p", "s", "local", "http://localhost:8000/v1"))
    get_config().consolidation.keyless_extraction_providers = []
    assert not keyless_route(local)
    get_config().consolidation.keyless_extraction_providers = ["local"]
    assert not keyless_route(ScheduledCredential("p", "s", "local", "https://hosted.example/v1"))


def test_profile_refuses_a_changed_actual_endpoint_before_keychain_read(monkeypatch):
    from unittest.mock import Mock

    reader = Mock(return_value="synthetic-keychain-key")
    monkeypatch.setattr(keychain, "read_generic_password", reader)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://new.example")
    with scheduled_credential_scope(
        ScheduledCredential("p", "s", "anthropic", "https://old.example")
    ):
        assert get_anthropic_api_key_optional() is None
    reader.assert_not_called()


def test_native_clients_cannot_leak_from_a_profile_into_ordinary_calls(monkeypatch):
    from particles.llm import client as native
    from particles.secrets import redact_known_secrets

    native.set_client(None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        keychain, "read_generic_password", lambda *a: "synthetic-keychain-native-key"
    )
    binding = ScheduledCredential("p", "s", "anthropic", "https://api.anthropic.com")
    with scheduled_credential_scope(binding):
        first = native.get_client()
        assert first.api_key == "synthetic-keychain-native-key"
        assert native.get_client() is first
        assert "synthetic-keychain-native-key" not in redact_known_secrets(
            "error echoed synthetic-keychain-native-key"
        )
    assert first.is_closed()
    assert native.sent_credentials() == []
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        native.get_client()
    native.set_client(None)


def test_native_profile_cache_is_bound_to_endpoint_and_environment_still_wins(monkeypatch):
    from particles.llm import client as native

    native.set_client(None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://first.example")
    monkeypatch.setattr(keychain, "read_generic_password", lambda *a: "synthetic-profile-key")
    binding = ScheduledCredential("p", "s", "anthropic", "https://first.example")
    with scheduled_credential_scope(binding):
        first = native.get_client()
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://second.example")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            native.get_client()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-environment-key")
        second = native.get_client()
        assert second is not first and second.api_key == "synthetic-environment-key"
    assert first.is_closed() and second.is_closed()
    native.set_client(None)


@pytest.mark.parametrize(
    ("provider", "names"),
    [
        ("anthropic", ("ANTHROPIC_API_KEY",)),
        ("local", ("PARTICLES_LLM_API_KEY_LOCAL", "PARTICLES_LOCAL_LLM_API_KEY")),
        ("my-gateway", ("PARTICLES_LLM_API_KEY_MY_GATEWAY",)),
    ],
)
def test_provider_names_are_centralized(provider, names):
    from particles.secrets import provider_key_names

    assert provider_key_names(provider) == names


@pytest.mark.asyncio
async def test_invalid_scheduled_binding_reports_blocked_without_paid_pass(monkeypatch, capsys):
    from unittest.mock import AsyncMock

    import typer

    from particles.api.cli import _scheduled_credentials as scheduled
    from particles.api.cli import memory

    def invalid(_store):
        raise ValueError("invalid DSN with sensitive detail")

    paid = AsyncMock(side_effect=AssertionError("paid pass must not run"))
    monkeypatch.setattr(scheduled, "credential_binding", invalid)
    monkeypatch.setattr(memory, "_consolidate_impl_scoped", paid)
    with pytest.raises(typer.Exit) as exc:
        await memory._consolidate_impl(
            store="default",
            if_due=True,
            structural_only=False,
            scope="global",
            output=None,
            fmt="json",
        )
    assert exc.value.exit_code == 2
    output = capsys.readouterr().err
    assert "learning blocked" in output and "sensitive" not in output
    paid.assert_not_called()
