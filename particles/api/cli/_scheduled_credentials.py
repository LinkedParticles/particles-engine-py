# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Non-secret scheduled credential planning and noninteractive preflight."""

from __future__ import annotations

from urllib.parse import urlsplit

from particles.api.cli._codex_runtime import physical_store, store_dsn
from particles.config import get_config
from particles.secrets import ScheduledCredential


def credential_binding(
    store: str, profile: str | None = None, *, dsn: str | None = None
) -> ScheduledCredential | None:
    """Bind the selected extraction route; never bind a different endpoint's key."""
    cfg = get_config()
    profile = profile or cfg.consolidation.credential_profile
    if not profile:
        return None
    provider = cfg.llm.for_purpose("extraction").provider
    if provider == "anthropic":
        from particles.secrets import get_anthropic_base_url_optional

        endpoint = get_anthropic_base_url_optional() or "https://api.anthropic.com"
    else:
        endpoint = cfg.llm.providers[provider].base_url
    return ScheduledCredential(profile, physical_store(dsn or store_dsn(store)), provider, endpoint)


def keyless_route(binding: ScheduledCredential, *, declared: list[str] | None = None) -> bool:
    """Only declared routes at literal 127.0.0.1 or ::1 may require no key."""
    names = (
        get_config().consolidation.keyless_extraction_providers if declared is None else declared
    )
    endpoint = urlsplit(binding.endpoint)
    return (
        binding.provider != "anthropic"
        and binding.provider in names
        and endpoint.scheme in {"http", "https"}
        and endpoint.hostname in {"127.0.0.1", "::1"}
    )


def environment_key(binding: ScheduledCredential) -> str | None:
    """The existing central helpers, outside a scheduled fallback context."""
    from particles.secrets import (
        get_anthropic_api_key_optional,
        get_llm_api_key_optional,
        scheduled_credential_scope,
    )

    with scheduled_credential_scope(None):
        return (
            get_anthropic_api_key_optional()
            if binding.provider == "anthropic"
            else get_llm_api_key_optional(binding.provider)
        )


def credential_status(binding: ScheduledCredential, *, use_environment: bool = True) -> str:
    """No network or prompt; doctor can model a job without the shell's keys."""
    from particles.keychain import KeychainError, read_generic_password

    if use_environment and environment_key(binding):
        return "environment"
    if keyless_route(binding):
        return "keyless local route"
    try:
        return "Keychain" if read_generic_password(binding.service, "extraction") else "missing"
    except KeychainError:
        return "locked/denied"


def environment_status(store: str) -> str:
    """Diagnose an ordinary consolidation process without enabling OS fallback."""
    binding = credential_binding(store, "diagnostic-only")
    if binding is None:
        return "unknown"
    if environment_key(binding):
        return "environment"
    return "keyless local route" if keyless_route(binding) else "missing"


def reused_credential_status(job: object) -> str:
    """Inspect a reused job's own route; never install a profile or read shell keys."""
    from pathlib import Path

    from particles.api.cli._codex_install import read_file
    from particles.api.cli._codex_schedule import Schedule
    from particles.api.cli._harness_pins import config_mapping
    from particles.config import ParticlesConfig
    from particles.secrets import ScheduledCredential, provider_key_names

    if not isinstance(job, Schedule):
        return "unverified reused job"
    config = job.env.get("PARTICLES_CONFIG")
    if not config:
        return "unverified reused job"
    raw = read_file(Path(config))
    if raw is None:
        return "unverified reused job"
    cfg = ParticlesConfig.model_validate(config_mapping(raw.decode()))
    provider = cfg.llm.for_purpose("extraction").provider
    endpoint = (
        job.env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        if provider == "anthropic"
        else cfg.llm.providers[provider].base_url
    )
    if any(job.env.get(name) for name in provider_key_names(provider)):
        return "job environment"
    binding = ScheduledCredential(
        cfg.consolidation.credential_profile or "diagnostic-only", job.store_id, provider, endpoint
    )
    if keyless_route(binding, declared=cfg.consolidation.keyless_extraction_providers):
        return "keyless local route"
    if cfg.consolidation.credential_profile:
        return credential_status(binding, use_environment=False)
    return "unverified reused job"
