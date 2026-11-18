# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Shared harness installer predicate for the MCP default-store database pin."""

from __future__ import annotations

from typing import Any

from particles.api.cli._claude_code import absolutize_sqlite_dsn, pinned_default_dsn


def mcp_registration_env(hook_env: dict[str, str], installed_dsn: str) -> dict[str, str]:
    """Pin the installed database only when the pinned default is relative.

    Read the file as written, without the installer's environment. Preserve
    an explicit absolute default chosen by the operator for the read tools.
    """
    env = dict(hook_env)
    if "DATABASE_URL" not in env:
        default = pinned_default_dsn({"PARTICLES_CONFIG": env["PARTICLES_CONFIG"]})
        if absolutize_sqlite_dsn(default) != default:
            env["DATABASE_URL"] = absolutize_sqlite_dsn(installed_dsn)
    return env


def config_mapping(text: str) -> dict[str, Any]:
    """Read installer/scheduler YAML through the bounded parser, without payload errors."""
    import yaml

    from particles.core.bounded_yaml import load

    try:
        value = load(text) or {}
    except yaml.YAMLError:
        raise ValueError("Particles configuration YAML cannot be parsed safely") from None
    if not isinstance(value, dict):
        raise ValueError("Particles configuration must be a mapping")
    return value
