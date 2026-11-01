# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The deploy files state the startup key gate as the engine applies it.

Security review 2026-10-27-1, G49 / E12: a live per-client key on the volume
satisfies the fail-closed gate as ``PARTICLES_API_KEY`` does, so neither the
compose file nor the Helm chart may require the operator key alone.
"""

from __future__ import annotations

import re
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def _claims_only_the_operator_key(message: str) -> bool:
    return "refuses to start" in message and "per-client" not in message


def test_compose_runs_without_the_operator_key() -> None:
    compose = (_DEPLOY / "compose.yaml").read_text(encoding="utf-8")
    assert "PARTICLES_API_KEY: ${PARTICLES_API_KEY:-}" in compose
    assert "${PARTICLES_API_KEY:?" not in compose
    # The comment beside it names the key-file alternative.
    assert "engine-keys.json" in compose and "keys create" in compose


def test_the_chart_refusal_states_the_startup_gate_truly() -> None:
    secret = (_DEPLOY / "helm" / "templates" / "secret.yaml").read_text(encoding="utf-8")
    messages = re.findall(r'fail "([^"]*)"', secret)
    assert messages
    for message in messages:
        assert not _claims_only_the_operator_key(message), message
        assert "keysFileOnly" in message


def test_the_chart_can_run_on_per_client_keys_alone() -> None:
    helm = _DEPLOY / "helm"
    secret = (helm / "templates" / "secret.yaml").read_text(encoding="utf-8")
    statefulset = (helm / "templates" / "statefulset.yaml").read_text(encoding="utf-8")
    values = (helm / "values.yaml").read_text(encoding="utf-8")
    assert "keysFileOnly: false" in values
    assert "(not .Values.auth.keysFileOnly)" in secret
    env = statefulset.split("- name: PARTICLES_API_KEY", 1)
    assert len(env) == 2
    ref = env[1].split("- name:", 1)[0]
    assert ".Values.auth.keysFileOnly" in ref and "optional: true" in ref
