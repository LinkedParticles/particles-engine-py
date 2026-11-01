# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""``/health`` shows the daemon's detail only to a key when reads are closed (2026-10-28-2 review).

AA50 (the Z43 remainder): with ``api.require_auth_for_reads`` on, a caller with
no key still gets the liveness answer and the aggregate status, but not each
task's runs, failures and last outcome. The daemon's state is stubbed; no task
runs, no model call is made.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

KEY = "health-audit-key"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    import particles.api.app as app_mod
    from particles.api.daemon import DaemonStatus, DaemonTaskStatus
    from particles.config import get_config, reset_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\napi:\n  require_auth_for_reads: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("PARTICLES_API_KEY", KEY)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}")
    reset_config()
    assert get_config().api.require_auth_for_reads is True
    status = DaemonStatus(
        enabled=True,
        healthy=False,
        tasks=[
            DaemonTaskStatus(
                name="inbox",
                interval_seconds=30,
                state="crashed",
                runs=4,
                failures=1,
                last_outcome="processed 2, failed 1",
            )
        ],
    )
    monkeypatch.setattr(app_mod, "daemon_status", lambda: status)
    yield TestClient(app_mod.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    reset_config()


def test_health_is_served_without_a_key_when_reads_are_closed(client: TestClient) -> None:
    """The liveness answer and the aggregate status need no key."""
    assert client.get("/particles").status_code == 401
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded" and "version" in body


def test_the_daemon_detail_needs_a_key_when_reads_are_closed(client: TestClient) -> None:
    """Each task's outcome line reaches only a caller whose key authenticates."""
    anonymous = client.get("/health").json()
    assert not anonymous.get("daemon")
    assert "processed 2" not in str(anonymous)
    wrong = client.get("/health", headers={"Authorization": "Bearer not-the-key"}).json()
    assert not wrong.get("daemon")
    keyed = client.get("/health", headers={"Authorization": f"Bearer {KEY}"}).json()
    assert keyed["daemon"]["tasks"][0]["last_outcome"] == "processed 2, failed 1"
