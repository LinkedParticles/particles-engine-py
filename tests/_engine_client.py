# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""An in-process client for an engine that has a real key.

A keyless engine serves reads only, so a test that writes over HTTP runs the
engine with :data:`TEST_ENGINE_KEY` and sends it as the bearer on every request.
A test about the keyless engine itself builds its own client under
``PARTICLES_API_KEY=dev-key`` instead.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

#: The key every keyed test engine runs under.
TEST_ENGINE_KEY = "test-engine-key"

#: The bearer header that key needs.
BEARER = {"Authorization": f"Bearer {TEST_ENGINE_KEY}"}


def keyed_engine_client(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> TestClient:
    """A loopback client of ``particles.api.app`` on an engine keyed with :data:`TEST_ENGINE_KEY`.

    The key is read per request, so a test may still switch the engine keyless
    with ``monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")`` after this.
    """
    from particles.api.app import app

    monkeypatch.setenv("PARTICLES_API_KEY", TEST_ENGINE_KEY)
    kwargs.setdefault("base_url", "http://127.0.0.1:8000")
    kwargs.setdefault("client", ("127.0.0.1", 50000))
    headers = {**BEARER, **kwargs.pop("headers", {})}
    return TestClient(app, headers=headers, **kwargs)
