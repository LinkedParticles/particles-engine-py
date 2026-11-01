# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""No route or mount answers a CORS preflight or sends an allow-origin header.

The engine serves its bearer-holding web UI on its own origin and offers
nothing to another one (security review 2026-10-28-1, J11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount, Route

from particles.api import _keys
from particles.api._keys import Role


@pytest.mark.parametrize("keyed", [False, True])
def test_no_path_answers_a_preflight_or_sends_allow_origin(
    cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keyed: bool
) -> None:
    from particles.api.app import app
    from particles.config import reset_config

    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(tmp_path / "engine-keys.json"))
    reset_config()
    _keys.reset_key_cache()
    try:
        if keyed:
            monkeypatch.delenv("CLAUDECODE", raising=False)
            _keys.mint(Role.READER, "viewer")
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        paths = {"/app/", "/app/index.html"}
        for route in app.routes:
            if isinstance(route, Route | Mount):
                paths.add(route.path.replace("{", "x").replace("}", "").replace(":path", ""))
        origin = {"Origin": "https://elsewhere.example"}
        preflight = {
            **origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        }
        sent = []
        for path in sorted(paths):
            for method, headers in (("OPTIONS", preflight), ("GET", origin), ("POST", origin)):
                resp = client.request(method, path, headers=headers)
                if any(h.lower().startswith("access-control-") for h in resp.headers):
                    sent.append((method, path))
    finally:
        _keys.reset_key_cache()
        _keys.forget_key_files()
    assert sent == []
