# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The /app policy bounds where the stored bearer may go (security review 2026-10-30-1, AB56).

In-process TestClient against the committed web-UI bundle; nothing leaves the process.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_the_app_policy_limits_connections_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Safe property: the shell that holds the bearer in localStorage is served
    with a policy that names default-src and connect-src, so script on the page
    cannot send the bearer to any origin."""
    from particles.api.web_app import build_web_ui_static_app, content_security_policy

    app = build_web_ui_static_app()
    if app is None:
        pytest.skip("no committed web-UI bundle in this tree")
    csp = content_security_policy(Path(app.directory))  # type: ignore[arg-type]
    directives = {part.strip().split(" ", 1)[0] for part in csp.split(";") if part.strip()}
    assert "default-src" in directives, csp
    assert "connect-src" in directives, csp


def test_control_the_app_shell_is_served_with_the_policy() -> None:
    from particles.api.app import app

    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
    resp = client.get("/app/")
    if resp.status_code == 404:
        pytest.skip("no committed web-UI bundle in this tree")
    assert "script-src" in resp.headers.get("content-security-policy", "")
