"""Tests for the web-UI same-origin static mount.

The unified web UI's bundle is served by the engine at ``/app``
**unauthenticated**, and the mount must degrade gracefully when the
bundle directory is absent (the bundle is built separately and is not part of
the Python wheel). These tests cover:

- ``build_web_ui_static_app`` returns ``None`` for a missing bundle dir and a
  ``WebUIStaticFiles`` instance for a present one.
- A mounted bundle is served (200) **with no bearer, even when a real
  ``PARTICLES_API_KEY`` is set** — the reversal, and the property a
  browser depends on, since a navigation cannot carry an ``Authorization``
  header. A wrong bearer is served too: the mount does not inspect credentials
  at all.

What did *not* change — that the API paths the loaded app calls stay
gated — is covered where it belongs, in ``tests/test_app.py``'s auth cases.

A FastAPI ``TestClient`` mounts a fresh app per case so the present/absent
branch is exercised without depending on whether the repo's real
``clients/web-ui/dist`` happens to be built.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from particles.api.web_app import (
    WEB_APP_MOUNT,
    WebUIStaticFiles,
    build_web_ui_static_app,
)


def _write_bundle(directory: Path) -> None:
    """Create a minimal built-bundle shell (index.html + one asset)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text(
        "<!doctype html><title>Particles</title><div id=app></div>", encoding="utf-8"
    )
    (directory / "app.js").write_text("console.log('web ui');", encoding="utf-8")


def test_factory_returns_none_when_bundle_absent(tmp_path: Path) -> None:
    """A missing ``dist/`` yields ``None`` so the engine skips the mount."""
    missing = tmp_path / "does-not-exist"
    assert build_web_ui_static_app(missing) is None


def test_factory_returns_static_app_when_bundle_present(tmp_path: Path) -> None:
    """A present bundle dir yields a ``WebUIStaticFiles`` app."""
    bundle = tmp_path / "dist"
    _write_bundle(bundle)
    static_app = build_web_ui_static_app(bundle)
    assert isinstance(static_app, WebUIStaticFiles)


def _app_with_mount(bundle: Path) -> FastAPI:
    """Build a minimal FastAPI app with only the web-UI static mount."""
    app = FastAPI()
    static_app = build_web_ui_static_app(bundle)
    assert static_app is not None
    app.mount(WEB_APP_MOUNT, static_app, name="web-ui")
    return app


def test_mounted_bundle_served_under_dev_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In dev-key mode the app shell is served without a bearer (unchanged)."""
    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    bundle = tmp_path / "dist"
    _write_bundle(bundle)
    client = TestClient(_app_with_mount(bundle), client=("127.0.0.1", 50000))

    resp = client.get(f"{WEB_APP_MOUNT}/index.html")
    assert resp.status_code == 200
    assert "Particles" in resp.text
    # html=True serves index.html for the mount root too.
    assert client.get(f"{WEB_APP_MOUNT}/").status_code == 200


def test_mounted_bundle_served_unauthenticated_with_a_real_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reversal: a real key no longer withholds the shell.

    This is the property a browser depends on — a navigation carries no
    ``Authorization`` header, and before that made the UI unopenable
    in any browser once a real key was set (the token prompt lives inside the
    bundle the gate withheld). Pinned here because re-adding the gate would
    look like a tightening and would in fact remove the entire surface.
    """
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    bundle = tmp_path / "dist"
    _write_bundle(bundle)
    client = TestClient(_app_with_mount(bundle), client=("127.0.0.1", 50000))

    unauth = client.get(f"{WEB_APP_MOUNT}/index.html")
    assert unauth.status_code == 200
    assert "Particles" in unauth.text

    # The SPA root and its assets too — a shell that loads without its JS is
    # no more usable than one that 401s.
    assert client.get(f"{WEB_APP_MOUNT}/").status_code == 200
    assert client.get(f"{WEB_APP_MOUNT}/app.js").status_code == 200


def test_mount_does_not_inspect_credentials_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong bearer is served, not rejected — the mount runs no auth check.

    Distinguishes "unauthenticated" from "authenticated leniently": there is no
    credential path through this mount to get wrong.
    """
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    bundle = tmp_path / "dist"
    _write_bundle(bundle)
    client = TestClient(_app_with_mount(bundle), client=("127.0.0.1", 50000))

    resp = client.get(
        f"{WEB_APP_MOUNT}/index.html",
        headers={"Authorization": "Bearer completely-wrong"},
    )
    assert resp.status_code == 200


def test_mount_serves_nothing_outside_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un-gating the shell must not un-gate the filesystem (path traversal)."""
    monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
    bundle = tmp_path / "dist"
    _write_bundle(bundle)
    (tmp_path / "secret.txt").write_text("not part of the bundle", encoding="utf-8")
    client = TestClient(_app_with_mount(bundle), client=("127.0.0.1", 50000))

    for attempt in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"):
        resp = client.get(f"{WEB_APP_MOUNT}{attempt}")
        assert resp.status_code in (307, 404), attempt
        assert "not part of the bundle" not in resp.text
