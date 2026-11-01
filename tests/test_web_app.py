# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
from tests._upstream import upstream_only


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
    client = TestClient(
        _app_with_mount(bundle), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
    )

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
    client = TestClient(
        _app_with_mount(bundle), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
    )

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
    client = TestClient(
        _app_with_mount(bundle), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
    )

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
    client = TestClient(
        _app_with_mount(bundle), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
    )

    for attempt in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"):
        resp = client.get(f"{WEB_APP_MOUNT}{attempt}")
        assert resp.status_code in (307, 404), attempt
        assert "not part of the bundle" not in resp.text


def test_bundle_served_with_a_script_policy(tmp_path: Path) -> None:
    """The shell holds the bearer, so only its own scripts may run (security review N7)."""
    import base64
    import hashlib

    bundle = tmp_path / "dist"
    _write_bundle(bundle)
    inline = "\n  document.title = 'x';\n"
    (bundle / "index.html").write_text(
        f"<!doctype html><script>{inline}</script><script type=module src=app.js></script>",
        encoding="utf-8",
    )
    client = TestClient(
        _app_with_mount(bundle), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
    )
    digest = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
    for path in ("/", "/app.js"):
        csp = client.get(f"{WEB_APP_MOUNT}{path}").headers["content-security-policy"]
        assert csp.startswith(f"script-src 'self' 'sha256-{digest}';")
        assert "frame-ancestors 'none'" in csp
        assert "unsafe-inline" not in csp


def test_committed_bundle_inline_scripts_are_all_hashed() -> None:
    """Every inline block in the committed bundle is allowed, so the theme stamp still runs."""
    import re

    from particles.api.web_app import _default_bundle_dir, content_security_policy

    index = _default_bundle_dir() / "index.html"
    if not index.is_file():
        pytest.skip("no committed web-UI bundle in this tree")
    blocks = re.findall(r"<script>(.*?)</script>", index.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "the bundle's theme stamp is an inline script"
    script_src = content_security_policy(index.parent).split(";", 1)[0]
    assert script_src.count("'sha256-") == len(blocks)


def test_every_data_href_sink_is_scheme_checked() -> None:
    """A URL from the store becomes an href only behind an http(s) test (security review N37).

    A ``javascript:`` URL in an href runs on the engine's origin, which holds the
    bearer. The route-hash links the app builds itself are not store data.
    """
    import re

    src = Path(__file__).resolve().parents[1] / "clients" / "web-ui" / "src"
    if not src.is_dir():
        pytest.skip("no web-UI source in this tree")
    sinks = 0
    for path in sorted(src.glob("*.ts")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            match = re.search(r"\.href\s*=\s*([\w.]+);", line)
            if match is None or "routeHash" in line:
                continue
            sinks += 1
            value = re.escape(match.group(1))
            window = "\n".join(lines[max(0, number - 6) : number])
            assert re.search(r"\^https\?:.*\.test\(" + value + r"\)", window), (
                f"{path.name}:{number + 1} assigns {match.group(1)} to an href unchecked"
            )
    assert sinks >= 3


def _build_stamp(web: Path) -> str:
    """The build stamp `esbuild.config.mjs` derives from the inputs under ``web``."""
    import hashlib
    import json

    files: list[str] = []

    def walk(rel: str) -> None:
        p = web / rel
        if p.is_dir():
            for name in sorted(x.name for x in p.iterdir()):
                walk(f"{rel}/{name}")
        else:
            files.append(rel)

    for root in (
        "src",
        "public",
        "package.json",
        "../../artifacts/graph/cytoscape.min.js",
        "../../design/tokens.css",
    ):
        walk(root)
    digest = hashlib.sha256()
    # JavaScript's default sort compares UTF-16 code units.
    for f in sorted(files, key=lambda s: s.encode("utf-16-be")):
        digest.update(f.encode())
        digest.update(b"\0")
        digest.update((web / f).read_bytes())
    version = json.loads((web / "package.json").read_text(encoding="utf-8"))["version"]
    return f"{version}+{digest.hexdigest()[:12]}"


@upstream_only  # reads the design tokens and vendored graph build this tree alone holds
def test_the_committed_build_stamp_matches_the_build_inputs(tmp_path: Path) -> None:
    """G48: the committed dist carries the stamp its build inputs produce, so the
    bundle the engine serves is the build of this tree's source.

    The stamp hashes the typed client that `npm run gen-api` writes, so this
    needs the web UI's installed toolchain. Where it is absent, CI's
    `Browser clients` job, which the required `Unit tests` check needs, runs
    the full rebuild-and-compare (security review 2026-10-28-1, O54; the need
    restored in 2026-10-28-2, G48).
    """
    import re
    import shutil
    import subprocess

    root = Path(__file__).resolve().parent.parent
    web = root / "clients" / "web-ui"
    if not (web / "dist" / "app.js").is_file():
        pytest.skip("no committed web-UI bundle in this tree")
    if (
        shutil.which("npm") is None
        or not (web / "node_modules" / ".bin" / "openapi-typescript").exists()
    ):
        pytest.skip("the web UI's toolchain is not installed (npm ci in clients/web-ui)")
    copy = tmp_path / "clients" / "web-ui"
    shutil.copytree(web / "src", copy / "src", ignore=shutil.ignore_patterns("openapi.d.ts"))
    shutil.copytree(web / "public", copy / "public")
    shutil.copy(web / "package.json", copy / "package.json")
    (copy / "node_modules").symlink_to(web / "node_modules")
    for rel in ("artifacts/openapi.json", "artifacts/graph/cytoscape.min.js", "design/tokens.css"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, tmp_path / rel)
    subprocess.run(
        ["npm", "run", "--silent", "gen-api"],
        cwd=copy,
        check=True,
        capture_output=True,
        timeout=120,
    )
    (copy / "node_modules").unlink()
    committed = set(
        re.findall(r"\d+\.\d+\.\d+\+[0-9a-f]{12}", (web / "dist" / "app.js").read_text())
    )
    assert committed == {_build_stamp(copy)}
