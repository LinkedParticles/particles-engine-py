# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/api/auth.py: the 2026-10-23 security review's D4, D5 and critic's gap.

- **Cross-site browser requests** (the completeness critic's highest-risk gap):
  any web page could send a multipart POST to ``/corpus/deposit/file`` or a form
  POST to ``/curation/rebuild?semantic=true`` on a dev-key loopback engine, with
  no preflight. Every gate now refuses a request the browser marks cross-site
  unless it carries a verified bearer, and under the dev-key skip a browser
  request must name a loopback ``Host`` (DNS rebinding).
- **D5**: the read routes now get the dev-key loopback backstop, and the startup
  gate reads the ``uvicorn`` command line's own bind.
- **D4**: the container's baked config requires the bearer on reads.

The paid-route rate limits and ``limit`` clamps from the same review (D16, D17,
D37) are endpoint wiring, pinned beside ``TestRateLimit`` in ``tests/test_app.py``.

Everything runs through ``TestClient`` against a scratch store: no network, no
model call, no Anthropic key.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from particles.config import get_config, reset_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = "http://127.0.0.1:8000"
_EVIL = "https://evil.example"

#: What a browser sends for a request a page on another site made.
_CROSS_SITE = {"Origin": _EVIL, "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors"}
#: What a browser sends for the engine's own web UI calling the engine.
_SAME_ORIGIN = {"Origin": _ENGINE, "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}


@pytest.fixture
def engine(cli_db: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A dev-key engine reached at ``127.0.0.1:8000`` from a loopback peer."""
    from particles.api.app import app

    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    yield TestClient(app, base_url=_ENGINE, client=("127.0.0.1", 50000))


def _deposit(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        "/corpus/deposit/file",
        files={"file": ("note.md", b"The attacker's planted claim.", "text/markdown")},
        data={"tags": "project:victim"},
        headers=headers,
    )


def _corpus_size(client: TestClient) -> int:
    resp = client.get("/corpus")
    assert resp.status_code == 200, resp.text
    return len(resp.json())


# ---------------------------------------------------------------------------
# Cross-site browser requests (the critic's gap)
# ---------------------------------------------------------------------------


class TestCrossSiteRefusedUnderDevKey:
    def test_multipart_deposit_from_another_site_refused(self, engine: TestClient) -> None:
        # The critic's reproduction: a 200 and a new entry carrying the page's tags.
        resp = _deposit(engine, _CROSS_SITE)
        assert resp.status_code == 403
        assert "cross-site" in resp.json()["detail"]
        assert _corpus_size(engine) == 0

    @pytest.mark.parametrize("site", ["cross-site", "same-site"])
    def test_sec_fetch_site_alone_decides(self, engine: TestClient, site: str) -> None:
        # same-site is refused too: a page on another localhost port is the same site.
        resp = _deposit(engine, {"Sec-Fetch-Site": site})
        assert resp.status_code == 403

    @pytest.mark.parametrize("origin", [_EVIL, "null", "http://127.0.0.1:3000"])
    def test_foreign_origin_without_fetch_metadata_refused(
        self, engine: TestClient, origin: str
    ) -> None:
        # An older browser sends Origin on a POST but no Sec-Fetch-Site.
        resp = _deposit(engine, {"Origin": origin})
        assert resp.status_code == 403
        assert _corpus_size(engine) == 0

    def test_form_post_to_semantic_rebuild_refused(self, engine: TestClient) -> None:
        # The paid store-wide probe a page could start with a plain form POST.
        resp = engine.post(
            "/curation/rebuild?semantic=true",
            data={"x": "1"},
            headers=_CROSS_SITE,
        )
        assert resp.status_code == 403

    def test_cross_site_paid_get_refused(self, engine: TestClient) -> None:
        # An <img src> can issue a GET; the semantic curation queue is paid.
        resp = engine.get("/curation?semantic=true", headers=_CROSS_SITE)
        assert resp.status_code == 403

    def test_cross_site_read_refused(self, engine: TestClient) -> None:
        resp = engine.get("/corpus", headers=_CROSS_SITE)
        assert resp.status_code == 403

    def test_dns_rebinding_host_refused(self, engine: TestClient) -> None:
        # A rebinding page reaches 127.0.0.1 under its own name, so the browser
        # calls the request same-origin; the Host header gives it away.
        resp = engine.get(
            "/corpus",
            headers={"Host": "evil.example:8000", "Sec-Fetch-Site": "same-origin"},
        )
        assert resp.status_code == 403
        assert "host" in resp.json()["detail"]


class TestRebindingWithoutBrowserMarkers:
    """Security review 2026-10-24, N1: the Host check ran only on browser-marked requests.

    A browser sends fetch metadata only to a potentially trustworthy URL, and a
    same-origin GET carries no Origin, so a GET from a page on a rebound
    plain-http hostname reached the engine with neither header and was served as
    if the CLI had sent it. The test above sent ``Sec-Fetch-Site``, which such a
    browser does not, and so passed while the read stayed open.
    """

    @pytest.mark.parametrize(
        "path",
        ["/corpus", "/corpus/blob/abc12345", "/events", "/particles", "/curation"],
    )
    def test_marker_less_get_for_a_foreign_host_refused(
        self, engine: TestClient, path: str
    ) -> None:
        resp = engine.get(path, headers={"Host": "attacker.example:8000"})
        assert resp.status_code == 403
        assert "attacker.example" in resp.json()["detail"]

    def test_all_interfaces_host_refused(self, engine: TestClient) -> None:
        # A page that GETs http://0.0.0.0:8000 names that address as the Host.
        assert engine.get("/corpus", headers={"Host": "0.0.0.0:8000"}).status_code == 403

    @pytest.mark.parametrize("path", ["/health", "/openapi.json", "/app/", "/no-such-route"])
    def test_ungated_paths_refused_for_a_foreign_host(self, engine: TestClient, path: str) -> None:
        # The middleware applies the bound before routing, so the paths that
        # carry no gate are covered too.
        resp = engine.get(path, headers={"Host": "attacker.example"})
        assert resp.status_code == 403

    def test_marker_less_write_for_a_foreign_host_refused(self, engine: TestClient) -> None:
        resp = _deposit(engine, {"Host": "attacker.example:8000"})
        assert resp.status_code == 403
        assert _corpus_size(engine) == 0

    def test_listed_host_served(self, engine: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_config().api, "allowed_hosts", ["particles.localhost"])
        resp = engine.get("/corpus", headers={"Host": "particles.localhost:8000"})
        assert resp.status_code == 200, resp.text

    def test_wildcard_entry_matches_a_subdomain_only(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().api, "allowed_hosts", ["*.tail.example"])
        assert engine.get("/corpus", headers={"Host": "box.tail.example"}).status_code == 200
        assert engine.get("/corpus", headers={"Host": "eviltail.example"}).status_code == 403


class TestLegitimateCallersStillServed:
    def test_web_ui_same_origin_deposit_served(self, engine: TestClient) -> None:
        resp = _deposit(engine, _SAME_ORIGIN)
        assert resp.status_code == 200, resp.text
        assert _corpus_size(engine) == 1

    def test_web_ui_on_localhost_served(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        client = TestClient(app, base_url="http://localhost:8000", client=("::1", 50000))
        resp = client.get(
            "/corpus",
            headers={"Origin": "http://localhost:8000", "Sec-Fetch-Site": "same-origin"},
        )
        assert resp.status_code == 200, resp.text

    def test_same_origin_by_origin_header_alone_served(self, engine: TestClient) -> None:
        resp = _deposit(engine, {"Origin": _ENGINE})
        assert resp.status_code == 200, resp.text

    def test_typed_url_served(self, engine: TestClient) -> None:
        # Sec-Fetch-Site: none is the user opening the URL themselves.
        resp = engine.get("/corpus", headers={"Sec-Fetch-Site": "none"})
        assert resp.status_code == 200

    def test_non_browser_client_served(self, engine: TestClient) -> None:
        # The CLI, the MCP HTTP backend and curl send no Origin and no fetch metadata.
        resp = _deposit(engine, {})
        assert resp.status_code == 200, resp.text

    def test_non_browser_client_with_a_foreign_host_refused(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Under the dev-key skip the Host bound applies to every request, not
        # only to browser-marked ones (security review 2026-10-24, N1): this
        # used to pin TestClient's own "testserver" as served.
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        client = TestClient(app, client=("127.0.0.1", 50000))
        assert client.get("/corpus").status_code == 403

    @pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000"])
    def test_cli_on_a_loopback_host_served(self, engine: TestClient, host: str) -> None:
        # The CLI's remote backend and the MCP backend name a loopback address.
        assert engine.get("/corpus", headers={"Host": host}).status_code == 200

    def test_request_without_a_host_header_passes_the_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every browser sends Host, so its absence (an HTTP/1.0 local client)
        # is not a rebinding page. httpx always sends one, hence the raw scope.
        from starlette.requests import Request

        from particles.api.auth import request_bound_refusal

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        scope = {"type": "http", "headers": [], "client": ("127.0.0.1", 50000)}
        assert request_bound_refusal(Request(scope)) is None

    def test_allowed_origin_served(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_config().api, "allowed_origins", ["https://notes.example.org/"])
        resp = _deposit(
            engine,
            {"Origin": "https://notes.example.org", "Sec-Fetch-Site": "cross-site"},
        )
        assert resp.status_code == 200, resp.text


class TestCrossSiteWithRealKey:
    @pytest.fixture
    def keyed(self, cli_db: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        yield TestClient(app, base_url="http://engine.tailnet:8000", client=("100.64.0.9", 50000))

    def test_verified_bearer_served_whatever_the_origin(self, keyed: TestClient) -> None:
        # A page on another site cannot attach this header without a CORS
        # preflight the engine never approves, so it is proof enough.
        resp = _deposit(keyed, {**_CROSS_SITE, "Authorization": "Bearer prod-secret"})
        assert resp.status_code == 200, resp.text

    def test_write_without_bearer_still_401(self, keyed: TestClient) -> None:
        assert _deposit(keyed, _CROSS_SITE).status_code == 401

    def test_open_read_refused_cross_site(self, keyed: TestClient) -> None:
        # Reads are open with a real key unless require_auth_for_reads; a page on
        # another site still does not get them.
        assert keyed.get("/corpus", headers=_CROSS_SITE).status_code == 403
        assert keyed.get("/corpus").status_code == 200
        bearer = {**_CROSS_SITE, "Authorization": "Bearer prod-secret"}
        assert keyed.get("/corpus", headers=bearer).status_code == 200

    def test_open_read_same_origin_served_on_a_named_host(self, keyed: TestClient) -> None:
        # The loopback Host rule is the dev-key skip's alone.
        resp = keyed.get(
            "/corpus",
            headers={"Origin": "http://engine.tailnet:8000", "Sec-Fetch-Site": "same-origin"},
        )
        assert resp.status_code == 200

    def test_listed_hosts_bound_every_path_under_a_real_key(
        self, keyed: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With api.allowed_hosts set, a rebinding page cannot reach a deployed
        # engine's open reads, or /health, under its own name.
        monkeypatch.setattr(get_config().api, "allowed_hosts", ["engine.tailnet"])
        assert keyed.get("/corpus").status_code == 200
        assert keyed.get("/health", headers={"Host": "127.0.0.1:8000"}).status_code == 200
        for path in ("/corpus", "/health"):
            resp = keyed.get(path, headers={"Host": "attacker.example"})
            assert resp.status_code == 403
            assert "api.allowed_hosts" in resp.json()["detail"]
        bearer = {"Host": "attacker.example", "Authorization": "Bearer prod-secret"}
        assert keyed.get("/events", headers=bearer).status_code == 403


class TestAllowedOriginsValidation:
    @pytest.mark.parametrize(
        "entry", ["*", "null", "https://*.example.org", "notes.example.org", "https://a.org/app"]
    )
    def test_rejects_what_is_not_an_origin(self, entry: str) -> None:
        from pydantic import ValidationError

        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(allowed_origins=[entry])

    def test_accepts_an_origin(self) -> None:
        from particles.config import ApiConfig

        assert ApiConfig(allowed_origins=["http://localhost:5173"]).allowed_origins


class TestAllowedHostsValidation:
    @pytest.mark.parametrize(
        "entry", ["*", "", "*.*.example", "https://engine.example", "engine.example:8000", "a/b"]
    )
    def test_rejects_what_is_not_a_host_name(self, entry: str) -> None:
        from pydantic import ValidationError

        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(allowed_hosts=[entry])

    def test_accepts_and_lowercases_names(self) -> None:
        from particles.config import ApiConfig

        hosts = ApiConfig(allowed_hosts=["Engine.Example", "*.tail.example", "[fd00::5]"])
        assert hosts.allowed_hosts == ["engine.example", "*.tail.example", "[fd00::5]"]


# ---------------------------------------------------------------------------
# D5: the dev-key loopback backstop on reads, and the real bind at startup
# ---------------------------------------------------------------------------


class TestDevKeyReadBackstop:
    @pytest.mark.parametrize("path", ["/corpus", "/corpus/blob/abc12345", "/particles"])
    def test_non_loopback_read_refused_503(
        self, cli_db: object, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        # The review's reproduction: a 10.0.0.5 peer got the raw blob bytes from
        # an engine launched by a raw `uvicorn --host 0.0.0.0` under dev-key.
        from particles.api.app import app

        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        client = TestClient(app, client=("10.0.0.5", 50000))
        resp = client.get(path)
        assert resp.status_code == 503
        assert "non-loopback" in resp.json()["detail"]


class TestStartupGateReadsTheUvicornCommandLine:
    @pytest.fixture(autouse=True)
    def _dev_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
        for var in ("UVICORN_HOST", "UVICORN_UDS", "UVICORN_FD", "PARTICLES_API_BIND_HOST"):
            monkeypatch.delenv(var, raising=False)
        reset_config()

    @pytest.mark.parametrize(
        "argv",
        [
            ["/venv/bin/uvicorn", "particles.api.app:app", "--host", "0.0.0.0"],
            ["/venv/bin/uvicorn", "particles.api.app:app", "--reload", "--host=::"],
            [
                "/venv/lib/python3.11/site-packages/uvicorn/__main__.py",
                "x:app",
                "--host",
                "10.1.2.3",
            ],
            ["/venv/bin/uvicorn", "particles.api.app:app", "--fd", "3"],
            # An empty --host binds every interface (security review 2026-10-24, N11).
            ["/venv/bin/uvicorn", "particles.api.app:app", "--host", ""],
            ["/venv/bin/uvicorn", "particles.api.app:app", "--host="],
        ],
    )
    def test_non_loopback_or_unprovable_bind_refused(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        # api.bind_host stays at its loopback default: it is not what uvicorn binds.
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setattr(sys, "argv", argv)
        assert get_config().api.bind_host == "127.0.0.1"
        with pytest.raises(RuntimeError, match="the uvicorn command line"):
            enforce_fail_closed_on_startup()

    def test_uvicorn_host_env_var_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setattr(sys, "argv", ["/venv/bin/uvicorn", "particles.api.app:app"])
        monkeypatch.setenv("UVICORN_HOST", "0.0.0.0")
        with pytest.raises(RuntimeError, match="'0.0.0.0'"):
            enforce_fail_closed_on_startup()

    @pytest.mark.parametrize(
        "argv",
        [
            ["/venv/bin/uvicorn", "particles.api.app:app", "--reload", "--port", "8000"],
            ["/venv/bin/uvicorn", "particles.api.app:app", "--host", "localhost"],
            ["/venv/bin/uvicorn", "particles.api.app:app", "--uds", "/tmp/particles.sock"],
        ],
    )
    def test_loopback_bind_boots(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setattr(sys, "argv", argv)
        enforce_fail_closed_on_startup()

    def test_uvicorn_loopback_bind_wins_over_a_stale_bind_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        monkeypatch.setattr(sys, "argv", ["/venv/bin/uvicorn", "particles.api.app:app"])
        enforce_fail_closed_on_startup()

    def test_other_launchers_still_read_bind_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `engine serve` is not the uvicorn CLI; it sets api.bind_host from its bind.
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setattr(sys, "argv", ["/venv/bin/particles", "engine", "serve", "0.0.0.0:8000"])
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        with pytest.raises(RuntimeError, match="api.bind_host"):
            enforce_fail_closed_on_startup()

    @pytest.mark.parametrize(
        "argv",
        [
            ["/venv/bin/gunicorn", "-k", "uvicorn.workers.UvicornWorker", "particles.api.app:app"],
            ["/venv/bin/hypercorn", "particles.api.app:app", "--bind", "0.0.0.0:8000"],
            ["/venv/bin/fastapi", "run", "particles/api/app.py"],
            ["/venv/lib/python3.11/site-packages/gunicorn/__main__.py", "x:app"],
        ],
    )
    def test_launchers_whose_bind_is_unreadable_refused(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        # Security review 2026-10-24, N11: these fell back to api.bind_host, still
        # the loopback default, whatever they actually bound.
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setattr(sys, "argv", argv)
        assert get_config().api.bind_host == "127.0.0.1"
        with pytest.raises(RuntimeError, match="whose bind this check cannot read"):
            enforce_fail_closed_on_startup()

    def test_an_unreadable_launcher_boots_with_a_real_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.auth import enforce_fail_closed_on_startup

        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        monkeypatch.setattr(sys, "argv", ["/venv/bin/gunicorn", "particles.api.app:app"])
        enforce_fail_closed_on_startup()


# ---------------------------------------------------------------------------
# D4: the container's baked config closes the read surface
# ---------------------------------------------------------------------------


class TestContainerConfigRequiresReadAuth:
    @pytest.fixture
    def container(self, cli_db: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        from particles.api.app import app

        # The baked file, with the scratch store's env overrides (DATABASE_URL,
        # PARTICLES_BLOB_DIR) still winning over its /data pins.
        monkeypatch.setenv("PARTICLES_CONFIG", str(_REPO_ROOT / "deploy" / "config.container.yaml"))
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        reset_config()
        yield TestClient(app, client=("172.17.0.1", 50000))

    def test_baked_config_sets_the_flag(self, container: TestClient) -> None:
        assert get_config().api.require_auth_for_reads is True

    @pytest.mark.parametrize(
        "path", ["/corpus", "/corpus/blob/abc12345", "/particles/p-missing/source", "/particles"]
    )
    def test_reads_without_bearer_refused(self, container: TestClient, path: str) -> None:
        assert container.get(path).status_code == 401

    def test_reads_with_bearer_served(self, container: TestClient) -> None:
        resp = container.get("/corpus", headers={"Authorization": "Bearer prod-secret"})
        assert resp.status_code == 200

    def test_health_stays_open(self, container: TestClient) -> None:
        assert container.get("/health").status_code == 200


class TestContainerImageBackstopsReadAuth:
    """Security review 2026-10-24, N9: a mounted config replaces the baked file entirely."""

    def test_image_env_requires_read_auth_whatever_the_file(self) -> None:
        dockerfile = (_REPO_ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
        runtime = dockerfile.split("AS runtime", 1)[1]
        assert "PARTICLES_API_REQUIRE_AUTH_FOR_READS=true" in runtime

    def test_the_env_wins_over_a_config_that_opens_the_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mounted = tmp_path / "config.yaml"
        mounted.write_text("api:\n  require_auth_for_reads: false\n", encoding="utf-8")
        monkeypatch.setenv("PARTICLES_CONFIG", str(mounted))
        monkeypatch.setenv("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "true")
        reset_config()
        assert get_config().api.require_auth_for_reads is True


class TestOpenReadsWarning:
    @pytest.mark.parametrize(
        ("key", "bind", "reads_gated", "warns"),
        [
            ("prod-secret", "0.0.0.0", False, True),
            ("prod-secret", "0.0.0.0", True, False),
            ("prod-secret", "127.0.0.1", False, False),
            ("dev-key", "127.0.0.1", False, False),
        ],
    )
    def test_warns_only_for_open_reads_beyond_loopback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        key: str,
        bind: str,
        reads_gated: bool,
        warns: bool,
    ) -> None:
        from particles.api.auth import open_reads_warning

        monkeypatch.setenv("PARTICLES_API_KEY", key)
        monkeypatch.setattr(get_config().api, "require_auth_for_reads", reads_gated)
        assert (open_reads_warning(bind) is not None) is warns
