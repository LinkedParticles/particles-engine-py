# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A thin client's tuning compared with the engine's.

The classification of every config section, the digests the engine serves on
``GET /config/tuning``, and the doctor's ``tuning:`` line on a thin client.
The engine in the doctor tests is a mocked transport; no socket is opened.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import particles
from particles import config_tuning as ct
from particles.api.cli import app
from particles.config import ParticlesConfig, env_override_fields, reset_config
from tests._claude_projects import isolate_claude_code
from tests._engine_client import keyed_engine_client

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_every_top_level_section_is_classified_exactly_once() -> None:
    """A new section fails here until it is named matched or per side."""
    fields = set(ParticlesConfig.model_fields)
    assert fields - ct.MATCHED_SECTIONS - ct.PER_SIDE_SECTIONS == set()
    assert set() == ct.MATCHED_SECTIONS & ct.PER_SIDE_SECTIONS
    assert (ct.MATCHED_SECTIONS | ct.PER_SIDE_SECTIONS) - fields == set()


def test_matched_subsections_sit_in_per_side_sections() -> None:
    for sub in ct.MATCHED_SUBSECTIONS:
        assert sub.split(".", 1)[0] in ct.PER_SIDE_SECTIONS


def test_the_incident_settings_are_matched() -> None:
    leaves = ct.tuning_leaves(ParticlesConfig())
    assert "consolidation.max_pending_entries" in leaves
    assert "consolidation.extract_budget_usd" in leaves
    assert "claude_code.observer_scope" in leaves
    assert "mcp.write.enabled_stores" in leaves


def test_per_side_sections_and_leaves_are_absent() -> None:
    leaves = ct.tuning_leaves(ParticlesConfig())
    for path in leaves:
        if path.split(".")[0] in ct.PER_SIDE_SECTIONS:
            assert path.startswith(tuple(ct.MATCHED_SUBSECTIONS)), path
    assert not any(p.startswith(("storage.", "api.", "engine.", "host.")) for p in leaves)
    assert "consolidation.credential_profile" not in leaves
    assert "graph.manifest_root" not in leaves
    assert "embeddings.progress_bars" not in leaves
    # The MCP server's own settings run on the client; only the write switch is shared.
    assert "mcp.rate_limit_per_minute" not in leaves
    assert "mcp.query_hits" not in leaves
    assert "mcp.write.max_asserted_confidence" in leaves
    assert "claude_code.state_dir" not in leaves


def test_a_provider_base_url_is_stripped_from_the_leaf() -> None:
    """A local model is localhost here and host.docker.internal in a container."""
    here = ParticlesConfig.model_validate(
        {"llm": {"providers": {"local": {"base_url": "http://localhost:11434/v1"}}}}
    )
    there = ParticlesConfig.model_validate(
        {"llm": {"providers": {"local": {"base_url": "http://host.docker.internal:11434/v1"}}}}
    )
    assert "base_url" not in ct.tuning_leaves(here)["llm.providers"]["local"]
    result = ct.compare_tuning(here, ct.tuning_digests(there), [])
    assert result.differing == {}


def test_a_dict_entry_on_one_side_is_a_difference_not_a_missing_leaf() -> None:
    here = ParticlesConfig.model_validate({"refetch_floors": {"WEB_PAGE": 60}})
    result = ct.compare_tuning(here, ct.tuning_digests(ParticlesConfig()), [])
    assert list(result.differing) == ["refetch_floors"]
    assert result.one_sided == 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _client_cfg() -> ParticlesConfig:
    return ParticlesConfig.model_validate(
        {"consolidation": {"max_pending_entries": 200, "extract_budget_usd": 4.0}}
    )


def test_the_2026_11_08_drift_is_named() -> None:
    result = ct.compare_tuning(_client_cfg(), ct.tuning_digests(ParticlesConfig()), [])
    assert result.differing == {
        "consolidation.extract_budget_usd": 4.0,
        "consolidation.max_pending_entries": 200,
    }
    assert result.matching == len(ct.tuning_leaves(ParticlesConfig())) - 2


def test_identical_configs_match_everywhere() -> None:
    cfg = _client_cfg()
    result = ct.compare_tuning(cfg, ct.tuning_digests(cfg), [])
    assert result.differing == {}
    assert result.one_sided == 0


def test_a_declared_difference_is_counted_not_warned() -> None:
    result = ct.compare_tuning(
        _client_cfg(),
        ct.tuning_digests(ParticlesConfig()),
        ["consolidation.max_pending_entries"],
    )
    assert list(result.differing) == ["consolidation.extract_budget_usd"]
    assert result.declared == ["consolidation.max_pending_entries"]


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("consolidation", "consolidation.max_pending_entries", True),
        ("consolidation.max_pending_entries", "consolidation.max_pending_entries", True),
        ("consolidation.budget_usd", "consolidation.max_pending_entries", False),
        ("*.max_pending_entries", "consolidation.max_pending_entries", True),
        ("llm.providers.local.model", "llm.providers", True),
        ("llm", "lint.max_findings", False),
    ],
)
def test_ignore_patterns(pattern: str, path: str, expected: bool) -> None:
    assert ct.ignored_by(path, [pattern]) is expected


def test_a_leaf_known_to_one_side_only_is_counted() -> None:
    digests = ct.tuning_digests(ParticlesConfig())
    digests["consolidation.from_a_newer_release"] = ct.leaf_digest(1)
    del digests["consolidation.max_pending_entries"]
    result = ct.compare_tuning(ParticlesConfig(), digests, [])
    assert result.one_sided == 2
    assert result.differing == {}


def test_digests_are_stable_and_carry_no_value() -> None:
    cfg = ParticlesConfig.model_validate(
        {"llm": {"providers": {"local": {"base_url": "http://secret-host.example:1/v1"}}}}
    )
    digests = ct.tuning_digests(cfg)
    assert digests == ct.tuning_digests(cfg)
    assert all(d.startswith("sha256:") and len(d) == 71 for d in digests.values())
    assert "secret-host" not in repr(digests)


def test_preview_caps_a_long_value() -> None:
    assert ct.preview(200) == "200"
    long = ct.preview(["x" * 50, "y" * 50])
    assert len(long) == ct.VALUE_PREVIEW_CHARS
    assert long.endswith("…")


def test_env_overrides_name_matched_fields_only() -> None:
    env = {
        "AUDIT_MAX_CONTRADICTION_PROBES": "5",
        "DATABASE_URL": "sqlite+aiosqlite:////x.db",
        "PARTICLES_ENGINE_BASE_URL": "",
    }
    assert env_override_fields(env) == {
        "AUDIT_MAX_CONTRADICTION_PROBES": "audit.max_contradiction_probes",
        "DATABASE_URL": "storage.database_url",
    }
    assert ct.env_overridden_tuning(env) == {
        "audit.max_contradiction_probes": "AUDIT_MAX_CONTRADICTION_PROBES"
    }


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_route_serves_digests_and_the_engines_own_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_file = tmp_path / "engine-config.yaml"
    cfg_file.write_text("consolidation:\n  extract_budget_usd: 4.0\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg_file))
    # Only this test's override may reach the engine's config, whatever the shell exports.
    for var in ct.env_overridden_tuning().values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUDIT_MAX_CONTRADICTION_PROBES", "7")
    reset_config()
    try:
        client = keyed_engine_client(monkeypatch)
        resp = client.get("/config/tuning")
        assert resp.status_code == 200, resp.text
        body = ct.TuningReport.model_validate(resp.json())
        from particles.config import get_config

        assert body.digests == ct.tuning_digests(get_config())
        assert body.version == particles.__version__
        assert body.env_overrides == ["audit.max_contradiction_probes"]
        # The engine still runs the compiled ceiling of 20 under a budget.
        assert any("max_pending_entries is 20" in w for w in body.warnings)
        assert not any(k.startswith("storage.") for k in body.digests)
    finally:
        reset_config()


def test_the_route_takes_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = keyed_engine_client(monkeypatch)
    assert client.get("/config/tuning", headers={"Authorization": ""}).status_code == 401


# ---------------------------------------------------------------------------
# The doctor on a thin client
# ---------------------------------------------------------------------------

_TOKEN = "engine-token-tuning-0123456789"
Handler = Callable[[httpx.Request], httpx.Response]


def _engine(tuning: Callable[[], httpx.Response]) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": particles.__version__})
        if path == "/digest/default":
            return httpx.Response(200, json={"markdown": "# Memory digest\n"})
        if path == "/auth/whoami":
            return httpx.Response(
                200, json={"principal": "operator", "role": "operator", "belief_write_stores": []}
            )
        if path == "/config/tuning":
            return tuning()
        return httpx.Response(404, json={"detail": "no such route"})

    return handle


def _report(cfg: ParticlesConfig, **kw: Any) -> Callable[[], httpx.Response]:
    def answer() -> httpx.Response:
        report = ct.TuningReport(
            version=particles.__version__, digests=ct.tuning_digests(cfg), **kw
        )
        return httpx.Response(200, json=report.model_dump(mode="json"))

    return answer


@pytest.fixture
def thin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Any]]:
    """A thin client whose own config tunes the budget; returns a doctor runner."""
    isolate_claude_code(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'absent.db'}")
    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "https://engine.test")
    monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", _TOKEN)
    cfg_file = tmp_path / "client-config.yaml"

    def run(handler: Handler, extra: str = "") -> Any:
        cfg_file.write_text(
            "consolidation:\n  max_pending_entries: 200\n  extract_budget_usd: 4.0\n" + extra,
            encoding="utf-8",
        )
        monkeypatch.setenv("PARTICLES_CONFIG", str(cfg_file))
        reset_config()
        real = httpx.AsyncClient

        def _client(
            *, base_url: str, timeout: float, headers: dict[str, str], purpose: str = ""
        ) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(handler), base_url=base_url, headers=headers)

        monkeypatch.setattr("particles.api.client.http.egress_client", _client)
        return CliRunner().invoke(app, ["hook", "doctor", "--store", "default"])

    yield run
    reset_config()


def test_the_doctor_names_the_drift_and_the_engines_warnings(thin: Callable[..., Any]) -> None:
    engine_cfg = ParticlesConfig()
    result = thin(
        _engine(_report(engine_cfg, warnings=["consolidation.max_pending_entries is 20, ..."]))
    )
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "tuning:         ⚠ 2 of" in out
    assert "⚠ drift: consolidation.max_pending_entries is 200 here; the engine runs another" in out
    assert "⚠ drift: consolidation.extract_budget_usd is 4.0 here" in out
    assert "set in its config file or left at its compiled default" in out
    assert "extract budget: ⚠ (the engine's config) consolidation.max_pending_entries is 20" in out
    # The client's own budget warnings judge a config that runs no consolidation.
    assert "extract budget: ⚠ consolidation" not in out


def test_the_doctor_says_when_everything_matches(thin: Callable[..., Any]) -> None:
    engine_cfg = ParticlesConfig.model_validate(
        {"consolidation": {"max_pending_entries": 200, "extract_budget_usd": 4.0}}
    )
    result = thin(_engine(_report(engine_cfg)))
    assert result.exit_code == 0, result.output
    assert "settings match the engine's" in result.stdout
    assert "⚠ drift:" not in result.stdout


def test_the_doctor_counts_a_declared_difference(thin: Callable[..., Any]) -> None:
    engine_cfg = ParticlesConfig.model_validate({"consolidation": {"extract_budget_usd": 4.0}})
    result = thin(
        _engine(_report(engine_cfg)),
        extra='engine:\n  tuning_drift_ignore: ["consolidation.max_pending_entries"]\n',
    )
    assert result.exit_code == 0, result.output
    assert "1 declared in engine.tuning_drift_ignore" in result.stdout
    assert "⚠ drift:" not in result.stdout


def test_the_doctor_names_an_engine_environment_override(thin: Callable[..., Any]) -> None:
    result = thin(
        _engine(_report(ParticlesConfig(), env_overrides=["consolidation.max_pending_entries"]))
    )
    assert result.exit_code == 0, result.output
    assert (
        "max_pending_entries is 200 here; the engine runs another value, set in its environment"
    ) in result.stdout


def test_an_engine_without_the_route_is_not_compared(thin: Callable[..., Any]) -> None:
    result = thin(_engine(lambda: httpx.Response(404, json={"detail": "Not Found"})))
    assert result.exit_code == 0, result.output
    assert "tuning:         not compared; the engine predates GET /config/tuning" in result.stdout
    assert "extract budget: not checked; consolidation runs on the engine" in result.stdout
    assert "extract budget: ⚠" not in result.stdout


def test_a_failing_route_warns_and_the_doctor_still_passes(thin: Callable[..., Any]) -> None:
    result = thin(_engine(lambda: httpx.Response(500, json={"detail": "boom"})))
    assert result.exit_code == 0, result.output
    assert "tuning:         ⚠ not compared; GET /config/tuning answered 500: boom" in result.stdout


def test_config_validate_points_a_thin_client_at_the_doctor(thin: Callable[..., Any]) -> None:
    thin(_engine(_report(ParticlesConfig())))  # writes and pins the client config
    result = CliRunner().invoke(app, ["config", "validate"])
    assert "`particles hook doctor` compares the engine's settings" in result.stdout
