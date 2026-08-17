"""Tests for ``particles/config.py`` — default values and env-var overrides.

Scoped to the knobs whose wiring is easy to get wrong (the ``_ENV_OVERRIDES``
tuples). The config loader itself is exercised indirectly across the suite;
these tests pin behaviour that other code relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from particles.config import get_config, reset_config


class TestApiBindHost:
    """the fail-closed bind-host knob and its env override."""

    def test_default_is_loopback(self) -> None:
        reset_config()
        assert get_config().api.bind_host == "127.0.0.1"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_BIND_HOST", "0.0.0.0")
        reset_config()
        assert get_config().api.bind_host == "0.0.0.0"


class TestEngineConfig:
    """the thin-client → remote-engine connection block."""

    def test_base_url_defaults_to_none(self) -> None:
        reset_config()
        assert get_config().engine.base_url is None

    def test_timeout_default(self) -> None:
        reset_config()
        assert get_config().engine.timeout_seconds == 60.0

    def test_base_url_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://mac-mini:8000")
        reset_config()
        assert get_config().engine.base_url == "http://mac-mini:8000"

    def test_timeout_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_ENGINE_TIMEOUT_SECONDS", "12.5")
        reset_config()
        assert get_config().engine.timeout_seconds == 12.5


class TestEmbeddingsConfig:
    """the embedding-encoder block; tqdm progress bars off by default."""

    def test_progress_bars_default_off(self) -> None:
        reset_config()
        assert get_config().embeddings.progress_bars is False

    def test_progress_bars_env_override_coerces_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_EMBEDDINGS_PROGRESS_BARS", "1")
        reset_config()
        assert get_config().embeddings.progress_bars is True


class TestObservabilityConfig:
    """the OpenTelemetry observability block (off by default)."""

    def test_defaults_off(self) -> None:
        reset_config()
        o = get_config().observability
        assert o.enabled is False
        assert o.exporter == "console"
        assert o.service_name == "particles"
        assert o.sample_ratio == 1.0

    def test_enabled_env_override_coerces_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENABLED", "true")
        reset_config()
        assert get_config().observability.enabled is True

    def test_exporter_and_endpoint_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_EXPORTER", "otlp")
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_ENDPOINT", "http://localhost:4318")
        reset_config()
        o = get_config().observability
        assert o.exporter == "otlp"
        assert o.endpoint == "http://localhost:4318"

    def test_sample_ratio_env_override_coerces_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_SAMPLE_RATIO", "0.25")
        reset_config()
        assert get_config().observability.sample_ratio == 0.25

    def test_sample_ratio_out_of_range_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OBSERVABILITY_SAMPLE_RATIO", "1.5")
        reset_config()
        with pytest.raises(ValueError):
            get_config()


class TestValidateConfig:
    """: the singleton-free validate seam."""

    def test_no_file_returns_none_path_and_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import validate_config

        monkeypatch.setenv("PARTICLES_CONFIG", "/nonexistent-validate-test.yaml")
        path, cfg = validate_config()
        assert path is None
        assert cfg.trust.reviewer_trust_rank == 0.8  # compiled-in default

    def test_valid_file_returns_path_and_value(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import validate_config

        cfg_file = tmp_path / "config.yaml"  # type: ignore[operator]
        cfg_file.write_text("trust:\n  reviewer_trust_rank: 0.9\n")
        monkeypatch.setenv("PARTICLES_CONFIG", str(cfg_file))
        path, cfg = validate_config()
        assert path == cfg_file
        assert cfg.trust.reviewer_trust_rank == 0.9

    def test_invalid_value_raises_validation_error(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import validate_config

        cfg_file = tmp_path / "config.yaml"  # type: ignore[operator]
        cfg_file.write_text("trust:\n  reviewer_trust_rank: 99\n")
        monkeypatch.setenv("PARTICLES_CONFIG", str(cfg_file))
        with pytest.raises(ValidationError):
            validate_config()

    def test_does_not_touch_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # validate_config() must read the file on disk, never the cached config.
        from particles.config import validate_config

        monkeypatch.setenv("PARTICLES_CONFIG", "/nonexistent-validate-test.yaml")
        reset_config()
        before = get_config()
        validate_config()
        assert get_config() is before  # singleton untouched


class TestSkipLiveAuthoritiesSourceTypes:
    """Conversational / journal-like sources skip live-ontology lookups."""

    def test_default_covers_conversation_and_journal(self) -> None:
        reset_config()
        assert get_config().subjects.skip_live_authorities_source_types == [
            "CONVERSATION",
            "JOURNAL",
        ]


class TestReconciliationStoreMode:
    """the consensus-mode knob and its env override."""

    def test_default_is_single(self) -> None:
        reset_config()
        assert get_config().reconciliation.store_mode == "single"

    def test_env_override_to_multi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECONCILIATION_STORE_MODE", "multi")
        reset_config()
        assert get_config().reconciliation.store_mode == "multi"

    def test_invalid_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # store_mode is a Literal["single", "multi"]; anything else must fail
        # validation rather than silently degrade to a default.
        monkeypatch.setenv("RECONCILIATION_STORE_MODE", "consensus")
        with pytest.raises(ValidationError):
            reset_config()
            get_config()


class TestMcpWriteGranularityKnobs:
    """claim-granularity soft-gate config knobs."""

    def test_defaults(self) -> None:
        reset_config()
        w = get_config().mcp.write
        assert w.max_assertion_chars == 320
        assert w.max_assertion_sentences == 3

    def test_negative_rejected(self) -> None:
        from particles.config import McpWriteConfig

        with pytest.raises(ValidationError):
            McpWriteConfig(max_assertion_chars=-1)


class TestLLMConfig:
    """per-purpose completion-provider selection."""

    def test_defaults_all_purposes_resolve_to_sonnet(self) -> None:
        reset_config()
        llm = get_config().llm
        for purpose in (
            "extraction",
            "semantic_lint",
            "query_response",
            "synthesis",
            "benchmark",
        ):
            sel = llm.for_purpose(purpose)
            assert sel.provider == "anthropic"
            assert sel.model == "claude-sonnet-4-6"

    def test_for_purpose_falls_back_to_default(self) -> None:
        from particles.config import LLMConfig, ProviderSelection

        llm = LLMConfig(default=ProviderSelection(model="default-model"))
        # No per-purpose override → default.
        assert llm.for_purpose("extraction").model == "default-model"

    def test_for_purpose_honours_override(self) -> None:
        from particles.config import LLMConfig, ProviderSelection

        llm = LLMConfig(
            default=ProviderSelection(model="default-model"),
            synthesis=ProviderSelection(model="synthesis-model"),
        )
        assert llm.for_purpose("synthesis").model == "synthesis-model"
        assert llm.for_purpose("extraction").model == "default-model"


class TestLLMModelKeyMigration:
    """extraction.model → llm.default.model; wiki.model → llm.synthesis.model."""

    def test_extraction_model_migrates_to_llm_default(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {"extraction": {"model": "legacy-opus"}}
        _migrate_legacy_keys(raw)
        assert "model" not in raw["extraction"]  # type: ignore[operator]
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.llm.default.model == "legacy-opus"
        assert cfg.llm.for_purpose("extraction").model == "legacy-opus"

    def test_wiki_model_migrates_to_llm_synthesis(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {"wiki": {"model": "legacy-synth"}}
        _migrate_legacy_keys(raw)
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.llm.for_purpose("synthesis").model == "legacy-synth"
        # Other purposes are unaffected by the wiki migration.
        assert cfg.llm.for_purpose("extraction").model == "claude-sonnet-4-6"

    def test_new_key_wins_over_legacy(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {
            "extraction": {"model": "legacy"},
            "llm": {"default": {"model": "explicit-new"}},
        }
        _migrate_legacy_keys(raw)
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.llm.default.model == "explicit-new"


class TestSubjectLinkThresholds:
    """the abstain ≤ suppress invariant on SubjectsConfig."""

    def test_defaults(self) -> None:
        from particles.config import ParticlesConfig

        subjects = ParticlesConfig().subjects
        assert subjects.external_link_abstain_threshold == 0.15
        # The whole point: abstain floor sits at or below the suppress/flag floor.
        assert subjects.external_link_abstain_threshold <= subjects.wikidata_link_suppress_threshold

    def test_abstain_above_suppress_rejected(self) -> None:
        from particles.config import SubjectsConfig

        with pytest.raises(ValidationError):
            SubjectsConfig(
                external_link_abstain_threshold=0.30,
                wikidata_link_suppress_threshold=0.25,
            )

    def test_abstain_equal_to_suppress_allowed(self) -> None:
        from particles.config import SubjectsConfig

        cfg = SubjectsConfig(
            external_link_abstain_threshold=0.25,
            wikidata_link_suppress_threshold=0.25,
        )
        assert cfg.external_link_abstain_threshold == 0.25


class TestTrustedProxiesValidator:
    """Security review F19 — reject overly-broad / loopback trusted_proxies ranges."""

    def test_empty_default_is_valid(self) -> None:
        from particles.config import ApiConfig

        assert ApiConfig().trusted_proxies == []
        assert ApiConfig(trusted_proxies=[]).trusted_proxies == []

    def test_normal_cidr_passes(self) -> None:
        from particles.config import ApiConfig

        cfg = ApiConfig(trusted_proxies=["10.0.0.0/8"])
        assert cfg.trusted_proxies == ["10.0.0.0/8"]

    def test_single_ip_passes(self) -> None:
        from particles.config import ApiConfig

        cfg = ApiConfig(trusted_proxies=["192.168.1.10"])
        assert cfg.trusted_proxies == ["192.168.1.10"]

    def test_ipv6_cidr_passes(self) -> None:
        from particles.config import ApiConfig

        cfg = ApiConfig(trusted_proxies=["2001:db8::/32"])
        assert cfg.trusted_proxies == ["2001:db8::/32"]

    def test_everything_range_v4_rejected(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["0.0.0.0/0"])

    def test_everything_range_v6_rejected(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["::/0"])

    def test_loopback_ip_rejected(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["127.0.0.1"])

    def test_loopback_range_rejected(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["127.0.0.0/8"])

    def test_loopback_ipv6_rejected(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["::1/128"])

    def test_garbage_rejected(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["not-an-ip"])

    def test_one_bad_entry_rejects_whole_list(self) -> None:
        from particles.config import ApiConfig

        with pytest.raises(ValidationError):
            ApiConfig(trusted_proxies=["10.0.0.0/8", "0.0.0.0/0"])


class TestConsolidationConfig:
    """the cycle's cadence/cost knobs, and the §5 structured-output knob."""

    def test_defaults(self) -> None:
        from particles.config import ConsolidationConfig, ParticlesConfig

        cfg = ConsolidationConfig()
        assert cfg.min_interval_hours == 20
        assert cfg.extract_pending is True
        assert cfg.max_pending_entries == 20
        assert cfg.semantic is True
        assert cfg.lock_timeout_minutes == 120
        assert isinstance(ParticlesConfig().consolidation, ConsolidationConfig)

    def test_local_structured_output_knob(self) -> None:
        from particles.config import LocalProviderConfig

        assert LocalProviderConfig().structured_output == "auto"
        assert LocalProviderConfig(structured_output="off").structured_output == "off"
        with pytest.raises(ValidationError):
            LocalProviderConfig(structured_output="json_object")  # type: ignore[arg-type]


class TestAbstractionConfig:
    """the abstraction-promotion pass knobs."""

    def test_defaults(self) -> None:
        from particles.config import AbstractionConfig, ConsolidationConfig

        cfg = AbstractionConfig()
        assert cfg.enabled is False
        assert cfg.mode == "propose"
        assert cfg.min_cluster_size == 3
        assert cfg.min_source_age_days == 14
        assert cfg.max_promotions_per_run == 5
        assert cfg.max_depth == 1
        assert cfg.require_entailment is True
        assert cfg.source_demotion == "suppress_in_projection"
        assert cfg.stale_support_discount == 0.5
        assert cfg.cluster_similarity_threshold == 0.55
        assert isinstance(ConsolidationConfig().abstraction, AbstractionConfig)

    def test_mode_literal_rejects_unknown(self) -> None:
        from particles.config import AbstractionConfig

        with pytest.raises(ValidationError):
            AbstractionConfig(mode="yolo")  # type: ignore[arg-type]

    def test_llm_abstraction_purpose_falls_back_to_default(self) -> None:
        from particles.config import LLMConfig, ProviderSelection

        cfg = LLMConfig()
        assert cfg.for_purpose("abstraction") is cfg.default
        override = LLMConfig(abstraction=ProviderSelection(model="local-x"))
        assert override.for_purpose("abstraction").model == "local-x"


# ---------------------------------------------------------------------------
# OwnerLensConfig
# ---------------------------------------------------------------------------


def test_owner_lens_defaults_are_inert() -> None:
    """The shipped default must change no ordering: no viewer, zero lift."""
    from particles.config import OwnerLensConfig

    cfg = OwnerLensConfig()
    assert cfg.enabled is True
    assert cfg.subjects == []
    assert cfg.rank_lift == 0.0


def test_owner_lens_rank_lift_must_be_non_negative() -> None:
    """Promotion-only is a config-level constraint, not just a runtime guard."""
    import pytest
    from pydantic import ValidationError

    from particles.config import OwnerLensConfig

    with pytest.raises(ValidationError):
        OwnerLensConfig(rank_lift=-0.01)


def test_owner_lens_mounted_on_particles_config() -> None:
    from particles.config import ParticlesConfig

    assert ParticlesConfig().owner_lens.rank_lift == 0.0


class TestNamedProviderRegistry:
    """llm.providers wiring, validation, and the llm.local migration."""

    def test_compiled_default_local_entry_always_present(self) -> None:
        from particles.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.providers["local"].base_url == "http://localhost:11434/v1"
        assert cfg.providers["local"].adapter == "openai_compat"

    def test_dangling_provider_name_fails_config_load(self) -> None:
        from pydantic import ValidationError

        from particles.config import LLMConfig, ProviderSelection

        with pytest.raises(ValidationError, match="neither 'anthropic' nor a key"):
            LLMConfig(extraction=ProviderSelection(provider="openai", model="m"))

    def test_dangling_adapter_kind_fails_at_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Kind validation is resolution-time, not load-time: a config → llm
        # import would mint a subpackage cycle (see LLMConfig's validator).
        from particles.config import (
            LLMConfig,
            OpenAICompatProviderConfig,
            ParticlesConfig,
            ProviderSelection,
        )
        from particles.llm.registry import CompletionError, get_provider

        cfg = ParticlesConfig(
            llm=LLMConfig(
                extraction=ProviderSelection(provider="x", model="m"),
                providers={"x": OpenAICompatProviderConfig(adapter="bogus")},
            )
        )
        monkeypatch.setattr("particles.config.get_config", lambda: cfg)
        with pytest.raises(CompletionError, match="unregistered adapter kind"):
            get_provider("extraction")

    def test_anthropic_is_a_reserved_provider_name(self) -> None:
        from pydantic import ValidationError

        from particles.config import LLMConfig, OpenAICompatProviderConfig

        with pytest.raises(ValidationError, match="reserved provider name"):
            LLMConfig(providers={"anthropic": OpenAICompatProviderConfig()})

    def test_legacy_llm_local_block_migrates_to_providers(self) -> None:
        from particles.config import LLMConfig, OpenAICompatProviderConfig

        cfg = LLMConfig(local=OpenAICompatProviderConfig(base_url="http://h:1/v1"))
        assert cfg.providers["local"].base_url == "http://h:1/v1"

    def test_explicit_providers_local_wins_over_legacy_block(self) -> None:
        from particles.config import LLMConfig, OpenAICompatProviderConfig

        cfg = LLMConfig(
            local=OpenAICompatProviderConfig(base_url="http://legacy/v1"),
            providers={"local": OpenAICompatProviderConfig(base_url="http://explicit/v1")},
        )
        assert cfg.providers["local"].base_url == "http://explicit/v1"

    def test_local_provider_config_is_an_alias(self) -> None:
        from particles.config import LocalProviderConfig, OpenAICompatProviderConfig

        assert LocalProviderConfig is OpenAICompatProviderConfig


class TestConfigDiscovery:
    """the git-style upward walk for ``config.yaml``."""

    @pytest.fixture(autouse=True)
    def _no_explicit_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drop the session-wide ``PARTICLES_CONFIG`` pin so discovery can run.

        conftest sets it unconditionally to a nonexistent path (so the dev's
        own config never leaks into a test); every case here exercises the
        fallback path that pin short-circuits.
        """
        monkeypatch.delenv("PARTICLES_CONFIG", raising=False)

    @staticmethod
    def _repo(root: Path, *, git_is_file: bool) -> None:
        """Mark ``root`` as a repository root the walk must stop at.

        ``git_is_file=True`` is the git-worktree (and submodule) shape, where
        ``.git`` is a file holding a ``gitdir:`` pointer rather than a
        directory — the case that motivated the walk in the first place.
        """
        if git_is_file:
            (root / ".git").write_text("gdir: /elsewhere/.git/worktrees/wt\n")
        else:
            (root / ".git").mkdir()

    def test_cwd_config_still_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-ADR behaviour is untouched: a local config.yaml loads."""
        from particles.config import _find_config_file

        (tmp_path / "config.yaml").write_text("storage:\n  blob_dir: /local\n")
        monkeypatch.chdir(tmp_path)

        assert _find_config_file() == tmp_path / "config.yaml"

    def test_cwd_config_beats_an_ancestor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import _find_config_file

        (tmp_path / "config.yaml").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "config.yaml").write_text("")
        monkeypatch.chdir(sub)

        assert _find_config_file() == sub / "config.yaml"

    @pytest.mark.parametrize("git_is_file", [False, True])
    def test_ancestor_config_is_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_is_file: bool
    ) -> None:
        """The fix: a subdirectory process loads the repo-root config.

        Parametrised over both ``.git`` shapes because a git *worktree* — the
        launch context behind the 2026-07-18 sharding incident — has ``.git``
        as a file, and an ``is_dir()`` bound would walk straight past it.
        """
        from particles.config import _find_config_file

        self._repo(tmp_path, git_is_file=git_is_file)
        (tmp_path / "config.yaml").write_text("")
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)

        assert _find_config_file() == tmp_path / "config.yaml"

    def test_nearest_ancestor_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import _find_config_file

        self._repo(tmp_path, git_is_file=False)
        (tmp_path / "config.yaml").write_text("")
        mid = tmp_path / "a"
        mid.mkdir()
        (mid / "config.yaml").write_text("")
        deep = mid / "b"
        deep.mkdir()
        monkeypatch.chdir(deep)

        assert _find_config_file() == mid / "config.yaml"

    @pytest.mark.parametrize("git_is_file", [False, True])
    def test_walk_stops_at_the_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_is_file: bool
    ) -> None:
        """A config above the repo root is never inherited (the $HOME case)."""
        from particles.config import _find_config_file

        (tmp_path / "config.yaml").write_text("")  # stands in for ~/config.yaml
        repo = tmp_path / "repo"
        repo.mkdir()
        self._repo(repo, git_is_file=git_is_file)
        deep = repo / "src"
        deep.mkdir()
        monkeypatch.chdir(deep)

        assert _find_config_file() is None

    def test_repo_root_config_is_still_found_when_it_holds_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound halts *after* examining the ``.git`` directory, not before."""
        from particles.config import _find_config_file

        self._repo(tmp_path, git_is_file=True)
        (tmp_path / "config.yaml").write_text("")
        deep = tmp_path / "src"
        deep.mkdir()
        monkeypatch.chdir(deep)

        assert _find_config_file() == tmp_path / "config.yaml"

    def test_no_config_anywhere_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import _find_config_file

        self._repo(tmp_path, git_is_file=False)
        deep = tmp_path / "src"
        deep.mkdir()
        monkeypatch.chdir(deep)

        assert _find_config_file() is None

    def test_explicit_env_var_still_wins_over_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import _find_config_file

        (tmp_path / "config.yaml").write_text("")
        named = tmp_path / "elsewhere.yaml"
        named.write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PARTICLES_CONFIG", str(named))

        assert _find_config_file() == named

    def test_missing_explicit_config_does_not_fall_through_to_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``PARTICLES_CONFIG`` stays absolute authority — including when wrong.

        This is the escape hatch the ADR promises anyone who wants the old
        defaults-only behaviour, so the walk must not quietly rescue a typo.
        """
        from particles.config import _find_config_file

        (tmp_path / "config.yaml").write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PARTICLES_CONFIG", str(tmp_path / "absent.yaml"))

        assert _find_config_file() is None
