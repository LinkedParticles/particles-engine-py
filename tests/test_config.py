# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
    """Private-referent and bulk-migration sources skip live-ontology lookups."""

    def test_default_covers_conversation_journal_and_migration(self) -> None:
        reset_config()
        assert get_config().subjects.skip_live_authorities_source_types == [
            "CONVERSATION",
            "JOURNAL",
            # different reason, same treatment: a per-entity live
            # lookup over a bulk export is slow, network-dependent, and can
            # rewrite a canonical_name the migrating user never chose.
            "MCP_MEMORY_EXPORT",
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
            assert sel.model == "claude-sonnet-5-5"

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

    def test_purpose_list_matches_the_port(self) -> None:
        """``LLM_PURPOSES`` is a literal because config cannot import the port."""
        from typing import get_args

        from particles.config import LLM_PURPOSES
        from particles.llm import LLMPurpose

        assert set(LLM_PURPOSES) == set(get_args(LLMPurpose))
        assert len(LLM_PURPOSES) == len(set(LLM_PURPOSES))

    def test_selection_source_names_where_the_model_came_from(self) -> None:
        """the compiled default can move, so say whether it is in force."""
        from particles.config import LLMConfig, ParticlesConfig

        compiled = LLMConfig()
        assert compiled.selection_source("default") == "compiled default"
        assert compiled.selection_source("extraction") == "compiled default"
        assert compiled.selection_source("use_judge") == "compiled llm.use_judge"

        cfg = ParticlesConfig.model_validate(
            {"llm": {"default": {"model": "m"}, "extraction": {"model": "x"}}}
        )
        assert cfg.llm.selection_source("default") == "llm.default"
        assert cfg.llm.selection_source("semantic_lint") == "llm.default"
        assert cfg.llm.selection_source("extraction") == "llm.extraction"

    def test_routed_elsewhere_lists_only_the_purposes_off_the_default(self) -> None:
        from particles.config import LLMConfig, ProviderSelection

        llm = LLMConfig(
            default=ProviderSelection(model="m"),
            semantic_lint=ProviderSelection(model="small"),
            use_judge=ProviderSelection(model="m"),
        )
        assert llm.routed_elsewhere() == {"semantic_lint": ProviderSelection(model="small")}


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
        assert cfg.llm.for_purpose("extraction").model == "claude-sonnet-5-5"

    def test_new_key_wins_over_legacy(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {
            "extraction": {"model": "legacy"},
            "llm": {"default": {"model": "explicit-new"}},
        }
        _migrate_legacy_keys(raw)
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.llm.default.model == "explicit-new"


class TestQueryAnswerBudgetMigration:
    """``extraction.query_max_tokens`` → ``query.answer_max_tokens``.

    The knob only ever drove the §9.3 answer call; under ``extraction`` it read
    as an extraction cap, and its 1024 default left an extended-thinking model
    no tokens for the answer at all.
    """

    def test_legacy_key_migrates_and_is_popped(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {"extraction": {"query_max_tokens": 2048}}
        _migrate_legacy_keys(raw)
        assert "query_max_tokens" not in raw["extraction"]  # type: ignore[operator]
        cfg = ParticlesConfig.model_validate(raw)
        # An operator who had deliberately raised it keeps their value.
        assert cfg.query.answer_max_tokens == 2048

    def test_new_key_wins_over_legacy(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {
            "extraction": {"query_max_tokens": 1024},
            "query": {"answer_max_tokens": 8192},
        }
        _migrate_legacy_keys(raw)
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.query.answer_max_tokens == 8192

    def test_defaults_leave_headroom_for_a_thinking_model(self) -> None:
        from particles.config import ParticlesConfig

        query = ParticlesConfig().query
        # The 1024 that produced the measured failures is gone…
        assert query.answer_max_tokens == 4096
        # …and the retry budget is strictly larger, or it would re-issue an
        # identical call and reproduce the same empty reply.
        assert query.answer_retry_max_tokens > query.answer_max_tokens

    def test_extraction_no_longer_carries_the_knob(self) -> None:
        from particles.config import ExtractionConfig

        assert not hasattr(ExtractionConfig(), "query_max_tokens")


class TestSharedEstimateKeys:
    """The price table and the prompt overhead are shared by the audit and the benchmarks."""

    def test_list_prices_ship_for_the_default_model(self) -> None:
        from particles.config import ParticlesConfig

        llm = ParticlesConfig().llm
        price = llm.price_for(llm.for_purpose("extraction"))
        assert price is not None and price.input > 0 and price.output > 0

    def test_operator_entries_merge_over_the_shipped_prices(self) -> None:
        from particles.config import ParticlesConfig

        cfg = ParticlesConfig.model_validate(
            {"llm": {"price_per_mtok": {"local:llama3.1": {"input": 0.1, "output": 0.2}}}}
        )
        assert cfg.llm.price_per_mtok["local:llama3.1"].output == 0.2
        assert "claude-sonnet-5" in cfg.llm.price_per_mtok

    def test_current_sonnet_is_priced_out_of_the_box(self) -> None:
        from particles.config import ParticlesConfig, ProviderSelection

        price = ParticlesConfig().llm.price_for(ProviderSelection(model="claude-sonnet-5-5"))
        assert price is not None and (price.input, price.output) == (2.00, 10.00)

    def test_price_for_prefers_the_provider_qualified_key(self) -> None:
        from particles.config import ParticlesConfig, ProviderSelection

        cfg = ParticlesConfig.model_validate(
            {"llm": {"price_per_mtok": {"anthropic:claude-sonnet-5": {"input": 9, "output": 9}}}}
        )
        price = cfg.llm.price_for(ProviderSelection(model="claude-sonnet-5"))
        assert price is not None and price.input == 9
        assert cfg.llm.price_for(ProviderSelection(provider="local", model="llama3.1")) is None

    def test_benchmark_price_table_migrates_to_llm(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {
            "benchmark_memory": {"price_per_mtok": {"m-a": {"input": 1, "output": 2}}},
            "llm": {"price_per_mtok": {"m-b": {"input": 3, "output": 4}}},
        }
        _migrate_legacy_keys(raw)
        assert "price_per_mtok" not in raw["benchmark_memory"]  # type: ignore[operator]
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.llm.price_per_mtok["m-a"].output == 2
        assert cfg.llm.price_per_mtok["m-b"].output == 4

    def test_benchmark_batch_discount_migrates_to_llm(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {"benchmark_memory": {"batch_discount": 0.25}}
        _migrate_legacy_keys(raw)
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.llm.batch_discount == 0.25

    def test_rot_prompt_overhead_migrates_to_extraction(self) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {
            "benchmark_rot": {"estimate_extraction_prompt_overhead_tokens": 1234}
        }
        _migrate_legacy_keys(raw)
        cfg = ParticlesConfig.model_validate(raw)
        assert cfg.extraction.estimate_prompt_overhead_tokens == 1234

    def test_per_type_output_scale_and_cache_reads_merge_over_the_shipped_entries(self) -> None:
        from pydantic import ValidationError

        from particles.config import AuditConfig, LLMConfig, ProviderSelection

        audit = AuditConfig(estimate_output_scale_by_source_type={"WEB_PAGE": 0.8})
        assert audit.estimate_output_scale_by_source_type == {
            "CONVERSATION": 0.4,
            "WEB_PAGE": 0.8,
        }
        with pytest.raises(ValidationError):
            AuditConfig(estimate_output_scale_by_source_type={"CONVERSATION": -0.1})

        llm = LLMConfig(cache_read_price_multiplier_by_model={"claude-haiku-5-5": 0.2})
        read = llm.cache_read_multiplier_for
        assert read(ProviderSelection(model="claude-sonnet-5-5")) == 0.05
        assert read(ProviderSelection(model="claude-opus-5-5")) == 0.05
        assert read(ProviderSelection(model="claude-fable-5-1")) == 0.025
        assert read(ProviderSelection(model="claude-haiku-5-5")) == 0.2
        assert read(ProviderSelection(model="claude-sonnet-5")) == 0.1
        with pytest.raises(ValidationError):
            LLMConfig(cache_read_price_multiplier_by_model={"claude-sonnet-5-5": -1})

    def test_audit_per_model_output_scale_must_be_non_negative(self) -> None:
        from pydantic import ValidationError

        from particles.config import AuditConfig

        with pytest.raises(ValidationError):
            AuditConfig(estimate_output_scale_by_model={"claude-sonnet-5": -1})
        with pytest.raises(ValidationError):
            AuditConfig(estimate_output_spread=1.0)
        with pytest.raises(ValidationError):
            AuditConfig(estimate_extraction_output_tokens_per_second=0)

    def test_audit_per_call_output_entries_carry_over_as_scales(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.config import ParticlesConfig, _migrate_legacy_keys

        raw: dict[str, object] = {
            "audit": {
                "estimate_output_tokens_per_extraction_call": 7000,
                "estimate_output_tokens_per_extraction_call_by_model": {
                    "claude-haiku-4-5": 3500,
                    "m-kept": 1,
                },
                "estimate_output_scale_by_model": {"m-kept": 0.9},
                "estimate_seconds_per_extraction_call": 50,
            }
        }
        with caplog.at_level("WARNING", logger="particles.config"):
            _migrate_legacy_keys(raw)
        audit = raw["audit"]
        assert isinstance(audit, dict)
        assert "estimate_output_tokens_per_extraction_call" not in audit
        assert "estimate_seconds_per_extraction_call" not in audit
        cfg = ParticlesConfig.model_validate(raw)
        # A ratio against the old flat 7,000; an entry already under the new
        # key wins.
        # The shipped entries stay beside the carried-over ones.
        from particles.config import _SHIPPED_OUTPUT_SCALE

        assert cfg.audit.estimate_output_scale_by_model == {
            **_SHIPPED_OUTPUT_SCALE,
            "claude-haiku-4-5": 0.5,
            "m-kept": 0.9,
        }
        assert "estimate_output_scale_by_model" in caplog.text
        assert "no longer read" in caplog.text


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


class TestMemoryBenchmarkConfig:
    """the published number describes the product, not a lab build."""

    def test_top_k_default_matches_the_product_query_default(self) -> None:
        # The benchmark's retrieval depth mirrors the shipped query default
        # (the API request model; the CLI --top-k and MCP query tool carry the
        # same literal). A lower k is a disclosed ablation on the run tuple,
        # never the default, so the two literals must not drift apart.
        from particles.config import MemoryBenchmarkConfig
        from particles.core.schema import QueryRequest

        assert MemoryBenchmarkConfig().top_k == QueryRequest.model_fields["top_k"].default
        assert MemoryBenchmarkConfig().top_k == 40


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


class TestConfigResolution:
    """to §3: ``PARTICLES_CONFIG``, then the user file, then defaults.

    The working directory and its ancestors are never read for settings; a
    Particles ``config.yaml`` found there is reported, once per process.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Drop the session-wide pin, give the test its own home, clear the once-flags.

        conftest pins ``PARTICLES_CONFIG`` to an empty file so the developer's
        config never leaks into a test; every case here needs the resolution
        that pin short-circuits, so ``HOME`` must be private too.
        """
        import particles.config as config_mod

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
        monkeypatch.setattr(config_mod, "_warned_resolution", set())
        monkeypatch.setattr(config_mod, "_resolution_warnings", [])
        return home

    @staticmethod
    def _user_file(home: Path, text: str) -> Path:
        path = home / ".particles" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    @staticmethod
    def _repo(root: Path, *, git_is_file: bool = False) -> None:
        if git_is_file:
            (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
        else:
            (root / ".git").mkdir()

    def test_a_working_directory_config_is_never_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Findings D1/D6/D8/D15: a cloned repository's file configures nothing."""
        from particles.config import _find_config_file, _load_config

        repo = tmp_path / "clone"
        repo.mkdir()
        self._repo(repo)
        (repo / "config.yaml").write_text(
            "engine:\n  base_url: http://127.0.0.1:18751\n"
            "mcp:\n  write:\n    enabled_stores: [default]\n"
            "claude_code:\n  harvest:\n    allow_remote: true\n"
            "agent_memory:\n  projection:\n    manifest: /etc/passwd\n"
        )
        monkeypatch.chdir(repo)

        assert _find_config_file() is None
        cfg = _load_config()
        assert cfg.engine.base_url is None
        assert cfg.mcp.write.enabled_stores == []
        assert cfg.claude_code.harvest.allow_remote is False
        assert cfg.agent_memory.projection.manifest != "/etc/passwd"

    def test_an_ancestor_config_is_never_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import _find_config_file

        self._repo(tmp_path)
        (tmp_path / "config.yaml").write_text("engine:\n  base_url: http://evil\n")
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)

        assert _find_config_file() is None

    def test_the_user_file_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated: Path
    ) -> None:
        from particles.config import _find_config_file, _load_config

        user = self._user_file(_isolated, "engine:\n  timeout_seconds: 7\n")
        monkeypatch.chdir(tmp_path)

        assert _find_config_file() == user
        assert _load_config().engine.timeout_seconds == 7

    def test_particles_config_beats_the_user_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated: Path
    ) -> None:
        from particles.config import _find_config_file

        self._user_file(_isolated, "")
        named = tmp_path / "named.yaml"
        named.write_text("")
        monkeypatch.setenv("PARTICLES_CONFIG", str(named))

        assert _find_config_file() == named

    def test_a_missing_explicit_file_does_not_fall_through_and_warns_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        _isolated: Path,
    ) -> None:
        """§3: absolute authority, now reported, with the quoted-tilde hint."""
        from particles.config import _find_config_file, _load_config, config_resolution_warnings

        self._user_file(_isolated, "engine:\n  timeout_seconds: 7\n")
        monkeypatch.setenv("PARTICLES_CONFIG", "~/src/absent/config.yaml")

        assert _find_config_file() is None
        with caplog.at_level("WARNING", logger="particles.config"):
            assert _load_config().engine.timeout_seconds == 60.0
            _load_config()
        messages = [
            r.getMessage() for r in caplog.records if "PARTICLES_CONFIG names" in r.getMessage()
        ]
        assert len(messages) == 1
        assert "does not expand `~`" in messages[0]
        assert config_resolution_warnings() == messages

    def test_a_nearby_particles_config_warns_once_and_names_the_remedies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.config import _load_config

        self._repo(tmp_path)
        (tmp_path / "config.yaml").write_text("storage:\n  blob_dir: /x\n")
        sub = tmp_path / "scripts"
        sub.mkdir()
        monkeypatch.chdir(sub)

        with caplog.at_level("WARNING", logger="particles.config"):
            _load_config()
            _load_config()
        messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Ignoring")]
        assert len(messages) == 1
        assert f"PARTICLES_CONFIG={tmp_path / 'config.yaml'}" in messages[0]
        assert ".particles/config.yaml" in messages[0]

    def test_another_tools_config_is_not_mentioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import nearby_unused_config

        self._repo(tmp_path)
        (tmp_path / "config.yaml").write_text("baseURL: https://example.org\ntheme: x\n")
        monkeypatch.chdir(tmp_path)

        assert nearby_unused_config() is None

    def test_the_look_stops_at_the_repository_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worktree's ``.git`` file bounds it, as walk was bounded."""
        from particles.config import nearby_unused_config

        (tmp_path / "config.yaml").write_text("storage: {}\n")
        wt = tmp_path / "worktree"
        wt.mkdir()
        self._repo(wt, git_is_file=True)
        monkeypatch.chdir(wt)

        assert nearby_unused_config() is None

    def test_particles_config_set_skips_the_look(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.config import _load_config, config_resolution_warnings

        (tmp_path / "config.yaml").write_text("storage: {}\n")
        named = tmp_path / "named.yaml"
        named.write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PARTICLES_CONFIG", str(named))

        _load_config()
        assert config_resolution_warnings() == []

    def test_only_a_read_file_can_silence_the_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated: Path
    ) -> None:
        from particles.config import _load_config, config_resolution_warnings

        repo = tmp_path / "repo"
        repo.mkdir()
        self._repo(repo)
        (repo / "config.yaml").write_text("cli:\n  warn_unused_config: false\n")
        monkeypatch.chdir(repo)

        _load_config()
        assert len(config_resolution_warnings()) == 1

        import particles.config as config_mod

        monkeypatch.setattr(config_mod, "_warned_resolution", set())
        monkeypatch.setattr(config_mod, "_resolution_warnings", [])
        self._user_file(_isolated, "cli:\n  warn_unused_config: false\n")
        _load_config()
        assert config_resolution_warnings() == []

    def test_the_user_file_itself_is_never_reported(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path
    ) -> None:
        from particles.config import nearby_unused_config

        user = self._user_file(_isolated, "storage: {}\n")
        monkeypatch.chdir(user.parent)

        assert nearby_unused_config(user) is None

    def test_describe_names_origin_missing_and_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated: Path
    ) -> None:
        from particles.config import describe_config_resolution

        self._repo(tmp_path)
        (tmp_path / "config.yaml").write_text("storage: {}\n")
        monkeypatch.chdir(tmp_path)
        assert describe_config_resolution() == (None, "none", None, tmp_path / "config.yaml")

        user = self._user_file(_isolated, "")
        assert describe_config_resolution().origin == "user file"
        assert describe_config_resolution().path == user

        monkeypatch.setenv("PARTICLES_CONFIG", str(tmp_path / "absent.yaml"))
        resolution = describe_config_resolution()
        assert (resolution.origin, resolution.missing) == ("none", tmp_path / "absent.yaml")


def test_retired_agent_extraction_keys_are_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """was deprecated: its keys are dropped once, never an error."""
    from particles.config import ParticlesConfig, _migrate_legacy_keys

    raw: dict[str, object] = {
        "claude_code": {"observer_scope": "project", "agent_extraction": {"mode": "off"}},
        "extraction": {"agent_expected_claims_per_kchar": 1.9},
    }
    with caplog.at_level("WARNING", logger="particles.config"):
        _migrate_legacy_keys(raw)
    assert raw["claude_code"] == {"observer_scope": "project"}
    assert raw["extraction"] == {}
    assert "claude_code.agent_extraction" in caplog.text
    ParticlesConfig.model_validate(raw)
