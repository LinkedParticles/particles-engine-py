"""Conformance Profile artifact + runner tests.

Covers ``particles/conformance/profile.py`` (the loader/validator),
``particles/conformance/runner.py`` (the L2/L3 self-certification), and the
single-source-of-truth **drift guard**: every constant published in
``artifacts/conformance/profile.yaml`` must still equal the live
``particles/config.py`` default at its declared ``config_path``. If someone
re-tunes a config default without updating the Profile (or vice versa), this
test fails — that is the mechanism that keeps "the Profile restates
the constants as the single source of truth" claim honest.

The L3 similarity self-check needs the real encoder, so it is ``integration``
(network/model download); the rest is pure and runs in the unit tier.
"""

from __future__ import annotations

import pytest

from particles.conformance.profile import (
    ConformanceProfile,
    load_profile,
    profile_path,
    resolve_config_value,
)
from particles.conformance.runner import run_l2, run_l3

_TOL = 1.0e-9


def test_profile_loads_and_validates() -> None:
    p = load_profile()
    assert p.profile_version == "1.2"
    assert p.float_tolerance == _TOL
    assert p.embedding_profile.reference.model == "all-MiniLM-L6-v2"
    assert p.embedding_profile.reference.dim == 384
    assert p.embedding_profile.reference.normalization == "l2"
    assert p.similarity_scale.clamp_negative is True
    assert p.similarity_vectors_ref == "similarity_vectors.json"
    assert len(p.all_constants()) >= 20
    # the four §4 formula families all carry vectors
    assert p.test_vectors.effective_confidence
    assert p.test_vectors.recency_factor
    assert p.test_vectors.calibration_apply
    assert p.test_vectors.noisy_or_merge
    # …and the three §5 algorithm families
    assert p.test_vectors.conflict_ladder
    assert p.test_vectors.context_fingerprint
    assert p.test_vectors.cascade_gate
    assert p.test_vectors.cascade_cap


def test_published_calibration_formula_matches_the_declared_transform() -> None:
    """§4's formula *string* must agree with the transform the artifact declares.

    ``formulas`` is published prose: the loader parses it as ``dict[str, str]``
    and nothing interprets it, so it is the one part of the profile an L2 run
    cannot falsify — every vector can pass while the formula printed beside them
    says something else. That gap shipped a real defect. The apply transform moved
    to logit space and synced ``calibration.transform`` plus all five
    ``calibration_apply`` vectors, but left the formula string naming the retired
    ``clamp(raw / T, 0, 1)`` form, so through 1.115.0 the artifact told an
    external implementer to build exactly what the vectors reject.
    """
    p = load_profile()
    formula = p.formulas["calibration_apply"]
    # A new transform must extend this guard rather than silently pass it.
    assert p.calibration.transform == "logit", p.calibration.transform
    assert "sigmoid(logit(raw) / T)" in formula, formula
    assert "clamp(" not in formula, f"retired pre-ADR-0238 form still published: {formula}"


def test_every_vector_id_is_unique() -> None:
    """Vector ids are how a failure is reported and how a port cross-references
    its own run — a duplicate would make a FAIL ambiguous."""
    tv = load_profile().test_vectors
    ids = [v["id"] for family in tv.model_dump().values() for v in family]
    assert len(ids) == len(set(ids)), "duplicate test-vector id"


def test_ladder_vectors_expect_real_verdicts() -> None:
    """A typo in `expected` would otherwise fail as a mismatch rather than as
    the malformed vector it is."""
    from particles.core.conflict_resolution import ConflictVerdict

    valid = {v.value for v in ConflictVerdict}
    for lv in load_profile().test_vectors.conflict_ladder:
        assert lv.expected in valid, f"{lv.id}: unknown verdict {lv.expected!r}"


def test_ladder_vectors_cover_every_rung() -> None:
    """The family's whole point is *ordering*, so each rung must have an
    outcome pinned — a rung with no vector is an unguarded reordering."""
    expected = {lv.expected for lv in load_profile().test_vectors.conflict_ladder}
    assert {
        "INCONSISTENT",  # rung 1 (ALEATORY) and rung 3 (default)
        "DOCUMENT_SUPERSEDES",  # rung 1.5
        "DOCUMENT_SUPERSEDED_BY_EXISTING",  # rung 1.5, mirror
        "CORROBORATES",  # rung 1.7 truth-apt gate
        "SUPERSEDES",  # rung 2
        "SUPERSEDED_BY_EXISTING",  # rung 2, mirror
    } <= expected


def test_fingerprint_vectors_are_well_formed() -> None:
    """Expectations are 64-char lowercase hex, and the ACTIVE-filter vector
    really carries non-ACTIVE noise (else fp-04 pins nothing)."""
    vectors = load_profile().test_vectors.context_fingerprint
    for fv in vectors:
        assert len(fv.expected) == 64 and fv.expected == fv.expected.lower()
    assert any(any(p.status != "ACTIVE" for p in fv.particles) for fv in vectors), (
        "no vector exercises the ACTIVE-only filter (§16.1 step 1)"
    )


def test_similarity_vectors_ref_resolves() -> None:
    p = load_profile()
    referenced = profile_path().parent / p.similarity_vectors_ref
    assert referenced.is_file(), f"profile points at missing {referenced}"


def test_constants_match_live_config_defaults() -> None:
    """The drift guard: published value == config default at config_path."""
    p = load_profile()
    mismatches: list[str] = []
    for c in p.all_constants():
        if c.config_path is None:
            continue
        live = float(resolve_config_value(c.config_path))
        if abs(live - c.value) > _TOL:
            mismatches.append(f"{c.name} ({c.config_path}): profile={c.value} config={live}")
    assert not mismatches, "profile.yaml drifted from config.py:\n" + "\n".join(mismatches)


def test_benchmark_match_constants_match_live_code() -> None:
    """The §2.6 / §13.3 benchmark match-semantics constants are
    code-level (no config_path), so the config drift guard skips them. Pin them
    against the live benchmark defaults directly, so profile.yaml can never drift
    behind the equivalence judge / ECE binning the runner actually uses."""
    import inspect

    from particles.benchmark import equivalence, metrics

    p = load_profile()
    by_name = {c.name: c.value for c in p.constants}

    eq_sig = inspect.signature(equivalence.match_emitted_to_expected)
    assert by_name["benchmark_equivalence_threshold"] == eq_sig.parameters["threshold"].default
    assert by_name["benchmark_llm_prefilter"] == eq_sig.parameters["llm_prefilter"].default

    ece_sig = inspect.signature(metrics.compute_calibration_error)
    assert by_name["benchmark_ece_bins"] == ece_sig.parameters["bins"].default


def test_recency_decay_matches_live_config() -> None:
    """Per-source decay table mirrors content_age_decay.sources exactly."""
    p = load_profile()
    live = resolve_config_value(p.recency_decay.config_path_root)
    mismatches: list[str] = []
    for source_type, entry in p.recency_decay.sources.items():
        cfg = live.get(source_type)
        if cfg is None:
            mismatches.append(
                f"{source_type}: absent from config {p.recency_decay.config_path_root}"
            )
            continue
        if abs(cfg.half_life_days - entry.half_life_days) > _TOL:
            mismatches.append(
                f"{source_type}.half_life_days: "
                f"profile={entry.half_life_days} config={cfg.half_life_days}"
            )
        if abs(cfg.floor - entry.floor) > _TOL:
            mismatches.append(f"{source_type}.floor: profile={entry.floor} config={cfg.floor}")
    assert not mismatches, "recency_decay drifted from config.py:\n" + "\n".join(mismatches)


def test_l2_runner_passes_every_vector() -> None:
    """The deterministic surface reproduces every published expectation."""
    report = run_l2(load_profile())
    assert report.status == "PASS", "\n".join(f"{c.name}: {c.detail}" for c in report.failures)
    # exercises all seven families
    names = {c.name.split("/")[0] for c in report.checks}
    assert names == {
        "effective_confidence",
        "recency_factor",
        "calibration_apply",
        "noisy_or_merge",
        "conflict_ladder",
        "context_fingerprint",
        "cascade_gate",
        "cascade_cap",
    }


def test_l2_runner_flags_a_tampered_vector() -> None:
    """A wrong `expected` must FAIL — proves the runner actually checks."""
    p = load_profile()
    tampered = ConformanceProfile.model_validate(p.model_dump())
    tampered.test_vectors.effective_confidence[0].expected = 0.123456
    report = run_l2(tampered)
    assert report.status == "FAIL"
    assert any(c.name.startswith("effective_confidence/") for c in report.failures)


def test_l2_runner_flags_a_tampered_ladder_vector() -> None:
    """The ladder check compares verdicts exactly, not loosely."""
    tampered = ConformanceProfile.model_validate(load_profile().model_dump())
    tampered.test_vectors.conflict_ladder[0].expected = "SUPERSEDES"
    report = run_l2(tampered)
    assert report.status == "FAIL"
    assert any(c.name.startswith("conflict_ladder/") for c in report.failures)


def test_l2_runner_recomputes_the_fingerprint_rather_than_echoing_it() -> None:
    """Perturbing an *input* (not the expectation) must FAIL — otherwise the
    family would pass for any digest at all."""
    tampered = ConformanceProfile.model_validate(load_profile().model_dump())
    # promote one of fp-04's deliberately non-ACTIVE rows into the baseline
    fp04 = next(v for v in tampered.test_vectors.context_fingerprint if v.id == "fp-04")
    next(p for p in fp04.particles if p.status != "ACTIVE").status = "ACTIVE"
    report = run_l2(tampered)
    assert report.status == "FAIL"
    assert any(c.name == "context_fingerprint/fp-04" for c in report.failures)


def test_l2_runner_flags_a_tampered_cascade_vector() -> None:
    """Both cascade families are live, not just parsed."""
    tampered = ConformanceProfile.model_validate(load_profile().model_dump())
    tampered.test_vectors.cascade_gate[0].expected = False
    tampered.test_vectors.cascade_cap[0].expected_processed = 99
    report = run_l2(tampered)
    assert report.status == "FAIL"
    failed = {c.name.split("/")[0] for c in report.failures}
    assert {"cascade_gate", "cascade_cap"} <= failed


@pytest.mark.integration
def test_l3_runner_reproduces_similarity_vectors() -> None:
    """Reference profile lands in every band + top-k (needs the real encoder)."""
    report = run_l3(load_profile())
    if report.status == "SKIPPED":  # pragma: no cover - model-less guard
        pytest.skip("embedding model unavailable")
    assert report.status == "PASS", "\n".join(f"{c.name}: {c.detail}" for c in report.failures)
