"""Live end-to-end run of the acceptance fixture (integration tier).

``particles audit tests/fixtures/memory_dir`` against a real key: harvest the
Claude-Code-shaped memory directory, extract with the live LLM, run the
semantic finders, and render the census. The fixture plants one cross-file
contradiction pair (Jenkins vs GitHub Actions staging deploys, above the 0.6
contradiction gate), one paraphrase pair (the Orion API rate limit, above the
0.88 co-evidential threshold), and one dated-stale fact (an expired TLS
certificate).

Per the ADR acceptance criterion the run MUST render nonzero headline counts
for all three classes and each exemplar MUST carry claim text and a next-verb
line. Extraction and the contradiction probe are LLM-driven, so the
assertions pin the *contract* (labels present, counts nonzero, next verbs,
claim text on exemplars) — never wording of extracted claims.

Runs only with a developer key (``uv run pytest tests/`` without the
``-m "not integration"`` filter); CI never runs it. One fixture directory of
three tiny files — cost discipline per tests/AGENTS.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.secrets import get_anthropic_api_key_optional

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        get_anthropic_api_key_optional() is None,
        reason="integration tier requires ANTHROPIC_API_KEY (tests/AGENTS.md § Integration tests)",
    ),
]

FIXTURE = Path(__file__).parent / "fixtures" / "memory_dir"


def test_fixture_audit_renders_nonzero_headline_classes(
    cli_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dated-stale plant is caught by the age-discount lens, which
    # fires only for source types with a decay horizon. Defaults carry none for
    # LOCAL_MARKDOWN, so the acceptance run supplies the operator config (an
    # existing knob — content_age_decay.sources — not a new detection default).
    config = tmp_path / "config.yaml"
    config.write_text(
        "content_age_decay:\n  sources:\n    LOCAL_MARKDOWN:\n      half_life_days: 180.0\n"
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    from particles.config import reset_config

    reset_config()

    runner = CliRunner()
    result = runner.invoke(app, ["audit", str(FIXTURE), "--yes"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Header: three memory files harvested into a real store.
    assert re.search(r"Audited 3 memory files → \d+ beliefs about \d+ subjects\.", out)

    # Acceptance: nonzero counts for all three headline classes.
    m = re.search(r"(\d+) potential contradictions", out)
    assert m and int(m.group(1)) > 0, out
    m = re.search(r"(\d+) likely-duplicate belief pairs", out)
    assert m and int(m.group(1)) > 0, out
    m = re.search(r"(\d+) probably-stale facts", out)
    assert m and int(m.group(1)) > 0, out

    # Exemplars carry claim text (a quoted line with a short id) and each
    # class ends with its next verb.
    assert re.search(r'• ".+"  \[[0-9a-f]{8}…\]', out)
    assert "next: particles review · particles curate --kind contradiction" in out
    assert "next: particles links suggest --judge" in out
    assert "next: particles curate --kind stale" in out

    # Honesty stance: hedged labels + the uncalibrated-confidence footnote.
    assert "unjudged similarity candidates; --judge to verify" in out
    assert "self-reported and capped, not benchmark-calibrated" in out
    assert "Run 'particles curate' to work these down a few at a time." in out

    # Idempotence: a re-run re-deposits nothing.
    rerun = runner.invoke(app, ["audit", str(FIXTURE), "--yes"])
    assert rerun.exit_code == 0, rerun.output
    assert re.search(r"Harvested 3 entries \(0 new, 3 unchanged\)", rerun.output)
