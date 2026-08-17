"""Polarity-benchmark suite YAML loader.

Mirrors :mod:`particles.benchmark.modality.loader`'s strictness contract —
missing required fields raise; unknown keys inside ``cases[]`` / ``labels[]``
raise (silently dropping gold-standard data would corrupt the metric); unknown
root-level keys log a warning for forward-compat. Suites live under
``tests/benchmark/polarity/`` — a sibling of the §13.3 ``suites/`` and the
``modality/`` directories, so each harness's discovery walker only ever parses
its own suites.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from particles.benchmark.polarity.schema import (
    ClaimPolarity,
    PolarityCase,
    PolarityLabel,
    PolaritySuite,
)

log = logging.getLogger(__name__)

_SUITE_KEYS = {
    "suite_id",
    "name",
    "version",
    "domain",
    "source_type",
    "cases",
    "metrics",
    "published_by",
    "published_at",
}
_CASE_KEYS = {"case_id", "document", "labels"}
_LABEL_KEYS = {"content", "polarity"}


class PolaritySuiteLoadError(ValueError):
    """Raised when a polarity-benchmark suite YAML is structurally invalid."""


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise PolaritySuiteLoadError(f"{path}: missing required field {key!r}")
    return d[key]


def _parse_label(raw: dict[str, Any], path: str) -> PolarityLabel:
    if not isinstance(raw, dict):
        raise PolaritySuiteLoadError(f"{path}: label must be a mapping")
    extra = set(raw) - _LABEL_KEYS
    if extra:
        raise PolaritySuiteLoadError(f"{path}: unknown PolarityLabel field(s): {sorted(extra)}")
    polarity_raw = _require(raw, "polarity", path)
    try:
        polarity = ClaimPolarity(str(polarity_raw).strip().upper())
    except ValueError as exc:
        raise PolaritySuiteLoadError(
            f"{path}: polarity {polarity_raw!r} is not a valid ClaimPolarity"
        ) from exc
    return PolarityLabel(content=str(_require(raw, "content", path)), polarity=polarity)


def _parse_case(raw: dict[str, Any], suite_path: Path, idx: int) -> PolarityCase:
    path = f"{suite_path.name}#cases[{idx}]"
    if not isinstance(raw, dict):
        raise PolaritySuiteLoadError(f"{path}: case must be a mapping")
    extra = set(raw) - _CASE_KEYS
    if extra:
        raise PolaritySuiteLoadError(f"{path}: unknown PolarityCase field(s): {sorted(extra)}")
    labels_raw = _require(raw, "labels", path)
    if not isinstance(labels_raw, list) or not labels_raw:
        raise PolaritySuiteLoadError(f"{path}: 'labels' must be a non-empty list")
    labels = [_parse_label(lbl, f"{path}.labels[{i}]") for i, lbl in enumerate(labels_raw)]
    return PolarityCase(
        case_id=str(_require(raw, "case_id", path)),
        document=str(_require(raw, "document", path)),
        labels=labels,
    )


def load_polarity_suite(suite_path: Path) -> PolaritySuite:
    """Read one polarity-suite YAML and return a :class:`PolaritySuite`.

    Raises :class:`PolaritySuiteLoadError` on any structural problem.
    """
    if not suite_path.exists():
        raise PolaritySuiteLoadError(f"Suite file not found: {suite_path}")
    try:
        raw = yaml.safe_load(suite_path.read_text())
    except yaml.YAMLError as exc:
        raise PolaritySuiteLoadError(f"{suite_path.name}: YAML parse error: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolaritySuiteLoadError(f"{suite_path.name}: top-level must be a mapping")

    extra = set(raw) - _SUITE_KEYS
    if extra:
        log.warning(
            "Polarity suite %s carries unrecognised root-level field(s) %s; "
            "ignoring (forward-compat)",
            suite_path.name,
            sorted(extra),
        )

    cases_raw = _require(raw, "cases", suite_path.name)
    if not isinstance(cases_raw, list) or not cases_raw:
        raise PolaritySuiteLoadError(f"{suite_path.name}: 'cases' must be a non-empty list")
    cases = [_parse_case(c, suite_path, i) for i, c in enumerate(cases_raw)]

    metrics_raw = raw.get("metrics", [])
    if metrics_raw is not None and not isinstance(metrics_raw, list):
        raise PolaritySuiteLoadError(f"{suite_path.name}: 'metrics' must be a list")

    published_at_raw = raw.get("published_at")
    published_at: datetime | None = None
    if published_at_raw is not None:
        if isinstance(published_at_raw, datetime):
            published_at = published_at_raw
        elif isinstance(published_at_raw, str):
            try:
                published_at = datetime.fromisoformat(published_at_raw)
            except ValueError as exc:
                raise PolaritySuiteLoadError(
                    f"{suite_path.name}: published_at {published_at_raw!r} is not ISO-8601"
                ) from exc
        else:
            raise PolaritySuiteLoadError(
                f"{suite_path.name}: published_at must be string or datetime"
            )

    return PolaritySuite(
        suite_id=str(_require(raw, "suite_id", suite_path.name)),
        name=str(_require(raw, "name", suite_path.name)),
        version=str(_require(raw, "version", suite_path.name)),
        domain=str(_require(raw, "domain", suite_path.name)),
        source_type=str(_require(raw, "source_type", suite_path.name)),
        cases=cases,
        metrics=[str(m) for m in (metrics_raw or [])],
        published_by=str(raw.get("published_by", "")),
        published_at=published_at,
    )


def discover_polarity_suites(suites_dir: Path) -> Iterator[PolaritySuite]:
    """Yield every ``*.yaml`` polarity suite under ``suites_dir``, sorted by name.

    Files that fail to load are logged and skipped (a malformed community suite
    must not block the bundled seed). Hidden files and ``__`` sentinels are
    skipped.
    """
    if not suites_dir.exists():
        return
    for entry in sorted(suites_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue
        try:
            yield load_polarity_suite(entry)
        except PolaritySuiteLoadError as exc:
            log.error("Skipping polarity suite %s: %s", entry.name, exc)
