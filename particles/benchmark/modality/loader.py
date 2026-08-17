"""Modality-benchmark suite YAML loader.

Mirrors :mod:`particles.benchmark.loader`'s strictness contract — missing
required fields raise; unknown keys inside ``cases[]`` / ``labels[]`` raise
(silently dropping gold-standard data would corrupt the metric); unknown
root-level keys log a warning for forward-compat. Suites live under
``tests/benchmark/modality/`` — a sibling of the §13.3 ``suites/`` directory,
so the content-benchmark discovery never tries (and fails) to parse a modality
suite, and vice versa.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from particles.benchmark.modality.schema import (
    ModalityCase,
    ModalityLabel,
    ModalitySuite,
)
from particles.core.schema import AssertionModality

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
_CASE_KEYS = {"case_id", "entry", "labels", "narrative_expected"}
_LABEL_KEYS = {"content", "modality"}


class ModalitySuiteLoadError(ValueError):
    """Raised when a modality-benchmark suite YAML is structurally invalid."""


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise ModalitySuiteLoadError(f"{path}: missing required field {key!r}")
    return d[key]


def _parse_label(raw: dict[str, Any], path: str) -> ModalityLabel:
    if not isinstance(raw, dict):
        raise ModalitySuiteLoadError(f"{path}: label must be a mapping")
    extra = set(raw) - _LABEL_KEYS
    if extra:
        raise ModalitySuiteLoadError(f"{path}: unknown ModalityLabel field(s): {sorted(extra)}")
    modality_raw = _require(raw, "modality", path)
    try:
        modality = AssertionModality(str(modality_raw).strip().upper())
    except ValueError as exc:
        raise ModalitySuiteLoadError(
            f"{path}: modality {modality_raw!r} is not a valid AssertionModality"
        ) from exc
    return ModalityLabel(content=str(_require(raw, "content", path)), modality=modality)


def _parse_case(raw: dict[str, Any], suite_path: Path, idx: int) -> ModalityCase:
    path = f"{suite_path.name}#cases[{idx}]"
    if not isinstance(raw, dict):
        raise ModalitySuiteLoadError(f"{path}: case must be a mapping")
    extra = set(raw) - _CASE_KEYS
    if extra:
        raise ModalitySuiteLoadError(f"{path}: unknown ModalityCase field(s): {sorted(extra)}")
    labels_raw = _require(raw, "labels", path)
    if not isinstance(labels_raw, list) or not labels_raw:
        raise ModalitySuiteLoadError(f"{path}: 'labels' must be a non-empty list")
    labels = [_parse_label(lbl, f"{path}.labels[{i}]") for i, lbl in enumerate(labels_raw)]
    return ModalityCase(
        case_id=str(_require(raw, "case_id", path)),
        entry=str(_require(raw, "entry", path)),
        labels=labels,
        narrative_expected=bool(raw.get("narrative_expected", True)),
    )


def load_modality_suite(suite_path: Path) -> ModalitySuite:
    """Read one modality-suite YAML and return a :class:`ModalitySuite`.

    Raises :class:`ModalitySuiteLoadError` on any structural problem.
    """
    if not suite_path.exists():
        raise ModalitySuiteLoadError(f"Suite file not found: {suite_path}")
    try:
        raw = yaml.safe_load(suite_path.read_text())
    except yaml.YAMLError as exc:
        raise ModalitySuiteLoadError(f"{suite_path.name}: YAML parse error: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModalitySuiteLoadError(f"{suite_path.name}: top-level must be a mapping")

    extra = set(raw) - _SUITE_KEYS
    if extra:
        log.warning(
            "Modality suite %s carries unrecognised root-level field(s) %s; "
            "ignoring (forward-compat)",
            suite_path.name,
            sorted(extra),
        )

    cases_raw = _require(raw, "cases", suite_path.name)
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ModalitySuiteLoadError(f"{suite_path.name}: 'cases' must be a non-empty list")
    cases = [_parse_case(c, suite_path, i) for i, c in enumerate(cases_raw)]

    metrics_raw = raw.get("metrics", [])
    if metrics_raw is not None and not isinstance(metrics_raw, list):
        raise ModalitySuiteLoadError(f"{suite_path.name}: 'metrics' must be a list")

    published_at_raw = raw.get("published_at")
    published_at: datetime | None = None
    if published_at_raw is not None:
        if isinstance(published_at_raw, datetime):
            published_at = published_at_raw
        elif isinstance(published_at_raw, str):
            try:
                published_at = datetime.fromisoformat(published_at_raw)
            except ValueError as exc:
                raise ModalitySuiteLoadError(
                    f"{suite_path.name}: published_at {published_at_raw!r} is not ISO-8601"
                ) from exc
        else:
            raise ModalitySuiteLoadError(
                f"{suite_path.name}: published_at must be string or datetime"
            )

    return ModalitySuite(
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


def discover_modality_suites(suites_dir: Path) -> Iterator[ModalitySuite]:
    """Yield every ``*.yaml`` modality suite under ``suites_dir``, sorted by name.

    Files that fail to load are logged and skipped (a malformed community
    suite must not block the bundled seed). Hidden files and ``__`` sentinels
    are skipped.
    """
    if not suites_dir.exists():
        return
    for entry in sorted(suites_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue
        try:
            yield load_modality_suite(entry)
        except ModalitySuiteLoadError as exc:
            log.error("Skipping modality suite %s: %s", entry.name, exc)
