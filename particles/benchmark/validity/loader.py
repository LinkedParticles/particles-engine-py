"""Validity-benchmark suite YAML loader.

Mirrors the modality / polarity sibling loaders' strictness contract — missing
required fields raise; unknown keys inside ``cases[]`` / ``labels[]`` raise
(silently dropping gold-standard data would corrupt the metric); unknown
root-level keys log a warning for forward-compat. Suites live under
``tests/benchmark/validity/`` — a sibling of the §13.3 ``suites/`` and the
``modality/`` / ``polarity/`` directories, so each harness's discovery walker
only ever parses its own suites.

The loader also enforces the :class:`ValidityLabel` consistency invariant —
``is_durable`` must equal ``expected_valid_until is None`` — so a suite can never
declare a durable decoy that also carries a boundary (or vice versa).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from particles.benchmark.validity.schema import ValidityCase, ValidityLabel, ValiditySuite

log = logging.getLogger(__name__)

_SUITE_KEYS = {
    "suite_id",
    "name",
    "version",
    "domain",
    "source_type",
    "cases",
    "date_tolerance_days",
    "metrics",
    "published_by",
    "published_at",
}
_CASE_KEYS = {"case_id", "document", "reference_date", "labels"}
_LABEL_KEYS = {"content", "expected_valid_until", "is_durable"}


class ValiditySuiteLoadError(ValueError):
    """Raised when a validity-benchmark suite YAML is structurally invalid."""


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise ValiditySuiteLoadError(f"{path}: missing required field {key!r}")
    return d[key]


def _coerce_date(raw: Any, path: str, field_name: str) -> date:
    """Parse a YAML date or ISO-8601 date string into a ``date``."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw.strip())
        except ValueError as exc:
            raise ValiditySuiteLoadError(
                f"{path}: {field_name} {raw!r} is not an ISO-8601 date"
            ) from exc
    raise ValiditySuiteLoadError(f"{path}: {field_name} must be a date or ISO-8601 string")


def _parse_label(raw: dict[str, Any], path: str) -> ValidityLabel:
    if not isinstance(raw, dict):
        raise ValiditySuiteLoadError(f"{path}: label must be a mapping")
    extra = set(raw) - _LABEL_KEYS
    if extra:
        raise ValiditySuiteLoadError(f"{path}: unknown ValidityLabel field(s): {sorted(extra)}")
    is_durable = bool(_require(raw, "is_durable", path))
    vu_raw = raw.get("expected_valid_until")
    expected: date | None = (
        None if vu_raw is None else _coerce_date(vu_raw, path, "expected_valid_until")
    )
    # Consistency invariant: durable ⇔ no boundary.
    if is_durable and expected is not None:
        raise ValiditySuiteLoadError(
            f"{path}: is_durable=true but expected_valid_until is set — "
            "a durable label carries no boundary"
        )
    if not is_durable and expected is None:
        raise ValiditySuiteLoadError(
            f"{path}: is_durable=false but expected_valid_until is missing — "
            "a bounded label needs a date"
        )
    return ValidityLabel(
        content=str(_require(raw, "content", path)),
        expected_valid_until=expected,
        is_durable=is_durable,
    )


def _parse_case(raw: dict[str, Any], suite_path: Path, idx: int) -> ValidityCase:
    path = f"{suite_path.name}#cases[{idx}]"
    if not isinstance(raw, dict):
        raise ValiditySuiteLoadError(f"{path}: case must be a mapping")
    extra = set(raw) - _CASE_KEYS
    if extra:
        raise ValiditySuiteLoadError(f"{path}: unknown ValidityCase field(s): {sorted(extra)}")
    labels_raw = _require(raw, "labels", path)
    if not isinstance(labels_raw, list) or not labels_raw:
        raise ValiditySuiteLoadError(f"{path}: 'labels' must be a non-empty list")
    labels = [_parse_label(lbl, f"{path}.labels[{i}]") for i, lbl in enumerate(labels_raw)]
    return ValidityCase(
        case_id=str(_require(raw, "case_id", path)),
        document=str(_require(raw, "document", path)),
        reference_date=_coerce_date(_require(raw, "reference_date", path), path, "reference_date"),
        labels=labels,
    )


def load_validity_suite(suite_path: Path) -> ValiditySuite:
    """Read one validity-suite YAML and return a :class:`ValiditySuite`.

    Raises :class:`ValiditySuiteLoadError` on any structural problem.
    """
    if not suite_path.exists():
        raise ValiditySuiteLoadError(f"Suite file not found: {suite_path}")
    try:
        raw = yaml.safe_load(suite_path.read_text())
    except yaml.YAMLError as exc:
        raise ValiditySuiteLoadError(f"{suite_path.name}: YAML parse error: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValiditySuiteLoadError(f"{suite_path.name}: top-level must be a mapping")

    extra = set(raw) - _SUITE_KEYS
    if extra:
        log.warning(
            "Validity suite %s carries unrecognised root-level field(s) %s; "
            "ignoring (forward-compat)",
            suite_path.name,
            sorted(extra),
        )

    cases_raw = _require(raw, "cases", suite_path.name)
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValiditySuiteLoadError(f"{suite_path.name}: 'cases' must be a non-empty list")
    cases = [_parse_case(c, suite_path, i) for i, c in enumerate(cases_raw)]

    metrics_raw = raw.get("metrics", [])
    if metrics_raw is not None and not isinstance(metrics_raw, list):
        raise ValiditySuiteLoadError(f"{suite_path.name}: 'metrics' must be a list")

    tolerance_raw = raw.get("date_tolerance_days", 3)
    if not isinstance(tolerance_raw, int) or isinstance(tolerance_raw, bool) or tolerance_raw < 0:
        raise ValiditySuiteLoadError(
            f"{suite_path.name}: date_tolerance_days must be a non-negative integer"
        )

    published_at_raw = raw.get("published_at")
    published_at: datetime | None = None
    if published_at_raw is not None:
        if isinstance(published_at_raw, datetime):
            published_at = published_at_raw
        elif isinstance(published_at_raw, str):
            try:
                published_at = datetime.fromisoformat(published_at_raw)
            except ValueError as exc:
                raise ValiditySuiteLoadError(
                    f"{suite_path.name}: published_at {published_at_raw!r} is not ISO-8601"
                ) from exc
        else:
            raise ValiditySuiteLoadError(
                f"{suite_path.name}: published_at must be string or datetime"
            )

    return ValiditySuite(
        suite_id=str(_require(raw, "suite_id", suite_path.name)),
        name=str(_require(raw, "name", suite_path.name)),
        version=str(_require(raw, "version", suite_path.name)),
        domain=str(_require(raw, "domain", suite_path.name)),
        source_type=str(_require(raw, "source_type", suite_path.name)),
        cases=cases,
        date_tolerance_days=tolerance_raw,
        metrics=[str(m) for m in (metrics_raw or [])],
        published_by=str(raw.get("published_by", "")),
        published_at=published_at,
    )


def discover_validity_suites(suites_dir: Path) -> Iterator[ValiditySuite]:
    """Yield every ``*.yaml`` validity suite under ``suites_dir``, sorted by name.

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
            yield load_validity_suite(entry)
        except ValiditySuiteLoadError as exc:
            log.error("Skipping validity suite %s: %s", entry.name, exc)
