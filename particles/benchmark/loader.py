"""BenchmarkSuite YAML loader.

Reads suite files matching the techspec §13.3 schema and resolves the
``fixture:`` convenience form against the conformance fixture corpus
(:mod:`particles.conformance.fixtures`). The output is fully-populated
:class:`BenchmarkSuite` instances — every case has either a real
``source_snapshot`` + ``inline_content`` or a resolvable ``fixture``
reference; the runner doesn't need to know which form the YAML used.

The loader is strict by default: missing required fields raise
:class:`SuiteLoadError`. Unknown keys at the suite level are surfaced
as warnings (a non-fatal log line) rather than failures so suite
authors can stage new fields without breaking older runners — but
unknown keys inside ``cases[]`` and ``expected[]`` raise, because
silently dropping them would mask gold-standard data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from particles.benchmark.schema import (
    BenchmarkCase,
    BenchmarkSuite,
    ExpectedParticle,
    RequiredMetric,
)
from particles.core.schema import UncertaintyNature

log = logging.getLogger(__name__)


# Fields the loader recognises on each model. Anything extra inside
# ``cases[]`` / ``expected[]`` raises; extra fields at the suite root
# log a warning (forward-compat).
_SUITE_KEYS = {
    "suite_id",
    "name",
    "version",
    "domain",
    "source_types",
    "cases",
    "metrics",
    "published_by",
    "published_at",
}
_CASE_KEYS = {"case_id", "fixture", "source_snapshot", "inline_content", "expected"}
_EXPECTED_KEYS = {"content", "confidence_min", "uncertainty_nature", "required"}


class SuiteLoadError(ValueError):
    """Raised when a benchmark suite YAML is structurally invalid."""


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise SuiteLoadError(f"{path}: missing required field {key!r}")
    return d[key]


def _parse_expected(raw: dict[str, Any], path: str) -> ExpectedParticle:
    extra = set(raw) - _EXPECTED_KEYS
    if extra:
        raise SuiteLoadError(f"{path}: unknown ExpectedParticle field(s): {sorted(extra)}")
    nature_raw = _require(raw, "uncertainty_nature", path)
    try:
        nature = UncertaintyNature(nature_raw)
    except ValueError as exc:
        raise SuiteLoadError(
            f"{path}: uncertainty_nature {nature_raw!r} is not a valid UncertaintyNature"
        ) from exc
    return ExpectedParticle(
        content=str(_require(raw, "content", path)),
        confidence_min=float(_require(raw, "confidence_min", path)),
        uncertainty_nature=nature,
        required=bool(raw.get("required", True)),
    )


def _parse_case(raw: dict[str, Any], suite_path: Path, idx: int) -> BenchmarkCase:
    path = f"{suite_path.name}#cases[{idx}]"
    extra = set(raw) - _CASE_KEYS
    if extra:
        raise SuiteLoadError(f"{path}: unknown BenchmarkCase field(s): {sorted(extra)}")
    case_id = str(_require(raw, "case_id", path))
    expected_raw = _require(raw, "expected", path)
    if not isinstance(expected_raw, list):
        raise SuiteLoadError(f"{path}: 'expected' must be a list")
    expected = [_parse_expected(e, f"{path}.expected[{i}]") for i, e in enumerate(expected_raw)]

    fixture = raw.get("fixture")
    snap_raw = raw.get("source_snapshot")
    inline = raw.get("inline_content")
    if fixture is None and snap_raw is None:
        raise SuiteLoadError(f"{path}: a case must set either 'fixture' or 'source_snapshot'")
    if fixture is not None and snap_raw is not None:
        raise SuiteLoadError(f"{path}: 'fixture' and 'source_snapshot' are mutually exclusive")

    snapshot = None
    inline_bytes: bytes | None = None
    if snap_raw is not None:
        from particles.core.schema import Snapshot

        try:
            snapshot = Snapshot.model_validate(snap_raw)
        except Exception as exc:
            raise SuiteLoadError(f"{path}: source_snapshot invalid: {exc}") from exc
        if inline is not None:
            inline_bytes = inline.encode("utf-8") if isinstance(inline, str) else bytes(inline)

    return BenchmarkCase(
        case_id=case_id,
        expected=expected,
        source_snapshot=snapshot,
        inline_content=inline_bytes,
        fixture=str(fixture) if fixture is not None else None,
    )


def _parse_metrics(raw: Any, suite_path: Path) -> list[RequiredMetric]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SuiteLoadError(f"{suite_path.name}: 'metrics' must be a list")
    out: list[RequiredMetric] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise SuiteLoadError(f"{suite_path.name}#metrics[{i}]: entry must be a mapping")
        out.append(
            RequiredMetric(
                name=str(_require(entry, "name", f"{suite_path.name}#metrics[{i}]")),
                definition=str(_require(entry, "definition", f"{suite_path.name}#metrics[{i}]")),
            )
        )
    return out


def load_suite(suite_path: Path) -> BenchmarkSuite:
    """Read one suite YAML and return a :class:`BenchmarkSuite`.

    Raises :class:`SuiteLoadError` on any structural problem; on a
    successfully loaded suite the result is ready for the runner —
    every field validated, every ``fixture`` reference *not yet*
    resolved (the runner does that against a fixture directory it
    knows about). The loader's job is pure structural validation.
    """
    if not suite_path.exists():
        raise SuiteLoadError(f"Suite file not found: {suite_path}")
    try:
        raw = yaml.safe_load(suite_path.read_text())
    except yaml.YAMLError as exc:
        raise SuiteLoadError(f"{suite_path.name}: YAML parse error: {exc}") from exc
    if not isinstance(raw, dict):
        raise SuiteLoadError(f"{suite_path.name}: top-level must be a mapping")

    extra = set(raw) - _SUITE_KEYS
    if extra:
        log.warning(
            "Benchmark suite %s carries unrecognised root-level field(s) %s; "
            "ignoring (forward-compat)",
            suite_path.name,
            sorted(extra),
        )

    cases_raw = _require(raw, "cases", suite_path.name)
    if not isinstance(cases_raw, list):
        raise SuiteLoadError(f"{suite_path.name}: 'cases' must be a list")
    cases = [_parse_case(c, suite_path, i) for i, c in enumerate(cases_raw)]

    source_types_raw = _require(raw, "source_types", suite_path.name)
    if not isinstance(source_types_raw, list):
        raise SuiteLoadError(f"{suite_path.name}: 'source_types' must be a list")

    published_at_raw = raw.get("published_at")
    published_at: datetime | None = None
    if published_at_raw is not None:
        if isinstance(published_at_raw, datetime):
            published_at = published_at_raw
        elif isinstance(published_at_raw, str):
            try:
                published_at = datetime.fromisoformat(published_at_raw)
            except ValueError as exc:
                raise SuiteLoadError(
                    f"{suite_path.name}: published_at {published_at_raw!r} is not ISO-8601"
                ) from exc
        else:
            raise SuiteLoadError(f"{suite_path.name}: published_at must be string or datetime")

    return BenchmarkSuite(
        suite_id=str(_require(raw, "suite_id", suite_path.name)),
        name=str(_require(raw, "name", suite_path.name)),
        version=str(_require(raw, "version", suite_path.name)),
        domain=str(_require(raw, "domain", suite_path.name)),
        source_types=[str(st) for st in source_types_raw],
        cases=cases,
        metrics=_parse_metrics(raw.get("metrics"), suite_path),
        published_by=str(raw.get("published_by", "")),
        published_at=published_at,
    )


def discover_suites(suites_dir: Path) -> Iterator[BenchmarkSuite]:
    """Yield every ``*.yaml`` suite under ``suites_dir``, sorted by filename.

    Files that fail to load are *logged and skipped* rather than
    aborting the whole iteration — a malformed community suite should
    not prevent the operator from running the reference suites that do
    parse. Hidden files and the ``__pycache__`` sentinel are skipped.
    """
    if not suites_dir.exists():
        return
    for entry in sorted(suites_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue
        try:
            yield load_suite(entry)
        except SuiteLoadError as exc:
            log.error("Skipping benchmark suite %s: %s", entry.name, exc)


def resolve_case_content(case: BenchmarkCase, fixture_dir: Path) -> tuple[Any, bytes, str]:
    """Resolve a case to ``(snapshot, content_bytes, source_type)``.

    Pulls inline data when the case carries a ``source_snapshot`` /
    ``inline_content`` pair; otherwise loads the named fixture from
    ``fixture_dir``. The runner needs all three values to invoke
    ``extractor.accepts(source_type)`` and ``extractor.extract(snapshot,
    content_bytes)``.
    """
    from particles.conformance.fixtures import iter_fixtures

    if case.fixture is not None:
        for fixture in iter_fixtures(fixture_dir):
            if fixture.fixture_id == case.fixture:
                return fixture.snapshot, fixture.content, fixture.source_type
        raise SuiteLoadError(
            f"case {case.case_id!r}: fixture {case.fixture!r} not found under {fixture_dir}"
        )

    assert case.source_snapshot is not None  # checked at parse time
    content = case.inline_content or b""
    # Inline cases don't carry a source_type field on the case itself;
    # the suite-level source_types[] is treated as authoritative for
    # such cases. The runner walks suite.source_types and pairs them
    # 1:1 with cases when the case carries no fixture — but for the v1
    # API we just return the first declared source_type and let the
    # extractor's accepts() filter handle the rest.
    return case.source_snapshot, content, ""
