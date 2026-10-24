# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Snapshot test for the OpenAPI 3.1 contract surface.

`artifacts/openapi.json` is the committed canonical FastAPI contract.
This test asserts the file on disk matches what `app.openapi()`
would produce right now — i.e. nobody has changed an endpoint
without regenerating the snapshot.

§ Decision, this is the discipline mechanism for the
1.0.0 strict-SemVer regime: every contract-changing PR sees the
diff and must classify the change (additive → minor, removal /
required-field addition → major).

If the test fails, run:

    uv run python scripts/gen_openapi.py

and review the diff against `artifacts/openapi.json` before committing.
"""

from __future__ import annotations

import json
from pathlib import Path

from particles import __version__
from particles.api.app import app

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOT_PATH = _REPO_ROOT / "artifacts" / "openapi.json"
_REGENERATE_HINT = (
    "OpenAPI schema drift — run `uv run python scripts/gen_openapi.py` "
    "to regenerate artifacts/openapi.json, review the diff, and classify "
    "the change as minor (additive) or major (breaking)."
)


def _render_live_schema() -> str:
    """Mirror of `scripts/gen_openapi.py::render` — kept in lockstep here so
    the test has no implicit dependency on the scripts/ package layout."""
    schema = app.openapi()
    # Lockstep with gen_openapi.py: the snapshot pins info.version to the
    # contract major version so package version bumps don't churn the file.
    schema["info"]["version"] = __version__.split(".", 1)[0]
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def test_openapi_snapshot_matches_live_schema() -> None:
    """The committed `artifacts/openapi.json` matches what FastAPI emits today."""
    assert _SNAPSHOT_PATH.exists(), (
        f"{_SNAPSHOT_PATH} missing — run `uv run python scripts/gen_openapi.py`"
    )
    committed = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    live = _render_live_schema()
    assert committed == live, _REGENERATE_HINT


def test_declared_responses_carry_their_own_description() -> None:
    """Every response a route declares names its own description.

    Without one, FastAPI fills in the running Python's reason phrase, and
    Python 3.13 renamed several (422 "Unprocessable Entity" became
    "Unprocessable Content"). The snapshot then matches on one interpreter
    and drifts on another: a regeneration on 3.14 failed the 1.210.1
    engine release, whose CI runs the suite on 3.11.
    """
    from fastapi.routing import APIRoute

    missing = [
        f"{route.path} {status}"
        for route in app.routes
        if isinstance(route, APIRoute)
        for status, spec in route.responses.items()
        if "description" not in spec
    ]
    assert missing == [], f"declared responses without a description: {missing}"
