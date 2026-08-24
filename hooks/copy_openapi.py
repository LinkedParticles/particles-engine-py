# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""mkdocs build hook — copy the canonical OpenAPI snapshot into the docs tree.

`artifacts/openapi.json` is the normative committed schema (see
`scripts/gen_openapi.py`). mkdocs only serves files under `docs/`,
so this hook copies the snapshot to `docs/api/openapi.json` at
build time. The docs-side copy is gitignored — the artifacts/ file
is the single source of truth and the snapshot test asserts it
matches the live FastAPI schema.

Wired up via `hooks:` in `mkdocs.yml`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "artifacts" / "openapi.json"
_DST = _REPO_ROOT / "docs" / "api" / "openapi.json"


def on_pre_build(config: Any, **_kwargs: object) -> None:
    """Copy the OpenAPI snapshot into the docs tree before mkdocs serves it."""
    if not _SRC.exists():
        # Surface a clear hint rather than letting mkdocs fail with a
        # 404-on-fetch later. The dev's likely missed running gen_openapi.
        raise FileNotFoundError(
            f"{_SRC.relative_to(_REPO_ROOT)} missing — "
            "run `uv run python scripts/gen_openapi.py` first."
        )
    _DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_SRC, _DST)
