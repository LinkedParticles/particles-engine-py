"""Locate the vendored graph-view assets at runtime.

Cytoscape.js (pinned, MIT — license text ships beside it) lives in
``artifacts/graph/`` at the repo root for source / editable use; a built wheel
force-includes it at ``particles/_artifacts/graph/`` (the mechanism,
see ``[tool.hatch.build.targets.wheel.force-include]`` in ``pyproject.toml``).
The exporter reads the file and inlines it into the output HTML, so the
exported artifact is self-contained and air-gap-safe — no CDN, ever.
"""

from __future__ import annotations

from pathlib import Path

#: The vendored Cytoscape.js version. Updates are ordinary
#: dependency-bump commits refreshing artifacts/graph/cytoscape.min.js,
#: LICENSE.cytoscape.txt, and this pin together.
CYTOSCAPE_VERSION = "3.34.0"


def graph_assets_dir() -> Path:
    """Return the directory holding the vendored graph assets.

    Prefers the wheel-packaged ``particles/_artifacts/graph``; falls back to
    the source-tree ``<repo-root>/artifacts/graph`` for editable / source use
    (the ``conformance._resources.schemas_dir()`` pattern).
    """
    packaged = Path(__file__).parent.parent.parent / "_artifacts" / "graph"
    if packaged.exists():
        return packaged
    return Path(__file__).parents[3] / "artifacts" / "graph"


def cytoscape_js() -> str:
    """Read the vendored Cytoscape.js source for inlining."""
    return (graph_assets_dir() / "cytoscape.min.js").read_text(encoding="utf-8")


def design_tokens_css() -> str:
    """Read the design tokens (``design/tokens.css``) for inlining.

    The single source of colour / type across every Particles surface (see
    ``design/README.md``). Same resolution as the graph assets: the wheel
    force-includes it at ``particles/_artifacts/design/tokens.css``; a source
    checkout falls back to ``<repo-root>/design/tokens.css``.
    """
    packaged = Path(__file__).parent.parent.parent / "_artifacts" / "design" / "tokens.css"
    if packaged.exists():
        return packaged.read_text(encoding="utf-8")
    return (Path(__file__).parents[3] / "design" / "tokens.css").read_text(encoding="utf-8")
