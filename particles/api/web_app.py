"""Same-origin static serving of the unified web UI.

The web UI (``clients/web-ui/``, formerly the curation PWA) is a thin
typed HTTP client of the FastAPI engine — Queue / Query / Graph hash routes in
one app shell. To make its authenticated ``fetch`` calls **same-origin** — so
they trigger no CORS preflight and the engine's fail-closed boundary gains no
CORS surface (the live "Failed to fetch" failure mode is
designed out, not patched around) — the engine serves the app's built bundle
from its own origin at ``GET /app``.

**The bundle is served unauthenticated.** The mount originally
gated it with the same bearer as every API path, on the consistency argument
that "the inert app shell is gated exactly like the API calls it later makes".
That turned out to be unreachable rather than strict: a browser navigation
cannot carry an ``Authorization`` header, and the settings view where an
operator would paste the token lives *inside* the bundle the gate refuses to
serve — so with a real ``PARTICLES_API_KEY`` the UI could not be opened in a
browser at all. The bundle is static, committed, carries no store data, and
becomes public at the repo split; ``/health`` already discloses the engine
unauthenticated. So the gate bought nothing and cost the whole surface.

**Every API path the loaded app then calls remains gated**, which is where the
data actually is. The app is fail-closed about it: with no bearer
configured it makes no call and says so, which is exactly the prompt the old
gate prevented anyone from ever reaching.

No new API endpoint, no CORS middleware (``spec_impact:
implementation``): this is a static-serving convenience over the frozen
contract. The separate-origin / engine-CORS variant is deferred.
"""

from __future__ import annotations

import logging
from pathlib import Path

from starlette.staticfiles import StaticFiles

log = logging.getLogger(__name__)

# Mount path for the web-UI app shell (unchanged by the
# unification — existing deep links and bearer setups survive the rename).
# Same-origin with the API the app calls (``/curation``, ``/query``,
# ``/graph``, …), so no cross-origin preflight.
WEB_APP_MOUNT = "/app"


def _default_bundle_dir() -> Path:
    """The built web-UI bundle directory: ``clients/web-ui/dist`` at repo root.

    Resolved relative to this file (``particles/api/web_app.py``), four
    parents up to the repo root, so it works for a source checkout regardless of
    the process working directory. (The bundle is excluded from the wheel, so an
    installed package never ships it — the mount is a source-tree affordance.)
    """
    return Path(__file__).resolve().parents[2] / "clients" / "web-ui" / "dist"


class WebUIStaticFiles(StaticFiles):
    """The web-UI bundle, served without the bearer gate.

    A plain :class:`StaticFiles` subclass — the type exists so callers (and
    tests) can assert *which* mount they got, and so the reasoning below has a
    home next to the code it explains.

    This deliberately does **not** run ``verify_request_bearer``. It used to
    , and that made the UI unopenable in any browser once a real
    key was set: the token prompt is inside the bundle the gate withheld. What
    is served here is the static shell only — no store data passes through this
    mount, and every ``/curation``, ``/query``, ``/graph`` call the loaded app
    makes still presents the bearer and is still refused without it.
    """


def build_web_ui_static_app(bundle_dir: Path | None = None) -> WebUIStaticFiles | None:
    """Return the static app for the web-UI bundle, or ``None`` if absent.

    ``bundle_dir`` defaults to ``clients/web-ui/dist`` at the repo root.
    The app is built separately (it is not in the wheel), so a missing bundle is
    the normal case for a fresh checkout / installed package — return ``None`` so
    the engine skips the mount rather than crashing at startup. ``html=True``
    serves ``index.html`` for the SPA root.

    The returned app is unauthenticated; the API paths it calls are
    not. See the module docstring.
    """
    directory = bundle_dir if bundle_dir is not None else _default_bundle_dir()
    if not directory.is_dir():
        log.info(
            "Web-UI bundle not found at %s — skipping the %s static mount "
            "(build it with `npm run build` in clients/web-ui/).",
            directory,
            WEB_APP_MOUNT,
        )
        return None
    log.info(
        "Serving the web UI from %s at %s — unauthenticated shell, gated API.",
        directory,
        WEB_APP_MOUNT,
    )
    return WebUIStaticFiles(directory=str(directory), html=True)
