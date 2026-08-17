# Particles Engine — the OCI image.
#
# One image, built from this repo: the wheel plus its locked dependency
# closure, the web-UI bundle, the embedding encoder baked at build time, and a
# baked /etc/particles/config.yaml whose every store-adjacent path is pinned
# absolute under a single /data volume. That last part is the point: it closes
# the §2-Correction trap class *by construction* — inside the image
# there is no cwd-relative default left to mis-fire, so the "successful cycle
# against an empty store" failure cannot happen the way it did on 2026-08-05.
#
# The entrypoint binds 0.0.0.0, which means the fail-closed gate
# refuses to start without a real PARTICLES_API_KEY. The image cannot come up
# open by accident; that refusal is a tested property, not a hope.
#
# Build (from the repo root):
#   docker build -f deploy/Dockerfile -t ghcr.io/linkedparticles/engine:dev .
#
# The image name is the owner's `linkedparticles` placeholder; the final names
# ride the sign-off. Until publication these images are built and used
# locally or in CI and are NEVER pushed to a public registry — the same
# the internal-marker rule applies to tags.

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve the locked closure into a self-contained venv and
# bake the encoder weights.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/huggingface

RUN pip install --no-cache-dir "uv==0.9.7"

WORKDIR /src

# Everything `uv sync --no-editable` needs to build and install the project:
# the manifest + lockfile, the package, and the trees the wheel force-includes
# (artifacts/, the Alembic migrations that land under particles/_alembic, and
# design/tokens.css — the design tokens the graph exporter inlines, which
# land under particles/_artifacts/design), plus the metadata files pyproject
# references. Only the one design file: the rest of design/ (previews, the
# card build tool) is not an SDK resource.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE CHANGELOG.md constraints.txt alembic.ini ./
COPY particles ./particles
COPY artifacts ./artifacts
COPY alembic ./alembic
COPY design/tokens.css ./design/tokens.css

# --frozen: install exactly the locked closure, never re-resolve. --no-dev:
# the dev group (pytest, mypy, mkdocs) has no business in a runtime image.
# --no-editable: build and install the real wheel, so the image carries the
# same artifact a `pip install` would.
RUN uv sync --frozen --no-dev --no-editable --no-install-project \
    && uv sync --frozen --no-dev --no-editable

# Bake the embedding encoder (the bake-vs-download question,
# routed here in favour of bake): predictable cold start, air-gap friendly, and
# the image is pulled far less often than the daemon restarts. There is no
# slim/no-encoder variant in v1 — rejected until a measured need exists; the
# size cost is disclosed in docs/operator-guide/container-deployment.md.
RUN /opt/venv/bin/python -c "\
from particles.embeddings import get_embedding_model, EMBEDDING_MODEL_ID; \
m = get_embedding_model(); \
assert m is not None, 'embedding encoder failed to load at build time'; \
print('baked encoder:', EMBEDDING_MODEL_ID)"

# ---------------------------------------------------------------------------
# Stage 2 — runtime.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/huggingface \
    # The encoder is on disk; forbid any reach to HuggingFace so a network
    # blip or an outage can never stall or fail a container start.
    HF_HUB_OFFLINE=1 \
    # The baked config below. Absolute, so nothing depends on the working
    # directory — PARTICLES_CONFIG is the loader's own bootstrap override and
    # is read directly by particles/config.py.
    PARTICLES_CONFIG=/etc/particles/config.yaml

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/huggingface /opt/huggingface

# The web-UI bundle is committed but excluded from the wheel, so the
# image has to put it back: `particles.api.web_app` resolves it three parents up
# from its own file, which for an installed package is the site-packages root.
# The move is followed by an assertion, so a layout change breaks the build here
# rather than surfacing as a silently missing /app mount in production.
#
# The Alembic migrations need no such move — the wheel force-includes them under
# `particles/_alembic` and `particles.db.alembic_paths()` prefers that
# copy. Asserted here too: `create_tables()` runs in the FastAPI lifespan, so a
# migrations dir the image cannot find is a crash on first boot.
COPY clients/web-ui/dist /tmp/web-ui-dist
RUN root="$(python -c 'import pathlib, particles; print(pathlib.Path(particles.__file__).resolve().parent.parent)')" \
 && mkdir -p "$root/clients/web-ui" \
 && mv /tmp/web-ui-dist "$root/clients/web-ui/dist" \
 && python -c "\
from particles.api.web_app import build_web_ui_static_app as build; \
assert build() is not None, 'the /app web-UI mount would not register'" \
 && python -c "\
from particles.db import alembic_paths; \
ini, scripts = alembic_paths(); \
assert ini.is_file(), f'alembic.ini not found at {ini}'; \
assert (scripts / 'versions').is_dir(), f'no migrations under {scripts}'"

COPY deploy/config.container.yaml /etc/particles/config.yaml

# Non-root by default. /data is the single mount point every store-adjacent
# path in the baked config points into.
RUN useradd --create-home --uid 10001 particles \
 && mkdir -p /data \
 && chown -R particles:particles /data
VOLUME ["/data"]
WORKDIR /data
USER particles

EXPOSE 8000

# /health carries no auth gate, so the probe needs no key. In daemon mode it
# also reports the background tasks; `status` flips to "degraded" when one has
# crashed, which this probe deliberately does NOT fail on — a dead consolidation
# tick must not restart a container that is still serving requests. Alert on the
# body, restart on the connection.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["particles"]
# Binding non-loopback is what arms the gate: no PARTICLES_API_KEY,
# no startup. --daemon turns on the in-process scheduler, which is the
# whole reason a container needs neither launchd nor cron.
CMD ["engine", "serve", "0.0.0.0:8000", "--daemon"]

# --- Provenance ------------------------------------------------------------
#
# Deliberately LAST: BUILD_DATE and VCS_REF change on every build, and an ARG
# invalidates the cache for everything after it. Down here that costs one
# metadata layer; anywhere above the encoder bake it would cost a full rebuild.
#
# The build context is a strict allowlist that does not admit `.git`, so the
# revision cannot be discovered inside the build — it has to be passed in. All
# three default to empty, so a bare `docker build` still succeeds; it just
# produces an image that declines to say what it is. See
# docs/operator-guide/container-deployment.md for the stamping build command.
ARG BUILD_DATE=""
ARG VCS_REF=""
ARG VERSION=""

# Standard OCI keys, so `docker inspect` answers "what is this, and when was it
# made?" without the image having to run. The three variable ones are the
# question an operator asks of a container they found running weeks later.
LABEL org.opencontainers.image.title="Particles Engine" \
      org.opencontainers.image.description="Provenance-tracked belief store with an HTTP + MCP surface and a same-origin web UI." \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.created="$BUILD_DATE" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.version="$VERSION"

# The date again, this time where the *running* process can read it: labels are
# invisible from inside the container, and GET /health is what the web UI asks.
# Config, not a raw env read — `build.date` is the field, and this is its
# registered override.
ENV PARTICLES_BUILD_DATE="$BUILD_DATE"
