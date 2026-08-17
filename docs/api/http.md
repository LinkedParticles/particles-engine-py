# HTTP API (OpenAPI)

The Particles FastAPI server exposes every operation as a typed HTTP
endpoint. The schema below is the **canonical 1.0.0 contract** — every
shipped endpoint, request/response model, and error response is
described here.

The snapshot is committed at
[`artifacts/openapi.json`](https://github.com/LinkedParticles/particles-engine-py/blob/main/artifacts/openapi.json)
and regenerated on every commit by `scripts/gen_openapi.py`. A snapshot
test (`tests/test_openapi_snapshot.py`) fails CI when the committed
schema disagrees with the live server — drift between code and contract
is impossible by construction.

For live exploration against a running server, hit `/docs` (Swagger UI)
or `/redoc` directly:

```bash
uv run uvicorn particles.api.app:app --reload
# then open http://localhost:8000/docs
```

<!--
ReDoc renders client-side from the local openapi.json. The file is
copied here at mkdocs build time by hooks/copy_openapi.py.
-->

<div id="redoc-container"></div>
<script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
<script>
  Redoc.init('openapi.json', {
    scrollYOffset: 60,
    hideDownloadButton: false,
    disableSearch: false,
    expandResponses: '200',
  }, document.getElementById('redoc-container'));
</script>
