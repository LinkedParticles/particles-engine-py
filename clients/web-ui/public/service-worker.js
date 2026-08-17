/*
 * Service worker for the Particles web UI.
 *
 * App-shell cache ONLY. The curation queue, query answers, graph renders, and
 * every write are LIVE engine state (§ Harder/cost): the worker must
 * NEVER cache an API response, or a stale "today's N" would render, a gesture
 * could act on a vanished card, and a graph would show yesterday's beliefs as
 * today's. It caches only the static shell (the HTML / JS / CSS / manifest /
 * icon / vendored Cytoscape) so "add to home screen" yields an installable app
 * icon and a fast cold start; everything under the engine API is fetched
 * network-only, untouched.
 *
 * Scope note: the engine serves this bundle under /app, so the
 * worker's scope is /app/ and it only ever sees requests for shell assets — API
 * calls go to sibling paths (/curation, /query, /graph, …) outside this scope.
 * The network-only guard below is belt-and-suspenders.
 */
const SHELL_CACHE = "particles-web-ui-shell-__BUILD_VERSION__";
const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./app.js",
  "./tokens.css",
  "./styles.css",
  "./manifest.webmanifest",
  "./icon.svg",
  "./cytoscape.min.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS)),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== SHELL_CACHE)
            .map((k) => caches.delete(k)),
        ),
      ),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  // Only ever serve GETs for same-scope shell assets from cache. Anything else
  // (every API call, every non-GET) is passed straight through to the network —
  // the worker never caches or replays a live engine response.
  if (req.method !== "GET") {
    return; // default: network
  }
  const url = new URL(req.url);
  const inScope = url.pathname.endsWith("/") || /\.(html|js|css|svg|webmanifest)$/.test(url.pathname);
  if (!inScope) {
    return; // API path → network-only
  }
  event.respondWith(
    caches.match(req).then((cached) => cached ?? fetch(req)),
  );
});
