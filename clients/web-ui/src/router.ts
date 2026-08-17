/*
 * Client-side hash router.
 *
 * The unified app has three primary surfaces — Curate (the curation
 * feed), Query, and Browse (the scoped graph) — addressed as hash routes so
 * every view is deep-linkable without any server-side routing: the engine
 * serves one static bundle at /app (web_app.py) and the fragment never
 * reaches the server.
 *
 *   /app#/curate
 *   /app#/query?q=is+Pluto+a+planet
 *   /app#/browse?scope=subject&subject_id=…&as_of=…
 *   /app#/settings
 *
 * The pre-rename route names (#/queue, #/graph) are accepted forever as
 * aliases — deep links are contract (MCP graph_view emitted #/graph URLs;
 * operators bookmarked #/queue) and an alias costs two lines.
 *
 * Scope/selector params ride the fragment's own query string (after the `?`
 * inside the hash), matching the GET /graph wire params one-to-one so a deep
 * link is a transcription of the API call it drives (posture: the
 * client renders server state; it adds no addressing scheme of its own).
 */

/**
 * `settings` is a route, not a tab: it is a full view that replaces the
 * others (not a sheet layered over one), so it gets an address like every
 * other full view. Without one, a reload dropped you back on the feed, Back
 * did not leave it, and it could not be linked to — the app's own onboarding
 * screen was the one surface nobody could send anyone to. It stays out of
 * NAV_ITEMS because the gear addresses it.
 */
export type RouteName = "curate" | "query" | "browse" | "settings";

export interface Route {
  name: RouteName;
  params: URLSearchParams;
}

/** Parse a location.hash value ("#/browse?scope=…") into a Route. */
export function parseHash(hash: string): Route {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const path = raw.startsWith("/") ? raw.slice(1) : raw;
  const qIndex = path.indexOf("?");
  const name = (qIndex === -1 ? path : path.slice(0, qIndex)).replace(/\/+$/, "");
  const params = new URLSearchParams(qIndex === -1 ? "" : path.slice(qIndex + 1));
  if (name === "query" || name === "curate" || name === "settings") {
    return { name, params };
  }
  if (name === "browse" || name === "graph") {
    return { name: "browse", params };
  }
  // Unknown, empty, or the legacy "queue" → curate (the app's home surface).
  return { name: "curate", params };
}

/** Build the hash string for a route ("#/browse?scope=…"). */
export function routeHash(name: RouteName, params?: URLSearchParams | Record<string, string>): string {
  const sp =
    params instanceof URLSearchParams ? params : new URLSearchParams(params ?? {});
  const qs = sp.toString();
  return `#/${name}${qs ? `?${qs}` : ""}`;
}

/** Navigate by setting location.hash (fires hashchange → the app re-renders). */
export function navigate(name: RouteName, params?: URLSearchParams | Record<string, string>): void {
  window.location.hash = routeHash(name, params);
}

/** Current route from the live location. */
export function currentRoute(): Route {
  return parseHash(window.location.hash);
}

/** Subscribe to route changes. Returns an unsubscribe function. */
export function onRouteChange(cb: (route: Route) => void): () => void {
  const handler = (): void => cb(currentRoute());
  window.addEventListener("hashchange", handler);
  return () => window.removeEventListener("hashchange", handler);
}
