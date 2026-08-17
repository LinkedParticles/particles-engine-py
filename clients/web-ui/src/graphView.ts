/*
 * #/browse — the scoped epistemic graph route.
 *
 * The SPA half of "one build, two presentations": GET /graph
 * returns the same GraphData the static `export graph` artifact embeds, and
 * this route renders it via the shared graphRender.ts mount (the same §5
 * encodings the inline query-page graph uses). Everything epistemic arrives
 * server-computed — this route only picks the scope.
 *
 * Scope lives in the hash query string (#/browse?scope=subject&subject_id=…),
 * transcribing the GET /graph wire params one-to-one, so every render is
 * deep-linkable and the MCP graph_view tool's `url` field lands here. The
 * engine enforces the anti-hairball invariant server-side with 422 (a
 * whole-store render does not exist).
 *
 * A scope-less visit is an invitation to browse, not a dead end: it seeds
 * the render with the store's most-connected subject (GET
 * /subjects?order=degree&limit=1 — the server picks; this client holds no
 * ranking of its own) and says so, with the scope picker kept in view. The
 * hash stays bare #/browse — the seed is a dynamic default, not an address;
 * the shell's per-tab hash memory makes return visits sticky to wherever
 * you actually browsed.
 */
import { ApiError, GraphData, GraphParams, ParticlesApiClient } from "./api";
import { CytoscapeCtor, hashParams, loadCytoscape, renderGraph } from "./graphRender";
import { navigate } from "./router";

export interface GraphViewDeps {
  client: ParticlesApiClient;
  onNeedsSettings: () => void;
}

// ---------------------------------------------------------------------------
// Params ↔ hash
// ---------------------------------------------------------------------------

/** Read GraphParams from the route's hash query string; null when unscoped. */
export function paramsFromHash(sp: URLSearchParams): GraphParams | null {
  const scope = sp.get("scope");
  if (
    scope !== "subject" &&
    scope !== "query" &&
    scope !== "inconsistency" &&
    scope !== "projection"
  ) {
    return null;
  }
  const p: GraphParams = { scope };
  if (scope === "subject") {
    const subjectId = sp.get("subject_id");
    if (!subjectId) return null;
    p.subject_id = subjectId;
  } else if (scope === "query") {
    const q = sp.get("q");
    if (!q) return null;
    p.q = q;
  } else if (scope === "inconsistency") {
    const iid = sp.get("inconsistency_id");
    if (!iid) return null;
    p.inconsistency_id = iid;
  } else {
    const manifest = sp.get("manifest");
    const section = sp.get("section");
    if (!manifest || !section) return null;
    p.manifest = manifest;
    p.section = section;
  }
  const hops = sp.get("hops");
  if (hops) p.hops = Number(hops);
  if (sp.get("history") === "true") p.history = true;
  const asOf = sp.get("as_of");
  if (asOf) p.as_of = asOf;
  const maxNodes = sp.get("max_nodes");
  if (maxNodes) p.max_nodes = Number(maxNodes);
  return p;
}

// ---------------------------------------------------------------------------
// View entry
// ---------------------------------------------------------------------------

export function renderGraphView(
  container: HTMLElement,
  deps: GraphViewDeps,
  params: URLSearchParams,
): void {
  container.innerHTML = "";
  const p = paramsFromHash(params);
  if (p === null) {
    // No (complete) scope in the hash. A truly bare visit gets the seeded
    // browse; a partial/broken scope (scope=subject with no id) keeps the
    // picker-first behaviour — the operator was mid-edit, don't yank them.
    if ([...params.keys()].length === 0) {
      void seedAndRender(container, deps);
    } else {
      container.appendChild(scopePicker(params));
    }
    return;
  }
  const view = el("div", "graph-view");
  container.appendChild(view);
  void loadAndRender(view, deps, p);
}

/** The unscoped landing: pick a scope (a whole-store render does not exist). */
function scopePicker(params: URLSearchParams): HTMLElement {
  // Preserve the non-scope deep-link params (hops / history / as_of /
  // max_nodes) across a resubmit: this picker also renders on a fetch error,
  // and rebuilding the URL from the scope fields alone silently stripped
  // whatever the operator had set — e.g. losing &history=true while
  // correcting a malformed as_of.
  const extras: Record<string, string> = {};
  for (const k of ["hops", "history", "as_of", "max_nodes"]) {
    const v = params.get(k);
    if (v) extras[k] = v;
  }
  const panel = el("div", "settings");
  panel.appendChild(el("h2", "", "Browse — pick a scope"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "Every render is a scoped subgraph: one subject's " +
        "neighbourhood, or one query's retrieval set. A whole-store render " +
        "does not exist — that is the anti-hairball rule, and it is the point.",
    ),
  );
  const subjInput = fieldInput(panel, "Subject name or id", params.get("subject_id") ?? "");
  const subjGo = button(panel, "Render subject neighbourhood", () => {
    const v = subjInput.value.trim();
    if (v) navigate("browse", { scope: "subject", subject_id: v, ...extras });
  });
  panel.appendChild(el("p", "hint", "— or —"));
  const qInput = fieldInput(panel, "Query", params.get("q") ?? "");
  button(panel, "Render query retrieval set", () => {
    const v = qInput.value.trim();
    if (v) navigate("browse", { scope: "query", q: v, ...extras });
  });
  subjInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") subjGo.click();
  });
  if (Object.keys(extras).length) {
    panel.appendChild(
      el(
        "p",
        "hint",
        "Kept from the current link: " +
          Object.entries(extras)
            .map(([k, v]) => `${k}=${v}`)
            .join(", "),
      ),
    );
  }
  return panel;
}

// ---------------------------------------------------------------------------
// Seeded first render (scope-less visit)
// ---------------------------------------------------------------------------

async function seedAndRender(container: HTMLElement, deps: GraphViewDeps): Promise<void> {
  container.appendChild(el("div", "empty", "Finding a place to start…"));
  let seed: { id: string; name: string } | null = null;
  try {
    const top = await deps.client.subjects({ order: "degree", limit: 1 });
    if (top.length) seed = { id: top[0].id ?? "", name: top[0].canonical_name };
  } catch (e) {
    if (e instanceof ApiError && e.kind === "not-configured") {
      deps.onNeedsSettings();
      return;
    }
    // Seed lookup failed (old engine, transient error) — the picker is the
    // honest fallback, not a blocker.
  }
  container.innerHTML = "";
  if (!seed || !seed.id) {
    container.appendChild(scopePicker(new URLSearchParams()));
    return;
  }
  // The auto-chosen anchor must say so — an unexplained render disorients
  // ("why am I looking at this?"). Banner first, then the graph (the point
  // of the seeded landing is a rendered graph, not a form), with the scope
  // picker kept visible below it — in view, never hidden behind an empty
  // state, but not shoving the render below the fold either.
  container.appendChild(
    banner(
      "info",
      `Starting from “${seed.name}”, the most-connected subject — pick another below the render.`,
    ),
  );
  const view = el("div", "graph-view");
  container.appendChild(view);
  container.appendChild(scopePicker(new URLSearchParams()));
  await loadAndRender(view, deps, { scope: "subject", subject_id: seed.id });
}

// ---------------------------------------------------------------------------
// Fetch + render
// ---------------------------------------------------------------------------

async function loadAndRender(
  view: HTMLElement,
  deps: GraphViewDeps,
  p: GraphParams,
): Promise<void> {
  view.innerHTML = "";
  view.appendChild(el("div", "empty", "Rendering the subgraph…"));
  let data: GraphData;
  let cy: CytoscapeCtor;
  try {
    [data, cy] = await Promise.all([deps.client.graph(p), loadCytoscape()]);
  } catch (e) {
    view.innerHTML = "";
    if (e instanceof ApiError && e.kind === "not-configured") {
      deps.onNeedsSettings();
      return;
    }
    const msg = e instanceof ApiError ? e.message : String(e);
    view.appendChild(subjectLinkifiedBanner("error", msg, p));
    view.appendChild(scopePicker(new URLSearchParams(hashParams(p))));
    return;
  }
  view.innerHTML = "";
  renderGraph(view, p, data, cy);
}

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------

function el(tag: string, cls?: string, text?: string): HTMLElement {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function banner(cls: "error" | "info", text: string): HTMLElement {
  const b = el("div", `banner ${cls}`);
  b.textContent = text;
  return b;
}

/**
 * An error banner whose subject ids are clickable. The engine's
 * unknown-subject error suggests candidates as "Name (uuid)"; forcing the
 * operator to copy-paste a uuid into the form defeats the suggestion. Each
 * uuid navigates the browse route to that subject (preserving the current
 * hops/history/as_of params). Built from text nodes — never innerHTML; the
 * message can quote store-controlled subject names.
 */
function subjectLinkifiedBanner(
  cls: "error" | "info",
  text: string,
  p: GraphParams,
): HTMLElement {
  const b = el("div", `banner ${cls}`);
  const uuidRe = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
  let last = 0;
  for (const m of text.matchAll(uuidRe)) {
    b.appendChild(document.createTextNode(text.slice(last, m.index)));
    const id = m[0];
    const a = el("a", "subject-link", id);
    a.addEventListener("click", () =>
      navigate("browse", hashParams({ ...p, subject_id: id, scope: "subject", q: undefined })),
    );
    b.appendChild(a);
    last = (m.index ?? 0) + id.length;
  }
  b.appendChild(document.createTextNode(text.slice(last)));
  return b;
}

function fieldInput(parent: HTMLElement, label: string, value: string): HTMLInputElement {
  const wrap = el("div", "field");
  const l = el("label", "", label);
  const input = document.createElement("input");
  input.type = "text";
  input.value = value;
  wrap.append(l, input);
  parent.appendChild(wrap);
  return input;
}

function button(parent: HTMLElement, label: string, onClick: () => void): HTMLButtonElement {
  const row = el("div", "btn-row");
  const btn = document.createElement("button");
  btn.className = "btn primary";
  btn.textContent = label;
  btn.onclick = onClick;
  row.appendChild(btn);
  parent.appendChild(row);
  return btn;
}
