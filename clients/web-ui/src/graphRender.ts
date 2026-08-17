/*
 * The pure Cytoscape mount for GraphData renders (encodings).
 *
 * Shared by the #/browse route (graphView.ts — full render with the as-of
 * scrubber) and the #/query page's inline "knowledge consulted" graph
 * (queryView.ts — same render, no scrubber). Everything epistemic arrives
 * server-computed: NO client-side confidence math, ever — a render here is
 * a projection of engine state.
 *
 * Deep links inside the render (anchor-here, show-history, tick jumps)
 * navigate the #/browse route: from the inline query context they
 * deliberately promote the reader into the full Browse view.
 */
import { GraphData, GraphParams, GraphParticleInfo } from "./api";
import { navigate } from "./router";
import { tokenPercent, tokenValue } from "./theme";

// ---------------------------------------------------------------------------
// Vendored-Cytoscape loader (same-origin <script>; loaded once, on demand)
// ---------------------------------------------------------------------------

export type CytoscapeCtor = (opts: Record<string, unknown>) => CyInstance;
type CyInstance = {
  layout(opts: Record<string, unknown>): { run(): void };
  fit(eles?: unknown, padding?: number): void;
  resize(): void;
  zoom(): number;
  pan(): { x: number; y: number };
  on(event: string, selectorOrCb: string | ((ev: CyEvent) => void), cb?: (ev: CyEvent) => void): void;
  edges(): { forEach(cb: (e: CyEle) => void): void };
  style(): {
    selector(sel: string): {
      style(map: Record<string, unknown>): { update(): void };
    };
  };
  destroy(): void;
};
type CyEvent = { target: CyEle | CyInstance };
type CyEle = {
  id(): string;
  data(key?: string): string;
  toggleClass(cls: string, state: boolean): void;
};

// Layout preference (presentation-only, so it lives in localStorage rather
// than the hash — deep links stay pure API transcriptions, PARITY.md §6).
// "rings" = the deterministic layouts; "force" = cose physics
// seeded from the deterministic layout (randomize:false), so the same store
// state still yields the same picture.
type LayoutMode = "rings" | "force";
const LAYOUT_KEY = "particles.graphLayout";

function loadLayoutMode(): LayoutMode {
  try {
    return localStorage.getItem(LAYOUT_KEY) === "force" ? "force" : "rings";
  } catch {
    return "rings";
  }
}

function saveLayoutMode(mode: LayoutMode): void {
  try {
    localStorage.setItem(LAYOUT_KEY, mode);
  } catch {
    // storage unavailable (private mode) — the toggle still works per render
  }
}

let cytoscapeLoader: Promise<CytoscapeCtor> | null = null;

export function loadCytoscape(): Promise<CytoscapeCtor> {
  if (cytoscapeLoader === null) {
    cytoscapeLoader = new Promise((resolve, reject) => {
      const w = window as unknown as { cytoscape?: CytoscapeCtor };
      if (w.cytoscape) {
        resolve(w.cytoscape);
        return;
      }
      const s = document.createElement("script");
      // Relative: the bundle is served under /app/, beside this asset.
      s.src = "cytoscape.min.js";
      s.onload = () => {
        if (w.cytoscape) resolve(w.cytoscape);
        else reject(new Error("cytoscape.min.js loaded but exposed no global"));
      };
      s.onerror = () => {
        cytoscapeLoader = null; // allow retry on next render
        reject(new Error("Failed to load cytoscape.min.js (rebuild dist/?)"));
      };
      document.head.appendChild(s);
    });
  }
  return cytoscapeLoader;
}

// ---------------------------------------------------------------------------
// Params → hash fragment (the #/browse deep-link vocabulary)
// ---------------------------------------------------------------------------

export function hashParams(p: GraphParams): Record<string, string> {
  const out: Record<string, string> = { scope: p.scope };
  if (p.subject_id) out.subject_id = p.subject_id;
  if (p.q) out.q = p.q;
  if (p.inconsistency_id) out.inconsistency_id = p.inconsistency_id;
  if (p.manifest) out.manifest = p.manifest;
  if (p.section) out.section = p.section;
  if (p.hops !== undefined && p.hops !== 1) out.hops = String(p.hops);
  if (p.history) out.history = "true";
  if (p.as_of) out.as_of = p.as_of;
  if (p.max_nodes !== undefined) out.max_nodes = String(p.max_nodes);
  return out;
}

export interface RenderGraphOptions {
  /**
   * Show the as-of scrubber (default true — the Browse route). The inline
   * query-page render passes false: time travel re-anchors the whole route,
   * which is a Browse concern, not an under-the-answer widget.
   */
  showScrubber?: boolean;
}

export function renderGraph(
  view: HTMLElement,
  p: GraphParams,
  data: GraphData,
  cytoscape: CytoscapeCtor,
  opts: RenderGraphOptions = {},
): void {
  const showScrubber = opts.showScrubber !== false;
  const P: Record<string, GraphParticleInfo> = data.particles ?? {};

  // --- header: meta + history toggle -------------------------------------
  const head = el("div", "graph-head");
  const metaBits = [
    `${data.nodes.length} subjects · ${data.edges.length} edges · ${Object.keys(P).length} particles`,
    data.census.scope,
  ];
  if (data.as_of) metaBits.push(`as of ${String(data.as_of).slice(0, 10)}`);
  head.appendChild(el("div", "graph-meta", metaBits.join(" · ")));

  const ghostIds = Object.keys(P).filter((id) => P[id].ghost);
  let showGhosts = true;
  let histToggle: HTMLInputElement | null = null;
  if (ghostIds.length) {
    const label = el("label", "hist");
    histToggle = document.createElement("input");
    histToggle.type = "checkbox";
    histToggle.checked = true;
    label.append(histToggle, document.createTextNode(" show history (ghosts)"));
    head.appendChild(label);
  }
  let layoutMode = loadLayoutMode();
  const layoutLabel = el("label", "hist");
  const layoutToggle = document.createElement("input");
  layoutToggle.type = "checkbox";
  layoutToggle.checked = layoutMode === "force";
  layoutToggle.title =
    "Force-directed (cose) layout, seeded from the deterministic layout — " +
    "pulls connected subjects together instead of ring placement";
  layoutLabel.append(layoutToggle, document.createTextNode(" force layout"));
  head.appendChild(layoutLabel);
  view.appendChild(head);

  // --- as-of scrubber -----------------------------
  // Each slider release re-issues GET /graph at that instant; the SERVER
  // recomputes every epistemic quantity per T (visibility + decay
  // move to T, judgment stays current). The client only picks the T: the
  // slider bounds come from the render's own timestamps, not from any
  // epistemic computation. Clicking a tick opens the change listing for that
  // day in the detail panel (wired below, after the panel exists).
  let inspectInstant: (day: string) => void = () => {};
  if (showScrubber) {
    const scrubber = buildScrubber(data, p, (day) => inspectInstant(day));
    if (scrubber) view.appendChild(scrubber);
  }

  // --- disclosures (discipline: a capped view says so) -----------
  for (const line of data.disclosures ?? []) {
    view.appendChild(banner("info", `⚠ ${line}`));
  }

  // --- canvas + panel ------------------------------------------------------
  const wrap = el("div", "graph-wrap");
  const cyEl = el("div", "cy");
  const panel = el("aside", "gpanel");
  wrap.append(cyEl, panel);
  view.appendChild(wrap);
  view.appendChild(legend());

  // --- elements, straight from the server payload (no epistemic math) -----
  const succOf: Record<string, string> = {};
  const predOf: Record<string, string> = {};
  for (const s of data.supersessions ?? []) {
    succOf[s.predecessor_id] = s.successor_id;
    predOf[s.successor_id] = s.predecessor_id;
  }

  // Structural mass per subject — how many rendered particles touch it
  // (cargo, including the truncated tail, plus incident edges). Drives the
  // node-size encoding together with utility; structural counting only, no
  // epistemic math.
  const massOf: Record<string, number> = {};
  for (const n of data.nodes) {
    massOf[n.subject_id] = (n.cargo ?? []).length + (n.cargo_truncated ?? 0);
  }
  for (const e of data.edges ?? []) {
    if (!P[e.particle_id]) continue;
    massOf[e.source] = (massOf[e.source] ?? 0) + 1;
    massOf[e.target] = (massOf[e.target] ?? 0) + 1;
  }

  const elements: Record<string, unknown>[] = [];
  for (const n of data.nodes) {
    elements.push({
      group: "nodes",
      data: {
        id: n.subject_id,
        label: (n.label ?? n.subject_id) + (n.contested ? " ⚠" : ""),
        hop: n.hop ?? 0,
        size: nodeSize(massOf[n.subject_id] ?? 0, n.utility_score ?? 0),
        shade: nodeShade(n.max_effective_confidence ?? 0),
        border: (n.hop ?? 0) === 0 ? 3 : 1.5,
      },
    });
  }
  (data.edges ?? []).forEach((e, i) => {
    const info = P[e.particle_id];
    if (!info) return;
    elements.push({
      group: "edges",
      classes: (info.ghost ? "ghost " : "") + (info.retrieval_hit ? "hit" : ""),
      data: {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        particleId: e.particle_id,
        opacity: Math.max(0.08, info.effective_confidence),
        lstyle: edgeStyleFor(info),
        label: info.contested ? "⚠" : info.status === "RETRACTED" ? "☒" : "",
      },
    });
  });

  // Colours come from the design tokens (design/tokens.css) resolved for the
  // active theme; Cytoscape needs concrete strings, so read them at render.
  const C = {
    label: tokenValue("--p-text"),
    nodeBorder: tokenValue("--p-graph-node-border"),
    edge: tokenValue("--p-graph-edge"),
    edgeGhost: tokenValue("--p-graph-edge-ghost"),
    edgeLabel: tokenValue("--p-badge-contested"),
    hit: tokenValue("--p-badge-hit"),
  };
  const cy = cytoscape({
    container: cyEl,
    elements,
    // Layout runs explicitly below; fit:false keeps the init layout from
    // resetting the viewport (see exporters/graph/render.py, same glue).
    layout: { name: "preset", fit: false },
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": 11,
          color: C.label,
          "text-valign": "bottom",
          "text-margin-y": 4,
          "text-wrap": "wrap",
          "text-max-width": 130,
          width: "data(size)",
          height: "data(size)",
          "background-color": "data(shade)",
          "border-width": "data(border)",
          "border-color": C.nodeBorder,
        },
      },
      {
        selector: "edge",
        style: {
          "curve-style": "bezier",
          "control-point-step-size": 12,
          width: 2,
          "line-color": C.edge,
          opacity: "data(opacity)",
          "line-style": "data(lstyle)",
          label: "data(label)",
          "font-size": 12,
          color: C.edgeLabel,
        },
      },
      { selector: "edge.hit", style: { "line-color": C.hit, width: 3.5 } },
      { selector: "edge.ghost", style: { "line-color": C.edgeGhost } },
      { selector: ".hidden", style: { display: "none" } },
    ],
  });

  // Deterministic layouts: concentric hop rings for subject
  // scope; a circle in retrieval-rank order for query scope. The opt-in
  // force layout runs cose ON TOP of the deterministic placement
  // (randomize:false), so it stays a pure function of the render too.
  const ringsLayout =
    data.scope_type === "subject"
      ? {
          name: "concentric",
          concentric: (n: { data: (k: string) => number }) => 10 - n.data("hop"),
          levelWidth: () => 1,
          minNodeSpacing: 60,
          animate: false,
          nodeDimensionsIncludeLabels: true,
          fit: false,
        }
      : {
          name: "circle",
          animate: false,
          spacingFactor: 1.2,
          nodeDimensionsIncludeLabels: true,
          fit: false,
        };
  const runLayout = (): void => {
    cy.layout(ringsLayout).run();
    if (layoutMode === "force") {
      cy.layout({
        name: "cose",
        animate: false,
        randomize: false,
        fit: false,
        nodeDimensionsIncludeLabels: true,
        idealEdgeLength: 110,
        nodeOverlap: 20,
        nodeRepulsion: 1200000,
        gravity: 10,
      }).run();
    }
  };
  runLayout();
  layoutToggle.addEventListener("change", () => {
    layoutMode = layoutToggle.checked ? "force" : "rings";
    saveLayoutMode(layoutMode);
    runLayout();
    cy.fit(undefined, 30);
  });
  // cy.fit() silently no-ops until the renderer is ready — retry per frame
  // until the viewport actually moves (same workaround as the static export).
  let fitTries = 0;
  const fitLoop = (): void => {
    cy.resize();
    cy.fit(undefined, 30);
    fitTries += 1;
    const untouched = cy.zoom() === 1 && cy.pan().x === 0 && cy.pan().y === 0;
    if (untouched && fitTries < 300) requestAnimationFrame(fitLoop);
  };
  fitLoop();

  // Counter-scale label text against zoom: font-size renders in canvas
  // units, so a fit that zooms out a large render (dozens of subjects)
  // shrinks 11px labels to unreadable specks. Keep labels at roughly
  // constant screen size when zoomed out; let them grow gently on zoom-in.
  const BASE_FONT = 11;
  let lastFont = BASE_FONT;
  const syncLabelScale = (): void => {
    const z = cy.zoom();
    if (!z || !Number.isFinite(z)) return;
    const fs = Math.round(Math.min(BASE_FONT * 6, BASE_FONT / Math.min(1, z)));
    if (fs === lastFont) return;
    lastFont = fs;
    cy.style().selector("node").style({ "font-size": fs }).update();
    cy.style().selector("edge").style({ "font-size": fs + 1 }).update();
  };
  let fontRaf = 0;
  cy.on("zoom", () => {
    if (fontRaf) return;
    fontRaf = requestAnimationFrame(() => {
      fontRaf = 0;
      syncLabelScale();
    });
  });
  syncLabelScale();

  // --- history toggle ------------------------------------------------------
  let panelSubject: string | null = null;
  if (histToggle) {
    histToggle.addEventListener("change", () => {
      showGhosts = histToggle.checked;
      cy.edges().forEach((e) => {
        const info = P[e.data("particleId")];
        if (info && info.ghost) e.toggleClass("hidden", !showGhosts);
      });
      if (panelSubject) showSubject(panelSubject);
    });
  }

  // --- detail panel (text via textContent: particle content is untrusted) --
  const chainOf = (pid: string): string[] => {
    let head_ = pid;
    while (predOf[head_]) head_ = predOf[head_];
    const chain = [head_];
    while (succOf[head_]) {
      head_ = succOf[head_];
      chain.push(head_);
    }
    return chain;
  };

  const particleRow = (pid: string): HTMLElement => {
    const info = P[pid];
    const row = el(
      "div",
      "prow" + (info.ghost ? " ghost" : "") + (info.retrieval_hit ? " hit" : ""),
    );
    const head_ = el("div");
    head_.appendChild(el("span", `chip ${info.status}`, info.status));
    if (info.contested) {
      const bases = (info.contested.bases ?? []).join("+");
      const c = el("span", "chip contested", `contested: ${bases}`);
      let t = `bases: ${(info.contested.bases ?? []).join(", ")}`;
      if (info.contested.inconsistency_id) t += `\nINCONSISTENCY ${info.contested.inconsistency_id}`;
      if (info.contested.caveat) t += `\n${info.contested.caveat}`;
      c.title = t;
      head_.appendChild(c);
    }
    if (info.retrieval_hit) head_.appendChild(el("span", "chip hitchip", "retrieval hit"));
    row.appendChild(head_);
    row.appendChild(
      el("div", "content", (info.status === "RETRACTED" ? "☒ " : "") + info.content),
    );
    const bar = el("div", "bar");
    const fill = document.createElement("i");
    fill.style.width = `${Math.round(100 * info.effective_confidence)}%`;
    fill.style.opacity = String(Math.max(0.25, info.effective_confidence));
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(
      el(
        "div",
        "kv",
        `confidence ${info.confidence.toFixed(2)} → effective ` +
          `${info.effective_confidence.toFixed(2)}` +
          ((info.utility_score ?? 0) > 0 ? ` · utility ${(info.utility_score ?? 0).toFixed(1)}` : "") +
          ` · asserted ${utcDay(info.asserted_at)}`,
      ),
    );
    if (info.contested) {
      // The chip's hover tooltip is invisible on touch and easy to miss —
      // spell the drill-down out as panel text. The INCONSISTENCY id links
      // to the contradiction's evidence render (scope=inconsistency
      //): the anchor record, both disputants with their true
      // statuses, their subjects and sources.
      const kv = el("div", "kv contested-kv");
      kv.appendChild(
        document.createTextNode(`contested — bases: ${(info.contested.bases ?? []).join(", ")}`),
      );
      if (info.contested.inconsistency_id) {
        const iid = info.contested.inconsistency_id;
        kv.appendChild(document.createTextNode(" · "));
        const a = el("a", "subject-link", `INCONSISTENCY ${iid.slice(0, 8)} — show the conflict`);
        a.addEventListener("click", () =>
          navigate("browse", { scope: "inconsistency", inconsistency_id: iid }),
        );
        kv.appendChild(a);
      }
      if (info.contested.caveat) {
        kv.appendChild(document.createTextNode(` · ${info.contested.caveat}`));
      }
      row.appendChild(kv);
    }
    if (info.as_of_note) {
      row.appendChild(
        el(
          "div",
          "kv",
          `as-of note: since retired (${info.as_of_note.status}, ` +
            `${utcDay(info.as_of_note.retired_at) || "undated"}, basis: ` +
            `${info.as_of_note.basis})`,
        ),
      );
    }
    const chain = chainOf(pid);
    if (chain.length > 1) {
      const c = el("div", "chain", "chain: ");
      chain.forEach((cid, i) => {
        if (i) c.appendChild(document.createTextNode(" ⟶ "));
        const a = el("a", "", cid.slice(0, 8) + (cid === pid ? " (this)" : ""));
        a.addEventListener("click", () => showParticle(cid));
        c.appendChild(a);
      });
      row.appendChild(c);
    } else if (info.supersedes) {
      // An ACTIVE belief that replaced an earlier one should say so even in
      // a current-only render (the predecessor itself loads with history).
      const c = el("div", "chain", `revises ${info.supersedes.slice(0, 8)} `);
      if (!p.history) {
        const a = el("a", "", "(show history)");
        a.addEventListener("click", () =>
          navigate("browse", hashParams({ ...p, history: true })),
        );
        c.appendChild(a);
      }
      row.appendChild(c);
    }
    if (info.source_uri && /^https?:\/\//.test(info.source_uri)) {
      const kv = el("div", "kv");
      const a = document.createElement("a");
      a.textContent = "source";
      a.href = info.source_uri;
      a.rel = "noopener noreferrer";
      a.target = "_blank";
      kv.appendChild(a);
      row.appendChild(kv);
    } else if (info.source_uri) {
      row.appendChild(el("div", "kv", `source: ${info.source_uri}`));
    }
    if ((info.subject_ids ?? []).length > 2) {
      row.appendChild(
        el(
          "div",
          "kv",
          `spans ${info.subject_ids.length} subjects (drawn as a pairwise clique)`,
        ),
      );
    }
    return row;
  };

  const showParticle = (pid: string): void => {
    panelSubject = null;
    panel.className = "gpanel open";
    panel.replaceChildren(el("h2", "", `Particle ${pid.slice(0, 8)}`), particleRow(pid));
  };

  const showSubject = (sid: string): void => {
    const n = data.nodes.find((x) => x.subject_id === sid);
    if (!n) return;
    panelSubject = sid;
    panel.className = "gpanel open";
    panel.replaceChildren();
    panel.appendChild(el("h2", "", n.label ?? sid));
    const idRow = el("div", "sub");
    const code = el("code", "", n.subject_id);
    idRow.appendChild(code);
    idRow.appendChild(document.createTextNode(" "));
    // Re-anchor the render on this subject — a deep-link navigation, so the
    // hash (and any copied URL) always names what is on screen.
    if (!(data.scope_type === "subject" && sid === data.scope_ref)) {
      const anchor = el("a", "copyid", "anchor here");
      anchor.addEventListener("click", () =>
        navigate("browse", {
          scope: "subject",
          subject_id: sid,
          ...(p.as_of ? { as_of: p.as_of } : {}),
          ...(p.history ? { history: "true" } : {}),
        }),
      );
      idRow.appendChild(anchor);
    }
    panel.appendChild(idRow);
    panel.appendChild(
      el(
        "div",
        "sub",
        (n.subject_class ? `${n.subject_class} · ` : "") +
          `hop ${n.hop ?? 0} · best-supported claim ` +
          `${(n.max_effective_confidence ?? 0).toFixed(2)}` +
          ((n.utility_score ?? 0) > 0
            ? ` · utility evidence ${(n.utility_score ?? 0).toFixed(1)}`
            : ""),
      ),
    );
    let listed = 0;
    for (const pid of n.cargo ?? []) {
      const info = P[pid];
      if (!info || (info.ghost && !showGhosts)) continue;
      panel.appendChild(particleRow(pid));
      listed += 1;
    }
    const edgeParticles = Object.keys(P).filter(
      (pid) =>
        (P[pid].subject_ids ?? []).length >= 2 &&
        (P[pid].subject_ids ?? []).indexOf(sid) !== -1 &&
        (showGhosts || !P[pid].ghost),
    );
    if (edgeParticles.length) {
      panel.appendChild(el("div", "sub", "edges (multi-subject particles):"));
      for (const pid of edgeParticles) panel.appendChild(particleRow(pid));
    }
    if ((n.cargo_truncated ?? 0) > 0) {
      panel.appendChild(
        el(
          "div",
          "sub",
          `⚠ ${n.cargo_truncated} more particle(s) beyond the panel cap — raise ` +
            "graph.max_particles_per_subject to see them",
        ),
      );
    }
    if (!listed && !edgeParticles.length) {
      panel.appendChild(el("div", "sub", "no particles on this lens"));
    }
  };

  cy.on("tap", "node", (ev) => showSubject((ev.target as CyEle).id()));
  cy.on("tap", "edge", (ev) => showParticle((ev.target as CyEle).data("particleId")));
  cy.on("tap", (ev) => {
    if (ev.target === (cy as unknown)) {
      panel.className = "gpanel";
      panelSubject = null;
    }
  });

  // --- tick-click change listing -------------------------------------------
  // "The tick showed me something changed — what?" List, from this render's
  // payload, everything that carries that day: assertions (a revision's
  // successor shows its 'revises' chain) and known retirements. A change the
  // current lens doesn't carry (e.g. a retirement in a current-only render)
  // is disclosed as such, with history / as-of jumps to widen the lens.
  inspectInstant = (day: string): void => {
    panelSubject = null;
    panel.className = "gpanel open";
    panel.replaceChildren(el("h2", "", `Changed on ${day}`));

    const actions = el("div", "sub");
    const jump = el("a", "copyid", `view render as of ${day}`);
    jump.addEventListener("click", () =>
      navigate("browse", hashParams({ ...p, as_of: day })),
    );
    actions.appendChild(jump);
    if (!p.history) {
      actions.appendChild(document.createTextNode(" · "));
      const hist = el("a", "copyid", "show history");
      hist.addEventListener("click", () =>
        navigate("browse", hashParams({ ...p, history: true })),
      );
      actions.appendChild(hist);
    }
    panel.appendChild(actions);

    const asserted = Object.keys(P).filter((pid) => utcDay(P[pid].asserted_at) === day);
    const retired = Object.keys(P).filter(
      (pid) => P[pid].as_of_note && utcDay(P[pid].as_of_note?.retired_at) === day,
    );
    // Cap each section: a store's initial-harvest day can carry hundreds of
    // assertions, and the panel is a drill-down, not a dump.
    const LIST_CAP = 30;
    const listSection = (label: string, pids: string[]): void => {
      if (!pids.length) return;
      panel.appendChild(el("div", "sub", `${label} (${pids.length}):`));
      for (const pid of pids.slice(0, LIST_CAP)) panel.appendChild(particleRow(pid));
      if (pids.length > LIST_CAP) {
        panel.appendChild(
          el("div", "sub", `…and ${pids.length - LIST_CAP} more ${label} this day`),
        );
      }
    };
    listSection("asserted", asserted);
    listSection("retired", retired);
    if (!asserted.length && !retired.length) {
      panel.appendChild(
        el(
          "div",
          "sub",
          "No rendered particle carries this instant — the change sits outside " +
            "this render's lens. Try 'show history' (retired beliefs) or view " +
            "the render as of this day.",
        ),
      );
    }
  };

  // The evidence scope's whole point is the conflict itself — open the panel
  // with the foreground listing immediately (the INCONSISTENCY anchor first,
  // then the disputants) instead of making the operator hunt for something
  // to tap. When the evidence predates subject binding there may be no nodes
  // at all (the engine disclosed it), so this panel IS the render.
  if (data.scope_type === "inconsistency") {
    const foreground = Object.keys(P).filter((pid) => P[pid].retrieval_hit);
    foreground.sort((x, y) =>
      x === data.scope_ref ? -1 : y === data.scope_ref ? 1 : x < y ? -1 : 1,
    );
    panel.className = "gpanel open";
    panel.replaceChildren(el("h2", "", "Conflict evidence"));
    panel.appendChild(
      el(
        "div",
        "sub",
        "The INCONSISTENCY record and the beliefs it disputes, with their " +
          "current statuses.",
      ),
    );
    for (const pid of foreground) panel.appendChild(particleRow(pid));
  }
}

// ---------------------------------------------------------------------------
// As-of scrubber: the slider maps [earliest assertion … now];
// releasing it navigates the deep link with as_of=<date> (or drops as_of at
// the "now" end), which re-renders the route via a fresh GET /graph — one
// re-query per instant, all epistemics recomputed server-side.
// ---------------------------------------------------------------------------

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * UTC calendar day of a timestamp string ("" when unparseable). The one day
 * convention across the graph render — scrubber ticks, the change listing,
 * and displayed dates — so an offset-bearing timestamp cannot render as one
 * day and match as another (as-of jumps are UTC-day granular).
 */
function utcDay(s: unknown): string {
  const t = Date.parse(String(s ?? ""));
  return Number.isNaN(t) ? "" : new Date(t).toISOString().slice(0, 10);
}

// Change instants discovered per scope, unioned across re-renders: an as-of
// render only carries the beliefs visible at its instant, so without this
// memory the tick marks (and slider bounds) ahead of T would vanish the
// moment you scrub back to T — leaving no visible way forward. Presentation
// state only (timestamps); keyed by scope so different renders don't mix.
const instantsByScope = new Map<string, Set<number>>();

function buildScrubber(
  data: GraphData,
  p: GraphParams,
  onInspect: (day: string) => void,
): HTMLElement | null {
  const scopeKey = `${data.scope_type}:${data.scope_ref}`;
  let known = instantsByScope.get(scopeKey);
  if (!known) {
    known = new Set<number>();
    instantsByScope.set(scopeKey, known);
  }
  // Fold this render's timestamps into the scope's known change instants:
  // every rendered assertion, plus every known retirement crossing.
  for (const info of Object.values(data.particles ?? {})) {
    const t = Date.parse(String(info.asserted_at));
    if (!Number.isNaN(t)) known.add(t);
    if (info.as_of_note && info.as_of_note.retired_at) {
      const r = Date.parse(String(info.as_of_note.retired_at));
      if (!Number.isNaN(r)) known.add(r);
    }
  }

  // Bounds (presentation only): earliest known change instant up to now.
  let earliest = Number.NaN;
  for (const t of known) {
    if (!(t >= earliest)) earliest = t;
  }
  // An as-of render may have excluded everything — the slider must survive
  // an empty instant so the operator can scrub back out of it.
  if (p.as_of) {
    const t = Date.parse(p.as_of);
    if (!Number.isNaN(t) && !(t >= earliest)) earliest = t;
  }
  const now = Date.now();
  if (Number.isNaN(earliest) || now - earliest < 2 * DAY_MS) {
    return null; // nothing to scrub over
  }
  // Start the window a day before the first assertion so the leftmost stop
  // shows the pre-belief store (an empty render is an honest answer).
  const minT = earliest - DAY_MS;

  const wrap = el("div", "asof-scrub");
  const label = el("span", "asof-label");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "1000";
  slider.title =
    "As-of scrubber: tick marks are the render's belief-change " +
    "instants (assertions, supersessions, retirements)";

  const posFor = (t: number): number =>
    Math.round(((t - minT) / (now - minT)) * 1000);
  const tFor = (pos: number): number => minT + (pos / 1000) * (now - minT);
  const dateStr = (t: number): string => new Date(t).toISOString().slice(0, 10);

  // Tick marks at the known change instants (the per-scope union above) so
  // the operator scrubs to change points instead of hunting for them.
  // Rendered as explicit marks (native datalist ticks are invisibly faint on
  // the dark theme), and clickable: a tick jumps the render to its instant.
  // One tick per DAY, not per instant: a real store asserts most particles
  // at distinct second-level timestamps (a 227-particle render carries ~225
  // instants), which renders as an unreadable forest. Day granularity also
  // matches the as-of jump granularity. Each tick sits at its day's earliest
  // instant.
  const days = new Map<string, number>();
  for (const t of known) {
    if (t < minT || t > now) continue;
    const d = dateStr(t);
    const cur = days.get(d);
    if (cur === undefined || t < cur) days.set(d, t);
  }
  const ticksBar = el("div", "asof-ticks");
  for (const [d, t] of [...days.entries()].sort((a, b) => a[1] - b[1])) {
    const tick = el("span", "tick");
    tick.style.left = `${(posFor(t) / 1000) * 100}%`;
    tick.title = `belief change ${d} — click to list what changed`;
    // Click → the change listing in the detail panel (which offers the
    // "view render as of this day" jump); the slider itself stays the way
    // to move the whole render through time.
    tick.onclick = () => onInspect(d);
    ticksBar.appendChild(tick);
  }

  const current = p.as_of ? Date.parse(p.as_of) : now;
  slider.value = String(Number.isNaN(current) ? 1000 : posFor(current));
  label.textContent = p.as_of ? `as of ${dateStr(current)}` : "now";

  // Dragging previews the date locally; releasing issues the re-query.
  slider.addEventListener("input", () => {
    const pos = Number(slider.value);
    label.textContent = pos >= 1000 ? "now" : `as of ${dateStr(tFor(pos))}`;
  });
  slider.addEventListener("change", () => {
    const pos = Number(slider.value);
    const next: GraphParams = { ...p };
    if (pos >= 1000) {
      delete next.as_of;
    } else {
      next.as_of = dateStr(tFor(pos));
    }
    navigate("browse", hashParams(next));
  });

  const track = el("div", "scrub-track");
  track.append(slider, ticksBar);
  wrap.append(track, label);
  return wrap;
}

// ---------------------------------------------------------------------------
// Encoding helpers — geometry only; the epistemic values arrive from the
// server. Kept in lockstep with exporters/graph/render.py.
// ---------------------------------------------------------------------------

function nodeSize(mass: number, u: number): number {
  // Structural mass (particles on the subject: cargo + incident edges) sets
  // the base scale; utility evidence (the size encoding) adds on
  // top. Both log-scaled and capped so a hub can't dwarf the render.
  return Math.round(22 + Math.min(30, 9 * Math.log1p(mass)) + Math.min(24, 12 * Math.log1p(u)));
}

function nodeShade(m: number): string {
  // Best-supported-claim aggregate → fill lightness. The ramp's ends are the
  // design tokens --p-graph-node-l-weak/-strong, which invert between themes
  // so "better supported" is always the higher-contrast end (darker on light,
  // lighter on dark — the exporter's light-canvas ramp is the light case).
  const weak = tokenPercent("--p-graph-node-l-weak");
  const strong = tokenPercent("--p-graph-node-l-strong");
  const t = Math.max(0, Math.min(1, m));
  const hue = tokenValue("--p-graph-node-hue") || "214";
  return `hsl(${hue}, 45%, ${Math.round(weak + (strong - weak) * t)}%)`;
}

function edgeStyleFor(info: GraphParticleInfo): string {
  if (info.status === "SUPERSEDED") return "dashed";
  if (info.status === "PROVENANCE_STALE") return "dotted";
  if (info.status === "RETRACTED") return "dashed";
  return "solid";
}

const LEGEND: ReadonlyArray<[string, string]> = [
  ["opacity", "effective confidence (server-computed — decay is literal fading)"],
  ["solid / dashed / dotted", "ACTIVE / SUPERSEDED ghost / PROVENANCE_STALE"],
  ["☒", "RETRACTED tombstone"],
  ["⚠", "contested — open the panel for the fired bases"],
  ["node size", "particles on the subject + utility evidence (how often used)"],
  ["node shade", "best-supported claim on that subject (display aggregate)"],
  ["bold blue", "retrieval hit (query scope)"],
];

function legend(): HTMLElement {
  const wrap = el("div", "graph-legend");
  for (const [mark, meaning] of LEGEND) {
    const span = el("span");
    span.appendChild(el("b", "", mark));
    span.appendChild(document.createTextNode(` ${meaning}`));
    wrap.appendChild(span);
  }
  return wrap;
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
