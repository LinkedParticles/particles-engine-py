/*
 * #/query — the query surface.
 *
 * Non-streaming v1 over the existing POST /query: the client shows staged
 * progress locally while one authenticated call runs (retrieval + the paid
 * NL completion happen engine-side); a streaming rung is deferred
 *. The answer renders with its cited particles from the existing
 * response contract — stored vs effective confidence, status, and the
 * the contested badge, all server-computed (no client-side confidence
 * math).
 *
 * The integration that pays for the unification: an inline, lazily-loaded
 * render of the retrieval-set graph under the answer ("the picture of the
 * knowledge this answer consulted" — the query scope already
 * defines, mounted via the shared graphRender.ts), and per-subject chips
 * deep-linking #/browse?scope=subject&subject_id=…. Lazy because the graph
 * is a second GET /graph that re-runs retrieval — it renders on request,
 * never as a hidden tax on every question. Under a refusal the
 * section inherits the refusal framing: the hits are transparency, not
 * support, and the graph must not visually contradict the refusal.
 */
import { ApiError, ParticlesApiClient, QueryResponse } from "./api";
import { loadCytoscape, renderGraph } from "./graphRender";
import { navigate, routeHash } from "./router";

export interface QueryViewDeps {
  client: ParticlesApiClient;
  onNeedsSettings: () => void;
}

export function renderQueryView(
  container: HTMLElement,
  deps: QueryViewDeps,
  params: URLSearchParams,
): void {
  container.innerHTML = "";
  const view = el("div", "query-view");
  container.appendChild(view);

  const form = el("div", "query-form");
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Ask the store a question…";
  input.value = params.get("q") ?? "";
  const ask = document.createElement("button");
  ask.className = "btn primary";
  ask.textContent = "Ask";
  form.append(input, ask);
  view.appendChild(form);

  const results = el("div", "query-results");
  view.appendChild(results);

  const run = (): void => {
    const q = input.value.trim();
    if (!q) return;
    // Keep the hash in sync so the answer view is deep-linkable/shareable —
    // but only navigate when it actually changes (navigate re-renders).
    const target = routeHash("query", { q });
    if (window.location.hash !== target) {
      window.history.replaceState(null, "", target);
    }
    void runQuery(results, deps, q);
  };
  ask.onclick = run;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") run();
  });

  const initialQ = (params.get("q") ?? "").trim();
  if (initialQ) {
    void runQuery(results, deps, initialQ);
  } else {
    input.focus();
  }
}

// ---------------------------------------------------------------------------
// One query round-trip with staged local progress (non-streaming)
// ---------------------------------------------------------------------------

const STAGES = [
  "Embedding the question…",
  "Retrieving and ranking beliefs…",
  "Generating the cited answer…",
];

async function runQuery(
  results: HTMLElement,
  deps: QueryViewDeps,
  q: string,
): Promise<void> {
  results.innerHTML = "";
  const progress = el("div", "query-progress");
  const stageLine = el("div", "stage", STAGES[0]);
  progress.appendChild(stageLine);
  results.appendChild(progress);

  // Staged *local* progress: one non-streaming call is in flight; the stages
  // pace expectation, they do not report engine state (that rung is deferred).
  let stage = 0;
  const ticker = window.setInterval(() => {
    stage = Math.min(stage + 1, STAGES.length - 1);
    stageLine.textContent = STAGES[stage];
  }, 1800);

  let resp: QueryResponse;
  try {
    resp = await deps.client.query(q);
  } catch (e) {
    window.clearInterval(ticker);
    results.innerHTML = "";
    if (e instanceof ApiError && e.kind === "not-configured") {
      deps.onNeedsSettings();
      return;
    }
    results.appendChild(banner("error", e instanceof ApiError ? e.message : String(e)));
    return;
  }
  window.clearInterval(ticker);
  results.innerHTML = "";
  renderResponse(results, deps, q, resp);
}

function renderResponse(
  results: HTMLElement,
  deps: QueryViewDeps,
  q: string,
  resp: QueryResponse,
): void {
  // Disclosed degradation (engine-computed): answer generation failed, so
  // the "answer" is the deterministic fallback listing, not prose. Never
  // render it as if the store answered.
  if (resp.answer_generation_error) {
    results.appendChild(
      banner(
        "error",
        `Answer generation failed engine-side: ${resp.answer_generation_error} — ` +
          "the text below is a listing of the retrieved beliefs, not an answer.",
      ),
    );
  }

  // --- the answer ----------------------------------------------------------
  const answer = el("div", "answer");
  answer.appendChild(el("div", "answer-text", resp.answer ?? ""));
  results.appendChild(answer);

  // --- the knowledge consulted, inline (the unification's
  // payoff). Collapsed until asked: expanding fires one GET /graph
  // (scope=query re-runs retrieval engine-side) and mounts the shared
  // renderer — no scrubber (time travel is a Browse concern; node taps
  // promote into the full Browse view).
  const actions = el("div", "query-actions");
  const showGraph = document.createElement("button");
  showGraph.className = "btn";
  showGraph.textContent = resp.answer_refused
    ? "Show nearest beliefs as a graph (likely unrelated)"
    : "Show the knowledge consulted (graph)";
  actions.appendChild(showGraph);
  results.appendChild(actions);
  const graphHost = el("div", "inline-graph");
  results.appendChild(graphHost);
  let graphLoaded = false;
  showGraph.onclick = () => {
    if (graphLoaded) {
      // Second click toggles visibility — the render is already paid for.
      graphHost.style.display = graphHost.style.display === "none" ? "" : "none";
      return;
    }
    graphLoaded = true;
    void (async () => {
      graphHost.appendChild(el("div", "empty", "Rendering the retrieval set…"));
      try {
        const [data, cy] = await Promise.all([
          deps.client.graph({ scope: "query", q }),
          loadCytoscape(),
        ]);
        graphHost.innerHTML = "";
        if (resp.answer_refused) {
          graphHost.appendChild(
            banner(
              "info",
              "Nearest beliefs, likely unrelated (the query was refused) — " +
                "this graph is transparency about what was retrieved, not support.",
            ),
          );
        }
        renderGraph(graphHost, { scope: "query", q }, data, cy, { showScrubber: false });
        const full = document.createElement("a");
        full.className = "subject-link";
        full.textContent = "open in Browse ↗";
        full.href = routeHash("browse", { scope: "query", q });
        graphHost.appendChild(full);
      } catch (e) {
        graphHost.innerHTML = "";
        graphLoaded = false; // allow retry
        graphHost.appendChild(
          banner("error", e instanceof ApiError ? e.message : String(e)),
        );
      }
    })();
  };

  if (resp.truncation_warning) {
    results.appendChild(banner("info", resp.truncation_warning));
  }
  if (resp.as_of) {
    results.appendChild(
      banner("info", `As believed at ${String(resp.as_of).slice(0, 10)} (lens).`),
    );
  }

  // --- cited particles ------------------------------------------------------
  const hits = resp.particles ?? [];
  if (!hits.length) {
    results.appendChild(el("div", "empty", "No beliefs matched."));
    return;
  }
  // under a refusal (below-floor or responder-declared) the hits
  // are transparency, not support — never present them as citations.
  const heading = resp.answer_refused
    ? `Nearest beliefs — likely unrelated (${hits.length})`
    : `Cited beliefs (${hits.length})`;
  results.appendChild(el("h3", "cited-heading", heading));
  const effs = resp.effective_confidences ?? [];
  const allBadges = resp.contested ?? [];
  const badges = allBadges.length === hits.length ? allBadges : [];
  hits.forEach((p, i) => {
    const row = el("div", "prow qhit");
    const head = el("div");
    head.appendChild(el("span", `chip ${p.status}`, String(p.status)));
    const badge = badges.length ? badges[i] : null;
    if (badge) {
      const c = el("span", "chip contested", `contested: ${(badge.bases ?? []).join("+")}`);
      if (badge.caveat) c.title = badge.caveat;
      head.appendChild(c);
    }
    row.appendChild(head);
    row.appendChild(el("div", "content", p.content ?? ""));

    const eff = typeof effs[i] === "number" ? effs[i] : null;
    if (eff !== null) {
      const bar = el("div", "bar");
      const fill = document.createElement("i");
      fill.style.width = `${Math.round(100 * eff)}%`;
      fill.style.opacity = String(Math.max(0.25, eff));
      bar.appendChild(fill);
      row.appendChild(bar);
    }
    row.appendChild(
      el(
        "div",
        "kv",
        `confidence ${p.confidence.value.toFixed(2)}` +
          (eff !== null ? ` → effective ${eff.toFixed(2)}` : "") +
          ` · asserted ${String(p.asserted_at).slice(0, 10)}` +
          // A belief that replaced an earlier one says so — ACTIVE alone
          // hides that the store's mind changed here.
          (p.supersedes ? " · revises an earlier belief" : ""),
      ),
    );

    // Subject chips → subject-scope graph deep links.
    const sids = p.subject_ids ?? [];
    if (sids.length) {
      const chips = el("div", "subject-chips");
      for (const sid of sids) {
        const chip = document.createElement("a");
        chip.className = "subject-chip";
        chip.textContent = sid.slice(0, 8);
        chip.title = `View subject ${sid} as graph`;
        chip.onclick = () => navigate("browse", { scope: "subject", subject_id: sid });
        chips.appendChild(chip);
      }
      row.appendChild(chips);
    }
    results.appendChild(row);
  });

  // --- coverage gaps (subjects the store is thin on) ------------------------
  for (const gap of resp.subject_coverage_gaps ?? []) {
    const label = gap.subject_name ?? gap.subject_id ?? "unknown subject";
    results.appendChild(
      banner("info", `Coverage gap: ${label} — ${gap.particle_count} belief(s) on file.`),
    );
  }
}

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
