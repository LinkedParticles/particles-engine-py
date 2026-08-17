/*
 * Particles web UI — app entry.
 *
 * One same-origin app, one mount: the bus-stop curation feed, the
 * query surface, and the graph view share a single
 * shell with client-side hash routing (#/curate, #/query, #/browse — all
 * deep-linkable; legacy #/queue and #/graph are permanent aliases), plus
 * deposit and settings as global actions. The tab order is the pipeline
 * order — curate what came in, query it, browse the graph it forms — with
 * deposit (the pipeline's start) as the always-available shell action. The app is a
 * thin typed HTTP client of the FastAPI engine: it holds no store, no schema
 * logic, no reconciliation, and renders only server-computed epistemics
 * (the mobile-parity rule generalizes "the client
 * only renders" app-wide).
 *
 * Served same-origin from the engine's /app mount (web_app.py), so its
 * authenticated fetch calls trigger no CORS preflight.
 */
import { EngineSettings, ParticlesApiClient } from "./api";
import { openDepositSheet } from "./deposit";
import { CurationFeed } from "./feed";
import { renderGraphView } from "./graphView";
import { renderQueryView } from "./queryView";
import { Route, RouteName, currentRoute, navigate, onRouteChange, routeHash } from "./router";
import { loadReviewerId, loadSettings } from "./settings";
import { renderSettings } from "./settingsView";
import { applyTheme } from "./theme";

// Injected by esbuild at build time (see esbuild.config.mjs); also names the
// service-worker cache, so a mismatch here means a stale bundle.
declare const __BUILD_VERSION__: string;

const NAV_ITEMS: ReadonlyArray<{ name: RouteName; label: string }> = [
  { name: "curate", label: "Curate" },
  { name: "query", label: "Query" },
  { name: "browse", label: "Browse" },
];

class WebUiApp {
  private readonly navEl: HTMLElement;
  private viewEl: HTMLElement;
  private settings: EngineSettings;
  private reviewerId: string;
  private client: ParticlesApiClient;
  private feed: CurationFeed | null = null;
  // Where "save" returns to. Settings is reached from wherever you were, so
  // leaving it should put you back there rather than always on the feed.
  private returnTo: string | null = null;
  // Last hash seen per route, so switching tabs returns to where you were —
  // the query you asked, the graph scope you were on — instead of a blank
  // route. Captured on every route render AND on tab-click (a view may have
  // updated its params via history.replaceState without a hashchange).
  private lastHash: Partial<Record<RouteName, string>> = {};

  constructor(root: HTMLElement) {
    this.settings = loadSettings();
    this.reviewerId = loadReviewerId();
    this.client = new ParticlesApiClient(this.settings);
    this.navEl = document.createElement("nav");
    this.navEl.className = "topnav";
    this.viewEl = document.createElement("div");
    this.viewEl.className = "view";
    root.append(this.navEl, this.viewEl);
  }

  start(): void {
    onRouteChange(() => this.renderRoute());
    // The fail-closed bounce to settings lives in renderRoute, so it covers
    // every entry point — first load, a hashchange, a deep link someone sent.
    this.renderRoute();
  }

  // --- Shell --------------------------------------------------------------

  private renderNav(active: RouteName): void {
    this.navEl.innerHTML = "";
    this.navEl.appendChild(renderBrand());
    for (const item of NAV_ITEMS) {
      const a = document.createElement("a");
      a.className = "tab" + (item.name === active ? " active" : "");
      a.textContent = item.label;
      a.href = this.lastHash[item.name] ?? routeHash(item.name);
      a.onclick = () => {
        // Before the hash changes, remember where the route being left is —
        // including replaceState updates the hashchange listener never saw.
        const cur = currentRoute();
        this.lastHash[cur.name] = window.location.hash || routeHash(cur.name);
      };
      this.navEl.appendChild(a);
    }
    const spacer = document.createElement("span");
    spacer.className = "spacer";
    this.navEl.appendChild(spacer);

    // Global actions: deposit and settings live on the shell,
    // not on any one route.
    const deposit = document.createElement("button");
    deposit.className = "navbtn";
    // The label is a separate span so the narrow-screen rule can drop the word
    // and keep the glyph. `title`/`aria-label` carry the name once it does —
    // the button must not become an unlabelled ＋ to a screen reader.
    deposit.title = "Deposit";
    deposit.setAttribute("aria-label", "Deposit");
    deposit.append("＋", labelSpan("Deposit"));
    deposit.onclick = () => {
      void openDepositSheet(this.client).then((outcome) => {
        if (outcome) this.flash(outcome.ok ? "info" : "error", outcome.message);
      });
    };
    // A route, so it takes the same treatment as a tab: it shows active state
    // and it pushes history, which is what makes Back leave settings. The
    // word is spelled out beside the glyph (dropped only on narrow screens,
    // where title/aria-label carry it) — `title` alone is a poor label: it is
    // invisible on touch and delayed on hover.
    const gear = document.createElement("button");
    gear.className = "navbtn" + (active === "settings" ? " active" : "");
    gear.title = "Settings";
    gear.setAttribute("aria-label", "Settings");
    gear.append("⚙", labelSpan("Settings"));
    gear.onclick = () => this.goto(routeHash("settings"));
    this.navEl.append(deposit, gear);
  }

  private flash(cls: "info" | "error", text: string): void {
    const el = document.createElement("div");
    el.className = `banner ${cls}`;
    el.textContent = text;
    this.viewEl.prepend(el);
  }

  /**
   * Swap in a fresh view container and return it. Every route render targets
   * its own element, so an in-flight async render from a *previous* route
   * (e.g. a slow GET /curation on a large store) completes into a detached
   * node instead of stomping whatever the operator navigated to since.
   */
  private freshView(): HTMLElement {
    const next = document.createElement("div");
    next.className = "view";
    this.viewEl.replaceWith(next);
    this.viewEl = next;
    return next;
  }

  // --- Routes -------------------------------------------------------------

  private renderRoute(route?: Route): void {
    let r = route ?? currentRoute();
    if (r.name !== "settings" && !this.client.isConfigured()) {
      // Fail-closed: no credential, no calls — settings is the app until one
      // is saved. replaceState rather than a push: every route
      // bounces here while unconfigured, so a pushed entry would make Back
      // ping-pong between the two forever. replaceState fires no hashchange,
      // hence the direct re-render.
      window.history.replaceState(null, "", routeHash("settings"));
      r = { name: "settings", params: new URLSearchParams() };
    }
    if (r.name !== "settings") {
      this.returnTo = window.location.hash || routeHash(r.name);
    }
    this.lastHash[r.name] = window.location.hash || routeHash(r.name);
    this.renderNav(r.name);
    const view = this.freshView();
    switch (r.name) {
      case "curate":
        this.showFeed(view);
        break;
      case "query":
        renderQueryView(view, this.viewDeps(), r.params);
        break;
      case "browse":
        renderGraphView(view, this.viewDeps(), r.params);
        break;
      case "settings":
        this.showSettings(view);
        break;
    }
  }

  private viewDeps(): { client: ParticlesApiClient; onNeedsSettings: () => void } {
    return {
      client: this.client,
      onNeedsSettings: () => this.goto(routeHash("settings")),
    };
  }

  private showFeed(view: HTMLElement): void {
    // The feed persists across route switches so a slow queue build survives
    // navigating away (it re-attaches to each fresh view container).
    if (!this.feed) {
      this.feed = new CurationFeed(view, {
        client: this.client,
        reviewerId: this.reviewerId,
        onNeedsSettings: () => this.goto(routeHash("settings")),
      });
    } else {
      this.feed.updateDeps({
        client: this.client,
        reviewerId: this.reviewerId,
        onNeedsSettings: () => this.goto(routeHash("settings")),
      });
    }
    this.feed.attach(view);
  }

  // --- Settings (the #/settings route) ------------------------

  private showSettings(view: HTMLElement): void {
    renderSettings(view, {
      onSaved: (settings, reviewerId) => {
        this.settings = settings;
        this.reviewerId = reviewerId;
        this.client.updateSettings(settings);
        this.goto(this.returnTo ?? routeHash("curate"));
      },
    });
  }

  /**
   * Go to a hash, re-rendering even when it is the one already showing.
   *
   * Assigning an unchanged `location.hash` fires no hashchange, so the
   * listener never runs — which is exactly the case on save when settings was
   * reached from the feed and the feed is where we return.
   */
  private goto(hash: string): void {
    if (window.location.hash === hash) this.renderRoute();
    else window.location.hash = hash;
  }
}

/** A nav-button word that the narrow-screen rule may hide, leaving its glyph. */
function labelSpan(text: string): HTMLElement {
  const span = document.createElement("span");
  span.className = "navbtn-label";
  span.textContent = ` ${text}`;
  return span;
}

/**
 * The app's name and mark, at the head of the nav on every route.
 *
 * `icon.svg` is the PWA's own installed icon, already beside the bundle under
 * `/app/` — one asset for the home screen, the browser tab and here, so the
 * installed app and the served page cannot drift apart. Relative `src`, like
 * every other asset in the shell, because the mount is `/app/` and not the
 * engine root.
 *
 * Deliberately not a link: the tab strip immediately to its right already owns
 * navigation, and a wordmark that silently duplicates "Curate" is a trap for
 * anyone who reads a logo as "go home".
 */
function renderBrand(): HTMLElement {
  const brand = document.createElement("span");
  brand.className = "brand";
  const mark = document.createElement("img");
  mark.className = "brand-mark";
  mark.src = "icon.svg";
  // Decorative: the wordmark beside it already carries the name, so announcing
  // it twice is noise to a screen reader.
  mark.alt = "";
  const name = document.createElement("span");
  name.className = "brand-name";
  name.textContent = "Particles";
  brand.append(mark, name);
  return brand;
}

/**
 * Diagnostics footer: which engine this is, and which bundle is rendering it.
 *
 * The **engine version** leads because it is the one that identifies the
 * software — it is the SDK release, so it maps to a tag and a changelog
 * section, and comparing it against the release you expect is how you learn a
 * long-running engine (a container started weeks ago, say) is behind. It is
 * fetched unconditionally: `GET /health` is unauthenticated, so this resolves
 * before a bearer is ever saved, on the settings screen where "what am I
 * pointed at?" is the live question. `unreachable` there means the engine is
 * down or the base URL is wrong — never that the token is bad, since none is
 * sent. The trailing `web-ui` string identifies only this bundle (an esbuild
 * hash of its own inputs, not a release): if it does not match your last
 * `npm run build`, the service worker is serving a stale bundle — hard-reload
 * or clear site data.
 */
function showBuildInfo(): void {
  const footer = document.getElementById("build-info");
  if (!footer) return;
  const stamp = (engine: string): void => {
    footer.textContent = `engine ${engine} · web-ui ${__BUILD_VERSION__}`;
  };
  stamp("…");
  new ParticlesApiClient(loadSettings())
    .health()
    .then((h) => {
      // `built_at` is present only when the artifact stamped itself — the
      // container image does, a source checkout does not — so the date
      // appears exactly where it means something. Rendered as the calendar
      // day: the question it answers ("is this months old?") has no use for
      // the time, and a full ISO instant in a footer reads as noise.
      const built = h.built_at ? ` (built ${String(h.built_at).slice(0, 10)})` : "";
      stamp(`${h.version}${built}`);
    })
    .catch(() => stamp("unreachable"));
}

function bootstrap(): void {
  const root = document.getElementById("app");
  if (!root) return;
  // Re-apply the saved appearance (index.html stamped it pre-paint; this keeps
  // the two in agreement if storage changed underneath a cached shell).
  applyTheme();
  // Land on the curation feed when no route is present, so pre-unification
  // bookmarks ("/app" with no fragment) behave exactly as before.
  if (!window.location.hash) navigate("curate");
  new WebUiApp(root).start();
  showBuildInfo();

  // Register the app-shell service worker (cache shell only, never API — see
  // public/service-worker.js). Best-effort: a failure here never blocks the app.
  if ("serviceWorker" in navigator) {
    // A rebuilt bundle installs a new worker (its cache name carries the build
    // version) which claims the page — but the PAGE is still the old cached
    // shell until the next reload. That gap is exactly the "the fix didn't
    // ship" confusion: reload once when a new worker takes over from an old
    // one, so a new build lands on the visit that discovers it. A first-ever
    // visit (no prior controller) is already the newest shell — no reload;
    // the boolean guard stops any loop.
    const wasControlled = navigator.serviceWorker.controller !== null;
    let reloadedForNewWorker = false;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!wasControlled || reloadedForNewWorker) return;
      reloadedForNewWorker = true;
      window.location.reload();
    });
    window.addEventListener("load", () => {
      void navigator.serviceWorker.register("service-worker.js").catch(() => {
        /* offline shell is a nicety, not a requirement */
      });
    });
  }
}

bootstrap();
