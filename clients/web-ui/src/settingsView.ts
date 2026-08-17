/*
 * Settings screen: engine base URL + the client bearer +
 * reviewer id. Fail-closed: until this screen has been saved, the feed shows it
 * and makes no API call. The base URL is optional (empty = same-origin, the
 * default served-from-the-engine case).
 *
 * Saving with an empty token probes the engine (`probeAuth`) instead of
 * refusing: whether a bearer is needed is the engine's fact, not the
 * operator's, and a keyless loopback engine used to be reachable only by
 * typing a value it would ignore. Every branch of the button reports what
 * happened — silence was the original defect here.
 */
import { type EngineSettings, ParticlesApiClient } from "./api";
import {
  loadReviewerId,
  loadSettings,
  saveReviewerId,
  saveSettings,
} from "./settings";
import { type ThemeChoice, applyTheme, loadTheme, saveTheme } from "./theme";

export interface SettingsViewDeps {
  onSaved: (settings: EngineSettings, reviewerId: string) => void;
}

export function renderSettings(root: HTMLElement, deps: SettingsViewDeps): void {
  const current = loadSettings();
  const reviewer = loadReviewerId();

  root.innerHTML = "";
  const panel = document.createElement("div");
  panel.className = "settings";

  const h = document.createElement("h2");
  h.textContent = "Engine connection";
  panel.appendChild(h);

  const intro = document.createElement("p");
  intro.className = "hint";
  intro.textContent =
    "The PWA is a thin HTTP client of the Particles engine, fail-closed: it calls nothing until you have saved this screen. When the engine serves this app same-origin, leave the base URL blank. The recommended exposure is a private mesh / SSH tunnel (Tailscale); a token over plain HTTP to a non-loopback host is your risk.";
  panel.appendChild(intro);

  const urlInput = field(
    panel,
    "Engine base URL (optional)",
    "Leave blank for same-origin (served from the engine). Or e.g. http://127.0.0.1:8000",
    current.baseUrl,
    "text",
  );

  // The token is the engine's own PARTICLES_API_KEY (the client bearer, ADR
  // 0137, checked against it by the gate). There is no mint/rotate
  // endpoint, so the hint has to name the env var: the operator who started
  // the engine is the only source, and nothing in this app can derive it.
  const tokenInput = field(
    panel,
    "Bearer token",
    "Paste the engine's PARTICLES_API_KEY — the secret whoever started the engine set. Nothing is minted here. Leave it blank if the engine was started without a key: saving will ask the engine and say so either way. Stored in this browser only; sent as 'Authorization: Bearer …' on every request.",
    current.bearerToken,
    "password",
  );

  const reviewerInput = field(
    panel,
    "Reviewer id",
    "Recorded on review resolutions (POST /review reviewer_id). Required before the Comment gesture.",
    reviewer,
    "text",
  );

  // Appearance is a presentation preference, not connection config: it applies
  // and persists on change, independent of the fail-closed Save below, and
  // gates nothing. The tokens (tokens.css) do the actual theming off the
  // <html data-theme> stamp that applyTheme sets.
  themeField(panel);

  // Every outcome of the button lands here. Before this existed, submitting an
  // empty token re-rendered this same screen (onSaved → renderRoute →
  // !isConfigured → showSettings), which is indistinguishable from a dead
  // button: no error, no change, nothing.
  const status = document.createElement("div");
  status.className = "hint";
  status.hidden = true;
  panel.appendChild(status);

  const say = (cls: "info" | "error", text: string): void => {
    status.className = `banner ${cls}`;
    status.textContent = text;
    status.hidden = false;
  };

  const row = document.createElement("div");
  row.className = "btn-row";
  const save = document.createElement("button");
  save.className = "btn primary";
  save.textContent = "Save & start curating";

  const submit = async (): Promise<void> => {
    const settings: EngineSettings = {
      baseUrl: urlInput.value.trim(),
      bearerToken: tokenInput.value.trim(),
      allowUnauthenticated: false,
    };

    // No token is a legitimate state, not a mistake: a loopback engine started
    // without PARTICLES_API_KEY skips the bearer check entirely. Which engine
    // this is, only the engine knows — so ask it rather than making the
    // operator guess, or making them type a value it would ignore.
    if (!settings.bearerToken) {
      save.disabled = true;
      say("info", "No token entered — asking the engine whether it needs one…");
      const probe = await new ParticlesApiClient(settings).probeAuth();
      save.disabled = false;
      switch (probe.kind) {
        case "open":
          settings.allowUnauthenticated = true;
          break;
        case "token-required":
          say(
            "error",
            "This engine requires a bearer token: it rejected an unauthenticated " +
              "request (401). Paste the PARTICLES_API_KEY it was started with.",
          );
          return;
        case "refuses-unauthenticated":
          say(
            "error",
            "This engine refuses unauthenticated requests from a non-loopback " +
              "client (503). It has to be started with a PARTICLES_API_KEY — " +
              "paste that value here.",
          );
          return;
        case "unreachable":
          say("error", `Could not reach the engine — ${probe.detail}`);
          return;
      }
    }

    saveSettings(settings);
    saveReviewerId(reviewerInput.value.trim());
    deps.onSaved(settings, reviewerInput.value.trim());
  };

  save.onclick = () => {
    void submit();
  };
  row.appendChild(save);
  panel.appendChild(row);

  root.appendChild(panel);
}

const THEME_CHOICES: ReadonlyArray<[ThemeChoice, string]> = [
  ["system", "Follow system"],
  ["light", "Light"],
  ["dark", "Dark"],
];

function themeField(parent: HTMLElement): void {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const l = document.createElement("label");
  l.textContent = "Appearance";
  const select = document.createElement("select");
  select.className = "theme-select";
  const current = loadTheme();
  for (const [value, label] of THEME_CHOICES) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    opt.selected = value === current;
    select.appendChild(opt);
  }
  select.onchange = () => {
    const choice = select.value as ThemeChoice;
    saveTheme(choice);
    applyTheme(choice);
  };
  const h = document.createElement("div");
  h.className = "hint";
  h.textContent =
    "Light, dark, or whatever the device prefers. Applies immediately; stored in this browser only.";
  wrap.append(l, select, h);
  parent.appendChild(wrap);
}

function field(
  parent: HTMLElement,
  label: string,
  hint: string,
  value: string,
  type: "text" | "password",
): HTMLInputElement {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const l = document.createElement("label");
  l.textContent = label;
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  const h = document.createElement("div");
  h.className = "hint";
  h.textContent = hint;
  wrap.append(l, input, h);
  parent.appendChild(wrap);
  return input;
}
