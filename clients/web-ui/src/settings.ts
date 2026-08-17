/*
 * PWA settings — engine URL + the client bearer.
 *
 * The bearer lives in the PWA's own client-side storage (browser localStorage),
 * NOT in config.yaml, secrets.py, or on ParticlesConfig — the same distinction
 * The pattern drawn for the Obsidian plugin holds: a separate non-Python process cannot
 * read secrets.py). What the PWA inherits from the lineage is the
 * discipline: a single declared credential, never in URLs/logs, read at call
 * time, with a loud early failure when missing or rejected.
 *
 * The base URL defaults to "" = same-origin. Because §2 serves the PWA from the
 * engine's `/app` mount, the operator reaches the app only through the * channel in the first place, and same-origin calls need no explicit base URL.
 */
import type { EngineSettings } from "./api";

const STORAGE_KEY = "particles.curation.settings";

export const DEFAULT_SETTINGS: EngineSettings = {
  // Same-origin: empty means "call the origin this app was served from" — the
  // engine's own host, so no cross-origin preflight.
  baseUrl: "",
  bearerToken: "",
  // Never a default: only a settings-screen probe that got a positive answer
  // out of the engine sets this (see EngineSettings.allowUnauthenticated).
  allowUnauthenticated: false,
};

export function loadSettings(): EngineSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<EngineSettings>;
    return {
      baseUrl:
        typeof parsed.baseUrl === "string"
          ? parsed.baseUrl
          : DEFAULT_SETTINGS.baseUrl,
      bearerToken:
        typeof parsed.bearerToken === "string"
          ? parsed.bearerToken
          : DEFAULT_SETTINGS.bearerToken,
      // Absent in anything stored before this field existed, which reads as
      // false — the fail-closed direction.
      allowUnauthenticated: parsed.allowUnauthenticated === true,
    };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: EngineSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

/** reviewer_id recorded on POST /review resolutions (separate from the bearer). */
const REVIEWER_KEY = "particles.curation.reviewerId";

export function loadReviewerId(): string {
  return localStorage.getItem(REVIEWER_KEY) ?? "";
}

export function saveReviewerId(reviewerId: string): void {
  localStorage.setItem(REVIEWER_KEY, reviewerId);
}
