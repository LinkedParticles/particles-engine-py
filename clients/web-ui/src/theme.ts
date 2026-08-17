/*
 * Appearance (theme) — system / light / dark.
 *
 * The design-system tokens (design/tokens.css, served as tokens.css beside the
 * bundle) resolve light on a bare <html>, dark under prefers-color-scheme, and
 * follow an explicit `<html data-theme="light|dark">` stamp over both. This
 * module owns that stamp: it persists the operator's choice in localStorage
 * (client-side only, like every other setting), applies it, and
 * exposes the resolved theme's token values for the one renderer that cannot
 * read CSS variables itself (Cytoscape in graphRender.ts).
 *
 * index.html re-applies the saved stamp inline before first paint so a chosen
 * theme never flashes the other one; the storage key and attribute there must
 * match the ones here.
 */

export type ThemeChoice = "system" | "light" | "dark";

const STORAGE_KEY = "particles.ui.theme";

export function loadTheme(): ThemeChoice {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function saveTheme(choice: ThemeChoice): void {
  try {
    if (choice === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    /* storage unavailable (private mode) — the choice still applies for this page */
  }
}

/** Stamp (or clear) the explicit theme on <html>; tokens.css does the rest. */
export function applyTheme(choice: ThemeChoice = loadTheme()): void {
  const root = document.documentElement;
  if (choice === "system") delete root.dataset.theme;
  else root.dataset.theme = choice;
}

/**
 * Read a design token's resolved value off <html> — e.g. `tokenValue("--p-text")`
 * → "#e6edf3" in dark. For renderers that need concrete colour strings.
 */
export function tokenValue(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** A percentage token ("22%") as a number (22). */
export function tokenPercent(name: string): number {
  const n = parseFloat(tokenValue(name));
  return Number.isFinite(n) ? n : 0;
}
