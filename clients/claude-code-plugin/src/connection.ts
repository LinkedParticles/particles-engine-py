/*
 * Where the pane may connect, and whether it may call without a key.
 * The pane talks to one engine only: the configured one.
 */

export interface Connection {
  /** The engine's base URL, without a trailing slash. */
  base: string;
  /** The bearer, empty for none. */
  token: string;
  /** True for an engine on this machine. */
  loopback: boolean;
}

/** Why the pane makes no call at all; null when it may. */
export type Refusal = { reason: string } | null;

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);

export function connectionOf(engineUrl: string | undefined, token: string | undefined): Connection | Refusal {
  const raw = (engineUrl ?? "").trim() || "http://127.0.0.1:8000";
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return { reason: `The engine URL "${raw}" is not a URL. Set it in the plugin's options.` };
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { reason: `The engine URL must be http or https, not ${url.protocol}` };
  }
  const loopback = LOOPBACK_HOSTS.has(url.hostname);
  if (!loopback && url.protocol !== "https:") {
    return {
      reason: `An engine that is not on this machine must use https; ${url.host} is configured with http.`,
    };
  }
  return { base: url.origin, token: (token ?? "").trim(), loopback };
}

export function isConnection(c: Connection | Refusal): c is Connection {
  return c !== null && "base" in c;
}

/** The command that starts an engine at this address. */
export function startCommand(base: string): string {
  const url = new URL(base);
  return `particles engine serve ${url.hostname}:${url.port || (url.protocol === "https:" ? "443" : "80")}`;
}

/**
 * The engine's address for a link the operator opens. The desktop app makes
 * a link only of https or `http://localhost`, so a loopback engine is named
 * `localhost` there; the pane's own calls keep the configured address.
 */
export function linkBase(c: Connection): string {
  if (!c.loopback) return c.base;
  const url = new URL(c.base);
  url.hostname = "localhost";
  return url.origin;
}
