/*
 * How an engine answer becomes a failure every client reports the same way.
 * The wording is the web UI's, moved here unchanged.
 */

/**
 * - `unauthorized` (401): the engine rejected the bearer token.
 * - `forbidden` (403): belief writes are disabled on the store, or the key's
 *   role cannot call the route. The message is the engine's detail; a client
 *   turns read-only.
 * - `unavailable` (503): a keyless engine refusing a write, or one refusing a
 *   non-loopback caller.
 * - `not-configured`: no token or reviewer set; the client makes no call.
 */
export type ApiErrorKind =
  | "unauthorized"
  | "forbidden"
  | "unavailable"
  | "not-configured"
  | "not-found"
  | "network"
  | "http";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

/** What the client knows about its own connection, for the messages. */
export interface ConnectionFacts {
  /** True when a bearer token is held. */
  hasToken: boolean;
  /** The engine address as configured; empty for same-origin. */
  baseUrl: string;
}

/**
 * The engine's `detail`, in both shapes it sends: a plain string, and
 * FastAPI's request-validation list of `{loc, msg}` objects.
 */
export function detailOf(body: unknown, fallback: string): string {
  const d = body && typeof body === "object" ? (body as { detail?: unknown }).detail : undefined;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.length) {
    return d
      .map((e) => {
        if (e && typeof e === "object") {
          const err = e as { loc?: unknown; msg?: unknown };
          const loc = Array.isArray(err.loc)
            ? err.loc.filter((x) => x !== "query" && x !== "body").join(".")
            : "";
          const msg = typeof err.msg === "string" ? err.msg : JSON.stringify(e);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return String(e);
      })
      .join(" · ");
  }
  return fallback;
}

/** The failure for a non-2xx answer, or null for a success. */
export function errorForStatus(
  status: number,
  body: unknown,
  facts: ConnectionFacts,
): ApiError | null {
  if (status >= 200 && status < 300) return null;
  if (status === 401) {
    return new ApiError(
      "unauthorized",
      facts.hasToken
        ? "The engine rejected the bearer token. Check the token in settings."
        : "This engine requires a bearer token and none is set — it may have " +
            "been restarted with a PARTICLES_API_KEY. Add it in settings.",
      401,
    );
  }
  if (status === 403) {
    return new ApiError(
      "forbidden",
      detailOf(body, "Belief writes are disabled on this engine (read-only)."),
      403,
    );
  }
  if (status === 503) {
    return new ApiError(
      "unavailable",
      detailOf(body, "The engine refuses unauthenticated non-loopback access."),
      503,
    );
  }
  if (status === 404) return new ApiError("not-found", detailOf(body, "Not found."), 404);
  return new ApiError("http", detailOf(body, `Engine returned HTTP ${status}.`), status);
}

/** The failure for an engine that could not be reached at all. */
export function networkError(facts: ConnectionFacts, cause: unknown): ApiError {
  return new ApiError(
    "network",
    `Could not reach the Particles engine${facts.baseUrl ? ` at ${facts.baseUrl}` : ""}: ${String(cause)}`,
  );
}

/** One line for any failure, for a client's error notice. */
export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  return `Unexpected error: ${String(e)}`;
}
