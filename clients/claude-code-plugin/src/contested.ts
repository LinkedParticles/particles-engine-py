/*
 * Reads the contested beliefs out of a Particles `query` result.
 * The result is untrusted text: an id is used only if it is a
 * canonical lowercase UUID, the form the engine mints.
 */

export interface ContestedHit {
  /** The open INCONSISTENCY's id. */
  inconsistencyId: string;
  /** The belief the agent used, when the hit names it. */
  beliefId?: string;
  /** Its claim, when the hit carries it. */
  content?: string;
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export function isUuid(s: unknown): s is string {
  return typeof s === "string" && UUID.test(s);
}

/** Every string a tool result carries: plain text, MCP content blocks, nested fields. */
function texts(x: unknown, out: string[], depth = 0): void {
  if (depth > 6 || x === null || x === undefined) return;
  if (typeof x === "string") {
    out.push(x);
    return;
  }
  if (Array.isArray(x)) {
    for (const item of x) texts(item, out, depth + 1);
    return;
  }
  if (typeof x === "object") {
    const o = x as Record<string, unknown>;
    if (typeof o.text === "string") out.push(o.text);
    for (const key of ["content", "result", "structuredContent"]) {
      if (key in o) texts(o[key], out, depth + 1);
    }
  }
}

/** Every object under a parsed value that carries a `contested` key. */
function walk(x: unknown, out: ContestedHit[], depth = 0): void {
  if (depth > 8 || x === null || typeof x !== "object") return;
  if (Array.isArray(x)) {
    for (const item of x) walk(item, out, depth + 1);
    return;
  }
  const o = x as Record<string, unknown>;
  if (isUuid(o.contested)) {
    const hit: ContestedHit = { inconsistencyId: o.contested };
    if (isUuid(o.id)) hit.beliefId = o.id;
    if (typeof o.content === "string") hit.content = o.content.slice(0, 500);
    out.push(hit);
  }
  for (const value of Object.values(o)) walk(value, out, depth + 1);
}

/** The contested hits in a tool result, each conflict once, in order. */
export function contestedHits(result: unknown): ContestedHit[] {
  const found: ContestedHit[] = [];
  walk(result, found);
  const strings: string[] = [];
  texts(result, strings);
  for (const s of strings) {
    const t = s.trim();
    if (!t.startsWith("{") && !t.startsWith("[")) continue;
    try {
      walk(JSON.parse(t), found);
    } catch {
      /* not JSON: nothing to read */
    }
  }
  const seen = new Set<string>();
  return found.filter((h) => {
    if (seen.has(h.inconsistencyId)) return false;
    seen.add(h.inconsistencyId);
    return true;
  });
}
