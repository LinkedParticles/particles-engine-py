/*
 * A scripted operator and engine for the core's fixtures: the same fixed
 * answers and engine replies the web UI's golden capture used
 * (clients/web-ui/test/golden/capture.mjs), so the two can be compared.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type Answer,
  type CurationCard,
  CurationClient,
  type Form,
  type WireRequest,
  type WireResponse,
  perform,
} from "../src/index.ts";

export const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

export function loadCards(): (CurationCard & { fixture: string })[] {
  return JSON.parse(readFileSync(join(FIXTURES, "cards.json"), "utf8")).cards;
}

const ANSWERS: Record<string, string> = {
  subject: "Example subject",
  subjects: "Alpha, Beta",
  content: "Scripted revised claim",
  entry_id: "entry-1",
  text: "Scripted source text",
  note: "scripted note",
};

/** The fixed operator: confirm with the first choice and the scripted values. */
export function scriptedAnswer(form: Form): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of form.fields ?? []) {
    if (f.type === "choice") out[f.name] = f.options?.[0]?.value ?? "";
    // A checkbox list keeps what it checked at first, else checks the first option.
    else if (f.type === "choices") out[f.name] = f.value ?? f.options?.[0]?.value ?? "";
    else out[f.name] = ANSWERS[f.name] ?? f.value ?? `scripted ${f.name}`;
  }
  return out;
}

export type Respond = (req: WireRequest) => WireResponse | undefined;

/** Engine replies that let each multi-step flow run to its end. */
export function engineFor(card: CurationCard): Respond {
  return (req) => {
    const body = req.body as { dry_run?: boolean } | undefined;
    if (/^\/corpus\/[^/]+\/retract$/.test(req.path) && body?.dry_run) {
      return { status: 200, body: { retracted_ids: ["r-1", "r-2"] } };
    }
    if (req.path === "/subjects/relink-gated") return { status: 200, body: { recoverable: 3 } };
    if (req.method === "GET" && req.path.startsWith("/curation/cards/")) {
      const full = structuredClone(card);
      full.particles = (card.particle_ids ?? []).map((id, i) => ({
        particle_id: id,
        content: `Full brief ${i}`,
        effective_confidence: 0.5,
        status: "ACTIVE",
        subject_labels: [],
      }));
      return { status: 200, body: full };
    }
    return undefined;
  };
}

export interface Trace {
  forms: Form[];
  requests: WireRequest[];
  resolved?: boolean;
  notice?: string;
  error?: unknown;
}

/** A client whose wire is a recording of every request. */
export function recordingClient(trace: Trace, respond: Respond): CurationClient {
  return new CurationClient({
    transport: async (req) => {
      trace.requests.push(req.body === undefined ? { method: req.method, path: req.path } : req);
      return respond(req) ?? { status: 200, body: {} };
    },
    facts: () => ({ hasToken: true, baseUrl: "http://engine.test" }),
    depositedBy: "web-ui",
  });
}

/** Run one gesture with the scripted operator; "cancel" refuses the first form. */
export async function run(
  gesture: string,
  card: CurationCard,
  respond: Respond,
  mode: "answer" | "cancel" = "answer",
): Promise<Trace> {
  const trace: Trace = { forms: [], requests: [] };
  const ask = async (form: Form): Promise<Answer> => {
    trace.forms.push(structuredClone(form));
    return mode === "cancel" ? null : scriptedAnswer(form);
  };
  try {
    const result = await perform(gesture, card, {
      client: recordingClient(trace, respond),
      ask,
      reviewerId: "reviewer-1",
    });
    trace.resolved = result.resolved;
    if (result.notice) trace.notice = result.notice;
  } catch (e) {
    trace.error = e;
  }
  return trace;
}
