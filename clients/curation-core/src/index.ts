/*
 * The shared curation core: everything about curation except
 * drawing. Every client (the web UI, the Claude Code pane, a future mobile
 * app) renders what this returns and checks itself against its fixtures.
 */
export * from "./schema.ts";
export * from "./errors.ts";
export * from "./client.ts";
export * from "./gestures.ts";
export * from "./status.ts";
export * from "./view.ts";
export * from "./queue.ts";
export * from "./actions.ts";

/** This core's version, shown by clients beside the engine's. */
export const CORE_VERSION = "0.1.0";
