/*
 * A belief's status, shown the same way on every surface. The wording lives
 * in the shared curation core; this module re-exports it for the
 * graph view.
 */
export { isQuarantined, reasonNote, statusText } from "../../curation-core/src/status.ts";
