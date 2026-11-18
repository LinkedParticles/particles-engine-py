/*
 * fixtures/contract.json is what the core produces today: a change to the
 * core's views, forms or requests must regenerate it (`npm run fixtures`).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { buildContract } from "./contract.mts";
import { FIXTURES } from "./harness.mts";

test("contract.json matches the core", async () => {
  const committed = JSON.parse(readFileSync(join(FIXTURES, "contract.json"), "utf8"));
  const built = JSON.parse(JSON.stringify(await buildContract()));
  assert.deepEqual(built, committed, "run `npm run fixtures` and commit fixtures/contract.json");
});
