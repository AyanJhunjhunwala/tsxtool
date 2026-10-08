import assert from "node:assert/strict";
import { test } from "node:test";
import { predict } from "../src/predictor.js";
import { generateSynthetic } from "../src/synthetic.js";
import { crossValidate, trainModel } from "../src/trainer.js";
import { extractPaths } from "../src/features.js";

test("heuristics alone (no model) route errors to reading code", () => {
  const p = predict(null, { last_message: "TypeError: x\n    at f (src/a.ts:1:2)", top_k: 1 });
  assert.equal(p[0].toolName, "view_file");
  assert.equal(p[0].safeToSpeculate, true);
});

test("confidences form a distribution", () => {
  const p = predict(null, { last_message: "hello", top_k: 10 });
  const sum = p.reduce((a, b) => a + b.confidence, 0);
  assert.ok(Math.abs(sum - 1) < 0.01);
});

test("trained model beats majority baseline on held-out traces", () => {
  const r = crossValidate(generateSynthetic(200, 7));
  assert.ok(r.top1 > r.baselineTop1 + 0.2, JSON.stringify(r));
  assert.ok(r.top3 > 0.95);
});

test("learns argument mappers (path flows from grep hit to view_file)", () => {
  const model = trainModel(generateSynthetic(200, 3));
  const out = predict(model, {
    last_message: "src/auth.ts:12: function parse(x)",
    history: [{ tool: "grep_search", args: { pattern: "parse" }, result: "src/auth.ts:12: function parse(x)" }],
    top_k: 3,
  });
  const view = out.find((x) => x.toolName === "view_file");
  assert.equal(view?.suggestedArguments.path, "src/auth.ts");
});

test("never speculates on mutating tools", () => {
  const model = trainModel(generateSynthetic(100, 1));
  const out = predict(model, { last_message: "ok", history: [{ tool: "view_file" }], top_k: 6 });
  for (const x of out) if (/replace|run_command/.test(x.toolName)) assert.equal(x.safeToSpeculate, false);
});

test("available_tools restricts candidates", () => {
  const out = predict(null, { last_message: "x", available_tools: ["Read", "Bash"], top_k: 5 });
  assert.deepEqual(out.map((x) => x.toolName).sort(), ["Bash", "Read"]);
});

test("extractPaths ignores versions and urls", () => {
  assert.deepEqual(extractPaths("see https://a.com/x.js v1.2.3 src/a.ts"), ["src/a.ts"]);
});
