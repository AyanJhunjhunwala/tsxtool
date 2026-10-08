// Bootstraps training data with a seeded generator of plausible agent workflows,
// so the classifier is useful before real traces exist. Real traces should be
// preferred: train with `npm run train -- <dir-of-traces>`.

import { Trace, TraceStep } from "./types.js";

function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

const FILES = ["src/auth.ts", "src/utils/parse.ts", "lib/db.py", "src/components/App.tsx", "pkg/server/main.go", "tests/api.test.ts"];

export function generateSynthetic(n: number, seed = 42): Trace[] {
  const r = rng(seed);
  const pick = <T,>(a: T[]) => a[Math.floor(r() * a.length)];
  const traces: Trace[] = [];
  for (let i = 0; i < n; i++) {
    const f = pick(FILES);
    const steps: TraceStep[] = [];
    const kind = r();
    if (kind < 0.45) {
      // bug fix loop
      const goal = `Fix the failing test in ${f}`;
      steps.push({ tool: "run_command", args: { command: "npm test" }, result: `FAIL ${f}\n  1 failed, 3 passed\nTypeError: Cannot read properties of undefined\n    at parse (${f}:12:5)` });
      if (r() < 0.5) steps.push({ tool: "grep_search", args: { pattern: "parse" }, result: `${f}:12: function parse(x)\n${f}:40: parse(y)` });
      steps.push({ tool: "view_file", args: { path: f }, result: "import x from 'y'\nexport function parse(a) { return a.b }" });
      steps.push({ tool: "replace_file_content", args: { path: f }, result: "File updated successfully" });
      steps.push({ tool: "run_command", args: { command: "npm test" }, result: r() < 0.8 ? "4 passed, all tests passed" : `FAIL ${f}\n1 failed` });
      if (r() < 0.3) steps.push({ tool: "run_command", args: { command: "git status" }, result: "On branch main\nChanges not staged for commit" });
      traces.push({ goal, steps });
    } else if (kind < 0.75) {
      // exploration
      const goal = "Explain how authentication works in this repo";
      steps.push({ tool: "list_dir", args: { path: "." }, result: "src\nREADME.md\npackage.json\n" + f });
      steps.push({ tool: "grep_search", args: { pattern: "auth" }, result: `${f}:3: export function auth()\n${pick(FILES)}:9: auth(` });
      steps.push({ tool: "view_file", args: { path: f }, result: "export function auth() { const x = 1 }\nimport y from 'z'" });
      if (r() < 0.5) steps.push({ tool: "view_file", args: { path: pick(FILES) }, result: "function helper() {}" });
      traces.push({ goal, steps });
    } else if (kind < 0.9) {
      // feature add
      const goal = `Add a retry option to ${f}`;
      steps.push({ tool: "view_file", args: { path: f }, result: "export function run() {}\nimport a from 'b'" });
      steps.push({ tool: "replace_file_content", args: { path: f }, result: "File updated successfully" });
      steps.push({ tool: "run_command", args: { command: "npx tsc --noEmit" }, result: r() < 0.4 ? `${f}(4,2): error TS2322: Type 'string' is not assignable` : "" });
      if (steps[2].result) {
        steps.push({ tool: "view_file", args: { path: f }, result: "export function run() { retry }" });
        steps.push({ tool: "replace_file_content", args: { path: f }, result: "File updated successfully" });
      }
      traces.push({ goal, steps });
    } else {
      // dependency / research
      const goal = "Research the docs for the zod library and install it";
      steps.push({ tool: "web_search", args: { query: "zod docs" }, result: "https://zod.dev - Zod documentation" });
      steps.push({ tool: "run_command", args: { command: "npm install zod" }, result: "added 1 packages" });
      steps.push({ tool: "run_command", args: { command: "npm test" }, result: "Error: Cannot find module 'zod'" });
      traces.push({ goal, steps });
    }
  }
  return traces;
}
