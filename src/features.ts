// Heuristic signal extraction: turns raw tool output text into named features.
// These feed both the naive-Bayes classifier and the rule-based prior.

const PATH_RE = /(?:[\w.-]+\/)+[\w.-]+\.\w{1,8}|\b[\w-]+\.(?:tsx?|jsx?|py|go|rs|java|json|md|ya?ml|toml|css|html|c|cpp|h|rb|sh)\b/g;

export function extractPaths(text: string): string[] {
  const seen = new Set<string>();
  for (const m of text.replace(/https?:\/\/\S+/g, " ").matchAll(PATH_RE)) {
    const p = m[0].replace(/^\.\//, "");
    if (!/^\d+\.\d+/.test(p) && !/^https?:/.test(p)) seen.add(p);
  }
  return [...seen];
}

const URL_RE = /https?:\/\/[^\s)\]"'<>`]+/g;

export function extractUrls(text: string): string[] {
  return [...new Set([...text.matchAll(URL_RE)].map((m) => m[0].replace(/[.,;:]+$/, "")))];
}

const SIGNALS: Array<[string, RegExp]> = [
  ["sig:stacktrace", /\n\s+at .+\(.+:\d+:\d+\)|Traceback \(most recent call last\)|panicked at|^\s+File ".+", line \d+/m],
  ["sig:error", /\b(error|exception|failed|failure|fatal|TypeError|ReferenceError|SyntaxError)\b/i],
  ["sig:test_fail", /\b(\d+ (failed|failing)|FAIL\b|assertion|expected .* (to|but)|✗|✕)/i],
  ["sig:test_pass", /\b(\d+ (passed|passing)|all tests passed|PASS\b|✓|✔)/i],
  ["sig:type_error", /\bTS\d{4}\b|type '.+' is not assignable|cannot find (name|module)/i],
  ["sig:lint", /\b(eslint|prettier|lint|warning)\b/i],
  ["sig:not_found", /no such file|not found|ENOENT|cannot find|does not exist|404/i],
  ["sig:permission", /permission denied|EACCES|unauthorized|403/i],
  ["sig:module_missing", /cannot find module|module not found|ModuleNotFoundError|No module named|command not found/i],
  ["sig:merge_conflict", /<<<<<<<|CONFLICT \(/],
  ["sig:git_status", /On branch|Changes not staged|nothing to commit|Untracked files/],
  ["sig:git_diff", /^diff --git|^@@ .+ @@/m],
  ["sig:search_hits", /^[\w./-]+:\d+[:-]/m],
  ["sig:file_list", /^(?:[\w.-]+\/)*[\w.-]+\.\w+$/m],
  ["sig:file_content", /\b(import |export |function |class |def |const |#include)/],
  ["sig:edit_ok", /(file|content)s? (has been )?(updated|written|created|modified)|successfully (edited|replaced|wrote)/i],
  ["sig:build_ok", /build (succeeded|successful)|compiled successfully|done in \d/i],
  ["sig:install", /\b(npm|yarn|pnpm|pip) (install|i)\b|added \d+ packages/i],
  ["sig:http", /\bhttps?:\/\/\S+/],
  ["sig:empty", /^\s*$/],
  ["sig:long", /[\s\S]{3000,}/],
];

const GOAL_VERBS: Array<[string, RegExp]> = [
  ["goal:fix", /\b(fix|debug|repair|resolve|failing)\b/i],
  ["goal:add", /\b(add|implement|create|build|write)\b/i],
  ["goal:refactor", /\b(refactor|rename|clean|simplify|migrate)\b/i],
  ["goal:explain", /\b(explain|understand|how does|what is|explore|find|where)\b/i],
  ["goal:test", /\b(test|coverage|spec)\b/i],
  ["goal:research", /\b(research|search the web|look up|docs?|documentation)\b/i],
  ["goal:git", /\b(commit|push|pull request|branch|merge|rebase)\b/i],
];

const STOP = new Set("the a an and or of to in is it for on with this that at by be as are was".split(" "));

export function extractFeatures(lastMessage: string, goal?: string): string[] {
  const f: string[] = [];
  const text = lastMessage.slice(0, 20000);
  for (const [name, re] of SIGNALS) if (re.test(text)) f.push(name);
  if (goal) for (const [name, re] of GOAL_VERBS) if (re.test(goal)) f.push(name);
  const paths = extractPaths(text);
  if (paths.length === 0) f.push("sig:no_paths");
  else if (paths.length === 1) f.push("sig:one_path");
  else f.push("sig:many_paths");
  // Bag of words (first 400 tokens) captures tool-specific vocabulary.
  const words = text.toLowerCase().match(/[a-z_]{3,}/g) ?? [];
  for (const w of words.slice(0, 400)) if (!STOP.has(w)) f.push("w:" + w);
  return f;
}

/** Tools whose execution has side effects: never speculatively run these. */
const SIDE_EFFECT = /(write|edit|replace|delete|remove|rm|create|commit|push|install|deploy|send|post|update|move|rename|apply|patch|bash|run_command|exec|shell)/i;
const READ_ONLY = /(read|view|cat|grep|search|find|glob|list|ls|get|fetch|status|diff|log|show|inspect)/i;

export function isSafeToSpeculate(tool: string): boolean {
  if (READ_ONLY.test(tool) && !SIDE_EFFECT.test(tool)) return true;
  return false;
}

/** Coarse intent of a shell command, so "Bash" is not one opaque action. */
export function commandClass(cmd: string): string {
  const c = cmd.trim().replace(/^(cd [^;&|]+(&&|;)\s*)+/, "");
  if (/^git\b/.test(c)) return /\b(commit|push|add|checkout|merge|rebase|stash)\b/.test(c) ? "gitwrite" : "gitread";
  if (/\b(pytest|jest|vitest|mocha|go test|cargo test|npm (run )?test|yarn test|pnpm test|bun test|unittest)\b/.test(c)) return "test";
  if (/\b(tsc|build|make|cargo build|go build|compile|webpack|vite build|next build)\b/.test(c)) return "build";
  if (/\b(npm|yarn|pnpm|pip3?|uv|brew|apt|cargo) (install|add|i)\b/.test(c)) return "install";
  if (/\b(lint|eslint|prettier|ruff|mypy|black)\b/.test(c)) return "lint";
  if (/^(grep|rg|find|ag)\b/.test(c)) return "search";
  if (/^(cat|head|tail|ls|sed -n|wc|tree|pwd|stat|file)\b/.test(c)) return "peek";
  if (/^(rm|mv|cp|mkdir|touch|chmod)\b/.test(c)) return "fs";
  if (/^(curl|wget|gh)\b/.test(c)) return "net";
  if (/^(node|python3?|tsx|ts-node|npx|bun|deno|ruby|go run|cargo run|\.\/)/.test(c)) return "exec";
  return "other";
}

const CMD_KEYS = ["command", "cmd", "input"];

/** Tool name refined by what it did, e.g. Bash:test, Bash:gitread. */
export function labelOf(step: { tool: string; args?: Record<string, unknown> }): string {
  const a = step.args ?? {};
  for (const k of CMD_KEYS) {
    const v = a[k];
    const cmd = Array.isArray(v) ? v.join(" ") : v;
    if (typeof cmd === "string" && /(bash|shell|exec|run|command)/i.test(step.tool)) return `${step.tool}:${commandClass(cmd)}`;
  }
  return step.tool;
}

function bucket(n: number): string {
  return n <= 2 ? String(n) : n <= 4 ? "3-4" : n <= 8 ? "5-8" : n <= 16 ? "9-16" : n <= 32 ? "17-32" : "33+";
}

/** Sequence/state features: what happened recently, not just what the last output said. */
export function contextFeatures(history: Array<{ tool: string; args?: Record<string, unknown>; result?: string }>): string[] {
  const f: string[] = [`n:${bucket(history.length)}`];
  const labs = history.map(labelOf);
  const last = history.length - 1;
  for (let k = 1; k <= 3 && k <= history.length; k++) {
    f.push(`p${k}:${history[last - k + 1].tool}`, `l${k}:${labs[last - k + 1]}`);
  }
  if (history.length >= 2) {
    f.push(`p12:${history[last - 1].tool}>${history[last].tool}`, `l12:${labs[last - 1]}>${labs[last]}`);
  }
  if (history.length >= 3) f.push(`l123:${labs[last - 2]}>${labs[last - 1]}>${labs[last]}`);
  let rep = 0;
  while (rep < 4 && rep < history.length && history[last - rep].tool === history[last].tool) rep++;
  if (rep) f.push(`rep:${rep}`);
  if (labs.some((l) => /(edit|write|patch)/i.test(l))) f.push("seen:edit");
  if (labs.some((l) => l.endsWith(":test"))) f.push("seen:test");
  if (history.length) {
    const prev = history[last];
    const res = prev.result ?? "";
    f.push(`rl:${bucket(Math.round(Math.log2(res.length + 1)))}`);
    const path = prev.args && (prev.args.file_path ?? prev.args.path);
    if (typeof path === "string") {
      f.push(`ext:${path.split(".").pop()}`);
      if (/test|spec/i.test(path)) f.push("path:test");
    }
  }
  return f;
}
