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
