// Inference: blends n-gram context lookup, naive-Bayes text classifier, and a
// rule-based heuristic prior into one distribution over tools; then fills in
// arguments with the learned argument mappers.

import { extractFeatures, extractPaths, isSafeToSpeculate } from "./features.js";
import { MAX_ORDER, Model, Prediction, TraceStep } from "./types.js";

type Dist = Record<string, number>;

const DEFAULT_TOOLS = [
  "view_file", "grep_search", "list_dir", "replace_file_content", "run_command", "web_search",
];

/** Map an arbitrary tool name to a coarse role so heuristics work for any toolset. */
export function roleOf(tool: string): "read" | "search" | "edit" | "run" | "web" | "other" {
  if (/(web|fetch|browse|http)/i.test(tool)) return "web";
  if (/(grep|search|find|glob|list|ls|tree)/i.test(tool)) return "search";
  if (/(edit|write|replace|patch|apply|create|update)/i.test(tool)) return "edit";
  if (/(run|bash|shell|exec|command|test|build|git)/i.test(tool)) return "run";
  if (/(read|view|cat|open|get_file)/i.test(tool)) return "read";
  return "other";
}

type Role = ReturnType<typeof roleOf>;

/** Hand-written rules: signals in the last output -> likely role of the next tool. */
function heuristicRoles(features: Set<string>, firstStep: boolean): Record<Role, number> {
  const r: Record<Role, number> = { read: 0.1, search: 0.1, edit: 0.1, run: 0.1, web: 0.02, other: 0.02 };
  const add = (role: Role, w: number) => (r[role] += w);
  if (firstStep) {
    add("search", 0.5); add("read", 0.3);
    if (features.has("goal:research")) add("web", 0.6);
  }
  if (features.has("sig:stacktrace") || features.has("sig:test_fail")) { add("read", 0.9); add("search", 0.3); }
  if (features.has("sig:type_error")) { add("read", 0.8); add("edit", 0.2); }
  if (features.has("sig:module_missing")) { add("run", 0.8); }
  if (features.has("sig:not_found")) { add("search", 0.8); }
  if (features.has("sig:permission")) { add("run", 0.4); }
  if (features.has("sig:search_hits")) { add("read", 0.9); }
  if (features.has("sig:file_list")) { add("read", 0.7); add("search", 0.2); }
  if (features.has("sig:file_content")) { add("edit", 0.6); add("search", 0.3); }
  if (features.has("sig:edit_ok")) { add("run", 0.9); add("read", 0.1); }
  if (features.has("sig:test_pass") || features.has("sig:build_ok")) { add("run", 0.5); add("edit", 0.2); }
  if (features.has("sig:git_status")) { add("run", 0.6); add("read", 0.3); }
  if (features.has("sig:git_diff")) { add("run", 0.5); add("edit", 0.2); }
  if (features.has("sig:merge_conflict")) { add("read", 0.8); }
  if (features.has("sig:http")) { add("web", 0.3); }
  if (features.has("sig:empty")) { add("search", 0.5); }
  return r;
}

function normalize(d: Dist): Dist {
  const s = Object.values(d).reduce((a, b) => a + b, 0);
  if (s <= 0) return d;
  const out: Dist = {};
  for (const [k, v] of Object.entries(d)) out[k] = v / s;
  return out;
}

function heuristicDist(tools: string[], features: Set<string>, firstStep: boolean, prior: Dist): Dist {
  const roles = heuristicRoles(features, firstStep);
  const byRole = new Map<Role, string[]>();
  for (const t of tools) {
    const role = roleOf(t);
    byRole.set(role, [...(byRole.get(role) ?? []), t]);
  }
  const d: Dist = {};
  for (const [role, ts] of byRole) {
    const mass = roles[role];
    const total = ts.reduce((a, t) => a + (prior[t] ?? 1), 0);
    for (const t of ts) d[t] = (mass * (prior[t] ?? 1)) / total;
  }
  return normalize(d);
}

function ngramDist(model: Model, tools: string[], names: string[]): { dist: Dist; strength: number } {
  const acc: Dist = {};
  let wsum = 0;
  let strength = 0;
  for (let k = Math.min(MAX_ORDER, names.length); k >= 1; k--) {
    const bucket = model.ngrams[k]?.[names.slice(-k).join(">")];
    if (!bucket) continue;
    const n = Object.values(bucket).reduce((a, b) => a + b, 0);
    const w = k * k * (n / (n + 2)); // longer, better-supported contexts count more
    for (const t of tools) acc[t] = (acc[t] ?? 0) + (w * ((bucket[t] ?? 0) + 0.01)) / (n + 0.01 * tools.length);
    wsum += w;
    strength = Math.max(strength, n / (n + 2));
  }
  if (names.length === 0) {
    const bucket = model.ngrams[1]?.["START"];
    if (bucket) {
      const n = Object.values(bucket).reduce((a, b) => a + b, 0);
      for (const t of tools) acc[t] = ((bucket[t] ?? 0) + 0.01) / (n + 0.01 * tools.length);
      wsum = 1;
      strength = n / (n + 2);
    }
  }
  if (wsum === 0) return { dist: {}, strength: 0 };
  return { dist: normalize(acc), strength };
}

function nbDist(model: Model, tools: string[], feats: string[]): Dist {
  const uniq = [...new Set(feats)];
  const totalDocs = Object.values(model.nb.toolCounts).reduce((a, b) => a + b, 0) || 1;
  const temper = 1 / Math.max(1, Math.sqrt(uniq.length / 4)); // NB is overconfident; soften
  const ll: Dist = {};
  for (const t of tools) {
    const docs = model.nb.toolCounts[t] ?? 0;
    let s = Math.log((docs + 1) / (totalDocs + tools.length));
    const fc = model.nb.featureCounts[t] ?? {};
    for (const f of uniq) {
      s += temper * Math.log(((fc[f] ?? 0) + 0.5) / (docs + 1));
    }
    ll[t] = s;
  }
  const mx = Math.max(...Object.values(ll));
  const out: Dist = {};
  for (const t of tools) out[t] = Math.exp(ll[t] - mx);
  return normalize(out);
}

/** Full distribution over candidate tools. `available` restricts the candidate set. */
export function predictDistribution(
  model: Model | null,
  history: TraceStep[],
  goal?: string,
  lastMessage?: string,
  available?: string[],
): Dist {
  const last = lastMessage ?? (history.length ? history[history.length - 1].result ?? "" : goal ?? "");
  const feats = extractFeatures(last, goal);
  const fset = new Set(feats);
  const names = history.map((s) => s.tool);

  let tools = available?.length ? available : model?.tools.length ? model.tools : DEFAULT_TOOLS;
  tools = [...new Set(tools)];

  const prior: Dist = {};
  if (model) for (const t of tools) prior[t] = (model.nb.toolCounts[t] ?? 0) + 1;

  const h = heuristicDist(tools, fset, names.length === 0, prior);
  if (!model || !model.trainedOn.transitions) return h;

  const ng = ngramDist(model, tools, names);
  const nbd = nbDist(model, tools, feats);
  let [wg, wn, wh] = model.weights;
  // If sequence context was never seen, shift its weight to the other signals.
  wg *= ng.strength;
  const z = wg + wn + wh;
  if (z <= 0) return normalize(Object.fromEntries(tools.map((t) => [t, prior[t] ?? 1])));
  const out: Dist = {};
  for (const t of tools) {
    out[t] = (wg * (ng.dist[t] ?? 0) + wn * (nbd[t] ?? 0) + wh * (h[t] ?? 0)) / z;
  }
  return normalize(out);
}

function resolveSource(
  src: Model["argMappers"][string][number]["source"],
  history: TraceStep[],
  goal?: string,
): unknown {
  switch (src.kind) {
    case "copyArg": {
      const s = history[history.length - src.fromBack];
      return s?.args?.[src.arg];
    }
    case "resultPath": {
      const s = history[history.length - src.fromBack];
      return extractPaths(s?.result ?? "")[src.index];
    }
    case "resultRegex": {
      const s = history[history.length - src.fromBack];
      return new RegExp(src.pattern).exec(s?.result ?? "")?.[src.group];
    }
    case "goalPath":
      return goal ? extractPaths(goal)[src.index] : undefined;
  }
}

export function suggestArguments(
  model: Model | null,
  tool: string,
  history: TraceStep[],
  goal?: string,
): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  const prev = history[history.length - 1];
  if (model && prev) {
    for (const m of model.argMappers[`${prev.tool}>${tool}`] ?? []) {
      const v = resolveSource(m.source, history, goal);
      if (v !== undefined) args[m.argName] = v;
    }
  }
  // Heuristic fallback when nothing was learned: first path mentioned in last output.
  if (Object.keys(args).length === 0 && prev && roleOf(tool) === "read") {
    const p = extractPaths(prev.result ?? "")[0];
    if (p) args.path = p;
  }
  return args;
}

const ROLE_RATIONALE: Record<Role, string> = {
  read: "Output points at specific files/locations worth inspecting.",
  search: "Need to locate relevant code or files before acting.",
  edit: "Context suggests enough information to modify code.",
  run: "Likely to execute a command (tests, build, install, git) next.",
  web: "Information likely needs to come from outside the repo.",
  other: "Statistically common follow-up in observed traces.",
};

export function predict(
  model: Model | null,
  input: { last_message: string; goal?: string; history?: TraceStep[]; available_tools?: string[]; top_k: number },
): Prediction[] {
  const history = input.history ?? [];
  const dist = predictDistribution(model, history, input.goal, input.last_message, input.available_tools);
  const feats = new Set(extractFeatures(input.last_message, input.goal));
  const triggers = [...feats].filter((f) => f.startsWith("sig:") && f !== "sig:no_paths").slice(0, 3);
  return Object.entries(dist)
    .sort((a, b) => b[1] - a[1])
    .slice(0, input.top_k)
    .map(([tool, p]) => ({
      toolName: tool,
      confidence: Math.round(p * 1000) / 1000,
      rationale:
        ROLE_RATIONALE[roleOf(tool)] + (triggers.length ? ` Signals: ${triggers.join(", ")}.` : ""),
      suggestedArguments: suggestArguments(model, tool, history, input.goal),
      safeToSpeculate: isSafeToSpeculate(tool),
    }));
}
