// Offline "pattern analyzer": mines tool-call traces into a Model.
// Mirrors PASTE's split: control flow (n-gram context -> next tool) is learned
// separately from data flow (argument mappers), plus a text classifier over the
// previous tool's output to disambiguate when sequence context is ambiguous.

import { extractFeatures, extractPaths } from "./features.js";
import { blend, computeComponents, lrFeatures, predictDistribution } from "./predictor.js";
import { featureIndices, trainLR } from "./lr.js";
import { ArgMapper, ArgSource, MAX_ORDER, Model, Trace } from "./types.js";

export interface Example {
  history: Trace["steps"]; // steps before the target
  goal?: string;
  target: Trace["steps"][number];
}

export function toExamples(traces: Trace[]): Example[] {
  const out: Example[] = [];
  for (const t of traces) {
    for (let i = 0; i < t.steps.length; i++) {
      out.push({ history: t.steps.slice(0, i), goal: t.goal, target: t.steps[i] });
    }
  }
  return out;
}

export function lastMessageOf(history: Trace["steps"], goal?: string): string {
  return history.length ? history[history.length - 1].result ?? "" : goal ?? "";
}

function bump(m: Record<string, number>, k: string, n = 1) {
  m[k] = (m[k] ?? 0) + n;
}

function candidateSources(ex: Example, argName: string, value: string): ArgSource[] {
  const out: ArgSource[] = [];
  for (let back = 1; back <= 2 && back <= ex.history.length; back++) {
    const prev = ex.history[ex.history.length - back];
    for (const [k, v] of Object.entries(prev.args ?? {})) {
      if (v === value) out.push({ kind: "copyArg", fromBack: back, arg: k });
    }
    const paths = extractPaths(prev.result ?? "");
    paths.slice(0, 5).forEach((p, index) => {
      if (p === value) out.push({ kind: "resultPath", fromBack: back, index });
    });
  }
  if (ex.goal) {
    extractPaths(ex.goal).forEach((p, index) => {
      if (p === value) out.push({ kind: "goalPath", index });
    });
  }
  void argName;
  return out;
}

export function trainModel(traces: Trace[], opts: { tuneWeights?: boolean; lr?: boolean; epochs?: number } = {}): Model {
  const tools = new Set<string>();
  const ngrams: Model["ngrams"] = {};
  for (let k = 1; k <= MAX_ORDER; k++) ngrams[k] = {};
  const nb: Model["nb"] = { toolCounts: {}, featureCounts: {}, featureTotals: {}, vocabSize: 0 };
  const vocab = new Set<string>();
  const mapperHits: Record<string, Record<string, number>> = {}; // pair -> argName|source -> hits
  const pairCounts: Record<string, number> = {};

  const examples = toExamples(traces);
  for (const ex of examples) {
    const tool = ex.target.tool;
    tools.add(tool);
    const names = ex.history.map((s) => s.tool);

    for (let k = 1; k <= MAX_ORDER; k++) {
      const ctx = k <= names.length ? names.slice(-k).join(">") : k === 1 ? "START" : null;
      if (ctx === null) continue;
      const bucket = (ngrams[k][ctx] ??= {});
      bump(bucket, tool);
    }

    const feats = new Set(extractFeatures(lastMessageOf(ex.history, ex.goal), ex.goal));
    bump(nb.toolCounts, tool);
    const fc = (nb.featureCounts[tool] ??= {});
    for (const f of feats) {
      bump(fc, f);
      bump(nb.featureTotals, tool);
      vocab.add(f);
    }

    if (ex.history.length) {
      const pair = `${names[names.length - 1]}>${tool}`;
      bump(pairCounts, pair);
      const hits = (mapperHits[pair] ??= {});
      for (const [argName, v] of Object.entries(ex.target.args ?? {})) {
        if (typeof v !== "string") continue;
        for (const src of candidateSources(ex, argName, v)) {
          bump(hits, `${argName}\u0000${JSON.stringify(src)}`);
        }
      }
    }
  }
  nb.vocabSize = vocab.size;

  const argMappers: Model["argMappers"] = {};
  for (const [pair, hits] of Object.entries(mapperHits)) {
    const best: Record<string, ArgMapper> = {};
    for (const [key, n] of Object.entries(hits)) {
      const [argName, srcJson] = key.split("\u0000");
      const support = n / pairCounts[pair];
      if (support < 0.3 || n < 2) continue;
      if (!best[argName] || best[argName].support < support) {
        best[argName] = { argName, source: JSON.parse(srcJson), support };
      }
    }
    if (Object.keys(best).length) argMappers[pair] = Object.values(best);
  }

  const model: Model = {
    version: 1,
    tools: [...tools].sort(),
    ngrams,
    nb,
    argMappers,
    weights: [0.5, 0.35, 0.15, 0],
    trainedOn: { traces: traces.length, transitions: examples.length },
  };

  if (opts.lr !== false && examples.length >= 50) {
    // Ignore the long tail of one-off tools: the LR needs enough examples per class.
    const lrTools = Object.entries(nb.toolCounts)
      .filter(([, n]) => n >= 20)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 40)
      .map(([t]) => t)
      .sort();
    const ix = new Map(lrTools.map((t, i) => [t, i]));
    const data = examples
      .filter((ex) => ix.has(ex.target.tool))
      .map((ex) => ({
        idx: featureIndices(lrFeatures(ex.history, ex.goal, lastMessageOf(ex.history, ex.goal))),
        y: ix.get(ex.target.tool)!,
      }));
    if (lrTools.length >= 2) model.lr = trainLR(lrTools, data, { epochs: opts.epochs });
    model.weights = [0.3, 0.1, 0.1, 0.5];
  }

  if (opts.tuneWeights !== false && traces.length >= 10) {
    model.weights = tuneWeights(traces, opts);
  }
  return model;
}

/** Grid-search blend weights on a deterministic held-out split by trace (maximize log-lik). */
function tuneWeights(traces: Trace[], opts: { lr?: boolean; epochs?: number }): Model["weights"] {
  const train = traces.filter((_, i) => i % 5 !== 0);
  const held = traces.filter((_, i) => i % 5 === 0);
  const m = trainModel(train, { tuneWeights: false, lr: opts.lr, epochs: opts.epochs });
  const comps = toExamples(held).map((ex) => ({
    ex,
    c: computeComponents(m, ex.history, ex.goal),
  }));
  const hasLR = !!m.lr;
  let best: Model["weights"] = m.weights;
  let bestScore = -Infinity;
  const steps = 6;
  for (let a = 0; a <= steps; a++) {
    for (let b = 0; a + b <= steps; b++) {
      for (let c = 0; a + b + c <= steps; c++) {
        const d = steps - a - b - c;
        if (!hasLR && d > 0) continue;
        const w = [a / steps, b / steps, c / steps, d / steps];
        let ll = 0;
        for (const { ex, c: cc } of comps) {
          ll += Math.log((blend(m, cc, w)[ex.target.tool] ?? 0) + 1e-6);
        }
        if (ll > bestScore) {
          bestScore = ll;
          best = w;
        }
      }
    }
  }
  return best;
}

export interface EvalResult {
  n: number;
  top1: number;
  top3: number;
  mrr: number;
  baselineTop1: number; // always predict most frequent tool
  /** top-1 of each signal alone, to show what the blend is buying. */
  parts?: Record<string, number>;
}

export function evaluate(model: Model, traces: Trace[]): EvalResult {
  const exs = toExamples(traces);
  const freq = model.nb.toolCounts;
  const majority = Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0];
  let t1 = 0, t3 = 0, mrr = 0, base = 0;
  const parts: Record<string, number> = { ngram: 0, nb: 0, heuristic: 0, lr: 0 };
  const top = (d: Record<string, number> | null) =>
    d && Object.keys(d).length ? Object.entries(d).sort((a, b) => b[1] - a[1])[0][0] : undefined;
  for (const ex of exs) {
    const comp = computeComponents(model, ex.history, ex.goal);
    if (top(comp.ng.dist) === ex.target.tool) parts.ngram++;
    if (top(comp.nb) === ex.target.tool) parts.nb++;
    if (top(comp.h) === ex.target.tool) parts.heuristic++;
    if (top(comp.lr) === ex.target.tool) parts.lr++;
    const dist = blend(model, comp);
    const ranked = Object.entries(dist).sort((a, b) => b[1] - a[1]).map(([t]) => t);
    const r = ranked.indexOf(ex.target.tool);
    if (r === 0) t1++;
    if (r >= 0 && r < 3) t3++;
    if (r >= 0) mrr += 1 / (r + 1);
    if (ex.target.tool === majority) base++;
  }
  const n = exs.length || 1;
  for (const k of Object.keys(parts)) parts[k] /= n;
  return { n: exs.length, top1: t1 / n, top3: t3 / n, mrr: mrr / n, baselineTop1: base / n, parts };
}

/** k-fold cross validation by trace so no trace leaks between train and test. */
export function crossValidate(traces: Trace[], folds = 5, opts: { lr?: boolean } = {}): EvalResult {
  const agg = { n: 0, top1: 0, top3: 0, mrr: 0, baselineTop1: 0 };
  const parts: Record<string, number> = {};
  for (let f = 0; f < folds; f++) {
    const train = traces.filter((_, i) => i % folds !== f);
    const test = traces.filter((_, i) => i % folds === f);
    if (!test.length || !train.length) continue;
    const r = evaluate(trainModel(train, { lr: opts.lr }), test);
    agg.n += r.n;
    agg.top1 += r.top1 * r.n;
    agg.top3 += r.top3 * r.n;
    agg.mrr += r.mrr * r.n;
    agg.baselineTop1 += r.baselineTop1 * r.n;
    for (const [k, v] of Object.entries(r.parts ?? {})) parts[k] = (parts[k] ?? 0) + v * r.n;
  }
  const n = agg.n || 1;
  for (const k of Object.keys(parts)) parts[k] /= n;
  return { n: agg.n, top1: agg.top1 / n, top3: agg.top3 / n, mrr: agg.mrr / n, baselineTop1: agg.baselineTop1 / n, parts };
}
