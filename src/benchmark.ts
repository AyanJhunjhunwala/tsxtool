// Level 1: offline replay benchmark. Cross-validated by session (trace), comparing
// the full predictor against ablations and baselines, and simulating speculation.

import { predict, predictDistribution } from "./predictor.js";
import { ScoreRecord, THRESHOLDS, pct, rankStats, speculationSweep, specTable } from "./scoring.js";
import { toExamples, trainModel } from "./trainer.js";
import { Model, Trace } from "./types.js";

type Variant = { name: string; weights?: Model["weights"]; majority?: boolean };

const VARIANTS: Variant[] = [
  { name: "majority baseline", majority: true },
  { name: "rules only", weights: [0, 0, 1] },
  { name: "n-gram only", weights: [1, 0, 0] },
  { name: "text classifier only", weights: [0, 1, 0] },
  { name: "full blend" },
];

export interface BenchResult {
  folds: number;
  sessions: number;
  transitions: number;
  variants: Array<{ name: string; n: number; top1: number; top3: number; mrr: number }>;
  speculation: ReturnType<typeof speculationSweep>;
  perTool: Array<{ tool: string; n: number; top1: number }>;
  markdown: string;
}

export function runBenchmark(traces: Trace[], folds = 5): BenchResult {
  const recs: Record<string, ScoreRecord[]> = Object.fromEntries(VARIANTS.map((v) => [v.name, []]));
  const perTool: Record<string, { n: number; hit: number }> = {};
  const used = Math.min(folds, traces.length);

  for (let f = 0; f < used; f++) {
    const train = traces.filter((_, i) => i % used !== f);
    const test = traces.filter((_, i) => i % used === f);
    if (!train.length || !test.length) continue;
    const model = trainModel(train);
    const majority = Object.entries(model.nb.toolCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";

    for (const ex of toExamples(test)) {
      const actual = { tool: ex.target.tool, args: ex.target.args };
      for (const v of VARIANTS) {
        let predicted;
        if (v.majority) {
          predicted = [{ toolName: majority, confidence: 1, rationale: "", suggestedArguments: {}, safeToSpeculate: false }];
        } else {
          const m = v.weights ? { ...model, weights: v.weights } : model;
          predicted = predict(m, {
            last_message: ex.history.length ? ex.history[ex.history.length - 1].result ?? "" : ex.goal ?? "",
            goal: ex.goal,
            history: ex.history,
            top_k: 3,
          });
        }
        recs[v.name].push({ predicted, actual });
      }
      const d = predictDistribution(model, ex.history, ex.goal);
      const top = Object.entries(d).sort((a, b) => b[1] - a[1])[0]?.[0];
      const pt = (perTool[actual.tool] ??= { n: 0, hit: 0 });
      pt.n++;
      if (top === actual.tool) pt.hit++;
    }
  }

  const variants = VARIANTS.map((v) => ({ name: v.name, ...rankStats(recs[v.name]) }));
  const speculation = speculationSweep(recs["full blend"], THRESHOLDS);
  const perToolRows = Object.entries(perTool)
    .map(([tool, x]) => ({ tool, n: x.n, top1: x.hit / x.n }))
    .sort((a, b) => b.n - a.n);

  const markdown = [
    `### Offline replay (${used}-fold CV by session)`,
    `${traces.length} sessions, ${variants[0].n} tool-call transitions.\n`,
    `| predictor | top-1 | top-3 | MRR |\n|---|---|---|---|`,
    ...variants.map((v) => `| ${v.name} | ${pct(v.top1)} | ${pct(v.top3)} | ${v.mrr.toFixed(3)} |`),
    `\n#### Speculation simulator (read-only, fully-specified args, exact match required)\n`,
    specTable(speculation, false),
    `\n#### Per-tool top-1 (full blend)\n`,
    `| tool | n | top-1 |\n|---|---|---|`,
    ...perToolRows.slice(0, 15).map((r) => `| ${r.tool} | ${r.n} | ${pct(r.top1)} |`),
  ].join("\n");

  return { folds: used, sessions: traces.length, transitions: variants[0].n, variants, speculation, perTool: perToolRows, markdown };
}
