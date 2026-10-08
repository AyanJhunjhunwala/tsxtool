// Shared scoring: turns (prediction, actual next call) records into the metrics
// we report, both for offline replay (benchmark.ts) and live shadow logs (report.ts).

import { Prediction } from "./types.js";

export interface ScoreRecord {
  /** Top-k predictions made before the call, best first. */
  predicted: Prediction[];
  actual: { tool: string; args?: Record<string, unknown> };
  /** Wall-clock time the actual call took, if known (ms). */
  durationMs?: number;
}

/** Every suggested argument must equal the one actually used, and there must be at least one. */
export function argsMatch(sug: Record<string, unknown>, actual?: Record<string, unknown>): boolean {
  const keys = Object.keys(sug);
  if (!keys.length || !actual) return false;
  return keys.every((k) => JSON.stringify(sug[k]) === JSON.stringify(actual[k]));
}

export interface SpecRow {
  threshold: number;
  speculated: number; // calls we would have launched early
  hits: number; // launched AND tool+args exactly right (cache would be used)
  wasted: number; // launched but wrong
  precision: number;
  coverage: number; // hits / all actual calls
  savedMs: number; // sum of durations of hits (0 if durations unknown)
  wastedMs: number; // estimated: mean known duration * wasted
}

/**
 * Speculation simulator: launch the top-1 prediction only if confidence >= threshold,
 * it is read-only, and arguments are fully specified (a call can't run without args).
 */
export function speculationSweep(recs: ScoreRecord[], thresholds: number[]): SpecRow[] {
  const known = recs.filter((r) => r.durationMs !== undefined);
  const meanMs = known.length ? known.reduce((a, r) => a + r.durationMs!, 0) / known.length : 0;
  return thresholds.map((threshold) => {
    let speculated = 0, hits = 0, savedMs = 0;
    for (const r of recs) {
      const top = r.predicted[0];
      if (!top || top.confidence < threshold || !top.safeToSpeculate) continue;
      if (!Object.keys(top.suggestedArguments).length) continue;
      speculated++;
      if (top.toolName === r.actual.tool && argsMatch(top.suggestedArguments, r.actual.args)) {
        hits++;
        savedMs += r.durationMs ?? 0;
      }
    }
    const wasted = speculated - hits;
    return {
      threshold,
      speculated,
      hits,
      wasted,
      precision: speculated ? hits / speculated : 0,
      coverage: recs.length ? hits / recs.length : 0,
      savedMs,
      wastedMs: wasted * meanMs,
    };
  });
}

export interface RankStats {
  n: number;
  top1: number;
  top3: number;
  mrr: number;
}

export function rankStats(recs: ScoreRecord[]): RankStats {
  let t1 = 0, t3 = 0, mrr = 0;
  for (const r of recs) {
    const i = r.predicted.findIndex((p) => p.toolName === r.actual.tool);
    if (i === 0) t1++;
    if (i >= 0 && i < 3) t3++;
    if (i >= 0) mrr += 1 / (i + 1);
  }
  const n = recs.length || 1;
  return { n: recs.length, top1: t1 / n, top3: t3 / n, mrr: mrr / n };
}

export const THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95];

const pct = (x: number) => (x * 100).toFixed(1) + "%";

export function specTable(rows: SpecRow[], haveDurations: boolean): string {
  const head = `| confidence ≥ | launched | exact hits | wasted | precision | coverage |${haveDurations ? " time saved | est. time wasted |" : ""}\n|---|---|---|---|---|---|${haveDurations ? "---|---|" : ""}`;
  const body = rows
    .map(
      (r) =>
        `| ${r.threshold} | ${r.speculated} | ${r.hits} | ${r.wasted} | ${pct(r.precision)} | ${pct(r.coverage)} |` +
        (haveDurations ? ` ${(r.savedMs / 1000).toFixed(1)}s | ${(r.wastedMs / 1000).toFixed(1)}s |` : ""),
    )
    .join("\n");
  return head + "\n" + body;
}

export { pct };
