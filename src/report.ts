// Level 2: aggregate shadow logs into a results table.
//   tsx src/report.ts [shadow-dir] [--out report.md]

import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ScoreRecord, THRESHOLDS, pct, rankStats, speculationSweep, specTable } from "./scoring.js";

export function buildReport(dir: string): string {
  const recs: ScoreRecord[] = [];
  let sessions = 0;
  for (const f of readdirSync(dir).filter((x) => x.endsWith(".jsonl"))) {
    sessions++;
    for (const line of readFileSync(join(dir, f), "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        recs.push(JSON.parse(line));
      } catch {
        /* skip */
      }
    }
  }
  if (!recs.length) return "No scored predictions yet. Run some sessions with the hook installed.";

  const rank = rankStats(recs);
  const timed = recs.filter((r) => r.durationMs !== undefined).length;
  const totalMs = recs.reduce((a, r) => a + (r.durationMs ?? 0), 0);
  const freq: Record<string, number> = {};
  for (const r of recs) freq[r.actual.tool] = (freq[r.actual.tool] ?? 0) + 1;
  const baseline = Math.max(...Object.values(freq)) / recs.length;
  const sweep = speculationSweep(recs, THRESHOLDS);

  return [
    `### Live shadow-mode results`,
    `${sessions} sessions, ${recs.length} scored tool calls (${timed} with timings, ${(totalMs / 1000).toFixed(0)}s total tool time).\n`,
    `| metric | value |\n|---|---|`,
    `| top-1 accuracy | ${pct(rank.top1)} |`,
    `| top-3 accuracy | ${pct(rank.top3)} |`,
    `| MRR | ${rank.mrr.toFixed(3)} |`,
    `| majority-tool baseline top-1 | ${pct(baseline)} |`,
    `\n#### Speculation simulator\n`,
    `Launch the top-1 prediction early only if read-only, args fully specified, and confidence ≥ threshold. A hit needs exact tool + args. "Time saved" is the summed duration of hit calls: an upper bound, since real saving is capped by how much of the call overlaps model thinking time.\n`,
    specTable(sweep, timed > 0),
  ].join("\n");
}

if (process.argv[1]?.endsWith("report.ts")) {
  const args = process.argv.slice(2);
  const oi = args.indexOf("--out");
  const out = oi >= 0 ? args.splice(oi, 2)[1] : undefined;
  const md = buildReport(args[0] ?? join(process.cwd(), "data", "shadow"));
  if (out) writeFileSync(out, md + "\n");
  console.log(md);
}
