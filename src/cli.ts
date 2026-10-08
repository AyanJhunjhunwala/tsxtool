// CLI: train / eval / predict.
//   tsx src/cli.ts train [traces-path ...] [--synthetic N] [--out data/model.json]
//   tsx src/cli.ts eval  [traces-path ...] [--synthetic N] [--json 1]
//   tsx src/cli.ts predict "<last message>" [--goal "..."] [--history a,b]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { loadTraces } from "./importer.js";
import { predict } from "./predictor.js";
import { generateSynthetic } from "./synthetic.js";
import { runBenchmark } from "./benchmark.js";
import { trainModel } from "./trainer.js";
import { loadModel, MODEL_PATH } from "./model-io.js";
import { Trace } from "./types.js";

const [cmd, ...rest] = process.argv.slice(2);
const flags: Record<string, string> = {};
const pos: string[] = [];
for (let i = 0; i < rest.length; i++) {
  if (rest[i].startsWith("--")) flags[rest[i].slice(2)] = rest[++i];
  else pos.push(rest[i]);
}

function gather(): Trace[] {
  const traces = pos.flatMap((p) => loadTraces(p));
  if (flags.synthetic) traces.push(...generateSynthetic(Number(flags.synthetic)));
  if (!traces.length) traces.push(...generateSynthetic(300));
  return traces;
}

if (cmd === "train") {
  const traces = gather();
  const model = trainModel(traces);
  const out = flags.out ?? MODEL_PATH;
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(model));
  console.error(`trained on ${model.trainedOn.traces} traces / ${model.trainedOn.transitions} transitions`);
  console.error(`tools: ${model.tools.join(", ")}`);
  console.error(`blend weights [ngram, nb, heuristic]: ${model.weights.join(", ")}`);
  console.error(`wrote ${out}`);
} else if (cmd === "eval") {
  const r = runBenchmark(gather());
  console.log(flags.json ? JSON.stringify(r, null, 2) : r.markdown);
} else if (cmd === "predict") {
  const history = (flags.history ?? "").split(",").filter(Boolean).map((tool) => ({ tool }));
  console.log(JSON.stringify(predict(loadModel(), { last_message: pos[0] ?? "", goal: flags.goal, history, top_k: 3 }), null, 2));
} else {
  console.error("usage: cli.ts <train|eval|predict> ...");
  process.exit(1);
}
