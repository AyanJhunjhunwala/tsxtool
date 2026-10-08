// Level 2: shadow-mode logger for Claude Code hooks (PreToolUse / PostToolUse).
// It NEVER changes agent behavior: no stdout, always exit 0. For every tool call
// it (a) scores the prediction made after the previous call against what the agent
// actually did, and (b) predicts the next call for the following turn.
//
//   tsx src/hook.ts pre    # PreToolUse: records start time
//   tsx src/hook.ts post   # PostToolUse: scores + predicts
//
// Logs: data/shadow/<session>.jsonl (override dir with TOOL_PREDICTOR_SHADOW_DIR).

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadModel } from "./model-io.js";
import { predict } from "./predictor.js";
import { Prediction, TraceStep } from "./types.js";

const DIR = process.env.TOOL_PREDICTOR_SHADOW_DIR ?? join(process.cwd(), "data", "shadow");

interface State {
  history: TraceStep[];
  pending?: Prediction[];
  starts: Record<string, number>;
}

function responseText(r: unknown): string {
  if (typeof r === "string") return r;
  if (r && typeof r === "object") {
    const o = r as Record<string, unknown>;
    for (const k of ["stdout", "output", "content", "text"]) if (typeof o[k] === "string") return o[k] as string;
    const file = o.file as { content?: unknown } | undefined;
    if (typeof file?.content === "string") return file.content;
    try {
      return JSON.stringify(r);
    } catch {
      return "";
    }
  }
  return "";
}

function main(mode: string, raw: string) {
  const input = JSON.parse(raw);
  const session = String(input.session_id ?? "unknown").replace(/[^\w-]/g, "_");
  mkdirSync(DIR, { recursive: true });
  const statePath = join(DIR, `${session}.state.json`);
  const state: State = existsSync(statePath)
    ? JSON.parse(readFileSync(statePath, "utf8"))
    : { history: [], starts: {} };
  const callId = String(input.tool_use_id ?? input.tool_name);

  if (mode === "pre") {
    state.starts[callId] = Date.now();
    writeFileSync(statePath, JSON.stringify(state));
    return;
  }

  const tool = String(input.tool_name);
  const args = (input.tool_input ?? {}) as Record<string, unknown>;
  const result = responseText(input.tool_response).slice(0, 8000);
  const start = state.starts[callId];
  delete state.starts[callId];
  const durationMs =
    typeof input.duration_ms === "number" ? input.duration_ms : start ? Date.now() - start : undefined;

  if (state.pending) {
    appendFileSync(
      join(DIR, `${session}.jsonl`),
      JSON.stringify({ ts: Date.now(), predicted: state.pending, actual: { tool, args }, durationMs }) + "\n",
    );
  }

  state.history.push({ tool, args, result });
  state.history = state.history.slice(-30);
  state.pending = predict(loadModel(), { last_message: result, history: state.history, top_k: 3 });
  writeFileSync(statePath, JSON.stringify(state));
}

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => {
  try {
    main(process.argv[2] ?? "post", raw);
  } catch (e) {
    console.error("tool-predictor hook error:", e); // stderr only; never block the agent
  }
  process.exit(0);
});
