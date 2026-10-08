// Loads training traces from (a) our JSONL format or (b) Claude Code session
// transcripts (~/.claude/projects/**/*.jsonl), pairing tool_use with tool_result.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { Trace, TraceStep } from "./types.js";

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((c) => (c && typeof c === "object" && "text" in c ? String((c as { text: unknown }).text) : ""))
      .join("\n");
  }
  return "";
}

function* walk(path: string): Generator<string> {
  const st = statSync(path);
  if (st.isFile()) {
    if (path.endsWith(".jsonl")) yield path;
    return;
  }
  for (const e of readdirSync(path)) yield* walk(join(path, e));
}

function parseLines(file: string): any[] {
  const out: any[] = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line));
    } catch {
      /* skip corrupt line */
    }
  }
  return out;
}

/** Native format: one trace per line: {"goal": "...", "steps": [{"tool","args","result"}]} */
function fromNative(rows: any[]): Trace[] {
  return rows.filter((r) => Array.isArray(r?.steps)).map((r) => ({ goal: r.goal, steps: r.steps as TraceStep[] }));
}

/** Claude Code transcript: one session file = one trace. */
function fromTranscript(rows: any[]): Trace | null {
  let goal: string | undefined;
  const steps: TraceStep[] = [];
  const byId = new Map<string, TraceStep>();
  for (const r of rows) {
    const content = r?.message?.content;
    if (r?.type === "user" && !goal && !r.isMeta) {
      const t = textOf(content).trim();
      if (t && !t.startsWith("<")) goal = t.slice(0, 500);
    }
    if (!Array.isArray(content)) continue;
    for (const c of content) {
      if (c?.type === "tool_use" && r.type === "assistant") {
        const step: TraceStep = { tool: String(c.name), args: c.input ?? {} };
        steps.push(step);
        byId.set(c.id, step);
      } else if (c?.type === "tool_result") {
        const step = byId.get(c.tool_use_id);
        if (step) step.result = textOf(c.content).slice(0, 8000);
      }
    }
  }
  return steps.length >= 2 ? { goal, steps } : null;
}

export function loadTraces(path: string): Trace[] {
  const traces: Trace[] = [];
  for (const file of walk(path)) {
    const rows = parseLines(file);
    const native = fromNative(rows);
    if (native.length) traces.push(...native);
    else {
      const t = fromTranscript(rows);
      if (t) traces.push(t);
    }
  }
  return traces;
}
