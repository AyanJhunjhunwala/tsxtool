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
function fromTranscript(rows: any[], minSteps = 2): Trace | null {
  let goal: string | undefined;
  const steps: TraceStep[] = [];
  const byId = new Map<string, TraceStep>();
  for (const r of rows) {
    const content = r?.message?.content;
    if (r?.type === "user" && !goal && !r.isMeta && !r.isSidechain) {
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
  return steps.length >= minSteps ? { goal, steps, harness: "claude" } : null;
}

function parseArgs(a: unknown): Record<string, unknown> {
  if (a && typeof a === "object") return a as Record<string, unknown>;
  if (typeof a === "string") {
    try {
      const v = JSON.parse(a);
      return v && typeof v === "object" ? v : { input: a };
    } catch {
      return { input: a };
    }
  }
  return {};
}

/** Codex CLI rollout: response_item payloads of function_call / function_call_output. */
function fromCodex(rows: any[]): Trace | null {
  let goal: string | undefined;
  const steps: TraceStep[] = [];
  const byId = new Map<string, TraceStep>();
  for (const r of rows) {
    if (r?.type !== "response_item") continue;
    const p = r.payload;
    if (p?.type === "message" && p.role === "user" && !goal) {
      const t = textOf(p.content).trim();
      if (t && !t.startsWith("<") && !t.startsWith("#")) goal = t.slice(0, 500);
    } else if (p?.type === "function_call" || p?.type === "custom_tool_call") {
      const step: TraceStep = { tool: String(p.name), args: parseArgs(p.arguments ?? p.input) };
      steps.push(step);
      byId.set(p.call_id, step);
    } else if (p?.type === "function_call_output" || p?.type === "custom_tool_call_output") {
      const step = byId.get(p.call_id);
      if (step) step.result = textOf(p.output).slice(0, 8000);
    }
  }
  return steps.length >= 2 ? { goal, steps, harness: "codex" } : null;
}

/**
 * OpenAI chat-completions style trajectories (OpenHands / SWE-rebench / Open-SWE-Traces):
 * a message list where assistant messages carry tool_calls and tool messages carry results.
 */
export function fromOpenAIMessages(messages: any[]): Trace | null {
  let goal: string | undefined;
  const steps: TraceStep[] = [];
  const byId = new Map<string, TraceStep>();
  for (const m of messages ?? []) {
    if (m?.role === "user" && !goal) goal = textOf(m.content).trim().slice(0, 500) || undefined;
    if (m?.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        const step: TraceStep = { tool: String(tc?.function?.name), args: parseArgs(tc?.function?.arguments) };
        steps.push(step);
        if (tc.id) byId.set(tc.id, step);
      }
    } else if (m?.role === "tool") {
      const step = byId.get(m.tool_call_id);
      if (step) step.result = textOf(m.content).slice(0, 8000);
    }
  }
  return steps.length >= 2 ? { goal, steps, harness: "openai" } : null;
}

/** Single Claude Code transcript, e.g. the live session a hook points at. Tolerates short sessions. */
export function loadTranscript(file: string): Trace | null {
  return fromTranscript(parseLines(file), 0);
}

export function loadTraces(path: string): Trace[] {
  const traces: Trace[] = [];
  for (const file of walk(path)) {
    const rows = parseLines(file);
    const native = fromNative(rows);
    if (native.length) traces.push(...native);
    else {
      const t = rows.some((r) => r?.type === "response_item") ? fromCodex(rows) : fromTranscript(rows);
      if (t) traces.push(t);
    }
  }
  return traces;
}
