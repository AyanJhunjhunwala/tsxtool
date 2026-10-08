# tool-call-predictor

A local Model Context Protocol (MCP) server that takes conversational context from an LLM or coding agent and returns ranked probabilities for which tools to call next.

## Overview

When coding agents work through problems, they make decisions about which tool to reach for next. This server acts as a prediction sidecar over stdio. An agent passes its current context (overarching goal, recent command output, or error text), and the server returns a ranked probability distribution over candidate tools, potential arguments, and confidence scores.

## Contract & Interface

The server exposes a single tool over the MCP standard:

### `predict_next_tool`

**Input parameters:**
- `last_message` (string, required): The latest output, error message, or prompt from the current turn.
- `goal` (string, optional): The broader objective the agent is trying to accomplish.
- `top_k` (number, optional): The maximum number of tool suggestions to return. Defaults to 3.

**Response shape:**
```json
{
  "receivedContext": {
    "goal": "Fix failing tests",
    "last_message": "TypeError: Cannot read properties of undefined"
  },
  "predictions": [
    {
      "toolName": "view_file",
      "confidence": 0.88,
      "rationale": "Detected error message; inspecting source file is typically the first step.",
      "suggestedArguments": {}
    },
    {
      "toolName": "run_command",
      "confidence": 0.65,
      "rationale": "Rerunning the command with verbose flags or running tests.",
      "suggestedArguments": {}
    }
  ]
}
```

## How prediction works (PASTE-inspired)

Modeled on Microsoft/SJTU's [PASTE](https://www.microsoft.com/en-us/research/publication/act-while-thinking-accelerating-llm-agents-via-pattern-aware-speculative-tool-execution/) ("Act While Thinking", arXiv 2603.18897): agent tool sequences are stable, and arguments flow predictably between calls. Three signals are blended into one distribution (`src/predictor.ts`):

1. **Sequence context (control flow)**: order-1..3 n-gram tables keyed on the last tool names, with backoff, learned from traces.
2. **Output classifier (text)**: multinomial naive Bayes over heuristic features of the last output (stack trace, test failure, search hits, git status, goal verbs, ... see `src/features.ts`) plus bag-of-words.
3. **Rule-based prior**: hand-written signal-to-role rules (read/search/edit/run/web), so it works with zero training data and any tool names.

Blend weights are grid-searched on a held-out split at train time. **Argument mappers** (PASTE's data-flow half) are mined per `prevTool>nextTool` pair: e.g. "`path` of `view_file` = first path in the previous `grep` output". Each prediction includes `safeToSpeculate`, false for anything that could mutate state, so a host can pre-run only read-only calls.

### Training

```bash
# From Claude Code session transcripts (tool_use/tool_result pairs are paired automatically)
npm run train -- ~/.claude/projects
# From your own JSONL: {"goal": "...", "steps": [{"tool","args","result"}]}
npm run train -- traces/
# Bootstrap with synthetic workflows
npm run train -- --synthetic 400
npm run eval  -- ~/.claude/projects     # 5-fold CV by trace: top-1 / top-3 / MRR vs majority baseline
npm run predict -- "FAIL src/a.ts" --goal "fix tests" --history run_command
```

The model is written to `data/model.json` (gitignored) and loaded by the server; without it the server runs heuristics only. `predict_next_tool` also accepts `history` (`[{tool,args,result}]`) and `available_tools`.

## Setup & Running

### Requirements
- Node.js 20+

### Install Dependencies
```bash
npm install
```

### Local Development
Run with hot reloading:
```bash
npm run dev
```

Run in production mode:
```bash
npm run start
```

### Inspect with MCP Inspector
To test JSON-RPC message exchanges interactively in the browser:
```bash
npm run inspect
```

## Client Configuration

Add this server to your local MCP client configuration (Cursor, Claude Desktop, Antigravity, or Cline):

```json
{
  "mcpServers": {
    "tool-call-predictor": {
      "command": "npx",
      "args": ["tsx", "/Users/ayanjhunjhunwala/Desktop/Code/tsxtool/main.tsx"]
    }
  }
}
```

## Technical Notes

- **Transport**: Communicates strictly over standard input/output (`stdio`) using JSON-RPC 2.0.
- **Diagnostics**: All internal logging writes to `stderr` via `console.error` to avoid corrupting `stdout` JSON-RPC messages.
- **Runtime Validation**: Incoming parameters are checked with Zod before processing to handle malformed calls from models without server crashes.
