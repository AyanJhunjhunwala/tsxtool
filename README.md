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
