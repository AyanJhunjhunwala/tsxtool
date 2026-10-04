import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// ============================================================================
// 1. INPUT SCHEMA VALIDATION (Zod)
// ============================================================================
// This schema defines what arguments the MCP client MUST or CAN pass to our
// predictor tool. Zod validates this untrusted JSON at runtime before our
// logic touches it.
const PredictNextToolInputSchema = z.object({
  // The overarching prompt or objective (e.g. "Fix the failing test in auth.ts")
  goal: z.string().optional(),

  // The immediate last message or previous tool result
  last_message: z.string().min(1, "last_message is required to make a prediction"),

  // How many candidate tools to suggest (defaults to 3 if omitted)
  top_k: z.number().int().positive().default(3),
});

// Infer the TypeScript type directly from the schema so we don't duplicate types
type PredictNextToolInput = z.infer<typeof PredictNextToolInputSchema>;

// ============================================================================
// 2. SERVER INITIALIZATION
// ============================================================================
// Create the MCP server instance. We specify the server identity and advertise
// our capabilities to the connecting client. Here, we declare we support "tools".
const server = new Server(
  {
    name: "tool-call-predictor",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {}, // Declares this server exposes callable tools
    },
  }
);

// ============================================================================
// 3. TOOL DISCOVERY HANDLER (ListTools)
// ============================================================================
// When an MCP client connects, it queries ListTools to learn what tools this
// server provides. We return the name, description, and JSON schema of each tool.
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "predict_next_tool",
        description:
          "Predicts the most probable next tool call and candidate arguments based on conversation context.",
        inputSchema: {
          type: "object",
          properties: {
            goal: {
              type: "string",
              description: "The main user goal or current task description.",
            },
            last_message: {
              type: "string",
              description: "The most recent output, error, or message.",
            },
            top_k: {
              type: "number",
              description: "Number of tool recommendations to return (default: 3).",
            },
          },
          required: ["last_message"],
        },
      },
    ],
  };
});

// ============================================================================
// 4. TOOL EXECUTION HANDLER (CallTool)
// ============================================================================
// When the client actually invokes a tool (e.g., "predict_next_tool"), MCP
// routes the request here with the tool name and arguments.
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "predict_next_tool") {
    // A. Validate incoming arguments using Zod
    const parsed = PredictNextToolInputSchema.safeParse(request.params.arguments);

    if (!parsed.success) {
      throw new Error(
        `Invalid arguments for predict_next_tool: ${JSON.stringify(parsed.error.format())}`
      );
    }

    const { goal, last_message, top_k } = parsed.data;

    // B. Baseline Prediction Heuristic (To be expanded with real models/logic)
    // For now, demonstrate pattern detection on error messages vs viewing files.
    const samplePredictions = [];

    if (
      last_message.toLowerCase().includes("error") ||
      last_message.toLowerCase().includes("failed")
    ) {
      samplePredictions.push({
        toolName: "view_file",
        confidence: 0.88,
        rationale: "Detected error message; inspecting source file is typically the first step.",
        suggestedArguments: {},
      });
      samplePredictions.push({
        toolName: "run_command",
        confidence: 0.65,
        rationale: "Rerunning the command with verbose flags or running tests.",
        suggestedArguments: {},
      });
    } else {
      samplePredictions.push({
        toolName: "replace_file_content",
        confidence: 0.75,
        rationale: "Context indicates readiness to modify target code.",
        suggestedArguments: {},
      });
    }

    // Limit output to the requested top_k results
    const predictions = samplePredictions.slice(0, top_k);

    // C. Return the result back to the MCP client
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              receivedContext: { goal, last_message },
              predictions,
            },
            null,
            2
          ),
        },
      ],
    };
  }

  // Handle unexpected tool names
  throw new Error(`Tool not found: ${request.params.name}`);
});

// ============================================================================
// 5. TRANSPORT SETUP & SERVER LIFECYCLE
// ============================================================================
// Connect the server to Node's stdio streams (standard input and output).
// IMPORTANT: Use console.error (stderr) for logging. console.log (stdout) will
// corrupt the JSON-RPC communication stream!
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("tool-call-predictor server started and listening on stdio.");
}

main().catch((err) => {
  console.error("Fatal error starting MCP server:", err);
  process.exit(1);
});
