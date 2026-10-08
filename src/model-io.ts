import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Model } from "./types.js";

const here = typeof __dirname !== "undefined" ? __dirname : dirname(fileURLToPath(import.meta.url));
export const MODEL_PATH = process.env.TOOL_PREDICTOR_MODEL ?? join(here, "..", "data", "model.json");

/** Returns the trained model, or null (heuristics-only mode) if none exists. */
export function loadModel(path = MODEL_PATH): Model | null {
  if (!existsSync(path)) return null;
  try {
    const m = JSON.parse(readFileSync(path, "utf8")) as Model;
    return m.version === 1 ? m : null;
  } catch {
    return null;
  }
}
