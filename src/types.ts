// Shared types for trace data, the trained model, and predictions.

export interface TraceStep {
  tool: string;
  args?: Record<string, unknown>;
  /** Text output (or error) the tool returned; drives the next decision. */
  result?: string;
}

export interface Trace {
  goal?: string;
  /** Which agent harness produced this trace (claude, codex, openai, native). */
  harness?: string;
  steps: TraceStep[];
}

/** PASTE-style argument mapper: how to build `argName` of the next call. */
export type ArgSource =
  | { kind: "copyArg"; fromBack: number; arg: string } // reuse arg of step N back
  | { kind: "resultPath"; fromBack: number; index: number } // Nth file path in result
  | { kind: "resultUrl"; fromBack: number; index: number } // Nth URL in result
  | { kind: "resultRegex"; fromBack: number; pattern: string; group: number }
  | { kind: "goalPath"; index: number };

export interface ArgMapper {
  argName: string;
  source: ArgSource;
  support: number; // fraction of training examples this mapper reproduced
}

/** Hashed multinomial logistic regression over text + context features. */
export interface LRModel {
  dim: number;
  tools: string[];
  bias: number[];
  /** Sparse rows: hashed feature index -> one weight per tool. */
  w: Record<string, number[]>;
}

export interface Model {
  version: 1;
  tools: string[];
  /** counts[order]["ctxKey"]["nextTool"] = n, orders 1..MAX_ORDER (PASTE-style context key lookup). */
  ngrams: Record<string, Record<string, Record<string, number>>>;
  /** Multinomial naive Bayes over features of the last result + goal. */
  nb: {
    toolCounts: Record<string, number>;
    featureCounts: Record<string, Record<string, number>>; // tool -> feature -> n
    featureTotals: Record<string, number>;
    vocabSize: number;
  };
  /** mappers["prevTool>nextTool"] = ordered best-first. */
  argMappers: Record<string, ArgMapper[]>;
  /** Discriminative model; absent in models trained before it existed. */
  lr?: LRModel;
  /** Learned blend weights for [ngram, nb, heuristic, lr]. Older models have 3. */
  weights: number[];
  trainedOn: { traces: number; transitions: number };
}

export interface Prediction {
  toolName: string;
  confidence: number;
  rationale: string;
  suggestedArguments: Record<string, unknown>;
  /** false for tools with side effects: do not speculatively execute. */
  safeToSpeculate: boolean;
}

export const MAX_ORDER = 3;
