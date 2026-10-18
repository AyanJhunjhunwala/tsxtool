// Hashed multinomial logistic regression trained with Adagrad. Unlike the naive
// Bayes slot it is discriminative, so correlated features (bag of words, path counts,
// sequence context) are weighted jointly instead of double-counted.

import { LRModel } from "./types.js";

export const LR_DIM = 1 << 15;

export function hashFeature(f: string, dim = LR_DIM): number {
  let h = 2166136261;
  for (let i = 0; i < f.length; i++) {
    h ^= f.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % dim;
}

export function featureIndices(feats: string[], dim = LR_DIM): number[] {
  return [...new Set(feats.map((f) => hashFeature(f, dim)))];
}

function softmax(z: number[]): number[] {
  const mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export interface LRExample {
  idx: number[];
  y: number; // index into tools
}

export interface LROptions {
  epochs?: number;
  lr?: number;
  l2?: number;
  minCount?: number;
}

export function trainLR(tools: string[], data: LRExample[], opts: LROptions = {}): LRModel {
  const { epochs = 6, lr = 0.15, l2 = 2e-6 } = opts;
  const K = tools.length;
  const W = new Float32Array(LR_DIM * K);
  const G = new Float32Array(LR_DIM * K).fill(1e-3);
  const b = new Float64Array(K);
  const gb = new Float64Array(K).fill(1e-3);
  const order = data.map((_, i) => i);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const z = new Array<number>(K);
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const oi of order) {
      const { idx, y } = data[oi];
      for (let k = 0; k < K; k++) z[k] = b[k];
      for (const f of idx) for (let k = 0; k < K; k++) z[k] += W[f * K + k];
      const p = softmax(z);
      for (let k = 0; k < K; k++) {
        const g = p[k] - (k === y ? 1 : 0);
        gb[k] += g * g;
        b[k] -= (lr * g) / Math.sqrt(gb[k]);
        for (const f of idx) {
          const o = f * K + k;
          const gg = g + l2 * W[o];
          G[o] += gg * gg;
          W[o] -= (lr * gg) / Math.sqrt(G[o]);
        }
      }
    }
  }
  const w: LRModel["w"] = {};
  for (let f = 0; f < LR_DIM; f++) {
    let big = false;
    for (let k = 0; k < K; k++) if (Math.abs(W[f * K + k]) > 0.02) big = true;
    if (big) w[f] = Array.from({ length: K }, (_, k) => Math.round(W[f * K + k] * 1000) / 1000);
  }
  return { dim: LR_DIM, tools, bias: Array.from(b, (v) => Math.round(v * 1000) / 1000), w };
}

/** Probability per tool in `lr.tools`. */
export function scoreLR(lr: LRModel, feats: string[]): Record<string, number> {
  const z = lr.bias.slice();
  for (const f of featureIndices(feats, lr.dim)) {
    const row = lr.w[f];
    if (row) for (let k = 0; k < z.length; k++) z[k] += row[k];
  }
  const p = softmax(z);
  const out: Record<string, number> = {};
  lr.tools.forEach((t, k) => (out[t] = p[k]));
  return out;
}
