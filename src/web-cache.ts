// Fetch-and-cache layer for speculative WebFetch. A prefetch is an outbound request
// the user never approved, so it is deliberately conservative: https GET only, public
// hosts only (re-checked on every redirect), no credentials, no secret-looking query
// params, bounded size and time.

import { lookup } from "node:dns/promises";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, unlinkSync, writeFileSync, appendFileSync } from "node:fs";
import { isIP } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = typeof __dirname !== "undefined" ? __dirname : dirname(fileURLToPath(import.meta.url));
export const CACHE_DIR = process.env.TOOLPRED_CACHE_DIR ?? join(here, "..", "data", "web-cache");
export const LOG_PATH = process.env.TOOLPRED_LOG ?? join(here, "..", "data", "prefetch-log.jsonl");
export const TTL_MS = Number(process.env.TOOLPRED_TTL_MS ?? 10 * 60_000);
const MAX_BYTES = 1_000_000;
const MAX_TEXT = 200_000;
const SECRETISH = /(token|key|secret|auth|sig|session|password|credential)/i;

export interface CachedPage {
  url: string;
  fetchedAt: number;
  ms: number;
  status: number;
  contentType: string;
  text: string;
  truncated: boolean;
}

export function log(entry: Record<string, unknown>): void {
  try {
    mkdirSync(dirname(LOG_PATH), { recursive: true });
    appendFileSync(LOG_PATH, JSON.stringify({ t: Date.now(), ...entry }) + "\n");
  } catch {
    /* logging must never break a hook */
  }
}

export function normalizeUrl(u: string): string {
  try {
    const x = new URL(u);
    x.hash = "";
    let s = x.toString();
    if (x.pathname === "/" && !x.search) s = s.replace(/\/$/, "");
    return s.replace(/\/$/, "");
  } catch {
    return u;
  }
}

const keyOf = (url: string) => createHash("sha1").update(normalizeUrl(url)).digest("hex").slice(0, 24);
const pathOf = (url: string, ext: string) => join(CACHE_DIR, keyOf(url) + ext);

function privateAddress(ip: string): boolean {
  if (ip.includes(":")) {
    const v = ip.toLowerCase();
    return v === "::1" || v === "::" || v.startsWith("fc") || v.startsWith("fd") || v.startsWith("fe80") || v.startsWith("::ffff:");
  }
  const [a, b] = ip.split(".").map(Number);
  return (
    a === 10 || a === 127 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127) || a >= 224
  );
}

/** Returns a reason string when the URL must not be prefetched, else null. */
export async function unsafeReason(raw: string): Promise<string | null> {
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    return "invalid url";
  }
  const allowLocal = process.env.TOOLPRED_ALLOW_LOCAL === "1"; // tests only
  if (u.protocol !== "https:" && !(allowLocal && u.protocol === "http:")) return "not https";
  if (u.username || u.password) return "credentials in url";
  for (const k of u.searchParams.keys()) if (SECRETISH.test(k)) return `secret-like query param ${k}`;
  const allow = (process.env.TOOLPRED_PREFETCH_ALLOW ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  if (allow.length && !allow.some((d) => u.hostname === d || u.hostname.endsWith("." + d))) return "host not in allowlist";
  if (allowLocal) return null;
  try {
    const addrs = isIP(u.hostname) ? [{ address: u.hostname }] : await lookup(u.hostname, { all: true });
    if (!addrs.length || addrs.some((a) => privateAddress(a.address))) return "non-public address";
  } catch {
    return "dns failure";
  }
  return null;
}

export function htmlToText(html: string): string {
  return html
    .replace(/<(script|style|noscript|svg)[\s\S]*?<\/\1>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<\/(p|div|li|h[1-6]|tr|br|section|article)>|<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, " ").replace(/\n\s*\n+/g, "\n\n").trim();
}

async function fetchSafe(url: string, hops = 3): Promise<Response> {
  const why = await unsafeReason(url);
  if (why) throw new Error(`blocked: ${why}`);
  const res = await fetch(url, {
    redirect: "manual",
    signal: AbortSignal.timeout(Number(process.env.TOOLPRED_TIMEOUT_MS ?? 12_000)),
    headers: { "user-agent": "tool-call-predictor-prefetch/1.0", accept: "text/html,text/plain,application/json;q=0.9,*/*;q=0.5" },
  });
  if (res.status >= 300 && res.status < 400 && res.headers.get("location")) {
    if (hops <= 0) throw new Error("too many redirects");
    return fetchSafe(new URL(res.headers.get("location")!, url).toString(), hops - 1);
  }
  return res;
}

async function readCapped(res: Response): Promise<string> {
  const reader = res.body?.getReader();
  if (!reader) return "";
  const chunks: Uint8Array[] = [];
  let n = 0;
  while (n < MAX_BYTES) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
  }
  void reader.cancel().catch(() => {});
  return Buffer.concat(chunks).toString("utf8");
}

export function readCache(url: string, now = Date.now()): CachedPage | null {
  const f = pathOf(url, ".json");
  if (!existsSync(f)) return null;
  try {
    const page = JSON.parse(readFileSync(f, "utf8")) as CachedPage;
    return now - page.fetchedAt <= TTL_MS ? page : null;
  } catch {
    return null;
  }
}

export const inFlight = (url: string) => {
  const f = pathOf(url, ".lock");
  try {
    return existsSync(f) && Date.now() - statSync(f).mtimeMs < 30_000;
  } catch {
    return false;
  }
};

/** Fetch `url` into the cache. Resolves to the page, or null if blocked/failed. */
export async function prefetch(url: string): Promise<CachedPage | null> {
  if (readCache(url)) return readCache(url);
  mkdirSync(CACHE_DIR, { recursive: true });
  const lock = pathOf(url, ".lock");
  writeFileSync(lock, String(Date.now()));
  const t0 = Date.now();
  try {
    const res = await fetchSafe(url);
    const contentType = res.headers.get("content-type") ?? "";
    if (!res.ok || !/(text|json|xml)/i.test(contentType)) throw new Error(`skip status=${res.status} type=${contentType}`);
    const body = await readCapped(res);
    const text = /html/i.test(contentType) ? htmlToText(body) : body;
    const page: CachedPage = {
      url, fetchedAt: Date.now(), ms: Date.now() - t0, status: res.status, contentType,
      text: text.slice(0, MAX_TEXT), truncated: text.length > MAX_TEXT,
    };
    writeFileSync(pathOf(url, ".json"), JSON.stringify(page));
    return page;
  } catch (e) {
    log({ event: "prefetch_failed", url, error: String((e as Error).message ?? e) });
    return null;
  } finally {
    try {
      unlinkSync(lock);
    } catch {
      /* already gone */
    }
  }
}
