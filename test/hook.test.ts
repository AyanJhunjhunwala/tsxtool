import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";

// Config is read at import time, so point everything at a temp dir first.
const dir = mkdtempSync(join(tmpdir(), "prefetch-"));
process.env.TOOLPRED_CACHE_DIR = join(dir, "cache");
process.env.TOOLPRED_LOG = join(dir, "log.jsonl");
process.env.TOOLPRED_ALLOW_LOCAL = "1";
process.env.TOOLPRED_PREFETCH_MIN_P = "0";
process.env.TOOLPRED_WAIT_MS = "300";

let base = "";
const server = createServer((req, res) => {
  if (req.url === "/page") {
    res.setHeader("content-type", "text/html");
    res.end("<html><script>evil()</script><body><h1>Title</h1><p>Hello &amp; welcome</p></body></html>");
  } else if (req.url === "/bin") {
    res.setHeader("content-type", "application/octet-stream");
    res.end("xx");
  } else {
    res.statusCode = 404;
    res.end("no");
  }
});
before(() => new Promise<void>((r) => server.listen(0, "127.0.0.1", () => { base = `http://127.0.0.1:${(server.address() as any).port}`; r(); })));
after(() => server.close());

test("unsafeReason blocks risky URLs", async () => {
  process.env.TOOLPRED_ALLOW_LOCAL = "0";
  const { unsafeReason } = await import("../src/web-cache.js");
  assert.equal(await unsafeReason("http://example.com/x"), "not https");
  assert.match((await unsafeReason("https://user:pw@example.com/")) ?? "", /credentials/);
  assert.match((await unsafeReason("https://example.com/?api_key=1")) ?? "", /secret-like/);
  assert.equal(await unsafeReason("https://127.0.0.1/"), "non-public address");
  assert.equal(await unsafeReason("https://10.1.2.3/"), "non-public address");
  assert.equal(await unsafeReason("https://169.254.169.254/latest/meta-data"), "non-public address");
  process.env.TOOLPRED_ALLOW_LOCAL = "1";
});

test("prefetch caches readable text, skips non-text", async () => {
  const { prefetch, readCache } = await import("../src/web-cache.js");
  const page = await prefetch(`${base}/page`);
  assert.ok(page);
  assert.ok(page.text.includes("Hello & welcome"));
  assert.ok(!page.text.includes("evil()"));
  assert.ok(readCache(`${base}/page`));
  assert.equal(await prefetch(`${base}/bin`), null);
  assert.equal(await prefetch(`${base}/missing`), null);
});

test("PreToolUse serves a cache hit, passes through a miss, and the report counts both", async () => {
  const { handle, report } = await import("../src/hook.js");
  const hit: any = await handle({ hook_event_name: "PreToolUse", tool_name: "WebFetch", tool_input: { url: `${base}/page`, prompt: "what is the title?" } });
  assert.equal(hit.hookSpecificOutput.permissionDecision, "deny");
  assert.match(hit.hookSpecificOutput.permissionDecisionReason, /Title/);
  assert.match(hit.hookSpecificOutput.permissionDecisionReason, /what is the title\?/);
  assert.equal(await handle({ hook_event_name: "PreToolUse", tool_name: "WebFetch", tool_input: { url: `${base}/other` } }), null);
  const r = report(process.env.TOOLPRED_LOG);
  assert.match(r, /hits 1, misses 1/);
});

test("pickPrefetch ranks URLs from a search result and ignores links on fetched pages", async () => {
  const { pickPrefetch } = await import("../src/hook.js");
  const a = `${base}/a`, b = `${base}/b`;
  const { urls } = await pickPrefetch({
    hook_event_name: "PostToolUse",
    tool_name: "WebSearch",
    tool_input: { query: "x" },
    tool_response: `results: ${a} and ${b}`,
  });
  assert.deepEqual(urls, [a, b]);
  const again = await pickPrefetch({
    hook_event_name: "PostToolUse",
    tool_name: "WebFetch",
    tool_input: { url: a },
    tool_response: `links: ${a} ${b}`,
  });
  assert.deepEqual(again.urls, []); // links on a fetched page are not prefetched
});
