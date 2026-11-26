// DopplerX release gate, Chromium surface. Serves one installed package (its files exactly as
// installed from the tarball) and two local model directories over loopback, and runs greedy text
// generation and a protein sequence encode on the real GPU in headless Chromium with WebGPU.
// The adapter must be a hardware adapter (not a fallback adapter); no off-machine request is allowed.
// usage: node browser-gpu-gate.js <install-dir> <text-model-dir> <protein-model-dir> <out.json>
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';

const [installDir, textModelDir, proteinModelDir, outFile] = process.argv.slice(2);
const require = createRequire(path.join(installDir, 'package.json'));
const { chromium } = require('playwright');
const pkg = JSON.parse(fs.readFileSync(path.join(installDir, 'node_modules/dopplerx/package.json'), 'utf8'));
const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

const PROMPT = 'Explain in two sentences why the sky is blue.';
const GREEDY = { maxTokens: 32, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, useChatTemplate: true };
const PROTEIN = 'MKTIIALSYIFCLVFADYKDDDDK';

const page = `<!doctype html><meta charset="utf-8"><script type="module">
const r = { surface: 'chromium', calls: {} };
try {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const info = adapter.info || {};
  r.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
    isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null, features: [...adapter.features].sort() };
  const base = '/node_modules/dopplerx/src/';
  const device = await import(base + 'gpu/device.js');
  const { createPipeline } = await import(base + 'inference/pipelines/text.js');
  await device.initDevice();
  const run = async (dir, fn) => {
    const manifest = await (await fetch(dir + 'manifest.json')).json();
    const t0 = performance.now();
    const pipeline = await createPipeline(manifest, { baseUrl: location.origin + dir,
      runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 512 } } } } });
    const loadMs = performance.now() - t0;
    try { return { loadMs, ...(await fn(pipeline)) }; } finally { await pipeline.unload?.(); }
  };
  r.text = await run('/models/text/', async (pipeline) => {
    const tokens = []; const t0 = performance.now(); let firstTokenMs = null;
    for await (const t of pipeline.generateTokens([{ role: 'user', content: ${JSON.stringify(PROMPT)} }], ${JSON.stringify(GREEDY)})) {
      if (firstTokenMs == null) firstTokenMs = performance.now() - t0;
      tokens.push(t);
    }
    return { tokens, generateMs: performance.now() - t0, firstTokenMs, useGPU: pipeline.useGPU ?? null,
      text: pipeline.tokenizer?.decode ? pipeline.tokenizer.decode(tokens, true, true) : null,
      stats: JSON.parse(JSON.stringify(pipeline.stats ?? {})) };
  });
  r.protein = await run('/models/protein/', async (pipeline) => {
    const t0 = performance.now();
    const res = await pipeline.encodeSequence(${JSON.stringify(PROTEIN)}, { includeTokenEmbeddings: true });
    return { encodeMs: performance.now() - t0, alphabet: res.alphabet, tokens: Array.from(res.tokens),
      embeddingDim: res.pooledEmbedding?.length ?? null, pooledEmbedding: Array.from(res.pooledEmbedding ?? []),
      tokenEmbeddings: Array.from(res.tokenEmbeddings ?? []) };
  });
} catch (e) { r.error = String(e && (e.stack || e.message) || e).slice(0, 1200); }
window.__result = r;
</script>`;

const types = { '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.svg': 'image/svg+xml', '.bin': 'application/octet-stream' };
const mounts = [['/models/text/', textModelDir], ['/models/protein/', proteinModelDir], ['/', installDir]];
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end(page); return; }
  const [prefix, dir] = mounts.find(([p]) => url.pathname.startsWith(p));
  const file = path.join(dir, path.normalize(decodeURIComponent(url.pathname.slice(prefix.length - 1))));
  if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  const stat = fs.statSync(file);
  const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range || '');
  const headers = { 'content-type': types[path.extname(file)] ?? 'application/octet-stream', 'accept-ranges': 'bytes' };
  if (range) {
    const start = Number(range[1]); const end = range[2] ? Number(range[2]) : stat.size - 1;
    res.writeHead(206, { ...headers, 'content-range': `bytes ${start}-${end}/${stat.size}`, 'content-length': end - start + 1 });
    fs.createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { ...headers, 'content-length': stat.size });
  fs.createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const origin = `http://127.0.0.1:${server.address().port}`;
const blocked = [];
const browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_CHANNEL ? { channel: process.env.BROWSER_CHANNEL } : {}), args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--ignore-gpu-blocklist'] });
const out = { schema: 'dopplerx-gate.chromium/v1', startedAtUtc: new Date().toISOString(), package: { name: pkg.name, version: pkg.version,
  lockIntegrity: JSON.parse(fs.readFileSync(path.join(installDir, 'package-lock.json'), 'utf8')).packages['node_modules/dopplerx'].integrity },
  models: { text: sha256(fs.readFileSync(path.join(textModelDir, 'manifest.json'))), protein: sha256(fs.readFileSync(path.join(proteinModelDir, 'manifest.json'))) },
  prompt: PROMPT, sampling: GREEDY, sequence: PROTEIN };
try {
  const context = await browser.newContext();
  await context.route('**/*', (route) => (route.request().url().startsWith(origin) ? route.continue() : (blocked.push(route.request().url()), route.abort())));
  const pg = await context.newPage();
  const consoleErrors = [];
  pg.on('pageerror', (e) => consoleErrors.push(String(e.message)));
  await pg.goto(origin + '/');
  await pg.waitForFunction(() => window.__result, null, { timeout: 600000 });
  Object.assign(out, await pg.evaluate(() => window.__result), { browserVersion: browser.version(), blockedExternalRequests: blocked, pageErrors: consoleErrors });
} finally {
  await browser.close();
  server.close();
}
const markers = [];
const walk = (v, t) => { if (v && typeof v === 'object') for (const [k, e] of Object.entries(v)) { if (/fallback|cpu/i.test(k) && e && e !== 0 && !(Array.isArray(e) && !e.length)) markers.push(`${t}.${k}=${JSON.stringify(e).slice(0, 60)}`); walk(e, `${t}.${k}`); } };
walk(out.text?.stats, 'text.stats');
out.noFallback = { isFallbackAdapter: out.adapter?.isFallbackAdapter ?? null, useGPU: out.text?.useGPU ?? null, fallbackOrCpuMarkersInStats: markers,
  pass: !out.error && out.adapter?.isFallbackAdapter !== true && out.text?.useGPU !== false && markers.length === 0 };
out.finishedAtUtc = new Date().toISOString();
fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
console.log(JSON.stringify({ version: pkg.version, browser: out.browserVersion, adapter: out.adapter && { ...out.adapter, features: out.adapter.features?.length },
  error: out.error, noFallback: out.noFallback, tokens: out.text?.tokens?.length, text: out.text?.text?.slice(0, 120),
  decodeMode: out.text?.stats?.decodeMode, stop: out.text?.stats?.stopReason, alphabet: out.protein?.alphabet, blocked: blocked.length,
  timing: { textLoad: out.text?.loadMs, gen: out.text?.generateMs, ttft: out.text?.firstTokenMs, protLoad: out.protein?.loadMs, enc: out.protein?.encodeMs } }, null, 1));
