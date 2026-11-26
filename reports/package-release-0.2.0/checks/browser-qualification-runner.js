// Usage: node run.js <consumer-dir> <out.json> <playwright-package-dir>
// Serves the installed dopplerx package (browser export conditions via an import
// map) and drives headless Chromium with WebGPU removed from the page.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [consumer, out, playwrightDir] = process.argv.slice(2);
const require = createRequire(path.join(playwrightDir, 'package.json'));
const { chromium } = require('playwright');
const pkg = JSON.parse(fs.readFileSync(path.join(consumer, 'node_modules/dopplerx/package.json'), 'utf8'));
const pick = (v) => (typeof v === 'string' ? v : (v.browser ? pick(v.browser) : (v.import ?? v.default)));
const imports = {};
for (const [k, v] of Object.entries(pkg.exports)) {
  imports[k === '.' ? 'dopplerx' : 'dopplerx' + k.slice(1)] = '/node_modules/dopplerx/' + pick(v).replace(/^\.\//, '');
}
const page = `<!doctype html><meta charset="utf-8"><script type="importmap">${JSON.stringify({ imports })}</script>
<script type="module">
const r = { navigatorGpu: !!navigator.gpu, entries: {}, calls: {} };
for (const spec of ${JSON.stringify(Object.keys(imports))}) {
  try { const m = await import(spec); r.entries[spec] = { ok: true, exports: Object.keys(m).length }; }
  catch (e) { r.entries[spec] = { ok: false, error: String(e && e.message || e).slice(0, 200) }; }
}
try { r.version = (await import('dopplerx')).DOPPLER_VERSION; } catch (e) { r.version = 'error: ' + e.message; }
try {
  const { createDopplerProvider } = await import('dopplerx/provider');
  const p = createDopplerProvider({ local: { model: 'qwen3-0.8b' }, policy: { mode: 'local-only', fallbackOn: ['gpu_unavailable'] } });
  try { const o = await p.generate('Hello'); r.calls.provider = { ok: true, out: String(o).slice(0, 80) }; }
  catch (e) { r.calls.provider = { ok: false, name: e.name, code: e.code ?? null, message: String(e.message).slice(0, 240),
    causeCode: e.cause?.code ?? null, causeMessage: e.cause ? String(e.cause.message).slice(0, 240) : null,
    failureClass: e.receipt?.failure?.failureClass ?? e.details?.failureClass ?? null }; }
} catch (e) { r.calls.provider = { ok: false, setupError: String(e.message).slice(0, 200) }; }
try {
  const { dr } = await import('dopplerx/compat');
  try { await dr.load('qwen3-0.8b'); r.calls.compatLoad = { ok: true }; }
  catch (e) { r.calls.compatLoad = { ok: false, name: e.name, code: e.code ?? null, message: String(e.message).slice(0, 240) }; }
} catch (e) { r.calls.compatLoad = { ok: false, setupError: String(e.message).slice(0, 200) }; }
window.__result = r;
</script>`;
const types = { '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.svg': 'image/svg+xml' };
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end(page); return; }
  const file = path.join(consumer, path.normalize(decodeURIComponent(url.pathname)));
  if (!file.startsWith(consumer) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const origin = `http://127.0.0.1:${server.address().port}`;
// Only the model manifest JSON may leave the machine; weights, tokenizer and everything else are blocked.
const MANIFEST_RE = /^https:\/\/huggingface\.co\/clocksmith\/rdrr\/resolve\/[0-9a-f]+\/models\/[^/]+\/manifest\.json$/;
const blocked = [];
const allowed = [];
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  await context.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
  });
  await context.route('**/*', (route) => {
    const u = route.request().url();
    if (u.startsWith(origin)) return route.continue();
    if (MANIFEST_RE.test(u)) { allowed.push(u); return route.continue(); }
    blocked.push(u);
    return route.abort();
  });
  const pg = await context.newPage();
  const pageErrors = [];
  pg.on('pageerror', (e) => pageErrors.push(String(e.message)));
  await pg.goto(origin + '/');
  await pg.waitForFunction(() => window.__result, null, { timeout: 120000 });
  const result = await pg.evaluate(() => window.__result);
  Object.assign(result, { browserVersion: browser.version(), allowedExternalRequests: allowed, blockedExternalRequests: blocked, pageErrors });
  fs.writeFileSync(out, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
  server.close();
}
