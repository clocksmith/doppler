// Complete-application comparison: the same triage application on Doppler and on
// Transformers.js, measured from an empty browser profile through offline reopening
// and crash recovery. One engine runs at a time; engine order alternates per repetition.
import fs from 'node:fs/promises';
import { readFileSync, readdirSync, createReadStream, statSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { loadSplit, stratifiedSample } from '../dataset.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, '..');
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const outDir = path.resolve(config.outDir);
await fs.mkdir(outDir, { recursive: true });

const readCsv = (split) => fs.readFile(path.join(config.csvDirectory, `issues_${split}.csv`), 'utf8');
const seed = 'issue-triage-20261021';
const train = stratifiedSample(await loadSplit('train', { readLocal: readCsv }), { split: 'train', perCell: 6, seed });
const test = stratifiedSample(await loadSplit('test', { readLocal: readCsv }), { split: 'test', perCell: 10, seed });
const strip = ({ index, repo, label, title, body }) => ({ index, repo, label, title, body });
const warmIssues = train.slice(0, 21).map(strip);
const longIssue = strip([...train].sort((a, b) => b.body.length - a.body.length)[0]);

// Minimal static server; counts model bytes served so acquisition is measured at the origin.
const mounts = [
  { prefix: '/app', dir: appRoot },
  { prefix: '/capsule', dir: path.resolve(config.capsuleDirectory), model: true },
  { prefix: '/tjs-package', dir: path.resolve(config.transformersPackageDirectory) },
  { prefix: `/tjs-remote/${config.transformersModelId}/resolve/main`, dir: path.resolve(config.transformersModelDirectory), model: true },
].sort((a, b) => b.prefix.length - a.prefix.length);
const served = { modelBytes: 0, modelRequests: 0 };
const TYPES = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.html': 'text/html',
  '.wasm': 'application/wasm', '.wgsl': 'text/plain' };
const httpServer = http.createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://local').pathname);
  const mount = mounts.find((m) => pathname.startsWith(`${m.prefix}/`));
  const file = mount && path.resolve(mount.dir, `.${pathname.slice(mount.prefix.length)}`);
  let stat = null;
  try { if (file && file.startsWith(mount.dir + path.sep)) stat = statSync(file); } catch { /* missing */ }
  if (!stat?.isFile()) { response.writeHead(404).end(); return; }
  const range = /bytes=(\d*)-(\d*)/.exec(request.headers.range ?? '');
  const start = range && range[1] !== '' ? Number(range[1]) : 0;
  const end = range && range[2] !== '' ? Math.min(Number(range[2]), stat.size - 1) : stat.size - 1;
  response.writeHead(range ? 206 : 200, {
    'Content-Type': TYPES[path.extname(file)] ?? 'application/octet-stream',
    'Content-Length': end - start + 1,
    'Accept-Ranges': 'bytes',
    ...(range ? { 'Content-Range': `bytes ${start}-${end}/${stat.size}` } : {}),
  });
  if (request.method === 'HEAD') { response.end(); return; }
  if (mount.model) served.modelRequests += 1;
  const stream = createReadStream(file, { start, end });
  if (mount.model) stream.on('data', (chunk) => { served.modelBytes += chunk.length; });
  stream.on('error', () => response.destroy());
  stream.pipe(response);
});
await new Promise((resolve) => httpServer.listen(0, '127.0.0.1', resolve));
const server = {
  baseUrl: `http://127.0.0.1:${httpServer.address().port}`,
  close: () => new Promise((resolve) => { httpServer.closeAllConnections(); httpServer.close(resolve); }),
};
const descriptor = JSON.parse(await fs.readFile(config.descriptorPath, 'utf8'));
const ARGS = ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--ignore-gpu-blocklist',
  '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'];
const MODEL_ROUTES = /\/(capsule|tjs-remote)\//;

function chromeProcesses(profileDir) {
  const rows = [];
  for (const pid of readdirSync('/proc').filter((p) => /^\d+$/.test(p))) {
    try {
      const cmd = readFileSync(`/proc/${pid}/cmdline`, 'utf8');
      if (!cmd.includes(profileDir)) continue;
      const rss = Number(/VmRSS:\s+(\d+)/.exec(readFileSync(`/proc/${pid}/status`, 'utf8'))?.[1] ?? 0) * 1024;
      let gtt = 0;
      for (const fd of readdirSync(`/proc/${pid}/fdinfo`)) {
        const info = readFileSync(`/proc/${pid}/fdinfo/${fd}`, 'utf8');
        if (!info.includes('drm-driver:\tamdgpu')) continue;
        gtt += Number(/drm-memory-gtt:\s+(\d+)/.exec(info)?.[1] ?? 0) * 1024;
        gtt += Number(/drm-memory-vram:\s+(\d+)/.exec(info)?.[1] ?? 0) * 1024;
      }
      rows.push({ pid: Number(pid), type: /--type=([a-z-]+)/.exec(cmd)?.[1] ?? 'browser', rss, gpuMemory: gtt });
    } catch { /* process exited */ }
  }
  return rows;
}

function startSampler(profileDir) {
  const peak = { totalRss: 0, rendererRss: 0, gpuProcessRss: 0, gpuMemory: 0 };
  const timer = setInterval(() => {
    const rows = chromeProcesses(profileDir);
    const sum = (f) => rows.filter(f).reduce((a, r) => a + r.rss, 0);
    peak.totalRss = Math.max(peak.totalRss, sum(() => true));
    peak.rendererRss = Math.max(peak.rendererRss, sum((r) => r.type === 'renderer'));
    peak.gpuProcessRss = Math.max(peak.gpuProcessRss, sum((r) => r.type === 'gpu-process'));
    peak.gpuMemory = Math.max(peak.gpuMemory, rows.reduce((a, r) => a + r.gpuMemory, 0));
  }, 200);
  return { stop: () => { clearInterval(timer); return peak; } };
}

async function launch(profileDir, { offline = false } = {}) {
  const context = await chromium.launchPersistentContext(profileDir, { channel: 'chrome', headless: true, args: ARGS });
  const startBytes = served.modelBytes;
  const startRequests = served.modelRequests;
  const blockedUrls = [];
  const transfer = {
    get modelBytes() { return served.modelBytes - startBytes; },
    get modelRequests() { return served.modelRequests - startRequests; },
    blockedUrls,
  };
  if (offline) {
    await context.route(MODEL_ROUTES, (route) => {
      blockedUrls.push(new URL(route.request().url()).pathname);
      return route.abort('internetdisconnected');
    });
  }
  const page = context.pages()[0] ?? await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error.message).slice(0, 300)));
  page.on('console', (message) => {
    if (message.type() === 'warning' || message.type() === 'error') errors.push(`${message.type()}: ${message.text().slice(0, 300)}`);
  });
  await page.goto(`${server.baseUrl}/app/tools/harness.html`);
  await page.waitForFunction(() => window.harnessReady === true);
  return { context, page, transfer, errors };
}

const openConfig = (engine, mode) => (engine === 'doppler'
  ? { descriptor, mode }
  : { modelId: config.transformersModelId });
const scoresOf = (s) => s.routes.map((r) => [r.id, r.yes, r.no]);

async function runEngine(engine, rep) {
  const profileDir = path.join(outDir, 'profiles', `${engine}-${rep}`);
  await fs.rm(profileDir, { recursive: true, force: true });
  const result = { engine, rep, profileDir };
  // 1. Empty profile: acquisition, opening, first decision, resident reuse, cancellation.
  {
    const sampler = startSampler(profileDir);
    const { context, page, transfer, errors } = await launch(profileDir);
    const launchedAt = Date.now();
    result.cold = await page.evaluate(([e, c]) => window.harness.open(e, c), [engine, openConfig(engine, 'install')]);
    const first = await page.evaluate((i) => window.harness.triage(i), warmIssues[0]);
    result.cold.firstDecisionMs = first.elapsedMs;
    result.cold.launchToFirstDecisionMs = Date.now() - launchedAt;
    result.referenceScores = scoresOf(first);
    result.warmMs = [];
    for (const issue of warmIssues.slice(1)) result.warmMs.push((await page.evaluate((i) => window.harness.triage(i), issue)).elapsedMs);
    result.cancel = await page.evaluate(([i, d]) => window.harness.cancelProbe(i, d), [longIssue, 60]);
    const after = await page.evaluate((i) => window.harness.triage(i), warmIssues[0]);
    result.cancel.reuseIdentical = JSON.stringify(scoresOf(after)) === JSON.stringify(result.referenceScores);
    if (rep === 0 && config.quality !== false) {
      result.quality = { train: [], test: [] };
      for (const [name, issues] of [['train', train], ['test', test]]) {
        for (const issue of issues) {
          const s = await page.evaluate((i) => window.harness.triage(i), strip(issue));
          result.quality[name].push({ index: issue.index, repo: issue.repo, label: issue.label,
            routes: s.routes, suggestedRoute: s.suggestedRoute, elapsedMs: s.elapsedMs });
        }
      }
    }
    result.cold.storage = await page.evaluate(async () => {
      const estimate = await navigator.storage.estimate();
      const cached = [];
      for (const name of await caches.keys()) {
        for (const request of await (await caches.open(name)).keys()) cached.push(`${name}: ${new URL(request.url).pathname}`);
      }
      return { usage: estimate.usage, quota: estimate.quota, usageDetails: estimate.usageDetails ?? null, cached };
    });
    await page.evaluate(() => window.harness.close());
    result.cold.transfer = { modelBytes: transfer.modelBytes, modelRequests: transfer.modelRequests };
    result.cold.pageErrors = errors;
    await context.close();
    result.cold.peakMemory = sampler.stop();
  }
  // 2. Offline reopening from retained storage.
  {
    const { context, page, transfer, errors } = await launch(profileDir, { offline: true });
    try {
      result.offline = await page.evaluate(([e, c]) => window.harness.open(e, c), [engine, openConfig(engine, 'open-retained')]);
      const s = await page.evaluate((i) => window.harness.triage(i), warmIssues[0]);
      result.offline.identical = JSON.stringify(scoresOf(s)) === JSON.stringify(result.referenceScores);
      result.offline.firstDecisionMs = s.elapsedMs;
      await page.evaluate(() => window.harness.close());
    } catch (error) {
      result.offline = { failed: true, error: String(error.message).slice(0, 400) };
    }
    result.offline.transfer = { modelBytes: transfer.modelBytes, modelRequests: transfer.modelRequests, blockedUrls: transfer.blockedUrls };
    result.offline.pageErrors = errors;
    await context.close().catch(() => {});
  }
  // 3. Browser killed during a request, then relaunched offline.
  {
    const { context, page } = await launch(profileDir, { offline: true });
    let relaunched = null;
    try {
      await page.evaluate(([e, c]) => window.harness.open(e, c), [engine, openConfig(engine, 'open-retained')]);
      page.evaluate((i) => window.harness.triage(i), longIssue).catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 150));
      for (const p of chromeProcesses(profileDir)) { try { process.kill(p.pid, 'SIGKILL'); } catch { /* gone */ } }
      await context.close().catch(() => {});
      const startedAt = Date.now();
      relaunched = await launch(profileDir, { offline: true });
      await relaunched.page.evaluate(([e, c]) => window.harness.open(e, c), [engine, openConfig(engine, 'open-retained')]);
      const s = await relaunched.page.evaluate((i) => window.harness.triage(i), warmIssues[0]);
      result.recovery = { relaunchToFirstDecisionMs: Date.now() - startedAt,
        identical: JSON.stringify(scoresOf(s)) === JSON.stringify(result.referenceScores) };
      await relaunched.page.evaluate(() => window.harness.close());
    } catch (error) {
      result.recovery = { failed: true, error: String(error.message).slice(0, 400),
        blockedUrls: relaunched?.transfer.blockedUrls ?? null };
    } finally {
      await context.close().catch(() => {});
      await relaunched?.context.close().catch(() => {});
    }
  }
  await fs.rm(profileDir, { recursive: true, force: true });
  return result;
}

const results = [];
try {
  for (let rep = 0; rep < config.repetitions; rep += 1) {
    const order = (rep % 2 === 0 ? ['doppler', 'transformersjs'] : ['transformersjs', 'doppler'])
      .filter((engine) => !config.engines || config.engines.includes(engine));
    for (const engine of order) {
      const r = await runEngine(engine, rep);
      results.push(r);
      await fs.writeFile(path.join(outDir, `${engine}-${rep}.json`), `${JSON.stringify(r, null, 2)}\n`);
      const med = [...r.warmMs].sort((a, b) => a - b)[Math.floor(r.warmMs.length / 2)];
      console.log(`${engine} rep${rep}: open ${r.cold.openMs.toFixed(0)} ms, first ${r.cold.firstDecisionMs.toFixed(0)} ms, warm median ${med.toFixed(0)} ms, `
        + `peak RSS ${(r.cold.peakMemory.totalRss / 2 ** 30).toFixed(2)} GiB, cancel ${r.cancel.status} ${r.cancel.abortToSettleMs?.toFixed(0)} ms, `
        + `offline ${r.offline.failed ? 'FAILED' : `${r.offline.openMs.toFixed(0)} ms identical=${r.offline.identical}`}, `
        + `recovery ${r.recovery.failed ? 'FAILED' : `${r.recovery.relaunchToFirstDecisionMs} ms identical=${r.recovery.identical}`}`);
    }
  }
} finally {
  await server.close();
}
