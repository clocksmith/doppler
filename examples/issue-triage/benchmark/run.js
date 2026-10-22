// Decision benchmark orchestrator (contract: reports/decision-benchmark-20261022/contract.json).
// Serves the application, the benchmark page, issue data and model files from one local origin,
// launches one browser per run with a fresh profile, samples memory over the browser's process
// tree, and stores the page's report together with the bytes it fetched.
//   node run.js <config.json>
// config: { outDir, csvDirectory, capsuleDirectory, descriptorPath, f16ModelDirectory,
//           hf: { "<repo id>": "<local dir>" }, browsers: [{ name, kind: chrome|firefox|manual,
//           executable?, command? }], lanes: [...], repetitions, quality (default true: the first
//           repetition decides every test and train issue), cpuFallback: bool, laneTimeoutMs }
import fs from 'node:fs/promises';
import { createReadStream, statSync, readFileSync, readdirSync, mkdtempSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadSplit, stratifiedSample } from '../dataset.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, '..');
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const outDir = path.resolve(config.outDir);
await fs.mkdir(outDir, { recursive: true });

const seed = 'issue-triage-20261021';
const readCsv = (split) => fs.readFile(path.join(config.csvDirectory, `issues_${split}.csv`), 'utf8');
const strip = ({ index, repo, label, title, body }) => ({ index, repo, label, title, body });
const train = stratifiedSample(await loadSplit('train', { readLocal: readCsv }), { split: 'train', perCell: 6, seed }).map(strip);
const test = stratifiedSample(await loadSplit('test', { readLocal: readCsv }), { split: 'test', perCell: 10, seed }).map(strip);
const descriptor = JSON.parse(await fs.readFile(config.descriptorPath, 'utf8'));
const runConfig = JSON.stringify({ descriptor, test, train, firstIssue: train[0] });

const mounts = [
  { prefix: '/app', dir: appRoot },
  { prefix: '/capsule', dir: path.resolve(config.capsuleDirectory), model: true },
  { prefix: '/models/doppler-f16', dir: path.resolve(config.f16ModelDirectory), model: true },
  ...Object.entries(config.hf).map(([repo, dir]) => ({ prefix: `/hf/${repo}/resolve/main`, dir: path.resolve(dir), model: true })),
].sort((a, b) => b.prefix.length - a.prefix.length);
const served = { modelBytes: 0, modelRequests: 0, missing: [] };
const pending = new Map();
const progress = new Map();
const TYPES = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.html': 'text/html',
  '.wasm': 'application/wasm', '.wgsl': 'text/plain' };
const httpServer = http.createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://local').pathname);
  if (pathname === '/api/run-config') {
    response.writeHead(200, { 'Content-Type': 'application/json' }).end(runConfig);
    return;
  }
  if (pathname.startsWith('/api/progress/') && request.method === 'POST') {
    let body = '';
    request.on('data', (chunk) => { body += chunk; });
    request.on('end', () => {
      response.writeHead(204).end();
      progress.set(decodeURIComponent(pathname.slice('/api/progress/'.length)), body.slice(0, 200));
    });
    return;
  }
  if (pathname.startsWith('/api/result/') && request.method === 'POST') {
    const chunks = [];
    request.on('data', (chunk) => chunks.push(chunk));
    request.on('end', () => {
      response.writeHead(204).end();
      pending.get(decodeURIComponent(pathname.slice('/api/result/'.length)))?.(JSON.parse(Buffer.concat(chunks).toString('utf8')));
    });
    return;
  }
  const mount = mounts.find((m) => pathname.startsWith(`${m.prefix}/`));
  const file = mount && path.resolve(mount.dir, `.${pathname.slice(mount.prefix.length)}`);
  let stat = null;
  try { if (file && file.startsWith(mount.dir + path.sep)) stat = statSync(file); } catch { /* missing */ }
  if (!stat?.isFile()) {
    if (served.missing.length < 50) served.missing.push(pathname);
    response.writeHead(404).end();
    return;
  }
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
const baseUrl = `http://127.0.0.1:${httpServer.address().port}`;

// Process tree memory. Linux reads /proc (RSS and amdgpu GTT+VRAM per process);
// macOS reads ps (RSS only).
function processTable() {
  if (process.platform === 'linux') {
    const rows = [];
    for (const pid of readdirSync('/proc').filter((p) => /^\d+$/.test(p))) {
      try {
        const status = readFileSync(`/proc/${pid}/status`, 'utf8');
        const ppid = Number(/PPid:\s+(\d+)/.exec(status)[1]);
        const rss = Number(/VmRSS:\s+(\d+)/.exec(status)?.[1] ?? 0) * 1024;
        let gpu = 0;
        for (const fd of readdirSync(`/proc/${pid}/fdinfo`)) {
          const info = readFileSync(`/proc/${pid}/fdinfo/${fd}`, 'utf8');
          if (!info.includes('drm-driver:\tamdgpu')) continue;
          gpu += Number(/drm-memory-gtt:\s+(\d+)/.exec(info)?.[1] ?? 0) * 1024;
          gpu += Number(/drm-memory-vram:\s+(\d+)/.exec(info)?.[1] ?? 0) * 1024;
        }
        rows.push({ pid: Number(pid), ppid, rss, gpu });
      } catch { /* exited or not ours */ }
    }
    return rows;
  }
  return execFileSync('ps', ['-A', '-o', 'pid=,ppid=,rss='], { encoding: 'utf8' }).trim().split('\n')
    .map((line) => line.trim().split(/\s+/).map(Number)).map(([pid, ppid, rss]) => ({ pid, ppid, rss: rss * 1024, gpu: null }));
}
function tree(rootPid, rows) {
  const ids = new Set([rootPid]);
  for (let grew = true; grew;) {
    grew = false;
    for (const row of rows) if (!ids.has(row.pid) && ids.has(row.ppid)) { ids.add(row.pid); grew = true; }
  }
  return rows.filter((row) => ids.has(row.pid));
}
function startSampler(rootPid) {
  const peak = { totalRss: 0, gpuMemory: process.platform === 'linux' ? 0 : null, processes: 0 };
  const timer = setInterval(() => {
    const rows = tree(rootPid, processTable());
    peak.totalRss = Math.max(peak.totalRss, rows.reduce((a, r) => a + r.rss, 0));
    if (peak.gpuMemory != null) peak.gpuMemory = Math.max(peak.gpuMemory, rows.reduce((a, r) => a + r.gpu, 0));
    peak.processes = Math.max(peak.processes, rows.length);
  }, 200);
  return { stop: () => { clearInterval(timer); return peak; }, pids: () => tree(rootPid, processTable()).map((r) => r.pid) };
}

const CHROME_ARGS = ['--headless=new', '--no-first-run', '--no-default-browser-check', '--enable-unsafe-webgpu',
  '--ignore-gpu-blocklist', ...(process.platform === 'linux' ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] : [])];
const FIREFOX_PREFS = ['user_pref("dom.webgpu.enabled", true);', 'user_pref("gfx.webgpu.ignore-blocklist", true);',
  'user_pref("browser.shell.checkDefaultBrowser", false);', 'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
  'user_pref("browser.aboutwelcome.enabled", false);'];

function browserVersion(browser) {
  if (browser.kind === 'manual') return browser.version ?? null;
  try { return execFileSync(browser.executable, ['--version'], { encoding: 'utf8' }).trim(); } catch (error) { return `unknown: ${error.message}`; }
}

async function launch(browser, url, profileDir) {
  if (browser.kind === 'chrome') {
    return spawn(browser.executable, [...CHROME_ARGS, `--user-data-dir=${profileDir}`, url], { stdio: 'ignore' });
  }
  if (browser.kind === 'firefox') {
    await fs.writeFile(path.join(profileDir, 'user.js'), `${FIREFOX_PREFS.join('\n')}\n`);
    return spawn(browser.executable, [...(browser.headless === false ? [] : ['-headless']), '-no-remote', '-profile', profileDir, url], { stdio: 'ignore' });
  }
  // manual: a command template such as ["open", "-a", "Safari", "{url}"], or print the URL.
  if (browser.command) return spawn(browser.command[0], browser.command.slice(1).map((a) => a.replace('{url}', url)), { stdio: 'ignore' });
  console.log(`open ${url}`);
  return null;
}

function killTree(sampler, child) {
  for (const pid of sampler?.pids() ?? []) { try { process.kill(pid, 'SIGKILL'); } catch { /* gone */ } }
  try { child?.kill('SIGKILL'); } catch { /* gone */ }
}

async function runOne(browser, lane, mode, rep, quality) {
  const runId = `${browser.name}-${lane}-${mode}-${rep}`;
  const profileDir = mkdtempSync(path.join(os.tmpdir(), 'decision-bench-'));
  const startBytes = served.modelBytes;
  const startRequests = served.modelRequests;
  served.missing = [];
  const url = `${baseUrl}/app/benchmark/bench.html?lane=${encodeURIComponent(lane)}&run=${encodeURIComponent(runId)}&mode=${mode}&quality=${quality ? 1 : 0}`;
  const result = new Promise((resolve) => pending.set(runId, resolve));
  const launchedAt = Date.now();
  const child = await launch(browser, url, profileDir);
  const sampler = child ? startSampler(child.pid) : null;
  let timer;
  const timeout = new Promise((resolve) => { timer = setTimeout(() => resolve({ error: `timeout after ${config.laneTimeoutMs} ms` }), config.laneTimeoutMs); });
  const report = await Promise.race([result, timeout]);
  clearTimeout(timer);
  if (report.error?.startsWith('timeout')) report.lastProgress = progress.get(runId) ?? null;
  const peakMemory = sampler?.stop() ?? null;
  killTree(sampler, child);
  pending.delete(runId);
  await new Promise((resolve) => setTimeout(resolve, 500));
  await fs.rm(profileDir, { recursive: true, force: true });
  const record = {
    runId, browser: { name: browser.name, kind: browser.kind, version: browserVersion(browser) }, lane, mode, rep, quality,
    launchToReportMs: Date.now() - launchedAt,
    transfer: { modelBytes: served.modelBytes - startBytes, modelRequests: served.modelRequests - startRequests, missing: served.missing },
    peakMemory, report,
  };
  const dir = path.join(outDir, browser.name);
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, `${lane}-${mode}-${rep}.json`), `${JSON.stringify(record, null, 1)}\n`);
  const test = report.test ?? [];
  const correct = test.filter((d) => d.choice === d.label).length;
  console.log(`${runId}: ${report.error ? `ERROR ${report.error.split('\n')[0]}` : `cold ${report.coldCompleteMs?.toFixed(0)} ms, open ${report.openMs?.toFixed(0)} ms, `
    + `first ${report.firstDecisionMs?.toFixed(0)} ms, test ${correct}/${test.length}, runtime ${JSON.stringify(report.runtime)}`}, `
    + `RSS ${((peakMemory?.totalRss ?? 0) / 2 ** 30).toFixed(2)} GiB, bytes ${(record.transfer.modelBytes / 1e6).toFixed(0)} MB`);
  return record;
}

const platform = { os: `${os.type()} ${os.release()}`, arch: os.arch(), cpu: os.cpus()[0]?.model ?? null, node: process.version, hostname: os.hostname() };
await fs.writeFile(path.join(outDir, 'platform.json'), `${JSON.stringify({ platform, browsers: config.browsers.map((b) => ({ ...b, version: browserVersion(b) })) }, null, 1)}\n`);
try {
  for (const browser of config.browsers) {
    for (let rep = 0; rep < config.repetitions; rep += 1) {
      const order = rep % 2 === 0 ? config.lanes : [...config.lanes].reverse();
      for (const lane of order) await runOne(browser, lane, 'cold', rep, rep === 0 && config.quality !== false);
    }
    if (config.cpuFallback) for (const lane of config.lanes) await runOne(browser, lane, 'cpu', 0, false);
  }
} finally {
  httpServer.closeAllConnections();
  httpServer.close();
}
