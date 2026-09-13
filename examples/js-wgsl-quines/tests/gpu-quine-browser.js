// Real browser compilation and readback, with actual GPU descendants as inputs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const backend = process.env.GPU_QUINE_BACKEND ?? 'default';
assert.ok(['default', 'swiftshader'].includes(backend), 'GPU_QUINE_BACKEND must be default or swiftshader');
const generations = 8;
const started = new Date().toISOString();
const evidence = path.join(root, 'verification/gpu-quine', `${started.replace(/[:.]/g, '-')}-${backend}`);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'gpu-quine-chrome-'));
const sourcePath = path.join(root, 'quines/gpu-quine.js');
const original = fs.readFileSync(sourcePath);
const probe = fs.readFileSync(new URL('./gpu-quine-probe.js', import.meta.url), 'utf8');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const report = {
  schema: 'gpu-quine-browser/v1', started, backend, requestedGenerations: generations,
  status: 'running', source: { path: 'quines/gpu-quine.js', bytes: original.length, sha256: hash(original) },
  harness: { sha256: hash(fs.readFileSync(fileURLToPath(import.meta.url))), probeSha256: hash(probe) },
  scope: 'Real browser WebGPU compilation, dispatch, and mapped bytes; no GPU mocks or CPU reproduction.',
  instrumentation: 'Fresh worker prepends an observer that delegates native calls; descendant bytes are unchanged.',
  consoleContract: 'The captured console string plus one LF must equal the complete mapped source bytes.',
  environment: { node: process.version, platform: process.platform, arch: process.arch },
  generations: [],
};
fs.mkdirSync(evidence, { recursive: true });
const server = http.createServer((request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const routes = { '/quines/index.html': ['quines/index.html', 'text/html'],
      '/quines/gpu-quine.js': ['quines/gpu-quine.js', 'text/javascript'] };
    const route = routes[pathname];
    if (!route) { response.writeHead(404); response.end(); return; }
    const bytes = fs.readFileSync(path.join(root, route[0]));
    response.writeHead(200, { 'Content-Type': route[1], 'Cache-Control': 'no-store' });
    response.end(bytes);
  } catch {
    response.writeHead(500);
    response.end();
  }
});
let chrome;
let socket;
let chromeError;
let diagnostics = '';
let sequence = 0;
const pending = new Map();
const consoleEvents = [];
const pageErrors = [];
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`CDP timed out: ${method}`));
    }, 120000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

try {
  assert.equal(original.at(-1), 10, 'The supplied program must retain its final newline');
  assert.ok(original.every(byte => byte < 128), 'This quine is ASCII, not a Unicode byte codec');
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const executable = process.env.CHROME_BIN ?? (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : '/usr/bin/google-chrome');
  const flags = ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run',
    '--no-default-browser-check', '--disable-background-networking', '--enable-unsafe-webgpu',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`];
  if (backend === 'swiftshader') flags.push('--enable-unsafe-swiftshader', '--use-angle=vulkan',
    '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader',
    '--disable-vulkan-surface');
  report.launch = { executable, flags: flags.filter(flag => !flag.startsWith('--user-data-dir=')) };
  chrome = spawn(executable, [...flags, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  chrome.on('error', error => { chromeError = error; });
  chrome.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-16384); });
  const activePort = path.join(profile, 'DevToolsActivePort');
  for (let i = 0; i < 100 && !fs.existsSync(activePort) && !chromeError; i += 1) await delay(100);
  if (chromeError) throw chromeError;
  if (!fs.existsSync(activePort)) throw new Error(`Chrome did not expose CDP: ${diagnostics}`);
  const port = Number(fs.readFileSync(activePort, 'utf8').split('\n')[0]);
  const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  delete version.webSocketDebuggerUrl;
  report.browser = version;
  const page = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.consoleAPICalled') consoleEvents.push(message.params);
    if (message.method === 'Runtime.exceptionThrown') pageErrors.push(message.params.exceptionDetails);
    const call = pending.get(message.id);
    if (!call) return;
    pending.delete(message.id);
    clearTimeout(call.timer);
    if (message.error) call.reject(new Error(JSON.stringify(message.error)));
    else call.resolve(message.result);
  });
  socket.addEventListener('close', () => {
    for (const call of pending.values()) {
      clearTimeout(call.timer);
      call.reject(new Error('Chrome debugging socket closed'));
    }
    pending.clear();
  });
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Page.navigate', { url: `${origin}/quines/index.html` });
  for (let i = 0; i < 600 && !consoleEvents.length && !pageErrors.length; i += 1) await delay(100);
  assert.deepEqual(pageErrors, [], 'The standalone launch page threw an exception');
  const launchError = consoleEvents.find(event => event.type === 'error');
  assert.ok(!launchError, JSON.stringify(launchError));
  const printed = consoleEvents.find(event => event.type === 'log')?.args[0]?.value;
  assert.equal(typeof printed, 'string', 'The uninstrumented launch page did not print a generation');
  assert.deepEqual(Buffer.from(printed + '\n'), original, 'The uninstrumented page changed source bytes');
  report.launchPage = { status: 'passed', outputSha256: hash(printed + '\n') };

  let current = original.toString('utf8');
  for (let generation = 1; generation <= generations; generation += 1) {
    const result = await send('Runtime.evaluate', { awaitPromise: true, returnByValue: true, expression: `
      new Promise((resolve) => {
        const url = URL.createObjectURL(new Blob([${JSON.stringify(probe)}, '\\n', ${JSON.stringify(current)}],
          { type: 'text/javascript' }));
        const worker = new Worker(url, { type: 'module' });
        let timer;
        const finish = value => {
          clearTimeout(timer);
          worker.terminate();
          URL.revokeObjectURL(url);
          resolve(value);
        };
        timer = setTimeout(() => finish({ status: 'failed', errors: ['GPU generation timed out'] }), 60000);
        worker.onmessage = event => finish(event.data);
        worker.onerror = event => finish({ status: 'failed', errors: [event.message] });
      })
    ` });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    const row = result.result.value;
    assert.ok(row, 'Worker returned no observation');
    // Keep failed observations before assertions, never replace them with a reference result.
    const prefix = `generation-${String(generation).padStart(2, '0')}`;
    fs.writeFileSync(path.join(evidence, `${prefix}-observation.json`), JSON.stringify(row, null, 2) + '\n');
    assert.equal(row.status, 'completed', JSON.stringify(row.errors));
    assert.deepEqual(row.errors, []);
    assert.equal(row.pipelines, 1, 'A real compute pipeline must finish creation');
    assert.deepEqual(row.dispatches, [[1]], 'The shader must dispatch exactly once');
    assert.equal(row.submissions, 1);
    assert.equal(row.mappings, 1);
    assert.equal(row.destroyedBuffers, 2);
    assert.equal(row.destroyedDevices, 1);
    assert.ok(row.compilation.every(message => message.type !== 'error'));
    assert.ok(Array.isArray(row.mappedBytes) && row.mappedBytes.every(byte => Number.isInteger(byte) && byte >= 0 && byte < 128));
    const bytes = Buffer.from(row.mappedBytes);
    fs.writeFileSync(path.join(evidence, `${prefix}.js`), bytes);
    assert.deepEqual(bytes, original, `GPU generation ${generation} changed the exact source bytes`);
    assert.equal(typeof row.printed, 'string');
    assert.deepEqual(Buffer.from(row.printed + '\n'), bytes, 'Console output disagrees with actual mapped GPU bytes');
    const adapter = row.adapter;
    assert.ok(adapter, 'The actual execution adapter must be recorded');
    const software = backend === 'swiftshader' || adapter.isFallbackAdapter === true
      || /swiftshader|llvmpipe|lavapipe|software/i.test(JSON.stringify(adapter));
    const adapterClass = software ? 'software' : adapter.isFallbackAdapter === false ? 'hardware-reported' : 'unknown';
    report.generations.push({ generation, status: 'passed', inputSha256: hash(current),
      outputSha256: hash(bytes), shaderSha256: hash(row.shader), bytes: bytes.length,
      adapter, adapterClass, compilation: row.compilation, artifact: `${prefix}.js` });
    current = bytes.toString('utf8');
    console.log(`PASS GPU quine generation ${generation}: ${bytes.length} exact bytes (${adapterClass})`);
  }
  report.status = 'passed';
} catch (error) {
  report.status = 'failed';
  report.error = error.stack;
  report.chromeDiagnostics = diagnostics;
  report.pageErrors = pageErrors;
  report.consoleErrors = consoleEvents.filter(event => event.type === 'error');
  console.error(error);
  process.exitCode = 1;
} finally {
  report.finished = new Date().toISOString();
  fs.writeFileSync(path.join(evidence, 'browser.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(`GPU quine receipt: ${path.relative(root, path.join(evidence, 'browser.json'))}`);
  for (const call of pending.values()) clearTimeout(call.timer);
  socket?.close();
  if (chrome && chrome.exitCode === null && !chromeError) {
    await new Promise(resolve => {
      const timer = setTimeout(() => { chrome.kill('SIGKILL'); resolve(); }, 2000);
      chrome.once('exit', () => { clearTimeout(timer); resolve(); });
      chrome.kill('SIGTERM');
    });
  }
  server.closeAllConnections();
  if (server.listening) await new Promise(resolve => server.close(resolve));
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 4, retryDelay: 250 });
}
