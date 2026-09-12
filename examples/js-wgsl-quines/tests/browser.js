/** Real Chrome WebGPU evidence, without a Node GPU shim or npm dependencies. */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const evidence = path.join(root, 'verification/sprout-three-backend-2026-09-12');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'sprout-chrome-'));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.wgsl': 'text/plain' };
const server = http.createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const filename = path.resolve(root, '.' + pathname);
    if (!filename.startsWith(root + path.sep)) throw new Error('Outside gallery');
    const body = fs.readFileSync(filename);
    response.writeHead(200, { 'Content-Type': mime[path.extname(filename)] ?? 'application/octet-stream' });
    response.end(body);
  } catch {
    response.writeHead(404);
    response.end();
  }
});
let chrome;
let socket;
let chromeError;
let diagnostics = '';
const pending = new Map();
let sequence = 0;
let onLoad;
const report = { schema: 'sprout-three-backend-browser-v1', started: new Date().toISOString() };
fs.mkdirSync(evidence, { recursive: true });

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
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const executable = process.env.CHROME_BIN ?? (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : '/usr/bin/google-chrome');
  chrome = spawn(executable, ['--headless=new', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'],
  { stdio: ['ignore', 'ignore', 'pipe'] });
  chrome.on('error', error => { chromeError = error; });
  chrome.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-8192); });
  const activePort = path.join(profile, 'DevToolsActivePort');
  for (let i = 0; i < 100 && !fs.existsSync(activePort) && !chromeError; i++) await delay(100);
  if (chromeError) throw chromeError;
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
    if (message.method === 'Page.loadEventFired') onLoad?.();
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
  const loaded = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Gallery page did not load')), 30000);
    onLoad = () => { clearTimeout(timer); resolve(); };
  });
  await send('Page.navigate', { url: `${origin}/demo.html` });
  await loaded;
  const result = await send('Runtime.evaluate', { awaitPromise: true, returnByValue: true, expression: `
    (async () => {
      const assert = (condition, message) => { if (!condition) throw new Error(message); };
      const sameBytes = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
      const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
        typeof bytes === 'string' ? new TextEncoder().encode(bytes) : bytes)),
        value => value.toString(16).padStart(2, '0')).join('');
      const load = source => { globalThis.__QUINE_LIBRARY__ = true; (0, eval)(source); return globalThis.Quine; };
      const source = await (await fetch('/quines/05-compiler.js')).text();
      const expectedShader = await (await fetch('/quines/05-compiler.wgsl')).text();
      const expectedBinary = new Uint8Array(await (await fetch('/quines/05-compiler.wasm')).arrayBuffer());
      const seed = load(source);
      const cycles = [];
      const checks = [];
      let current = source;
      let returnedShader;
      let returnedBinary;
      let native;
      let adapter;
      for (let generation = 0; generation < 3; generation++) {
        const js = load(current);
        const shader = js.wgsl();
        assert(shader === expectedShader, 'JS descendant changed its WGSL output');
        const self = await js.run('wgsl', null, shader);
        assert(self.text === shader, 'GPU WGSL self-reproduction changed bytes');
        const emitted = await js.run('wasm', null, self.text);
        assert(sameBytes(emitted.bytes, expectedBinary), 'GPU emitted different Wasm bytes');
        assert(!emitted.adapter.isFallbackAdapter, 'A software fallback is not physical GPU evidence');
        native = await js.fromWasm(emitted.bytes);
        assert(sameBytes(native.self(), emitted.bytes), 'Native Wasm self-reproduction changed bytes');
        assert(native.wgsl() === self.text, 'Wasm changed its WGSL descendant');
        const back = native.js();
        assert(back === current && back === source, 'Wasm did not close the exact JS cycle');
        assert(load(back).self() === back, 'Returned JS does not execute as a quine');
        current = back;
        returnedShader = self.text;
        returnedBinary = emitted.bytes;
        adapter = emitted.adapter;
        cycles.push({ generation: generation + 1, path: 'JS -> GPU WGSL -> Wasm -> JS',
          jsSha256: await hash(back), wgslSha256: await hash(self.text), wasmSha256: await hash(emitted.bytes),
          wasmBytes: emitted.bytes.length, adapter });
      }
      const examples = await (await fetch('/compiler/examples.json')).json();
      for (const { name, expected } of examples) {
        const program = await (await fetch('/compiler/' + name + '.json')).json();
        for (const compilerRuntime of ['js', 'gpu-wgsl', 'wasm']) {
          for (const target of ['js', 'wgsl', 'wasm']) {
            const compiled = compilerRuntime === 'gpu-wgsl'
              ? await seed.run(target, program, returnedShader)
              : { output: (compilerRuntime === 'js' ? seed : native).compile(program, target) };
            const output = compiled.output ?? compiled.bytes ?? compiled.text;
            const canonical = seed.compile(program, target);
            assert(target === 'wasm' ? sameBytes(output, canonical) : output === canonical,
              compilerRuntime + ' compiled different ' + target + ' for ' + name);
            let actual;
            if (target === 'js') actual = load(output).execute();
            if (target === 'wgsl') actual = (await seed.run('js', null, output)).text;
            if (target === 'wasm') actual = (await seed.fromWasm(output)).js();
            assert(actual === expected, compilerRuntime + ' -> ' + target + ' failed ' + name);
            checks.push({ program: name, compilerRuntime, targetRuntime: target === 'wgsl' ? 'gpu-wgsl' : target,
              expected, actual, status: 'passed' });
          }
        }
      }
      return { status: 'passed', adapter, cycles, checks, artifacts: {
        'gpu-returned.wgsl': returnedShader,
        'gpu-emitted.wasm': Array.from(returnedBinary),
        'wasm-returned.js': current
      } };
    })()
  ` });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  const value = result.result.value;
  if (!value || value.status !== 'passed') throw new Error('Browser returned no completed evidence');
  const { artifacts, ...outcome } = value;
  report.artifacts = {};
  for (const [name, content] of Object.entries(artifacts)) {
    const bytes = Buffer.from(content);
    fs.writeFileSync(path.join(evidence, name), bytes);
    report.artifacts[name] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
  }
  Object.assign(report, outcome);
  console.log(`${report.cycles.length} physical-GPU three-runtime cycles and ${report.checks.length} unrelated-program paths passed.`);
  console.log(JSON.stringify({ browser: report.browser.Browser, adapter: report.adapter }));
} catch (error) {
  report.status = 'failed';
  report.error = error.stack;
  report.chromeDiagnostics = diagnostics;
  console.error(error);
  process.exitCode = 1;
} finally {
  report.finished = new Date().toISOString();
  fs.writeFileSync(path.join(evidence, 'browser.json'), JSON.stringify(report, null, 2) + '\n');
  for (const call of pending.values()) clearTimeout(call.timer);
  socket?.close();
  chrome?.kill('SIGTERM');
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  await delay(400);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 4, retryDelay: 250 });
}
