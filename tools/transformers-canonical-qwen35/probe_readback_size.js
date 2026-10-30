import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const configBytes = await fs.readFile(process.argv[2]), config = JSON.parse(configBytes);
const fixtureBytes = await fs.readFile(path.join(config.fixtureDir, 'fixtures.json')), fixtures = JSON.parse(fixtureBytes);
const here = path.dirname(fileURLToPath(import.meta.url));
const sha = value => createHash('sha256').update(value).digest('hex');
await fs.mkdir(config.outputDir, { recursive: false });
for (const test of fixtures.cases) {
  if (sha(await fs.readFile(path.join(config.fixtureDir, test.file))) !== test.graphSha256) throw new Error('Synthetic fixture bytes changed');
}
const identity = { configSha256: sha(configBytes), fixturesSha256: sha(fixtureBytes),
  files: await Promise.all([fileURLToPath(import.meta.url), path.join(here, 'observe_gpu.js'),
    path.join(config.ortDist, 'ort.webgpu.min.mjs'), path.join(config.ortDist, 'ort-wasm-simd-threaded.jsep.wasm')]
    .map(async file => ({ path: file, sha256: sha(await fs.readFile(file)) }))) };
const logs = [];
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') { response.writeHead(200, { 'Content-Type': 'text/html' }); response.end('bounded observation micro'); return; }
    const [prefix, root] = pathname.startsWith('/ort/') ? ['/ort/', config.ortDist] :
      pathname.startsWith('/adapter/') ? ['/adapter/', here] : ['/fixtures/', config.fixtureDir];
    const file = path.resolve(root, decodeURIComponent(pathname.slice(prefix.length)));
    if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error('outside served fixture root');
    const stat = await fs.stat(file), extension = path.extname(file);
    response.writeHead(200, { 'Content-Length': stat.size, 'Content-Type': ['.js', '.mjs'].includes(extension) ?
      'text/javascript' : extension === '.wasm' ? 'application/wasm' : 'application/octet-stream' });
    createReadStream(file).on('error', error => response.destroy(error)).pipe(response);
  } catch (error) { response.writeHead(500); response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let context, timer;
try {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, 'browser-profile'), {
    executablePath: config.browserExecutable, args: config.browserArgs, headless: true,
    timeout: fixtures.policy.operationTimeoutMs });
  const page = await context.newPage();
  page.on('console', message => logs.push(message.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { timeout: fixtures.policy.operationTimeoutMs });
  const result = await Promise.race([page.evaluate(async ({ config, fixtures }) => {
    const ort = await import('/ort/ort.webgpu.min.mjs');
    const { readGpuTensor } = await import('/adapter/observe_gpu.js');
    ort.env.wasm.wasmPaths = '/ort/'; ort.env.wasm.numThreads = 1;
    const policy = fixtures.policy, attempts = [], devices = new Set();
    const bounded = async (operation, phase) => {
      let timer;
      try { return await Promise.race([operation, new Promise((resolve, reject) => {
        timer = setTimeout(() => reject(new Error('size micro timed out at ' + phase)), policy.operationTimeoutMs);
      })]); } finally { clearTimeout(timer); }
    };
    try {
      for (const test of fixtures.cases) {
        let session, outputs, input;
        const attempt = { bytes: test.bytes, expectedSha256: test.expectedSha256, phases: [], gpuErrors: [] };
        attempts.push(attempt);
        const phase = name => { attempt.phases.push(name); console.log('size micro ' + test.bytes + ' ' + name); };
        try {
          phase('create-session');
          session = await bounded(ort.InferenceSession.create('/fixtures/' + test.file, {
            executionProviders: ['webgpu'], graphOptimizationLevel: 'disabled',
            preferredOutputLocation: { selected: 'gpu-buffer', observed: 'gpu-buffer' },
            extra: { session: { disable_cpu_ep_fallback: '1' } } }), 'create-session');
          const device = await ort.env.webgpu.device;
          devices.add(device);
          if (!device.adapterInfo.vendor.toLowerCase().includes(config.requiredVendor)) throw new Error('wrong actual ORT adapter');
          attempt.adapter = { vendor: device.adapterInfo.vendor, architecture: device.adapterInfo.architecture };
          device.addEventListener('uncapturederror', event => attempt.gpuErrors.push(event.error.message));
          input = new ort.Tensor('int64', BigInt64Array.from([0n]), [1]);
          phase('run-gather-and-identity');
          outputs = await bounded(session.run({ ids: input }), 'run-gather-and-identity');
          attempt.sourceLocation = outputs.observed.location;
          phase('read-exact-helper');
          const observed = await readGpuTensor(device, outputs.observed, test.bytes, policy.chunkBytes,
            policy.operationTimeoutMs, policy.maxAllocationBytes);
          phase('browser-sha256');
          const digest = await bounded(crypto.subtle.digest('SHA-256', observed), 'browser-sha256');
          attempt.observedSha256 = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
          attempt.locationAfterRead = outputs.observed.location;
          attempt.pass = attempt.observedSha256 === test.expectedSha256 && outputs.observed.location === 'gpu-buffer' && !attempt.gpuErrors.length;
          phase('complete');
        } catch (error) { attempt.error = String(error); attempt.pass = false; }
        finally {
          for (const tensor of [...Object.values(outputs ?? {}), input].filter(Boolean)) {
            try { tensor.dispose(); } catch (error) { attempt.cleanupError = String(error); }
          }
          if (session) try { await bounded(session.release(), 'session-release'); } catch (error) { attempt.cleanupError = String(error); }
        }
      }
    } finally { for (const device of devices) device.destroy(); }
    return { schema: 'doppler.ort-sized-initializer-readback-micro.v1', qualification: false,
      scope: fixtures.policy.scope, attempts, pass: attempts.every(attempt => attempt.pass) };
  }, { config, fixtures }), new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error('Bounded observation micro overall timeout')), fixtures.policy.overallTimeoutMs);
  })]);
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify({ ...result, identity }, null, 2));
  if (!result.pass) process.exitCode = 1;
} catch (error) {
  await fs.writeFile(path.join(config.outputDir, 'failure.json'), JSON.stringify({ scope: fixtures.policy.scope,
    pass: false, error: String(error), identity }, null, 2));
  throw error;
} finally {
  clearTimeout(timer); await context?.close(); await new Promise(resolve => server.close(resolve));
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
