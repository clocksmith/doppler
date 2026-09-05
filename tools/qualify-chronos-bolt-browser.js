import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const model = path.resolve(process.argv[2] ?? '');
const sealed = process.argv.includes('--sealed');
const started = new Date().toISOString();
const server = http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Doppler forecast qualification</title>'); return; }
    const root = pathname.startsWith('/model/') ? model : repo;
    const relative = pathname.startsWith('/model/') ? pathname.slice(7) : pathname.slice(1);
    const filename = path.resolve(root, relative);
    if (!filename.startsWith(root + path.sep)) throw new Error('Outside qualification roots');
    response.setHeader('Content-Type', filename.endsWith('.js') ? 'text/javascript' : filename.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    response.setHeader('Cache-Control', 'no-store');
    response.end(await fs.readFile(filename));
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
const consoleMessages = [];
let report;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  page.on('console', message => consoleMessages.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => consoleMessages.push({ type: 'pageerror', text: String(error) }));
  await page.goto(origin);
  report = await page.evaluate(async (sealed) => {
    const { createForecastProgramFactory } = await import('/src/inference/pipelines/forecast/pack-program.js');
    const { computeCanonicalSha256, hashBytesSha256 } = await import('/src/formats/canonical-hash.js');
    const candidate = await (await fetch('/model/candidate.json')).json();
    const reference = await (await fetch('/model/source/reference.json')).json();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('Browser has no WebGPU adapter');
    const adapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, adapter.info[k] ?? null]));
    const device = await adapter.requestDevice();
    const rows = [];
    let failure = null;
    let program;
    let profile;
    let checkpoint = { sequence: 0, digest: null };
    try {
      const artifactStore = { async readArtifact(artifact) {
        const bytes = new Uint8Array(await (await fetch('/model/' + artifact.path)).arrayBuffer());
        if (hashBytesSha256(bytes) !== artifact.hash || bytes.length !== artifact.sizeBytes) throw new Error('Browser artifact custody mismatch');
        return bytes;
      } };
      if (sealed) {
        const { openPack } = await import('/src/pack-runtime.js');
        const pack = await (await fetch('/model/pack.json')).json();
        profile = await (await fetch('/model/trust-profile.json')).json();
        const releaseEvents = await (await fetch('/model/release-events.json')).json();
        program = await openPack(pack, { artifactStore, trustedSigners: profile.trustedSigners,
          device: { getDevice: () => device, getProfile: () => ({ surface: 'browser-webgpu', hasF16: device.features.has('shader-f16'),
            hasSubgroups: device.features.has('subgroups'), maxBufferSize: device.limits.maxBufferSize }) },
          programFactory: createForecastProgramFactory(device), session: { releaseEvents,
            releaseTrustedSigners: profile.trustedSigners, acceptedTargetPlanDigests: profile.acceptedTargetPlanDigests,
            releasePolicy: { now: new Date().toISOString(), minimumSequence: profile.minimumSequence, checkpoint },
            persistReleaseCheckpoint: value => {
              localStorage.setItem('doppler-release-checkpoint', JSON.stringify(value)); checkpoint = value;
            } } });
      } else program = await createForecastProgramFactory(device)({ pack: candidate, targetPlan: candidate.targetPlan, artifactStore });
      for (const testCase of reference.cases) {
        const start = performance.now();
        const request = { context: testCase.context, horizon: testCase.horizon };
        const output = sealed ? await program.forecast({ ...request, application: profile.application, assignmentHash: null })
          : await program.forecast(request, {});
        let maxAbsoluteError = 0;
        let mismatches = 0;
        output.values.forEach((value, i) => {
          const difference = Math.abs(value - testCase.values[i]);
          maxAbsoluteError = Math.max(maxAbsoluteError, difference);
          if (difference > reference.tolerance.absolute + reference.tolerance.relative * Math.abs(testCase.values[i])) mismatches++;
        });
        rows.push({ id: testCase.id, elapsedMs: performance.now() - start, maxAbsoluteError, mismatches, output });
      }
      if (rows.some(row => row.mismatches > 0)) throw new Error('Browser forecasting reference parity failed');
      if (/swiftshader|llvmpipe|software/i.test(Object.values(adapterInfo).join(' '))) throw new Error('Software adapter is not physical hardware evidence');
    } catch (error) { failure = String(error?.stack ?? error); }
    finally { try { await program?.close(); } finally { device.destroy(); } }
    return { schema: 'doppler.forecast-qualification/v1', status: failure ? 'failed' : 'passed', surface: 'browser-webgpu',
      boundary: sealed ? 'signed-pack-session' : 'candidate-program', packIdentity: program?.packIdentity ?? null, checkpoint,
      adapter: adapterInfo, userAgent: navigator.userAgent, candidateHash: computeCanonicalSha256(candidate),
      executionGraphHash: candidate.program.executionGraphHash, artifactClosureHash: computeCanonicalSha256(candidate.artifacts),
      referenceHash: computeCanonicalSha256(reference), tolerance: reference.tolerance, rows, failure };
  }, sealed);
} catch (error) {
  report = { schema: 'doppler.forecast-qualification/v1', surface: 'browser-webgpu', status: 'failed', failure: String(error?.stack ?? error) };
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
  const filename = `qualification-${sealed ? 'signed-' : ''}browser-${started.replace(/[:.]/g, '-')}.json`;
  await fs.writeFile(path.join(model, filename), JSON.stringify({ ...report, startedAt: started, completedAt: new Date().toISOString(), consoleMessages }));
  console.log(JSON.stringify({ report: path.join(model, filename), status: report.status, failure: report.failure,
    rows: report.rows?.map(({ output, ...row }) => row), adapter: report.adapter }));
  if (report.status !== 'passed') process.exitCode = 1;
}
