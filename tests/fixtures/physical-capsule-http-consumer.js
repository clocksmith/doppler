import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import { once } from 'node:events';
import { createHash, randomBytes } from 'node:crypto';
import { createRequire } from 'node:module';
import { openCapsule } from 'doppler-gpu/host';
import { createCapsuleServeHandler } from 'doppler-gpu/serve';
import { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } from './node_modules/doppler-gpu/src/tooling/node-webgpu.js';
import { destroyDevice } from './node_modules/doppler-gpu/src/gpu/device.js';
import { evaluateRerankReference } from './node_modules/doppler-gpu/src/config/rerank-reference.js';

// Copy into an isolated installed consumer. Only explicit retained inputs are used.
const [configPath, outputPath] = process.argv.slice(2);
const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const digest = async filename => createHash('sha256').update(await fs.readFile(filename)).digest('hex');
const config = await read(configPath);
const capsule = await read(config.capsulePath);
const options = await read(config.openOptionsPath);
const reference = await read(config.referencePath);
const require = createRequire(import.meta.url);
const report = { schema: 'doppler.capsule-http-physical-probe/v1', passed: false,
  evidenceClass: 'internal-physical-installed-package', externalAdoption: false,
  startedAtUtc: new Date().toISOString(), nodeVersion: process.version,
  config: { path: configPath, sha256: await digest(configPath) },
  package: { path: config.packagePath, sha256: await digest(config.packagePath) },
  sourceReference: { path: config.referencePath, sha256: await digest(config.referencePath) },
  capsule: { path: config.capsulePath, sha256: await digest(config.capsulePath) },
  requests: [], stage: 'provider' };
let session;
let handler;
let server;
const originalFetch = globalThis.fetch;
try {
  assert.throws(() => require.resolve('doe-gpu/node-webgpu'), /Cannot find/);
  report.doeAbsent = true;
  const provider = await bootstrapNodeWebGPUProvider(config.provider.module, { createArgs: config.provider.createArgs });
  report.provider = provider.receipt;
  report.providerVersion = require('webgpu/package.json').version;
  const info = provider.session.adapter.info;
  report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, info[key]]));
  assert(!/swiftshader|llvmpipe|software/i.test(JSON.stringify(report.hardware)));
  assert.match(JSON.stringify(report.hardware), /AMD|amd/);
  globalThis.fetch = async input => {
    report.requests.push(String(input));
    throw new Error('Artifact network access is disabled; retained files are required.');
  };
  report.stage = 'capsule-open';
  report.loadingObservations = [];
  report.memoryBeforeLoad = process.memoryUsage();
  report.resourceUsageBeforeLoad = process.resourceUsage();
  report.measurementScope = { cache: 'fresh process; retained artifacts; operating-system and driver caches not reset',
    loadSamples: 1, directRuns: 1, httpRuns: 1, warmupRuns: 0,
    memory: 'Node process accounting, including native allocations; not dedicated GPU residency',
    performanceClaim: false };
  const before = performance.now();
  session = await openCapsule(config.capsulePath, { ...options, observer: { observe(event) {
    report.loadingObservations.push({ elapsedMs: performance.now() - before, event });
  } } });
  report.loadMs = performance.now() - before;
  report.memoryAfterLoad = process.memoryUsage();
  report.resourceUsageAfterLoad = process.resourceUsage();
  report.identity = session.capsuleIdentity;
  report.targetPlanDigest = session.selectedTargetPlanDigest;
  const request = { ...config.operation, input: { ...reference.input, application: capsule.release.application },
    limits: { ...config.operation.limits, deadlineAt: Date.now() + config.jobDurationMs } };
  report.request = request;
  report.stage = 'direct-operation';
  const directStarted = performance.now();
  report.direct = [];
  for await (const event of session.executeOperation(request)) report.direct.push(event);
  report.directElapsedMs = performance.now() - directStarted;
  report.stage = 'http-operation';
  const token = randomBytes(32).toString('hex');
  handler = createCapsuleServeHandler({ session, policy: config.servingPolicy, token });
  server = http.createServer(handler);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const url = `http://127.0.0.1:${server.address().port}/v1/operations`;
  const httpStarted = performance.now();
  const response = await originalFetch(url, { method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify(request) });
  report.httpStatus = response.status;
  report.httpBody = await response.text();
  report.httpElapsedMs = performance.now() - httpStarted;
  assert.equal(response.status, 200);
  report.served = report.httpBody.trim().split('\n').map(line => JSON.parse(line));
  report.comparisons = [];
  for (const events of [report.direct, report.served]) {
    assert.equal(events.at(-1).status, 'completed');
    const receipt = events.at(-1).output;
    const observation = { input: { query: receipt.evidence.query, documents: receipt.evidence.documents },
      scoringConfig: session.manifest.inference.rerank, outputs: receipt.evidence.scores };
    const comparison = evaluateRerankReference(reference, observation);
    report.comparisons.push(comparison);
    assert.equal(comparison.passed, true, 'unchanged source reference');
    assert.equal(events.at(-1).receipt.targetPlanDigest, report.targetPlanDigest);
  }
  // Observation timings can differ; the actual outputs must retain exact parity.
  assert.deepEqual(report.served.at(-1).output.evidence.scores, report.direct.at(-1).output.evidence.scores);
  assert.deepEqual(report.served.at(-1).output.evidence.ranking, report.direct.at(-1).output.evidence.ranking);
  assert.equal(report.served.at(-1).requestHash, report.direct.at(-1).requestHash);
  assert.equal(report.requests.length, 0);
  report.passed = true;
  report.stage = 'complete';
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  const errors = [];
  for (const close of [() => handler?.close(), () => server && new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve())),
    () => session?.close(), () => destroyDevice(), () => releaseNodeWebGPU()]) {
    try { await close(); } catch (error) { errors.push(error.message); }
  }
  globalThis.fetch = originalFetch;
  report.memoryAfterCleanup = process.memoryUsage();
  report.resourceUsageAfterCleanup = process.resourceUsage();
  report.cleanup = { passed: errors.length === 0, errors };
  report.passed &&= report.cleanup.passed;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(outputPath, JSON.stringify(report, null, 2), { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error?.message, outputPath }));
if (!report.passed) process.exitCode = 1;
