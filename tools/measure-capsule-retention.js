#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { createRetainedReleaseCheckpoint } from './retained-release-checkpoint.js';
import { closeInstalledCapsule } from './close-installed-capsule.js';

// One fresh process per measurement. This adapter changes only the declared
// verified-artifact retention policy; model programs and reference tolerances stay fixed.
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const config = await read(process.argv[2]);
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.capsule-retention-measurement/v1', passed: false,
  config, startedAtUtc: new Date().toISOString(), runtime: { name: 'node', version: process.version },
  physicalExecution: false, stage: 'verify-inputs' };
let session, release, destroyDevice, restoreFetch, checkpointStore;
try {
  for (const input of [config.reference, config.candidate]) {
    assert.equal(hash(await fs.readFile(input.path)), input.digest, 'Measurement input changed.');
  }
  const candidate = await read(config.candidate.path), reference = await read(config.reference.path);
  assert.equal(candidate.schema, 'doppler.capsule-retention-candidate/v1');
  const installed = await read(path.join(config.packageBundlePath, 'receipt.json'));
  assert(installed.passed);
  const archive = path.join(config.packageBundlePath, installed.package.filename);
  assert.equal(hash(await fs.readFile(archive)), candidate.runtimeHash);
  assert.equal(candidate.runtimeHash, `sha256:${installed.package.sha256}`);
  const installedRoot = path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu');
  const load = file => import(pathToFileURL(path.join(installedRoot, file)).href);
  const { openCapsule } = await load('src/client/capsule-host.js');
  const { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } = await load('src/tooling/node-webgpu.js');
  const { evaluateRerankReference } = await load('src/config/rerank-reference.js');
  ({ destroyDevice } = await load('src/gpu/device.js'));
  release = releaseNodeWebGPU;
  const provider = await bootstrapNodeWebGPUProvider('webgpu', { createArgs: config.providerCreateArgs });
  const info = provider.session.adapter.info;
  report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, info[key]]));
  assert.equal(report.hardware.vendor, config.requiredVendor);
  assert.equal(report.hardware.isFallbackAdapter, false);
  assert.equal(hash(Buffer.from(JSON.stringify(report.hardware))), config.hardwareDigest);
  report.provider = provider.receipt;
  const capsuleFile = path.join(config.capsuleRoot, 'distribution/capsule-v3.json');
  assert.equal(hash(await fs.readFile(capsuleFile)), candidate.capsuleHash);
  const options = await read(path.join(config.capsuleRoot, 'current-open-options.json'));
  const capsule = await read(capsuleFile);
  checkpointStore = await createRetainedReleaseCheckpoint(path.join(config.capsuleRoot, 'release-checkpoints.json'), config.outputDir, capsule.semanticRoot);
  options.persistReleaseCheckpoint = checkpointStore.persist;
  options.releasePolicy = { ...options.releasePolicy, now: new Date().toISOString() };
  options.maxRetainedArtifactBytes = candidate.maxRetainedArtifactBytes;
  const application = options.releaseEvents.at(-1).release.application;
  const originalFetch = globalThis.fetch;
  restoreFetch = () => { globalThis.fetch = originalFetch; };
  globalThis.fetch = async () => { throw new Error('Network acquisition is disabled during retention measurements.'); };
  report.stage = 'open';
  const before = performance.now();
  session = await openCapsule(capsuleFile, options);
  report.loadMs = performance.now() - before;
  report.identity = session.capsuleIdentity;
  report.selectedTargetPlanDigest = session.selectedTargetPlanDigest;
  assert.equal(report.selectedTargetPlanDigest, candidate.targetPlanDigest);
  report.stage = 'inference';
  const started = performance.now();
  report.receipt = await session.rerank({ application, ...reference.input });
  report.inferenceMs = performance.now() - started;
  report.comparison = evaluateRerankReference(reference, { input: reference.input,
    scoringConfig: session.manifest.inference.rerank, outputs: report.receipt.evidence.scores });
  assert(report.comparison.passed, 'Frozen source reference comparison failed.');
  report.physicalExecution = true;
  report.peakRssBytes = process.resourceUsage().maxRSS * 1024;
  report.budgetsPassed = report.loadMs <= config.maxLoadMs && report.inferenceMs <= config.maxInferenceMs;
  report.passed = report.comparison.passed && report.budgetsPassed;
  report.stage = 'complete';
} catch (error) {
  report.error = { message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally {
  report.cleanup = await closeInstalledCapsule({ closeSession: () => session?.close(),
    destroyDevice: () => destroyDevice?.(), releaseProvider: () => release?.() });
  if (!report.cleanup.passed) { report.passed = false; process.exitCode = 1; }
  try { await checkpointStore?.verifySourceUnchanged(); }
  catch (error) { report.cleanupError = error.message; report.passed = false; process.exitCode = 1; }
  restoreFetch?.();
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'measurement.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, outputDir: config.outputDir, loadMs: report.loadMs,
  peakRssBytes: report.peakRssBytes, error: report.error?.message }));
