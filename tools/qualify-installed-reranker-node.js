#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Repository qualification imports execution exclusively from the installed archive.
const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const config = await read(process.argv[2]);
for (const key of ['packageBundlePath', 'capsuleRoot', 'deniedRoot', 'referencePath', 'outputDir']) {
  if (!path.isAbsolute(config[key] ?? '')) throw new Error(`Absolute ${key} required.`);
}
if (!config.requiredVendor || !Array.isArray(config.providerCreateArgs)
  || !Number.isSafeInteger(config.repeatRuns) || config.repeatRuns < 1) {
  throw new Error('Explicit hardware, provider creation and repeat policy required.');
}
const { capsuleRoot, deniedRoot, referencePath, outputDir } = config;
const installed = await read(path.join(config.packageBundlePath, 'receipt.json'));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
assert(installed.passed, 'Passing installed-package receipt required.');
assert.equal(sha256(await fs.readFile(path.join(config.packageBundlePath, installed.package.filename))), installed.package.sha256);
const installedRoot = path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu');
const installedModule = relative => import(pathToFileURL(path.join(installedRoot, relative)).href);
const { openCapsule } = await installedModule('src/client/capsule-host.js');
const { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } = await installedModule('src/tooling/node-webgpu.js');
const { getDevice, destroyDevice } = await installedModule('src/gpu/device.js');
const { evaluateRerankReference } = await installedModule('src/config/rerank-reference.js');
await fs.mkdir(outputDir);
await fs.writeFile(path.join(outputDir, 'config.json'), JSON.stringify(config, null, 2), { flag: 'wx' });
const outputPath = path.join(outputDir, 'qualification.json');
const pack = await read(`${capsuleRoot}/distribution/capsule-v3.json`);
const options = await read(`${capsuleRoot}/current-open-options.json`);
options.releasePolicy = { ...options.releasePolicy, now: new Date().toISOString() };
const application = options.releaseEvents.at(-1).release.application;
const sourceLedgerPath = `${capsuleRoot}/release-checkpoints.json`;
const sourceLedgerBytes = await fs.readFile(sourceLedgerPath);
const ledgerPath = path.join(outputDir, 'release-checkpoints.json');
await fs.writeFile(ledgerPath, sourceLedgerBytes, { flag: 'wx' });
const priorLedger = await read(ledgerPath);
options.persistReleaseCheckpoint = async checkpoint => {
  const ledger = await read(ledgerPath);
  assert(checkpoint.sequence >= ledger[pack.semanticRoot].sequence);
  if (checkpoint.sequence === ledger[pack.semanticRoot].sequence) assert.equal(checkpoint.digest, ledger[pack.semanticRoot].digest);
  ledger[pack.semanticRoot] = checkpoint;
  await fs.writeFile(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
};
const reference = await read(referencePath);
const require = createRequire(path.join(installedRoot, 'package.json'));
const report = { schema: 'doppler.standalone-node-capsule-probe/v1', startedAtUtc: new Date().toISOString(),
  passed: false, installedPackage: installed.package, config, stage: 'provider', nodeVersion: process.version, platform: process.platform,
  qualifierSha256: sha256(await fs.readFile(new URL(import.meta.url))),
  consumerLockSha256: sha256(await fs.readFile(path.join(config.packageBundlePath, 'consumer/package-lock.json'))),
  releaseEvaluationTime: options.releasePolicy.now,
  evidenceClass: 'internal-physical-installed-package', externalAdoption: false,
  sourceReference: { path: referencePath, digest: createHash('sha256').update(await fs.readFile(referencePath)).digest('hex') },
  signedCapsule: { capsuleId: pack.capsuleId, semanticRoot: pack.semanticRoot }, requests: [] };
let session;
try {
  assert.throws(() => require.resolve('doe-gpu/node-webgpu'), /Cannot find/);
  report.doeAbsent = true;
  const provider = await bootstrapNodeWebGPUProvider('webgpu', {
    createArgs: config.providerCreateArgs,
  });
  report.provider = provider.receipt;
  const info = provider.session.adapter.info;
  report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, info[key]]));
  assert(!/swiftshader|llvmpipe|software/i.test(JSON.stringify(report.hardware)));
  assert.equal(report.hardware.isFallbackAdapter, false);
  assert.equal(report.hardware.vendor.toLowerCase(), config.requiredVendor);
  report.providerVersion = require('webgpu/package.json').version;
  globalThis.fetch = async input => { report.requests.push(String(input)); throw new Error('Network disabled for retained Capsule probe.'); };
  report.stage = 'capsule-open';
  const before = performance.now();
  session = await openCapsule(`${capsuleRoot}/distribution/capsule-v3.json`, options);
  report.loadMs = performance.now() - before;
  report.identity = session.capsuleIdentity;
  report.plan = session.selectedTargetPlanDigest;
  report.stage = 'rerank';
  report.executions = [];
  for (let repeat = 0; repeat < config.repeatRuns; repeat++) {
    const began = performance.now();
    const receipt = await session.rerank({ application, ...reference.input });
    const observation = { input: { query: receipt.evidence.query, documents: receipt.evidence.documents },
      scoringConfig: session.manifest.inference.rerank, outputs: receipt.evidence.scores };
    const comparison = evaluateRerankReference(reference, observation);
    report.executions.push({ repeat, elapsedMs: performance.now() - began, receipt, comparison });
    assert.equal(comparison.passed, true, 'pinned source oracle');
    assert.equal(session.selectedTargetPlanDigest, report.plan);
  }
  const lostDevice = getDevice(); lostDevice.destroy(); await lostDevice.lost;
  await assert.rejects(session.rerank({ application, ...reference.input }), /lost|closed|destroyed|device/i);
  await session.close(); session = null;
  session = await openCapsule(`${capsuleRoot}/distribution/capsule-v3.json`, options);
  assert.notEqual(getDevice(), lostDevice);
  const restored = await session.rerank({ application, ...reference.input });
  const restoredComparison = evaluateRerankReference(reference, { input: reference.input, scoringConfig: session.manifest.inference.rerank, outputs: restored.evidence.scores });
  assert(restoredComparison.passed);
  report.deviceLossRecovery = { passed: true, distinctDevice: true, receipt: restored, comparison: restoredComparison };
  const denialBytes = await fs.readFile(deniedRoot + '/recovery-checkpoint.json');
  const deniedOptions = await read(deniedRoot + '/retained-open-options.json');
  const events = await read(deniedRoot + '/recovery-release-events.json');
  const checkpoint = JSON.parse(denialBytes);
  const rejectOptions = { ...deniedOptions, releasePolicy: { ...deniedOptions.releasePolicy,
    now: new Date().toISOString(), minimumSequence: 2, checkpoint },
    persistReleaseCheckpoint: value => assert.deepEqual(value, checkpoint),
    artifactStore: { readArtifact() { throw Error('A denied release must not acquire artifacts'); } } };
  await assert.rejects(openCapsule(deniedRoot + '/distribution/capsule-v3.json', { ...rejectOptions, releaseEvents: [events.eligible, events.revoked] }), /revoked|denied/i);
  await assert.rejects(openCapsule(deniedRoot + '/distribution/capsule-v3.json', rejectOptions), /checkpoint|rollback|sequence|history/i);
  assert.deepEqual(await fs.readFile(deniedRoot + '/recovery-checkpoint.json'), denialBytes);
  assert.deepEqual(await read(ledgerPath), priorLedger);
  assert.deepEqual(await fs.readFile(sourceLedgerPath), sourceLedgerBytes);
  report.priorDenialPreserved = { passed: true, checkpoint, rollbackRejected: true };
  report.passed = true;
  report.stage = 'complete';
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  const errors = [];
  for (const close of [() => session?.close(), () => destroyDevice(), async () => { report.release = await releaseNodeWebGPU(); }]) {
    try { await close(); } catch (error) { errors.push(error.message); }
  }
  report.cleanup = { passed: errors.length === 0, errors };
  report.passed &&= report.cleanup.passed;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(outputPath, JSON.stringify(report, null, 2), { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error?.message, outputPath }));
if (!report.passed) process.exitCode = 1;
