#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const build = await read(path.join(config.applicationDir, 'build-receipt.json'));
const fixtureBytes = await fs.readFile(config.fixturePath);
const fixture = JSON.parse(fixtureBytes);
const report = { schema: 'doppler.node-document-search-qualification/v1', passed: false,
  generatedAtUtc: new Date().toISOString(), config, build, nodeVersion: process.version,
  probeSha256: hash(await fs.readFile(fileURLToPath(import.meta.url))), fixtureSha256: hash(fixtureBytes),
  physicalExecution: false, externalAdoption: false, phases: {}, networkRequests: [], checks: {} };
let app;
let sampler;
let stage = 'installed-assets';
const began = performance.now();
const fetchOriginal = globalThis.fetch;
let peakRssBytes = 0;
let metrics;
let originals;
let constructors;
function compareResults(actual, expected) {
  assert.deepEqual(actual.results.map(row => row.document.id), expected.results.map(row => row.document.id));
  for (const [index, row] of actual.results.entries()) {
    assert(Math.abs(row.rerankScore - expected.results[index].rerankScore) <= fixture.acceptance.offlineScoreMaxAbs);
  }
}
function sample() { peakRssBytes = Math.max(peakRssBytes, process.memoryUsage().rss); }
async function timed(name, action) {
  stage = name;
  process.stdout.write(name + '\n');
  const start = performance.now();
  const result = await action();
  report.phases[name] = performance.now() - start;
  sample();
  return result;
}
try {
  assert.equal(build.schema, 'doppler.document-search-node-build/v1');
  if (config.offlineKernelRequired) {
    const { createConnection } = await import('node:net');
    report.networkIsolation = [];
    for (const host of ['127.0.0.1', '::1']) {
      const code = await new Promise((resolve, reject) => {
        const socket = createConnection({ host, port: 9 });
        socket.once('connect', () => { socket.destroy(); reject(new Error('Network socket unexpectedly succeeded')); });
        socket.once('error', error => resolve(error.code));
      });
      assert.equal(code, 'EPERM', 'Kernel must deny socket creation, not merely reject the connection');
      report.networkIsolation.push({ host, error: code });
    }
  }
  for (const asset of build.assets) {
    const bytes = await fs.readFile(path.join(config.applicationDir, asset.path));
    assert.equal(bytes.length, asset.sizeBytes, asset.path);
    assert.equal(hash(bytes), asset.sha256, asset.path);
  }
  assert(build.runtimeAssets?.length > 0, 'Installed runtime asset manifest required');
  for (const asset of build.runtimeAssets) {
    const bytes = await fs.readFile(path.join(config.applicationDir, 'node_modules/doppler-gpu', asset.path));
    assert.equal(bytes.length, asset.sizeBytes, asset.path);
    assert.equal(hash(bytes), asset.sha256, asset.path);
  }
  const installedPackage = path.join(config.applicationDir, 'node_modules/doppler-gpu/package.json');
  assert.equal((await read(installedPackage)).version, build.installedPackage.version);
  assert.equal(hash(await fs.readFile(path.join(config.applicationDir, 'vendor', build.installedPackage.filename))), build.installedPackage.sha256);
  assert.equal((await read(path.join(config.applicationDir, 'node_modules/webgpu/package.json'))).version, build.provider.version);
  report.lockSha256 = hash(await fs.readFile(path.join(config.applicationDir, 'package-lock.json')));
  report.storageFilesystem = { type: (await fs.statfs(config.storageDir)).type,
    rebootTested: false, physicalGpuResidencyMeasured: false };
  if (config.install) assert.deepEqual(await fs.readdir(config.storageDir), [], 'Clean installation requires empty application storage');
  globalThis.fetch = async (...args) => {
    report.networkRequests.push(String(args[0]));
    if (config.offline) throw new Error('Network disabled during retained Node acceptance.');
    return fetchOriginal(...args);
  };
  const { createNodeDocumentSearch } = await import(pathToFileURL(path.join(config.applicationDir, 'node.js')));
  const { getDevice } = await import(pathToFileURL(path.join(config.applicationDir, 'node_modules/doppler-gpu/src/tooling-exports/device.js')));
  app = await createNodeDocumentSearch({ storageDir: config.storageDir });
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter, 'Physical GPU adapter required');
  report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']
    .map(key => [key, adapter.info[key]]));
  assert.equal(report.hardware.vendor, config.requiredVendor);
  assert.equal(report.hardware.isFallbackAdapter, false);
  originals = { submit: GPUQueue.prototype.submit, createBuffer: GPUDevice.prototype.createBuffer,
    destroyBuffer: GPUBuffer.prototype.destroy, destroyDevice: GPUDevice.prototype.destroy };
  constructors = { queue: GPUQueue, device: GPUDevice, buffer: GPUBuffer };
  metrics = { submitCalls: 0, createdBufferBytes: 0, liveRequestedBufferBytes: 0, peakRequestedBufferBytes: 0 };
  const buffers = new WeakMap();
  const devices = new WeakMap();
  GPUQueue.prototype.submit = function(commands) {
    const result = originals.submit.call(this, commands);
    metrics.submitCalls++;
    globalThis.documentSearchSubmissionProbe?.();
    return result;
  };
  GPUDevice.prototype.createBuffer = function(descriptor) {
    const buffer = originals.createBuffer.call(this, descriptor);
    let device = devices.get(this);
    if (!device) { device = { live: 0, destroyed: false }; devices.set(this, device); }
    buffers.set(buffer, { device, bytes: descriptor.size, destroyed: false });
    device.live += descriptor.size;
    metrics.createdBufferBytes += descriptor.size;
    metrics.liveRequestedBufferBytes += descriptor.size;
    metrics.peakRequestedBufferBytes = Math.max(metrics.peakRequestedBufferBytes, metrics.liveRequestedBufferBytes);
    return buffer;
  };
  GPUBuffer.prototype.destroy = function() {
    const result = originals.destroyBuffer.call(this);
    const entry = buffers.get(this);
    if (entry && !entry.destroyed && !entry.device.destroyed) {
      entry.destroyed = true; entry.device.live -= entry.bytes; metrics.liveRequestedBufferBytes -= entry.bytes;
    }
    return result;
  };
  GPUDevice.prototype.destroy = function() {
    const result = originals.destroyDevice.call(this);
    const entry = devices.get(this);
    if (entry && !entry.destroyed) { entry.destroyed = true; metrics.liveRequestedBufferBytes -= entry.live; entry.live = 0; }
    return result;
  };
  sampler = setInterval(sample, 100);
  await timed(config.install ? 'install' : 'openRetained', () => config.install ? app.controller.install() : app.controller.openRetained());
  assert.deepEqual(app.controller.getState().sessionRoles.sort(), ['embedding', 'reranker']);
  report.sessions = Object.fromEntries(Object.entries(app.controller.getSessions()).map(([role, session]) => [role, {
    capsuleIdentity: session.capsuleIdentity, selectedTargetId: session.selectedTargetId,
    selectedTargetPlanDigest: session.selectedTargetPlanDigest, deviceProfile: session.deviceProfile,
    observedInitialExecutionIdentity: session.observedInitialExecutionIdentity,
  }]));
  report.physicalExecution = true;
  await timed('index', () => app.controller.indexDocuments(fixture.documents));
  const index = JSON.stringify(app.controller.getIndex());
  const beforeReuse = metrics.submitCalls;
  await timed('unchangedIndex', () => app.controller.indexDocuments(fixture.documents));
  assert.equal(metrics.submitCalls, beforeReuse, 'Unchanged documents must not dispatch GPU work');
  report.checks.unchangedDocumentReuse = true;
  report.searches = [];
  for (let repeat = 0; repeat < 2; repeat++) {
    for (const query of fixture.queries) {
      const result = await timed(`search-${repeat}-${query.id}`, () => app.controller.search(query.text));
      assert.equal(result.results[0]?.document.id, query.expectedTopId);
      const previous = report.searches.find(entry => entry.queryId === query.id);
      if (previous) compareResults(result, previous.result);
      report.searches.push({ repeat, queryId: query.id, result });
    }
  }
  report.checks.repeatedSearch = true;
  if (config.previousReportPath) {
    const previous = await read(config.previousReportPath);
    for (const entry of report.searches) compareResults(entry.result, previous.searches.find(item => item.queryId === entry.queryId).result);
    report.checks.restartResults = true;
  }
  if (config.lifecycle) {
    let submits = 0;
    globalThis.documentSearchSubmissionProbe = () => { submits++; app.controller.cancelIndexing(); };
    await assert.rejects(app.controller.indexDocuments([...fixture.documents, { ...fixture.documents[0], id: 'cancelled', text: 'A cancelled revision.' }]), { name: 'AbortError' });
    delete globalThis.documentSearchSubmissionProbe;
    assert(submits > 0);
    assert.equal(JSON.stringify(app.controller.getIndex()), index);
    report.checks.cancelledIndexAfterSubmission = { passed: true, submissions: submits };
    submits = 0;
    globalThis.documentSearchSubmissionProbe = () => { submits++; app.controller.cancelSearch(); };
    await assert.rejects(app.controller.search(fixture.queries[0].text), { name: 'AbortError' });
    delete globalThis.documentSearchSubmissionProbe;
    assert(submits > 0);
    report.checks.cancelledQueryAfterSubmission = { passed: true, submissions: submits };
    const first = app.controller.search(fixture.queries[0].text);
    const second = app.controller.search(fixture.queries[1].text);
    assert.equal((await first).superseded, true);
    assert.equal((await second).results[0].document.id, fixture.queries[1].expectedTopId);
    report.checks.supersededQuery = true;
    const snapshotPath = path.join(config.storageDir, 'documents/document-snapshot.json');
    const savedSnapshot = await fs.readFile(snapshotPath);
    const rename = fs.rename;
    fs.rename = async (from, to) => {
      if (String(to).endsWith('document-snapshot.json')) throw Object.assign(new Error('Injected interrupted save'), { code: 'EIO' });
      return rename(from, to);
    };
    try { await assert.rejects(app.controller.indexDocuments(fixture.documents), { code: 'EIO' }); }
    finally { fs.rename = rename; }
    assert.equal(JSON.stringify(app.controller.getIndex()), index);
    assert.deepEqual(await fs.readFile(snapshotPath), savedSnapshot);
    report.checks.interruptedSave = true;
  }
  report.observations = app.controller.getObservations();
  report.gpuWhileResident = { ...metrics };
  await timed('close', () => app.close());
  report.checks.explicitClosure = !app.controller.getState().hasSessions;
  app = null;
  assert.equal(metrics.liveRequestedBufferBytes, 0);
  if (config.lifecycle) {
    app = await createNodeDocumentSearch({ storageDir: config.storageDir });
    const model = app.config.models[0];
    const capsule = await read(path.join(config.applicationDir, model.capsuleUrl));
    const artifact = capsule.artifacts.find(item => item.role === 'tokenizer');
    const damaged = path.join(config.storageDir, model.storageId, 'artifacts', artifact.hash.slice(7));
    await fs.writeFile(damaged, new Uint8Array([0]));
    await timed('detectCorruption', () => assert.rejects(app.controller.openRetained(), /integrity/i));
    await timed('repair', () => app.controller.repair());
    assert.equal((await app.controller.search(fixture.queries[0].text)).results[0].document.id, fixture.queries[0].expectedTopId);
    report.checks.corruptionRepair = true;
    const lost = getDevice();
    lost.destroy();
    await lost.lost;
    await assert.rejects(app.controller.search(fixture.queries[0].text));
    await app.close();
    app = await createNodeDocumentSearch({ storageDir: config.storageDir });
    await timed('deviceLossRecovery', () => app.controller.openRetained());
    assert.equal((await app.controller.search(fixture.queries[0].text)).results[0].document.id, fixture.queries[0].expectedTopId);
    report.checks.deviceLossRecovery = true;
    await app.close(); app = null;
    assert.equal(metrics.liveRequestedBufferBytes, 0);
  }
  if (config.offline) assert.deepEqual(report.networkRequests, []);
  report.passed = true;
} catch (error) { report.failure = { stage, message: error.message, stack: error.stack }; }
finally {
  clearInterval(sampler);
  try { await app?.close(); } catch (error) { report.cleanupFailure = error.message; report.passed = false; }
  if (originals) {
    constructors.queue.prototype.submit = originals.submit;
    constructors.device.prototype.createBuffer = originals.createBuffer;
    constructors.device.prototype.destroy = originals.destroyDevice;
    constructors.buffer.prototype.destroy = originals.destroyBuffer;
  }
  globalThis.fetch = fetchOriginal;
  delete globalThis.documentSearchSubmissionProbe;
  sample();
  report.memory = { sampledPeakProcessRssBytes: peakRssBytes, processHighWaterRssBytes: process.resourceUsage().maxRSS * 1024,
    gpu: metrics, gpuScope: 'Requested buffer allocation sizes and explicit destruction; not measured physical GPU residency.' };
  report.elapsedMs = performance.now() - began;
  await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ passed: report.passed, failure: report.failure, outputPath: config.outputPath }));
if (!report.passed) process.exitCode = 1;
