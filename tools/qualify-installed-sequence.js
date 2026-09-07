#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { evaluateSequenceReference, validateSequenceReference } from './lib/sequence-model-qualification.js';
import { closeInstalledCapsule } from './close-installed-capsule.js';

const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const read = async file => JSON.parse(await fs.readFile(file));

export function assertCompleteSequenceReferences(references, dimension) {
  assert(Array.isArray(references) && references.length > 0);
  assert(Number.isSafeInteger(dimension) && dimension > 0);
  const indices = Array.from({ length: dimension }, (_, index) => index);
  for (const reference of references) {
    validateSequenceReference(reference);
    assert.equal(reference.outputs.logits, false);
    assert.deepEqual(reference.probes.pooledEmbedding.indices, indices, 'Every pooled element needs a source reference.');
    assert.equal(reference.probes.pooledEmbedding.values.length, dimension);
    assert.equal(reference.probes.tokenEmbeddings.length, reference.input.tokenIds.length);
    reference.probes.tokenEmbeddings.forEach((probe, position) => {
      assert.equal(probe.position, position); assert.deepEqual(probe.indices, indices, 'Every token element needs a source reference.');
      assert.equal(probe.values.length, dimension);
    });
    for (const probe of [reference.probes.pooledEmbedding, ...reference.probes.tokenEmbeddings]) {
      assert(probe.values.every(Number.isFinite));
    }
  }
}

export async function qualifyInstalledSequence(config) {
  assert.equal(config.schema, 'doppler.installed-sequence-qualification/v1');
  assert.equal(process.versions.bun, undefined, 'This qualification declares Node explicitly.');
  assert(Number.isSafeInteger(config.maxRecoveryMs) && config.maxRecoveryMs > 0);
  assert(Array.isArray(config.providerCreateArgs) && config.requiredVendor);
  for (const field of ['packageBundlePath', 'modelDir', 'outputDir']) assert(path.isAbsolute(config[field]));
  const referenceBytes = await fs.readFile(config.reference.path); assert.equal(hash(referenceBytes), config.reference.digest);
  const manifestBytes = await fs.readFile(path.join(config.modelDir, 'manifest.json')); assert.equal(hash(manifestBytes), config.manifestDigest);
  const manifest = JSON.parse(manifestBytes);
  const source = JSON.parse(referenceBytes); assert.equal(source.schema, 'doppler.sequence-source-reference-set/v1');
  assertCompleteSequenceReferences(source.references, manifest.architecture.hiddenSize);
  assert.deepEqual(source.references.map(reference => reference.input.sequence), source.policy.sequences);
  assert.equal(config.maxRecoveryMs, source.policy.maxRecoveryMs);
  for (const reference of source.references) {
    assert.equal(reference.modelId, source.policy.modelId);
    assert.equal(reference.source.repository, source.policy.repository);
    assert.equal(reference.source.revision, source.policy.revision);
    assert.deepEqual(reference.tolerances, source.policy.tolerances);
  }
  const installed = await read(path.join(config.packageBundlePath, 'receipt.json')); assert(installed.passed);
  assert.equal(hash(await fs.readFile(path.join(config.packageBundlePath, installed.package.filename))), `sha256:${installed.package.sha256}`);
  const installedRoot = path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu');
  const installedModule = file => import(pathToFileURL(path.join(installedRoot, file)).href);
  const api = await installedModule('src/client/doppler-api.js');
  const { getDevice, destroyDevice } = await installedModule('src/gpu/device.js');
  const { observeInitialExecutionIdentity } = await installedModule('src/config/initial-execution-identity.js');
  const { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } = await installedModule('src/tooling/node-webgpu.js');
  await fs.mkdir(config.outputDir);
  const report = { schema: 'doppler.installed-sequence-qualification-result/v1', passed: false, config,
    startedAtUtc: new Date().toISOString(), installedPackage: installed.package,
    qualifierDigest: hash(await fs.readFile(new URL(import.meta.url))),
    referenceEvaluatorDigest: hash(await fs.readFile(new URL('./lib/sequence-model-qualification.js', import.meta.url))),
    cleanupDigest: hash(await fs.readFile(new URL('./close-installed-capsule.js', import.meta.url))),
    source: source.policy, manifestDigest: config.manifestDigest, referenceDigest: config.reference.digest,
    runtime: { name: 'node', version: process.version }, phases: [], requests: [],
    signedCapsule: false, externalAdoption: false, scope: 'Complete declared encoder outputs and installed Node device-loss recovery; no language-model head or biological claim.' };
  let model;
  const originalFetch = globalThis.fetch;
  async function open() {
    const began = performance.now();
    model = await api.load({ url: pathToFileURL(config.modelDir + '/').href },
      { runtimeConfig: { inference: { session: manifest.inference.session } } });
    for (const reference of source.references) {
      assert.equal(model.manifest.artifactIdentity.sourceCheckpointId, reference.source.checkpointId);
      assert.equal(model.manifest.artifactIdentity.sourceRepo, reference.source.repository);
      assert.equal(model.manifest.artifactIdentity.sourceRevision, reference.source.revision);
    }
    return { loadMs: performance.now() - began,
      executionIdentity: observeInitialExecutionIdentity(model.advanced.getResolvedRuntimeSession()) };
  }
  async function execute(stage) {
    for (const [index, reference] of source.references.entries()) {
      const began = performance.now();
      const result = await model.encodeSequence(reference.input.sequence, { includeTokenEmbeddings: true, includeLogits: false });
      const comparison = evaluateSequenceReference({ manifest: model.manifest, result, reference });
      report.phases.push({ stage, index, durationMs: performance.now() - began, comparison,
        observation: { tokens: Array.from(result.tokens), embeddingDim: result.embeddingDim,
          pooledEmbedding: Array.from(result.pooledEmbedding), tokenEmbeddings: Array.from(result.tokenEmbeddings) } });
      await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(report.phases, null, 2) + '\n');
      assert(comparison.passed, `Full sequence reference failed at ${stage}/${index}.`);
    }
  }
  try {
    const provider = await bootstrapNodeWebGPUProvider('webgpu', { createArgs: config.providerCreateArgs });
    assert(provider.ok); report.provider = provider.receipt;
    report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']
      .map(key => [key, provider.session.adapter.info[key]]));
    assert.equal(report.hardware.vendor, config.requiredVendor); assert.equal(report.hardware.isFallbackAdapter, false);
    globalThis.fetch = async input => { report.requests.push(String(input)); throw new Error('Network disabled for retained sequence qualification.'); };
    const { installNodeFileFetchShim } = await installedModule('src/tooling/node-file-fetch.js'); installNodeFileFetchShim();
    report.initial = await open(); await execute('initial');
    const lost = getDevice(); lost.destroy(); const loss = await lost.lost;
    report.deviceLoss = { reason: loss.reason, message: loss.message };
    try {
      await model.encodeSequence(source.references[0].input.sequence, { includeTokenEmbeddings: true, includeLogits: false });
      assert.fail('A lost device must reject further execution.');
    } catch (error) {
      if (error.code === 'ERR_ASSERTION') throw error;
      report.lostSessionRejection = { name: error.name, message: error.message };
    }
    const recoveryStarted = performance.now();
    await model.unload(); model = null; destroyDevice();
    report.reopened = await open(); assert.notEqual(getDevice(), lost);
    await execute('recovered'); report.recoveryMs = performance.now() - recoveryStarted;
    assert(report.recoveryMs <= config.maxRecoveryMs, 'Recovery exceeded the declared budget.');
    report.peakProcessRssBytes = process.resourceUsage().maxRSS * 1024;
    report.passed = true;
  } catch (error) { report.error = { message: error.message, stack: error.stack }; }
  finally {
    report.cleanup = await closeInstalledCapsule({ closeSession: () => model?.unload(), destroyDevice, releaseProvider: releaseNodeWebGPU });
    globalThis.fetch = originalFetch; report.passed &&= report.cleanup.passed;
    report.completedAtUtc = new Date().toISOString();
    await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  }
  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const report = await qualifyInstalledSequence(await read(process.argv[2]));
  console.log(JSON.stringify({ passed: report.passed, outputDir: report.config.outputDir, error: report.error }));
  if (!report.passed) process.exitCode = 1;
}
