#!/usr/bin/env node

import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { openPack } from '../src/client/doppler-api.js';
import { getPackIdentity } from '../src/config/pack.js';
import { hashPackSequenceInput, hashPackSequenceOutput } from '../src/config/pack-sequence-receipt.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';
import { destroyDevice, resetDeviceState } from '../src/gpu/device.js';
import { releaseNodeWebGPU } from '../src/tooling/node-webgpu.js';
import { evaluateSequenceReference, validateSequenceReference } from './lib/sequence-model-qualification.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIGEST = /^sha256:[0-9a-f]{64}$/u;

export function validateSequencePackQualificationConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) throw new Error('Qualification config must be an object.');
  for (const field of ['packPath', 'referencePath', 'outputPath']) {
    if (typeof config[field] !== 'string' || !config[field].trim()) throw new Error(`Qualification requires ${field}.`);
  }
  if (typeof config.expectedPack?.packId !== 'string' || !config.expectedPack.packId.trim()
    || !['doppler.pack/v2', 'doppler.pack/v3'].includes(config.expectedPack.schema)
    || ['semanticRoot', 'envelopeDigest', 'artifactClosureDigest'].some((field) => !DIGEST.test(config.expectedPack[field]))) {
    throw new Error('Qualification requires an exact expectedPack identity.');
  }
  if (!DIGEST.test(config.referenceDigest)) throw new Error('Qualification requires the frozen referenceDigest.');
  if (config.originPolicy !== 'disabled') throw new Error('Qualification originPolicy must be disabled.');
  const options = config.openOptions;
  if (!options?.trustedSigners || typeof options.trustedSigners !== 'object' || Array.isArray(options.trustedSigners)
    || Object.keys(options.trustedSigners).length === 0
    || !Array.isArray(options.acceptedTargetPlanDigests) || options.acceptedTargetPlanDigests.length === 0
    || options.acceptedTargetPlanDigests.some((entry) => !DIGEST.test(entry))) {
    throw new Error('Qualification requires explicit trustedSigners and acceptedTargetPlanDigests.');
  }
  if (config.sequenceOptions?.includeTokenEmbeddings !== true || config.sequenceOptions?.includeLogits !== false
    || !config.sequenceOptions.assignment || typeof config.sequenceOptions.assignment !== 'object'
    || Array.isArray(config.sequenceOptions.assignment)) {
    throw new Error('Qualification requires explicit embedding options and assignment.');
  }
  return config;
}

function gitValue(args) {
  return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
}

export async function qualifySequencePack(input) {
  const config = structuredClone(validateSequencePackQualificationConfig(input));
  const observations = [];
  const fetchAttempts = [];
  const report = {
    schema: 'doppler.sequencePackQualification.v1',
    passed: false,
    generatedAt: new Date().toISOString(),
    config,
    configDigest: computeCanonicalSha256(config),
    source: { revision: gitValue(['rev-parse', 'HEAD']), dirty: Boolean(gitValue(['status', '--short'])) },
    boundary: { publicAPI: 'openPack.encodeSequence', modelImplementation: 'doppler', originPolicy: config.originPolicy, independentParticipants: false },
    stage: 'input-verification',
    observations,
    fetchAttempts,
  };
  let session;
  const originalFetch = globalThis.fetch;
  try {
    const pack = JSON.parse(await readFile(config.packPath, 'utf8'));
    const identity = getPackIdentity(pack);
    if (computeCanonicalSha256(identity) !== computeCanonicalSha256(config.expectedPack)) {
      throw new Error('Pack identity differs from the frozen qualification request.');
    }
    const referenceBytes = await readFile(config.referencePath);
    if (hashBytesSha256(referenceBytes) !== config.referenceDigest) throw new Error('Reference digest differs from the frozen qualification request.');
    const reference = validateSequenceReference(JSON.parse(referenceBytes.toString('utf8')));
    globalThis.fetch = async (resource) => {
      const url = String(resource?.url ?? resource);
      fetchAttempts.push(url);
      throw new Error(`Origin-disabled qualification prohibits fetch: ${url}`);
    };
    report.stage = 'pack-open';
    session = await openPack(config.packPath, {
      ...config.openOptions,
      observer: { observe: (event) => observations.push(structuredClone(event)) },
    });
    report.runtime = { device: session.deviceProfile, initialExecutionIdentity: session.observedInitialExecutionIdentity };
    report.stage = 'sequence-execution';
    const result = await session.encodeSequence(reference.input.sequence, config.sequenceOptions);
    report.stage = 'acceptance';
    report.result = evaluateSequenceReference({ manifest: session.manifest, result, reference });
    report.executionReceipt = result.receipt;
    const { receiptDigest, ...receipt } = result.receipt;
    report.bindingChecks = [
      { id: 'pack.identity', passed: computeCanonicalSha256(receipt.pack) === computeCanonicalSha256(identity) },
      { id: 'plan.accepted', passed: config.openOptions.acceptedTargetPlanDigests.includes(receipt.targetPlanDigest) },
      { id: 'operation', passed: receipt.operation === 'encodeSequence' },
      { id: 'assignment', passed: receipt.assignmentHash === computeCanonicalSha256(config.sequenceOptions.assignment) },
      { id: 'input', passed: receipt.inputHash === hashPackSequenceInput(reference.input.sequence, config.sequenceOptions) },
      { id: 'output', passed: receipt.outputHash === hashPackSequenceOutput(result) },
      { id: 'receipt.integrity', passed: receiptDigest === computeCanonicalSha256(receipt) },
      { id: 'artifacts.complete', passed: receipt.artifactReceipts.length === pack.artifacts.length
        && pack.artifacts.every((artifact) => receipt.artifactReceipts.some((entry) => entry.artifactId === artifact.artifactId
          && entry.hash === artifact.hash && entry.sizeBytes === artifact.sizeBytes)) },
      { id: 'origin.unused', passed: fetchAttempts.length === 0 },
    ];
    report.passed = report.result.passed && report.bindingChecks.every((check) => check.passed);
    report.stage = 'complete';
  } catch (error) {
    report.error = { name: error.name, message: error.message };
  } finally {
    try {
      await session?.close();
    } catch (error) {
      report.passed = false;
      report.cleanupError = { name: error.name, message: error.message };
    } finally {
      globalThis.fetch = originalFetch;
      for (const cleanup of [destroyBufferPool, destroyDevice, resetDeviceState, releaseNodeWebGPU]) {
        try { await cleanup(); } catch (error) {
          report.passed = false;
          report.cleanupError = { name: error.name, message: error.message };
        }
      }
    }
    await mkdir(dirname(resolve(config.outputPath)), { recursive: true });
    await writeFile(config.outputPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  return report;
}

async function main(argv) {
  if (argv.length !== 2 || argv[0] !== '--config') throw new Error('Usage: node tools/qualify-sequence-pack.js --config <path>');
  const report = await qualifySequencePack(JSON.parse(await readFile(argv[1], 'utf8')));
  console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error ?? null, outputPath: report.config.outputPath }));
  if (!report.passed) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
