import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  FORGE_VERSION,
  buildForgeOptions,
  forgeModelPack,
  parseArgs,
  usage,
} from '../../tools/forge-model-pack.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-forge-test-'));
const fixtureRoot = path.join(tmpRoot, 'fixture');
const modelDir = path.join(fixtureRoot, 'model');
const reportDir = path.join(fixtureRoot, 'reports');
await fs.mkdir(modelDir, { recursive: true });
await fs.mkdir(reportDir, { recursive: true });

const gatherDigest = `sha256:${KERNEL_REF_CONTENT_DIGESTS['gather.wgsl#main']}`;
const manifestPath = path.join(modelDir, 'manifest.json');
const reportPath = path.join(reportDir, 'report.json');
const conversionConfigPath = path.join(fixtureRoot, 'conversion.json');
const promptTokenDigest = `sha256:${'a'.repeat(64)}`;
const logitsDigest = `sha256:${'b'.repeat(64)}`;
const kvByteDigest = `sha256:${'c'.repeat(64)}`;
const kvKeyDigest = `sha256:${'d'.repeat(64)}`;
const kvValueDigest = `sha256:${'e'.repeat(64)}`;

await fs.writeFile(path.join(modelDir, 'tokenizer.json'), '{"model":"unit"}\n', 'utf8');
await fs.writeFile(conversionConfigPath, '{"modelId":"forge-unit-model"}\n', 'utf8');
await fs.writeFile(manifestPath, `${JSON.stringify({
  version: 1,
  modelId: 'forge-unit-model',
  modelType: 'llm',
  hashAlgorithm: 'sha256',
  shards: [
    {
      index: 0,
      filename: 'shard_00000.bin',
      size: 16,
      hash: '1'.repeat(64),
      offset: 0,
    },
  ],
  tokenizer: {
    type: 'bundled',
    file: 'tokenizer.json',
  },
  inference: {
    schema: 'doppler.execution/v1',
    execution: {
      kernels: {
        embed: {
          kernel: 'gather.wgsl',
          entry: 'main',
          digest: gatherDigest,
        },
      },
      preLayer: [['embed', 'embed']],
      decode: [],
      prefill: [],
      postLayer: [],
    },
  },
}, null, 2)}\n`, 'utf8');

await fs.writeFile(reportPath, `${JSON.stringify({
  mode: 'debug',
  workload: 'inference',
  suite: 'debug',
  modelId: 'forge-unit-model',
  timestamp: '2026-04-22T00:00:00.000Z',
  results: [{ name: 'generation', passed: true }],
  metrics: {
    prompt: 'The sky is',
    maxTokens: 1,
    tokensGenerated: 1,
    prefillMs: 1.5,
    decodeMs: 2.5,
    prefillTokens: 3,
    decodeTokens: 1,
    generationDiagnostics: {
      preview: [{ id: 42, text: ' blue', fallbackText: ' blue' }],
      total: 1,
      omitted: 0,
    },
    referenceTranscript: {
      prompt: {
        identity: 'The sky is',
        hash: `sha256:${'f'.repeat(64)}`,
        tokenIdsHash: promptTokenDigest,
        tokenCount: 3,
      },
      kvCache: {
        mode: 'stats+sha256-layer-kv-bytes',
        layout: 'contiguous',
        kvDtype: 'f16',
        seqLen: 4,
        maxSeqLen: 8,
        usedBytes: 64,
        allocatedBytes: 128,
        counters: null,
        byteDigestMode: 'sha256-layer-kv-bytes',
        byteDigest: kvByteDigest,
        byteDigests: [{
          layer: 0,
          seqLen: 4,
          keyBytes: 32,
          valueBytes: 32,
          keyDigest: kvKeyDigest,
          valueDigest: kvValueDigest,
        }],
      },
      logits: {
        mode: 'sha256-per-step',
        perStepDigests: [logitsDigest],
        steps: [{
          index: 0,
          tokenId: 42,
          inputTokenCount: 3,
          dtype: 'f32',
          elementCount: 8,
          digest: logitsDigest,
        }],
      },
    },
  },
  output: ' blue',
}, null, 2)}\n`, 'utf8');

// Test 1: CLI parseArgs and usage
const parsed = parseArgs([
  '--manifest', manifestPath,
  '--reference-report', reportPath,
  '--conversion-config', conversionConfigPath,
  '--out', path.join(tmpRoot, 'output.pack.json'),
  '--json',
]);
assert.equal(parsed.manifest, manifestPath);
assert.equal(parsed.json, true);
assert.match(usage(), /Doppler Forge/);

// Test 2: buildForgeOptions
const options = await buildForgeOptions(parsed);
assert.equal(options.manifestPath, manifestPath);
assert.equal(options.referenceReportPath, reportPath);

// Test 3: forgeModelPack compilation
const outputPath = path.join(tmpRoot, 'compiled.pack.json');
const receipt = await forgeModelPack({
  ...options,
  outputPath,
});

assert.equal(receipt.ok, true);
assert.equal(receipt.forgeVersion, FORGE_VERSION);
assert.equal(receipt.modelId, 'forge-unit-model');
assert.equal(typeof receipt.bundleId, 'string');
assert.equal(receipt.wgslModuleCount, 1);
assert.equal(receipt.reachableKernelDigests[0], gatherDigest);
assert.ok(receipt.artifactCount >= 4);

// Verify file written to disk is valid JSON
const writtenRaw = await fs.readFile(outputPath, 'utf8');
const writtenPack = JSON.parse(writtenRaw);
assert.equal(writtenPack.schema, 'doppler.program-bundle/v1');
assert.equal(writtenPack.modelId, 'forge-unit-model');
assert.equal(writtenPack.wgslModules[0].file, 'gather.wgsl');

console.log('✔ forge-model-pack.test.js: all tests passed');
