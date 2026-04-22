import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { exportProgramBundle } from '../../src/tooling/program-bundle.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const repoRoot = process.cwd();
const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-program-bundle-'));
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
await fs.writeFile(conversionConfigPath, '{"modelId":"unit-model"}\n', 'utf8');
await fs.writeFile(manifestPath, `${JSON.stringify({
  version: 1,
  modelId: 'unit-model',
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
        embed_alias: {
          kernel: 'gather.wgsl',
          entry: 'main',
          digest: gatherDigest,
        },
      },
      preLayer: [['embed', 'embed_alias']],
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
  modelId: 'unit-model',
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

const bundle = await exportProgramBundle({
  repoRoot,
  manifestPath,
  modelDir,
  referenceReportPath: reportPath,
  conversionConfigPath,
  createdAtUtc: '2026-04-22T00:00:00.000Z',
});

assert.equal(bundle.modelId, 'unit-model');
assert.equal(bundle.artifacts.filter((artifact) => artifact.role === 'weight-shard').length, 1);
assert.equal(bundle.wgslModules.length, 1);
assert.equal(bundle.wgslModules[0].id, 'embed_alias');
assert.equal(bundle.wgslModules[0].digest, gatherDigest);
assert.equal(bundle.execution.kernelClosure.expandedStepCount, 1);
assert.equal(bundle.execution.steps[0].kernelId, 'embed_alias');
assert.equal(bundle.referenceTranscript.output.tokensGenerated, 1);
assert.equal(bundle.referenceTranscript.phase.prefillTokens, 3);
assert.equal(bundle.referenceTranscript.prompt.tokenIdsHash, promptTokenDigest);
assert.equal(bundle.referenceTranscript.prompt.tokenCount, 3);
assert.equal(bundle.referenceTranscript.logits.mode, 'sha256-per-step');
assert.deepEqual(bundle.referenceTranscript.logits.perStepDigests, [logitsDigest]);
assert.equal(bundle.referenceTranscript.kvCache.byteDigest, kvByteDigest);
assert.equal(bundle.referenceTranscript.kvCache.byteDigests[0].keyDigest, kvKeyDigest);

await fs.writeFile(path.join(reportDir, 'manual-receipt.json'), `${JSON.stringify({
  suite: 'debug',
  modelId: 'unit-model',
  results: [{ name: 'generation', passed: true }],
  notes: 'manual receipt without token transcript',
}, null, 2)}\n`, 'utf8');

await assert.rejects(
  () => exportProgramBundle({
    repoRoot,
    manifestPath,
    modelDir,
    referenceReportPath: path.join(reportDir, 'manual-receipt.json'),
    conversionConfigPath,
  }),
  /reference report must include metrics/
);

await assert.rejects(
  () => exportProgramBundle({
    repoRoot,
    manifestPath: path.join(
      repoRoot,
      'models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json'
    ),
    modelDir: path.join(
      repoRoot,
      'models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple'
    ),
    referenceReportPath: path.join(
      repoRoot,
      'tests/fixtures/reports/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/2026-04-16T00-00-00.000Z.json'
    ),
    conversionConfigPath: path.join(
      repoRoot,
      'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json'
    ),
  }),
  /reference report must include metrics\.prompt or metrics\.promptInput/
);

await fs.rm(tmpRoot, { recursive: true, force: true });

console.log('program-bundle-exporter.test: ok');
