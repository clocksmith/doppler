import assert from 'node:assert/strict';
import { runForgePipeline, stageAnalyze } from '../../src/converter/forge-stages.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import { TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';

const privateKeyJwk = {
  ...TEST_PACK_PUBLIC_KEY,
  d: 'WQi2FHRfw0jZxl_IXiMp5TAuehMfssojWd2Oj3WaUKU',
};
const manifest = {
  modelId: 'forge-model', modelType: 'transformer',
  artifactIdentity: { sourceCheckpointId: 'test/forge-model' },
  architecture: {
    numLayers: 1, hiddenSize: 4, intermediateSize: 8,
    numAttentionHeads: 1, numKeyValueHeads: 1, headDim: 4, vocabSize: 8,
  },
  inference: {
    attention: { causal: true, slidingWindow: 4, queryKeyNorm: true },
    normalization: { rmsNormEps: 1e-6, rmsNormWeightOffset: true },
    ffn: { activation: 'gelu', gatedActivation: true },
    rope: { ropeTheta: 10000, ropeLocalTheta: 10000 },
    output: { tieWordEmbeddings: false },
    layerPattern: { type: 'every_n', period: 1, offset: 0 },
    session: {
      compute: { defaults: { activationDtype: 'f32' } },
      kvcache: { kvDtype: 'f32', layout: 'contiguous' },
    },
  },
  quantizationInfo: { weights: 'f32' },
  shards: [{ filename: 'weights.bin', size: 4, hash: '1'.repeat(64) }],
  tensors: { weight: { role: 'matmul', shape: [4, 4], dtype: 'F32' } },
};
const manifestRaw = `${JSON.stringify(manifest)}\n`;
const hash = (value) => `sha256:${sha256Hex(value)}`;
const wgslHash = hash('@compute @workgroup_size(1) fn main() {}\n');
const graphHash = `sha256:${'2'.repeat(64)}`;
const artifacts = [
  { role: 'manifest', path: 'manifest.json', hash: hash(manifestRaw), sizeBytes: manifestRaw.length },
  { role: 'tokenizer', path: 'tokenizer.json', hash: `sha256:${'3'.repeat(64)}`, sizeBytes: 1 },
  { role: 'weight-shard', path: 'weights.bin', hash: `sha256:${'4'.repeat(64)}`, sizeBytes: 4 },
  { role: 'wgsl-source', path: 'program/wgsl/main.wgsl', hash: wgslHash, sizeBytes: 45 },
  { role: 'reference-report', path: 'reference.json', hash: `sha256:${'5'.repeat(64)}`, sizeBytes: 1 },
];
const programBundle = {
  schema: 'doppler.program-bundle/v1', schemaVersion: 1, bundleId: 'fixture', modelId: manifest.modelId,
  createdAtUtc: '2026-08-22T00:00:00.000Z',
  sources: { manifest: { hash: hash(manifestRaw) }, executionGraph: { hash: graphHash } },
  artifacts,
  execution: {
    graphHash,
    steps: [
      { id: 'prefill', phase: 'prefill' },
      { id: 'decode', phase: 'decode' },
    ],
  },
  wgslModules: [{
    id: 'main', file: 'main.wgsl', entry: 'main', digest: wgslHash,
    sourceHash: wgslHash, metadata: { requiresSubgroups: false },
  }],
  captureProfile: { surfaces: ['test-webgpu'] },
  referenceTranscript: {
    generationConfig: { temperature: 0 },
    tokens: { ids: [1, 2, 3, 4] },
  },
};
const programBundleRaw = `${JSON.stringify(programBundle)}\n`;

const result = await runForgePipeline({
  manifest, manifestRaw, programBundle, programBundleRaw,
  programBundlePath: '/tmp/program-bundle.json', repoRoot: '/tmp', outputPath: '/tmp/model.pack.json',
}, {
  authority: TEST_PACK_AUTHORITY,
  privateKeyJwk,
  publicKeyJwk: TEST_PACK_PUBLIC_KEY,
});
assert.deepEqual(result.stages.map((stage) => stage.stage), [
  'inspect', 'normalize', 'analyze', 'lower', 'specialize',
  'search', 'verify', 'qualify', 'package', 'sign',
]);
assert.equal(result.pack.schema, 'doppler.pack/v2');
assert.equal(result.pack.signature.authority, TEST_PACK_AUTHORITY);
assert.equal(result.pack.modelIR.hiddenSize, 4);
assert.equal(result.pack.targetPlans.length, 1, 'Forge must not invent unsupported target variants');

assert.throws(
  () => stageAnalyze({ manifest: { ...manifest, architecture: { ...manifest.architecture, headDim: undefined } }, artifacts, manifestHash: hash(manifestRaw) }),
  /headDim/
);

console.log('✔ forge-stages.test.js passed');
