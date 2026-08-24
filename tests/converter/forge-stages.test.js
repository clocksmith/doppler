import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { runForgePipeline, stageAnalyze } from '../../src/converter/forge-stages.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import {
  TEST_PACK_AUTHORITY,
  TEST_PACK_PUBLIC_KEY,
  createPackReleaseFixture,
} from '../helpers/pack-v2-fixture.js';

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
    surface: 'test-webgpu',
    sourceParity: {
      schema: 'doppler.source-token-parity/v1',
      status: 'passed',
      expectedTranscriptPath: 'reports/source.json',
      expectedTranscriptHash: `sha256:${'8'.repeat(64)}`,
      sourceModel: 'test/forge-model',
      sourceRevision: 'fixture-revision',
      sampling: 'greedy',
      prompt: { passed: true, expectedCount: 1, observedCount: 1, firstMismatchIndex: null },
      generation: { passed: true, expectedCount: 4, observedCount: 4, firstMismatchIndex: null },
    },
    generationConfig: { temperature: 0 },
    tokens: { ids: [1, 2, 3, 4] },
  },
};
const programBundleRaw = `${JSON.stringify(programBundle)}\n`;
const release = createPackReleaseFixture({ targetIds: ['webgpu-f32-f32-portable'] });

const result = await runForgePipeline({
  manifest, manifestRaw, programBundle, programBundleRaw,
  programBundlePath: '/tmp/program-bundle.json', repoRoot: '/tmp', outputPath: '/tmp/model.pack.json',
  release,
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

const qwenReceipt = JSON.parse(await fs.readFile(
  'reports/model-ir-v2/qwen3.8-27b.model-ir-receipt.json',
  'utf8'
));
const modelIRV2 = {
  ...qwenReceipt.modelIR,
  modelId: manifest.modelId,
  sourceIdentity: {
    ...qwenReceipt.modelIR.sourceIdentity,
    checkpointId: manifest.artifactIdentity.sourceCheckpointId,
    repository: 'test/forge-model',
    revision: 'fixture-revision',
  },
};
const modelIREvidenceRaw = `${JSON.stringify({ modelIR: modelIRV2 })}\n`;
const modelIREvidence = {
  sourcePath: '/tmp/model-ir-receipt.json',
  hash: hash(modelIREvidenceRaw),
  sizeBytes: modelIREvidenceRaw.length,
};
const v2Manifest = {
  ...manifest,
  artifactIdentity: {
    ...manifest.artifactIdentity,
    sourceRepo: 'test/forge-model',
    sourceRevision: 'fixture-revision',
  },
  inference: {
    ...manifest.inference,
    linearAttention: { stateDtype: 'f32' },
  },
};
const v2ManifestRaw = `${JSON.stringify(v2Manifest)}\n`;
const v2ProgramBundle = {
  ...programBundle,
  sources: {
    ...programBundle.sources,
    manifest: { hash: hash(v2ManifestRaw) },
  },
  artifacts: programBundle.artifacts.map((artifact) => (
    artifact.role === 'manifest'
      ? { ...artifact, hash: hash(v2ManifestRaw), sizeBytes: v2ManifestRaw.length }
      : artifact
  )),
};
const initialExecutionIdentity = createInitialExecutionIdentityV2({
  executionGraphHash: graphHash,
  resolvedGraphHash: `sha256:${'6'.repeat(64)}`,
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: wgslHash }],
  dtypeLane: { activation: 'f32', kv: 'f32', weight: 'f32' },
  fusionSet: [],
  kvLayout: { layout: 'contiguous' },
  memoryPolicy: { kvcache: { layout: 'contiguous' } },
  executionPlanDigest: `sha256:${'7'.repeat(64)}`,
  runtimeEngine: { schema: 'fixture' },
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v2',
    runtimeConfig: {
      inference: {
        session: {}, compute: {}, generation: { disableMultiTokenDecode: false },
      },
    },
  },
});
const v2Result = await runForgePipeline({
  manifest: v2Manifest,
  manifestRaw: v2ManifestRaw,
  programBundle: v2ProgramBundle,
  programBundleRaw: `${JSON.stringify(v2ProgramBundle)}\n`,
  programBundlePath: '/tmp/program-bundle-v2.json',
  repoRoot: '/tmp',
  outputPath: '/tmp/model-v2.pack.json',
  modelIR: modelIRV2,
  modelIREvidence,
  initialExecutionIdentity,
  release,
}, {
  authority: TEST_PACK_AUTHORITY,
  privateKeyJwk,
  publicKeyJwk: TEST_PACK_PUBLIC_KEY,
});
assert.equal(v2Result.pack.modelIR.schema, 'doppler.model-ir/v2');
assert.deepEqual(v2Result.pack.modelIR.supportScope.qualifiedEntryPoints, ['text.generate']);
assert.equal(v2Result.pack.targetPlans[0].schema, 'doppler.target-plan/v2');
assert.equal(v2Result.pack.targetPlans[0].initialExecutionIdentity.digest, initialExecutionIdentity.digest);
assert.equal(
  v2Result.pack.artifacts.find((artifact) => (
    artifact.artifactId === v2Result.pack.program.modelIREvidenceArtifactId
  ))?.role,
  'source-truth-evidence'
);
assert.ok(v2Result.pack.targetPlans[0].memoryLayout.bufferSlots.some((slot) => slot.slotId === 'recurrent_state'));
assert.ok(v2Result.pack.targetPlans[0].memoryLayout.bufferSlots.some((slot) => slot.slotId === 'convolutional_state'));

const wrongKernelIdentity = createInitialExecutionIdentityV2({
  executionGraphHash: graphHash,
  resolvedGraphHash: `sha256:${'6'.repeat(64)}`,
  kernelClosure: [{ moduleId: 'other', file: 'other.wgsl', entry: 'main', digest: wgslHash }],
  dtypeLane: { activation: 'f32', kv: 'f32', weight: 'f32' },
  fusionSet: [],
  kvLayout: { layout: 'contiguous' },
  memoryPolicy: { kvcache: { layout: 'contiguous' } },
  executionPlanDigest: `sha256:${'7'.repeat(64)}`,
  runtimeEngine: { schema: 'fixture' },
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v2',
    runtimeConfig: {
      inference: {
        session: {}, compute: {}, generation: { disableMultiTokenDecode: false },
      },
    },
  },
});
await assert.rejects(
  () => runForgePipeline({
    manifest: v2Manifest,
    manifestRaw: v2ManifestRaw,
    programBundle: v2ProgramBundle,
    programBundleRaw: `${JSON.stringify(v2ProgramBundle)}\n`,
    programBundlePath: '/tmp/program-bundle-v2.json',
    repoRoot: '/tmp',
    outputPath: '/tmp/model-v2.pack.json',
    modelIR: modelIRV2,
    modelIREvidence,
    initialExecutionIdentity: wrongKernelIdentity,
    release,
  }, {
    authority: TEST_PACK_AUTHORITY,
    privateKeyJwk,
    publicKeyJwk: TEST_PACK_PUBLIC_KEY,
  }),
  /kernel closure different from the observed initial execution/
);

assert.throws(
  () => stageAnalyze({ manifest: { ...manifest, architecture: { ...manifest.architecture, headDim: undefined } }, artifacts, manifestHash: hash(manifestRaw) }),
  /headDim/
);

console.log('✔ forge-stages.test.js passed');
