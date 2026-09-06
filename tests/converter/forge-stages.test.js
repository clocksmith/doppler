import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { runForgePipeline, stageAnalyze } from '../../src/converter/forge-stages.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createRerankReferenceFixture } from '../helpers/rerank-reference-fixture.js';
import { createEmbeddingReferenceFixture } from '../helpers/embedding-reference-fixture.js';
import { createForgeEvaluationFixture } from '../helpers/forge-evaluation-fixture.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import {
  TEST_CAPSULE_AUTHORITY,
  TEST_CAPSULE_PUBLIC_KEY,
  createCapsuleReleaseFixture,
} from '../helpers/capsule-v2-fixture.js';

const privateKeyJwk = {
  ...TEST_CAPSULE_PUBLIC_KEY,
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
const release = createCapsuleReleaseFixture({ targetIds: ['webgpu-f32-f32-portable'] });

const result = await runForgePipeline({
  manifest, manifestRaw, programBundle, programBundleRaw,
  programBundlePath: '/tmp/program-bundle.json', repoRoot: '/tmp', outputPath: '/tmp/model.capsule.json',
  release,
}, {
  authority: TEST_CAPSULE_AUTHORITY,
  privateKeyJwk,
  publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
});
assert.deepEqual(result.stages.map((stage) => stage.stage), [
  'inspect', 'normalize', 'analyze', 'lower', 'specialize',
  'search', 'verify', 'qualify', 'package', 'sign',
]);
assert.equal(result.capsule.schema, 'doppler.capsule/v2');
assert.equal(result.capsule.signature.authority, TEST_CAPSULE_AUTHORITY);
assert.equal(result.capsule.modelIR.hiddenSize, 4);
assert.equal(result.capsule.targetPlans.length, 1, 'Forge must not invent unsupported target variants');
const candidateEvaluation = createForgeEvaluationFixture(result.capsule.targetPlans[0].modelIRHash,
  result.capsule.targetPlans.map(hashTargetPlan));
const evaluatedCapsule = await runForgePipeline({ manifest, manifestRaw, programBundle, programBundleRaw,
  programBundlePath: '/tmp/program-bundle.json', repoRoot: '/tmp', outputPath: '/tmp/model.capsule.json', release, candidateEvaluation }, {
  authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
});
assert.equal(evaluatedCapsule.searchReceipt.policy, 'observed-range-pareto');
assert.equal(evaluatedCapsule.searchReceipt.evaluationReceipt.promotionAllowed, false);
assert.equal(evaluatedCapsule.capsule.semanticRoot, result.capsule.semanticRoot, 'selection must not rewrite executable identity');
const failedEvaluation = structuredClone(candidateEvaluation);
failedEvaluation.observations[0].output.tokens = [9];
await assert.rejects(runForgePipeline({ manifest, manifestRaw, programBundle, programBundleRaw,
  programBundlePath: '/tmp/program-bundle.json', repoRoot: '/tmp', outputPath: '/tmp/model.capsule.json', release,
  candidateEvaluation: failedEvaluation }, { authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY }),
error => /no Capsule may be signed/.test(error.message) && error.evaluationReceipt.selectedCandidateHashes.length === 0);

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
    schema: 'doppler.capsule-program-load-policy/v2',
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
  outputPath: '/tmp/model-v2.capsule.json',
  modelIR: modelIRV2,
  modelIREvidence,
  initialExecutionIdentity,
  release,
}, {
  authority: TEST_CAPSULE_AUTHORITY,
  privateKeyJwk,
  publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
});
assert.equal(v2Result.capsule.modelIR.schema, 'doppler.model-ir/v2');
assert.deepEqual(v2Result.capsule.modelIR.supportScope.qualifiedEntryPoints, ['text.generate']);
assert.equal(v2Result.capsule.targetPlans[0].schema, 'doppler.target-plan/v2');
assert.equal(v2Result.capsule.targetPlans[0].initialExecutionIdentity.digest, initialExecutionIdentity.digest);
assert.equal(
  v2Result.capsule.artifacts.find((artifact) => (
    artifact.artifactId === v2Result.capsule.program.modelIREvidenceArtifactId
  ))?.role,
  'source-truth-evidence'
);
assert.ok(v2Result.capsule.targetPlans[0].memoryLayout.bufferSlots.some((slot) => slot.slotId === 'recurrent_state'));
assert.ok(v2Result.capsule.targetPlans[0].memoryLayout.bufferSlots.some((slot) => slot.slotId === 'convolutional_state'));

// Synthetic source closure only: proves Forge preserves declared adapter policy
// through signing, not that these fixture shader bytes implement LoRA.
const adapterModules = ['matmul_f16.wgsl', 'scale.wgsl', 'residual.wgsl'].map(file => ({
  ...v2ProgramBundle.wgslModules[0], id: `adapter-${file}`, file,
}));
const adapterBundle = { ...v2ProgramBundle, wgslModules: [...v2ProgramBundle.wgslModules, ...adapterModules] };
const adapterExecution = { schema: 'doppler.capsule-adapter-execution/v1', maxAdapters: 1,
  combination: 'single', formats: ['peft_safetensors'], operations: ['generate'],
  kernelModules: adapterModules.map(module => module.id) };
const adapterIdentity = createInitialExecutionIdentityV2({ ...initialExecutionIdentity,
  kernelClosure: [...initialExecutionIdentity.kernelClosure,
    ...adapterModules.map(module => ({ moduleId: module.id, file: module.file, entry: module.entry, digest: module.digest }))],
});
const adapterInput = { manifest: v2Manifest, manifestRaw: v2ManifestRaw, programBundle: adapterBundle,
  programBundleRaw: JSON.stringify(adapterBundle), programBundlePath: '/tmp/adapter-bundle.json',
  repoRoot: '/tmp', outputPath: '/tmp/adapter-model.capsule.json', modelIR: modelIRV2,
  modelIREvidence, initialExecutionIdentity: adapterIdentity, adapterExecution, release };
const adapterResult = await runForgePipeline(adapterInput,
  { authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY });
assert.deepEqual(adapterResult.capsule.targetPlans[0].adapterExecution, adapterExecution);
assert.notEqual(adapterResult.capsule.semanticRoot, v2Result.capsule.semanticRoot);
await assert.rejects(runForgePipeline({ ...adapterInput, adapterExecution: {
  ...adapterExecution, kernelModules: [...adapterExecution.kernelModules, 'undeclared'],
} }, { authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY }), /outside signed execution closure/);

// Actual Forge stages with synthetic source/output evidence, not hardware proof.
const rerankIR = structuredClone(modelIRV2);
rerankIR.sourceIdentity.revision = '1'.repeat(40);
const rerankEntry = rerankIR.entryPoints.find((entry) => entry.kind === 'generate');
rerankEntry.kind = 'rerank';
rerankEntry.phases = ['prefill'];
const rerankTranscript = createRerankReferenceFixture();
Object.assign(rerankTranscript.reference.source, {
  checkpointId: rerankIR.sourceIdentity.checkpointId,
  repository: rerankIR.sourceIdentity.repository,
});
rerankTranscript.referenceDigest = computeCanonicalSha256(rerankTranscript.reference);
const rerankManifest = structuredClone(v2Manifest);
rerankManifest.artifactIdentity.sourceRevision = rerankIR.sourceIdentity.revision;
rerankManifest.inference.supportsRerank = true;
rerankManifest.inference.rerank = rerankTranscript.reference.scoringConfig;
const rerankManifestRaw = `${JSON.stringify(rerankManifest)}\n`;
Object.assign(rerankTranscript, {
  modelId: rerankManifest.modelId, manifestHash: hash(rerankManifestRaw), executionGraphHash: graphHash,
});
const rerankBundle = structuredClone(v2ProgramBundle);
rerankBundle.referenceTranscript = rerankTranscript;
rerankBundle.sources.manifest.hash = hash(rerankManifestRaw);
Object.assign(rerankBundle.artifacts.find((artifact) => artifact.role === 'manifest'), {
  hash: hash(rerankManifestRaw), sizeBytes: rerankManifestRaw.length,
});
const rerankEvidenceRaw = JSON.stringify({ modelIR: rerankIR });
const rerankInput = {
  manifest: rerankManifest, manifestRaw: rerankManifestRaw, programBundle: rerankBundle,
  programBundleRaw: JSON.stringify(rerankBundle), programBundlePath: '/tmp/rerank-bundle.json',
  repoRoot: '/tmp', outputPath: '/tmp/rerank.capsule.json', modelIR: rerankIR,
  modelIREvidence: { sourcePath: '/tmp/rerank-ir.json', hash: hash(rerankEvidenceRaw), sizeBytes: rerankEvidenceRaw.length },
  initialExecutionIdentity, release,
};
const signer = { authority: TEST_CAPSULE_AUTHORITY, privateKeyJwk, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY };
const rerankResult = await runForgePipeline(rerankInput, signer);
assert.equal(rerankResult.capsule.modelIR.schema, 'doppler.model-ir/v2');
assert.deepEqual(rerankResult.capsule.modelIR.supportScope.qualifiedEntryPoints, [rerankEntry.id]);
assert.equal(rerankResult.capsule.targetPlans[0].qualification[0].operation, 'rerank');
for (const [change, expected] of [
  [(input) => { input.programBundle.referenceTranscript.observation.outputs[0].score += 10; }, /source comparison failed/],
  [(input) => {
    input.programBundle.referenceTranscript.reference.source.revision = '2'.repeat(40);
    input.programBundle.referenceTranscript.referenceDigest = computeCanonicalSha256(input.programBundle.referenceTranscript.reference);
  }, /source.*identity/i],
  [(input) => { input.modelIR.entryPoints.find((entry) => entry.kind === 'rerank').kind = 'generate'; }, /lowered rerank/],
  [(input) => { input.programBundle.referenceTranscript.surface = 'unknown-webgpu'; }, /explicit physical/],
]) {
  const invalid = structuredClone(rerankInput);
  change(invalid);
  await assert.rejects(() => runForgePipeline(invalid, signer), expected);
}

const embeddingInput = structuredClone(rerankInput);
embeddingInput.modelIR.entryPoints.find(entry => entry.kind === 'rerank').kind = 'embed';
embeddingInput.manifest.modelType = 'embedding';
delete embeddingInput.manifest.inference.supportsRerank;
delete embeddingInput.manifest.inference.rerank;
const embeddingTranscript = createEmbeddingReferenceFixture();
Object.assign(embeddingTranscript.reference.source, {
  checkpointId: embeddingInput.modelIR.sourceIdentity.checkpointId,
  repository: embeddingInput.modelIR.sourceIdentity.repository,
});
embeddingTranscript.referenceDigest = computeCanonicalSha256(embeddingTranscript.reference);
embeddingInput.manifest.inference.output.embeddingPostprocessor = embeddingTranscript.reference.embeddingContract.postprocessor;
embeddingInput.manifestRaw = JSON.stringify(embeddingInput.manifest);
Object.assign(embeddingTranscript, { modelId: embeddingInput.manifest.modelId,
  manifestHash: hash(embeddingInput.manifestRaw), executionGraphHash: graphHash });
embeddingInput.programBundle.referenceTranscript = embeddingTranscript;
embeddingInput.programBundle.sources.manifest.hash = embeddingTranscript.manifestHash;
Object.assign(embeddingInput.programBundle.artifacts.find(artifact => artifact.role === 'manifest'), {
  hash: embeddingTranscript.manifestHash, sizeBytes: embeddingInput.manifestRaw.length,
});
embeddingInput.programBundleRaw = JSON.stringify(embeddingInput.programBundle);
const embeddingResult = await runForgePipeline(embeddingInput, signer);
assert.equal(embeddingResult.capsule.targetPlans[0].qualification[0].operation, 'embed');
assert.equal(embeddingResult.capsule.targetPlans[0].qualification[0].embeddedTexts, 2);
assert.deepEqual(embeddingResult.capsule.modelIR.supportScope.qualifiedEntryPoints, [rerankEntry.id]);
for (const [change, expected] of [
  [input => { input.programBundle.referenceTranscript.observation.outputs[0].embedding[3] = 1; }, /source comparison failed/],
  [input => { input.modelIR.entryPoints.find(entry => entry.kind === 'embed').kind = 'generate'; }, /lowered embed/],
  [input => { input.programBundle.referenceTranscript.surface = 'unknown-webgpu'; }, /explicit physical/],
  [input => { input.programBundle.captureProfile.surfaces = ['different-webgpu']; }, /capture surface/],
]) {
  const invalid = structuredClone(embeddingInput);
  change(invalid);
  await assert.rejects(() => runForgePipeline(invalid, signer), expected);
}

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
    schema: 'doppler.capsule-program-load-policy/v2',
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
    outputPath: '/tmp/model-v2.capsule.json',
    modelIR: modelIRV2,
    modelIREvidence,
    initialExecutionIdentity: wrongKernelIdentity,
    release,
  }, {
    authority: TEST_CAPSULE_AUTHORITY,
    privateKeyJwk,
    publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
  }),
  /kernel closure different from the observed initial execution/
);

assert.throws(
  () => stageAnalyze({ manifest: { ...manifest, architecture: { ...manifest.architecture, headDim: undefined } }, artifacts, manifestHash: hash(manifestRaw) }),
  /headDim/
);

console.log('✔ forge-stages.test.js passed');
