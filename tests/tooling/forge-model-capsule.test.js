import assert from 'node:assert/strict';
import { createForgeEvaluationFixture } from '../helpers/forge-evaluation-fixture.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  FORGE_VERSION,
  buildForgeOptions,
  forgeModelCapsule,
  parseArgs,
  usage,
} from '../../tools/forge-model-capsule.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { createCapsuleReleaseFixture } from '../helpers/capsule-v2-fixture.js';
import { createEmbeddingReferenceFixture } from '../helpers/embedding-reference-fixture.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

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
const releaseManifestPath = path.join(fixtureRoot, 'capsule-release.json');
const promptTokenDigest = `sha256:${'a'.repeat(64)}`;
const logitsDigest = `sha256:${'b'.repeat(64)}`;
const kvByteDigest = `sha256:${'c'.repeat(64)}`;
const kvKeyDigest = `sha256:${'d'.repeat(64)}`;
const kvValueDigest = `sha256:${'e'.repeat(64)}`;
const shardBytes = Buffer.alloc(16);
const shardHash = createHash('sha256').update(shardBytes).digest('hex');

await fs.writeFile(path.join(modelDir, 'tokenizer.json'), '{"model":"unit"}\n', 'utf8');
await fs.writeFile(path.join(modelDir, 'shard_00000.bin'), shardBytes);
await fs.writeFile(conversionConfigPath, '{"modelId":"forge-unit-model"}\n', 'utf8');
await fs.writeFile(releaseManifestPath, `${JSON.stringify(createCapsuleReleaseFixture({
  targetIds: ['webgpu-f32-f32-portable'],
}), null, 2)}\n`, 'utf8');
await fs.writeFile(manifestPath, `${JSON.stringify({
  version: 1,
  modelId: 'forge-unit-model',
  modelType: 'llm',
  artifactIdentity: {
    sourceCheckpointId: 'test/forge-unit-model',
    sourceRepo: 'test/forge-unit-model',
    sourceRevision: 'fixture-revision',
  },
  architecture: {
    numLayers: 1,
    hiddenSize: 4,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 4,
    vocabSize: 8,
  },
  quantizationInfo: { weights: 'f32' },
  hashAlgorithm: 'sha256',
  shards: [
    {
      index: 0,
      filename: 'shard_00000.bin',
      size: 16,
      hash: shardHash,
      offset: 0,
    },
  ],
  tokenizer: {
    type: 'bundled',
    file: 'tokenizer.json',
  },
  tensors: {
    weight: { role: 'matmul', shape: [4, 4], dtype: 'F32' },
  },
  inference: {
    schema: 'doppler.execution/v1',
    attention: { causal: true, slidingWindow: 4, queryKeyNorm: false },
    normalization: { rmsNormEps: 1e-6, rmsNormWeightOffset: false },
    ffn: { activation: 'gelu', gatedActivation: false },
    rope: { ropeTheta: 10000, ropeLocalTheta: 10000 },
    output: { tieWordEmbeddings: false },
    layerPattern: { type: 'every_n', period: 1, offset: 0 },
    session: {
      compute: { defaults: { activationDtype: 'f32' } },
      kvcache: { kvDtype: 'f32', layout: 'contiguous' },
    },
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
  surface: 'node',
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
      generationConfig: {
        maxTokens: 1, temperature: 0, topP: 1, topK: 1,
        repetitionPenalty: 1, repetitionPenaltyWindow: 8,
        seed: null, useChatTemplate: false,
      },
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
      tokens: {
        ids: [42],
      },
      output: {
        tokensGenerated: 1,
        stopReason: 'max-tokens',
      },
    },
    sourceParity: {
      schema: 'doppler.source-token-parity/v1',
      status: 'passed',
      expectedTranscriptPath: 'reports/source.json',
      expectedTranscriptHash: `sha256:${'0'.repeat(64)}`,
      sourceModel: 'test/forge-unit-model',
      sourceRevision: 'fixture-revision',
      sampling: 'greedy',
      prompt: { passed: true, expectedCount: 3, observedCount: 3, firstMismatchIndex: null },
      generation: { passed: true, expectedCount: 1, observedCount: 1, firstMismatchIndex: null },
    },
  },
  output: ' blue',
}, null, 2)}\n`, 'utf8');

// Test 1: CLI parseArgs and usage
const parsed = parseArgs([
  '--manifest', manifestPath,
  '--reference-report', reportPath,
  '--conversion-config', conversionConfigPath,
  '--release-manifest', releaseManifestPath,
  '--out', path.join(tmpRoot, 'output.capsule.json'),
  '--json',
]);
assert.equal(parsed.manifest, manifestPath);
assert.equal(parsed.json, true);
assert.match(usage(), /Doppler Forge/);

// Test 2: buildForgeOptions
const options = await buildForgeOptions(parsed);
assert.equal(options.manifestPath, manifestPath);
assert.equal(options.referenceReportPath, reportPath);
assert.equal(options.releaseManifestPath, releaseManifestPath);

// Test 3: forgeModelCapsule compilation
const outputPath = path.join(tmpRoot, 'compiled.capsule.json');
const receipt = await forgeModelCapsule({
  ...options,
  outputPath,
});

assert.equal(receipt.ok, true);
assert.equal(receipt.forgeVersion, FORGE_VERSION);
assert.equal(receipt.modelId, 'forge-unit-model');
assert.equal(typeof receipt.capsuleId, 'string');
assert.match(receipt.semanticRoot, /^sha256:[0-9a-f]{64}$/);
assert.equal(receipt.wgslModuleCount, 1);
assert.ok(receipt.artifactCount >= 4);
assert.deepEqual(receipt.stages.map((stage) => stage.stage), [
  'inspect', 'normalize', 'analyze', 'lower', 'specialize',
  'search', 'verify', 'qualify', 'package', 'sign',
]);

// Verify file written to disk is valid JSON
const writtenRaw = await fs.readFile(outputPath, 'utf8');
const writtenCapsule = JSON.parse(writtenRaw);
assert.equal(writtenCapsule.schema, 'doppler.capsule/v2');
assert.equal(writtenCapsule.modelId, 'forge-unit-model');
assert.equal(writtenCapsule.wgslModules[0].file, 'gather.wgsl');
assert.ok(writtenCapsule.signature);

const secondOutputPath = path.join(tmpRoot, 'second', 'compiled.capsule.json');
const second = await forgeModelCapsule({ ...options, outputPath: secondOutputPath });
assert.equal(second.semanticRoot, receipt.semanticRoot);
assert.equal(second.envelopeHash, receipt.envelopeHash);

const evaluation = createForgeEvaluationFixture(writtenCapsule.targetPlans[0].modelIRHash, writtenCapsule.targetPlans.map(hashTargetPlan));
const candidateEvaluationPath = path.join(tmpRoot, 'evaluation.json');
const evaluationRaw = `${JSON.stringify(evaluation)}\n`;
await fs.writeFile(candidateEvaluationPath, evaluationRaw);
const evaluatedOptions = await buildForgeOptions(parseArgs(['--candidate-evaluation', candidateEvaluationPath]));
assert.equal(evaluatedOptions.candidateEvaluationPath, candidateEvaluationPath);
const evaluated = await forgeModelCapsule({ ...options, candidateEvaluationPath, outputPath: path.join(tmpRoot, 'evaluated', 'capsule.json') });
assert.equal(evaluated.semanticRoot, receipt.semanticRoot);
assert.equal(evaluated.searchReceipt.policy, 'observed-range-pareto');
assert.equal(evaluated.searchReceipt.evaluationReceipt.claimAllowed, false);
assert.equal(evaluated.candidateEvaluation.path, candidateEvaluationPath);
assert.equal(evaluated.candidateEvaluation.sizeBytes, Buffer.byteLength(evaluationRaw));
evaluation.observations[0].output.tokens = [999];
await fs.writeFile(candidateEvaluationPath, JSON.stringify(evaluation));
const rejectedOutput = path.join(tmpRoot, 'rejected-evaluation', 'capsule.json');
await assert.rejects(forgeModelCapsule({ ...options, candidateEvaluationPath, outputPath: rejectedOutput }), /no Capsule may be signed/);
await assert.rejects(fs.access(rejectedOutput), /ENOENT/);

const qualificationReportPath = path.join(reportDir, 'browser-qualification-report.json');
const qualificationReport = JSON.parse(await fs.readFile(reportPath, 'utf8'));
qualificationReport.metrics.referenceTranscript.surface = 'browser-webgpu';
qualificationReport.metrics.referenceTranscript.executionGraphHash = writtenCapsule.program.executionGraphHash;
await fs.writeFile(qualificationReportPath, `${JSON.stringify(qualificationReport, null, 2)}\n`, 'utf8');
const qualifiedOutputPath = path.join(tmpRoot, 'qualified', 'compiled.capsule.json');
const qualified = await forgeModelCapsule({
  ...options,
  programBundlePath: receipt.programBundlePath,
  outputPath: qualifiedOutputPath,
  qualificationReportPaths: [qualificationReportPath],
});
const qualifiedCapsule = JSON.parse(await fs.readFile(qualifiedOutputPath, 'utf8'));
assert.ok(qualifiedCapsule.targetPlans[0].qualification.some((entry) => entry.surface === 'browser-webgpu'));
const browserEvidence = qualifiedCapsule.artifacts.find((artifact) => artifact.role === 'qualification-evidence');
assert.ok(browserEvidence);
assert.equal(
  await fs.readFile(path.join(path.dirname(qualifiedOutputPath), browserEvidence.path), 'utf8'),
  await fs.readFile(qualificationReportPath, 'utf8')
);

const sourceModelIR = JSON.parse(await fs.readFile(
  'reports/model-ir-v2/qwen3.8-27b.model-ir-receipt.json',
  'utf8'
)).modelIR;
const capsuleModelIR = {
  ...sourceModelIR,
  modelId: 'forge-unit-model',
  sourceIdentity: {
    ...sourceModelIR.sourceIdentity,
    checkpointId: 'test/forge-unit-model',
    repository: 'test/forge-unit-model',
    revision: 'fixture-revision',
  },
};
const modelIRReceiptPath = path.join(tmpRoot, 'forge-unit-model.model-ir.json');
const modelIRReceiptRaw = `${JSON.stringify({ modelIR: capsuleModelIR }, null, 2)}\n`;
await fs.writeFile(modelIRReceiptPath, modelIRReceiptRaw, 'utf8');
const identityPath = path.join(tmpRoot, 'forge-unit-model.initial-identity.json');
const initialExecutionIdentity = createInitialExecutionIdentityV2({
  executionGraphHash: writtenCapsule.program.executionGraphHash,
  resolvedGraphHash: `sha256:${'6'.repeat(64)}`,
  kernelClosure: [{ moduleId: 'embed', file: 'gather.wgsl', entry: 'main', digest: gatherDigest }],
  dtypeLane: { activation: 'f32', kv: 'f32' },
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
await fs.writeFile(identityPath, `${JSON.stringify(initialExecutionIdentity, null, 2)}\n`, 'utf8');
const v2OutputPath = path.join(tmpRoot, 'v2', 'compiled.capsule.json');
await forgeModelCapsule({
  ...options,
  programBundlePath: receipt.programBundlePath,
  outputPath: v2OutputPath,
  modelIRReceiptPath,
  initialExecutionIdentityPath: identityPath,
});
const v2Capsule = JSON.parse(await fs.readFile(v2OutputPath, 'utf8'));
assert.equal(v2Capsule.modelIR.schema, 'doppler.model-ir/v2');
assert.deepEqual(v2Capsule.modelIR.supportScope.qualifiedEntryPoints, ['text.generate']);
const modelIREvidence = v2Capsule.artifacts.find((artifact) => (
  artifact.artifactId === v2Capsule.program.modelIREvidenceArtifactId
));
assert.equal(modelIREvidence.role, 'source-truth-evidence');
assert.equal(
  await fs.readFile(path.join(path.dirname(v2OutputPath), modelIREvidence.path), 'utf8'),
  modelIRReceiptRaw
);

console.log('✔ forge-model-capsule.test.js: all tests passed');

// Embeddings require their own qualification transcript, with no invented
// generation tokens. Additional reports must bind the same frozen source.
const embeddingFixture = createEmbeddingReferenceFixture();
const embeddingManifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
embeddingManifest.modelType = 'embedding';
embeddingManifest.artifactIdentity = { ...embeddingManifest.artifactIdentity,
  sourceCheckpointId: embeddingFixture.reference.source.checkpointId,
  sourceRepo: embeddingFixture.reference.source.repository,
  sourceRevision: embeddingFixture.reference.source.revision };
embeddingManifest.inference.output.embeddingPostprocessor = embeddingFixture.reference.embeddingContract.postprocessor;
const embeddingManifestPath = path.join(modelDir, 'embedding-manifest.json');
const embeddingManifestRaw = JSON.stringify(embeddingManifest);
await fs.writeFile(embeddingManifestPath, embeddingManifestRaw);
const embeddingReport = {
  schema: 'doppler.embeddingModelQualification.v1', passed: true,
  model: { modelId: embeddingManifest.modelId, manifestHash: 'sha256:' + createHash('sha256').update(embeddingManifestRaw).digest('hex'),
    artifactIdentity: embeddingManifest.artifactIdentity },
  runtime: { surface: 'browser-webgpu', executionGraphHash: writtenCapsule.program.executionGraphHash, adapterInfo: { vendor: 'synthetic' } },
  reference: embeddingFixture.reference, referenceDigest: embeddingFixture.referenceDigest, observation: embeddingFixture.observation,
};
const embeddingReportPath = path.join(reportDir, 'embedding.json');
await fs.writeFile(embeddingReportPath, JSON.stringify(embeddingReport));
const embeddingOptions = { ...options, createdAtUtc: '2026-09-05T00:00:00.000Z', manifestPath: embeddingManifestPath, referenceReportPath: embeddingReportPath,
  outputPath: path.join(tmpRoot, 'embedding', 'capsule.json') };
const embedded = await forgeModelCapsule(embeddingOptions);
assert.equal(embedded.ok, true);
await forgeModelCapsule({ ...embeddingOptions, outputPath: path.join(tmpRoot, 'embedding-extra', 'capsule.json'),
  qualificationReportPaths: [embeddingReportPath] });
const other = structuredClone(embeddingReport);
other.reference.tolerances.embeddingMaxAbs = 0.02;
other.referenceDigest = computeCanonicalSha256(other.reference);
const otherPath = path.join(reportDir, 'embedding-other.json');
await fs.writeFile(otherPath, JSON.stringify(other));
await assert.rejects(forgeModelCapsule({ ...embeddingOptions, qualificationReportPaths: [otherPath] }), /same model, source reference, and request/);
console.log('forge-model-capsule: embedding qualification passed (synthetic reference)');
