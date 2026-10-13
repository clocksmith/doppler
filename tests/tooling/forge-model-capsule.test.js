import assert from 'node:assert/strict';
import { createForgeEvaluationFixture } from '../helpers/forge-evaluation-fixture.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  RIG_VERSION as FORGE_VERSION,
  buildRigOptions as buildForgeOptions,
  rigModelCapsule as forgeModelCapsule,
  parseArgs,
  usage,
} from '../../tools/rig-model-capsule.js';
import * as legacyForge from '../../tools/forge-model-capsule.js';
assert.equal(forgeModelCapsule, legacyForge.forgeModelCapsule);
assert.equal(buildForgeOptions, legacyForge.buildForgeOptions);
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
    numLayers: 2,
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
assert.match(usage(), /Doppler Rig/);

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
const second = await legacyForge.forgeModelCapsule({ ...options, outputPath: secondOutputPath });
assert.equal(second.semanticRoot, receipt.semanticRoot);
assert.equal(second.envelopeHash, receipt.envelopeHash);
assert.equal(await fs.readFile(secondOutputPath, 'utf8'), writtenRaw);

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
        session: { intermediatePrecision: 'f32', linearRecurrentStatePrecision: 'f32' }, compute: {}, generation: { disableMultiTokenDecode: false },
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

// Entirely synthetic producer wiring. These records do not qualify a physical model.
const partitionContract = await import('../../src/inference/pipelines/text/layer-partition-contract.js');
const physicalManifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
const partitionPlan = partitionContract.createLayerPartitionPlan({ modelId: physicalManifest.modelId,
  ...physicalManifest.architecture, splitLayer: 1, activationDtype: 'f32' });
const manifestHash = 'sha256:' + createHash('sha256').update(await fs.readFile(manifestPath)).digest('hex');
const indexPath = path.join(modelDir, 'pieces.json');
const pieceIndex = { schema: 'doppler.verified-pieces/v1', manifestIdentity: manifestHash, maxPieceBytes: 16,
  files: [{ path: 'shard_00000.bin', size: 16, pieces: [{ offset: 0, size: 16, identity: 'sha256:' + shardHash }] }] };
await fs.writeFile(indexPath, JSON.stringify(pieceIndex));
const residentContract = { schema: 'doppler.resident-qualification-contract/v1', referenceHash: promptTokenDigest,
  evaluatorHash: kvByteDigest, tolerance: 0.001, comparedSteps: 1 };
const residentContractPath = path.join(reportDir, 'resident-contract.json');
await fs.writeFile(residentContractPath, JSON.stringify(residentContract));
const residentReport = { schema: 'doppler.resident-partition-qualification/v1', operation: 'residentPartition',
  modelId: physicalManifest.modelId, manifestIdentity: manifestHash, surface: 'node-webgpu', status: 'passed',
  executionGraphHash: writtenCapsule.program.executionGraphHash, initialExecutionIdentity,
  partitionPlan, partitionPlanHash: partitionContract.hashLayerPartitionPlan(partitionPlan), partitionIndex: 0,
  comparison: { contractHash: computeCanonicalSha256(residentContract), referenceHash: promptTokenDigest, captureHash: logitsDigest, evaluatorHash: kvByteDigest,
    tolerance: 0.001, steps: [{ index: 0, maxAbsoluteError: 0.0002 }], complete: true },
};
const residentReportPath = path.join(reportDir, 'resident.json');
await fs.writeFile(residentReportPath, JSON.stringify(residentReport));
const residentOptions = { ...options, programBundlePath: receipt.programBundlePath, modelIRReceiptPath,
  initialExecutionIdentityPath: identityPath, residentQualificationContractPath: residentContractPath, pieceIndexPath: indexPath,
  qualificationReportPaths: [residentReportPath], outputPath: path.join(tmpRoot, 'resident', 'capsule.json') };
await forgeModelCapsule(residentOptions);
const residentCapsule = JSON.parse(await fs.readFile(residentOptions.outputPath, 'utf8'));
const signedIndex = residentCapsule.artifacts.find(row => row.artifactId === residentCapsule.program.pieceIndexArtifactId);
assert.equal(signedIndex.role, 'piece-index');
assert.equal(await fs.readFile(path.join(path.dirname(residentOptions.outputPath), signedIndex.path), 'utf8'), JSON.stringify(pieceIndex));
const residentRecord = residentCapsule.targetPlans[0].qualification.find(row => row.operation === 'residentPartition');
assert.equal(residentRecord.partitionIndex, 0);
assert.equal(residentRecord.comparedSteps, 1);
assert.equal(residentRecord.partitionPlanHash, residentReport.partitionPlanHash);
const reverseReport = { ...residentReport, surface: 'browser-webgpu', partitionIndex: 1 };
const reversePath = path.join(reportDir, 'resident-reverse.json');
await fs.writeFile(reversePath, JSON.stringify(reverseReport));
await forgeModelCapsule({ ...residentOptions, qualificationReportPaths: [residentReportPath, reversePath] });
const bothPlacements = JSON.parse(await fs.readFile(residentOptions.outputPath, 'utf8')).targetPlans[0]
  .qualification.filter(row => row.operation === 'residentPartition');
assert.deepEqual(bothPlacements.map(row => [row.surface, row.partitionIndex]), [['node-webgpu', 0], ['browser-webgpu', 1]]);
assert.equal(bothPlacements.length, 2);

for (const mutate of [
  row => { delete row.partitionPlanHash; }, row => { delete row.partitionPlan; },
  row => { row.partitionPlan.hiddenSize += 1; }, row => { row.executionGraphHash = logitsDigest; }, row => { row.partitionIndex = 2; },
  row => { row.comparison.steps = []; }, row => { row.comparison.steps[0].maxAbsoluteError = 0.00101; },
  row => { row.status = 'failed'; }, row => { row.comparison.tolerance = 0.01; },
  row => { row.comparison.referenceHash = logitsDigest; }, row => { row.comparison.complete = false; }, row => { row.manifestIdentity = promptTokenDigest; },
  row => { row.initialExecutionIdentity.programLoadPolicy.runtimeConfig.inference.session.linearRecurrentStatePrecision = 'two-f32'; },
]) {
  const invalid = structuredClone(residentReport); mutate(invalid);
  await fs.writeFile(residentReportPath, JSON.stringify(invalid));
  await assert.rejects(forgeModelCapsule(residentOptions), /resident|Resident|qualification|identity|comparison|Partition/);
}
await fs.writeFile(residentReportPath, JSON.stringify(residentReport));
await assert.rejects(forgeModelCapsule({ ...residentOptions, pieceIndexPath: null }), /piece/i);
const badPieces = structuredClone(pieceIndex); badPieces.files[0].pieces[0].identity = promptTokenDigest;
await fs.writeFile(indexPath, JSON.stringify(badPieces));
await assert.rejects(forgeModelCapsule(residentOptions), /integrity|piece/i);
console.log('forge-model-capsule: signed resident producer passed (synthetic evidence only)');

// Ordered report v2 passes through the real Rig signing path; these receipts are synthetic.
await fs.writeFile(indexPath, JSON.stringify(pieceIndex));
const orderedPlan = partitionContract.createOrderedLayerPartitionPlan({ modelId: physicalManifest.modelId,
  ...physicalManifest.architecture, splitLayers: [1], activationDtype: 'f32' });
const orderedProviders = [{ adapter: 'synthetic-browser' }, { adapter: 'synthetic-node' }];
const orderedContract = { ...residentContract, schema: 'doppler.resident-qualification-contract/v2',
  manifestIdentity: manifestHash, executionGraphHash: residentReport.executionGraphHash,
  partitionPlan: orderedPlan, partitionPlanHash: partitionContract.hashLayerPartitionPlan(orderedPlan),
  stages: orderedProviders.map((provider, index) => ({ index, surface: index ? 'node-webgpu' : 'browser-webgpu',
    providerIdentityHash: computeCanonicalSha256(provider) })) };
await fs.writeFile(residentContractPath, JSON.stringify(orderedContract));
const orderedPaths = [];
for (const stage of orderedContract.stages) {
  const report = { ...residentReport, schema: 'doppler.resident-partition-qualification/v2', partitionPlan: orderedPlan,
    partitionPlanHash: orderedContract.partitionPlanHash, partitionIndex: stage.index, surface: stage.surface,
    providerIdentity: orderedProviders[stage.index], comparison: { ...residentReport.comparison,
      contractHash: computeCanonicalSha256(orderedContract) } };
  const reportPath = path.join(reportDir, `ordered-${stage.index}.json`);
  await fs.writeFile(reportPath, JSON.stringify(report)); orderedPaths.push(reportPath);
}
await forgeModelCapsule({ ...residentOptions, qualificationReportPaths: orderedPaths });
const orderedCapsule = JSON.parse(await fs.readFile(residentOptions.outputPath, 'utf8'));
const orderedRecords = orderedCapsule.targetPlans[0].qualification.filter(row => row.operation === 'residentPartition');
assert.deepEqual(orderedRecords.map(row => row.partitionIndex), [0, 1]);
assert.deepEqual(orderedRecords[0].orderedContract, orderedContract);
assert.equal(orderedRecords[1].providerIdentityHash, orderedContract.stages[1].providerIdentityHash);
await assert.rejects(forgeModelCapsule({ ...residentOptions, qualificationReportPaths: orderedPaths.slice(0, 1) }), /missing ordered stage/);
await assert.rejects(forgeModelCapsule({ ...residentOptions, qualificationReportPaths: [...orderedPaths, orderedPaths[0]] }), /duplicate stage/);
console.log('forge-model-capsule: ordered report binding and complete-set signing passed (synthetic evidence only)');
