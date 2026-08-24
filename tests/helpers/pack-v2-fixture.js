import { createModelIR, hashModelIR } from '../../src/config/model-ir.js';
import { createTargetPlan, createTargetPlanV2 } from '../../src/config/target-plan.js';
import { buildPackV2, signPackV2 } from '../../src/config/pack-v2.js';
import { sha256BytesHex } from '../../src/utils/sha256.js';

export const TEST_PACK_AUTHORITY = 'doppler-pack-test';
export const TEST_PACK_PUBLIC_KEY = Object.freeze({
  crv: 'Ed25519',
  x: 'FLU5-eSyW8ORkAf8HupzJn8juiJ2TrGSw2rgMNqGPfc',
  kty: 'OKP',
});
const TEST_PACK_PRIVATE_KEY = Object.freeze({
  crv: 'Ed25519',
  d: 'WQi2FHRfw0jZxl_IXiMp5TAuehMfssojWd2Oj3WaUKU',
  x: TEST_PACK_PUBLIC_KEY.x,
  kty: 'OKP',
});

function bytes(value) {
  return new TextEncoder().encode(value);
}

function digest(value) {
  return `sha256:${sha256BytesHex(value)}`;
}

export function createPackReleaseFixture(options = {}) {
  const targetIds = options.targetIds ?? ['webgpu-f32-portable'];
  const previousPackId = options.previousPackId ?? 'pack-test-model-pack-v2-previous';
  const previousSemanticRoot = options.previousSemanticRoot ?? `sha256:${'9'.repeat(64)}`;
  return {
    schema: 'doppler.pack-release/v1',
    source: {
      repository: 'https://example.invalid/pack-test-model',
      revision: 'fixture-revision',
      revisionDigest: `sha256:${'a'.repeat(64)}`,
      provenanceDigest: `sha256:${'b'.repeat(64)}`,
      license: {
        spdxId: 'Apache-2.0',
        name: 'Apache License 2.0',
        sourceUrl: 'https://www.apache.org/licenses/LICENSE-2.0.txt',
        textDigest: `sha256:${'c'.repeat(64)}`,
      },
    },
    application: {
      applicationId: 'pack-test-application',
      applicationRevision: 'fixture-application-revision',
      applicationRevisionDigest: `sha256:${'d'.repeat(64)}`,
      workload: { id: 'fixture-workload', digest: `sha256:${'e'.repeat(64)}` },
      oracle: { id: 'fixture-oracle', digest: `sha256:${'f'.repeat(64)}` },
    },
    exclusions: {
      rejectionTypes: [
        'acceptance-failed',
        'application-gate-failed',
        'artifact-invalid',
        'evidence-expired',
        'migration-required',
        'revoked',
        'unsupported-device',
      ],
      known: [{
        code: 'unsupported-device',
        scope: 'fixture-unsupported-device',
        reason: 'The reference Pack is qualified only for its declared target.',
        evidenceDigest: `sha256:${'1'.repeat(64)}`,
      }],
    },
    lifecycle: {
      releaseVersion: '1.0.0',
      supersedes: { packId: previousPackId, semanticRoot: previousSemanticRoot },
      migration: {
        id: 'fixture-v1-migration',
        policyDigest: `sha256:${'2'.repeat(64)}`,
        required: true,
      },
      failedUpgrade: {
        preservePrevious: true,
        previousPackId,
        previousSemanticRoot,
      },
    },
    revocation: {
      authorityId: 'doppler-pack-test-authority',
      policyDigest: `sha256:${'4'.repeat(64)}`,
      offlineExpirySeconds: 86400,
      failClosedAfterExpiry: true,
    },
    stateSnapshot: {
      schema: 'doppler.pack-state-snapshot/v1',
      format: 'canonical-json',
      identityDigest: `sha256:${'5'.repeat(64)}`,
      portableAcrossTargetIds: targetIds,
    },
  };
}

export async function createSignedPackFixture(options = {}) {
  const artifactBytes = new Map([
    ['manifest', bytes('{"modelId":"pack-test-model"}\n')],
    ['tokenizer', bytes('{"type":"test"}\n')],
    ['weights', Uint8Array.from([1, 2, 3, 4])],
    ['wgsl', bytes('@compute @workgroup_size(1) fn main() {}\n')],
    ['evidence', bytes('{"passed":true}\n')],
    ['program-bundle', bytes('{"schema":"doppler.program-bundle/v1"}\n')],
  ]);
  const artifacts = [
    ['manifest', 'manifest', 'manifest.json'],
    ['tokenizer', 'tokenizer', 'tokenizer.json'],
    ['weights', 'weight-shard', 'weights.bin'],
    ['wgsl', 'wgsl-source', 'main.wgsl'],
    ['evidence', 'reference-report', 'reference.json'],
    ['program-bundle', 'program-bundle', 'program-bundle.json'],
  ].map(([artifactId, role, path]) => ({
    artifactId,
    role,
    path,
    hash: digest(artifactBytes.get(artifactId)),
    sizeBytes: artifactBytes.get(artifactId).byteLength,
  }));
  const modelIR = createModelIR({
    modelId: 'pack-test-model',
    architecture: 'transformer',
    vocabSize: 8,
    hiddenSize: 4,
    numLayers: 1,
    sourceIdentity: { manifestArtifactId: 'manifest', manifestHash: artifacts[0].hash },
    tensorRoles: { weight: { role: 'matmul', shape: [4, 4], semanticDtype: 'f32' } },
    layers: [{ index: 0, type: 'global-attention' }],
    attentionGeometry: { numHeads: 1, numKvHeads: 1, headDim: 4 },
    normalization: { type: 'rmsnorm', eps: 1e-6 },
    rope: { dimension: 4, baseFreq: 10000 },
    ffn: { type: 'gelu', intermediateSize: 8 },
    outputTopology: { headType: 'causal-lm', tieWeights: false },
    phases: ['prefill', 'decode'],
  });
  const executionGraphHash = `sha256:${'3'.repeat(64)}`;
  const programBundleHash = digest(artifactBytes.get('program-bundle'));
  const wgslHash = artifacts.find((artifact) => artifact.artifactId === 'wgsl').hash;
  const phaseCommand = (phase) => [{
    kind: 'program-phase',
    phase,
    executionGraphHash,
    declaredStepIds: [`${phase}-step`],
  }];
  const createPlan = options.initialExecutionIdentity ? createTargetPlanV2 : createTargetPlan;
  const targetPlan = createPlan({
    targetId: 'webgpu-f32-portable',
    modelId: modelIR.modelId,
    modelIRHash: hashModelIR(modelIR),
    executionGraphHash,
    programBundleHash,
    capabilityPredicate: { requiresF16: false, requiresSubgroups: false, minBufferSize: 4 },
    dtypes: { activation: 'f32', kv: 'f32', weight: 'f32' },
    fusions: [],
    kernelClosure: [{ moduleId: 'main', digest: wgslHash, sourceHash: wgslHash }],
    memoryLayout: {
      kvCacheLayout: 'contiguous',
      bufferSlots: [{
        slotId: 'input_tokens', role: 'token-ids', scope: 'transient', owner: 'runtime',
        usageBits: 1,
        size: { op: 'affine', constantBytes: 0, terms: { seqLen: 4 }, alignment: 4, minimumBytes: 4 },
      }],
    },
    phases: { prefill: phaseCommand('prefill'), decode: phaseCommand('decode') },
    qualification: [{
      surface: 'test-webgpu', status: 'passed', evidenceArtifactId: 'evidence',
      evidenceHash: artifacts.find((artifact) => artifact.artifactId === 'evidence').hash,
      generatedTokens: 4,
    }],
    ...(options.initialExecutionIdentity
      ? { initialExecutionIdentity: options.initialExecutionIdentity }
      : {}),
  });
  const unsignedPack = buildPackV2({
    modelId: modelIR.modelId,
    createdAtUtc: '2026-08-22T00:00:00.000Z',
    modelIR,
    targetPlans: [targetPlan],
    wgslModules: [{
      id: 'main', file: 'main.wgsl', entry: 'main', digest: wgslHash,
      sourceHash: wgslHash, sourceArtifactId: 'wgsl',
    }],
    artifacts,
    program: {
      schema: 'doppler.pack-program/v1',
      programBundleHash,
      programBundleArtifactId: 'program-bundle',
      executionGraphHash,
      manifestArtifactId: 'manifest',
      tokenizerArtifactIds: ['tokenizer'],
      weightArtifactIds: ['weights'],
      execution: { steps: [{ id: 'prefill-step' }, { id: 'decode-step' }] },
      referenceTranscript: { tokens: { ids: [4, 5, 6, 7] } },
    },
    release: createPackReleaseFixture({ targetIds: [targetPlan.targetId] }),
  });
  const pack = await signPackV2(unsignedPack, {
    authority: TEST_PACK_AUTHORITY,
    privateKeyJwk: TEST_PACK_PRIVATE_KEY,
    publicKeyJwk: TEST_PACK_PUBLIC_KEY,
  });
  const artifactStore = {
    async hashArtifact(artifact) {
      const value = artifactBytes.get(artifact.artifactId);
      return { hash: digest(value), sizeBytes: value.byteLength };
    },
    async readArtifact(artifact) {
      return artifactBytes.get(artifact.artifactId);
    },
  };
  return { pack, modelIR, targetPlan, artifactBytes, artifactStore };
}
