import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { promoteExecutionCandidate, searchExecutionCandidates } from '../../src/converter/execution-candidate-forge.js';

const qwenReceipt = JSON.parse(await fs.readFile('reports/model-ir-v2/qwen3.8-27b.model-ir-receipt.json', 'utf8'));
const glimmerReceipt = JSON.parse(await fs.readFile('reports/model-ir-v2/glimmer-30b.model-ir-receipt.json', 'utf8'));
const digest = (character) => `sha256:${character.repeat(64)}`;
const kernel = (id, character) => ({
  file: `${id}.wgsl`, entry: 'main', digest: digest(character), sourceHash: digest(character),
});
const vocabulary = {
  schema: 'doppler.execution-candidate-forge/v1',
  kernels: {
    embed: kernel('embed', '1'), full_prefill: kernel('full_prefill', '2'), full_decode: kernel('full_decode', '3'),
    linear_prefill: kernel('linear_prefill', '4'), linear_decode: kernel('linear_decode', '5'),
    recurrent_update: kernel('recurrent_update', '6'), sample: kernel('sample', '7'),
  },
  entryPointKernels: { generate: ['embed', 'sample'] },
  blockLowerings: [
    {
      id: 'portable-full-attention', blockKinds: ['full-attention'],
      phases: {
        prefill: [{ id: 'attention', kernelId: 'full_prefill' }],
        decode: [{ id: 'attention', kernelId: 'full_decode' }],
      },
    },
    {
      id: 'portable-linear-recurrent', blockKinds: ['linear-recurrent-attention'],
      phases: {
        prefill: [{ id: 'linear', kernelId: 'linear_prefill' }, { id: 'state', kernelId: 'recurrent_update' }],
        decode: [{ id: 'linear', kernelId: 'linear_decode' }, { id: 'state', kernelId: 'recurrent_update' }],
      },
    },
  ],
};
const common = {
  supportedStateKinds: ['kv', 'recurrent', 'convolutional'],
  targetId: 'webgpu-f32-f16-portable',
  capabilityPredicate: { requiresF16: true, requiresSubgroups: false, minBufferSize: 1024 },
  dtypes: { activation: 'f32', kv: 'f16', weight: 'q4k' },
  fusions: [],
  memoryLayout: {
    kvCacheLayout: 'heterogeneous-contiguous',
    bufferSlots: [{
      slotId: 'input', role: 'token-ids', scope: 'transient', owner: 'runtime', usageBits: 1,
      size: { op: 'affine', constantBytes: 0, terms: { seqLen: 4 }, alignment: 4, minimumBytes: 4 },
    }],
  },
};
const proposals = [
  {
    ...common, id: 'ai-portable', score: 10,
    author: { kind: 'ai', actor: 'openai-codex', proposalId: 'candidate-1' },
    selections: {
      'full-attention': 'portable-full-attention',
      'linear-recurrent': 'portable-linear-recurrent',
    },
  },
  {
    ...common, id: 'ai-incomplete', score: 1,
    author: { kind: 'ai', actor: 'openai-codex', proposalId: 'candidate-2' },
    selections: { 'full-attention': 'portable-full-attention' },
  },
  {
    ...common, id: 'human-portable', score: 20,
    author: { kind: 'human', actor: 'test-author' },
    selections: {
      'full-attention': 'portable-full-attention',
      'linear-recurrent': 'portable-linear-recurrent',
    },
  },
];

const search = searchExecutionCandidates({
  modelIR: qwenReceipt.modelIR,
  entryPointId: 'text.generate',
  vocabulary,
  proposals,
});
assert.equal(search.generatedCandidates, 3);
assert.equal(search.acceptedProposalId, 'ai-portable');
assert.equal(search.rejectedCandidates.length, 2);
assert.equal(search.acceptedCandidate.programBundle.schema, 'doppler.generated-program-bundle/v2');
assert.equal(search.acceptedCandidate.executionGraph.schedule.length, 64);
assert.ok(search.acceptedCandidate.kernelClosure.some((entry) => entry.moduleId === 'recurrent_update'));

const candidate = search.acceptedCandidate;
const identity = createInitialExecutionIdentityV2({
  executionGraphHash: candidate.executionGraphHash,
  resolvedGraphHash: digest('8'),
  kernelClosure: [{ moduleId: 'embed', file: 'embed.wgsl', entry: 'main', digest: digest('1') }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f16', math: 'f32', accumulation: 'f32' },
  fusionSet: [], kvLayout: { layout: 'heterogeneous-contiguous' },
  memoryPolicy: { kvcache: { layout: 'heterogeneous-contiguous' } },
  executionPlanDigest: digest('9'), runtimeEngine: { schema: 'test' },
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v2',
    runtimeConfig: {
      inference: {
        session: {}, compute: {}, generation: { disableMultiTokenDecode: false },
      },
    },
  },
});
const plan = promoteExecutionCandidate(candidate, {
  qualification: [{
    surface: 'test-webgpu', status: 'passed', evidenceArtifactId: 'evidence',
    evidenceHash: digest('a'), generatedTokens: 1,
  }],
  initialExecutionIdentity: identity,
});
assert.equal(plan.schema, 'doppler.target-plan/v2');
assert.equal(plan.executionGraphHash, candidate.executionGraphHash);
const legacyPolicyIdentity = createInitialExecutionIdentityV2({
  ...identity,
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v1',
    runtimeConfig: { inference: { session: {}, compute: {} } },
  },
});
assert.throws(
  () => promoteExecutionCandidate(candidate, {
    qualification: plan.qualification,
    initialExecutionIdentity: legacyPolicyIdentity,
  }),
  /current signed program-load policy/,
  'Forge must not promote a non-reconstructive legacy load policy'
);

assert.throws(() => searchExecutionCandidates({
  modelIR: glimmerReceipt.modelIR,
  entryPointId: 'text.generate',
  vocabulary,
  proposals,
}), /not lowered/);

console.log('✔ execution-candidate-forge.test.js passed');
