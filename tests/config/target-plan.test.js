import assert from 'node:assert/strict';
import { createTargetPlan, hashTargetPlan, matchesDeviceCapability, validateTargetPlan } from '../../src/config/target-plan.js';

const digest = `sha256:${'a'.repeat(64)}`;
const plan = createTargetPlan({
  targetId: 'webgpu-f16-subgroups', modelId: 'test-model', modelIRHash: digest,
  executionGraphHash: digest, programBundleHash: digest,
  capabilityPredicate: { requiresF16: true, requiresSubgroups: true, minBufferSize: 64 },
  dtypes: { activation: 'f16', kv: 'f16', weight: 'q4k' }, fusions: [],
  kernelClosure: [{ moduleId: 'main', digest, sourceHash: digest }],
  memoryLayout: {
    kvCacheLayout: 'contiguous',
    bufferSlots: [{ slotId: 'input', role: 'input', scope: 'transient', owner: 'runtime', usageBits: 1, size: { op: 'constant', bytes: 256 } }],
  },
  phases: {
    prefill: [{ kind: 'program-phase', phase: 'prefill', executionGraphHash: digest, declaredStepIds: ['prefill'] }],
    decode: [{ kind: 'program-phase', phase: 'decode', executionGraphHash: digest, declaredStepIds: ['decode'] }],
  },
  qualification: [{ surface: 'test', status: 'passed', evidenceArtifactId: 'evidence', evidenceHash: digest, generatedTokens: 1 }],
});

assert.equal(validateTargetPlan(plan).ok, true);
assert.match(hashTargetPlan(plan), /^sha256:[0-9a-f]{64}$/);
assert.equal(matchesDeviceCapability(plan, { hasF16: true, hasSubgroups: true, maxBufferSize: 128 }), true);
assert.equal(matchesDeviceCapability(plan, { hasF16: false, hasSubgroups: true, maxBufferSize: 128 }), false);
assert.equal(matchesDeviceCapability(plan, { hasF16: true, hasSubgroups: false, maxBufferSize: 128 }), false);
assert.equal(matchesDeviceCapability(plan, { hasF16: true, hasSubgroups: true, maxBufferSize: 16 }), false);
assert.throws(() => createTargetPlan({ ...plan, qualification: [] }), /qualification/);

console.log('✔ target-plan.test.js passed');
