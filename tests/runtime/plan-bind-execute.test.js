import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { createModelIR } from '../../src/config/model-ir.js';
import { createTargetPlan } from '../../src/config/target-plan.js';
import { buildPackV2 } from '../../src/tooling/pack-v2.js';

const ir = createModelIR({
  modelId: 'plan-bind-exec-model',
  architecture: 'qwen3',
  hiddenSize: 1024,
  numLayers: 4,
  vocabSize: 1000,
});

const targetPlan = createTargetPlan({
  targetId: 'webgpu-f16',
  modelId: 'plan-bind-exec-model',
  capabilityPredicate: { requiresF16: true, requiresSubgroups: false, minBufferSize: 0 },
  dtypes: { activation: 'f16', kv: 'f16', weight: 'q4k' },
  kernelClosure: [
    { id: 'k_main', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` },
  ],
  memoryLayout: {
    kvCacheLayout: 'paged',
    estimatedPeakBytes: 1024 * 1024,
    bufferSlots: [
      { slotId: 'hidden_state', role: 'activation', scope: 'layer-recycled' },
      { slotId: 'kv_cache', role: 'kv', scope: 'session' },
    ],
  },
  phases: {
    prefill: [{ step: 'prefill_kernel' }],
    decode: [{ step: 'decode_kernel' }],
  },
});

const pack = buildPackV2({
  modelId: 'plan-bind-exec-model',
  modelIR: ir,
  targetPlans: [targetPlan],
  wgslModules: [
    { id: 'k_main', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` },
  ],
  artifacts: [
    { role: 'manifest', path: 'manifest.json', hash: `sha256:${'b'.repeat(64)}`, sizeBytes: 64 },
  ],
});

// Create runtime with mock device supporting F16
const runtime = createDopplerRuntime({
  device: { hasF16: true, hasSubgroups: false },
});

// Open pack -> Selects target without mutating
const session = await runtime.openPack(pack);
assert.equal(session.selectedTargetId, 'webgpu-f16');
assert.equal(session.modelId, 'plan-bind-exec-model');

// Run forward generation through sessionController
const tokens = [];
for await (const token of session.generate({ promptTokens: [1, 2, 3], maxTokens: 4 })) {
  tokens.push(token);
}

assert.equal(tokens.length, 4);
assert.deepEqual(tokens, [4, 5, 6, 7]);

console.log('✔ plan-bind-execute.test.js passed');
