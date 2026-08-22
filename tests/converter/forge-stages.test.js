import assert from 'node:assert/strict';
import { stageAnalyze, stageInspect, stageSpecialize } from '../../src/converter/forge-stages.js';

// Stage 1: Inspect
const intake = await stageInspect({
  modelDir: '/models/qwen3.8-27b',
  manifest: {
    modelId: 'qwen3.8-27b',
    modelType: 'transformer',
    hiddenSize: 5120,
    numLayers: 64,
    numHeads: 40,
    numKvHeads: 8,
    normType: 'rmsnorm',
  },
});
assert.equal(intake.ok, true);

// Stage 2 & 3: Analyze -> ModelIR
const analyzed = stageAnalyze(intake.data);
assert.equal(analyzed.ok, true);
assert.equal(analyzed.modelIR.modelId, 'qwen3.8-27b');
assert.equal(analyzed.modelIR.hiddenSize, 5120);
assert.equal(analyzed.modelIR.numLayers, 64);
assert.match(analyzed.modelIRHash, /^sha256:[0-9a-f]{64}$/);

// Stage 4 & 5: Specialize -> TargetPlans
const specialized = stageSpecialize(analyzed.modelIR, [
  { id: 'k_main', file: 'matmul.wgsl', entry: 'main', digest: `sha256:${'1'.repeat(64)}` },
]);
assert.equal(specialized.ok, true);
assert.equal(specialized.targetPlans.length, 3);
assert.equal(specialized.targetPlans[0].targetId, 'webgpu-f16-subgroups');
assert.equal(specialized.targetPlans[1].targetId, 'webgpu-f16');
assert.equal(specialized.targetPlans[2].targetId, 'webgpu-f32-safe');
assert.equal(specialized.targetPlanHashes.length, 3);

console.log('✔ forge-stages.test.js passed');
