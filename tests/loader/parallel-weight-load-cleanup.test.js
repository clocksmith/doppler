import assert from 'node:assert/strict';
import { loadLayer } from '../../src/loader/layer-loader.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';

async function checkFailureSettlement(failingSuffix, lateSuffix, overrides = {}) {
  let finishLate;
  const late = new Promise(resolve => { finishLate = resolve; });
  const failure = Object.assign(new Error('allocation rejected'), { code: 'RESOURCE_EXHAUSTED' });
  const owned = new Set();
  const buffer = { size: 8, destroyed: false, destroy() { this.destroyed = true; } };
  let cleaned = false;
  const loading = loadLayer({
    tensorLocations: new Map(), gpuBuffers: owned,
    needsNormWeightOffset: () => false, keepF32Weights: true,
    isMoE: false, isExpertLayer: () => false, ...overrides,
    async loadTensor(name) {
      if (name.endsWith(failingSuffix)) throw failure;
      if (name.endsWith(lateSuffix)) {
        await late;
        owned.add(buffer);
        return createWeightBuffer(buffer, 'f16', 'row', [2, 2], name);
      }
      return null;
    },
  }, 0).catch(error => {
    // This is the model-load owner's cleanup boundary. No load may outlive it.
    cleaned = true;
    for (const allocation of owned) allocation.destroy();
    return error;
  });
  await new Promise(resolve => setImmediate(resolve));
  const rejectedBeforeSiblingSettled = cleaned;
  finishLate();
  assert.equal(await loading, failure, 'Preserve the original allocation rejection');
  assert.equal(rejectedBeforeSiblingSettled, false, 'Cleanup cannot race an unfinished tensor load');
  assert.equal(buffer.destroyed, true, 'Late allocations must be present when their owner cleans up');
}

await checkFailureSettlement('linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight');
await checkFailureSettlement('mlp.gate_proj.weight', 'mlp.down_proj.weight');
await checkFailureSettlement('mlp.router.weight', 'mlp.router.bias', { isMoE: true, isExpertLayer: () => true });
console.log('parallel-weight-load-cleanup.test: ok');
