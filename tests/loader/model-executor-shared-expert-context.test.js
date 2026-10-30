import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
setDevice({ features: new Set(), limits: {}, createBindGroup() {}, lost: new Promise(() => {}) }, { platformConfig: null });
import { _loadLayer } from '../../src/loader/model-executor.js';
const suffixes = ['gate_proj', 'up_proj', 'down_proj'];
const tensors = new Map(suffixes.map(role => [`model.language_model.layers.0.mlp.shared_expert.${role}.weight`, Float32Array.of(1, 2, 3, 4)]));
tensors.set('model.language_model.layers.0.mlp.shared_expert_gate.weight', Float32Array.of(1, 2));
const model = { manifest: { moeConfig: { expertFormat: 'qwen3_5' }, architecture: { hiddenSize: 2 } },
  tensorLocations: new Map(), gpuBuffers: new Set(), keepF32Weights: true, isMoE: true, layers: new Map(),
  _isExpertLayer: () => true, _needsNormWeightOffset: () => false,
  async _loadTensor(name) { return tensors.get(name) ?? null; } };
await _loadLayer.call(model, 0);
for (const key of ['sharedExpertGate', 'sharedExpertUp', 'sharedExpertDown', 'sharedExpertRouter']) {
  assert.ok(model.layers.get(0)[key], 'production model-executor context must carry shared expert: ' + key);
}
tensors.delete('model.language_model.layers.0.mlp.shared_expert.gate_proj.weight');
await assert.rejects(_loadLayer.call(model, 0), /missing shared expert/);
await assert.rejects(_loadLayer.call({ ...model, manifest: { architecture: { hiddenSize: 2 } } }, 0), /expertFormat/);
setDevice(null, { platformConfig: null });
console.log('model-executor-shared-expert-context.test: ok');
