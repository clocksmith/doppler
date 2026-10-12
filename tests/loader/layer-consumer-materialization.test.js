import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { loadLayer } from '../../src/loader/layer-loader.js';

setDevice({ features: new Set(), limits: {}, createBindGroup() {}, createBuffer() {}, lost: new Promise(() => {}) });
try {
  const calls = new Map();
  await loadLayer({
    tensorLocations: new Map(),
    loadTensor: async (name, _gpu, _silent, consumerRole) => { calls.set(name, consumerRole); return null; },
    needsNormWeightOffset: () => false, gpuBuffers: new Set(), keepF32Weights: true,
    isMoE: false, isExpertLayer: () => false,
  }, 0);
  for (const [suffix, role] of [
    ['self_attn.q_proj', 'q_proj'], ['self_attn.k_proj', 'k_proj'], ['self_attn.v_proj', 'v_proj'],
    ['self_attn.o_proj', 'o_proj'], ['linear_attn.out_proj', 'o_proj'],
    ['linear_attn.in_proj_qkv', 'linear_qkv_proj'], ['linear_attn.in_proj_z', 'linear_z_proj'],
    ['linear_attn.in_proj_a', 'linear_a_proj'], ['linear_attn.in_proj_b', 'linear_b_proj'],
    ['mlp.gate_proj', 'ffn_gate'], ['mlp.up_proj', 'ffn_up'], ['mlp.down_proj', 'ffn_down'],
  ]) assert.equal(calls.get(`model.layers.0.${suffix}.weight`), role, suffix);
} finally { setDevice(null); }
console.log('layer consumer materialization: named slots carry semantic roles without rewriting artifact roles');
