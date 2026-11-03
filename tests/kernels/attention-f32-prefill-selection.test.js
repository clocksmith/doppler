import assert from 'node:assert/strict';
import { validateAttentionVariant, calculateAttentionWorkgroups } from '../../src/gpu/kernels/attention/plan.js';
import { getKernelPathAttentionVariant } from '../../src/config/kernel-path-loader.js';
import { resolvePipelineIdentity } from '../../src/gpu/kernels/pipeline-cache.js';

const variant = 'prefill_query_tile_f32';
// Caller constants cannot change either registered tile's admitted geometry.
for (const [selectedVariant, tile] of [[variant, 8], ['prefill_query_tile16_f32', 16]]) {
  for (const constants of [null, { HEAD_DIM: 128, WORKGROUP_SIZE: 128, QUERY_TILE: 3 }]) {
    const identity = resolvePipelineIdentity({ operation: 'attention', variant: selectedVariant, constants });
    assert.equal(identity.constants.HEAD_DIM, 256);
    assert.equal(identity.constants.QUERY_TILE, tile);
    assert.equal(identity.constants.WORKGROUP_SIZE, tile * 32);
  }
}
const path = { prefill: { steps: [{ op: 'attention', kernel: 'attention_prefill_query_tile_f32.wgsl', entry: 'main' }] } };
assert.equal(getKernelPathAttentionVariant('prefill', 3, path), variant);
for (const width of [32, 64]) {
  const caps = { hasSubgroups: true, wgslLanguageFeatures: ['subgroup_id'], adapterInfo: { subgroupMinSize: width, subgroupMaxSize: width } };
  const validate = (overrides = {}) => {
    const p = { isDecode: false, f16KV: false, f16Q: false, caps, headDim: 256, kvLen: 2287, shared: 16384, ...overrides };
    return validateAttentionVariant(variant, p.isDecode, p.f16KV, p.f16Q, p.caps, p.headDim, p.kvLen, p.shared);
  };
  assert.equal(validate(), variant);
  assert.throws(() => validate({ headDim: 128 }), /headDim == 256/);
  assert.throws(() => validate({ shared: 8192 }), /shared >= 16384/);
  assert.throws(() => validate({ isDecode: true }), /prefill-only/);
  assert.throws(() => validate({ f16KV: true }), /requires f32/);
  assert.throws(() => validate({ f16Q: true }), /requires f32/);
  assert.throws(() => validate({ caps: { hasSubgroups: false } }), /unsupported GPU features/);
  assert.throws(() => validate({ caps: { ...caps, wgslLanguageFeatures: [] } }), /unsupported GPU features/);
  assert.throws(() => validate({ caps: { ...caps, adapterInfo: { subgroupMinSize: 16, subgroupMaxSize: 64 } } }), /partitions of 32/);
  assert.throws(() => validate({ caps: { hasSubgroups: true, wgslLanguageFeatures: ['subgroup_id'] } }), /unsupported or unavailable/);
}
assert.equal(calculateAttentionWorkgroups('tiled_small', 256, 16, variant), 512);
assert.equal(calculateAttentionWorkgroups('tiled_small', 239, 16, variant), 480);
assert.equal(calculateAttentionWorkgroups('tiled_small', 256, 16, 'prefill_query_tile16_f32'), 256);
assert.equal(calculateAttentionWorkgroups('tiled_small', 239, 16, 'prefill_query_tile16_f32'), 240);
assert.equal(calculateAttentionWorkgroups('tiled_small', 256, 16, 'prefill_small'), 128);
console.log('attention-f32-prefill-selection: passed (CPU contract only)');
