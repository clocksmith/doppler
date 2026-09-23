import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { selectRuleValue } from '../../src/gpu/kernels/rule-registry.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

const capsuleRoot = new URL('../../examples/document-search/capsules/reranker/', import.meta.url);
const capsule = JSON.parse(await readFile(new URL('capsule-v3.json', capsuleRoot), 'utf8'));
const digest = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
// Selector coverage is synthetic, not physical NVIDIA/Apple acceptance. The
// post-review removal of redundant platform rows must preserve these decisions.
for (const platformId of ['apple-m2', 'apple-m3', 'nvidia-rtx30', 'nvidia-rtx40', 'amd-rdna3']) {
  for (const hasSubgroups of [false, true]) for (const useVec4 of [false, true]) for (const wantsF16Out of [false, true]) {
    const expected = wantsF16Out ? (useVec4 ? 'subgroup_vec4_f16out' : 'subgroup_f16out')
      : hasSubgroups ? (useVec4 ? 'subgroup_vec4' : 'subgroup') : 'shared';
    const variant = selectRuleValue('dequant', 'variant', { platformId, hasSubgroups, useVec4, wantsF16Out });
    assert.equal(variant, expected, JSON.stringify({ platformId, hasSubgroups, useVec4, wantsF16Out }));
    // This accepted plan declares vectorized F16 dequantization, not every
    // possible selector result. Never expand its signed shader closure for a test.
    if (useVec4 && wantsF16Out) {
      const config = getKernelConfig('dequant', variant);
      assert.equal(config.entryPoint, 'main_vec4');
      assert.equal(config.shaderFile, 'dequant_f16_out_vec4.wgsl');
      const bytes = await readFile(new URL('../../src/gpu/kernels/dequant_f16_out_vec4.wgsl', import.meta.url));
      const artifact = capsule.artifacts.find(entry => entry.role === 'wgsl-source' && entry.hash === digest(bytes));
      assert.ok(artifact, 'selected shader must belong to the accepted Capsule');
      assert.deepEqual(await readFile(new URL(artifact.path, capsuleRoot)), bytes);
    }
  }
}
console.log('document-search-dequant-selection: 40 selector cases and declared shader identity passed');
