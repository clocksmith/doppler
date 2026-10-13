// Source/dataflow control. Independent physical arithmetic remains required.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
const recurrent = await readFile(new URL('../../src/gpu/kernels/gated_delta_recurrent_paired.wgsl', import.meta.url), 'utf8');
const main = recurrent.slice(recurrent.indexOf('@compute'));
assert(!main.includes('pair_softplus('), 'Recurrent state update must consume prepared paired gates.');
assert(!main.includes('pair_exp('));
assert(main.includes('beta = gates[ab_index * 2u];'));
assert(main.includes('decay = gates[ab_index * 2u + 1u];'));
const prep = await readFile(new URL('../../src/gpu/kernels/qwen_linear_attention_prepare_paired.wgsl', import.meta.url), 'utf8');
const expected = [
  'let beta = pair_sigmoid(b[b_index]);',
  'let log_decay = -pair_mul(pair_exp(vec2<f32>(a_log[head], 0.0)), pair_softplus(pair_add(a[ab_index], vec2<f32>(dt_bias[head], 0.0))));',
  'let decay = pair_exp(log_decay);',
  'gates[ab_index * 2u] = beta;',
  'gates[ab_index * 2u + 1u] = decay;',
];
let previous = -1;
for (const expression of expected) {
  const at = prep.indexOf(expression); assert(at > previous, `Original ordered expression missing: ${expression}`); previous = at;
}
assert(prep.includes('array<vec2<f32>>'));
// Both F32 words, including signed zero and tiny residuals, survive the only
// added storage boundary exactly. No high+low scalar reconstruction is used.
const words = new Uint32Array([0, 0x80000000, 1, 0x80000001, 0x3f800000, 0x32800000, 0xbf800000, 0xb2800000]);
const stored = new Float32Array(words.buffer).slice();
assert.deepEqual(new Uint32Array(stored.buffer), words);
const config = getKernelConfig('qwen_linear_attention_prepare', 'paired_gates');
assert.equal(config.shaderFile, 'qwen_linear_attention_prepare_paired.wgsl');
assert.equal(config.variantMetadata.outputPrecision, 'two-f32');
assert.deepEqual(config.bindings.map(binding => binding.name), ['uniforms', 'a', 'b', 'dt_bias', 'a_log', 'gates']);
console.log('paired gate preparation: original expression order, complete component storage and registry ABI pass');
