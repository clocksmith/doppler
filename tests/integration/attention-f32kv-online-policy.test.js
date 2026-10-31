import assert from 'node:assert/strict';
import { removeSubgroups, widenToF32Activations } from '../../src/converter/execution-graph/validation.js';
import { widenToF32CorrectnessFallback } from '../../src/converter/execution-graph-transforms.js';
import { validateAttentionVariant } from '../../src/gpu/kernels/attention/plan.js';

for (const kernel of ['attention_decode_online_f16kv.wgsl', 'attention_decode_online_head256_f16kv.wgsl']) {
  const graph = { kernels: { attn: { kernel, entry: 'main', digest: 'sha256:original',
    precision: { activationDtype: 'f32', kvDtype: 'f16' } } }, decode: [['attention', 'attn']], prefill: [], postLayer: [] };
  const original = structuredClone(graph);
  for (const hasF16 of [false, true]) {
    const context = { activationDtype: 'f32', capabilities: { hasSubgroups: true, hasF16 } };
    const result = hasF16 ? widenToF32CorrectnessFallback(graph, context) : widenToF32Activations(graph, context);
    assert.equal(result.kernels.attn.kernel, 'attention_decode_online.wgsl');
    assert.deepEqual(result.kernels.attn.precision, { activationDtype: 'f32', kvDtype: 'f32' });
    assert.equal(result.kernels.attn.digest, null, 'derived execution invalidates original digest');
    assert.deepEqual(graph, original, 'transform preserves source graph');
    const noSubgroups = removeSubgroups(graph, { ...context, capabilities: { hasSubgroups: false, hasF16 } });
    const fallback = hasF16 ? widenToF32CorrectnessFallback(noSubgroups, context) : widenToF32Activations(noSubgroups, context);
    assert.equal(fallback.kernels[fallback.decode[0][1]].kernel, 'attention_streaming.wgsl');
    const widenedWithoutSubgroups = removeSubgroups(result, { ...context, capabilities: { hasSubgroups: false, hasF16 } });
    assert.equal(widenedWithoutSubgroups.kernels[widenedWithoutSubgroups.decode[0][1]].kernel, 'attention_streaming.wgsl');
  }
}
const caps = { hasF16: false, hasSubgroups: true, wgslLanguageFeatures: [] };
for (const headDim of [64, 128, 256, 512]) {
  assert.equal(validateAttentionVariant('decode_online', true, false, false, caps, headDim, 2287, 32768), 'decode_online');
}
assert.throws(() => validateAttentionVariant('decode_online', true, true, false, caps, 256, 28, 32768), /requires f32 Q\/K\/V/);
assert.throws(() => validateAttentionVariant('decode_online', true, false, false, { ...caps, hasSubgroups: false }, 256, 28, 32768), /unsupported GPU features/);
assert.throws(() => validateAttentionVariant('decode_online', true, false, false, caps, 513, 28, 32768), /headDim <= 512/);
console.log('attention-f32kv-online-policy: passed');
