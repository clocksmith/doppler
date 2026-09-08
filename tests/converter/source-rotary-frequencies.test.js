import assert from 'node:assert/strict';
import { readSourceRotaryFrequencies } from '../../src/converter/source-rotary-frequencies.js';
import { createConverterConfig } from '../../src/config/schema/converter.schema.js';

const values = [1, 0.464111328125, 0.2154541015625];
const bytes = Float32Array.from(values).buffer;
const tensors = [0, 1].map((layer) => ({
  name: `layer.${layer}.inv_freq`, dtype: 'F32', shape: [values.length],
  offset: layer * bytes.byteLength, size: bytes.byteLength,
}));
const policy = { match: '^layer\\.\\d+\\.inv_freq$', expectedMatches: tensors.length };
assert.deepEqual(createConverterConfig({ sourceRotaryFrequencies: policy }).sourceRotaryFrequencies, policy);
assert.deepEqual(await readSourceRotaryFrequencies(tensors, policy, async () => bytes), values);
assert.equal(await readSourceRotaryFrequencies(tensors, null, () => assert.fail('Unexpected source read')), null);
await assert.rejects(() => readSourceRotaryFrequencies(tensors, { ...policy, expectedMatches: 3 }, async () => bytes), /matched 2 tensors/);
await assert.rejects(() => readSourceRotaryFrequencies(tensors, policy,
  async (tensor) => tensor.offset === 0 ? bytes : Float32Array.from([1, 0.5, 0.25]).buffer), /differ across layers/);
await assert.rejects(() => readSourceRotaryFrequencies(tensors, policy, async () => bytes.slice(4)), /byte length mismatch/);
await assert.rejects(() => readSourceRotaryFrequencies(tensors, policy,
  async () => Float32Array.from([1, NaN, 0.25]).buffer), /invalid values/);
await assert.rejects(() => readSourceRotaryFrequencies([{ ...tensors[0], dtype: 'F16' }],
  { ...policy, expectedMatches: 1 }, async () => bytes), /contiguous f32 vector/);
console.log('source-rotary-frequencies.test: ok');
