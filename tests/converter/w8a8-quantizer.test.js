import assert from 'node:assert/strict';
import { quantizeW8A8Reference } from '../kernels/reference/w8a8.js';
import { validateManifest } from '../../src/formats/rdrr/validation.js';
import { W8A8_POLICY, validateW8A8Policy } from '../../src/config/schema/w8a8.schema.js';
import { quantizeToW8A8Rows, roundW8A8Code } from '../../src/converter/w8a8-quantizer.js';
import { validateW8A8StorageBytes } from '../../src/formats/rdrr/w8a8-storage.js';
import { transformTensorBytes } from '../../src/converter/tensor-transform.js';
import { buildQuantizationInfo } from '../../src/converter/quantization-info.js';

assert.throws(() => validateW8A8Policy(null), /explicitly/);
for (const key of Object.keys(W8A8_POLICY)) {
  const incomplete = { ...W8A8_POLICY }; delete incomplete[key];
  assert.throws(() => validateW8A8Policy(incomplete), /explicitly/);
  assert.throws(() => validateW8A8Policy({ ...W8A8_POLICY, [key]: 'unsupported' }), /must be/);
}
const scratch = new DataView(new ArrayBuffer(4));
function bits(value) { scratch.setFloat32(0, value, true); return scratch.getUint32(0, true); }
function f32(raw) { scratch.setUint32(0, raw, true); return scratch.getFloat32(0, true); }
function rational(value) {
  const raw = bits(Math.abs(value));
  return { mantissa: BigInt((raw & 0x7fffff) | 0x800000), exponent: ((raw >>> 23) & 255) - 150 };
}
function exactCode(value, scale) {
  if (value === 0) return 0;
  const a = rational(value); const b = rational(scale);
  let numerator = a.mantissa; let denominator = b.mantissa;
  const shift = a.exponent - b.exponent;
  if (shift >= 0) numerator <<= BigInt(shift); else denominator <<= BigInt(-shift);
  let q = numerator / denominator; const remainder = numerator % denominator;
  if (2n * remainder > denominator || (2n * remainder === denominator && (q & 1n))) q++;
  const bounded = Math.min(127, Number(q));
  return value < 0 ? -bounded : bounded;
}
// Every half-integer tie, neighboring F32 values, signs and exponent scales.
for (const exponent of [-110, -20, 0, 20, 100]) {
  const scale = 2 ** exponent;
  for (let integer = 0; integer < 127; integer++) {
    const midpoint = (integer + 0.5) * scale;
    for (const delta of [-1, 0, 1]) {
      const value = f32(bits(midpoint) + delta);
      assert.equal(roundW8A8Code(value, scale), exactCode(value, scale));
      assert.equal(roundW8A8Code(-value, scale), exactCode(-value, scale));
    }
  }
}
// Non-power-of-two F32 scale divisions are checked against exact rational rounding.
let random = 0x3ac712f5;
for (let i = 0; i < 5000; i++) {
  random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
  const scale = f32(0x20000000 + (random % 0x30000000));
  const value = Math.fround(scale * ((random % 25401) / 100 - 127));
  assert.equal(roundW8A8Code(value, scale), exactCode(value, scale));
}
for (const exponent of [-110, -20, 0, 20, 100]) {
  const values = Float32Array.from({ length: 257 }, (_, i) => Math.sin(i * 1.7) * 127 * 2 ** exponent);
  assert.deepEqual(quantizeToW8A8Rows(values, [1, 257], W8A8_POLICY).quantized,
    quantizeW8A8Reference(values, 1, 257).bytes);
}
const input = new Float32Array([0, -0, 0, 0, 0, -127, -2.5, -1.5, 1.5, 127]);
const encoded = quantizeToW8A8Rows(input, [2, 5], W8A8_POLICY);
assert.equal(encoded.storage.rowStrideBytes, 12);
assert.deepEqual(Array.from(encoded.quantized.slice(4, 12)), Array(8).fill(0));
assert.deepEqual(Array.from(new Int8Array(encoded.quantized.buffer, 16, 8)), [-127, -2, -2, 2, 127, 0, 0, 0]);
assert.deepEqual(validateW8A8StorageBytes(encoded.quantized, [2, 5], encoded.storage), encoded.storage);
for (const value of [NaN, Infinity, -Infinity, 2 ** -149, -(2 ** -149), 2 ** -126]) {
  assert.throws(() => quantizeToW8A8Rows(new Float32Array([value]), [1, 1], W8A8_POLICY), /subnormal|nonfinite/);
}
assert.throws(() => quantizeToW8A8Rows(new Float32Array(133145), [1, 133145], W8A8_POLICY), /133144/);
for (const corrupt of ['padding', 'code', 'scale', 'zeroScale', 'length', 'layout']) {
  const bytes = encoded.quantized.slice(); const storage = { ...encoded.storage };
  if (corrupt === 'padding') bytes[9] = 1;
  if (corrupt === 'code') bytes[16] = 128;
  if (corrupt === 'scale') new DataView(bytes.buffer).setFloat32(12, NaN, true);
  if (corrupt === 'zeroScale') new DataView(bytes.buffer).setFloat32(0, 2, true);
  if (corrupt === 'layout') delete storage.paddedK;
  assert.throws(() => validateW8A8StorageBytes(corrupt === 'length' ? bytes.slice(1) : bytes, [2, 5], storage), /W8A8/);
}
const config = { quantization: { weights: 'w8a8', embeddings: 'f16', lmHead: 'f16', computePrecision: 'f32', w8a8: W8A8_POLICY } };
const info = buildQuantizationInfo(config, 'f32', 'f32', 'f32');
assert.equal(info.weights, 'w8a8'); assert.deepEqual(info.w8a8, W8A8_POLICY);
assert.throws(() => buildQuantizationInfo({ quantization: { ...config.quantization, w8a8: null } }, 'f32', 'f32', 'f32'), /explicitly/);
const tensor = { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F32', shape: [2, 5], role: 'matmul' };
const transformed = transformTensorBytes(tensor, new Uint8Array(input.buffer), { targetQuant: 'w8a8', quantizationInfo: info });
assert.equal(transformed.outDtype, 'W8A8'); assert.deepEqual(transformed.tensorData, encoded.quantized);
const dense = transformTensorBytes({ ...tensor, name: 'model.embed_tokens.weight', role: 'embedding' }, new Uint8Array(input.buffer), { targetQuant: 'w8a8', quantizationInfo: info });
assert.equal(dense.outDtype, 'F16');
const rowA = quantizeToW8A8Rows(input.slice(0, 5), [1, 5], W8A8_POLICY);
const rowB = quantizeToW8A8Rows(input.slice(5), [1, 5], W8A8_POLICY);
assert.deepEqual(new Uint8Array([...rowA.quantized, ...rowB.quantized]), encoded.quantized);
console.log('W8A8 converter policy, exact rounding, storage and role controls passed.');

const tensorRecord = { dtype: 'W8A8', role: 'matmul', layout: 'row', shape: [2, 5],
  size: encoded.quantized.length, storage: encoded.storage };
for (const mutate of [
  m => { delete m.quantizationInfo.w8a8; },
  m => { m.quantizationInfo.compute = 'f16'; },
  m => { m.tensors.weight.dtype = 'F16'; },
  m => { m.tensors.weight.size++; },
  m => { m.tensors.weight.storage.blockShape[1]++; },
]) {
  const manifest = { quantizationInfo: structuredClone(info), tensors: { weight: structuredClone(tensorRecord) } };
  mutate(manifest);
  assert.ok(validateManifest(manifest).errors.some(message => /W8A8|w8a8/.test(message)));
}
assert.throws(() => buildQuantizationInfo({ quantization: { ...config.quantization, embeddings: 'q4k' } }, 'f32', 'f32', 'f32'), /dense embedding/);
