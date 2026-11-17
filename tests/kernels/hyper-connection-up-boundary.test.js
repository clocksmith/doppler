import assert from 'node:assert/strict';
import { recordHyperConnectionUpMixBranch } from '../../src/gpu/kernels/hyper-connection-up.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';

globalThis.GPUBufferUsage ??= { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 };
const buffer = size => ({ size, usage: GPUBufferUsage.STORAGE });
const tensor = (shape, dtype = 'f32') => ({ dtype, shape, buffer: buffer(shape.reduce((a, b) => a * b) * 4) });
const device = { limits: { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64, maxComputeWorkgroupStorageSize: 16384,
  maxStorageBufferBindingSize: 1 << 20, maxComputeWorkgroupsPerDimension: 65535 } };
const recorder = { device };
const input = tensor([1, 32]), normed = tensor([1, 16]);
const weights = createWeightBuffer(buffer(16 * 18), 'q4_0', 'row', [16, 32]);
const options = { numTokens: 1, streamCount: 4, streamWidth: 4, projectionRank: 32 };
const run = (w, o = options, x = input, n = normed) => recordHyperConnectionUpMixBranch(recorder, x, w, n, o);
await assert.rejects(run(createWeightBuffer(buffer(1024), 'bf16', 'row', [16, 32])), /no dtype fallback/);
await assert.rejects(run(createWeightBuffer(buffer(1024), 'q8_0', 'row', [16, 32])), /no dtype fallback/);
await assert.rejects(run(createWeightBuffer(buffer(1024), 'q4_0', 'column', [16, 32])), /row-packed/);
await assert.rejects(run(weights, { ...options, numTokens: 2 }), /one token/);
await assert.rejects(run(weights, { ...options, streamCount: 5 }), /four streams/);
await assert.rejects(run(weights, { ...options, projectionRank: 33 }), /whole Q4_0 blocks/);
await assert.rejects(run(weights, options, tensor([1, 32], 'f16')), /scalar F32/);
await assert.rejects(run(weights, options, input, tensor([1, 15])), /scalar F32/);
await assert.rejects(run({ ...weights, buffer: buffer(4) }), /undersized/);
await assert.rejects(run(weights, { ...options, outputBuffer: normed.buffer }), /must not alias/);
for (const [name, unsupported] of [['maxComputeWorkgroupSizeX', 64], ['maxComputeWorkgroupSizeY', 0], ['maxComputeWorkgroupSizeZ', 0]]) {
  const supported = device.limits[name];
  device.limits[name] = unsupported;
  await assert.rejects(run(weights), /supported gemv32 geometry/, `${name} must independently admit the declared workgroup`);
  device.limits[name] = supported;
}
const cancelled = new AbortController(); cancelled.abort(new Error('cancelled before publication'));
await assert.rejects(run(weights, { ...options, signal: cancelled.signal }), /cancelled before publication/);
console.log('HC Q4_0 up boundary: incompatible representations, aliases, geometry and pre-dispatch cancellation fail closed (CPU only)');
