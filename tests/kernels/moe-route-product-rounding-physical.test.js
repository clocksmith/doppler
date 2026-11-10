import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { runRouteScatter, recordRouteScatter } from '../../src/gpu/kernels/moe-route.js';

// Source MoE materializes each weighted expert output as F32 before index_add.
// Cancellation separates that contract from an FMA of product plus accumulator.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical route-product test requires WebGPU: ${probe.reason}`);
const device = getDevice(), errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const source = await readFile(new URL('../../src/gpu/kernels/scatter_add_routes_f32.wgsl', import.meta.url));
const receipt = { schema: 'doppler.route-product-rounding-physical/v1',
  sourceSha256: createHash('sha256').update(source).digest('hex'), adapter: getKernelCapabilities().adapterInfo,
  scope: 'Route-order weighted product rounding, immediate and recorded; not expert-ID summation order or model qualification.',
  acceptance: 'Exact F32 products followed by left-associated F32 sums', rows: [], errors };
assert.doesNotMatch(JSON.stringify(receipt.adapter), /swiftshader|llvmpipe|software/i);
try {
  for (const [numTokens, hiddenSize, topK] of [[2, 257, 2], [3, 5, 10], [1, 1, 2]]) {
    const weights = Float32Array.from({ length: numTokens * topK }, (_, i) => Math.fround((1 + i % topK) / (topK * (topK + 1) / 2)));
    const values = Float32Array.from({ length: numTokens * topK * hiddenSize }, (_, i) => Math.sin(i * 0.17) * 3);
    weights[0] = 0.5; weights[1] = Math.fround(0.5 * (1 + 2 ** -23));
    for (let channel = 0; channel < hiddenSize; channel++) {
      values[channel] = -1;
      values[hiddenSize + channel] = 1 - 2 ** -23;
      for (let slot = 2; slot < topK; slot++) values[slot * hiddenSize + channel] = 0;
    }
    const expected = new Float32Array(numTokens * hiddenSize);
    for (let token = 0; token < numTokens; token++) for (let channel = 0; channel < hiddenSize; channel++) {
      let sum = 0;
      for (let slot = 0; slot < topK; slot++) {
        const route = token * topK + slot;
        sum = Math.fround(sum + Math.fround(weights[route] * values[route * hiddenSize + channel]));
      }
      expected[token * hiddenSize + channel] = sum;
    }
    assert.equal(expected[0], 0);
    const owned = [];
    const upload = data => { const buffer = acquireBuffer(data.byteLength); owned.push(buffer); uploadData(buffer, data); return buffer; };
    try {
      const input = createTensor(upload(values), 'f32', [numTokens * topK, hiddenSize]);
      const routing = upload(weights);
      for (const recorded of [false, true]) {
        const recorder = recorded ? createCommandRecorder('route-product-rounding') : null;
        let output;
        try {
          const options = { numTokens, hiddenSize, topK };
          output = recorder ? await recordRouteScatter(recorder, input, routing, options)
            : await runRouteScatter(input, routing, options);
          if (recorder) await recorder.submitAndWait();
          const actual = new Float32Array(await readBuffer(output.buffer, expected.byteLength));
          let mismatches = 0, maxAbsoluteError = 0;
          for (let i = 0; i < expected.length; i++) {
            if (actual[i] !== expected[i]) mismatches++;
            maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual[i] - expected[i]));
          }
          receipt.rows.push({ numTokens, hiddenSize, topK, recorded, values: expected.length,
            mismatches, maxAbsoluteError, cancellationActual: actual[0], cancellationExpected: expected[0] });
        } finally {
          if (output) releaseBuffer(output.buffer);
          recorder?.abort();
        }
      }
    } finally {
      await device.queue.onSubmittedWorkDone();
      for (const buffer of owned) releaseBuffer(buffer);
    }
  }
} finally {
  destroyDevice();
  receipt.passed = errors.length === 0 && receipt.rows.length === 6 && receipt.rows.every(row => row.mismatches === 0);
  if (process.argv[2]) await writeFile(process.argv[2], `${JSON.stringify(receipt, null, 2)}\n`);
}
console.log(JSON.stringify(receipt));
assert(receipt.passed, 'Route combination must round weighted outputs before accumulation.');
