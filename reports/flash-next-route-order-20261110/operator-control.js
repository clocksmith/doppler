import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [rootArgument, mode, outputPath] = process.argv.slice(2);
assert(rootArgument && outputPath && ['route-slot', 'expert-id'].includes(mode),
  'Usage: operator-control.js RUNTIME_ROOT route-slot|expert-id NEW_RECEIPT.json');
const root = path.resolve(rootArgument);
const load = name => import(pathToFileURL(path.join(root, name)).href);
const { probeNodeGPU } = await load('tests/helpers/gpu-probe.js');
const { getDevice, getKernelCapabilities, destroyDevice } = await load('src/gpu/device.js');
const { createCommandRecorder } = await load('src/gpu/command-recorder.js');
const { createTensor } = await load('src/gpu/tensor.js');
const { acquireBuffer, releaseBuffer, readBuffer, uploadData } = await load('src/memory/buffer-pool.js');
const { runRouteScatter, recordRouteScatter } = await load('src/gpu/kernels/moe-route.js');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const source = await readFile(path.join(root, 'src/gpu/kernels/scatter_add_routes_f32.wgsl'));
const receipt = { schema: 'doppler.route-order-physical-control/v1', mode,
  scope: 'Fixed weighted expert outputs; arithmetic ordering and product materialization, not full-model correctness',
  shaderSha256: hash(source), rows: [], gpuErrors: [] };
await writeFile(outputPath, JSON.stringify(receipt), { flag: 'wx' });
assert((await probeNodeGPU({ installFileFetchShim: true })).ready, 'Physical GPU required');
const device = getDevice();
receipt.adapter = getKernelCapabilities().adapterInfo;
assert.doesNotMatch(JSON.stringify(receipt.adapter), /software|swiftshader|llvmpipe/i);
device.addEventListener('uncapturederror', event => receipt.gpuErrors.push(event.error.message));
try {
  for (const [numTokens, hiddenSize, topK, kind] of [
    [2, 257, 3, 'order-cancellation'], [3, 5, 10, 'mixed-routes'], [1, 1, 2, 'product-rounding'],
  ]) {
    const weights = Float32Array.from({ length: numTokens * topK }, (_, i) => (i % topK + 1) / (topK * (topK + 1) / 2));
    const indices = Uint32Array.from({ length: numTokens * topK }, (_, i) => (i % topK * (topK - 1) + 7) % topK);
    const values = Float32Array.from({ length: numTokens * topK * hiddenSize }, (_, i) => Math.sin(i * 0.17) * 3);
    if (kind === 'order-cancellation') {
      for (let token = 0; token < numTokens; token++) {
        indices.set([0, 2, 1], token * topK);
        weights.set([0.25, 0.25, 0.5], token * topK);
        for (let channel = 0; channel < hiddenSize; channel++) {
          values[(token * topK) * hiddenSize + channel] = 4 * 2 ** 24;
          values[(token * topK + 1) * hiddenSize + channel] = -4 * 2 ** 24;
          values[(token * topK + 2) * hiddenSize + channel] = 2;
        }
      }
    }
    if (kind === 'product-rounding') {
      weights.set([0.5, Math.fround(0.5 * (1 + 2 ** -23))]);
      values.set([-1, 1 - 2 ** -23]);
    }
    const reference = order => Float32Array.from({ length: numTokens * hiddenSize }, (_, i) => {
      const token = Math.floor(i / hiddenSize), channel = i % hiddenSize;
      const slots = Array.from({ length: topK }, (_, slot) => slot);
      if (order === 'expert-id') slots.sort((a, b) => indices[token * topK + a] - indices[token * topK + b] || a - b);
      let sum = 0;
      for (const slot of slots) {
        const route = token * topK + slot;
        sum = Math.fround(sum + Math.fround(weights[route] * values[route * hiddenSize + channel]));
      }
      return sum;
    });
    const expected = reference(mode), other = reference(mode === 'expert-id' ? 'route-slot' : 'expert-id');
    if (kind === 'order-cancellation') {
      assert.equal(reference('route-slot')[0], 1);
      assert.equal(reference('expert-id')[0], 0);
    }
    if (kind === 'product-rounding') assert.equal(expected[0], 0);
    const buffers = [];
    const upload = data => { const buffer = acquireBuffer(data.byteLength); buffers.push(buffer); uploadData(buffer, data); return buffer; };
    try {
      const input = createTensor(upload(values), 'f32', [numTokens * topK, hiddenSize]);
      const routing = upload(weights), indicesBuffer = upload(indices);
      for (const recorded of [false, true]) {
        const recorder = recorded ? createCommandRecorder('route-order-control') : null;
        let output;
        try {
          const options = { numTokens, hiddenSize, topK, indicesBuffer };
          output = recorder ? await recordRouteScatter(recorder, input, routing, options)
            : await runRouteScatter(input, routing, options);
          if (recorder) await recorder.submitAndWait();
          const actual = new Float32Array(await readBuffer(output.buffer, expected.byteLength));
          const mismatches = actual.reduce((count, value, i) => count + Number(value !== expected[i]), 0);
          const alternativeMismatches = actual.reduce((count, value, i) => count + Number(value !== other[i]), 0);
          receipt.rows.push({ kind, numTokens, hiddenSize, topK, recorded,
            outputs: actual.length, mismatches, alternativeOrderMismatches: alternativeMismatches,
            actualFirst: actual[0], expectedFirst: expected[0],
            inputsSha256: hash(Buffer.concat([Buffer.from(values.buffer), Buffer.from(weights.buffer), Buffer.from(indices.buffer)])),
            outputSha256: hash(Buffer.from(actual.buffer)) });
          assert.equal(mismatches, 0);
          if (kind === 'order-cancellation') assert.equal(alternativeMismatches, actual.length);
        } finally {
          if (output) releaseBuffer(output.buffer);
          recorder?.abort();
        }
      }
    } finally {
      await device.queue.onSubmittedWorkDone();
      for (const buffer of buffers) releaseBuffer(buffer);
    }
  }
  assert.deepEqual(receipt.gpuErrors, []);
  receipt.passed = receipt.rows.length === 6;
} finally {
  destroyDevice();
  await writeFile(outputPath, JSON.stringify(receipt, null, 2) + '\n');
}
assert(receipt.passed);
console.log(JSON.stringify({ mode, passed: receipt.passed, rows: receipt.rows.length, adapter: receipt.adapter }));
