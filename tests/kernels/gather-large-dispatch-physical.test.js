import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { runGather, recordGather } from '../../src/gpu/kernels/gather.js';
import { runSplitQG, recordSplitQG } from '../../src/gpu/kernels/split_qg.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { readBuffer, releaseBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert(probe.ready, `Physical gather needs WebGPU: ${probe.reason}`);
assert.doesNotMatch(JSON.stringify(getKernelCapabilities().adapterInfo), /swiftshader|llvmpipe|software/i);
const device = getDevice(), buffers = [], rows = [];
const tokens = 3784, hidden = 5120, vocabulary = 4;
assert(Math.ceil(tokens * hidden / 256) > device.limits.maxComputeWorkgroupsPerDimension);
function upload(data) {
  const buffer = device.createBuffer({ size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  buffers.push(buffer); device.queue.writeBuffer(buffer, 0, data); return buffer;
}
try {
  const ids = Uint32Array.from({ length: tokens + 1 }, (_, i) => i % (vocabulary + 1));
  const indices = upload(ids);
  const values = Float32Array.from({ length: vocabulary * hidden }, (_, i) => Math.floor(i / hidden) * 2 + i % 4 - 3);
  const weights = { f32: upload(values), f16: upload(f32ToF16Array(values)) };
  for (const input of ['f32', 'f16']) for (const output of ['f32', 'f16']) for (const useVec4 of [false, true]) {
    for (const recorded of [false, true]) {
      const options = { embeddingDtype: input, outputDtype: output, useVec4, indexOffset: 1 };
      let tensor, recorder;
      try {
        if (recorded) {
          recorder = createCommandRecorder('large-gather');
          tensor = await recordGather(recorder, indices, weights[input], tokens, hidden, vocabulary, options);
          await recorder.submitAndWait();
        } else tensor = await runGather(indices, weights[input], tokens, hidden, vocabulary, options);
        const bytes = await readBuffer(tensor.buffer);
        const actual = output === 'f16' ? new Uint16Array(bytes) : new Float32Array(bytes);
        for (let i = 0; i < tokens * hidden; i++) {
          const id = ids[Math.floor(i / hidden) + 1];
          const expected = id >= vocabulary ? 0 : id * 2 + i % 4 - 3;
          const got = output === 'f16' ? f16ToF32Bits(actual[i]) : actual[i];
          if (got !== expected) assert.fail(`gather ${input}/${output}/${useVec4}/${recorded} index ${i}: ${got} != ${expected}`);
        }
        rows.push({ input, output, useVec4, recorded, elements: tokens * hidden, passed: true });
      } finally { await recorder?.abort(); if (tensor) releaseBuffer(tensor.buffer); }
    }
  }
  for (const dtype of ['f32', 'f16']) {
    const headDim = 256, numHeads = 24, size = tokens * numHeads * headDim;
    const source = Float32Array.from({ length: size * 2 }, (_, i) => i % 31 - 15);
    const inputBuffer = upload(dtype === 'f16' ? f32ToF16Array(source) : source);
    const input = createTensor(inputBuffer, dtype, [tokens, numHeads * headDim * 2], 'qg-physical-input');
    for (const recorded of [false, true]) {
      let result, recorder;
      try {
        const options = { numTokens: tokens, numHeads, headDim };
        if (recorded) {
          recorder = createCommandRecorder('large-split-qg');
          result = await recordSplitQG(recorder, input, options);
          await recorder.submitAndWait();
        } else result = await runSplitQG(input, options);
        for (const [name, offset] of [['Q', 0], ['G', headDim]]) {
          const data = await readBuffer(result[name].buffer);
          const values = dtype === 'f16' ? new Uint16Array(data) : new Float32Array(data);
          for (let i = 0; i < size; i++) {
            const expected = source[Math.floor(i / headDim) * headDim * 2 + i % headDim + offset];
            const got = dtype === 'f16' ? f16ToF32Bits(values[i]) : values[i];
            if (got !== expected) assert.fail(`split QG ${dtype}/${recorded}/${name} index ${i}`);
          }
        }
        rows.push({ operation: 'split_qg', dtype, recorded, elements: size, passed: true });
      } finally {
        await recorder?.abort();
        if (result) { releaseBuffer(result.Q.buffer); releaseBuffer(result.G.buffer); }
      }
    }
    inputBuffer.destroy();
  }
  await device.queue.onSubmittedWorkDone();
  assert.equal(getBufferPool().getStats().resources.active.bytes, 0);
  console.log(JSON.stringify({ schema: 'doppler.gather-large-dispatch/v1', rows, passed: true }));
} finally {
  for (const buffer of buffers) buffer.destroy();
  destroyDevice(); await releaseNodeWebGPU();
}
