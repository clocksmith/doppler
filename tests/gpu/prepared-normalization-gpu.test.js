import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { recordRMSNorm } from '../../src/gpu/kernels/rmsnorm.js';
import { readBuffer } from '../../src/memory/buffer-pool.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  if (process.env.DOPPLER_REQUIRE_PREPARED_NORM_GPU === '1') throw Error(probe.reason);
  console.log(`prepared-normalization-gpu: skipped (${probe.reason})`);
  process.exit(0);
}
const device = getDevice();
const report = { provider: probe.providerReceipt, cases: [] };
const createBuffer = device.createBuffer.bind(device);
const createBindGroup = device.createBindGroup.bind(device);
let created = 0, bound = 0;
device.createBuffer = descriptor => { created++; return createBuffer(descriptor); };
device.createBindGroup = descriptor => { bound++; return createBindGroup(descriptor); };
try {
  for (const hiddenSize of [1024, 5120]) {
    const workspace = new PreparedDecodeWorkspace(device);
    const inputBuffer = device.createBuffer({ size: hiddenSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const weights = Float32Array.from({ length: hiddenSize }, (_, i) => 0.5 + (i % 7) / 8);
    const weightBuffer = device.createBuffer({ size: weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(weightBuffer, 0, weights);
    const input = createTensor(inputBuffer, 'f32', [1, hiddenSize]);
    const weight = createWeightBuffer(weightBuffer, 'f32', 'row', [hiddenSize]);
    const kernelPath = { activationDtype: 'f32',
      decode: { steps: [{ op: 'input_norm', kernel: 'rmsnorm.wgsl', entry: 'main' }] } };
    const options = { batchSize: 1, hiddenSize, phase: 'decode', layerIdx: 0,
      role: 'input_norm', section: 'layer', kernelPath, rmsNormWeightOffset: false };
    let previousOutput = null;
    try {
      for (let step = 0; step < 3; step++) {
        const values = Float32Array.from({ length: hiddenSize }, (_, i) => Math.sin(i * 0.17 + step) / 4);
        const meanSquare = values.reduce((sum, value) => sum + value * value, 0) / hiddenSize;
        const scale = 1 / Math.sqrt(meanSquare + 1e-6);
        device.queue.writeBuffer(inputBuffer, 0, values);
        const recorder = createCommandRecorder('prepared_normalization');
        recorder.decodeWorkspace = workspace;
        const before = { created, bound };
        let output;
        try {
          output = await recordRMSNorm(recorder, input, weight, 1e-6, options);
          await recorder.submitAndWait();
          const resources = { gpuBuffers: created - before.created, bindings: bound - before.bound };
          if (step > 0) {
            assert.equal(resources.gpuBuffers, 0, 'Warm prepared normalization must not create GPU buffers');
            assert.equal(resources.bindings, 0, 'Warm prepared normalization must reuse its binding');
            assert.equal(output.buffer, previousOutput, 'Decode steps reuse the same named output');
          }
          previousOutput = output.buffer;
          const observed = new Float32Array(await readBuffer(output.buffer, hiddenSize * 4));
          let maxAbs = 0;
          for (let i = 0; i < hiddenSize; i++) {
            const error = Math.abs(observed[i] - values[i] * scale * weights[i]);
            assert.ok(Number.isFinite(observed[i]) && error <= 1e-5, `normalization[${i}] error ${error}`);
            maxAbs = Math.max(maxAbs, error);
          }
          report.cases.push({ hiddenSize, step, maxAbs, ...resources });
        } finally { await recorder.abort(); }
      }
      const cancelled = createCommandRecorder('cancelled_prepared_normalization');
      cancelled.decodeWorkspace = workspace;
      await recordRMSNorm(cancelled, input, weight, 1e-6, options);
      await cancelled.abort();
      assert(workspace.ownsBuffer(previousOutput), 'Abort preserves the session output owner');
    } finally {
      workspace.release();
      inputBuffer.destroy();
      weightBuffer.destroy();
    }
  }
  console.log(`prepared-normalization-gpu: ok ${JSON.stringify(report)}`);
} finally {
  device.createBuffer = createBuffer;
  device.createBindGroup = createBindGroup;
  await destroyDevice();
  await releaseNodeWebGPU();
}
