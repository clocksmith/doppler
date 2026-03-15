import assert from 'node:assert/strict';

const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { initDevice } = await import('../../src/gpu/device.js');
const {
  acquireBuffer,
  uploadData,
  readBuffer,
  releaseBuffer,
} = await import('../../src/memory/buffer-pool.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { initConvLayerState } = await import('../../src/inference/pipelines/text/ops.js');
const {
  quantizeToQ4KM,
  dequantizeQ4KM,
} = await import('../../src/converter/quantizer.js');

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('lfm2-conv-kernel-init.test: skipped (no WebGPU runtime)');
  process.exit(0);
}

await initDevice();

const convKernelF32 = new Float32Array([
  0.5, -1.25, 2.0,
  -0.75, 1.5, 0.25,
]);
const convKernelShape = [2, 1, 3];
const { quantized, numBlocks } = quantizeToQ4KM(convKernelF32, convKernelShape);
const expected = dequantizeQ4KM(quantized, numBlocks, [convKernelF32.length]);

const quantizedBuffer = acquireBuffer(quantized.byteLength, undefined, 'lfm2_conv_kernel_q4k');
uploadData(quantizedBuffer, quantized);

const convState = {};
try {
  await initConvLayerState(
    convState,
    createWeightBuffer(quantizedBuffer, 'q4k', null, convKernelShape, 'lfm2_conv_kernel_q4k'),
    null,
    2,
    'L0.conv',
    0
  );

  assert.ok(convState.convWeightGPU, 'Conv init must upload dequantized F32 weights.');
  assert.ok(convState.convStateGPU, 'Conv init must allocate conv state.');
  assert.equal(convState.kernelSize, 3);
  assert.equal(convState.hiddenSize, 2);

  const actual = new Float32Array(
    await readBuffer(convState.convWeightGPU, expected.byteLength)
  );

  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < 1e-5,
      `dequantized weight mismatch at ${i}: expected=${expected[i]}, got=${actual[i]}`
    );
  }
} finally {
  if (convState.convWeightGPU) {
    releaseBuffer(convState.convWeightGPU);
  }
  if (convState.convStateGPU) {
    releaseBuffer(convState.convStateGPU);
  }
  releaseBuffer(quantizedBuffer);
}

console.log('lfm2-conv-kernel-init.test: ok');
