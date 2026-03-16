import assert from 'node:assert/strict';

const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { initDevice } = await import('../../src/gpu/device.js');
const { acquireBuffer, releaseBuffer, uploadData } = await import('../../src/memory/buffer-pool.js');
const { createWeightBuffer, isWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { loadLayer } = await import('../../src/loader/layer-loader.js');
const { quantizeToQ4KM } = await import('../../src/converter/quantizer.js');

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('layer-loader-conv-precision.test: skipped (no WebGPU runtime)');
  process.exit(0);
}

await initDevice();

function makeQ4KWeight(values, shape, label) {
  const { quantized } = quantizeToQ4KM(new Float32Array(values), shape);
  const buffer = acquireBuffer(quantized.byteLength, undefined, label);
  uploadData(buffer, quantized);
  return createWeightBuffer(buffer, 'q4k', 'row', shape, label);
}

async function loadConvLayer(keepF32Weights) {
  const convInProj = makeQ4KWeight([
    0.1, -0.2,
    0.3, 0.4,
    -0.5, 0.6,
    0.7, -0.8,
    0.9, 1.0,
    -1.1, 1.2,
  ], [6, 2], 'conv_in_proj_q4k');
  const convKernel = makeQ4KWeight([
    0.5, -1.25, 2.0,
    -0.75, 1.5, 0.25,
  ], [2, 1, 3], 'conv_kernel_q4k');
  const convOutProj = makeQ4KWeight([
    0.2, -0.1,
    0.3, 0.7,
  ], [2, 2], 'conv_out_proj_q4k');

  const tensors = new Map([
    ['model.layers.0.conv.in_proj.weight', convInProj],
    ['model.layers.0.conv.conv.weight', convKernel],
    ['model.layers.0.conv.out_proj.weight', convOutProj],
  ]);

  const gpuBuffers = new Set();
  let layerWeights;
  try {
    layerWeights = await loadLayer({
      tensorLocations: new Map(),
      loadTensor: async (name) => tensors.get(name) ?? null,
      needsNormWeightOffset: () => false,
      gpuBuffers,
      keepF32Weights,
      isMoE: false,
      isExpertLayer: () => false,
    }, 0);

    return {
      layerWeights,
      gpuBuffers,
    };
  } catch (error) {
    for (const buffer of gpuBuffers) {
      releaseBuffer(buffer);
    }
    for (const weight of tensors.values()) {
      releaseBuffer(weight.buffer);
    }
    throw error;
  }
}

const f16Run = await loadConvLayer(false);
try {
  assert.ok(isWeightBuffer(f16Run.layerWeights.convInProj));
  assert.ok(isWeightBuffer(f16Run.layerWeights.convKernel));
  assert.ok(isWeightBuffer(f16Run.layerWeights.convOutProj));
  assert.equal(f16Run.layerWeights.convInProj.dtype, 'f32');
  assert.equal(f16Run.layerWeights.convKernel.dtype, 'q4k');
  assert.equal(f16Run.layerWeights.convOutProj.dtype, 'f32');
} finally {
  for (const buffer of f16Run.gpuBuffers) {
    releaseBuffer(buffer);
  }
}

const f32Run = await loadConvLayer(true);
try {
  assert.ok(isWeightBuffer(f32Run.layerWeights.convInProj));
  assert.ok(isWeightBuffer(f32Run.layerWeights.convKernel));
  assert.ok(isWeightBuffer(f32Run.layerWeights.convOutProj));
  assert.equal(f32Run.layerWeights.convInProj.dtype, 'f32');
  assert.equal(f32Run.layerWeights.convKernel.dtype, 'q4k');
  assert.equal(f32Run.layerWeights.convOutProj.dtype, 'f32');
} finally {
  for (const buffer of f32Run.gpuBuffers) {
    releaseBuffer(buffer);
  }
}

console.log('layer-loader-conv-precision.test: ok');
