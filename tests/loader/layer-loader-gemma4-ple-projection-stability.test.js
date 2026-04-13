import assert from 'node:assert/strict';

const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { initDevice } = await import('../../src/gpu/device.js');
const { acquireBuffer, releaseBuffer, uploadData } = await import('../../src/memory/buffer-pool.js');
const { createWeightBuffer, isWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { loadLayer } = await import('../../src/loader/layer-loader.js');
const { quantizeToQ4KM } = await import('../../src/converter/quantizer.js');
const { f32ToF16Array } = await import('../../src/inference/kv-cache/types.js');

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('layer-loader-gemma4-ple-projection-stability.test: skipped (no WebGPU runtime)');
  process.exit(0);
}

await initDevice();

function makeF16Buffer(values, label) {
  const packed = f32ToF16Array(new Float32Array(values));
  const buffer = acquireBuffer(packed.byteLength, undefined, label);
  uploadData(buffer, packed);
  return buffer;
}

function makeQ4KBuffer(values, shape, label) {
  const { quantized } = quantizeToQ4KM(new Float32Array(values), shape);
  const buffer = acquireBuffer(quantized.byteLength, undefined, label);
  uploadData(buffer, quantized);
  return buffer;
}

function releaseTrackedBuffers(buffers) {
  const released = new Set();
  for (const buffer of buffers) {
    if (!buffer || released.has(buffer)) {
      continue;
    }
    released.add(buffer);
    releaseBuffer(buffer);
  }
}

const shape = [4, 2];
const values = [
  0.25, -0.5,
  1.0, 0.75,
  -1.25, 0.5,
  0.125, -0.875,
];
const gateValues = [
  -0.75, 0.5,
  1.25, -1.0,
  0.375, 0.625,
  -0.25, 0.875,
];
const denseF16Buffer = makeF16Buffer(values, 'ple_projection_f16');
const packedQ4KBuffer = makeQ4KBuffer(values, shape, 'ple_projection_q4k');
const denseGateF16Buffer = makeF16Buffer(gateValues, 'ple_gate_f16');
const packedGateQ4KBuffer = makeQ4KBuffer(gateValues, shape, 'ple_gate_q4k');
const perLayerProjection = createWeightBuffer(
  denseF16Buffer,
  'f16',
  'row',
  shape,
  'per_layer_projection',
  {
    q4k: {
      buffer: packedQ4KBuffer,
      layout: 'row',
    },
  }
);
const perLayerInputGate = createWeightBuffer(
  denseGateF16Buffer,
  'f16',
  'row',
  shape,
  'per_layer_input_gate',
  {
    q4k: {
      buffer: packedGateQ4KBuffer,
      layout: 'row',
    },
  }
);

const tensors = new Map([
  ['model.layers.0.per_layer_input_gate.weight', perLayerInputGate],
  ['model.layers.0.per_layer_projection.weight', perLayerProjection],
]);
const gpuBuffers = new Set();

try {
  const layerWeights = await loadLayer({
    tensorLocations: new Map(),
    loadTensor: async (name) => tensors.get(name) ?? null,
    needsNormWeightOffset: () => false,
    gpuBuffers,
    keepF32Weights: false,
    isMoE: false,
    isExpertLayer: () => false,
  }, 0);

  assert.ok(isWeightBuffer(layerWeights.perLayerProjection));
  assert.equal(
    layerWeights.perLayerProjection.dtype,
    'f32',
    'Gemma 4 per-layer projection should promote to a stable f32 base materialization'
  );
  assert.notEqual(
    layerWeights.perLayerProjection.buffer,
    denseF16Buffer,
    'Gemma 4 per-layer projection f32 base should be a new promoted GPU buffer'
  );
  assert.equal(
    layerWeights.perLayerProjection.materializations?.f16?.buffer,
    denseF16Buffer,
    'Gemma 4 per-layer projection should retain the original dense f16 materialization'
  );
  assert.equal(
    layerWeights.perLayerProjection.materializations?.q4k?.buffer,
    packedQ4KBuffer,
    'Gemma 4 per-layer projection should retain the original q4k materialization'
  );
  assert.ok(isWeightBuffer(layerWeights.perLayerInputGate));
  assert.equal(
    layerWeights.perLayerInputGate.dtype,
    'f32',
    'Gemma 4 per-layer input gate should promote to a stable f32 base materialization'
  );
  assert.notEqual(
    layerWeights.perLayerInputGate.buffer,
    denseGateF16Buffer,
    'Gemma 4 per-layer input gate f32 base should be a new promoted GPU buffer'
  );
  assert.equal(
    layerWeights.perLayerInputGate.materializations?.f16?.buffer,
    denseGateF16Buffer,
    'Gemma 4 per-layer input gate should retain the original dense f16 materialization'
  );
  assert.equal(
    layerWeights.perLayerInputGate.materializations?.q4k?.buffer,
    packedGateQ4KBuffer,
    'Gemma 4 per-layer input gate should retain the original q4k materialization'
  );
} finally {
  releaseTrackedBuffers([
    denseF16Buffer,
    packedQ4KBuffer,
    denseGateF16Buffer,
    packedGateQ4KBuffer,
    ...gpuBuffers,
  ]);
}

const referenceShape = [4, 256];
const referenceValues = new Float32Array(referenceShape[0] * referenceShape[1]).map((_, index) => (
  ((index % 17) - 8) / 8
));
const { quantized: referenceQuantized } = quantizeToQ4KM(referenceValues, referenceShape);
const referenceGpuBuffers = new Set();

try {
  const referenceLayerWeights = await loadLayer({
    tensorLocations: new Map([
      ['model.layers.0.per_layer_input_gate.weight', {
        shape: referenceShape,
        dtype: 'Q4_K_M',
        role: 'other',
        layout: 'row',
        size: referenceQuantized.byteLength,
      }],
      ['model.layers.0.per_layer_projection.weight', {
        shape: referenceShape,
        dtype: 'Q4_K_M',
        role: 'matmul',
        layout: 'row',
        size: referenceQuantized.byteLength,
      }],
    ]),
    async loadTensor(name, toGPU) {
      if (name !== 'model.layers.0.per_layer_projection.weight' && name !== 'model.layers.0.per_layer_input_gate.weight') {
        return null;
      }
      if (toGPU === false) {
        return referenceQuantized;
      }
      throw new Error('reference path should prefer CPU q4k bytes for per_layer_input_gate/per_layer_projection');
    },
    needsNormWeightOffset: () => false,
    gpuBuffers: referenceGpuBuffers,
    keepF32Weights: false,
    isMoE: false,
    isExpertLayer: () => false,
  }, 0);

  assert.ok(isWeightBuffer(referenceLayerWeights.perLayerProjection));
  assert.equal(
    referenceLayerWeights.perLayerProjection.dtype,
    'f32',
    'Gemma 4 q4k per-layer projection should use CPU reference dequant before runtime matmul'
  );
  assert.equal(
    referenceLayerWeights.perLayerProjection.buffer.size,
    referenceValues.byteLength,
    'Gemma 4 q4k per-layer projection reference path should upload a dense f32 buffer'
  );
  assert.ok(isWeightBuffer(referenceLayerWeights.perLayerInputGate));
  assert.equal(
    referenceLayerWeights.perLayerInputGate.dtype,
    'f32',
    'Gemma 4 q4k per-layer input gate should use CPU reference dequant before runtime matmul'
  );
  assert.equal(
    referenceLayerWeights.perLayerInputGate.buffer.size,
    referenceValues.byteLength,
    'Gemma 4 q4k per-layer input gate reference path should upload a dense f32 buffer'
  );
} finally {
  releaseTrackedBuffers(referenceGpuBuffers);
}

console.log('layer-loader-gemma4-ple-projection-stability.test: ok');
