import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  STORAGE: 0x0080,
  COPY_DST: 0x0008,
};

globalThis.GPUBuffer = class GPUBuffer {
  constructor({ size, usage, label = null }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
};

const { setDevice } = await import('../../src/gpu/device.js');
const { Q4K_BLOCK_BYTES, q4kBlockCount } = await import('../../src/config/schema/index.js');
const { createWeightBuffer, getWeightDtype, tagBufferDtype } = await import('../../src/gpu/weight-buffer.js');
const { fuseQKVWeights } = await import('../../src/inference/pipelines/text/init.js');

function createFakeDevice() {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
    },
    features: new Set(['shader-f16']),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    createBuffer({ size, usage, label }) {
      return new GPUBuffer({ size, usage, label });
    },
    createBindGroup() {
      return {};
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer() {},
        finish() {
          return {};
        },
      };
    },
  };
}

setDevice(createFakeDevice(), { platformConfig: null });

try {
  const tagged = new GPUBuffer({
    size: 80,
    usage: GPUBufferUsage.STORAGE,
    label: 'tagged_raw_weight',
  });
  tagBufferDtype(tagged, 'f16');
  assert.equal(getWeightDtype(tagged), 'f16');

  const qProj = new GPUBuffer({ size: 80, usage: GPUBufferUsage.STORAGE, label: 'q_proj_raw' });
  const kProj = new GPUBuffer({ size: 80, usage: GPUBufferUsage.STORAGE, label: 'k_proj_raw' });
  const vProj = new GPUBuffer({ size: 80, usage: GPUBufferUsage.STORAGE, label: 'v_proj_raw' });
  tagBufferDtype(qProj, 'f16');
  tagBufferDtype(kProj, 'f16');
  tagBufferDtype(vProj, 'f16');

  const layerWeights = new Map();
  layerWeights.set('layer_0', {
    qProj,
    kProj,
    vProj,
    qkvProj: null,
  });

  fuseQKVWeights(layerWeights, {
    numLayers: 1,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 4,
    hiddenSize: 4,
  });

  const fused = layerWeights.get('layer_0').qkvProj;
  assert.ok(fused, 'expected fused QKV weight');
  assert.equal(fused.dtype, 'f16');
  assert.deepEqual(Array.from(fused.shape), [12, 4]);
  assert.equal(fused.buffer.size, 12 * 4 * 2);

  const qProjBias = new GPUBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE, label: 'q_proj_bias' });
  const kProjBias = new GPUBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE, label: 'k_proj_bias' });
  const vProjBias = new GPUBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE, label: 'v_proj_bias' });
  tagBufferDtype(qProjBias, 'f32');
  tagBufferDtype(kProjBias, 'f32');
  tagBufferDtype(vProjBias, 'f32');
  const biasedLayerWeights = new Map();
  biasedLayerWeights.set('layer_0', {
    qProj,
    qProjBias,
    kProj,
    kProjBias,
    vProj,
    vProjBias,
    qkvProj: null,
  });
  fuseQKVWeights(biasedLayerWeights, {
    numLayers: 1,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 4,
    hiddenSize: 4,
  });
  const fusedBias = biasedLayerWeights.get('layer_0').qkvProjBias;
  assert.ok(fusedBias, 'expected fused QKV projection bias');
  assert.equal(getWeightDtype(fusedBias), 'f32');
  assert.equal(fusedBias.size, 12 * 4);

  const q4RowBytes = q4kBlockCount(4) * Q4K_BLOCK_BYTES;
  const qProjQ4 = createWeightBuffer(
    new GPUBuffer({ size: 4 * q4RowBytes, usage: GPUBufferUsage.STORAGE, label: 'q_proj_q4' }),
    'q4k',
    'row',
    [4, 4],
    'q_proj_q4'
  );
  const kProjQ4 = createWeightBuffer(
    new GPUBuffer({ size: 4 * q4RowBytes, usage: GPUBufferUsage.STORAGE, label: 'k_proj_q4' }),
    'q4k',
    'row',
    [4, 4],
    'k_proj_q4'
  );
  const vProjQ4 = createWeightBuffer(
    new GPUBuffer({ size: 4 * q4RowBytes, usage: GPUBufferUsage.STORAGE, label: 'v_proj_q4' }),
    'q4k',
    'row',
    [4, 4],
    'v_proj_q4'
  );

  const q4KernelPath = {
    id: 'test_q4_prefill',
    name: 'Test Q4 Prefill',
    activationDtype: 'f32',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'fused_matmul_q4_batched_multicol_shared.wgsl', entry: 'main' },
      ],
    },
  };

  const q4LayerWeights = new Map();
  q4LayerWeights.set('layer_0', {
    qProj: qProjQ4,
    kProj: kProjQ4,
    vProj: vProjQ4,
    qkvProj: null,
  });

  fuseQKVWeights(q4LayerWeights, {
    numLayers: 1,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 4,
    hiddenSize: 4,
  }, q4KernelPath);

  assert.equal(
    q4LayerWeights.get('layer_0').qkvProj,
    null,
    'Q4K QKV fusion must require explicit opt-in'
  );

  const q4AllowedLayerWeights = new Map();
  q4AllowedLayerWeights.set('layer_0', {
    qProj: qProjQ4,
    kProj: kProjQ4,
    vProj: vProjQ4,
    qkvProj: null,
  });

  fuseQKVWeights(q4AllowedLayerWeights, {
    numLayers: 1,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 4,
    hiddenSize: 4,
  }, q4KernelPath, { allowQ4K: true });

  const q4Fused = q4AllowedLayerWeights.get('layer_0').qkvProj;
  assert.ok(q4Fused, 'expected explicit Q4K fused QKV weight');
  assert.equal(q4Fused.dtype, 'q4k');
  assert.equal(q4Fused.layout, 'row');
  assert.deepEqual(Array.from(q4Fused.shape), [12, 4]);
  assert.equal(q4Fused.buffer.size, 12 * q4RowBytes);

  const gatedLayerWeights = new Map();
  const gatedQProj = new GPUBuffer({ size: 8 * 4 * 2, usage: GPUBufferUsage.STORAGE, label: 'q_gate_proj_raw' });
  tagBufferDtype(gatedQProj, 'f16');
  gatedLayerWeights.set('layer_0', {
    qProj: gatedQProj,
    kProj,
    vProj,
    qkvProj: null,
  });
  fuseQKVWeights(gatedLayerWeights, {
    numLayers: 1,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 4,
    hiddenSize: 4,
    attentionOutputGate: true,
  }, null, { allowQ4K: true });
  const gatedFused = gatedLayerWeights.get('layer_0').qkvProj;
  const gatedGate = gatedLayerWeights.get('layer_0').qGateProj;
  assert.ok(gatedFused, 'expected gated QKV fusion to preserve Q projection rows');
  assert.ok(gatedGate, 'expected gated QKV fusion to emit a separate gate projection');
  assert.equal(gatedFused.dtype, 'f16');
  assert.equal(gatedGate.dtype, 'f16');
  assert.deepEqual(Array.from(gatedFused.shape), [12, 4]);
  assert.deepEqual(Array.from(gatedGate.shape), [4, 4]);
  assert.deepEqual(Array.from(gatedLayerWeights.get('layer_0').qkvSizes), [4, 4, 4]);
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('weight-buffer-runtime-dtype.test: ok');
