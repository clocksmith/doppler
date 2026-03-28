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
const { getWeightDtype, tagBufferDtype } = await import('../../src/gpu/weight-buffer.js');
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

  const qProjQ4 = new GPUBuffer({ size: 80, usage: GPUBufferUsage.STORAGE, label: 'q_proj_q4_dense' });
  const kProjQ4 = new GPUBuffer({ size: 80, usage: GPUBufferUsage.STORAGE, label: 'k_proj_q4_dense' });
  const vProjQ4 = new GPUBuffer({ size: 80, usage: GPUBufferUsage.STORAGE, label: 'v_proj_q4_dense' });
  tagBufferDtype(qProjQ4, 'f16');
  tagBufferDtype(kProjQ4, 'f16');
  tagBufferDtype(vProjQ4, 'f16');

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
    'QKV fusion must skip when the active kernel path requires q4k weights for qkv_proj'
  );
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('weight-buffer-runtime-dtype.test: ok');
