import assert from 'node:assert/strict';

const { setDevice } = await import('../../src/gpu/device.js');
const { resolveQ4KConfig } = await import('../../src/inference/pipelines/text/init.js');
const { selectRuleValue } = await import('../../src/rules/rule-registry.js');

function createFakeDevice() {
  const features = new Set(['shader-f16', 'subgroups']);
  return {
    queue: {
      submit() {},
    },
    features,
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    createBuffer() {
      return {
        __dopplerFakeGPUBuffer: true,
        size: 0,
        usage: 0,
        destroy() {},
      };
    },
    createBindGroup() {
      return {};
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() {
          return {};
        },
      };
    },
    createShaderModule() {
      return {};
    },
    lost: new Promise(() => {}),
  };
}

const manifest = {
  modelId: 'unit-gemma3-q4k',
  quantization: 'Q4_K_M',
  quantizationInfo: {
    layout: 'row',
  },
};

const defaultPath = {
  id: 'unit-q4k-dequant',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4' },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main' },
    ],
  },
  postLayer: [],
  preLayer: [],
};
const f32WeightPath = {
  ...defaultPath,
  id: 'unit-q4k-dequant-f32w',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f32.wgsl', entry: 'main' },
    ],
  },
};
const mixedPath = {
  id: 'unit-q4k-mixed',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'fused_matmul_q4.wgsl', entry: 'main_multicol' },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main' },
    ],
  },
  postLayer: [],
  preLayer: [],
};
const fusedLmHeadPath = {
  id: 'unit-q4k-lm-head',
  decode: defaultPath.decode,
  prefill: defaultPath.prefill,
  postLayer: [
    { op: 'lm_head', kernel: 'fused_matmul_q4.wgsl', entry: 'main_gemv' },
  ],
  preLayer: [],
};
const linearMixedPath = {
  id: 'unit-q4k-linear-mixed',
  decode: {
    steps: [
      { op: 'linear_qkv_proj', kernel: 'fused_matmul_q4.wgsl', entry: 'main_multicol' },
    ],
  },
  prefill: {
    steps: [
      { op: 'linear_out_proj', kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main' },
    ],
  },
  postLayer: [],
  preLayer: [],
};

setDevice(createFakeDevice(), { platformConfig: null });

try {
  assert.deepEqual(resolveQ4KConfig(manifest, defaultPath, 'config', false), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: false,
    q4kMaterializationMode: 'dense',
    q4kFusedRoles: [],
  });

  assert.deepEqual(resolveQ4KConfig(manifest, defaultPath, 'config', true), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: true,
    q4kMaterializationMode: 'dense',
    q4kFusedRoles: [],
  });

  assert.deepEqual(resolveQ4KConfig(manifest, f32WeightPath, 'config', false), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: true,
    q4kMaterializationMode: 'dense',
    q4kFusedRoles: [],
  });

  assert.deepEqual(resolveQ4KConfig(manifest, fusedLmHeadPath, 'execution-v1', false), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: false,
    q4kMaterializationMode: 'dense',
    q4kFusedRoles: ['lm_head'],
  });
  assert.equal(
    selectRuleValue('loader', 'tensorLoader', 'gpuLoaderPath', {
      dtype: 'Q4_K_M',
      role: 'lm_head',
      useFusedQ4K: false,
      requiresFusedQ4KRole: true,
      q4kMaterializationMode: 'dense',
      q4kCpuReferenceEligible: false,
      q4kBasicBackendClass: false,
    }),
    'q4k_fused'
  );

  assert.deepEqual(resolveQ4KConfig(manifest, mixedPath, 'execution-v1', false), {
    useFusedQ4K: true,
    q4kLayout: 'row',
    keepF32Weights: false,
    q4kMaterializationMode: 'mixed',
    q4kFusedRoles: ['q_proj'],
  });

  assert.deepEqual(resolveQ4KConfig(manifest, linearMixedPath, 'execution-v1', false), {
    useFusedQ4K: true,
    q4kLayout: 'row',
    keepF32Weights: false,
    q4kMaterializationMode: 'mixed',
    q4kFusedRoles: ['linear_qkv_proj'],
  });
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('q4k-loader-kernel-path-contract.test: ok');
