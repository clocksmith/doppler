import assert from 'node:assert/strict';

const {
  resolveMatmulConstants,
  resolveMatmulPhase,
  selectMatmulVariantAndFlags,
} = await import('../../src/gpu/kernels/matmul-selection.js');
const { setDevice } = await import('../../src/gpu/device.js');

function createFakeDevice({ hasF16 = true, hasSubgroups = true } = {}) {
  const features = new Set();
  if (hasF16) {
    features.add('shader-f16');
  }
  if (hasSubgroups) {
    features.add('subgroups');
  }
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

const kernelPath = {
  id: 'unit-q4k-f32a-path',
  name: 'unit-q4k-f32a-path',
  description: 'Unit test path for matmul weight dtype validation.',
  activationDtype: 'f32',
  kvDtype: 'f16',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4' },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main' },
    ],
  },
};

const f32WeightKernelPath = {
  id: 'unit-q4k-f32w-f32a-path',
  name: 'unit-q4k-f32w-f32a-path',
  description: 'Unit test path for F32 matmul weight validation.',
  activationDtype: 'f32',
  kvDtype: 'f16',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f32.wgsl', entry: 'main' },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f32.wgsl', entry: 'main' },
    ],
  },
};

const lmHeadKernelPath = {
  id: 'unit-lm-head-phase-path',
  name: 'unit-lm-head-phase-path',
  description: 'Unit test path for lm_head decode/prefill phase pinning.',
  activationDtype: 'f32',
  kvDtype: 'f16',
  decode: {
    steps: [
      { op: 'lm_head', kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_multicol' },
    ],
  },
  prefill: {
    steps: [],
  },
  postLayer: [
    { op: 'lm_head', kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_multicol' },
    { op: 'lm_head_prefill', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main' },
  ],
};

setDevice(createFakeDevice(), { platformConfig: null });

try {
  assert.throws(
    () => selectMatmulVariantAndFlags(
      'run',
      8,
      16,
      32,
      'f32',
      'f32',
      true,
      'f32',
      {
        role: 'q_proj',
        layerIdx: 0,
        kernelPath,
      }
    ),
    /requires f16 weights but B dtype is f32/
  );

  assert.throws(
    () => selectMatmulVariantAndFlags(
      'run',
      1,
      16,
      32,
      'f32',
      'f32',
      true,
      'f32',
      {
        role: 'q_proj',
        layerIdx: 0,
        kernelPath,
      }
    ),
    /requires f16 weights but B dtype is f32/
  );

  const selected = selectMatmulVariantAndFlags(
    'run',
    8,
    16,
    32,
    'f32',
    'f16',
    true,
    'f32',
    {
      role: 'q_proj',
      layerIdx: 0,
      kernelPath,
    }
  );
  assert.equal(selected.variant, 'f16w_f32a');

  // Q4K weights are valid here because this kernel path is a dequant-before-dispatch path.
  const selectedQ4KPrefill = selectMatmulVariantAndFlags(
    'run',
    8,
    16,
    32,
    'f32',
    'q4k',
    true,
    'f32',
    {
      role: 'q_proj',
      layerIdx: 0,
      kernelPath,
    }
  );
  assert.equal(selectedQ4KPrefill.variant, 'f16w_f32a');

  const selectedQ4KDecode = selectMatmulVariantAndFlags(
    'run',
    1,
    16,
    32,
    'f32',
    'q4k',
    true,
    'f32',
    {
      role: 'q_proj',
      layerIdx: 0,
      kernelPath,
    }
  );
  assert.equal(selectedQ4KDecode.variant, 'gemv_subgroup_vec4');

  assert.throws(
    () => selectMatmulVariantAndFlags(
      'run',
      8,
      16,
      32,
      'f32',
      'f16',
      true,
      'f32',
      {
        role: 'q_proj',
        layerIdx: 0,
        kernelPath: f32WeightKernelPath,
      }
    ),
    /requires f32 weights but B dtype is f16/
  );

  assert.throws(
    () => selectMatmulVariantAndFlags(
      'run',
      8,
      16,
      32,
      'f32',
      'q4k',
      true,
      'f32',
      {
        role: 'q_proj',
        layerIdx: 0,
        kernelPath: f32WeightKernelPath,
      }
    ),
    /requires f32 weights but B dtype is q4k/
  );

  const selectedF32Weights = selectMatmulVariantAndFlags(
    'run',
    8,
    16,
    32,
    'f32',
    'f32',
    true,
    'f32',
    {
      role: 'q_proj',
      layerIdx: 0,
      kernelPath: f32WeightKernelPath,
    }
  );
  assert.equal(selectedF32Weights.variant, 'f32');

  const selectedLmHeadDecode = selectMatmulVariantAndFlags(
    'run',
    1,
    64,
    32,
    'f32',
    'f16',
    true,
    'f32',
    {
      role: 'lm_head',
      kernelPath: lmHeadKernelPath,
    }
  );
  assert.equal(selectedLmHeadDecode.variant, 'gemv_subgroup_multicol');

  const selectedLmHeadPrefillPhaseOverride = selectMatmulVariantAndFlags(
    'run',
    1,
    64,
    32,
    'f32',
    'f16',
    true,
    'f32',
    {
      role: 'lm_head',
      phaseOverride: 'prefill',
      kernelPath: lmHeadKernelPath,
    }
  );
  assert.equal(selectedLmHeadPrefillPhaseOverride.variant, 'f16w_f32a');

  const resolvedPhase = resolveMatmulPhase(1, 'prefill');
  assert.equal(resolvedPhase, 'prefill');
  assert.equal(
    resolveMatmulConstants(
      {
        role: 'lm_head',
        phaseOverride: 'prefill',
        kernelPath: lmHeadKernelPath,
      },
      resolvedPhase
    ),
    null
  );
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('matmul-kernel-path-weight-dtype-contract.test: ok');
