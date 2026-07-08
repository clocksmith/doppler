import assert from 'node:assert/strict';

const {
  resolveMatmulConstants,
  resolveMatmulPhase,
  selectMatmulKernel,
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

const litertInt4KernelPath = {
  id: 'unit-litert-int4-fused-path',
  name: 'unit-litert-int4-fused-path',
  description: 'Unit test path for LiteRT INT4 fused matmul validation.',
  activationDtype: 'f32',
  kvDtype: 'f16',
  decode: {
    steps: [
      {
        op: 'q_proj',
        kernel: 'fused_matmul_litert_int4_f32a_f32out.wgsl',
        entry: 'main_multicol_f32a_f32out',
      },
    ],
  },
  prefill: {
    steps: [
      {
        op: 'q_proj',
        kernel: 'fused_matmul_litert_int4_f32a_f32out.wgsl',
        entry: 'main_multicol_f32a_f32out',
      },
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

  assert.equal(
    selectMatmulKernel({
      aDtype: 'f32',
      bDtype: 'f16',
      outputDtype: 'f32',
      isPrefill: true,
      prefillRows: 8,
      transposeB: true,
    }),
    'f16w_f32a',
    'small f16-weight/f32-activation prefill should keep the base f16w_f32a matmul'
  );

  assert.equal(
    selectMatmulKernel({
      aDtype: 'f32',
      bDtype: 'f16',
      outputDtype: 'f32',
      isPrefill: true,
      prefillRows: 64,
      transposeB: true,
    }),
    'f16w_f32a_tiled',
    'large f16-weight/f32-activation prefill should use the register-tiled matmul'
  );

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
  assert.equal(selectedQ4KPrefill.variant, 'q4_fused_batched');

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
  assert.equal(selectedQ4KDecode.variant, 'q4_fused_multicol');

  const selectedQ4KDecodeWideTile = selectMatmulVariantAndFlags(
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
      useWideTileQ4KDecode: true,
    }
  );
  assert.equal(selectedQ4KDecodeWideTile.variant, 'q4_fused_widetile');

  const selectedQ4KDecodeWideTileResidual = selectMatmulVariantAndFlags(
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
      residualTensor: {},
      useWideTileQ4KDecode: true,
      useWideTileResidualFusion: true,
    }
  );
  assert.equal(selectedQ4KDecodeWideTileResidual.variant, 'q4_fused_widetile_residual');

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

  const selectedF32WeightsQ4K = selectMatmulVariantAndFlags(
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
  );
  assert.equal(selectedF32WeightsQ4K.variant, 'q4_fused_batched');

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

  const selectedLiteRTInt4Override = selectMatmulVariantAndFlags(
    'run',
    8,
    16,
    32,
    'f32',
    'litert_int4',
    true,
    'f32',
    {
      role: 'q_proj',
      layerIdx: 0,
      kernelPath: litertInt4KernelPath,
    }
  );
  assert.equal(selectedLiteRTInt4Override.variant, 'litert_int4_multicol_f32a_f32out');
  assert.equal(selectedLiteRTInt4Override.useLiteRTInt4Fused, true);

  assert.throws(
    () => selectMatmulVariantAndFlags(
      'run',
      8,
      16,
      32,
      'f32',
      'litert_int4',
      false,
      'f32',
      {
        role: 'q_proj',
        layerIdx: 0,
        kernelPath: litertInt4KernelPath,
      }
    ),
    /requires transposeB=true/
  );

  const selectedLiteRTInt4Auto = selectMatmulVariantAndFlags(
    'run',
    8,
    16,
    32,
    'f32',
    'litert_int4',
    true,
    'f32',
    {
      role: 'q_proj',
      layerIdx: 0,
    }
  );
  assert.equal(selectedLiteRTInt4Auto.variant, 'litert_int4_multicol_f32a_f32out');
  assert.equal(selectedLiteRTInt4Auto.useLiteRTInt4Fused, true);

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

  const selectedLmHeadPrefillRole = selectMatmulVariantAndFlags(
    'run',
    1,
    64,
    32,
    'f32',
    'f16',
    true,
    'f32',
    {
      role: 'lm_head_prefill',
      phaseOverride: 'prefill',
      kernelPath: lmHeadKernelPath,
    }
  );
  assert.equal(selectedLmHeadPrefillRole.variant, 'f16w_f32a');

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
