import assert from 'node:assert/strict';

const { selectMatmulVariantAndFlags } = await import('../../src/gpu/kernels/matmul-selection.js');
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
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('matmul-kernel-path-weight-dtype-contract.test: ok');
