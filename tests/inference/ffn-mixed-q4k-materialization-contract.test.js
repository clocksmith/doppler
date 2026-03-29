import assert from 'node:assert/strict';

const { setDevice } = await import('../../src/gpu/device.js');
const { resolveFusedGateUpWeights } = await import('../../src/inference/pipelines/text/ffn/dense.js');

function createMixedWeight(label) {
  const denseBuffer = { label: `${label}_dense` };
  const q4kBuffer = { label: `${label}_q4k` };
  return {
    buffer: denseBuffer,
    dtype: 'f16',
    layout: 'row',
    shape: [6912, 1152],
    label,
    materializations: {
      f16: { buffer: denseBuffer, layout: 'row' },
      q4k: { buffer: q4kBuffer, layout: 'row' },
    },
  };
}

const fusedKernelPath = {
  id: 'unit-q4k-inline',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'fused_matmul_q4.wgsl', entry: 'main' },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'fused_matmul_q4_batched_multicol_shared.wgsl', entry: 'main' },
    ],
  },
};

const denseKernelPath = {
  id: 'unit-dense-inline',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main' },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main' },
    ],
  },
};

const mixedWeights = {
  gate: createMixedWeight('gate'),
  up: createMixedWeight('up'),
};

setDevice({
  lost: new Promise(() => {}),
  queue: {
    submit() {},
  },
  features: new Set(['shader-f16', 'subgroups']),
  limits: {
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 16384,
  },
  createBuffer() {
    return {
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
}, { platformConfig: null });

try {
  const widenedFused = resolveFusedGateUpWeights(mixedWeights, {
    activationDtype: 'f32',
    hiddenSize: 1152,
    kernelPath: fusedKernelPath,
  });
  assert.equal(widenedFused.hasQ4KMaterialization, true);
  assert.equal(widenedFused.gateDtype, 'q4k');
  assert.equal(widenedFused.upDtype, 'q4k');
  assert.equal(widenedFused.gate.buffer.label, 'gate_q4k');
  assert.equal(widenedFused.up.buffer.label, 'up_q4k');

const nativeF16 = resolveFusedGateUpWeights(mixedWeights, {
  activationDtype: 'f16',
  hiddenSize: 1152,
  kernelPath: fusedKernelPath,
});
assert.equal(nativeF16.gateDtype, 'q4k');
assert.equal(nativeF16.upDtype, 'q4k');
assert.equal(nativeF16.gate.buffer.label, 'gate_q4k');
assert.equal(nativeF16.up.buffer.label, 'up_q4k');

  const misalignedHidden = resolveFusedGateUpWeights(mixedWeights, {
    activationDtype: 'f32',
    hiddenSize: 1150,
    kernelPath: fusedKernelPath,
  });
  assert.equal(misalignedHidden.gateDtype, 'f16');
  assert.equal(misalignedHidden.upDtype, 'f16');

  const densePath = resolveFusedGateUpWeights(mixedWeights, {
    activationDtype: 'f32',
    hiddenSize: 1152,
    kernelPath: denseKernelPath,
  });
  assert.equal(densePath.gateDtype, 'f16');
  assert.equal(densePath.upDtype, 'f16');
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('ffn-mixed-q4k-materialization-contract.test: ok');
