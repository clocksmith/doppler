import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { getMatmulConfig, selectMatmulVariantAndFlags } from '../../src/gpu/kernels/matmul-selection.js';
import { calculateMatmulDispatch } from '../../src/gpu/kernels/matmul-dispatch.js';
import { GPU_LIMITS } from '../../src/gpu/kernels/constants.js';

const step = { op: 'q_proj', kernel: 'matmul_gemv_f32.wgsl', entry: 'main' };
const kernelPath = { activationDtype: 'f32', decode: { steps: [step] }, prefill: { steps: [step] } };
setDevice({
  features: new Set(),
  limits: { maxStorageBufferBindingSize: 1 << 24, maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256, maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupStorageSize: 16384 },
  queue: { submit() {} },
  createBindGroup() { return {}; },
  lost: new Promise(() => {}),
});

try {
  const select = (M, weight = 'f32', output = 'f32') => selectMatmulVariantAndFlags(
    'run', M, 17, 35, 'f32', weight, true, output,
    { role: 'q_proj', kernelPath, phaseOverride: 'decode' }
  );
  const selected = select(1);
  assert.equal(selected.variant, 'gemv_f32');
  assert.equal(selected.useGemv, true);
  assert.throws(() => select(2), /requires M=1/);
  assert.throws(() => select(1, 'f16'), /requires f32 weights/);
  assert.throws(() => select(1, 'f32', 'f16'), /outputs f32/);

  const config = getMatmulConfig(selected.variant, null);
  const columns = config.variantMetadata.colsPerWg;
  const N = columns * GPU_LIMITS.MAX_WORKGROUPS + 1;
  const { workgroups, uniformWorkgroupsX } = calculateMatmulDispatch(
    selected.variant, false, true, false, 1, N, config
  );
  assert.equal(workgroups[1], 2);
  assert.equal(uniformWorkgroupsX, workgroups[0]);
  assert.ok(workgroups[0] * workgroups[1] * columns >= N);
  assert.ok(workgroups[0] <= GPU_LIMITS.MAX_WORKGROUPS);
  console.log('matmul-f32-gemv-contract.test: ok (explicit admission, no f16/subgroup requirement, 2D grid)');
} finally {
  setDevice(null);
}
