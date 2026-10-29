import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { selectMatmulVariantAndFlags } from '../../src/gpu/kernels/matmul-selection.js';

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
  for (const [kernel, variant, M] of [
    ['matmul_compensated_f32.wgsl', 'f32_compensated', 3],
    ['matmul_gemv_compensated_f32.wgsl', 'gemv_f32_compensated', 1],
  ]) {
    const step = { op: 'q_proj', kernel, entry: 'main' };
    const kernelPath = { activationDtype: 'f32', decode: { steps: [step] } };
    const select = (rows, weightDtype = 'f32', outputDtype = 'f32') => selectMatmulVariantAndFlags(
      'run', rows, 35, 3584, 'f32', weightDtype, true, outputDtype,
      { role: 'q_proj', kernelPath, phaseOverride: 'decode' }
    );
    assert.equal(select(M).variant, variant);
    assert.throws(() => select(M, 'f16'), /requires f32 weights/);
    assert.throws(() => select(M, 'f32', 'f16'), /outputs f32/);
    if (M === 1) assert.throws(() => select(2), /requires M=1/);
  }
  console.log('matmul-compensated-f32-contract.test: ok (explicit single-F32 admission, no f16/subgroup requirement)');
} finally {
  setDevice(null);
}
