import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setDevice } from '../../src/gpu/device.js';
import { createKernelRegistry, enterKernelRegistry, getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
import { calculateMatmulDispatch } from '../../src/gpu/kernels/matmul-dispatch.js';
import { resolvePipelineIdentity } from '../../src/gpu/kernels/pipeline-cache.js';

const source = JSON.parse(await readFile(new URL('../../src/config/kernels/registry.json', import.meta.url), 'utf8'));
const variants = source.operations.matmul.variants;
const limits = { maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64, maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupStorageSize: 16384 };
const device = overrides => setDevice({ features: new Set(), limits: { ...limits, ...overrides },
  queue: { submit() {} }, createBindGroup() { return {}; }, lost: new Promise(() => {}) });

function withVariant(variant, change, check) {
  const schema = structuredClone(variants[variant]);
  change(schema);
  const restore = enterKernelRegistry(createKernelRegistry({ extensions: { matmul: { variants: { [variant]: schema } } } }));
  try { check(); } finally { restore(); }
}

device();
try {
  for (const [variant, constants, expected, M, grid] of [
    ['f32_compensated', { TILE_SIZE: 8 }, [8, 8, 1], 17, [3, 5, 1]],
    ['gemv_f32_compensated', { THREADS_PER_COL: 7, COLS_PER_WG: 3 }, [7, 3, 1], 1, [12, 1, 1]],
    ['gemv_f32', { THREADS_PER_COL: 7, COLS_PER_WG: 3 }, [7, 3, 1], 1, [12, 1, 1]],
    ['gemv_f32_subgroup', { WORKGROUP_SIZE: 96, COLS_PER_WG: 7 }, [96, 1, 1], 1, [5, 1, 1]],
  ]) {
    // Change only JSON override geometry; deliberately keep the old hand fields.
    withVariant(variant, schema => Object.assign(schema.wgslOverrides, constants), () => {
      const raw = getKernelConfig('matmul', variant);
      const conflict = Object.fromEntries(Object.keys(constants).map(name => [name, 1]));
      const config = getMatmulConfig(variant, conflict);
      assert.notDeepEqual(config.workgroupSize, raw.workgroupSize);
      assert.deepEqual(config.workgroupSize, expected);
      assert.deepEqual(calculateMatmulDispatch(variant, false, false, false, M, 35, config).workgroups, grid);
      const identity = resolvePipelineIdentity({ operation: 'matmul', variant, constants: conflict });
      for (const [name, value] of Object.entries(constants)) assert.equal(identity.constants[name], value,
        'registry override must retain precedence over conflicting caller constants');
      assert.ok(Object.isFrozen(config.workgroupSize));
      const cached = getMatmulConfig(variant, null);
      device({ maxComputeWorkgroupSizeX: 4 });
      assert.throws(() => getMatmulConfig(variant, null), /maxComputeWorkgroupSizeX/);
      device();
      assert.equal(getMatmulConfig(variant, null), cached);
    });
  }

  const rejects = (change, message) => withVariant('f32_compensated', change,
    () => assert.throws(() => getMatmulConfig('f32_compensated', null), message));
  for (const value of [0, -1, 1.5]) rejects(schema => { schema.wgslOverrides.TILE_SIZE = value; }, /positive u32/);
  rejects(schema => { delete schema.wgslOverrides.TILE_SIZE; }, /positive u32/);
  rejects(schema => { schema.variantMetadata.workgroupConstants = ['TILE_SIZE']; }, /three workgroupConstants/);
  rejects(schema => { delete schema.variantMetadata.workgroupStorageBytesPerInvocation; }, /storage bytes/);
  rejects(schema => { schema.wgslOverrides.TILE_SIZE = 17; }, /maxComputeInvocationsPerWorkgroup/);
  rejects(schema => { schema.wgslOverrides.TILE_SIZE = 257; }, /maxComputeWorkgroupSizeX/);
  withVariant('f32_compensated', schema => { schema.wgslOverrides.TILE_SIZE = 8; }, () => {
    const cached = getMatmulConfig('f32_compensated', null);
    device({ maxComputeInvocationsPerWorkgroup: 32 });
    assert.throws(() => getMatmulConfig('f32_compensated', null), /maxComputeInvocationsPerWorkgroup/);
    device({ maxComputeWorkgroupStorageSize: 511 });
    assert.throws(() => getMatmulConfig('f32_compensated', null), /maxComputeWorkgroupStorageSize/);
    device();
    assert.equal(getMatmulConfig('f32_compensated', null), cached);
  });
  withVariant('f32_compensated', schema => { schema.sharedMemory = 64; schema.wgslOverrides.TILE_SIZE = 8; }, () => {
    device({ maxComputeWorkgroupStorageSize: 575 });
    assert.throws(() => getMatmulConfig('f32_compensated', null), /maxComputeWorkgroupStorageSize/);
    device();
  });
  withVariant('gemv_f32_subgroup', () => {}, () => {
    device({ maxComputeWorkgroupStorageSize: 0 });
    const cached = getMatmulConfig('gemv_f32_subgroup', null);
    assert.equal(cached.sharedMemory, 0);
    assert.equal(cached.variantMetadata.workgroupStorageBytesPerInvocation, 0);
    device({ maxComputeWorkgroupStorageSize: 0, maxComputeWorkgroupSizeX: 64 });
    assert.throws(() => getMatmulConfig('gemv_f32_subgroup', null), /maxComputeWorkgroupSizeX/);
    device({ maxComputeWorkgroupStorageSize: 0, maxComputeInvocationsPerWorkgroup: 64 });
    assert.throws(() => getMatmulConfig('gemv_f32_subgroup', null), /maxComputeInvocationsPerWorkgroup/);
    device({ maxComputeWorkgroupStorageSize: 0 });
    assert.equal(getMatmulConfig('gemv_f32_subgroup', null), cached);
    device();
  });
  for (const bytes of [-1, 0.5, NaN, undefined]) {
    assert.throws(() => withVariant('gemv_f32_subgroup', schema => { schema.variantMetadata.workgroupStorageBytesPerInvocation = bytes; },
      () => getMatmulConfig('gemv_f32_subgroup', null)), /storage bytes|JSON|json|finite/);
  }
  withVariant('gemv_f32_subgroup', schema => { schema.sharedMemory = 64; }, () => {
    device({ maxComputeWorkgroupStorageSize: 63 });
    assert.throws(() => getMatmulConfig('gemv_f32_subgroup', null), /maxComputeWorkgroupStorageSize/);
    device({ maxComputeWorkgroupStorageSize: 64 });
    assert.equal(getMatmulConfig('gemv_f32_subgroup', null).sharedMemory, 64);
    device();
  });
  const legacy = getMatmulConfig('f32', null);
  assert.equal(getMatmulConfig('f32', { TILE_SIZE: 8 }), legacy, 'unopted variants must keep existing specialization behavior');
  console.log('matmul-workgroup-geometry.test: ok (final JSON geometry, precedence, device/cache capacity, legacy isolation)');
} finally {
  setDevice(null);
}
