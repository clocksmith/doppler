import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setDevice } from '../../src/gpu/device.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { clearShaderCaches, registerShaderSources } from '../../src/gpu/kernels/shader-cache.js';
import { clearPipelineCaches, getPipelineFast } from '../../src/gpu/kernels/pipeline-cache.js';
import { getMatmulConfig, selectMatmulVariantAndFlags } from '../../src/gpu/kernels/matmul-selection.js';
import { getKernelWgslRequirements, hasRequiredFeatures } from '../../src/gpu/kernels/feature-check.js';
import { calculateMatmulDispatch } from '../../src/gpu/kernels/matmul-dispatch.js';

const step = { op: 'q_proj', kernel: 'matmul_gemv_f32_subgroup.wgsl', entry: 'main' };
const kernelPath = { activationDtype: 'f32', decode: { steps: [step] }, prefill: { steps: [step] } };
setDevice({
  features: new Set(['subgroups']),
  limits: {
    maxStorageBufferBindingSize: 1 << 24, maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64, maxComputeWorkgroupStorageSize: 0,
  },
  queue: { submit() {} }, createBindGroup() { return {}; }, lost: new Promise(() => {}),
});
try {
  const select = (M, weight = 'f32', output = 'f32') => selectMatmulVariantAndFlags(
    'run', M, 17, 35, 'f32', weight, true, output, { role: 'q_proj', kernelPath, phaseOverride: 'decode' }
  );
  const selected = select(1);
  assert.equal(selected.variant, 'gemv_f32_subgroup');
  assert.equal(selected.useGemv, true);
  assert.throws(() => select(2), /requires M=1/);
  assert.throws(() => select(1, 'f16'), /requires f32 weights/);
  assert.throws(() => select(1, 'f32', 'f16'), /outputs f32/);
  const config = getMatmulConfig(selected.variant, null);
  const language = getKernelWgslRequirements(config);
  const supported = { hasF16: false, hasSubgroups: true, wgslLanguageFeatures: ['subgroup_id'] };
  assert.equal(hasRequiredFeatures(config.requires, supported, language), true);
  assert.equal(hasRequiredFeatures(config.requires, { ...supported, hasSubgroups: false }, language), false);
  assert.equal(hasRequiredFeatures(config.requires, { ...supported, wgslLanguageFeatures: [] }, language), false);
  const grid = calculateMatmulDispatch(selected.variant, false, true, false, 1, 262151, config);
  assert.equal(grid.workgroups[1], 2);
  assert.equal(grid.uniformWorkgroupsX, grid.workgroups[0]);
} finally {
  setDevice(null);
}

// Independently enumerate ownership/contributions for both widths and partial
// active sets. No formula uses workgroup-local invocation indices or lane IDs.
for (const subgroupSize of [32, 64]) {
  for (const workgroup of [7, 96, 128, 256]) {
    for (const columns of [1, 3, 4, 7, 16]) {
      for (const transpose of [true, false]) {
        for (const K of [1, 2, 3, 4, 35, 129]) {
          const groups = Array.from({ length: Math.ceil(workgroup / subgroupSize) }, (_, id) =>
            ({ id, active: Math.min(subgroupSize, workgroup - id * subgroupSize) }));
          const visits = Array.from({ length: columns }, () => new Uint32Array(K));
          const writes = new Uint32Array(columns);
          const width = transpose ? 1 : 4;
          const batches = Math.ceil(Math.ceil(columns / width) / groups.length);
          for (let batch = 0; batch < batches; batch++) {
            for (const group of groups.toReversed()) {
              const first = (batch * groups.length + group.id) * width;
              for (let component = 0; component < width && first + component < columns; component++) {
                writes[first + component]++;
              }
              // Every active lane has one compact prefix rank, regardless of physical ID.
              for (let rank = 0; rank < group.active; rank++) {
                if (transpose) {
                  for (let chunk = rank * 4; chunk < K; chunk += group.active * 4) {
                    for (let element = chunk; element < Math.min(chunk + 4, K); element++) {
                      if (first < columns) visits[first][element]++;
                    }
                  }
                } else {
                  for (let k = rank; k < K; k += group.active) {
                    for (let component = 0; component < 4 && first + component < columns; component++) {
                      visits[first + component][k]++;
                    }
                  }
                }
              }
            }
          }
          assert.ok(writes.every(count => count === 1), 'each column requires exactly one elected writer');
          for (const column of visits) {
            assert.ok(column.every(count => count === 1), 'every K contribution must occur exactly once');
          }
        }
      }
    }
  }
}

// Arbitrary workgroup sizes require an explicit JSON algorithm declaration.
// Every omitted/true/malformed legacy policy retains the previous restriction.
const registry = JSON.parse(await readFile(new URL('../../src/config/kernels/registry.json', import.meta.url), 'utf8'));
for (const [policy, size] of [[undefined, 7], [true, 7], ['false', 7], ['false', 128], [false, 7], [false, 0], [false, -1], [false, 1.5]]) {
  const schema = structuredClone(registry.operations.rmsnorm.variants.default);
  schema.workgroup = [size, 1, 1];
  schema.wgslOverrides = { WORKGROUP_SIZE: size };
  schema.variantMetadata = { workgroupPowerOfTwo: policy };
  if (policy === undefined) delete schema.variantMetadata.workgroupPowerOfTwo;
  const restore = enterKernelRegistry(createKernelRegistry({ extensions: { rmsnorm: { variants: { default: schema } } } }));
  const device = {
    features: new Set(), limits: {}, lost: new Promise(() => {}),
    queue: { submit() {} }, createBindGroup() { return {}; },
    createShaderModule: descriptor => descriptor,
    createComputePipelineAsync: async descriptor => descriptor,
  };
  setDevice(device);
  clearShaderCaches();
  clearPipelineCaches();
  registerShaderSources({ 'rmsnorm.wgsl': '@compute @workgroup_size(7) fn main() {}' });
  try {
    if (policy === false && size === 7) {
      assert.ok(await getPipelineFast('rmsnorm', 'default', null, null, device));
    } else {
      await assert.rejects(getPipelineFast('rmsnorm', 'default', null, null, device), /power-of-two|positive integer|must be boolean/);
    }
  } finally {
    clearPipelineCaches();
    clearShaderCaches();
    setDevice(null);
    restore();
  }
}
console.log('matmul-f32-subgroup-contract.test: ok (features/dtypes/M, 2D dispatch, 32/64-lane ownership, explicit geometry policy)');
