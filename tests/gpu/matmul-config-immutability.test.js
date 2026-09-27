import assert from 'node:assert/strict';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
import { getUniformByteLength, writeUniformsFromObject } from '../../src/gpu/kernels/uniform-utils.js';

const canonical = getKernelConfig('matmul', 'gemv_subgroup_multicol');
const specialized = getMatmulConfig('gemv_subgroup_multicol', { WORKGROUP_SIZE: 128, MULTICOL_COLS_PER_WG: 8 });
assert.notEqual(specialized, canonical);
assert.ok(Object.isFrozen(specialized));
assert.ok(Object.isFrozen(specialized.workgroupSize));
assert.ok(Object.isFrozen(specialized.variantMetadata));
assert.equal(specialized.uniforms, canonical.uniforms);
assert.throws(() => { specialized.variantMetadata.colsPerWg = 1; }, TypeError);
assert.throws(() => { specialized.workgroupSize[0] = 1; }, TypeError);
assert.equal(specialized.workgroupSize[0], 128);
assert.equal(specialized.variantMetadata.colsPerWg, 8);
const buffer = new ArrayBuffer(getUniformByteLength(specialized));
writeUniformsFromObject(new DataView(buffer), specialized,
  { M: 1, N: 16, K: 16, alpha: 1, transpose_b: 1, workgroups_x: 1, num_blocks_per_row: 1 });
assert.equal(getMatmulConfig('gemv_subgroup_multicol', null), canonical);
console.log('matmul-config-immutability: specialized dispatch retains immutable generated uniform contract');
