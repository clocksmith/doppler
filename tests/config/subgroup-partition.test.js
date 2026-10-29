import assert from 'node:assert/strict';
import { assertSubgroupPartition } from '../../src/gpu/kernels/subgroup-partition.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
for (const lanes of [1, 2, 4, 8, 16, 32]) assertSubgroupPartition(lanes, { subgroupMinSize: 32, subgroupMaxSize: 64 });
for (const info of [null, {}, { subgroupMinSize: 4, subgroupMaxSize: 32 }, { subgroupMinSize: 16, subgroupMaxSize: 8 }]) {
  assert.throws(() => assertSubgroupPartition(8, info), /unsupported or unavailable/);
}
assert.throws(() => assertSubgroupPartition(3, { subgroupMinSize: 32, subgroupMaxSize: 64 }), /power of two/);
const config = getMatmulConfig('q4_fused_subgroup8', { COLS_PER_WG: 1, WORKGROUP_SIZE: 128, THREADS_PER_COL_GEMV: 1 });
assert.equal(config.variantMetadata.colsPerWg, 32, 'dispatch must use the same registry overrides as shader compilation');
assert.equal(config.workgroupSize[0], 256);
console.log('subgroup-partition.test: passed');
