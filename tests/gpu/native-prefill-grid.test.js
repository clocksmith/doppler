import assert from 'node:assert/strict';
import { q40Dispatch } from '../../src/gpu/kernels/q4_0.js';
const device = { limits: { maxComputeWorkgroupsPerDimension: 65535 } };
const groups = (M, N) => Math.ceil(M / 16) * Math.ceil(N / 16);
const full = groups(2287, 8192), chunk = groups(256, 8192);
assert.equal(full, 73216);
assert.deepEqual(q40Dispatch(device, full), [65535, 2, 1]);
assert.equal(65535 * 2 - full, 57854);
assert.equal(chunk, 8192);
assert.deepEqual(q40Dispatch(device, chunk), [8192, 1, 1]);
assert.deepEqual(q40Dispatch({ limits: { maxComputeWorkgroupsPerDimension: 4 } }, 6), [4, 2, 1]);
for (const count of [1, 4, 5, 6, 15, 16]) {
 const grid = q40Dispatch({ limits: { maxComputeWorkgroupsPerDimension: 4 } }, count);
 assert.ok(grid[0] <= 4 && grid[1] <= 4 && grid[0] * grid[1] >= count);
 const valid = [], padding = [];
 for (let y = 0; y < grid[1]; y++) for (let x = 0; x < grid[0]; x++) {
  const index = y * grid[0] + x;
  (index < count ? valid : padding).push(index);
 }
 assert.deepEqual(valid, Array.from({ length: count }, (_, i) => i));
 assert.equal(padding.length, grid[0] * grid[1] - count);
}
console.log(JSON.stringify({ status: 'passed', scope: 'CPU dispatch geometry; not GPU timing or watchdog attribution',
 full: { required: full, grid: [65535, 2, 1], padding: 57854 },
 chunked: { requiredPer256: chunk, grid: [8192, 1, 1] } }));
