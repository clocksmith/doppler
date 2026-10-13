import assert from 'node:assert/strict';
import { BufferPool, BufferUsage } from '../../src/memory/buffer-pool.js';
import { DEFAULT_BUFFER_POOL_CONFIG } from '../../src/config/schema/buffer-pool.schema.js';
import { setDeviceMemoryBudget, getDeviceMemorySnapshot } from '../../src/memory/device-budget.js';
const rows = [];
for (const [tokens, expected] of [[1, 256], [184, 65536], [256, 65536], [52, 16384], [1588, 458752]]) {
  const device = { lost: new Promise(() => {}), limits: { maxBufferSize: 2 ** 28, maxStorageBufferBindingSize: 2 ** 28 },
    queue: { onSubmittedWorkDone: async () => {} }, createBuffer({ size, usage }) { return { size, usage, destroy() {} }; } };
  setDeviceMemoryBudget(device, 1420000000);
  const pool = new BufferPool(false, structuredClone(DEFAULT_BUFFER_POOL_CONFIG), device);
  const owned = [];
  // Conservative recorded bound: two attempts, six linear layers per 8-layer
  // partition. Actual selected prefill submits each layer independently.
  for (let attempt = 0; attempt < 2; attempt++) for (let layer = 0; layer < 6; layer++) {
    const gates = pool.acquire(tokens * 16 * 16, BufferUsage.STORAGE, 'paired_linear_gates');
    assert.equal(gates.size, expected); owned.push(gates);
    owned.push(pool.acquire(80, BufferUsage.UNIFORM, 'gate_uniform'));
  }
  const bound = 12 * (expected + 256);
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, bound);
  rows.push({ tokens, logicalGateBytes: tokens * 256, poolGateBytes: expected, extraUniformBytes: 256, twoAttemptsSixLayersBytes: bound });
  for (const buffer of owned) pool.release(buffer);
  pool.destroy(); await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
}
{
  let complete; const completion = new Promise(resolve => { complete = resolve; });
  const device = { lost: new Promise(() => {}), limits: { maxBufferSize: 2 ** 28, maxStorageBufferBindingSize: 2 ** 28 },
    queue: { onSubmittedWorkDone: () => completion }, createBuffer({ size, usage }) { return { size, usage, destroy() {} }; } };
  setDeviceMemoryBudget(device, 131072);
  const pool = new BufferPool(false, structuredClone(DEFAULT_BUFFER_POOL_CONFIG), device);
  const survivor = pool.acquire(65536, BufferUsage.STORAGE, 'surviving_attempt_gate');
  const cancelled = pool.acquire(65536, BufferUsage.STORAGE, 'cancelled_attempt_gate');
  pool.discard(cancelled);
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 131072, 'Cancelled unconfirmed work stays charged');
  assert.throws(() => pool.acquire(65536, BufferUsage.STORAGE, 'third_attempt'), e => e.code === 'RESOURCE_EXHAUSTED');
  complete(); await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 65536, 'Only cancelled owner is destroyed after completion');
  pool.release(survivor); pool.destroy(); await pool.settleDeferredDestruction();
}
console.log(JSON.stringify({ scope: 'Calculated incremental buffer bounds using actual pool allocator; no hardware peak claim', rows }, null, 2));
