import assert from 'node:assert/strict';
import { BufferPool, BufferUsage } from '../../src/memory/buffer-pool.js';
import { DEFAULT_BUFFER_POOL_CONFIG } from '../../src/config/schema/buffer-pool.schema.js';
import { setDeviceMemoryBudget, getDeviceMemorySnapshot } from '../../src/memory/device-budget.js';

function fixture(hardFailOnBudgetExceeded) {
  let settle;
  const completion = new Promise(resolve => { settle = resolve; });
  const buffers = [];
  const device = {
    lost: new Promise(() => {}),
    limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
    queue: { onSubmittedWorkDone: () => completion },
    createBuffer({ size, usage }) {
      const buffer = { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
      buffers.push(buffer);
      return buffer;
    },
  };
  setDeviceMemoryBudget(device, 512);
  const config = structuredClone(DEFAULT_BUFFER_POOL_CONFIG);
  config.budget = { maxTotalBytes: 512, highWatermarkRatio: 0.9,
    emergencyTrimTargetRatio: 0.75, hardFailOnBudgetExceeded };
  return { pool: new BufferPool(false, config, device), device, buffers, settle };
}

for (const hardFail of [false, true]) {
  const { pool, device, buffers, settle } = fixture(hardFail);
  const active = pool.acquire(256, BufferUsage.STORAGE, 'another-owner');
  const reusable = pool.acquire(256, BufferUsage.STORAGE, 'reusable');
  pool.release(reusable);
  const reused = pool.acquire(128, BufferUsage.STORAGE, 'reuse-without-allocation');
  assert.equal(reused, reusable, 'Compatible reuse must precede projected new-allocation trimming');
  assert.equal(buffers.length, 2);
  assert.equal(getDeviceMemorySnapshot(device).rejected, 0);
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 512);
  assert.equal(active.destroyed, false);
  assert.equal(pool.getStats().currentBytesRequested, 512, 'Requests retain configured alignment');
  pool.release(reused);
  pool.release(active);
  settle();
  pool.destroy();
  await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
}

{
  const { pool, device, settle } = fixture(false);
  const active = pool.acquire(256, BufferUsage.STORAGE, 'another-owner');
  const pending = pool.acquire(256, BufferUsage.STORAGE, 'pending-completion');
  pool.discard(pending);
  assert.throws(() => pool.acquire(256, BufferUsage.STORAGE, 'new-allocation'),
    error => error.code === 'RESOURCE_EXHAUSTED');
  assert.equal(active.destroyed, false, 'Pressure never destroys another active owner');
  assert.equal(pending.destroyed, false, 'Pressure never treats pending GPU completion as settled');
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 512);
  settle();
  await pool.settleDeferredDestruction();
  const fresh = pool.acquire(256, BufferUsage.STORAGE, 'after-confirmed-settlement');
  assert.equal(pending.destroyed, true);
  assert.equal(active.destroyed, false);
  pool.release(fresh);
  pool.release(active);
  pool.destroy();
  await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
}
console.log('buffer-pool-device-budget.test: ok');
