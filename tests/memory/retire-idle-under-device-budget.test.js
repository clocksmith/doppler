// A sequence boundary under an explicit device budget retires idle pooled
// buffers (charged as live) only after confirmed GPU completion, never touches
// owned buffers, and leaves pool reuse unchanged when no budget is set.
import assert from 'node:assert/strict';
import { getBufferPool, destroyBufferPool, BufferUsage } from '../../src/memory/buffer-pool.js';
import { retireIdleBuffersUnderDeviceBudget } from '../../src/memory/sequence-retirement.js';
import { setDeviceMemoryBudget, getDeviceMemorySnapshot, installDeviceMemoryAccounting } from '../../src/memory/device-budget.js';

function device() {
  let completions = 0;
  const value = {
    lost: new Promise(() => {}),
    limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
    queue: { onSubmittedWorkDone: async () => { completions++; } },
    createBuffer({ size, usage }) {
      return { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
    },
    get completions() { return completions; },
  };
  return value;
}

{
  const gpu = device();
  setDeviceMemoryBudget(gpu, 1 << 20);
  const pool = getBufferPool(gpu);
  const owned = pool.acquire(4096, BufferUsage.STORAGE, 'owned');
  const idleA = pool.acquire(8192, BufferUsage.STORAGE, 'idle-a');
  const idleB = pool.acquire(16384, BufferUsage.STORAGE, 'idle-b');
  pool.release(idleA);
  pool.release(idleB);
  const before = getDeviceMemorySnapshot(gpu).liveBytes;
  assert.equal(pool.getStats().pooledBuffers, 2, 'Released buffers are pooled');
  assert.equal(await retireIdleBuffersUnderDeviceBudget(gpu), true);
  assert.equal(pool.getStats().pooledBuffers, 0);
  assert.equal(idleA.destroyed && idleB.destroyed, true, 'Idle buffers retire');
  assert.equal(owned.destroyed, false, 'An owned buffer is never retired');
  assert.ok(gpu.completions >= 1, 'Retirement waits for submitted work');
  assert.equal(getDeviceMemorySnapshot(gpu).liveBytes, before - idleA.size - idleB.size);
  pool.release(owned);
  destroyBufferPool(gpu);
}

{
  const gpu = device();
  installDeviceMemoryAccounting(gpu);
  const pool = getBufferPool(gpu);
  const idle = pool.acquire(8192, BufferUsage.STORAGE, 'idle');
  pool.release(idle);
  assert.equal(await retireIdleBuffersUnderDeviceBudget(gpu), false, 'No budget keeps pool reuse');
  assert.equal(pool.getStats().pooledBuffers, 1);
  assert.equal(pool.acquire(8192, BufferUsage.STORAGE, 'reuse'), idle);
  destroyBufferPool(gpu);
}

assert.equal(await retireIdleBuffersUnderDeviceBudget(null), false);
console.log('retire-idle-under-device-budget: ok');
