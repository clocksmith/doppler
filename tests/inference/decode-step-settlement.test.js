import assert from 'node:assert/strict';
import { BufferPool, BufferUsage } from '../../src/memory/buffer-pool.js';
import { DEFAULT_BUFFER_POOL_CONFIG } from '../../src/config/schema/buffer-pool.schema.js';
import { setDeviceMemoryBudget, getDeviceMemorySnapshot } from '../../src/memory/device-budget.js';
import { createDecodeStepResources } from '../../src/inference/pipelines/text/generator/decode-components.js';

// Real ownership/pool/budget contracts, simulated GPU queue completion only.
let confirmed = false;
const device = {
  lost: new Promise(() => {}),
  limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
  queue: { onSubmittedWorkDone: async () => { if (!confirmed) throw Error('unknown GPU completion'); } },
  createBuffer({ size, usage }) { return { size, usage, destroyed: false, destroy() { this.destroyed = true; } }; },
};
setDeviceMemoryBudget(device, 512);
const config = structuredClone(DEFAULT_BUFFER_POOL_CONFIG);
config.budget = { maxTotalBytes: 512, highWatermarkRatio: 0.9, emergencyTrimTargetRatio: 0.75, hardFailOnBudgetExceeded: true };
const pool = new BufferPool(false, config, device);
const borrowed = pool.acquire(256, BufferUsage.STORAGE, 'session-hidden');
const owned = pool.acquire(256, BufferUsage.STORAGE, 'decode-output');
const calls = [];
const lifecycle = createDecodeStepResources({
  async abort() { calls.push('abort'); throw Error('abort unsettled'); },
  async completeDeferredCleanup() { calls.push('deferred'); throw Error('cleanup unsettled'); },
}, buffer => pool.release(buffer), buffer => pool.discard(buffer));
lifecycle.resources.register(borrowed, 'borrowed', 'borrowed');
lifecycle.resources.register(owned, 'owned');
const primary = Error('sampling failed');
await assert.rejects(lifecycle.close(true, primary), error => {
  assert(error instanceof AggregateError); assert.equal(error.cause, primary); assert.equal(error.errors.length, 3); return true;
});
assert.deepEqual(calls, ['abort', 'deferred']);
assert.equal(pool.getStats().resources.deferredCleanup.bytes, 256);
assert.equal(pool.getStats().resources.active.bytes, 256);
assert.equal(pool.getStats().resources.reusable.bytes, 0);
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 512);
assert.equal(owned.destroyed, false);
assert.throws(() => pool.acquire(256, BufferUsage.STORAGE, 'cannot-reuse-unknown'), error => error.code === 'RESOURCE_EXHAUSTED');
await assert.rejects(pool.settleDeferredDestruction(), /unknown GPU completion/);
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 512);
confirmed = true;
await pool.settleDeferredDestruction();
assert.equal(owned.destroyed, true); assert.equal(borrowed.destroyed, false);
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 256);
pool.destroy(); await pool.settleDeferredDestruction();
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
console.log('decode-step-settlement: unknown ownership remains charged until confirmed settlement');
