import assert from 'node:assert/strict';
import { BufferPool, BufferUsage } from '../../src/memory/buffer-pool.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';
import { DEFAULT_BUFFER_POOL_CONFIG } from '../../src/config/schema/runtime/buffer-pool.schema.js';

function fixture() {
  let release;
  const completed = new Promise(resolve => { release = resolve; });
  const state = { submissions: 0, encoders: 0, maps: 0, failMap: false, failSubmit: false };
  const device = {
    lost: new Promise(() => {}), features: new Set(),
    limits: { maxBufferSize: 8192, maxStorageBufferBindingSize: 8192 },
    queue: {
      submit(commands) {
        if (state.failSubmit) throw new Error('Submission failed');
        state.submissions++;
        for (const command of commands) for (const op of command) op();
      },
      onSubmittedWorkDone: () => completed,
    },
    createBuffer({ size, usage }) {
      const bytes = new Uint8Array(size);
      return { size, usage, bytes, mapped: false, destroyed: false,
        async mapAsync() {
          state.maps++;
          await completed;
          if (state.failMap) throw new Error('Mapping failed');
          this.mapped = true;
        },
        getMappedRange(offset, length) { assert(this.mapped); return bytes.buffer.slice(offset, offset + length); },
        unmap() { assert(this.mapped); this.mapped = false; },
        destroy() { this.destroyed = true; },
      };
    },
    createCommandEncoder() {
      state.encoders++;
      const ops = [];
      return {
        writeForTest(buffer, data) { ops.push(() => buffer.bytes.set(data)); },
        copyBufferToBuffer(source, offset, target, destination, length) {
          ops.push(() => target.bytes.set(source.bytes.subarray(offset, offset + length), destination));
        },
        finish() { return ops; },
      };
    },
  };
  setDeviceMemoryBudget(device, 8192);
  return { device, pool: new BufferPool(false, structuredClone(DEFAULT_BUFFER_POOL_CONFIG), device), state, release };
}

{
  const { pool, device, state, release } = fixture();
  const source = pool.acquire(40, BufferUsage.STORAGE, 'route_ids');
  const recorder = new CommandRecorder(device);
  const ids = Uint32Array.from({ length: 10 }, (_, index) => index + 17);
  recorder.getEncoder().writeForTest(source, new Uint8Array(ids.buffer));
  let published = false;
  const reading = pool.readBufferSlice(source, 0, 40, recorder).then(bytes => { published = true; return bytes; });
  await Promise.resolve();
  assert.equal(state.submissions, 1, 'Route production and its ID copy share one submission');
  assert.equal(state.encoders, 2, 'Only the original encoder and next recorder prefix exist');
  assert.equal(published, false, 'CPU cannot consume selected IDs before map completion');
  assert.equal(pool.getStats().activeBuffers, 2, 'Source and staging remain owned during mapping');
  release();
  assert.deepEqual(new Uint32Array(await reading), ids);
  assert.equal(pool.getStats().activeBuffers, 1);
  const allocated = pool.getStats().allocations;
  await pool.readBufferSlice(source, 0, 40, recorder);
  assert.equal(pool.getStats().allocations, allocated, 'Later route cuts reuse the unmapped staging allocation');
  assert.equal(state.submissions, 2);
  await recorder.abort();
  pool.release(source); pool.destroy(); await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
}

for (const failure of ['failMap', 'failSubmit']) {
  const { pool, device, state, release } = fixture();
  const source = pool.acquire(40, BufferUsage.STORAGE, 'route_ids');
  const recorder = new CommandRecorder(device);
  state[failure] = true; release();
  await assert.rejects(pool.readBufferSlice(source, 0, 40, recorder), /failed/i);
  assert.equal(pool.getStats().activeBuffers, 1, 'A failed read relinquishes its staging ownership');
  assert.equal(source.destroyed, false, 'Readback failure does not destroy model-owned input');
  await recorder.abort();
  pool.release(source); pool.destroy(); await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
}

{
  const { pool, device, state, release } = fixture();
  const source = pool.acquire(40, BufferUsage.STORAGE, 'route_ids');
  const other = new CommandRecorder(fixture().device);
  const before = pool.getStats().allocations;
  await assert.rejects(pool.readBufferSlice(source, 0, 40, other), /owning device/);
  assert.equal(pool.getStats().allocations, before);
  assert.equal(state.submissions, 0);
  await assert.rejects(pool.readBufferSlice(source, 2, 40), /aligned/);
  release();
  const standalone = await pool.readBufferSlice(source, 0, 40);
  assert.equal(standalone.byteLength, 40);
  assert.equal(state.submissions, 1, 'Immediate callers preserve their standalone readback');
  pool.release(source); pool.destroy(); await pool.settleDeferredDestruction();
  assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
}
console.log('recorded readback: ordered route copy, one submission, completion, reuse, device isolation and failure cleanup passed');
