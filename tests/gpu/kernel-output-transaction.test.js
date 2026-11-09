import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// The executor's buffer-pool port is replaced; the real transaction helpers run.
const target = new URL('../../src/gpu/kernels/kernel-execution.js', import.meta.url).href;
const live = new Set();
const pool = {
  failOn: null,
  acquired: [],
  acquire(size, usage, label) {
    if (label === this.failOn) throw new Error(`allocation of ${label} rejected`);
    const buffer = { size, label };
    live.add(buffer);
    this.acquired.push(buffer);
    return buffer;
  },
  release(buffer) { assert(live.delete(buffer), `release of an unknown buffer ${buffer?.label}`); },
};
globalThis.kernelOutputPool = pool;
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../../memory/buffer-pool.js') return {
    url: `data:text/javascript,${encodeURIComponent(`export const acquireBuffer = (size, usage, label) => kernelOutputPool.acquire(size, usage, label);
      export const releaseBuffer = (buffer) => kernelOutputPool.release(buffer);`)}`,
    shortCircuit: true,
  };
  return next(specifier, context);
} });
const { withKernelOutputs, withKernelOutput } = await import(target);

try {
  // A failed second acquisition rolls the first back.
  pool.failOn = 'gate';
  await assert.rejects(() => withKernelOutputs(null, { query: { bytes: 64, label: 'query' }, gate: { bytes: 64, label: 'gate' } }, async () => assert.fail('executor must not run')), /allocation of gate rejected/);
  assert.equal(live.size, 0, 'the query buffer was released when the gate allocation failed');

  // A failing executor rolls every acquired output back, in acquisition order.
  pool.failOn = null;
  pool.acquired.length = 0;
  await assert.rejects(() => withKernelOutputs(null, { a: { bytes: 16 }, b: { bytes: 32, label: 'second' } }, async (outputs) => {
    assert.deepEqual(Object.keys(outputs), ['a', 'b']);
    assert.equal(outputs.b.label, 'second');
    assert.equal(outputs.a.label, 'a', 'an output without a label is labelled by its name');
    throw new Error('dispatch failed');
  }), /dispatch failed/);
  assert.equal(live.size, 0);

  // On a recorder, failed outputs become temporaries instead of being released.
  const temporaries = [];
  const recorder = { beginComputePass() {}, trackTemporaryBuffer: (buffer) => temporaries.push(buffer) };
  await assert.rejects(() => withKernelOutputs(recorder, { out: { bytes: 8 } }, async () => { throw new Error('recorded failure'); }), /recorded failure/);
  assert.equal(temporaries.length, 1);
  assert.equal(live.size, 1, 'the recorder owns the temporary until it cleans up');
  pool.release(temporaries.pop());

  // Supplied buffers are never acquired or released, even on failure.
  const supplied = { size: 8, label: 'caller-owned' };
  await assert.rejects(() => withKernelOutputs(null, { out: { supplied }, scratch: { bytes: 8 } }, async (outputs) => {
    assert.equal(outputs.out, supplied);
    throw new Error('boom');
  }), /boom/);
  assert.equal(live.size, 0);

  // Success transfers ownership: nothing is released and the result is returned.
  const result = await withKernelOutputs(null, { x: { bytes: 4 }, y: { bytes: 4 } }, async ({ x, y }) => ({ x, y }));
  assert.equal(live.size, 2);
  assert.ok(live.has(result.x) && live.has(result.y));
  pool.release(result.x); pool.release(result.y);

  // The single-output helper is the same transaction.
  await assert.rejects(() => withKernelOutput(null, null, 4, 'single', async () => { throw new Error('single failure'); }), /single failure/);
  assert.equal(live.size, 0);
  const single = await withKernelOutput(null, null, 4, 'single', async (output) => output);
  assert.equal(single.label, 'single');
  pool.release(single);
  console.log('kernel-output-transaction: rollback on acquisition and execution failure, recorder temporaries and ownership transfer verified');
} finally {
  hooks.deregister();
  delete globalThis.kernelOutputPool;
}
