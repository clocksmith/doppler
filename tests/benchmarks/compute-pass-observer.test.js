import assert from 'node:assert/strict';
import { observeComputePasses } from '../../tools/experiments/compute-pass-observer.js';

// CPU boundary test: an encoded pass is not evidence of submitted GPU work.
class Pass {
  setPipeline() {}
  dispatchWorkgroups() {}
  dispatchWorkgroupsIndirect() {}
  end() {}
}
class Encoder {
  constructor() { this.operations = []; }
  beginComputePass({ timestampWrites }) {
    if (timestampWrites) this.operations.push(() => {
      const { querySet, beginningOfPassWriteIndex: a, endOfPassWriteIndex: b } = timestampWrites;
      if (!missing) { querySet.values[a] = 100n; querySet.values[b] = 1000100n; }
    });
    return new Pass();
  }
  finish() { return { operations: this.operations }; }
  resolveQuerySet(q, start, count, buffer) {
    this.operations.push(() => new BigUint64Array(buffer.data).set(q.values.slice(start, start + count)));
  }
  copyBufferToBuffer(src, a, dst, b, size) {
    this.operations.push(() => new Uint8Array(dst.data, b, size).set(new Uint8Array(src.data, a, size)));
  }
}
class Queue {
  submit(buffers) { for (const buffer of buffers) for (const op of buffer.operations) op(); }
}
const globals = ['GPUCommandEncoder', 'GPUComputePassEncoder', 'GPUQueue', 'GPUBufferUsage', 'GPUMapMode'];
const originalGlobals = globals.map(key => Object.getOwnPropertyDescriptor(globalThis, key));
Object.assign(globalThis, { GPUCommandEncoder: Encoder, GPUComputePassEncoder: Pass, GPUQueue: Queue,
  GPUBufferUsage: { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 }, GPUMapMode: { READ: 1 } });
const originals = [Encoder.prototype.beginComputePass, Encoder.prototype.finish, Queue.prototype.submit];
let missing = false;
const resources = [];
const device = {
  features: new Set(['timestamp-query']), queue: new Queue(),
  createCommandEncoder: () => new Encoder(),
  createQuerySet({ count }) {
    const result = { values: new BigUint64Array(count), destroyed: false, destroy() { this.destroyed = true; } };
    resources.push(result); return result;
  },
  createBuffer({ size }) {
    const result = { data: new ArrayBuffer(size), mapState: 'unmapped', destroyed: false,
      async mapAsync() { this.mapState = 'mapped'; }, getMappedRange() { return this.data; },
      unmap() { this.mapState = 'unmapped'; }, destroy() { this.destroyed = true; } };
    resources.push(result); return result;
  },
};
// Doppler wraps the queue instance at device initialization, before diagnostics.
const trackedSubmit = device.queue.submit.bind(device.queue);
device.queue.submit = buffers => trackedSubmit(buffers);
const originalQueueSubmit = device.queue.submit;
const options = { queriesPerBlock: 4, maxPasses: 4 };
function encode() {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass({ label: 'work' });
  pass.setPipeline({ label: 'kernel' }); pass.dispatchWorkgroups(1); pass.end();
  return encoder;
}
function restored() {
  assert.deepEqual([Encoder.prototype.beginComputePass, Encoder.prototype.finish, Queue.prototype.submit], originals);
  assert.ok(resources.every(value => value.destroyed));
  assert.equal(device.queue.submit, originalQueueSubmit);
}
try {
  let observer = observeComputePasses(device, options);
  device.queue.submit([encode().finish()]);
  encode().finish(); encode();
  let result = await observer.collect();
  assert.equal(result.submittedPasses, 1); assert.equal(result.unsubmittedPasses, 2);
  assert.equal(result.sumPassMs, 1); assert.equal(result.byPipeline.kernel.ms, 1);
  assert.ok(result.records.slice(1).every(value => value.status === 'encoded-not-submitted' && value.ms === undefined));
  restored();

  // Native Node objects need not have matching browser constructor globals.
  delete globalThis.GPUCommandEncoder;
  delete globalThis.GPUComputePassEncoder;
  observer = observeComputePasses(device, options);
  device.queue.submit([encode().finish()]);
  result = await observer.collect();
  assert.equal(result.submittedPasses, 1); assert.equal(result.sumPassMs, 1);
  restored();
  Object.assign(globalThis, { GPUCommandEncoder: Encoder, GPUComputePassEncoder: Pass });

  observer = observeComputePasses(device, options);
  const pending = encode(); observer.stop(); device.queue.submit([pending.finish()]);
  result = await observer.collect(); assert.equal(result.submittedPasses, 1); assert.equal(result.sumPassMs, 1);
  restored();

  missing = true; observer = observeComputePasses(device, options);
  device.queue.submit([encode().finish()]);
  await assert.rejects(observer.collect(), /Missing or reversed/); restored(); missing = false;

  observer = observeComputePasses(device, { queriesPerBlock: 2, maxPasses: 1 });
  assert.throws(() => device.createCommandEncoder().beginComputePass({ timestampWrites: {} }), /timestamp owner/);
  encode(); assert.throws(encode, /budget exhausted/); observer.dispose(); restored();
  console.log('compute-pass-observer: submitted ownership, late submit, missing timestamps, cleanup passed');
} finally {
  for (let i = 0; i < globals.length; i++) {
    if (originalGlobals[i]) Object.defineProperty(globalThis, globals[i], originalGlobals[i]);
    else delete globalThis[globals[i]];
  }
}
