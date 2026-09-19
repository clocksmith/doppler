import assert from 'node:assert/strict';
import { installGpuObservation } from '../../tools/lib/installed-gpu-observation.js';

let submissions = 0, fences = 0, maps = 0, copies = 0, dispatches = 0;
let finish;
const pending = new Promise(resolve => { finish = resolve; });
class Queue {
  submit() { submissions++; return 'submitted'; }
  onSubmittedWorkDone() { fences++; return pending; }
}
class Buffer {
  constructor(label, usage) { this.label = label; this.usage = usage; }
  mapAsync() { maps++; return pending; }
}
class Encoder { copyBufferToBuffer() { copies++; } }
class Pass {
  dispatchWorkgroups() { dispatches++; }
  dispatchWorkgroupsIndirect() { dispatches++; }
}
Object.assign(globalThis, { GPUQueue: Queue, GPUBuffer: Buffer, GPUCommandEncoder: Encoder,
  GPUComputePassEncoder: Pass, GPUBufferUsage: { MAP_READ: 1 } });
const original = Queue.prototype.submit;
const observer = installGpuObservation();
try {
  const queue = new Queue();
  const source = new Buffer('logits', 0), target = new Buffer('staging', 1);
  queue.submit(); // Disabled observation must not add work or counters.
  observer.start();
  assert.equal(queue.submit(), 'submitted');
  assert.equal(queue.onSubmittedWorkDone(), pending, 'preserve native promise identity');
  assert.equal(target.mapAsync(), pending);
  new Encoder().copyBufferToBuffer(source, 0, target, 0, 608);
  new Encoder().copyBufferToBuffer(source, 0, source, 0, 608);
  new Pass().dispatchWorkgroups(1);
  new Pass().dispatchWorkgroupsIndirect(source, 0);
  finish(); await pending;
  const observation = observer.stop();
  assert.deepEqual([submissions, fences, maps, copies, dispatches], [2, 1, 1, 2, 2]);
  assert.equal(observation.counts.submissions, 1);
  assert.equal(observation.counts.dispatches, 1);
  assert.equal(observation.counts.indirectDispatches, 1);
  assert.equal(observation.rows.filter(row => row.kind === 'readback-copy').length, 1);
  assert.equal(observation.rows.find(row => row.kind === 'readback-copy').bytes, 608);
  assert.equal(observation.rows.filter(row => row.failed === false).length, 2);
} finally { observer.restore(); }
assert.equal(Queue.prototype.submit, original);

const failure = new Error('device lost');
const rejected = Promise.reject(failure);
Buffer.prototype.mapAsync = () => rejected;
const failedObserver = installGpuObservation();
try {
  failedObserver.start();
  assert.equal(new Buffer('failed', 1).mapAsync(), rejected);
  await assert.rejects(rejected, error => error === failure);
  assert.equal(failedObserver.stop().rows[0].failed, true);
} finally { failedObserver.restore(); }
console.log('installed-gpu-observation.test: counters, no extra GPU work, native result/failure identity and restoration passed');
