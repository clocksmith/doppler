import assert from 'node:assert/strict';
import { captureRequestedDevices, observeSettled, observeSubmittedCancellation } from '../../tools/reranker-lifecycle-observation.js';

const previousAdapter = globalThis.GPUAdapter, previousQueue = globalThis.GPUQueue;
class Queue { submit() { this.submitted = true; } }
class Adapter { async requestDevice() { return { queue: new Queue() }; } }
globalThis.GPUQueue = Queue; globalThis.GPUAdapter = Adapter;
try {
  const originalRequest = Adapter.prototype.requestDevice;
  const capture = captureRequestedDevices();
  const device = await new Adapter().requestDevice();
  capture.restore();
  assert.equal(Adapter.prototype.requestDevice, originalRequest);
  assert.deepEqual(capture.devices, [device]);
  // Doppler's submit tracker installs a bound own-property wrapper.
  const boundSubmit = device.queue.submit.bind(device.queue);
  device.queue.submit = boundSubmit;
  const originalSubmit = Queue.prototype.submit;
  const cancelled = await observeSubmittedCancellation(async signal => {
    assert.equal(signal.aborted, false);
    device.queue.submit([]);
    assert.equal(device.queue.submitted, true);
    signal.throwIfAborted();
  }, capture.devices, 1000);
  assert.equal(cancelled.device, device);
  assert.equal(cancelled.observation.cancellationHonored, true);
  assert.equal(Queue.prototype.submit, originalSubmit);
  assert.equal(device.queue.submit, boundSubmit);
  const ignored = await observeSubmittedCancellation(async () => {
    device.queue.submit([]); return 'completed';
  }, capture.devices, 1000);
  assert.equal(ignored.observation.cancellationHonored, false);
  assert.equal(ignored.observation.outcome.value, 'completed');
  const failure = await observeSubmittedCancellation(() => { throw new Error('before submit'); }, capture.devices, 1000);
  assert.equal(failure.observation.triggeredAfterSubmission, false);
  assert.equal(failure.observation.cancellationHonored, false);
  assert.equal(Queue.prototype.submit, originalSubmit);
  assert.equal(device.queue.submit, boundSubmit);
  const deadline = await observeSettled(() => new Promise(() => {}), 1);
  assert.equal(deadline.status, 'timeout');
} finally {
  if (previousAdapter === undefined) delete globalThis.GPUAdapter; else globalThis.GPUAdapter = previousAdapter;
  if (previousQueue === undefined) delete globalThis.GPUQueue; else globalThis.GPUQueue = previousQueue;
}
console.log('reranker lifecycle observation preserves submissions, records ignored aborts and restores hooks on failure');
