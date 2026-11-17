import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, readBufferSlice, getBufferPool } from '../../src/memory/buffer-pool.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';

// Physical command ordering and resource retirement only; no model or timing claim.
const probe = await probeNodeGPU();
if (!probe.ready) throw new Error(`Physical recorded readback unavailable: ${probe.reason}`);
const device = getDevice(), capabilities = getKernelCapabilities();
assert(!/swiftshader|llvmpipe|lavapipe/.test(JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase()));
const allocationCapBytes = 1024 * 1024;
setDeviceMemoryBudget(device, allocationCapBytes);
const pool = getBufferPool(device), baseline = pool.getStats().resources;
const source = acquireBuffer(40), routed = acquireBuffer(40);
const recorder = new CommandRecorder(device, 'physical_route_readback');
const originalSubmit = device.queue.submit;
let submissions = 0, allocations;
device.queue.submit = function (...args) { submissions++; return originalSubmit.apply(this, args); };
try {
  for (let iteration = 0; iteration < 17; iteration++) {
    const expected = Uint32Array.from({ length: 10 }, (_, index) => iteration * 512 + index);
    device.queue.writeBuffer(source, 0, expected);
    // Producing routing work and its staging copy share one real encoder.
    recorder.getEncoder().copyBufferToBuffer(source, 0, routed, 0, 40);
    const actual = new Uint32Array(await readBufferSlice(routed, 0, 40, recorder));
    assert.deepEqual(actual, expected);
    assert.equal(submissions, iteration + 1);
    if (iteration === 0) allocations = pool.getStats().allocations;
    else assert.equal(pool.getStats().allocations, allocations, 'Readback staging is reused');
  }
  await recorder.abort();
  releaseBuffer(source); releaseBuffer(routed);
  await device.queue.onSubmittedWorkDone();
  const after = pool.getStats().resources;
  for (const key of ['active', 'retainedModel', 'deferredCleanup']) assert.equal(after[key].bytes, baseline[key].bytes);
  console.log(JSON.stringify({ completed: true, physical: true, adapter: capabilities?.adapterInfo ?? null,
    submissions, selections: 17, allocationCapBytes, peakBytes: getDeviceMemorySnapshot(device).peakBytes,
    scope: 'Recorded readback ordering, staging reuse and model-free physical retirement' }));
} finally {
  device.queue.submit = originalSubmit;
  await recorder.abort();
  if (pool.isActiveBuffer(source)) releaseBuffer(source);
  if (pool.isActiveBuffer(routed)) releaseBuffer(routed);
  await destroyDevice();
  assert.equal(getDeviceMemorySnapshot(device)?.liveBytes ?? 0, 0);
}
