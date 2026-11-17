import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { MappedExpertUploadCache } from '../../src/loader/experts/mapped-expert-upload-cache.js';

// A short physical upload/lifetime component check. Synthetic bytes only;
// neither this test nor the cache establishes model-source authentication.
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, `Physical mapped-cache check unavailable: ${probe.reason}`);
assert.ok(!/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(getKernelCapabilities()?.adapterInfo)));
const device = getDevice(), capBytes = 8 * 1024 * 1024;
setDeviceMemoryBudget(device, capBytes);
const errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const cache = new MappedExpertUploadCache({ device, maxBytes: 8192, maxMappedBytes: 4096, maxEntries: 2 });
const destination = device.createBuffer({ size: 4096, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
const ranges = [
  { sourceOffset: 0, destination, destinationOffset: 0, bytes: 1024 },
  { sourceOffset: 1024, destination, destinationOffset: 1024, bytes: 2048 },
  { sourceOffset: 3072, destination, destinationOffset: 3072, bytes: 1024 },
];
let fills = 0, beforeClose;
const pattern = tag => Uint8Array.from({ length: 4096 }, (_, i) => (i * 13 + tag) & 255);
const admit = (key, tag, pinned = false) => cache.admit(key, 4096, bytes => {
  fills++; bytes.set(pattern(tag)); return bytes.byteLength;
}, { pinned });
try {
  assert.equal(await admit('synthetic:parked:0', 17, true), true);
  assert.equal(await admit('synthetic:t1:A', 33), true);
  for (const [key, tag] of [['synthetic:parked:0', 17], ['synthetic:t1:A', 33], ['synthetic:t1:A', 33]]) {
    const ticket = cache.copyInto(key, ranges);
    assert.ok(ticket); await ticket.settled;
    assert.deepEqual(new Uint8Array(await readBuffer(destination, 4096)), pattern(tag));
  }
  assert.equal(fills, 2, 'copy hits never call the payload provider again');
  assert.equal(await admit('synthetic:t1:B', 55), true);
  assert.equal(cache.copyInto('synthetic:t1:A', ranges), null, 'bounded T1 eviction leaves the parked entry pinned');
  const ticket = cache.copyInto('synthetic:parked:0', ranges);
  assert.ok(ticket);
  beforeClose = cache.getStats();
  await cache.close(); await ticket.settled;
  assert.deepEqual(new Uint8Array(await readBuffer(destination, 4096)), pattern(17), 'source retirement preserves the submitted copy and caller destination');
  assert.equal(cache.getStats().allocatedBytes, 0);
  assert.equal(cache.getStats().mappedBytes, 0);
  assert.equal(beforeClose.peakBytes, 8192);
  assert.equal(beforeClose.reuses, 1, 'idle same-size upload buffer is remapped and refilled');
  assert.ok(beforeClose.peakMappedBytes <= 4096);
} finally {
  try {
    await cache.close();
    await device.queue.onSubmittedWorkDone();
    destination.destroy();
    const pool = getBufferPool(device); pool.destroy(); await pool.settleDeferredDestruction();
    const retired = getDeviceMemorySnapshot(device);
    assert.equal(retired.liveBytes, 0, 'explicit retirement must complete before device destruction');
    assert.ok(retired.peakBytes <= capBytes);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ test: 'mapped-expert-upload-cache-physical', adapter: getKernelCapabilities()?.adapterInfo,
      beforeClose, final: cache.getStats(), fills, apiCapBytes: capBytes, peakApiBytes: retired.peakBytes,
      retiredLiveBytes: retired.liveBytes, scope: 'synthetic mapped upload, exact copy and retirement; no model or authentication claim' }));
  } finally { destroyDevice(); await releaseNodeWebGPU(); }
}
