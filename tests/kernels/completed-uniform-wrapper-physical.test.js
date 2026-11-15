import assert from 'node:assert/strict';
import registry from '../../src/config/kernels/registry.json' with { type: 'json' };
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { getCompletedUniformCache, getCompletedUniformCacheStats } from '../../src/gpu/uniform-cache.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { unifiedKernelWrapper } from '../../src/gpu/kernels/kernel-execution.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Completed uniform wrapper requires physical WebGPU: ${probe.reason}`);
const device = getDevice(), errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const a = acquireBuffer(16), b = acquireBuffer(16), output = acquireBuffer(16);
uploadData(a, new Float32Array([1, 2, 3, 4])); uploadData(b, new Float32Array([4, 3, 2, 1]));
const uniforms = { size: 4, scale: 1, dispatch_stride: 4 };
const execute = target => unifiedKernelWrapper('residual', target, 'default', [a, b, output], uniforms, [1, 1, 1]);
const settle = async () => { await device.queue.onSubmittedWorkDone(); await Promise.resolve(); };
const check = async () => assert.deepEqual(Array.from(new Float32Array(await readBuffer(output, 16))), [5, 5, 5, 5]);
try {
  // Async cleanup; two overlapping leases reuse immutable uniform content.
  await execute(device); await execute(device); await settle(); await check();
  assert.ok(getCompletedUniformCacheStats(device).hits >= 1);
  assert.equal(getCompletedUniformCacheStats(device).pendingDestruction, 0);
  const beforeRecord = getCompletedUniformCacheStats(device);
  const recorder = createCommandRecorder('unchanged-recorder');
  await execute(recorder); await recorder.submitAndWait(); await check();
  assert.deepEqual(getCompletedUniformCacheStats(device), beforeRecord);
  const spec = registry.operations.residual.variants.default;
  const leave = enterKernelRegistry(createKernelRegistry({ extensions: {
    residual: { variants: { default: { ...spec, immediateCompletion: 'settled' } } },
  } }));
  try { await execute(device); await check(); } finally { leave(); }
  // Failure after acquire, before submission must release without destroying cached content.
  const createBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = () => { throw new Error('injected bind failure'); };
  try { await assert.rejects(execute(device), /injected bind/); }
  finally { device.createBindGroup = createBindGroup; }
  await execute(device); await settle(); await check();
  // Uncertain dispatch failure may follow a real submission: defer release until completion.
  const submit = device.queue.submit.bind(device.queue);
  device.queue.submit = commands => { submit(commands); throw new Error('injected post-submit failure'); };
  try { await assert.rejects(execute(device), /injected post-submit/); }
  finally { device.queue.submit = submit; }
  await settle(); await execute(device); await settle(); await check();
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ exact: true, asynchronous: true, settled: true, failures: true,
    recorderUnchanged: true, stats: getCompletedUniformCacheStats(device) }));
} finally {
  await settle(); getCompletedUniformCache(device).clear();
  for (const buffer of [a, b, output]) releaseBuffer(buffer); destroyDevice();
}
