// Explicit isolated process. No model, shader, cache or public host is opened.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
const [providerEntry, output] = process.argv.slice(2);
assert(providerEntry && output);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const result = { schema: 'doppler.ordered-provider-allocation-preflight/v1', pid: process.pid,
  startedAt: new Date().toISOString(), node: process.version, platform: process.platform,
  operationDeadlineMs: 45000, externalProcessDeadlineMs: 60000, peakExplicitBufferBytes: 8192,
  scope: 'Adapter/features/limits and 8192-byte clear/copy/readback only; no shader or model qualification', events: [] };
const persist = () => writeFile(output, JSON.stringify(result, null, 2));
const event = async phase => { result.events.push({ phase, at: new Date().toISOString() }); await persist(); };
let gpu, adapter, device, source, target, timer;
const saved = new Map();
try {
  result.providerEntry = { path: providerEntry, hash: hash(await readFile(providerEntry)) };
  result.executable = { path: process.execPath, hash: hash(await readFile(process.execPath)) };
  const provider = await import(pathToFileURL(providerEntry));
  for (const key of Object.keys(provider.globals)) saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  Object.assign(globalThis, provider.globals);
  await event('provider-imported');
  await Promise.race([(async () => {
    gpu = provider.create(['backend=vulkan']);
    adapter = await gpu.requestAdapter(); assert(adapter, 'No Vulkan adapter available');
    result.adapter = Object.fromEntries(['vendor','architecture','device','description','subgroupMinSize','subgroupMaxSize','isFallbackAdapter'].map(key => [key, adapter.info[key]]));
    result.features = [...adapter.features].sort();
    result.limits = {};
    for (const key of Object.getOwnPropertyNames(Object.getPrototypeOf(adapter.limits))) {
      if (key !== 'constructor' && typeof adapter.limits[key] === 'number') result.limits[key] = adapter.limits[key];
    }
    await event('adapter-observed');
    device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', e => result.events.push({ phase: 'uncaptured-error', message: e.error.message }));
    device.lost.then(info => { result.deviceLost = { reason: info.reason, message: info.message }; });
    device.pushErrorScope('validation');
    source = device.createBuffer({ size: 4096, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    target = device.createBuffer({ size: 4096, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.clearBuffer(source); encoder.copyBufferToBuffer(source, 0, target, 0, 4096);
    device.queue.submit([encoder.finish()]); await event('copy-submitted');
    await device.queue.onSubmittedWorkDone(); result.queueSettled = true;
    await target.mapAsync(GPUMapMode.READ); assert(new Uint8Array(target.getMappedRange()).every(value => value === 0)); target.unmap();
    const error = await device.popErrorScope(); assert.equal(error, null); result.readbackPassed = true;
    await event('copy-settled');
  })(), new Promise((_, reject) => { timer = setTimeout(() => reject(Error('45-second preflight operation deadline')), 45000); })]);
} catch (error) { result.error = { name: error.name, message: error.message }; process.exitCode = 1; }
finally {
  clearTimeout(timer);
  source?.destroy(); target?.destroy(); source = null; target = null;
  device?.destroy(); device = null; adapter = null; gpu = null;
  for (const [key, descriptor] of saved) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key]; }
  globalThis.gc?.(); await new Promise(resolve => setTimeout(resolve, 20));
  result.providerReferencesReleased = true; result.finishedAt = new Date().toISOString();
  result.sharedObjects = [];
  for (const path of process.report.getReport().sharedObjects) {
    if (path.startsWith('/')) result.sharedObjects.push({ path, hash: hash(await readFile(path)) });
    else result.sharedObjects.push({ path, virtual: true });
  }
  await persist();
}
