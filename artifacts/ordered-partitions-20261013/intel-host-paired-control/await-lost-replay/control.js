// Adapted from the frozen paired-cast physical fixture; installed package-local wrappers.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
const [root, providerPath, output] = process.argv.slice(2); assert(root && providerPath && output);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const receipt = { schema: 'doppler.intel-host-paired-cast-control/v1', pid: process.pid, startedAt: new Date().toISOString(),
 operationDeadlineMs: 300000, cleanupDeadlineMs: 10000, poolBudgetBytes: 16777216, cases: 4,
 scope: 'Installed package-local wrapper compatibility; not public package API or model qualification', events: [] };
const persist = () => writeFile(output, JSON.stringify(receipt, null, 2));
const checkpoint = async event => { receipt.events.push({ ...event, at: new Date().toISOString() }); await persist(); };
const bounded = async (promise, ms, label) => { let timer; try { return await Promise.race([promise,
 new Promise((_, reject) => { timer = setTimeout(() => reject(Error(label + ' deadline')), ms); })]); } finally { clearTimeout(timer); } };
const load = path => import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu/src', path)));
const inventory = JSON.parse(await readFile(resolve(root, 'inventory.json')));
for (const file of inventory) assert.equal('sha256:' + sha(await readFile(resolve(root, 'node_modules/doppler-gpu', file.path))), file.hash);
receipt.installedFilesVerified = inventory.length;
let providerReleaseSafe = false;
const provider = await import(pathToFileURL(providerPath)); const saved = new Map();
for (const key of Object.keys(provider.globals).concat('navigator')) saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
Object.assign(globalThis, provider.globals);
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: provider.create(['backend=vulkan']) } });
const { setRuntimeConfig } = await load('config/runtime.js');
setRuntimeConfig({ shared: { bufferPool: { budget: { maxTotalBytes: 16777216, hardFailOnBudgetExceeded: true } } } });
try {
 receipt.sourceHash = sha(await readFile(new URL(import.meta.url)));
 receipt.result = await bounded((async () => {

    const { setDevice, destroyDevice } = await load('gpu/device.js');
    const { createTensor } = await load('gpu/tensor.js');
    const { CommandRecorder } = await load('gpu/command-recorder.js');
    const { expandF32ToPaired, recordExpandF32ToPaired, castPairedToF16, recordCastPairedToF16 } = await load('gpu/kernels/paired-cast.js');
    const { readBufferSlice, releaseBuffer, destroyBufferPool, getBufferPool } = await load('memory/buffer-pool.js');
    await checkpoint({ phase: 'adapter-start' });
    const adapter = await navigator.gpu.requestAdapter();
    await checkpoint({ phase: 'adapter-complete' });
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    await checkpoint({ phase: 'device-start' });
    const device = await adapter.requestDevice(); setDevice(device, { platformConfig: null });
    await checkpoint({ phase: 'device-complete' });
    const errors = [], rows = [], owned = new Set(), pooled = new Set();
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const make = (bytes, upload) => {
      const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      owned.add(buffer); if (upload) device.queue.writeBuffer(buffer, 0, upload); return buffer;
    };
    try {
      const bits = new Uint32Array([0, 0x80000000, 1, 0x80000001, 0x3f800000, 0x7f7fffff, 0x3f801000]);
      for (const recorded of [false, true]) for (const borrowed of [true]) {
        const input = createTensor(make(bits.byteLength, bits), 'f32', [7], 'exact_scalar_bits');
        const options = borrowed ? { outputBuffer: make(bits.length * 8) } : {};
        const recorder = recorded ? new CommandRecorder(device, 'paired_cast_control') : null;
        let output;
        try {
          await checkpoint({ operation: 'expand', recorded, phase: 'dispatch-start' });
          output = recorded ? await recordExpandF32ToPaired(recorder, input, options) : await expandF32ToPaired(input, options);
          if (!borrowed) pooled.add(output.buffer);
          await checkpoint({ phase: 'wrapper-complete', recorded });
          if (recorder) { await checkpoint({ phase: 'submit-start' }); await recorder.submitAndWait(); await checkpoint({ phase: 'submit-complete' }); }
          await checkpoint({ phase: 'readback-start', recorded });
          const actual = Array.from(new Uint32Array(await readBufferSlice(output.buffer, 0, bits.length * 8)));
          await checkpoint({ phase: 'readback-complete', recorded, actual });
          const expected = Array.from(bits).flatMap(word => [word, 0]);
          if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error('Expansion changed source bits.');
          if (output.precision !== 'two-f32' || output.components !== 2 || output.elementStrideBytes !== 8) throw new Error('Paired output metadata missing.');
          rows.push({ operation: 'expand', recorded, borrowed, actual });
          receipt.completedRows = [...rows]; await checkpoint({ phase: 'case-assertions-passed', recorded });
        } finally { if (recorder) { await checkpoint({ phase: 'abort-start' }); await recorder.abort(); await checkpoint({ phase: 'abort-complete' }); } }
      }
      // Halfway 1+2^-11 plus/minus minimum F32 subnormal; exact signs break the tie.
      const pairWords = new Uint32Array([0x3f801000, 1, 0x3f801000, 0x80000001,
        0xbf801000, 1, 0xbf801000, 0x80000001, 0x80000000, 0]);
      const expected = [0x3c01, 0x3c00, 0xbc00, 0xbc01, 0x8000, 0];
      for (const recorded of [false, true]) for (const borrowed of [true]) {
        const input = createTensor(make(pairWords.byteLength, pairWords), 'f32', [5], 'half_midpoints',
          { precision: 'two-f32', components: 2, elementStrideBytes: 8 });
        const options = borrowed ? { outputBuffer: make(12) } : {};
        const recorder = recorded ? new CommandRecorder(device, 'paired_half_control') : null;
        try {
          await checkpoint({ operation: 'half', recorded, phase: 'dispatch-start' });
          const output = recorded ? await recordCastPairedToF16(recorder, input, options) : await castPairedToF16(input, options);
          if (!borrowed) pooled.add(output.buffer);
          await checkpoint({ phase: 'wrapper-complete', recorded });
          if (recorder) { await checkpoint({ phase: 'submit-start' }); await recorder.submitAndWait(); await checkpoint({ phase: 'submit-complete' }); }
          await checkpoint({ phase: 'readback-start', recorded });
          const actual = Array.from(new Uint16Array(await readBufferSlice(output.buffer, 0, 12)));
          await checkpoint({ phase: 'readback-complete', recorded, actual });
          if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`Half bits ${actual} != ${expected}`);
          if (output.dtype !== 'f16' || output.precision !== undefined) throw new Error('Half output contract wrong.');
          rows.push({ operation: 'half', recorded, borrowed, actual });
          receipt.completedRows = [...rows]; await checkpoint({ phase: 'case-assertions-passed', recorded });
        } finally { if (recorder) { await checkpoint({ phase: 'abort-start' }); await recorder.abort(); await checkpoint({ phase: 'abort-complete' }); } }
      }
      await checkpoint({ phase: 'queue-settlement-start' });
      await bounded(device.queue.onSubmittedWorkDone(), 10000, 'cleanup queue');
      await checkpoint({ phase: 'queue-settlement-complete' });
      for (const buffer of pooled) releaseBuffer(buffer); pooled.clear();
      const stats = getBufferPool().getStats();
      if (stats.activeBuffers !== 0 || errors.length) throw new Error(JSON.stringify({ stats, errors }));
      return { rows, errors, stats, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture }, settled: true };
    } finally {
      const cleanupEnd = Date.now() + 10000;
      const cleanupRemaining = () => { const remaining = cleanupEnd - Date.now(); if (remaining <= 0) throw Error('Total cleanup deadline'); return remaining; };
      receipt.cleanupStartedAt = new Date().toISOString();
      await checkpoint({ phase: 'device-finally-entered' });
      await checkpoint({ phase: 'queue-settlement-start' });
      await bounded(device.queue.onSubmittedWorkDone(), cleanupRemaining(), 'cleanup queue');
      await checkpoint({ phase: 'queue-settlement-complete' });
      for (const buffer of pooled) releaseBuffer(buffer);
      await checkpoint({ phase: 'owned-buffer-destroy-start' });
      for (const buffer of owned) buffer.destroy();
      await checkpoint({ phase: 'owned-buffer-destroy-complete' });
      await checkpoint({ phase: 'pool-destroy-start' }); destroyBufferPool(); await checkpoint({ phase: 'pool-destroy-complete' });
      const lost = device.lost;
      assert(lost && typeof lost.then === 'function', 'Owned device lost promise required before provider release');
      await checkpoint({ phase: 'device-destroy-start' }); destroyDevice(); await checkpoint({ phase: 'device-destroy-complete' });
      await checkpoint({ phase: 'device-lost-settlement-start' });
      receipt.deviceLost = await bounded(lost, cleanupRemaining(), 'owned device lost');
      cleanupRemaining();
      receipt.cleanupCompletedAt = new Date().toISOString();
      providerReleaseSafe = true;
      await checkpoint({ phase: 'device-lost-settlement-complete' });
    }
 })(), 300000, 'paired cast operation');
 receipt.passed = receipt.result.rows.length === 4 && receipt.result.settled;
 assert(receipt.passed);
} catch (error) { receipt.error = { name: error.name, message: error.message, stack: error.stack }; process.exitCode = 1; }
finally {
 await checkpoint({ phase: 'provider-finally-entered' });
 if (!providerReleaseSafe) {
   receipt.ownership = 'unknown'; receipt.providerRetained = true;
   await checkpoint({ phase: 'provider-retained-until-ownership-settles' });
 } else {
 for (const [key, descriptor] of saved) { if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key]; }
 await checkpoint({ phase: 'provider-globals-restored' });
 await checkpoint({ phase: 'forced-gc-start' }); globalThis.gc?.(); await checkpoint({ phase: 'forced-gc-complete' }); receipt.finishedAt = new Date().toISOString(); await persist();
 }
}
