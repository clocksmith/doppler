// Actual installed native host owns provider/device retirement; fixtures retain exact paired bits.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
const [root, output] = process.argv.slice(2); assert(root && output);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const receipt = { schema: 'doppler.actual-native-host-cast-control/v1', pid: process.pid,
 startedAt: new Date().toISOString(), operationDeadlineMs: 300000, cleanupDeadlineMs: 10000,
 scope: 'Actual installed native host lifecycle with four package-local paired cast controls; no model qualification', events: [] };
const persist = () => writeFile(output, JSON.stringify(receipt, null, 2));
const checkpoint = async event => { receipt.events.push({ ...event, at: new Date().toISOString() }); await persist(); };
const bounded = async (promise, ms, label) => { let timer; try { return await Promise.race([promise,
 new Promise((_, reject) => { timer = setTimeout(() => reject(Error(label + ' deadline')), ms); })]); } finally { clearTimeout(timer); } };
const load = path => import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu/src', path)));
const descriptor = JSON.parse(await readFile(resolve(root, 'native-delivery.json')));
for (const entry of descriptor.entries) {
 const bytes = await readFile(resolve(root, entry.path)); assert.equal(bytes.length, entry.bytes); assert.equal(sha(bytes), entry.sha256);
}
receipt.nativeFilesVerified = descriptor.entries.length;
receipt.hostSourceSha256 = sha(await readFile(resolve(root, 'server/native-participant.js')));
receipt.runtimeArchiveSha256 = sha(await readFile(resolve(root, 'deploy/artifacts/doppler-gpu-0.6.28.tgz')));
assert.equal(receipt.runtimeArchiveSha256, '5e0d60bc433d332187d422721fd7afcaec780c202e3082adcab389b47fd27fc5');
const { createNativeGpuHost } = await import(pathToFileURL(resolve(root, 'server/native-participant.js')));
const host = createNativeGpuHost(() => import(pathToFileURL(resolve(root, 'node_modules/webgpu/index.js'))));
const navigatorBefore = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
let device = null, lostObserved = false;
const owned = new Set(), pooled = new Set();
try {
 receipt.result = await bounded((async () => {
    await checkpoint({ phase: 'native-host-open-start' }); await host.open(); await checkpoint({ phase: 'native-host-open-complete' });
    const { setRuntimeConfig } = await load('config/runtime.js');
    setRuntimeConfig({ shared: { bufferPool: { budget: { maxTotalBytes: 16777216, hardFailOnBudgetExceeded: true } } } });
    const { setDevice } = await load('gpu/device.js');
    const { createTensor } = await load('gpu/tensor.js');
    const { CommandRecorder } = await load('gpu/command-recorder.js');
    const { expandF32ToPaired, recordExpandF32ToPaired, castPairedToF16, recordCastPairedToF16 } = await load('gpu/kernels/paired-cast.js');
    const { readBufferSlice, releaseBuffer, getBufferPool } = await load('memory/buffer-pool.js');
    const adapter = await navigator.gpu.requestAdapter(); assert(adapter && !adapter.info.isFallbackAdapter);
    device = await adapter.requestDevice(); setDevice(device, { platformConfig: null });
    device.lost.then(info => { lostObserved = true; receipt.deviceLost = { reason: info.reason, message: info.message };
      receipt.events.push({ phase: 'owned-device-lost-observed', at: new Date().toISOString() }); });
    const runtimePackage = JSON.parse(await readFile(resolve(root, 'node_modules/doppler-gpu/package.json')));
    const tooling = await import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu', runtimePackage.exports['./tooling'].import)));
    assert.equal(tooling.getDevice(), device, 'Actual host and fixtures must share the same runtime device owner');
    const errors = [], rows = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const make = (bytes, upload) => { const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      owned.add(buffer); if (upload) device.queue.writeBuffer(buffer, 0, upload); return buffer; };
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

 })(), 300000, 'paired cast operation');
} catch (error) { receipt.error = { name: error.name, message: error.message, stack: error.stack }; process.exitCode = 1; }
finally {
 await checkpoint({ phase: 'actual-native-host-close-start' });
 try {
   await host.close();
   assert(!device || lostObserved, 'Host returned before owned device loss observation');
   assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis, 'navigator'), navigatorBefore);
   receipt.hostClosed = true;
   await checkpoint({ phase: 'actual-native-host-close-complete' });
   for (const buffer of owned) buffer.destroy();
 } catch (error) { receipt.cleanupError = { name: error.name, message: error.message }; receipt.ownership = 'unknown'; process.exitCode = 1; }
 receipt.passed = !receipt.error && !receipt.cleanupError && receipt.hostClosed && receipt.result?.rows.length === 4;
 receipt.finishedAt = new Date().toISOString(); await persist();
}
