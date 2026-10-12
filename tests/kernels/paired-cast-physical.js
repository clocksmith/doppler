// Actual public wrapper dispatch; no model or runtime policy changes.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const destination = process.argv[2];
if (!destination) throw new Error('Provide receipt destination.');
await mkdir(dirname(destination), { recursive: true });
const root = resolve(fileURLToPath(new URL('../../', import.meta.url)));
const receipt = { schema: 'doppler-paired-cast-wrapper-physical-v1', sources: {},
  scope: 'Immediate and recorded public wrappers, exact expansion bits, direct half midpoint residuals, odd count padding, caller supplied outputs and settled resources; no whole-model qualification.' };
for (const path of ['src/gpu/tensor.js', 'src/gpu/kernels/paired-cast.js', 'src/gpu/kernels/cast_f32_to_two_f32.wgsl', 'src/gpu/kernels/cast_two_f32_to_f16.wgsl', 'src/config/kernels/registry.json']) {
  receipt.sources[path] = createHash('sha256').update(await readFile(resolve(root, path))).digest('hex');
}
const server = createServer(async (req, res) => {
  try {
    const path = resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
    if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    if (path === root) { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html>'); return; }
    res.setHeader('Content-Type', extname(path) === '.js' ? 'text/javascript' : extname(path) === '.json' ? 'application/json' : 'text/plain');
    res.end(await readFile(path));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}/`);
  receipt.result = await page.evaluate(async () => {
    const { setDevice, destroyDevice } = await import('/src/gpu/device.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { expandF32ToPaired, recordExpandF32ToPaired, castPairedToF16, recordCastPairedToF16 } = await import('/src/gpu/kernels/paired-cast.js');
    const { readBufferSlice, releaseBuffer, destroyBufferPool, getBufferPool } = await import('/src/memory/buffer-pool.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const device = await adapter.requestDevice(); setDevice(device, { platformConfig: null });
    const errors = [], rows = [], owned = new Set(), pooled = new Set();
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const make = (bytes, upload) => {
      const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      owned.add(buffer); if (upload) device.queue.writeBuffer(buffer, 0, upload); return buffer;
    };
    try {
      const bits = new Uint32Array([0, 0x80000000, 1, 0x80000001, 0x3f800000, 0x7f7fffff, 0x3f801000]);
      for (const recorded of [false, true]) for (const borrowed of [false, true]) {
        const input = createTensor(make(bits.byteLength, bits), 'f32', [7], 'exact_scalar_bits');
        const options = borrowed ? { outputBuffer: make(bits.length * 8) } : {};
        const recorder = recorded ? new CommandRecorder(device, 'paired_cast_control') : null;
        let output;
        try {
          output = recorded ? await recordExpandF32ToPaired(recorder, input, options) : await expandF32ToPaired(input, options);
          if (!borrowed) pooled.add(output.buffer);
          if (recorder) await recorder.submitAndWait();
          const actual = Array.from(new Uint32Array(await readBufferSlice(output.buffer, 0, bits.length * 8)));
          const expected = Array.from(bits).flatMap(word => [word, 0]);
          if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error('Expansion changed source bits.');
          if (output.precision !== 'two-f32' || output.components !== 2 || output.elementStrideBytes !== 8) throw new Error('Paired output metadata missing.');
          rows.push({ operation: 'expand', recorded, borrowed, actual });
        } finally { if (recorder) await recorder.abort(); }
      }
      // Halfway 1+2^-11 plus/minus minimum F32 subnormal; exact signs break the tie.
      const pairWords = new Uint32Array([0x3f801000, 1, 0x3f801000, 0x80000001,
        0xbf801000, 1, 0xbf801000, 0x80000001, 0x80000000, 0]);
      const expected = [0x3c01, 0x3c00, 0xbc00, 0xbc01, 0x8000, 0];
      for (const recorded of [false, true]) for (const borrowed of [false, true]) {
        const input = createTensor(make(pairWords.byteLength, pairWords), 'f32', [5], 'half_midpoints',
          { precision: 'two-f32', components: 2, elementStrideBytes: 8 });
        const options = borrowed ? { outputBuffer: make(12) } : {};
        const recorder = recorded ? new CommandRecorder(device, 'paired_half_control') : null;
        try {
          const output = recorded ? await recordCastPairedToF16(recorder, input, options) : await castPairedToF16(input, options);
          if (!borrowed) pooled.add(output.buffer);
          if (recorder) await recorder.submitAndWait();
          const actual = Array.from(new Uint16Array(await readBufferSlice(output.buffer, 0, 12)));
          if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`Half bits ${actual} != ${expected}`);
          if (output.dtype !== 'f16' || output.precision !== undefined) throw new Error('Half output contract wrong.');
          rows.push({ operation: 'half', recorded, borrowed, actual });
        } finally { if (recorder) await recorder.abort(); }
      }
      await device.queue.onSubmittedWorkDone();
      for (const buffer of pooled) releaseBuffer(buffer); pooled.clear();
      const stats = getBufferPool().getStats();
      if (stats.activeBuffers !== 0 || errors.length) throw new Error(JSON.stringify({ stats, errors }));
      return { rows, errors, stats, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture }, settled: true };
    } finally {
      await device.queue.onSubmittedWorkDone();
      for (const buffer of pooled) releaseBuffer(buffer);
      for (const buffer of owned) buffer.destroy();
      destroyBufferPool(); destroyDevice();
    }
  });
  receipt.browser = browser.version(); receipt.passed = receipt.result.rows.length === 8 && receipt.result.settled;
} catch (error) { receipt.error = error.stack; throw error; }
finally {
  await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n');
}
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired cast wrappers failed.');
