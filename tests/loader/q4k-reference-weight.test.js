import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Buffer-pool and weight-buffer ports are replaced; the real dequantization,
// allocation/upload/adoption transaction and layout handling run.
const target = new URL('../../src/loader/q4k-reference-weight.js', import.meta.url).href;
const live = new Set();
const port = {
  failUpload: false,
  acquire(size, usage, label) { const buffer = { size, label }; live.add(buffer); return buffer; },
  release(buffer) { assert(live.delete(buffer), 'release of an unknown buffer'); },
  upload(buffer, data) { if (this.failUpload) throw new Error('upload rejected'); buffer.bytes = data.byteLength; },
};
globalThis.q4kReferencePort = port;
const modules = {
  '../memory/buffer-pool.js': `export const acquireBuffer = (size, usage, label) => q4kReferencePort.acquire(size, usage, label);
    export const releaseBuffer = (buffer) => q4kReferencePort.release(buffer);
    export const uploadData = (buffer, data) => q4kReferencePort.upload(buffer, data);`,
  '../gpu/weight-buffer.js': 'export const createWeightBuffer = (buffer, dtype, layout, shape, name) => ({ buffer, dtype, layout, shape, name });',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
const { loadReferenceQ4KWeight, acquireUploadedBuffer, isQ4KLocationDtype, toUint8Array } = await import(target);
const { QK_K, Q4K_BLOCK_BYTES } = await import('../../src/config/schema/quantization.schema.js');

const rows = 2, cols = QK_K;
const blocks = Math.ceil((rows * cols) / QK_K);
const quantized = new Uint8Array(blocks * Q4K_BLOCK_BYTES);
const location = { dtype: 'Q4_K_M', shape: [rows, cols], size: quantized.byteLength };
const owned = new Set();
const ctx = { gpuBuffers: owned, loadTensor: async () => quantized, resolveWeightLayout: () => 'col' };

try {
  // Success: dequantized f32 bytes uploaded, buffer adopted, caller's layout applied once.
  let layoutCalls = 0;
  const weight = await loadReferenceQ4KWeight(ctx, { name: 'w', location, layout: () => { layoutCalls += 1; return ctx.resolveWeightLayout(location); } });
  assert.equal(weight.dtype, 'f32');
  assert.equal(weight.layout, 'col');
  assert.deepEqual(weight.shape, [rows, cols]);
  assert.equal(weight.buffer.label, 'w_f32_reference');
  assert.equal(weight.buffer.bytes, rows * cols * 4);
  assert.equal(layoutCalls, 1);
  assert.ok(owned.has(weight.buffer), 'an uploaded buffer joins the owner set');
  const explicit = await loadReferenceQ4KWeight(ctx, { name: 'v', location, layout: 'row' });
  assert.equal(explicit.layout, 'row');

  // The defect: a failed upload used to leak the acquired buffer.
  port.failUpload = true;
  const before = live.size;
  await assert.rejects(() => loadReferenceQ4KWeight(ctx, { name: 'leak', location, layout: 'row' }), /upload rejected/);
  assert.equal(live.size, before, 'the buffer acquired for a failed upload is released');
  assert.ok(![...owned].some((buffer) => buffer.label === 'leak_f32_reference'), 'a failed upload is never adopted');
  port.failUpload = false;

  // Non-Q4K, non-2D and unavailable tensors return null without allocating.
  const liveBefore = live.size;
  assert.equal(await loadReferenceQ4KWeight(ctx, { name: 'f16', location: { ...location, dtype: 'F16' }, layout: 'row' }), null);
  assert.equal(await loadReferenceQ4KWeight(ctx, { name: 'vec', location: { ...location, shape: [cols] }, layout: 'row' }), null);
  assert.equal(await loadReferenceQ4KWeight({ ...ctx, loadTensor: async () => null }, { name: 'none', location, layout: 'row' }), null);
  assert.equal(live.size, liveBefore);

  // The transaction helper and its small predicates.
  const adopted = new Set();
  const buffer = acquireUploadedBuffer(new Float32Array(4), 'direct', adopted);
  assert.ok(adopted.has(buffer));
  assert.equal(acquireUploadedBuffer(new Float32Array(2), 'unowned').label, 'unowned');
  assert.equal(isQ4KLocationDtype('Q4_K'), true);
  assert.equal(isQ4KLocationDtype('Q6_K'), false);
  assert.equal(toUint8Array(new Float32Array([1])).byteLength, 4);
  assert.equal(toUint8Array('nope'), null);
  console.log('q4k-reference-weight: shared allocation/upload/adoption transaction and layout handling verified');
} finally {
  hooks.deregister();
  delete globalThis.q4kReferencePort;
}
