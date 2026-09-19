import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/generator/token-selection.js', import.meta.url).href;
globalThis.GPUBufferUsage ??= { COPY_DST: 8, MAP_READ: 1 };
globalThis.GPUMapMode ??= { READ: 1 };
const createDevice = () => ({ createBuffer: () => ({
  async mapAsync() { state.bytes = await state.read(); }, getMappedRange: () => state.bytes,
  unmap() { state.unmapped++; }, destroy() { state.destroyed++; },
}) });
const state = { device: createDevice(), released: [], submitted: 0, aborted: 0, output: {} };
globalThis.tokenSelectionLifecycle = state;
const modules = {
  '../../../../gpu/command-recorder.js': `export const createCommandRecorder = () => ({ device: globalThis.tokenSelectionLifecycle.device,
    getEncoder: () => ({ copyBufferToBuffer() {} }), submit() { globalThis.tokenSelectionLifecycle.submitted++; }, abort() { globalThis.tokenSelectionLifecycle.aborted++; } });`,
  '../../../../gpu/kernels/sample.js': `export const recordArgmax = async () => { await globalThis.tokenSelectionLifecycle.compile(); return globalThis.tokenSelectionLifecycle.output; };
    export const recordGPUSample = recordArgmax;`,
  '../../../../gpu/kernels/rep-penalty.js': 'export const recordHistoryPenalties = async () => {};',
  '../../../../gpu/kernels/logit-suppress.js': 'export const recordSuppressLogits = async () => {};',
  '../../../../memory/buffer-pool.js': `export const releaseBuffer = buffer => globalThis.tokenSelectionLifecycle.released.push(buffer);
`,
  '../../../../gpu/device.js': 'export const getDevice = () => globalThis.tokenSelectionLifecycle.device;',
  '../../../../gpu/device-state.js': 'export const isDeviceLost = () => globalThis.tokenSelectionLifecycle.lost === true;',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { selectTokenFromGpuLogits } = await import(target);
  const result = { logitsBuffer: { size: 16 }, logitsDtype: 'f32', vocabSize: 4 };
  for (const failure of ['compile-abort', 'compile-loss', 'compile-replace', 'compile-error', 'read-abort', 'read-error', 'success']) {
    const abort = new AbortController();
    Object.assign(state, { device: createDevice(), lost: false, unmapped: 0, destroyed: 0, released: [], submitted: 0, aborted: 0,
      compile: async () => {
        await Promise.resolve();
        if (failure === 'compile-abort') abort.abort();
        if (failure === 'compile-loss') state.lost = true;
        if (failure === 'compile-replace') state.device = {};
        if (failure === 'compile-error') throw Error('compile failed');
      }, read: async () => {
        await Promise.resolve();
        if (failure === 'read-abort') abort.abort();
        if (failure === 'read-error') throw Error('read failed');
        return Uint32Array.of(2).buffer;
      } });
    const pending = selectTokenFromGpuLogits(result, [], { temperature: 0, signal: abort.signal }, { padTokenId: null });
    if (failure === 'success') assert.equal((await pending).tokenId, 2);
    else await assert.rejects(pending, /aborted|lost or replaced|failed/);
    assert.equal(state.submitted, failure.startsWith('compile') ? 0 : 1, failure);
    assert.equal(state.aborted, 1, failure);
    assert.equal(state.destroyed, failure.startsWith('compile') ? 0 : 1, failure);
    assert.equal(state.unmapped, failure.startsWith('compile') || failure === 'read-error' ? 0 : 1, failure);
    assert.deepEqual(state.released, failure === 'compile-error' ? [] : [state.output], failure);
  }
  const abort = new AbortController(); abort.abort();
  state.aborted = 0;
  await assert.rejects(selectTokenFromGpuLogits(result, [], { signal: abort.signal }, {}), /aborted/);
  assert.equal(state.aborted, 0, 'already cancelled requests create no recorder');
} finally { hooks.deregister(); delete globalThis.tokenSelectionLifecycle; }
console.log('token-selection-lifecycle: passed');
