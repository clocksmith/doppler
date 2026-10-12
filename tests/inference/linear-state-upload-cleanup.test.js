import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/linear-attention/plan.js', import.meta.url).href;
const live = new Set();
let calls = 0, failAt = -1, failure = 'allocate';
globalThis.linearUploadControl = {
  acquire(size) { if (failure === 'allocate' && calls++ === failAt) throw Error('injected allocation'); const b = { size, __dopplerFakeGPUBuffer: true }; live.add(b); return b; },
  upload() { if (failure === 'upload' && calls++ === failAt) throw Error('injected upload'); },
  release(b) { assert(live.delete(b)); },
};
const source = 'export const acquireBuffer=(...a)=>linearUploadControl.acquire(...a);export const uploadData=(...a)=>linearUploadControl.upload(...a);export const releaseBuffer=b=>linearUploadControl.release(b);export const readBuffer=()=>{throw Error("unexpected read");};';
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../../../../memory/buffer-pool.js') return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { ensureLayerRuntimeGpuBuffers } = await import(target);
  const fields = ['convWeight', 'dtBias', 'aLog', 'normWeight', 'convState', 'recurrentState'];
  for (failure of ['allocate', 'upload']) for (failAt of [0, 1, 2, 3, 4]) {
    calls = 0;
    const borrowed = { size: 4, __dopplerFakeGPUBuffer: true };
    const state = Object.fromEntries(fields.map(name => [name, new Float32Array(2)])); state.convWeightGPU = borrowed;
    assert.throws(() => ensureLayerRuntimeGpuBuffers(state), /injected/);
    assert.equal(state.convWeightGPU, borrowed, 'borrowed resident weight survives failed preparation');
    assert.equal(live.size, 0, 'all newly prepared resources settle on any partial upload failure');
  }
} finally { hooks.deregister(); delete globalThis.linearUploadControl; }
console.log('linear state upload: allocation/upload rollback preserves resident ownership');
