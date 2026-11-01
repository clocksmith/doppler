import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/loader/weight-downcast.js', import.meta.url).href;
const live = new Set();
const originalGPUBuffer = globalThis.GPUBuffer;
globalThis.GPUBuffer = class { constructor() { this.size = 16; } };
let failure = '';
globalThis.downcastFixture = {
  cast() { if (failure === 'cast') throw Error('injected cast'); const buffer = {}; live.add(buffer); return { buffer }; },
  release(buffer) { assert(live.delete(buffer), 'release once'); },
};
const modules = {
  '../gpu/device.js': 'export const getKernelCapabilities=()=>({hasF16:true});',
  '../gpu/kernels/cast.js': 'export const castF32ToF16=async()=>downcastFixture.cast();',
  '../memory/buffer-pool.js': 'export const releaseBuffer=b=>downcastFixture.release(b);',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { maybeDowncastToF16 } = await import(target);
  for (const raw of [false, true]) {
  for (failure of ['cast', 'wrap', '']) {
    const input = new GPUBuffer(); live.add(input);
    const weight = raw ? input : { buffer: input, dtype: 'f32', layout: 'row', shape: [2, 2] };
    const options = { label: 'test', keepF32: false, dtype: 'f32', ...(failure === 'wrap' ? { shape: 7 } : {}) };
    if (failure) {
      await assert.rejects(maybeDowncastToF16(weight, options), /Failed to downcast test to f16/);
      assert.deepEqual([...live], [input], 'input retained, temporary output released');
      live.delete(input);
    } else {
      const result = await maybeDowncastToF16(weight, options);
      assert.equal(result.wasDowncast, true);
      assert.equal(result.buffer.dtype, 'f16');
      assert.deepEqual([...live], [result.newBuffer]);
      downcastFixture.release(result.newBuffer);
    }
  }
  }
} finally { hooks.deregister(); delete globalThis.downcastFixture; globalThis.GPUBuffer = originalGPUBuffer; }
console.log('weight downcast: cast/wrapping failures propagate, temporary cleanup and successful transfer passed');
