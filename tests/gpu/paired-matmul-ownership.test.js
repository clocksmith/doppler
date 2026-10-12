import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';
const target = new URL('../../src/gpu/kernels/paired-matmul.js', import.meta.url).href;
const live = new Set(), calls = [];
globalThis.pairedMatmulControl = {
  fail: false,
  acquire(size) { const b = { size }; live.add(b); return b; },
  release(b) { assert(live.delete(b)); },
  dispatch(...args) { if (this.fail) throw Error('injected dispatch failure'); calls.push(args); },
};
const modules = {
  '../../memory/buffer-pool.js': 'export const acquireBuffer=size=>pairedMatmulControl.acquire(size); export const releaseBuffer=b=>pairedMatmulControl.release(b);',
  './kernel-execution.js': 'export const unifiedKernelWrapper=async(...args)=>pairedMatmulControl.dispatch(...args);',
  '../device.js': 'export const getDevice=()=>({limits:{maxComputeWorkgroupsPerDimension:65535}});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
const { executePairedMatmul } = await import(target);
const input = createTensor({ size: 2048 }, 'f32', [1, 256], 'input', { precision: 'two-f32', components: 2, elementStrideBytes: 8 });
const weight = createTensor({ size: 3072 }, 'f32', [3, 256], 'weight');
try {
  for (const recorded of [false, true]) {
    const held = new Set(), recorder = recorded ? { device: { limits: { maxComputeWorkgroupsPerDimension: 65535 } }, trackTemporaryBuffer: b => held.add(b) } : null;
    const settle = () => { for (const b of held) pairedMatmulControl.release(b); held.clear(); };
    const run = (config = {}) => executePairedMatmul(recorder, input, weight, 1, 3, 256, { outputDtype: 'f32', transposeB: true, ...config });
    pairedMatmulControl.fail = true; await assert.rejects(run(), /injected/); settle(); assert.equal(live.size, 0);
    pairedMatmulControl.fail = false;
    for (const outputPrecision of ['two-f32', 'f32']) {
      const result = await run({ outputPrecision, alpha: 1.1 });
      assert.equal(result.buffer.size, outputPrecision === 'f32' ? 12 : 24);
      assert.equal(result.precision, outputPrecision === 'f32' ? undefined : 'two-f32');
      const uniforms = calls.at(-1)[4]; assert.equal(uniforms.alpha_lo, Math.fround(1.1 - Math.fround(1.1)));
      pairedMatmulControl.release(result.buffer);
    }
    for (const config of [{ aOffset: 4 }, { bOffset: 4 }, { outputBuffer: input.buffer }, { outputBuffer: { size: 8 } }, { outputPrecision: 'unknown' }, { alpha: Infinity }]) {
      const count = calls.length; await assert.rejects(run(config)); assert.equal(calls.length, count); assert.equal(live.size, 0);
    }
    await assert.rejects(executePairedMatmul(recorder, { ...input, precision: undefined }, weight, 1, 3, 256, { outputDtype: 'f32' }), /Interleaved/);
    await assert.rejects(executePairedMatmul(recorder, input, weight, NaN, 3, 256, { outputDtype: 'f32' }), /geometry/);
  }
} finally { hooks.deregister(); delete globalThis.pairedMatmulControl; }
console.log('paired matmul: explicit representation/output, source alpha, invalid geometry and immediate/recorded ownership passed');
