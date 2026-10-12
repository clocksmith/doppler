import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';
const target = new URL('../../src/gpu/kernels/paired-rmsnorm.js', import.meta.url).href;
const live = new Set(), calls = [];
globalThis.pairedNormControl = {
  failure: null,
  acquire(size, label) {
    if (this.failure === 'placeholder' && label.includes('placeholder')) throw Error('injected allocation failure');
    const buffer = { size }; live.add(buffer); return buffer;
  },
  release(buffer) { assert(live.delete(buffer)); },
  dispatch(...args) { if (this.failure === 'dispatch') throw Error('injected dispatch failure'); calls.push(args); },
};
const modules = {
  '../../memory/buffer-pool.js': `export const acquireBuffer=(size,usage,label)=>pairedNormControl.acquire(size,label); export const releaseBuffer=b=>pairedNormControl.release(b);`,
  './kernel-execution.js': 'export const unifiedKernelWrapper=async(...args)=>pairedNormControl.dispatch(...args);',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
const { runRMSNorm, recordRMSNorm } = await import('../../src/gpu/kernels/rmsnorm.js');
const representation = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
const input = createTensor({ size: 8192 }, 'f32', [1, 1024], 'input', representation);
const weight = createTensor({ size: 4096 }, 'f32', [1024], 'weight');
try {
  for (const recorded of [false, true]) {
    const held = new Set();
    const recorder = recorded ? { device: { limits: { maxComputeWorkgroupsPerDimension: 65535 } },
      trackTemporaryBuffer: b => held.add(b) } : null;
    const settle = () => { for (const b of held) pairedNormControl.release(b); held.clear(); };
    const run = (options = {}) => recorded
      ? recordRMSNorm(recorder, input, weight, 1e-6, { batchSize: 1, hiddenSize: 1024, ...options })
      : runRMSNorm(input, weight, 1e-6, { batchSize: 1, hiddenSize: 1024, ...options });
    for (const failure of ['placeholder', 'dispatch']) {
      pairedNormControl.failure = failure;
      await assert.rejects(run(), /injected/); settle();
      assert.equal(live.size, 0, `${failure} must release every owned buffer`);
    }
    pairedNormControl.failure = null;
    const output = await run({ preResidual: input, residualSumOutput: { size: 8192 } });
    const uniforms = calls.at(-1)[4];
    assert.equal(uniforms.eps_hi, Math.fround(1e-6));
    assert.equal(uniforms.eps_lo, Math.fround(1e-6 - Math.fround(1e-6)));
    assert.equal(output.precision, 'two-f32'); assert.equal(output.buffer.size, 8192);
    pairedNormControl.release(output.buffer); settle();
    for (const options of [{ batchSize: NaN }, { hiddenSize: 0 }, { outputBuffer: input.buffer },
      { outputBuffer: { size: 4096 } }, { residualSumOutput: { size: 8192 } },
      { preResidual: input, residual: input }, { residual: { ...input, precision: undefined } }]) {
      const before = calls.length;
      await assert.rejects(run(options)); assert.equal(calls.length, before); assert.equal(live.size, 0);
    }
  }
} finally { hooks.deregister(); delete globalThis.pairedNormControl; }
console.log('paired RMSNorm: source epsilon, immediate/recorded ownership, allocation/dispatch failure and geometry rejection passed');
