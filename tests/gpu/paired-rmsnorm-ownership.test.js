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
    // Unpinned calls dispatch the shipped program, one 64-lane workgroup per token.
    pairedNormControl.release((await run()).buffer); settle();
    assert.deepEqual([calls.at(-1)[2], calls.at(-1)[5], calls.at(-1)[6].WORKGROUP_SIZE], ['paired', [1, 1, 1], 64]);
    // A declared sliced program adds one workgroup per 256-element output slice; its width is a registry override.
    const declare = kernel => ({ activationDtype: 'f32', prefill: { steps: [{ op: 'input_norm', kernel, entry: 'main' }] },
      decode: { steps: [{ op: 'input_norm', kernel, entry: 'main' }] } });
    const pinned = { role: 'input_norm', section: 'layer', phase: 'decode', layerIdx: 0 };
    pairedNormControl.release((await run({ ...pinned, kernelPath: declare('rmsnorm_paired_sliced.wgsl') })).buffer); settle();
    assert.deepEqual([calls.at(-1)[2], calls.at(-1)[5], calls.at(-1)[6].WORKGROUP_SIZE], ['paired_sliced', [1, 1, 4], undefined]);
    const before = calls.length;
    await assert.rejects(run({ ...pinned, kernelPath: declare('rmsnorm.wgsl') }), /RMSNorm/);
    assert.equal(calls.length, before); assert.equal(live.size, 0);
    for (const options of [{ batchSize: NaN }, { hiddenSize: 0 }, { outputBuffer: input.buffer },
      { outputBuffer: { size: 4096 } }, { residualSumOutput: { size: 8192 } },
      { preResidual: input, residual: input }, { residual: { ...input, precision: undefined } }]) {
      const before = calls.length;
      await assert.rejects(run(options)); assert.equal(calls.length, before); assert.equal(live.size, 0);
    }
  }
  // Q/K head normalization (head 256, 8 query and 2 key heads per token) dispatches the programs the
  // graph's q_norm and k_norm steps declare for the phase and layer. A graph declaring neither step
  // runs the interim rmsnorm/paired program; a missing step, a scalar program or no graph dispatches
  // nothing and leaves nothing allocated.
  const { runRMSNormQK, recordRMSNormQK } = await import('../../src/gpu/kernels/rmsnorm-qk.js');
  const q = createTensor({ size: 8 * 256 * 8 }, 'f32', [8, 256], 'q', representation);
  const k = createTensor({ size: 2 * 256 * 8 }, 'f32', [2, 256], 'k', representation);
  const headWeight = createTensor({ size: 1024 }, 'f32', [256], 'head_weight');
  const graph = (qKernel, kKernel) => {
    const steps = [{ op: 'input_norm', kernel: 'rmsnorm_paired.wgsl', entry: 'main' },
      ...(qKernel ? [{ op: 'q_norm', kernel: qKernel, entry: 'main' }] : []),
      ...(kKernel ? [{ op: 'k_norm', kernel: kKernel, entry: 'main' }] : [])];
    return { activationDtype: 'f32', prefill: { steps }, decode: { steps } };
  };
  const recorder = { device: { limits: { maxComputeWorkgroupsPerDimension: 65535 } }, trackTemporaryBuffer: b => pairedNormControl.release(b) };
  const heads = kernelPath => [q, k, headWeight, headWeight, 1e-6,
    { numTokens: 1, numHeads: 8, numKVHeads: 2, headDim: 256, rmsNormWeightOffset: true, kernelPath, phase: 'decode', layerIdx: 3 }];
  for (const [kernelPath, expected] of [
    [graph('rmsnorm_paired_sliced.wgsl', 'rmsnorm_paired.wgsl'), [['paired_sliced', [8, 1, 1], undefined], ['paired', [2, 1, 1], 64]]],
    [graph('rmsnorm_paired_sliced.wgsl', 'rmsnorm_paired_sliced.wgsl'), [['paired_sliced', [8, 1, 1], undefined], ['paired_sliced', [2, 1, 1], undefined]]],
    [graph(null, null), [['paired', [8, 1, 1], 64], ['paired', [2, 1, 1], 64]]],
  ]) {
    for (const run of [(...args) => runRMSNormQK(...args), (...args) => recordRMSNormQK(recorder, ...args)]) {
      const before = calls.length;
      const result = await run(...heads(kernelPath));
      assert.equal(calls.length, before + 2);
      assert.deepEqual(calls.slice(-2).map(call => [call[2], call[5], call[6].WORKGROUP_SIZE]), expected);
      assert.deepEqual(calls.slice(-2).map(call => call[6].WEIGHT_OFFSET), [true, true]);
      pairedNormControl.release(result.q.buffer); pairedNormControl.release(result.k.buffer);
      assert.equal(live.size, 0);
    }
  }
  for (const [kernelPath, failure] of [[graph('rmsnorm_paired_sliced.wgsl', null), /k_norm requires exactly one declared step/],
    [graph('rmsnorm.wgsl', 'rmsnorm.wgsl'), /not a declared paired RMSNorm program/], [null, /runs only from the execution graph/]]) {
    const before = calls.length;
    await assert.rejects(runRMSNormQK(...heads(kernelPath)), failure);
    assert.equal(calls.length, before); assert.equal(live.size, 0);
  }
} finally { hooks.deregister(); delete globalThis.pairedNormControl; }
console.log('paired RMSNorm: source epsilon, immediate/recorded ownership, allocation/dispatch failure, declared sliced program geometry, geometry rejection and graph-declared Q/K head-norm programs passed');
