import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';

const target = new URL('../../src/inference/pipelines/text/ffn/paired.js', import.meta.url).href;
const live = new Set(), held = new Set(), calls = [];
const pair = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
const input = createTensor({ size: 2 * 256 * 8 }, 'f32', [2, 256], 'input', pair);
const weight = { buffer: { size: 256 * 512 * 4 }, dtype: 'f32', shape: [512, 256], layout: 'row' };
const weights = { gate: weight, up: weight, down: weight };
const makeTensor = (shape) => {
  const buffer = { size: shape.reduce((a, b) => a * b, 1) * 8 };
  live.add(buffer);
  return createTensor(buffer, 'f32', shape, 'owned', pair);
};
globalThis.pairedFfnTest = {
  fail: null,
  release(buffer) { assert(live.delete(buffer), 'must release owned buffer once'); },
  async matmul(tensor, weight, M, N, K, options) {
    calls.push(options.role);
    if (this.fail === options.role) throw Error('injected ' + options.role);
    assert.equal(tensor.precision, 'two-f32');
    assert.equal(options.outputPrecision, 'two-f32');
    assert.equal(options.phaseOverride, 'prefill');
    assert.deepEqual(tensor.shape, [M, K]);
    return makeTensor([M, N]);
  },
  async silu(up, options) {
    calls.push('silu');
    if (this.fail === 'silu') throw Error('injected silu');
    assert.equal(options.gate.precision, 'two-f32');
    assert.equal(options.inputActivation, 'identity');
    assert.equal(options.gateActivation, 'silu');
    assert.equal(options.swigluLimit, null);
    assert.equal(options.size, 1024);
    return makeTensor(up.shape);
  },
  async probe(stage, buffer, options) {
    if (this.fail === stage + '_probe') throw Error('injected probe');
    assert.equal(options.precision, 'two-f32');
    assert.equal(options.components, 2);
    assert.equal(options.elementStrideBytes, 8);
  },
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../ops.js') {
    return { url: 'data:text/javascript,' + encodeURIComponent(`
      export const doMatmul=(...args)=>pairedFfnTest.matmul(...args);
      export const doSiLU=(...args)=>pairedFfnTest.silu(...args);
      export const releaseOrTrack=(recorder,buffer)=>recorder ? recorder.trackTemporaryBuffer(buffer) : pairedFfnTest.release(buffer);
    `), shortCircuit: true };
  }
  if (context.parentURL === target && specifier === '../weights.js') {
    return { url: 'data:text/javascript,export const getWeightBuffer=value=>value;', shortCircuit: true };
  }
  if (context.parentURL === target && specifier === '../probes.js') {
    return { url: 'data:text/javascript,export const runProbes=(...args)=>pairedFfnTest.probe(...args);', shortCircuit: true };
  }
  return next(specifier, context);
} });
const { runPairedFFNGPU, validatePairedFFN } = await import(target);
const { runDenseFFNGPU } = await import('../../src/inference/pipelines/text/ffn/dense.js');
const projection = op => ({ op, kernel: 'matmul_paired.wgsl', entry: 'main',
  constants: { WEIGHT_F16: false, WEIGHT_Q4K: false, SCALAR_OUTPUT: false } });
const steps = [projection('gate_proj'), projection('up_proj'),
  { op: 'activation', kernel: 'paired_silu.wgsl', entry: 'main' }, projection('down_proj')];
const base = { config: { hiddenSize: 256, intermediateSize: 512, hiddenActivation: 'silu', swigluLimit: null,
  gatedActivation: true, useDoubleWideMlp: false }, phase: 'prefill', kernelPath: { prefill: { steps }, decode: { steps } } };
const settle = () => { for (const buffer of held) pairedFfnTest.release(buffer); held.clear(); };
try {
  for (const recorded of [false, true]) {
    const recorder = recorded ? { trackTemporaryBuffer: buffer => { assert(!held.has(buffer)); held.add(buffer); } } : null;
    const context = { ...base, recorder };
    for (const failure of ['ffn_gate', 'ffn_up', 'silu', 'ffn_down', 'capture', 'ffn_gate_probe', 'ffn_up_probe', 'ffn_act_probe']) {
      pairedFfnTest.fail = failure;
      const failingContext = failure === 'capture' ? { ...context,
        trainingCapture: { layerIdx: 0, stage: 'ffn_act', capture: () => { throw Error('injected capture'); } } } : context;
      await assert.rejects(runPairedFFNGPU(0, input, 2, failingContext, weights), /injected/);
      settle(); assert.equal(live.size, 0); assert.equal(input.buffer.size, 4096);
    }
    pairedFfnTest.fail = null; calls.length = 0;
    const output = await runPairedFFNGPU(0, input, 2, context, weights);
    assert.deepEqual(calls, ['ffn_gate', 'ffn_up', 'silu', 'ffn_down']);
    assert.deepEqual(output.shape, [2, 256]); assert.equal(output.elementStrideBytes, 8);
    settle(); assert.deepEqual([...live], [output.buffer]); pairedFfnTest.release(output.buffer);
  }
  const rejects = (context = base, tensor = input, supplied = weights) => {
    const count = calls.length;
    assert.throws(() => validatePairedFFN(0, tensor, 2, context, supplied), /PairedFFN/);
    assert.equal(calls.length, count); assert.equal(live.size, 0);
  };
  pairedFfnTest.fail = null; calls.length = 0;
  const throughPublicDense = await runDenseFFNGPU(0, input, 2, base, weights);
  assert.deepEqual(calls, ['ffn_gate', 'ffn_up', 'silu', 'ffn_down']);
  pairedFfnTest.release(throughPublicDense.buffer);
  rejects(base, { ...input, components: 1 });
  rejects(base, { ...input, shape: [512] });
  rejects(base, { ...input, buffer: { size: 2048 } });
  rejects({ ...base, config: { ...base.config, swigluLimit: 7 } });
  rejects({ ...base, config: { ...base.config, hiddenActivation: 'gelu' } });
  rejects({ ...base, lora: {} });
  rejects(base, input, { ...weights, gateUp: weight });
  rejects(base, input, { ...weights, down: null });
  for (const badSteps of [steps.slice(1), [steps[1], steps[0], ...steps.slice(2)],
    [steps[0], steps[1], { op: 'ffn', kernel: 'fused_ffn_q4k.wgsl', entry: 'main' }, steps[3]],
    [...steps, projection('gate_proj')],
    steps.map(step => step.op === 'up_proj' ? { ...step, precision: { outputDtype: 'f16' } } : step),
    steps.map(step => step.op === 'activation' ? { ...step, constants: { HAS_GATE: false } } : step),
    steps.map(step => step.op === 'down_proj' ? { ...step, constants: { ...step.constants, SCALAR_OUTPUT: true } } : step)]) {
    rejects({ ...base, kernelPath: { prefill: { steps: badSteps }, decode: { steps: badSteps } } });
  }
} finally { hooks.deregister(); delete globalThis.pairedFfnTest; }
console.log('paired FFN declared split sequence, shape/precision, immediate/recorded failures and borrowed input ownership passed');
