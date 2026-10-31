import assert from 'node:assert/strict';
import { validateSigmoidGatedAdd } from '../../src/gpu/kernels/sigmoid-gated-add.js';
import { resolveQwenSharedExpertCombine } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/config/kernel-registry-contract.js';
const tensor = (rows, columns) => ({ dtype: 'f32', shape: [rows, columns], buffer: { size: rows * columns * 4 } });
const input = tensor(33, 2048), gate = tensor(33, 1), residual = tensor(33, 2048);
const options = { numTokens: 33, hiddenSize: 2048, swigluLimit: null };
validateSigmoidGatedAdd(input, gate, residual, options);
for (const swigluLimit of [0, -1, 0.25]) validateSigmoidGatedAdd(input, gate, residual, { ...options, swigluLimit });
for (const override of [{ hiddenSize: 0 }, { numTokens: 2.5 }, { numTokens: 0x100000000 },
  { swigluLimit: undefined }, { swigluLimit: NaN }, { outputBuffer: input.buffer },
  { outputBuffer: { size: 4 } }]) {
  assert.throws(() => validateSigmoidGatedAdd(input, gate, residual, { ...options, ...override }));
}
assert.throws(() => validateSigmoidGatedAdd({ ...input, dtype: 'f16' }, gate, residual, options));
assert.throws(() => validateSigmoidGatedAdd(input, tensor(1, 33), residual, options));
assert.throws(() => validateSigmoidGatedAdd(input, { ...gate, buffer: { size: 4 } }, residual, options));
assert.equal(resolveQwenSharedExpertCombine({ kernelPath: null }), false);
const step = { op: 'moe_shared_combine', kernel: 'sigmoid_gated_add_f32.wgsl', entry: 'main' };
const path = { decode: { steps: [step] }, prefill: { steps: [] } };
const lookup = { kernelPath: path, phase: 'decode', layerIdx: 0 };
assert.equal(resolveQwenSharedExpertCombine(lookup), true);
assert.equal(resolveQwenSharedExpertCombine({ ...lookup, phase: 'prefill' }), false);
assert.throws(() => resolveQwenSharedExpertCombine({ ...lookup, phase: undefined }));
assert.throws(() => resolveQwenSharedExpertCombine({ ...lookup, layerIdx: -1 }));
assert.throws(() => resolveQwenSharedExpertCombine({ ...lookup,
  kernelPath: { decode: { steps: [step, step] } } }));
assert.throws(() => resolveQwenSharedExpertCombine({ ...lookup,
  kernelPath: { decode: { steps: [{ ...step, kernel: 'residual.wgsl' }] } } }));
assert.throws(() => resolveQwenSharedExpertCombine({ ...lookup,
  kernelPath: { decode: { steps: [{ ...step, constants: { HIDDEN_SIZE: 7 } }] } } }));
const overridden = { ...path, layerOverrides: [{ layers: [2], decode: { steps: [] }, steps: [{ op: 'other' }] }] };
assert.equal(resolveQwenSharedExpertCombine({ ...lookup, layerIdx: 2, kernelPath: overridden }), false);
console.log('sigmoid gated add contract checks passed (CPU only; no numerical qualification)');

const incompatible = createKernelRegistry({ extensions: { sigmoid_gated_add: { variants: { f32: {
  wgsl: 'residual.wgsl', entryPoint: 'main', workgroup: [256, 1, 1], requires: [],
} } } } });
const restoreRegistry = enterKernelRegistry(incompatible);
try { assert.throws(() => resolveQwenSharedExpertCombine(lookup)); }
finally { restoreRegistry(); }
assert.equal(resolveQwenSharedExpertCombine(lookup), true);
