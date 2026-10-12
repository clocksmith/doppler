import { createTensor } from '../tensor.js';
import { getDevice } from '../device.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { WORKGROUP_SIZES } from './constants.js';
import { selectRuleValue } from './rule-registry.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const PAIR = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });
/** @param {import('../tensor.js').Tensor} value @param {number} count */
function validate(value, count) {
  if (!Number.isSafeInteger(count) || count <= 0 || count > 0xffffffff) throw new Error('[PairedActivation] Positive U32 count required.');
  if (value?.dtype !== 'f32' || value.precision !== 'two-f32' || value.components !== 2
    || value.elementStrideBytes !== 8 || !Array.isArray(value.shape) || value.shape.length === 0
    || value.shape.some(size => !Number.isSafeInteger(size) || size <= 0)) {
    throw new Error('[PairedActivation] Explicit interleaved two-F32 tensor required.');
  }
  const elements = value.shape.reduce((total, size) => total * size, 1);
  if (!Number.isSafeInteger(elements) || elements < count || !value.buffer || !Number.isSafeInteger(value.buffer.size) || value.buffer.size < count * 8) {
    throw new Error('[PairedActivation] Tensor shape or buffer is undersized.');
  }
}
/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} a
 * @param {import('../tensor.js').Tensor} b
 * @param {number} count
 * @param {string} operation
 * @param {import('./paired-activation.js').PairedPointwiseOptions} options
 * @param {number} scale
 * @param {Record<string, number | boolean> | null} constants
 * @param {readonly number[]} outputShape
 */
async function execute(target, a, b, count, operation, options, scale, constants = null, outputShape = [count]) {
  validate(a, count); validate(b, count);
  if (!Number.isFinite(scale) || !Number.isFinite(Math.fround(scale))) throw new Error('[PairedActivation] Finite F32-pair scale required.');
  const inplace = options.inplace === true;
  if (inplace && operation !== 'scale') throw new Error('[PairedActivation] In-place mode requires scale.');
  if (inplace && options.outputBuffer && options.outputBuffer !== a.buffer) throw new Error('[PairedActivation] In-place scale cannot target another output.');
  const bytes = count * 8, supplied = inplace ? a.buffer : options.outputBuffer;
  if (supplied && ((!inplace && (supplied === a.buffer || supplied === b.buffer)) || supplied.size < bytes)) {
    throw new Error('[PairedActivation] Output must be distinct and sufficiently sized.');
  }
  const device = target?.device ?? getDevice();
  if (!device) throw new Error('[PairedActivation] GPU device required.');
  const workgroups = Math.ceil(count / WORKGROUP_SIZES.DEFAULT);
  const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
  const variant = selectRuleValue('silu', 'pairedVariant', { operation, inplace });
  return withKernelOutput(target, supplied ?? null, bytes, 'paired_pointwise', async output => {
    await unifiedKernelWrapper('paired_pointwise', target, variant, inplace ? [output] : (operation === 'scale' || operation === 'silu_unary' ? [a, output] : [a, b, output]), {
      size: count, scale_hi: Math.fround(scale), scale_lo: Math.fround(scale - Math.fround(scale)),
    }, [x, Math.ceil(workgroups / x), 1], constants);
    return createTensor(output, 'f32', outputShape, 'paired_pointwise', PAIR);
  });
}
/** @type {typeof import('./paired-activation.js').executePairedAdd} */
export function executePairedAdd(target, a, b, count, options = {}) {
  return execute(target, a, b, count, 'add', options, options.scale ?? 1);
}
/** @type {typeof import('./paired-activation.js').executePairedScale} */
export function executePairedScale(target, input, scale, count, options = {}) {
  const shape = input.shape.reduce((total, size) => total * size, 1) === count ? input.shape : [count];
  return execute(target, input, input, count, 'scale', options, scale, null, shape);
}
/** @type {typeof import('./paired-activation.js').executePairedMultiply} */
export function executePairedMultiply(target, a, b, count, options = {}) {
  return execute(target, a, b, count, 'multiply', options, 1);
}
/** @type {typeof import('./paired-activation.js').executePairedSiLU} */
export async function executePairedSiLU(target, input, count, options) {
  if (options.swigluLimit !== null) throw new Error('[PairedActivation] Paired SiLU requires explicitly disabled swigluLimit.');
  if (!['silu', 'sigmoid'].includes(options.gateActivation) || !['silu', 'identity'].includes(options.inputActivation)) {
    throw new Error('[PairedActivation] Explicit supported gate/input activation required.');
  }
  const shape = input.shape.reduce((total, size) => total * size, 1) === count ? input.shape : [count];
  if (!options.gate) return execute(target, input, input, count, options.inputActivation === 'identity' ? 'scale' : 'silu_unary', options, 1, null, shape);
  return execute(target, input, options.gate, count, options.gateActivation, options, 1, {
    HAS_GATE: options.gate != null, INPUT_USE_IDENTITY: options.inputActivation === 'identity',
  }, shape);
}
