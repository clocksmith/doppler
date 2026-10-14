import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getWeightDtype, getLayout } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';
import { getDevice } from '../device.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const REPRESENTATION = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });

/** @type {typeof import('./paired-matmul.js').resolvePairedMatmulPipeline} */
export function resolvePairedMatmulPipeline(weight, options) {
  const dtype = getWeightDtype(weight);
  if (typeof dtype !== 'string' || !['f32', 'f16', 'q4k'].includes(dtype)) {
    throw new Error('[PairedMatmul] Explicit F32/F16/Q4K GPU weights are required.');
  }
  const layout = getLayout(weight);
  const automatic = options.transposeB === undefined || options.transposeB === 'auto';
  if (automatic && layout !== 'row' && layout !== 'column') throw new Error('[PairedMatmul] Explicit row/column layout required.');
  if (options.outputPrecision !== undefined && options.outputPrecision !== 'f32' && options.outputPrecision !== 'two-f32') {
    throw new Error('[PairedMatmul] Invalid output precision.');
  }
  const transpose = automatic ? layout === 'row' : options.transposeB === true;
  const scalarOutput = options.outputPrecision === 'f32';
  return { operation: 'matmul', variant: selectRuleValue('matmul', 'pairedVariant', { weightDtype: dtype, scalarOutput }),
    constants: { WORKGROUP_SIZE: 64, WEIGHT_F16: dtype === 'f16', WEIGHT_Q4K: dtype === 'q4k',
      TRANSPOSE_B: transpose, SCALAR_OUTPUT: scalarOutput } };
}

/** @type {typeof import('./paired-matmul.js').executePairedMatmul} */
export async function executePairedMatmul(recorder, input, weight, M, N, K, options) {
  if (![M, N, K].every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)
    || !Number.isSafeInteger(M * N) || M * N > 0xffffffff || !Number.isSafeInteger(M * K * 8)) {
    throw new Error('[PairedMatmul] Invalid explicit M/N/K geometry.');
  }
  if (options.outputDtype !== 'f32') throw new Error('[PairedMatmul] Component outputDtype must be explicitly F32.');
  if (input.dtype !== 'f32' || input.precision !== 'two-f32' || input.components !== 2 || input.elementStrideBytes !== 8) {
    throw new Error('[PairedMatmul] Interleaved two-F32 input is required.');
  }
  const buffer = getBuffer(weight);
  if (!buffer || !('size' in buffer)) throw new Error('[PairedMatmul] Explicit F32/F16/Q4K GPU weights are required.');
  const pipeline = resolvePairedMatmulPipeline(weight, options);
  const { WEIGHT_Q4K: q4k, TRANSPOSE_B: transpose, SCALAR_OUTPUT: scalarOutput } = pipeline.constants;
  const dtype = getWeightDtype(weight);
  if (q4k && (K % 256 !== 0 || !transpose)) throw new Error('[PairedMatmul] Q4K requires row layout and a K multiple of256.');
  const aOffset = options.aOffset ?? 0, bOffset = options.bOffset ?? 0, cOffset = options.cOffset ?? 0;
  if ([aOffset, bOffset, cOffset].some(value => !Number.isSafeInteger(value) || value < 0)
    || aOffset % 8 || bOffset % 4 || cOffset % (scalarOutput ? 4 : 8)) {
    throw new Error('[PairedMatmul] Byte offsets must align to their declared element representations.');
  }
  // This bounded wrapper currently exposes whole operands. Offset views require
  // a separately declared binding contract, never a silent offset discard.
  if (aOffset || bOffset || cOffset) throw new Error('[PairedMatmul] Nonzero operand offsets are not yet supported.');
  const weightBytes = dtype === 'q4k' ? N * (K / 256) * 144 : N * K * (dtype === 'f16' ? 2 : 4);
  if (input.buffer.size < M * K * 8 || buffer.size < weightBytes) throw new Error('[PairedMatmul] Operand buffer is undersized.');
  const alpha = options.alpha ?? 1;
  if (!Number.isFinite(alpha) || !Number.isFinite(Math.fround(alpha))) throw new Error('[PairedMatmul] Invalid alpha.');
  const bytes = M * N * (scalarOutput ? 4 : 8), supplied = options.outputBuffer;
  if (supplied && (supplied.size < bytes || supplied === input.buffer || supplied === buffer)) {
    throw new Error('[PairedMatmul] Output must be distinct and adequately sized.');
  }
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[PairedMatmul] GPU device is not initialized.');
  const output = supplied ?? acquireBuffer(bytes, undefined, 'matmul_paired_output');
  const gridWidth = Math.min(M * N, device.limits.maxComputeWorkgroupsPerDimension);
  try {
    await unifiedKernelWrapper(pipeline.operation, recorder, pipeline.variant, [input, buffer, output], {
      M, N, K, grid_width: gridWidth, alpha_hi: Math.fround(alpha), alpha_lo: Math.fround(alpha - Math.fround(alpha)),
    }, [gridWidth, Math.ceil(M * N / gridWidth), 1], pipeline.constants);
    return createTensor(output, 'f32', [M, N], 'matmul_paired', scalarOutput ? undefined : REPRESENTATION);
  } catch (error) {
    if (!supplied) {
      if (recorder) recorder.trackTemporaryBuffer(output); else releaseBuffer(output);
    }
    throw error;
  }
}
