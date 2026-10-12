import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const REPRESENTATION = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });

/** @param {import('../tensor.js').Tensor} input @param {number} bytes @param {string} label */
function assertPair(input, bytes, label) {
  if (input?.dtype !== 'f32' || input.precision !== 'two-f32' || input.components !== 2
    || input.elementStrideBytes !== 8 || !input.buffer || !Number.isFinite(input.buffer.size) || input.buffer.size < bytes) {
    throw new Error(`[PairedRMSNorm] ${label} must be an adequately sized interleaved two-F32 tensor.`);
  }
}

/** @type {typeof import('./paired-rmsnorm.js').executePairedRMSNorm} */
export async function executePairedRMSNorm(recorder, input, weight, eps, options) {
  const { batchSize, hiddenSize, preResidual = null, residual = null,
    residualSumOutput = null, outputBuffer = null, outputScale = null } = options;
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0 || !Number.isSafeInteger(hiddenSize)
    || hiddenSize <= 0 || hiddenSize > 0xffffff || !Number.isSafeInteger(batchSize * hiddenSize * 8)) {
    throw new Error('[PairedRMSNorm] Explicit positive batchSize and hiddenSize are required.');
  }
  if (!Number.isFinite(eps) || eps <= 0 || Math.fround(eps) === 0) {
    throw new Error('[PairedRMSNorm] Epsilon must be positive and representable as a normalized F32 pair.');
  }
  if (preResidual && residual) throw new Error('[PairedRMSNorm] Pre- and post-residual are mutually exclusive.');
  if (residualSumOutput && !preResidual) throw new Error('[PairedRMSNorm] Prenorm output requires a pre-residual.');
  const scale = outputScale === null ? 1 : outputScale;
  if (!Number.isFinite(scale) || !Number.isFinite(Math.fround(scale))) throw new Error('[PairedRMSNorm] Invalid output scale.');
  const bytes = batchSize * hiddenSize * 8;
  assertPair(input, bytes, 'Input');
  const residualInput = preResidual ?? residual;
  if (residualInput) assertPair(residualInput, bytes, 'Residual');
  const weightDtype = getWeightDtype(weight), weightBuffer = getBuffer(weight);
  if (weightDtype !== 'f32' && weightDtype !== 'f16') throw new Error('[PairedRMSNorm] Explicit F32 or F16 weights required.');
  if (!weightBuffer || !('size' in weightBuffer) || weightBuffer.size < hiddenSize * (weightDtype === 'f16' ? 2 : 4)) {
    throw new Error('[PairedRMSNorm] Weight buffer is undersized.');
  }
  const prenormProvided = residualSumOutput && 'buffer' in residualSumOutput ? residualSumOutput.buffer : residualSumOutput;
  for (const supplied of [outputBuffer, prenormProvided]) {
    if (supplied && (supplied.size < bytes || supplied === input.buffer || supplied === residualInput?.buffer)) {
      throw new Error('[PairedRMSNorm] Output must be distinct from inputs and adequately sized.');
    }
  }
  if (outputBuffer && outputBuffer === prenormProvided) throw new Error('[PairedRMSNorm] Outputs must not alias.');
  let output = null, prenorm = null;
  try {
    output = outputBuffer ?? acquireBuffer(bytes, undefined, 'rmsnorm_paired_output');
    prenorm = prenormProvided ?? acquireBuffer(8, undefined, 'rmsnorm_paired_prenorm_placeholder');
    const tokenStride = Math.min(batchSize, recorder?.device?.limits?.maxComputeWorkgroupsPerDimension ?? 65535);
    await unifiedKernelWrapper('rmsnorm', recorder, 'paired',
      [input, weightBuffer, output, residualInput ?? input, prenorm], {
        size: hiddenSize, num_tokens: batchSize,
        eps_hi: Math.fround(eps), eps_lo: Math.fround(eps - Math.fround(eps)),
        token_stride: tokenStride, scale_hi: Math.fround(scale), scale_lo: Math.fround(scale - Math.fround(scale)),
      }, [tokenStride, Math.ceil(batchSize / tokenStride), 1], {
        WORKGROUP_SIZE: 64, WEIGHT_F16: weightDtype === 'f16', WEIGHT_OFFSET: options.rmsNormWeightOffset === true,
        PRE_RESIDUAL: preResidual != null, POST_RESIDUAL: residual != null, OUTPUT_PRENORM: prenormProvided != null,
      });
    return createTensor(output, 'f32', [batchSize, hiddenSize], 'rmsnorm_paired', REPRESENTATION);
  } catch (error) {
    if (output && !outputBuffer) {
      if (recorder) recorder.trackTemporaryBuffer(output); else releaseBuffer(output);
    }
    throw error;
  } finally {
    if (prenorm && !prenormProvided) {
      if (recorder) recorder.trackTemporaryBuffer(prenorm); else releaseBuffer(prenorm);
    }
  }
}
