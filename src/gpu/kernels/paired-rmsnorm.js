import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { getKernelConfig } from './kernel-configs.js';
import { getKernelPathQKNormSpec } from '../../config/kernel-path-loader.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const REPRESENTATION = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });

/** @param {import('../tensor.js').Tensor} input @param {number} bytes @param {string} label */
function assertPair(input, bytes, label) {
  if (input?.dtype !== 'f32' || input.precision !== 'two-f32' || input.components !== 2
    || input.elementStrideBytes !== 8 || !input.buffer || !Number.isFinite(input.buffer.size) || input.buffer.size < bytes) {
    throw new Error(`[PairedRMSNorm] ${label} must be an adequately sized interleaved two-F32 tensor.`);
  }
}

/** Program used when no execution path pins one (direct calls outside a graph). */
const UNPINNED_PAIRED_RMSNORM_VARIANT = 'paired';

/**
 * Interim: the Q/K head-normalization program of a paired graph that declares no q_norm or
 * k_norm step. Capsules signed before those steps existed ran and qualified it. Run admission
 * and Rig signing accept such a graph only when its closure declares this program
 * (assertPairedQKNormClosure), as the `paired_qk_head_norm` mechanism kernel or a step kernel.
 * Remove it, with that acceptance, once every supported manifest declares the steps.
 */
export const UNDECLARED_PAIRED_QK_NORM_VARIANT = 'paired';

/**
 * Registry declaration of a paired RMSNorm program. `tokenSlices` programs dispatch one workgroup
 * per token and output slice of the declared workgroup width; the others one workgroup per token.
 * @param {string} variant
 */
function resolvePairedRMSNormProgram(variant) {
  const config = typeof variant === 'string' ? getKernelConfig('rmsnorm', variant) : null;
  const metadata = config?.variantMetadata;
  const geometry = metadata?.dispatchGeometry;
  if (!config || metadata?.inputPrecision !== 'two-f32' || metadata.outputPrecision !== 'two-f32'
    || (geometry !== undefined && geometry !== 'tokenSlices')) {
    throw new Error(`[PairedRMSNorm] rmsnorm/${String(variant)} is not a declared paired RMSNorm program.`);
  }
  return { sliced: geometry === 'tokenSlices', sliceWidth: config.workgroupSize[0] };
}

/** @type {typeof import('./paired-rmsnorm.js').resolvePairedRMSNormVariant} */
export function resolvePairedRMSNormVariant(declaredVariant) {
  const variant = declaredVariant ?? UNPINNED_PAIRED_RMSNORM_VARIANT;
  resolvePairedRMSNormProgram(variant);
  return variant;
}

/** @type {typeof import('./paired-rmsnorm.js').resolvePairedQKNormVariant} */
export function resolvePairedQKNormVariant(op, phase, layerIdx, kernelPath) {
  if (kernelPath == null) throw new Error(`[PairedRMSNorm] ${op} runs only from the execution graph that declares it.`);
  const declared = getKernelPathQKNormSpec(op, phase, layerIdx, kernelPath);
  return resolvePairedRMSNormVariant(declared === null ? UNDECLARED_PAIRED_QK_NORM_VARIANT : declared.variant);
}

/** @type {typeof import('./paired-rmsnorm.js').resolvePairedRMSNormPipeline} */
export function resolvePairedRMSNormPipeline(weightDtype, fusion, variant) {
  if (weightDtype !== 'f32' && weightDtype !== 'f16') throw new Error('[PairedRMSNorm] Explicit F32 or F16 weights required.');
  const flags = { WEIGHT_F16: weightDtype === 'f16', WEIGHT_OFFSET: fusion.rmsNormWeightOffset === true,
    PRE_RESIDUAL: fusion.preResidual, POST_RESIDUAL: fusion.residual, OUTPUT_PRENORM: fusion.outputPrenorm };
  // Sliced programs declare their workgroup width in registry wgslOverrides.
  if (resolvePairedRMSNormProgram(variant).sliced) return { operation: 'rmsnorm', variant, constants: flags };
  return { operation: 'rmsnorm', variant, constants: { WORKGROUP_SIZE: 64, ...flags } };
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
  const prenormProvided = residualSumOutput && 'buffer' in residualSumOutput ? residualSumOutput.buffer : residualSumOutput;
  const program = resolvePairedRMSNormProgram(options.variant);
  const pipeline = resolvePairedRMSNormPipeline(weightDtype, { rmsNormWeightOffset: options.rmsNormWeightOffset === true,
    preResidual: preResidual != null, residual: residual != null, outputPrenorm: prenormProvided != null }, options.variant);
  if (!weightBuffer || !('size' in weightBuffer) || weightBuffer.size < hiddenSize * (weightDtype === 'f16' ? 2 : 4)) {
    throw new Error('[PairedRMSNorm] Weight buffer is undersized.');
  }
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
    const slices = program.sliced ? Math.ceil(hiddenSize / program.sliceWidth) : 1;
    await unifiedKernelWrapper(pipeline.operation, recorder, pipeline.variant,
      [input, weightBuffer, output, residualInput ?? input, prenorm], {
        size: hiddenSize, num_tokens: batchSize,
        eps_hi: Math.fround(eps), eps_lo: Math.fround(eps - Math.fround(eps)),
        token_stride: tokenStride, scale_hi: Math.fround(scale), scale_lo: Math.fround(scale - Math.fround(scale)),
      }, [tokenStride, Math.ceil(batchSize / tokenStride), slices], pipeline.constants);
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
