import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { validateFP8StorageDescriptor, fp8StorageBytes } from '../../formats/rdrr/fp8-storage.js';
import { getMatmulConfig } from './matmul-selection.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

/** @param {import('../command-recorder.js').CommandRecorder | null} recorder
 * @param {import('../tensor.js').Tensor} input @param {import('../weight-buffer.js').WeightBuffer} weights
 * @param {number} M @param {number} N @param {number} K @param {import('./fp8.js').FP8MatmulOptions} options */
async function executeFP8(recorder, input, weights, M, N, K, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('FP8 matmul requires an initialized GPU device.');
  options.signal?.throwIfAborted();
  if (![M, N, K].every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)
    || M * K > 0xffffffff || M * N > 0xffffffff) throw new Error('FP8 matmul requires bounded explicit M/N/K.');
  if (input.shape.length !== 2 || input.shape[0] !== M || input.shape[1] !== K) throw new Error('FP8 input shape differs from explicit [M,K].');
  if (input.dtype !== 'f32' || options.outputDtype !== 'f32' || input.precision === 'two-f32'
    || (options.alpha !== undefined && options.alpha !== 1)
    || (options.transposeB !== undefined && options.transposeB !== true && options.transposeB !== 'auto')
    || [options.aOffset, options.bOffset, options.cOffset].some(value => value !== undefined && value !== 0)) {
    throw new Error('FP8 matmul requires scalar F32 activation/output, alpha=1 and whole row-packed [N,K] weights.');
  }
  const storage = validateFP8StorageDescriptor(weights.metadata);
  const ruled = selectRuleValue('kernels', 'matmul', 'fp8Variant', { weightDtype: weights.dtype });
  const variant = options.variant ?? ruled;
  const config = getMatmulConfig(variant, null);
  if (config.weightDtype !== weights.dtype || config.outputDtype !== 'f32'
    || config.wgslOverrides?.E4M3FN !== (storage.encoding === 'e4m3fn')
    || config.variantMetadata?.dispatchGeometry !== 'column-row-tile' || config.variantMetadata?.tileM !== 1) {
    throw new Error('FP8 variant does not match the declared weight encoding and reduction geometry.');
  }
  if (options.constants && Object.entries(options.constants).some(([key, value]) => config.wgslOverrides?.[key] !== value)) {
    throw new Error('FP8 caller constants must match the declared registry program.');
  }
  const width = config.workgroupSize[0];
  if (!Number.isSafeInteger(width) || width < 1 || (width & (width-1)) !== 0
    || config.workgroupSize[1] !== 1 || config.workgroupSize[2] !== 1) throw new Error('FP8 reduction requires a power-of-two one-dimensional workgroup.');
  if (weights.layout !== 'row' || weights.shape.length !== 2 || weights.shape[0] !== N || weights.shape[1] !== K) {
    throw new Error('FP8 weight shape/layout differs from explicit [N,K].');
  }
  if (N > device.limits.maxComputeWorkgroupsPerDimension || M > device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error('FP8 dispatch exceeds device axis limits.');
  }
  const weightBytes = fp8StorageBytes([N, K], storage);
  for (const [buffer, size] of /** @type {[GPUBuffer, number][]} */ ([[input.buffer, M*K*4], [weights.buffer, weightBytes]])) {
    if (!buffer || buffer.size < size || !(buffer.usage & GPUBufferUsage.STORAGE)
      || buffer.size > device.limits.maxStorageBufferBindingSize || size > device.limits.maxStorageBufferBindingSize) throw new Error('FP8 operand binding is undersized or unsupported.');
  }
  const supplied = options.outputBuffer ?? null;
  if (supplied && (supplied.size < M*N*4 || !(supplied.usage & GPUBufferUsage.STORAGE)
    || supplied === input.buffer || supplied === weights.buffer)) throw new Error('FP8 output must be a distinct adequately sized storage buffer.');
  if (M*N*4 > device.limits.maxStorageBufferBindingSize) throw new Error('FP8 output exceeds device binding limits.');
  return withKernelOutput(recorder, supplied, M*N*4, 'fp8_output', async output => {
    await unifiedKernelWrapper('matmul', recorder, variant, [input, weights.buffer, output], {
      M, N, K, block_n: storage.blockShape[0], block_k: storage.blockShape[1],
      blocks_k: Math.ceil(K / storage.blockShape[1]), block_words: storage.blockBytes / 4,
    }, [N, M, 1], null, null, null, options.signal ?? null);
    return createTensor(output, 'f32', [M, N], 'fp8_output');
  });
}

/** @type {typeof import('./fp8.js').runFP8Matmul} */
export function runFP8Matmul(input, weights, M, N, K, options) {
  return executeFP8(null, input, weights, M, N, K, options);
}
/** @type {typeof import('./fp8.js').recordFP8Matmul} */
export function recordFP8Matmul(recorder, input, weights, M, N, K, options) {
  return executeFP8(recorder, input, weights, M, N, K, options);
}
