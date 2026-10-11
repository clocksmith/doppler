import { getDevice, getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import { getKernelConfig } from './kernel-configs.js';
import { hasRequiredFeatures, getKernelWgslRequirements } from './feature-check.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { W8A8_POLICY } from '../../config/schema/w8a8.schema.js';

/** @type {ReadonlyArray<readonly [number, string]>} */
const STATUS_MESSAGES = Object.freeze([
  [1, 'nonfinite activation'], [2, 'subnormal activation'], [4, 'invalid or unrepresentable row scale'],
  [8, 'invalid signed byte or nonzero padding'], [16, 'rescale overflow or invalid intermediate'],
]);
/** @type {WeakMap<import('../command-recorder.js').CommandRecorder, GPUBuffer>} */
const recorderStatuses = new WeakMap();

/** @param {import('../command-recorder.js').CommandRecorder} recorder */
function recorderStatus(recorder) {
  let status = recorderStatuses.get(recorder);
  if (status) return status;
  status = acquireBuffer(4, undefined, 'w8a8_status');
  recorder.trackTemporaryBuffer(status);
  recorder.getEncoder().clearBuffer(status, 0, 4);
  recorderStatuses.set(recorder, status);
  const owned = status;
  recorder.enqueueValidationTask(async () => {
    try { await assertW8A8Status(owned); }
    finally { recorderStatuses.delete(recorder); }
  });
  return status;
}

/** @type {import('./w8a8.js').assertW8A8Status} */
export async function assertW8A8Status(status) {
  const flags = new Uint32Array(await readBuffer(status, 4))[0];
  if (flags === 0) return;
  const error = Object.assign(new Error(`W8A8 rejected GPU computation: ${STATUS_MESSAGES.filter(([bit]) => flags & bit)
    .map(([, message]) => message).join(', ')} (status ${flags}).`),
  { code: 'DOPPLER_W8A8_INVALID_INPUT', statusBits: flags });
  throw error;
}

/** @param {GPUDevice} device @param {GPUBuffer} buffer @param {number} bytes @param {string} label */
function checkBuffer(device, buffer, bytes, label) {
  if (!buffer || !Number.isSafeInteger(bytes) || bytes <= 0 || bytes > 0xffffffff
    || buffer.size < bytes || bytes > device.limits.maxStorageBufferBindingSize
    || buffer.size > device.limits.maxStorageBufferBindingSize
    || !(buffer.usage & GPUBufferUsage.STORAGE)) {
    throw new Error(`W8A8 ${label} requires a bounded storage buffer covering ${bytes} bytes.`);
  }
}

/** @type {import('./w8a8.js').validateW8A8Matmul} */
export function validateW8A8Matmul(device, input, weights, M, N, K, options) {
  if (![M, N, K].every(value => Number.isSafeInteger(value) && value > 0)
    || K > W8A8_POLICY.maxReductionWidth || !Number.isSafeInteger(M * N) || M * N > 0xffffffff) {
    throw new Error('W8A8 dimensions exceed the exact I32 accumulation contract.');
  }
  if (!options || !['w8a8_portable', 'w8a8_packed'].includes(options.variant)) {
    throw new Error('W8A8 requires an explicitly declared portable or packed variant.');
  }
  if (input?.dtype !== 'f32' || weights?.dtype !== 'w8a8' || weights.layout !== 'row'
    || options.outputDtype !== 'f32' || (options.transposeB !== undefined && options.transposeB !== true
      && options.transposeB !== 'auto') || (options.alpha !== undefined && options.alpha !== 1)) {
    throw new Error('W8A8 requires F32 activation/output, row-packed [N,K] weights, transposeB=true and alpha=1.');
  }
  if ([options.aOffset, options.bOffset, options.cOffset].some(value => value !== undefined && value !== 0)) {
    throw new Error('W8A8 does not accept buffer offsets.');
  }
  const declaredConstants = getKernelConfig('matmul', options.variant).wgslOverrides;
  if (options.constants && Object.entries(options.constants).some(([key, value]) => declaredConstants?.[key] !== value)) {
    throw new Error('W8A8 specialization overrides must match the declared registry constants.');
  }
  if (weights.shape?.length !== 2 || weights.shape[0] !== N || weights.shape[1] !== K) {
    throw new Error('W8A8 weight shape must match the explicit [N,K] matrix.');
  }
  const paddedK = Math.ceil(K / 4) * 4, rowStrideBytes = 4 + paddedK;
  const metadata = weights.metadata;
  if (metadata?.packing !== W8A8_POLICY.packing || metadata.paddedK !== paddedK
    || metadata.rowStrideBytes !== rowStrideBytes || metadata.scaleOffsetBytes !== 0 || metadata.dataOffsetBytes !== 4) {
    throw new Error('W8A8 weight metadata differs from the declared row-interleaved packing.');
  }
  checkBuffer(device, input.buffer, M * K * 4, 'activation');
  checkBuffer(device, weights.buffer, N * rowStrideBytes, 'weight');
  if (M * rowStrideBytes > device.limits.maxStorageBufferBindingSize
    || M * N * 4 > device.limits.maxStorageBufferBindingSize) throw new Error('W8A8 scratch/output exceeds device binding limits.');
  if (options.outputBuffer) {
    checkBuffer(device, options.outputBuffer, M * N * 4, 'output');
    if ([input.buffer, weights.buffer].includes(options.outputBuffer)) throw new Error('W8A8 output must not alias either operand.');
  }
  const capabilities = getKernelCapabilities();
  for (const [operation, variant] of [['matmul', options.variant], ['w8a8_quantize', 'f32'],
    ['w8a8_rescale', 'activation'], ['w8a8_rescale', 'weight']]) {
    const config = getKernelConfig(operation, variant);
    if (!hasRequiredFeatures([...config.requires], capabilities, [...getKernelWgslRequirements(config)])) {
      throw new Error(`W8A8 declared kernel ${operation}/${variant} requires unavailable capabilities.`);
    }
    const workgroups = operation === 'w8a8_quantize' ? M : Math.ceil(M * N / config.workgroupSize[0]);
    if (workgroups > device.limits.maxComputeWorkgroupsPerDimension) throw new Error('W8A8 dispatch exceeds device workgroup limits.');
  }
  return { rows: M, columns: N, inner: K, row_words: rowStrideBytes / 4 };
}

/** @param {import('../command-recorder.js').CommandRecorder | null} recorder
 * @param {import('../tensor.js').Tensor} input @param {import('../weight-buffer.js').WeightBuffer} weights
 * @param {number} M @param {number} N @param {number} K @param {import('./w8a8.js').W8A8Options} options */
async function executeW8A8(recorder, input, weights, M, N, K, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('W8A8 requires an initialized GPU device.');
  options?.signal?.throwIfAborted();
  const uniforms = validateW8A8Matmul(device, input, weights, M, N, K, options);
  if (recorder && typeof recorder.enqueueValidationTask !== 'function') {
    throw new Error('W8A8 recording requires mandatory validation at recorder completion.');
  }
  /** @type {GPUBuffer[]} */
  const temporaries = [];
  /** @param {number} bytes @param {string} label */
  const temporary = (bytes, label) => {
    const buffer = acquireBuffer(bytes, undefined, label);
    temporaries.push(buffer);
    if (recorder) {
      try { recorder.trackTemporaryBuffer(buffer); }
      catch (error) { releaseBuffer(buffer); throw error; }
    }
    return buffer;
  };
  try {
    const packed = temporary(M * uniforms.row_words * 4, 'w8a8_activations');
    const accumulator = temporary(M * N * 4, 'w8a8_accumulator');
    const intermediate = temporary(M * N * 4, 'w8a8_activation_scaled');
    const status = recorder ? recorderStatus(recorder) : temporary(4, 'w8a8_status');
    if (!recorder) device.queue.writeBuffer(status, 0, new Uint32Array(1));
    return await withKernelOutput(recorder, options.outputBuffer ?? null, M * N * 4, 'w8a8_output', async output => {
      await unifiedKernelWrapper('w8a8_quantize', recorder, 'f32', [input.buffer, packed, status], uniforms,
        M, null, null, null, options.signal ?? null);
      await unifiedKernelWrapper('matmul', recorder, options.variant, [packed, weights.buffer, accumulator, status], uniforms,
        Math.ceil(M * N / getKernelConfig('matmul', options.variant).workgroupSize[0]), null, null, null, options.signal ?? null);
      await unifiedKernelWrapper('w8a8_rescale', recorder, 'activation', [accumulator, packed, intermediate, status], uniforms,
        Math.ceil(M * N / getKernelConfig('w8a8_rescale', 'activation').workgroupSize[0]), null, null, null, options.signal ?? null);
      await unifiedKernelWrapper('w8a8_rescale', recorder, 'weight', [intermediate, weights.buffer, output, status], uniforms,
        Math.ceil(M * N / getKernelConfig('w8a8_rescale', 'weight').workgroupSize[0]), null, null, null, options.signal ?? null);
      if (!recorder) await assertW8A8Status(status);
      options.signal?.throwIfAborted();
      return createTensor(output, 'f32', [M, N], 'w8a8_output');
    });
  } finally {
    if (!recorder) for (const buffer of temporaries) releaseBuffer(buffer);
  }
}

/** @type {import('./w8a8.js').runW8A8Matmul} */
export function runW8A8Matmul(input, weights, M, N, K, options) {
  return executeW8A8(null, input, weights, M, N, K, options);
}

/** @type {import('./w8a8.js').recordW8A8Matmul} */
export function recordW8A8Matmul(recorder, input, weights, M, N, K, options) {
  return executeW8A8(recorder, input, weights, M, N, K, options);
}
