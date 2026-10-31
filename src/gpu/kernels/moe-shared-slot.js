import { ggufBlockFormat, ggufBlockBytes } from '../../formats/gguf/block-storage.js';
import { getDevice, getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import { getKernelConfig } from './kernel-configs.js';
import { assertQ40Buffer, q40Dispatch } from './q4_0.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';

const COMBINE_WORKGROUP = 256;

/** @type {typeof import('./moe-shared-slot.js').validateSharedSlotGeometry} */
export function validateSharedSlotGeometry(options) {
  for (const value of [options.numTokens, options.numExperts, options.topK, options.hiddenSize,
    options.intermediateSize, options.sharedSize]) {
    if (!Number.isSafeInteger(value) || value < 1 || value > 0xffffff) {
      throw new Error('[MoE shared slot] Explicit positive bounded geometry is required.');
    }
  }
  if (options.topK > options.numExperts || options.hiddenSize % 32
    || options.intermediateSize % 32 || options.sharedSize % 32
    || (options.swigluLimit !== null && (!Number.isFinite(options.swigluLimit) || !Number.isFinite(Math.fround(options.swigluLimit))))) {
    throw new Error('[MoE shared slot] Invalid topK, complete native blocks, or explicit SwiGLU limit.');
  }
}

/** @param {GPUDevice} device @param {import('../weight-buffer.js').WeightBuffer} weight
 * @param {number[]} shape */
function validateWeight(device, weight, shape) {
  if (!weight || weight.layout !== 'row' || !['q4_0', 'q4_1', 'q5_0', 'q8_0'].includes(weight.dtype)
    || weight.shape.length !== shape.length || weight.shape.some((n, i) => n !== shape[i])) {
    throw new Error('[MoE shared slot] Expected declared native GGUF row-packed weight geometry.');
  }
  assertQ40Buffer(device, weight.buffer, ggufBlockBytes(shape, weight.dtype), GPUBufferUsage.STORAGE, 'shared slot weight');
  return ggufBlockFormat(weight.dtype);
}

/** @param {GPUDevice} device @param {import('../tensor.js').Tensor} input @param {number} rows @param {number} columns */
function validateInput(device, input, rows, columns) {
  if (input.dtype !== 'f32' || input.precision != null || input.components != null
    || input.elementStrideBytes != null || input.shape.length !== 2
    || input.shape[0] !== rows || input.shape[1] !== columns) {
    throw new Error('[MoE shared slot] Explicit scalar F32 input geometry required.');
  }
  assertQ40Buffer(device, input.buffer, rows * columns * 4, GPUBufferUsage.STORAGE, 'shared slot input');
}

/** @param {import('../command-recorder.js').CommandRecorder | null} recorder
 * @param {import('./moe-shared-slot.js').SharedSlotOptions} options
 * @param {string} operation @param {number} storageSlots @param {number} inputWidth */
function prepare(recorder, options, operation, storageSlots, inputWidth) {
  validateSharedSlotGeometry(options);
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[MoE shared slot] Active device required.');
  const variant = resolveSharedSlotVariant(options.variant);
  const config = getKernelConfig(operation, variant);
  const width = config.wgslOverrides?.WORKGROUP_SIZE;
  if (typeof width !== 'number') throw new Error('[MoE shared slot] Registered workgroup size required.');
  let rowsPerWorkgroup = 0;
  const slotStride = Math.max(options.intermediateSize, options.sharedSize);
  /** @type {Record<string, number>} */
  let constants = { WORKGROUP_SIZE: width };
  if (variant === PAIR_VARIANT || variant === PAIR_COMBINE_VARIANT || variant === PAIR_COMBINE_RESIDUAL_VARIANT) {
    const threadsPerRow = config.wgslOverrides?.THREADS_PER_ROW;
    if (typeof threadsPerRow !== 'number') throw new Error('[MoE shared slot] Registered row group size required.');
    rowsPerWorkgroup = width / threadsPerRow;
    const pairs = inputWidth / 64;
    const minSubgroup = device.features?.has('subgroups') ? getKernelCapabilities()?.adapterInfo?.subgroupMinSize : null;
    if (typeof minSubgroup !== 'number' || typeof threadsPerRow !== 'number'
      || threadsPerRow < 1 || (threadsPerRow & (threadsPerRow - 1)) || minSubgroup < threadsPerRow
      || typeof width !== 'number' || width % minSubgroup || width % threadsPerRow
      || width > device.limits.maxComputeInvocationsPerWorkgroup) {
      throw new Error('[MoE shared slot] Pair geometry requires subgroups no narrower than one row group.');
    }
    if (options.hiddenSize % 64 || options.intermediateSize % 64 || options.sharedSize % 64
      || pairs * 17 * 16 > device.limits.maxComputeWorkgroupStorageSize
      || options.numTokens * (options.topK + 1) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error('[MoE shared slot] Pair geometry requires even block counts and bounded staging.');
    }
    constants = variant !== PAIR_VARIANT
      ? { WORKGROUP_SIZE: width, THREADS_PER_ROW: threadsPerRow }
      : { WORKGROUP_SIZE: width, THREADS_PER_ROW: threadsPerRow, MAX_INPUT_PAIRS: pairs };
  } else if (typeof width !== 'number' || width < 4 || width % 4 || (width & (width - 1))
    || width > device.limits.maxComputeInvocationsPerWorkgroup
    || width * 16 * storageSlots > device.limits.maxComputeWorkgroupStorageSize) {
    throw new Error('[MoE shared slot] Registered reduction exceeds device workgroup limits.');
  }
  const uniforms = { num_tokens: options.numTokens, hidden_size: options.hiddenSize,
    intermediate_size: options.intermediateSize, shared_size: options.sharedSize,
    num_experts: options.numExperts, slot_stride: slotStride, clamp_max: options.swigluLimit ?? 0, _pad: 0 };
  assertQ40Buffer(device, options.indices, options.numTokens * options.topK * 4, GPUBufferUsage.STORAGE, 'route indices');
  return { device, width, slotStride, uniforms, variant, rowsPerWorkgroup, constants };
}

const PAIR_VARIANT = 'pair_f32';
// Down only: the pair row geometry with the slot combine in the same dispatch.
const PAIR_COMBINE_VARIANT = 'pair_combine_f32';
// Down only: pair_combine_f32 that also adds the layer's outer residual.
const PAIR_COMBINE_RESIDUAL_VARIANT = 'pair_combine_residual_f32';
/** @param {string | undefined} variant */
function resolveSharedSlotVariant(variant) {
  if (variant !== 'native_f32' && variant !== PAIR_VARIANT && variant !== PAIR_COMBINE_VARIANT
    && variant !== PAIR_COMBINE_RESIDUAL_VARIANT) {
    throw new Error('[MoE shared slot] Explicit registered shared-slot variant required.');
  }
  return variant;
}


/** @param {GPUDevice} device @param {GPUBuffer | null | undefined} output
 * @param {number} bytes @param {GPUBuffer[]} inputs */
function validateOutput(device, output, bytes, inputs) {
  if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff || bytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error('[MoE shared slot] Output extent exceeds binding limits.');
  }
  if (!output) return;
  assertQ40Buffer(device, output, bytes, GPUBufferUsage.STORAGE, 'shared slot output');
  if (inputs.includes(output)) throw new Error('[MoE shared slot] Output must not alias inputs.');
}

/** @type {typeof import('./moe-shared-slot.js').executeSharedSlotGateUp} */
export async function executeSharedSlotGateUp(recorder, input, weights, options) {
  const { device, slotStride, uniforms, variant, rowsPerWorkgroup, constants } =
    prepare(recorder, options, 'moe_shared_slot_gu', 2, options.hiddenSize);
  if (variant === PAIR_COMBINE_VARIANT || variant === PAIR_COMBINE_RESIDUAL_VARIANT) {
    throw new Error(`[MoE shared slot] ${variant} is a down-only variant.`);
  }
  const { numTokens, numExperts, topK, hiddenSize, intermediateSize, sharedSize } = options;
  validateInput(device, input, numTokens, hiddenSize);
  const routed = validateWeight(device, weights.gate, [numExperts, intermediateSize, hiddenSize]);
  validateWeight(device, weights.up, [numExperts, intermediateSize, hiddenSize]);
  const shared = validateWeight(device, weights.sharedGate, [sharedSize, hiddenSize]);
  validateWeight(device, weights.sharedUp, [sharedSize, hiddenSize]);
  if (weights.gate.dtype !== weights.up.dtype || weights.sharedGate.dtype !== weights.sharedUp.dtype) {
    throw new Error('[MoE shared slot] Gate/up storage must match within each bank.');
  }
  const buffers = [input.buffer, weights.gate.buffer, weights.up.buffer,
    weights.sharedGate.buffer, weights.sharedUp.buffer, options.indices];
  const rows = numTokens * (topK + 1), bytes = rows * slotStride * 4;
  validateOutput(device, options.outputBuffer, bytes, buffers);
  return withKernelOutput(recorder, options.outputBuffer ?? null, bytes, 'moe_shared_slot_gu', async output => {
    const workgroups = variant === PAIR_VARIANT
      ? [Math.ceil(slotStride / rowsPerWorkgroup), rows, 1]
      : q40Dispatch(device, Math.ceil(slotStride / 4) * rows);
    await unifiedKernelWrapper('moe_shared_slot_gu', recorder, variant, [...buffers, output], uniforms,
      workgroups,
      { ...constants, TOP_K: topK, ROUTED_FORMAT: routed.code, ROUTED_BYTES: routed.bytes,
        SHARED_FORMAT: shared.code, SHARED_BYTES: shared.bytes });
    return createTensor(output, 'f32', [rows, slotStride], 'moe_shared_slot_gu');
  });
}

/** @type {typeof import('./moe-shared-slot.js').executeSharedSlotDown} */
export async function executeSharedSlotDown(recorder, input, weights, gate, options) {
  const { device, width, slotStride, uniforms, variant, rowsPerWorkgroup, constants } = prepare(recorder, options,
    'moe_shared_slot_down', options.topK + 1, Math.max(options.intermediateSize, options.sharedSize));
  const { numTokens, numExperts, topK, hiddenSize, intermediateSize, sharedSize } = options;
  validateInput(device, input, numTokens * (topK + 1), slotStride);
  validateInput(device, gate, numTokens, 1);
  const routed = validateWeight(device, weights.down, [numExperts, hiddenSize, intermediateSize]);
  const shared = validateWeight(device, weights.sharedDown, [hiddenSize, sharedSize]);
  assertQ40Buffer(device, options.routeWeights, numTokens * topK * 4, GPUBufferUsage.STORAGE, 'route weights');
  const buffers = [input.buffer, weights.down.buffer, weights.sharedDown.buffer,
    options.indices, options.routeWeights, gate.buffer];
  const bytes = numTokens * hiddenSize * 4;
  // The residual-bearing variant and a residual input come together or not at all:
  // the caller skips its own residual add only when this kernel adds it.
  const residual = options.residual ?? null;
  if ((variant === PAIR_COMBINE_RESIDUAL_VARIANT) !== (residual !== null)) {
    throw new Error(`[MoE shared slot] ${variant} ${residual ? 'does not add' : 'requires'} an outer residual input.`);
  }
  if (residual) validateInput(device, residual, numTokens, hiddenSize);
  validateOutput(device, options.outputBuffer, bytes, residual ? [...buffers, residual.buffer] : buffers);
  const kernelConstants = { ...constants, TOP_K: topK, ROUTED_FORMAT: routed.code, ROUTED_BYTES: routed.bytes,
    SHARED_FORMAT: shared.code, SHARED_BYTES: shared.bytes };
  if (residual) {
    return withKernelOutput(recorder, options.outputBuffer ?? null, bytes, 'moe_shared_slot_down', async output => {
      await unifiedKernelWrapper('moe_shared_slot_down', recorder, variant, [...buffers, output, residual.buffer], uniforms,
        [Math.ceil(hiddenSize / rowsPerWorkgroup), numTokens, 1], kernelConstants);
      return createTensor(output, 'f32', [numTokens, hiddenSize], 'moe_shared_slot_down_residual');
    });
  }
  if (variant === PAIR_COMBINE_VARIANT) {
    return withKernelOutput(recorder, options.outputBuffer ?? null, bytes, 'moe_shared_slot_down', async output => {
      await unifiedKernelWrapper('moe_shared_slot_down', recorder, variant, [...buffers, output], uniforms,
        [Math.ceil(hiddenSize / rowsPerWorkgroup), numTokens, 1], kernelConstants);
      return createTensor(output, 'f32', [numTokens, hiddenSize], 'moe_shared_slot_down');
    });
  }
  if (variant === PAIR_VARIANT) {
    const slotsBytes = numTokens * (topK + 1) * hiddenSize * 4;
    validateOutput(device, options.slotsBuffer, slotsBytes, buffers);
    const slots = options.slotsBuffer ?? acquireBuffer(slotsBytes, undefined, 'moe_shared_slot_down_slots');
    try {
      return await withKernelOutput(recorder, options.outputBuffer ?? null, bytes, 'moe_shared_slot_down', async output => {
        if (output === slots) throw new Error('[MoE shared slot] Output must not alias per-slot outputs.');
        await unifiedKernelWrapper('moe_shared_slot_down', recorder, variant,
          [input.buffer, weights.down.buffer, weights.sharedDown.buffer, options.indices, slots], uniforms,
          [Math.ceil(hiddenSize / rowsPerWorkgroup), numTokens * (topK + 1), 1], kernelConstants);
        await unifiedKernelWrapper('moe_shared_slot_combine', recorder, 'f32',
          [slots, options.routeWeights, gate.buffer, output], uniforms,
          q40Dispatch(device, Math.ceil(numTokens * hiddenSize / COMBINE_WORKGROUP)),
          { WORKGROUP_SIZE: COMBINE_WORKGROUP, TOP_K: topK });
        return createTensor(output, 'f32', [numTokens, hiddenSize], 'moe_shared_slot_down');
      });
    } finally {
      if (!options.slotsBuffer) {
        if (recorder) recorder.trackTemporaryBuffer(slots);
        else releaseBuffer(slots);
      }
    }
  }
  return withKernelOutput(recorder, options.outputBuffer ?? null, bytes, 'moe_shared_slot_down', async output => {
    await unifiedKernelWrapper('moe_shared_slot_down', recorder, 'native_f32', [...buffers, output], uniforms,
      q40Dispatch(device, Math.ceil(hiddenSize / 4) * numTokens),
      { WORKGROUP_SIZE: width, TOP_K: topK, ROUTED_FORMAT: routed.code, ROUTED_BYTES: routed.bytes,
        SHARED_FORMAT: shared.code, SHARED_BYTES: shared.bytes });
    return createTensor(output, 'f32', [numTokens, hiddenSize], 'moe_shared_slot_down');
  });
}
