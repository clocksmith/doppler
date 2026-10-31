import { ggufBlockFormat, ggufBlockBytes } from '../../formats/gguf/block-storage.js';
import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { getKernelConfig } from './kernel-configs.js';
import { assertQ40Buffer, q40Dispatch } from './q4_0.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

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
 * @param {string} operation @param {number} storageSlots */
function prepare(recorder, options, operation, storageSlots) {
  validateSharedSlotGeometry(options);
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[MoE shared slot] Active device required.');
  const config = getKernelConfig(operation, 'native_f32');
  const width = config.wgslOverrides?.WORKGROUP_SIZE;
  if (typeof width !== 'number' || width < 4 || width % 4 || (width & (width - 1))
    || width > device.limits.maxComputeInvocationsPerWorkgroup
    || width * 16 * storageSlots > device.limits.maxComputeWorkgroupStorageSize) {
    throw new Error('[MoE shared slot] Registered reduction exceeds device workgroup limits.');
  }
  const slotStride = Math.max(options.intermediateSize, options.sharedSize);
  const uniforms = { num_tokens: options.numTokens, hidden_size: options.hiddenSize,
    intermediate_size: options.intermediateSize, shared_size: options.sharedSize,
    num_experts: options.numExperts, slot_stride: slotStride, clamp_max: options.swigluLimit ?? 0, _pad: 0 };
  assertQ40Buffer(device, options.indices, options.numTokens * options.topK * 4, GPUBufferUsage.STORAGE, 'route indices');
  return { device, width, slotStride, uniforms };
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
  const { device, width, slotStride, uniforms } = prepare(recorder, options, 'moe_shared_slot_gu', 2);
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
    await unifiedKernelWrapper('moe_shared_slot_gu', recorder, 'native_f32', [...buffers, output], uniforms,
      q40Dispatch(device, Math.ceil(slotStride / 4) * rows),
      { WORKGROUP_SIZE: width, TOP_K: topK, ROUTED_FORMAT: routed.code, ROUTED_BYTES: routed.bytes,
        SHARED_FORMAT: shared.code, SHARED_BYTES: shared.bytes });
    return createTensor(output, 'f32', [rows, slotStride], 'moe_shared_slot_gu');
  });
}

/** @type {typeof import('./moe-shared-slot.js').executeSharedSlotDown} */
export async function executeSharedSlotDown(recorder, input, weights, gate, options) {
  const { device, width, slotStride, uniforms } = prepare(recorder, options, 'moe_shared_slot_down', options.topK + 1);
  const { numTokens, numExperts, topK, hiddenSize, intermediateSize, sharedSize } = options;
  validateInput(device, input, numTokens * (topK + 1), slotStride);
  validateInput(device, gate, numTokens, 1);
  const routed = validateWeight(device, weights.down, [numExperts, hiddenSize, intermediateSize]);
  const shared = validateWeight(device, weights.sharedDown, [hiddenSize, sharedSize]);
  assertQ40Buffer(device, options.routeWeights, numTokens * topK * 4, GPUBufferUsage.STORAGE, 'route weights');
  const buffers = [input.buffer, weights.down.buffer, weights.sharedDown.buffer,
    options.indices, options.routeWeights, gate.buffer];
  const bytes = numTokens * hiddenSize * 4;
  validateOutput(device, options.outputBuffer, bytes, buffers);
  return withKernelOutput(recorder, options.outputBuffer ?? null, bytes, 'moe_shared_slot_down', async output => {
    await unifiedKernelWrapper('moe_shared_slot_down', recorder, 'native_f32', [...buffers, output], uniforms,
      q40Dispatch(device, Math.ceil(hiddenSize / 4) * numTokens),
      { WORKGROUP_SIZE: width, TOP_K: topK, ROUTED_FORMAT: routed.code, ROUTED_BYTES: routed.bytes,
        SHARED_FORMAT: shared.code, SHARED_BYTES: shared.bytes });
    return createTensor(output, 'f32', [numTokens, hiddenSize], 'moe_shared_slot_down');
  });
}
