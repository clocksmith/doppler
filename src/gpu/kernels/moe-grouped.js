import { ggufBlockFormat, ggufBlockBytes } from '../../formats/gguf/block-storage.js';
import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { getKernelConfig } from './kernel-configs.js';
import { assertQ40Buffer } from './q4_0.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

/** @type {typeof import('./moe-grouped.js').groupedPrefillGeometry} */
export function groupedPrefillGeometry(options, limits) {
  const sort = getKernelConfig('moe_grouped_sort', 'native_f32');
  const gu = getKernelConfig('moe_grouped_gu', 'native_f32');
  const down = getKernelConfig('moe_grouped_down', 'native_f32');
  const groupSize = Number(sort.wgslOverrides?.GROUP_SIZE);
  const maxExperts = Number(sort.wgslOverrides?.MAX_EXPERTS);
  const width = Number(sort.wgslOverrides?.WORKGROUP_SIZE);
  const { numTokens, numExperts, topK, hiddenSize, intermediateSize } = options;
  const reject = (/** @type {string} */ reason) => {
    throw new Error(`[MoE grouped prefill] Unsupported candidate: ${reason}. Select the unchanged ungrouped execution graph explicitly; no automatic fallback.`);
  };
  if (![numTokens, numExperts, topK, hiddenSize, intermediateSize].every(n => Number.isSafeInteger(n) && n > 0)
    || numExperts > maxExperts || topK > numExperts || topK > Number(sort.variantMetadata?.maxTopK)
    || hiddenSize % 32 || intermediateSize % 32) reject('requires bounded experts/topK and complete 32-value native blocks');
  if (groupSize !== 8 || maxExperts !== 256 || width !== 256
    || gu.wgslOverrides?.ROWS_PER_THREAD !== 2 || down.wgslOverrides?.ROWS_PER_THREAD !== 4
    || gu.wgslOverrides?.GROUP_SIZE !== groupSize || down.wgslOverrides?.GROUP_SIZE !== groupSize
    || gu.wgslOverrides?.WORKGROUP_SIZE !== width || down.wgslOverrides?.WORKGROUP_SIZE !== width
    || gu.wgslOverrides?.INPUT_IS_TOKEN !== true || gu.wgslOverrides?.SPLIT_GATE_UP !== true
    || down.wgslOverrides?.INPUT_IS_TOKEN !== false || down.wgslOverrides?.SPLIT_GATE_UP !== false) {
    reject('registered tile geometry differs from this implementation');
  }
  if (![limits.maxComputeInvocationsPerWorkgroup, limits.maxComputeWorkgroupSizeX,
    limits.maxComputeWorkgroupStorageSize, limits.maxStorageBuffersPerShaderStage,
    limits.maxComputeWorkgroupsPerDimension, limits.maxStorageBufferBindingSize, limits.maxBufferSize]
    .every(n => Number.isSafeInteger(n) && n > 0)) reject('explicit device limits required');
  if (limits.maxComputeInvocationsPerWorkgroup < width || limits.maxComputeWorkgroupSizeX < width
    || limits.maxComputeWorkgroupStorageSize < 12324 || limits.maxStorageBuffersPerShaderStage < 5) {
    reject('device cannot hold the cooperative tile or its bindings');
  }
  const numRoutes = numTokens * topK;
  const maxChunks = Math.ceil(numRoutes / groupSize) + numExperts;
  const pairOffset = 4 * maxChunks;
  const guGroups = Math.ceil(intermediateSize * 2 / (32 * Number(gu.wgslOverrides?.ROWS_PER_THREAD)));
  const downGroups = Math.ceil(hiddenSize / (32 * Number(down.wgslOverrides?.ROWS_PER_THREAD)));
  if ([maxChunks, guGroups, downGroups].some(n => n > limits.maxComputeWorkgroupsPerDimension)) reject('dispatch exceeds device axis limit');
  const groupBytes = (pairOffset + numRoutes) * 4;
  for (const bytes of [groupBytes, numRoutes * intermediateSize * 8, numRoutes * hiddenSize * 4, numTokens * hiddenSize * 4]) {
    if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff || bytes > limits.maxStorageBufferBindingSize
      || bytes > limits.maxBufferSize) reject('scratch extent exceeds u32 or device buffer limits');
  }
  return { numRoutes, maxChunks, pairOffset, groupBytes, guGroups, downGroups };
}

/** @type {typeof import('./moe-grouped.js').executeGroupedSort} */
export async function executeGroupedSort(recorder, indices, groups, indirect, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[MoE grouped prefill] Active device required.');
  const geometry = groupedPrefillGeometry(options, device.limits);
  assertQ40Buffer(device, indices, geometry.numRoutes * 4, GPUBufferUsage.STORAGE, 'grouped route IDs');
  assertQ40Buffer(device, groups, geometry.groupBytes, GPUBufferUsage.STORAGE, 'grouped schedule');
  assertQ40Buffer(device, indirect, 24, GPUBufferUsage.STORAGE, 'grouped indirect');
  if (!(indirect.usage & GPUBufferUsage.INDIRECT) || new Set([indices, groups, indirect]).size !== 3) {
    throw new Error('[MoE grouped prefill] Distinct schedule/indices/indirect buffers with INDIRECT usage required.');
  }
  await unifiedKernelWrapper('moe_grouped_sort', recorder, 'native_f32', [indices, groups, indirect], {
    num_routes: geometry.numRoutes, num_experts: options.numExperts, top_k: options.topK,
    pair_offset: geometry.pairOffset, gu_groups: geometry.guGroups, down_groups: geometry.downGroups,
  }, [1, 1, 1]);
}

/** @type {typeof import('./moe-grouped.js').executeGroupedProjection} */
export async function executeGroupedProjection(recorder, operation, input, weights, upWeights, groups, indirect, options) {
  if (operation !== 'moe_grouped_gu' && operation !== 'moe_grouped_down') throw new Error('[MoE grouped prefill] Unknown projection.');
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[MoE grouped prefill] Active device required.');
  const geometry = groupedPrefillGeometry(options, device.limits);
  const config = getKernelConfig(operation, 'native_f32');
  const isGateUp = config.wgslOverrides?.INPUT_IS_TOKEN === true;
  const N = isGateUp ? options.intermediateSize * 2 : options.hiddenSize;
  const K = isGateUp ? options.hiddenSize : options.intermediateSize;
  const inputRows = isGateUp ? options.numTokens : geometry.numRoutes;
  if (input.dtype !== 'f32' || input.precision != null || input.components != null || input.elementStrideBytes != null
    || input.shape.length !== 2 || input.shape[0] !== inputRows || input.shape[1] !== K) {
    throw new Error('[MoE grouped prefill] Requires explicit scalar F32 input geometry.');
  }
  const shape = [options.numExperts, isGateUp ? N / 2 : N, K];
  const banks = isGateUp ? [weights, upWeights] : [weights];
  const admittedDtypes = config.variantMetadata?.weightDtypes;
  for (const weight of banks) {
    if (!weight || !Array.isArray(admittedDtypes) || !admittedDtypes.includes(weight.dtype) || weight.dtype !== weights.dtype
      || weight.layout !== 'row' || weight.shape.length !== 3 || weight.shape.some((n, i) => n !== shape[i])) {
      throw new Error('[MoE grouped prefill] Unsupported layout: separate row-packed Q4_0/Q4_1/Q8_0 banks required; select the ungrouped graph explicitly.');
    }
    assertQ40Buffer(device, weight.buffer, ggufBlockBytes(shape, weight.dtype), GPUBufferUsage.STORAGE, 'grouped weight');
  }
  assertQ40Buffer(device, input.buffer, inputRows * K * 4, GPUBufferUsage.STORAGE, 'grouped input');
  assertQ40Buffer(device, groups, geometry.groupBytes, GPUBufferUsage.STORAGE, 'grouped schedule');
  assertQ40Buffer(device, indirect, 24, GPUBufferUsage.INDIRECT, 'grouped indirect');
  const format = ggufBlockFormat(weights.dtype);
  return withKernelOutput(recorder, null, geometry.numRoutes * N * 4, operation, async output => {
    await unifiedKernelWrapper(operation, recorder, 'native_f32',
      [input.buffer, weights.buffer, upWeights?.buffer ?? weights.buffer, groups, output], {
        num_routes: geometry.numRoutes, num_experts: options.numExperts, top_k: options.topK,
        pair_offset: geometry.pairOffset, N, K,
      }, { indirectBuffer: indirect, indirectOffset: isGateUp ? 0 : 12 },
      { QUANT_FORMAT: format.code, BLOCK_BYTES: format.bytes });
    return createTensor(output, 'f32', [geometry.numRoutes, N], operation);
  });
}
