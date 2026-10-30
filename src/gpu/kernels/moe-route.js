import { QK_K, Q4K_BLOCK_BYTES } from '../../config/schema/index.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getWeightDtype, isGpuBufferInstance } from '../weight-buffer.js';
import { getKernelConfig } from './kernel-configs.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

/** @param {unknown} value @param {string} label @returns {number} */
function positive(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > 0xFFFFFFFF) {
    throw new Error(`[MoE route] ${label} must be a positive u32.`);
  }
  return value;
}

/** @param {{size: number} | null | undefined} buffer @param {number} bytes @param {string} label */
function extent(buffer, bytes, label) {
  positive(bytes, `${label} byte extent`);
  if (!buffer || !Number.isSafeInteger(buffer.size) || buffer.size < bytes) {
    throw new Error(`[MoE route] ${label} buffer is smaller than ${bytes} bytes.`);
  }
}

/** @param {import('../tensor.js').Tensor} input @param {number} rows @param {number} cols */
function tensor(input, rows, cols) {
  if (input?.dtype !== 'f32' || input.precision !== undefined || input.components !== undefined
      || input.elementStrideBytes !== undefined || input.shape.length !== 2
      || input.shape[0] !== rows || input.shape[1] !== cols) {
    throw new Error('[MoE route] Expected a scalar F32 tensor with explicit row/column geometry.');
  }
  extent(input.buffer, rows * cols * 4, 'input');
}

/** @param {GPUBuffer | null | undefined} output @param {GPUBuffer[]} inputs */
function outputAlias(output, inputs) {
  if (output && inputs.includes(output)) throw new Error('[MoE route] Output must not alias an input.');
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer} indices
 * @param {import('../weight-buffer.js').WeightBuffer} weights
 * @param {import('./moe-route.js').RouteQ4MatmulOptions} options
 */
async function matmul(target, input, indices, weights, options) {
  const { numRoutes, numExperts, topK, N, K, inputMode, alpha, outputBuffer } = options;
  for (const [key, value] of Object.entries({ numRoutes, numExperts, topK, N, K })) positive(value, key);
  if (numRoutes % topK || topK > numExperts) throw new Error('[MoE route] Invalid route/topK/expert geometry.');
  if (inputMode !== 'token' && inputMode !== 'route') throw new Error('[MoE route] Explicit token or route inputMode required.');
  if (!Number.isFinite(alpha) || !Number.isFinite(Math.fround(alpha))) throw new Error('[MoE route] Finite F32 alpha required.');
  tensor(input, inputMode === 'token' ? numRoutes / topK : numRoutes, K);
  const bank = getBuffer(weights);
  if (!isGpuBufferInstance(bank)) throw new Error('[MoE route] A resident GPU weight bank is required.');
  if (getWeightDtype(weights) !== 'q4k' || weights.layout !== 'row'
      || weights.shape?.length !== 3 || weights.shape[0] !== numExperts
      || weights.shape[1] !== N || weights.shape[2] !== K) {
    throw new Error('[MoE route] Expected a row-packed Q4_K bank shaped [experts, N, K].');
  }
  const blocks = Math.ceil(K / QK_K);
  extent(bank, numExperts * N * blocks * Q4K_BLOCK_BYTES, 'weight bank');
  extent(indices, numRoutes * 4, 'routing indices');
  outputAlias(outputBuffer, [input.buffer, indices, bank]);
  if (outputBuffer) extent(outputBuffer, numRoutes * N * 4, 'output');
  const config = getKernelConfig('moe_route_matmul', 'q4k_f32');
  const constants = config.wgslOverrides;
  const columns = positive(constants?.COLS_PER_WG, 'COLS_PER_WG');
  const threads = positive(constants?.THREADS_PER_COL_GEMV, 'THREADS_PER_COL_GEMV');
  const workgroup = positive(constants?.WORKGROUP_SIZE, 'WORKGROUP_SIZE');
  if (workgroup !== columns * threads || workgroup > 256) throw new Error('[MoE route] Invalid registered reduction geometry.');
  if (constants?.USE_FULL_BLOCK_FAST_PATH === true && K % QK_K !== 0) {
    throw new Error('[MoE route] Full-block specialization cannot execute a partial Q4_K row.');
  }
  return withKernelOutput(target, outputBuffer ?? null, numRoutes * N * 4, 'moe_route_matmul', async output => {
    await unifiedKernelWrapper('moe_route_matmul', target, 'q4k_f32',
      [input.buffer, bank, output, indices], {
        M: numRoutes, N, K, alpha, num_blocks_per_row: blocks,
        num_experts: numExperts, top_k: topK, input_mode: inputMode === 'token' ? 0 : 1,
      }, [Math.ceil(N / columns), numRoutes, 1]);
    return createTensor(output, 'f32', [numRoutes, N], 'moe_route_matmul');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer} weights
 * @param {import('./moe-route.js').RouteScatterOptions} options
 */
async function scatter(target, input, weights, options) {
  const { numTokens, hiddenSize, topK, outputBuffer } = options;
  for (const [key, value] of Object.entries({ numTokens, hiddenSize, topK })) positive(value, key);
  tensor(input, numTokens * topK, hiddenSize);
  extent(weights, numTokens * topK * 4, 'routing weights');
  outputAlias(outputBuffer, [input.buffer, weights]);
  if (outputBuffer) extent(outputBuffer, numTokens * hiddenSize * 4, 'output');
  const config = getKernelConfig('scatter_add_routes', 'f32');
  const threads = positive(config.workgroupSize[0], 'scatter workgroup');
  return withKernelOutput(target, outputBuffer ?? null, numTokens * hiddenSize * 4, 'moe_route_scatter', async output => {
    await unifiedKernelWrapper('scatter_add_routes', target, 'f32', [input.buffer, weights, output], {
      num_tokens: numTokens, hidden_size: hiddenSize, top_k: topK,
    }, [Math.ceil(numTokens * hiddenSize / threads), 1, 1]);
    return createTensor(output, 'f32', [numTokens, hiddenSize], 'moe_route_scatter');
  });
}

/** @type {typeof import('./moe-route.js').runRouteQ4Matmul} */
export function runRouteQ4Matmul(input, indices, weights, options) {
  return matmul(null, input, indices, weights, options);
}
/** @type {typeof import('./moe-route.js').recordRouteQ4Matmul} */
export function recordRouteQ4Matmul(recorder, input, indices, weights, options) {
  return matmul(recorder, input, indices, weights, options);
}
/** @type {typeof import('./moe-route.js').runRouteScatter} */
export function runRouteScatter(input, weights, options) {
  return scatter(null, input, weights, options);
}
/** @type {typeof import('./moe-route.js').recordRouteScatter} */
export function recordRouteScatter(recorder, input, weights, options) {
  return scatter(recorder, input, weights, options);
}
