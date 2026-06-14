import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { QK_K } from '../../config/schema/index.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { VEC4_ELEMENTS_PER_WG } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';

const ROUTE_TILE_M = 4;
const ROUTE_THREADS_PER_COL = 64;

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer, got ${String(value)}.`);
  }
}

function resolveInputMode(value) {
  if (value === 'token') return 0;
  if (value === 'route') return 1;
  throw new Error(`Gemma4 route expert inputMode must be "token" or "route", got ${String(value)}.`);
}

export async function runGemma4RouteQ4MatmulF16A(input, routeIndices, weight, options = {}) {
  const numRoutes = options.numRoutes;
  const topK = options.topK;
  const n = options.N;
  const k = options.K;
  assertPositiveInteger(numRoutes, 'numRoutes');
  assertPositiveInteger(topK, 'topK');
  assertPositiveInteger(n, 'N');
  assertPositiveInteger(k, 'K');
  if (input?.dtype !== 'f16') {
    throw new Error(`Gemma4 route expert matmul requires f16 input, got ${String(input?.dtype)}.`);
  }
  if (getWeightDtype(weight) !== 'q4k') {
    throw new Error(`Gemma4 route expert matmul requires q4k weights, got ${String(getWeightDtype(weight))}.`);
  }

  const outputBytes = numRoutes * n * 2;
  const outputBuffer = options.outputBuffer ?? acquireBuffer(outputBytes, undefined, 'gemma4_route_q4_output');
  const ownsOutput = options.outputBuffer == null;
  const numBlocksPerRow = Math.ceil(k / QK_K);

  try {
    await unifiedKernelWrapper(
      'gemma4_route_q4_matmul',
      null,
      'f16a',
      [
        input.buffer,
        routeIndices,
        getBuffer(weight),
        outputBuffer,
      ],
      {
        num_routes: numRoutes,
        top_k: topK,
        N: n,
        K: k,
        num_blocks_per_row: numBlocksPerRow,
        input_mode: resolveInputMode(options.inputMode ?? 'token'),
        alpha: options.alpha ?? 1.0,
      },
      [n, Math.ceil(numRoutes / ROUTE_TILE_M), 1],
      {
        TILE_M: ROUTE_TILE_M,
        THREADS_PER_COL: ROUTE_THREADS_PER_COL,
      }
    );
    return createTensor(outputBuffer, 'f16', [numRoutes, n], options.label ?? 'gemma4_route_q4_output');
  } catch (error) {
    if (ownsOutput) {
      releaseBuffer(outputBuffer);
    }
    throw error;
  }
}

export async function runScatterAddRoutesF16ExpertScale(
  routeOutputs,
  routingIndices,
  routingWeights,
  expertScales,
  numTokens,
  hiddenSize,
  topK,
  options = {}
) {
  assertPositiveInteger(numTokens, 'numTokens');
  assertPositiveInteger(hiddenSize, 'hiddenSize');
  assertPositiveInteger(topK, 'topK');
  if (routeOutputs?.dtype !== 'f16') {
    throw new Error(`Route scatter requires f16 route outputs, got ${String(routeOutputs?.dtype)}.`);
  }

  const outputBytes = numTokens * hiddenSize * 2;
  const outputBuffer = options.outputBuffer ?? acquireBuffer(outputBytes, undefined, 'scatter_add_routes_output');
  const ownsOutput = options.outputBuffer == null;

  try {
    await unifiedKernelWrapper(
      'scatter_add_routes',
      null,
      'f16_w16_expert_scale',
      [
        routeOutputs.buffer,
        routingIndices,
        routingWeights,
        expertScales,
        outputBuffer,
      ],
      {
        num_tokens: numTokens,
        hidden_size: hiddenSize,
        top_k: topK,
      },
      [Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG), 1, 1]
    );
    return createTensor(outputBuffer, 'f16', [numTokens, hiddenSize], options.label ?? 'scatter_add_routes_output');
  } catch (error) {
    if (ownsOutput) {
      releaseBuffer(outputBuffer);
    }
    throw error;
  }
}
