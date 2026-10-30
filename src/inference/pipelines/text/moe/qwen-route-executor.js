import { runRouteQ4Matmul, runRouteScatter } from '../../../../gpu/kernels/moe-route.js';
import { runSiLURowSplit } from '../../../../gpu/kernels/silu.js';
import { releaseBuffer } from '../../../../memory/buffer-pool.js';

/** @type {typeof import('./qwen-route-executor.js').runQwenRouteExperts} */
export async function runQwenRouteExperts(options) {
  const { inputTensor, indicesBuffer, weightsBuffer, weights, expectedExpertFormat,
    numTokens, topK, hiddenSize, intermediateSize, activationDtype, hiddenActivation, swigluLimit } = options;
  if (activationDtype !== 'f32' || hiddenActivation !== 'silu' || expectedExpertFormat !== 'qwen3_5'
      || weights?.expertFormat !== expectedExpertFormat || swigluLimit === undefined) {
    throw new Error('[MoE] Qwen topk-route requires explicit F32, SiLU, Qwen packed weights, and swigluLimit.');
  }
  const projection = { numRoutes: numTokens * topK, numExperts: weights.numExperts, topK, alpha: 1 };
  let gateUp = null, activated = null, down = null;
  try {
    gateUp = await runRouteQ4Matmul(inputTensor, indicesBuffer, weights.gateUp, {
      ...projection, N: intermediateSize * 2, K: hiddenSize, inputMode: 'token',
    });
    activated = await runSiLURowSplit(gateUp, {
      numTokens: projection.numRoutes, dim: intermediateSize, activation: hiddenActivation, swigluLimit,
    });
    releaseBuffer(gateUp.buffer);
    gateUp = null;
    down = await runRouteQ4Matmul(activated, indicesBuffer, weights.down, {
      ...projection, N: hiddenSize, K: intermediateSize, inputMode: 'route',
    });
    releaseBuffer(activated.buffer);
    activated = null;
    return await runRouteScatter(down, weightsBuffer, { numTokens, hiddenSize, topK });
  } finally {
    if (gateUp) releaseBuffer(gateUp.buffer);
    if (activated) releaseBuffer(activated.buffer);
    if (down) releaseBuffer(down.buffer);
  }
}
