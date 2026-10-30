import { runRouteQ4Matmul, recordRouteQ4Matmul, runRouteScatter, recordRouteScatter } from '../../../../gpu/kernels/moe-route.js';
import { runSiLURowSplit, recordSiLURowSplit } from '../../../../gpu/kernels/silu.js';
import { releaseOrTrack } from '../ops.js';

/** @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} recorder
 * @param {import('./qwen-route-executor.js').QwenRouteExpertOptions} options */
async function execute(recorder, options) {
  const { inputTensor, indicesBuffer, weightsBuffer, weights, expectedExpertFormat,
    numTokens, topK, hiddenSize, intermediateSize, activationDtype, hiddenActivation, swigluLimit } = options;
  if (activationDtype !== 'f32' || hiddenActivation !== 'silu' || expectedExpertFormat !== 'qwen3_5'
      || weights?.expertFormat !== expectedExpertFormat || swigluLimit === undefined) {
    throw new Error('[MoE] Qwen topk-route requires explicit F32, SiLU, Qwen packed weights, and swigluLimit.');
  }
  const projection = { numRoutes: numTokens * topK, numExperts: weights.numExperts, topK, alpha: 1 };
  const workspace = recorder?.decodeWorkspace && numTokens === 1 ? recorder.decodeWorkspace : null;
  if (workspace && (typeof options.layerIdx !== 'number' || !Number.isInteger(options.layerIdx) || options.layerIdx < 0)) {
    throw new Error('[MoE] Prepared route execution requires an explicit layerIdx.');
  }
  const matmul = recorder ? recordRouteQ4Matmul.bind(null, recorder) : runRouteQ4Matmul;
  const activate = recorder ? recordSiLURowSplit.bind(null, recorder) : runSiLURowSplit;
  const scatter = recorder ? recordRouteScatter.bind(null, recorder) : runRouteScatter;
  let gateUp = null, activated = null, down = null;
  try {
    gateUp = await matmul(inputTensor, indicesBuffer, weights.gateUp, {
      ...projection, N: intermediateSize * 2, K: hiddenSize, inputMode: 'token',
      outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:gate-up`, topK * intermediateSize * 2 * 4),
    });
    activated = await activate(gateUp, {
      numTokens: projection.numRoutes, dim: intermediateSize, activation: hiddenActivation, swigluLimit,
      outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:activated`, topK * intermediateSize * 4),
    });
    releaseOrTrack(recorder, gateUp.buffer);
    gateUp = null;
    down = await matmul(activated, indicesBuffer, weights.down, {
      ...projection, N: hiddenSize, K: intermediateSize, inputMode: 'route',
      outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:down`, topK * hiddenSize * 4),
    });
    releaseOrTrack(recorder, activated.buffer);
    activated = null;
    return await scatter(down, weightsBuffer, { numTokens, hiddenSize, topK,
      outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:routed`, hiddenSize * 4) });
  } finally {
    if (gateUp) releaseOrTrack(recorder, gateUp.buffer);
    if (activated) releaseOrTrack(recorder, activated.buffer);
    if (down) releaseOrTrack(recorder, down.buffer);
  }
}

/** @type {typeof import('./qwen-route-executor.js').runQwenRouteExperts} */
export function runQwenRouteExperts(options) { return execute(null, options); }

/** @type {typeof import('./qwen-route-executor.js').recordQwenRouteExperts} */
export function recordQwenRouteExperts(recorder, options) { return execute(recorder, options); }
