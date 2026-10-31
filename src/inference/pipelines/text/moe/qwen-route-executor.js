import { executeGroupedPrefill } from './grouped-prefill.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { getWeightDtype, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { runMatmul, recordMatmul } from '../../../../gpu/kernels/matmul.js';
import { executeSharedSlotGateUp, executeSharedSlotDown } from '../../../../gpu/kernels/moe-shared-slot.js';
import { validateQwenSharedExpert } from '../ffn/qwen-shared-expert.js';
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
  const gateUpWeight = weights.gateUp ?? weights.gate;
  if (!gateUpWeight || (!weights.gateUp && !weights.up)) throw new Error('[MoE] Missing explicit fused or separate gate/up banks.');
  for (const weight of [gateUpWeight,weights.down,...(weights.up ? [weights.up] : [])]) {
    selectRuleValue('inference','moe','qwenRouteWeightDtype',{ dtype:getWeightDtype(weight) });
  }
  if (options.groupedPrefill === true) {
    if (options.sharedSlot) throw new Error('[MoE grouped prefill] Shared-slot execution must remain separate.');
    return executeGroupedPrefill(recorder, options);
  }
  if (options.sharedSlot) return executeSharedSlot(recorder, options);
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
    gateUp = await matmul(inputTensor, indicesBuffer, gateUpWeight, {
      upWeights: weights.gateUp ? null : weights.up,
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

/** @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} recorder
 * @param {import('./qwen-route-executor.js').QwenRouteExpertOptions} options */
async function executeSharedSlot(recorder, options) {
  const shared = options.sharedSlot;
  if (!shared) throw new Error('[MoE] Missing shared-slot contract.');
  const { inputTensor, weights, numTokens, hiddenSize, intermediateSize, topK, swigluLimit } = options;
  const sharedOptions = { numTokens, hiddenSize, intermediateSize: shared.intermediateSize,
    kernelPath: shared.kernelPath, phase: shared.phase, layerIdx: options.layerIdx, swigluLimit };
  validateQwenSharedExpert(inputTensor, inputTensor, shared.weights, sharedOptions);
  const { sharedExpertGate, sharedExpertUp, sharedExpertDown, sharedExpertRouter } = shared.weights;
  if (!weights.gate || !weights.up || weights.gateUp || !isWeightBuffer(sharedExpertGate)
    || !isWeightBuffer(sharedExpertUp) || !isWeightBuffer(sharedExpertDown)) {
    throw new Error('[MoE] Shared slot requires separate native packed gate/up and shared banks.');
  }
  const geometry = { numTokens, numExperts: weights.numExperts, topK, hiddenSize, intermediateSize,
    sharedSize: shared.intermediateSize, swigluLimit, indices: options.indicesBuffer };
  const workspace = recorder?.decodeWorkspace && numTokens === 1 ? recorder.decodeWorkspace : null;
  if (workspace && (typeof options.layerIdx !== 'number' || !Number.isSafeInteger(options.layerIdx) || options.layerIdx < 0)) {
    throw new Error('[MoE] Prepared shared slot requires an explicit layerIdx.');
  }
  const matmul = recorder ? recordMatmul.bind(null, recorder) : runMatmul;
  let activated = null, gate = null;
  try {
    gate = await matmul(inputTensor, sharedExpertRouter, numTokens, 1, hiddenSize, {
      transposeB: true, outputDtype: 'f32', kernelPath: shared.kernelPath,
      phaseOverride: shared.phase, layerIdx: options.layerIdx, role: 'moe_shared_router',
      outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:shared-router`, 4),
    });
    activated = await executeSharedSlotGateUp(recorder, inputTensor,
      { gate: weights.gate, up: weights.up, sharedGate: sharedExpertGate, sharedUp: sharedExpertUp },
      { ...geometry, outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:shared-slots`,
        (topK + 1) * Math.max(intermediateSize, shared.intermediateSize) * 4) });
    return await executeSharedSlotDown(recorder, activated,
      { down: weights.down, sharedDown: sharedExpertDown }, gate,
      { ...geometry, routeWeights: options.weightsBuffer,
        outputBuffer: workspace?.getBuffer(`moe:${options.layerIdx}:shared-output`, hiddenSize * 4) });
  } finally {
    if (activated) releaseOrTrack(recorder, activated.buffer);
    if (gate) releaseOrTrack(recorder, gate.buffer);
  }
}

/** @type {typeof import('./qwen-route-executor.js').runQwenRouteExperts} */
export function runQwenRouteExperts(options) { return execute(null, options); }

/** @type {typeof import('./qwen-route-executor.js').recordQwenRouteExperts} */
export function recordQwenRouteExperts(recorder, options) { return execute(recorder, options); }
