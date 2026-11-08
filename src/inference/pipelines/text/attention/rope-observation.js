import { runProbes, tensorProbeOptions } from '../probes.js';

export async function observeAttentionRoPE(options) {
  const {
    state,
    recorder,
    ropeApplied,
    disableRoPE,
    qTensor,
    kTensor,
    layerIdx,
    numTokens,
    numHeads,
    numKVHeads,
    headDim,
  } = options;
  if (!qTensor || (!ropeApplied && (disableRoPE || !state.ropeFreqsCos || !state.ropeFreqsSin))) {
    return;
  }
  await runProbes('q_rope', qTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numHeads * headDim,
    probes: state.debugProbes,
    recorder,
    operatorDiagnostics: state.operatorDiagnostics,
    ...tensorProbeOptions(qTensor),
  });
  if (kTensor) {
    await runProbes('k_rope', kTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes,
      recorder,
      operatorDiagnostics: state.operatorDiagnostics,
      ...tensorProbeOptions(kTensor),
    });
  }
}

export async function observeActivityAttention(options) {
  const activity = options.state.operatorDiagnostics?.activity;
  if (!activity) return;
  const { kvState, qTensor, cachedKTensor, cachedVTensor, recorder, layerIdx,
    numTokens, numHeads, numKVHeads, headDim, attnScale, attnSoftcap,
    causalForAttention, effectiveSlidingWindow } = options;
  if (kvState.kvLayout !== 'contiguous' || kvState.kvStart !== 0 || !cachedKTensor || !cachedVTensor) {
    activity.unavailable('attention', 'This cache layout has no activity capture adapter.');
    return;
  }
  const width = numKVHeads * headDim;
  const metadata = { heads: numHeads, kvHeads: numKVHeads, headDim, scale: attnScale,
    softcap: attnSoftcap, causal: causalForAttention, slidingWindow: effectiveSlidingWindow,
    kvLength: kvState.kvLenForAttention, queryStart: kvState.startPosForMask };
  for (const [role, tensor] of [['kv-key', cachedKTensor], ['kv-value', cachedVTensor]]) {
    await activity.capture(role, tensor.buffer, { layer: layerIdx, dtype: tensor.dtype,
      recorder, shape: [numTokens, width], offset: kvState.startPosForMask * width,
      metadata, ...tensorProbeOptions(tensor) });
  }
  await activity.capture('attention-query', qTensor.buffer, { layer: layerIdx,
    shape: [numTokens, numHeads * headDim], recorder, metadata, ...tensorProbeOptions(qTensor) });
}
