import { runProbes } from '../probes.js';

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
    dtype: qTensor.dtype,
  });
  if (kTensor) {
    await runProbes('k_rope', kTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes,
      recorder,
      operatorDiagnostics: state.operatorDiagnostics,
      dtype: kTensor.dtype,
    });
  }
}
