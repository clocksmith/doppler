export function projectSeparateAttentionGate({
  runMatmul,
  projectionInput,
  gateWeight,
  numTokens,
  outputSize,
  hiddenSize,
  layerIdx,
  kernelPath,
  outputDtype,
  matmulDebug,
  executionPolicies,
  fusedNormWeight,
  fusedNormEps,
  fusedNormOffset,
}) {
  if (!gateWeight) return null;
  return runMatmul(projectionInput, gateWeight, numTokens, outputSize, hiddenSize, {
    transposeB: 'auto',
    role: 'q_gate_proj',
    layerIdx,
    kernelPath,
    outputDtype,
    matmulDebug,
    executionPolicies,
    normWeight: fusedNormWeight,
    rmsNormEps: fusedNormEps,
    rmsNormOffset: fusedNormOffset,
  });
}
