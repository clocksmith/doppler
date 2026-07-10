function kernelPathUsesHead256DecodeAttention(kernelPath) {
  const steps = kernelPath?.decode?.steps;
  if (!Array.isArray(steps)) return false;
  return steps.some((step) => (
    step?.op === 'attention'
    && step?.kernel === 'attention_decode_online_head256_f16kv_output_gate.wgsl'
    && (step?.entry === undefined || step.entry === 'main')
  ));
}

function tensorElementCount(tensor) {
  if (!Array.isArray(tensor?.shape)) return null;
  return tensor.shape.reduce((total, value) => total * value, 1);
}

export function canUseAttentionOutputGateFusion(options = {}) {
  const {
    session,
    qGateTensor,
    numTokens,
    numHeads,
    headDim,
    cachedKDtype,
    cachedVDtype,
    kernelPath,
    diffusionGemmaDecoder,
  } = options;
  if (diffusionGemmaDecoder === true) return false;
  if (session?.attentionDecodeOnline?.useOutputGateFusion !== true) return false;
  if (!qGateTensor?.buffer || qGateTensor.dtype !== 'f32') return false;
  if (numTokens !== 1 || headDim !== 256) return false;
  if (cachedKDtype !== 'f16' || cachedVDtype !== 'f16') return false;
  if (!kernelPathUsesHead256DecodeAttention(kernelPath)) return false;
  const gateElements = tensorElementCount(qGateTensor);
  if (gateElements !== null && gateElements < numTokens * numHeads * headDim) return false;
  return true;
}
