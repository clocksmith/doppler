// Admission for the implemented paired attention path, before any tensor allocation.
/** @type {typeof import('./paired-contract.js').assertPairedAttentionExecution} */
export function assertPairedAttentionExecution(input, config, state, session, weights) {
  const metadata = input?.precision !== undefined || input?.components !== undefined || input?.elementStrideBytes !== undefined;
  if (!metadata && session.intermediatePrecision !== 'two-f32') return;
  if (session.intermediatePrecision !== 'two-f32' || input?.dtype !== 'f32' || input.precision !== 'two-f32'
    || input.components !== 2 || input.elementStrideBytes !== 8) {
    throw new Error('[PairedAttention] Input representation must match the immutable paired session.');
  }
  if (!weights || state.skipKVCacheWrites === true || !state.kvCache?.hasGPUCache?.()
    || state.kvCache.kvDtype !== 'f16' || state.kvCache.layout !== 'contiguous'
    || config.diffusionGemmaDecoder === true || config.sharedKVSourceLayerIdx != null || config.storeSharedKV === true) {
    throw new Error('[PairedAttention] Declared ordinary resident contiguous F16 cache and owned Q/K/V projections required.');
  }
  if ((config.inputDtype != null && config.inputDtype !== 'f32')
    || (config.outputDtype != null && config.outputDtype !== 'f32') || config.activationDtype !== 'f32') {
    throw new Error('[PairedAttention] Paired attention must retain F32 component storage throughout its declared projection path.');
  }
  if (config.normalizationType !== 'rmsnorm' || config.valueNorm === true || config.multimodalBidirectionalSpan != null) {
    throw new Error('[PairedAttention] Declared RMSNorm text attention required; other normalization or multimodal paths are not paired.');
  }
  if ([weights.qProjBias, weights.kProjBias, weights.vProjBias, weights.qkvProjBias, weights.oProjBias].some(value => value != null)) {
    throw new Error('[PairedAttention] Projection bias requires a declared paired bias operator.');
  }
  if (config.precomputedInputNorm && (config.precomputedInputNorm.precision !== 'two-f32'
    || config.precomputedInputNorm.components !== 2 || config.precomputedInputNorm.elementStrideBytes !== 8)) {
    throw new Error('[PairedAttention] Precomputed normalization must retain the paired representation.');
  }
}
