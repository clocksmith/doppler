function isProdTurboQuantMode(mode) {
  return mode === 'turboquant_prod';
}

export function buildTieredQuantAttentionOptions(kvState, options = {}) {
  const isProd = isProdTurboQuantMode(kvState?.coldQuantMode);
  return {
    seqLen: options.seqLen ?? 1,
    coldLen: kvState?.coldLen ?? 0,
    hotLen: kvState?.hotLen ?? 0,
    numKVHeads: options.numKVHeads,
    causal: options.causal,
    startPos: options.startPos ?? 0,
    slidingWindow: options.slidingWindow ?? 0,
    attnSoftcap: options.attnSoftcap ?? 0,
    scale: options.scale,
    hotWindow: kvState?.hotWindow ?? 0,
    hotStart: kvState?.hotStart ?? 0,
    packedStride: kvState?.coldPackedStride ?? 0,
    mode: kvState?.coldQuantMode ?? 'none',
    rotationMatrixBuffer: kvState?.rotationMatrixBuffer ?? null,
    codebookCentroidsBuffer: kvState?.codebookCentroidsBuffer ?? null,
    residualKBuffer: isProd ? kvState?.residualKGPU ?? null : null,
    residualVBuffer: isProd ? kvState?.residualVGPU ?? null : null,
    residualNormsKBuffer: isProd ? kvState?.residualNormsKGPU ?? null : null,
    residualNormsVBuffer: isProd ? kvState?.residualNormsVGPU ?? null : null,
    qjlMatrixBuffer: isProd ? kvState?.qjlMatrixBuffer ?? null : null,
  };
}

export function buildContiguousQuantAttentionOptions(kvState, options = {}) {
  const isProd = kvState?.prodMode === true || isProdTurboQuantMode(kvState?.coldQuantMode);
  return {
    seqLen: options.seqLen ?? 1,
    kvLen: options.kvLen ?? 0,
    numKVHeads: options.numKVHeads,
    causal: options.causal,
    startPos: options.startPos ?? 0,
    slidingWindow: options.slidingWindow ?? 0,
    attnSoftcap: options.attnSoftcap ?? 0,
    scale: options.scale,
    packedStride: kvState?.coldPackedStride ?? 0,
    mode: kvState?.coldQuantMode ?? 'turboquant',
    rotationMatrixBuffer: kvState?.rotationMatrixBuffer ?? null,
    codebookCentroidsBuffer: kvState?.codebookCentroidsBuffer ?? null,
    residualKBuffer: isProd ? kvState?.residualKGPU ?? null : null,
    residualVBuffer: isProd ? kvState?.residualVGPU ?? null : null,
    residualNormsKBuffer: isProd ? kvState?.residualNormsKGPU ?? null : null,
    residualNormsVBuffer: isProd ? kvState?.residualNormsVGPU ?? null : null,
    qjlMatrixBuffer: isProd ? kvState?.qjlMatrixBuffer ?? null : null,
    packedStrideMSE: isProd ? kvState?.coldPackedStride : undefined,
    packedStrideResidual: isProd ? kvState?.residualPackedStride : undefined,
  };
}
