

export function resolveRotaryDim(headDim, partialRotaryFactor, modelId) {
  if (partialRotaryFactor == null) {
    return headDim;
  }
  if (typeof partialRotaryFactor !== 'number' || Number.isNaN(partialRotaryFactor)) {
    throw new Error(`Manifest "${modelId}" has invalid rope.partialRotaryFactor.`);
  }
  if (partialRotaryFactor <= 0 || partialRotaryFactor > 1) {
    throw new Error(
      `Manifest "${modelId}" requires 0 < rope.partialRotaryFactor <= 1; got ${partialRotaryFactor}.`
    );
  }
  const rotaryDim = Math.trunc(headDim * partialRotaryFactor);
  if (rotaryDim <= 0 || (rotaryDim % 2) !== 0) {
    throw new Error(
      `Manifest "${modelId}" resolves rope rotary dim ${rotaryDim} from headDim=${headDim} ` +
      `and partialRotaryFactor=${partialRotaryFactor}, but rotary dim must be a positive even integer.`
    );
  }
  return rotaryDim;
}

export function resolveFrequencyBaseDim(headDim, rotaryDim, frequencyBaseDim, modelId, fieldName) {
  if (frequencyBaseDim == null) {
    return rotaryDim;
  }
  if (typeof frequencyBaseDim !== 'number' || Number.isNaN(frequencyBaseDim)) {
    throw new Error(`Manifest "${modelId}" has invalid ${fieldName}.`);
  }
  const resolved = Math.trunc(frequencyBaseDim);
  if (resolved <= 0 || (resolved % 2) !== 0) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} to be a positive even integer; got ${frequencyBaseDim}.`
    );
  }
  if (resolved < rotaryDim) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} (${resolved}) to be >= rotary dim (${rotaryDim}).`
    );
  }
  if (resolved > headDim) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} (${resolved}) to be <= attention head dim (${headDim}).`
    );
  }
  return resolved;
}

export function buildPerLayerIntermediateSizes({
  baseIntermediateSize,
  numLayers,
  numKvSharedLayers,
  useDoubleWideMlp,
  modelId,
}) {
  if (!Number.isFinite(baseIntermediateSize) || baseIntermediateSize <= 0) {
    throw new Error(
      `Manifest "${modelId}" has invalid architecture.intermediateSize (${String(baseIntermediateSize)}).`
    );
  }
  if (!Number.isFinite(numLayers) || numLayers <= 0) {
    throw new Error(
      `Manifest "${modelId}" has invalid architecture.numLayers (${String(numLayers)}).`
    );
  }

  const resolvedBaseIntermediateSize = Math.trunc(baseIntermediateSize);
  const resolvedNumLayers = Math.trunc(numLayers);
  const intermediateSizes = new Array(resolvedNumLayers).fill(resolvedBaseIntermediateSize);

  if (!useDoubleWideMlp) {
    return intermediateSizes;
  }

  if (!Number.isFinite(numKvSharedLayers) || numKvSharedLayers <= 0 || numKvSharedLayers >= resolvedNumLayers) {
    throw new Error(
      `Manifest "${modelId}" enables ffn.useDoubleWideMlp, but architecture.numKvSharedLayers=${String(numKvSharedLayers)} ` +
      `must be a positive integer smaller than numLayers=${resolvedNumLayers}.`
    );
  }

  const firstKvSharedLayerIdx = resolvedNumLayers - Math.trunc(numKvSharedLayers);
  if (firstKvSharedLayerIdx <= 0 || firstKvSharedLayerIdx >= resolvedNumLayers) {
    throw new Error(
      `Manifest "${modelId}" enables ffn.useDoubleWideMlp, but the derived first KV-shared layer index ` +
      `(${firstKvSharedLayerIdx}) is invalid for numLayers=${resolvedNumLayers}.`
    );
  }

  const widenedIntermediateSize = resolvedBaseIntermediateSize * 2;
  for (let layerIdx = firstKvSharedLayerIdx; layerIdx < resolvedNumLayers; layerIdx += 1) {
    intermediateSizes[layerIdx] = widenedIntermediateSize;
  }

  return intermediateSizes;
}

export function resolveLayerIntermediateSize(config, layerIdx) {
  const intermediateSizes = Array.isArray(config?.intermediateSizes) ? config.intermediateSizes : null;
  if (Number.isFinite(layerIdx) && intermediateSizes) {
    const resolved = intermediateSizes[Math.trunc(layerIdx)];
    if (Number.isFinite(resolved) && resolved > 0) {
      return Math.trunc(resolved);
    }
  }

  const fallback = Number(config?.intermediateSize);
  if (!Number.isFinite(fallback) || fallback <= 0) {
    throw new Error(`Invalid modelConfig.intermediateSize: ${String(config?.intermediateSize)}`);
  }
  return Math.trunc(fallback);
}
