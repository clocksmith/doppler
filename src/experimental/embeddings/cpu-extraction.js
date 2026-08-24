import { layerNormCPU, matmulCPU, rmsNormCPU } from '../logits/cpu.js';

export function extractTokenEmbeddingsFromHidden(
  hiddenStates,
  numTokens,
  hiddenSize,
  finalNormWeights,
  config,
  finalNormBias = null
) {
  const expectedLength = numTokens * hiddenSize;
  if (hiddenStates.length !== expectedLength) {
    throw new Error(
      `[Pipeline] Hidden state length mismatch for embedding extraction: expected=${expectedLength}, got=${hiddenStates.length}`
    );
  }
  const tokenEmbeddings = new Float32Array(expectedLength);
  let effectiveLayerNormBias = null;
  if (config.normalizationType === 'layernorm') {
    if (config.finalNormBiasTensor !== null && !(finalNormBias instanceof Float32Array)) {
      throw new Error(
        `[Pipeline] LayerNorm declares bias tensor "${config.finalNormBiasTensor}" but it was not loaded.`
      );
    }
    effectiveLayerNormBias = finalNormBias instanceof Float32Array
      ? finalNormBias
      : new Float32Array(hiddenSize);
    if (effectiveLayerNormBias.length !== hiddenSize) {
      throw new Error(
        `[Pipeline] LayerNorm final bias length must be ${hiddenSize}; got ${effectiveLayerNormBias.length}.`
      );
    }
  }
  for (let tokenIndex = 0; tokenIndex < numTokens; tokenIndex += 1) {
    const offset = tokenIndex * hiddenSize;
    const tokenHidden = hiddenStates.subarray(offset, offset + hiddenSize);
    const normalized = config.normalizationType === 'layernorm'
      ? layerNormCPU(tokenHidden, finalNormWeights, effectiveLayerNormBias, config.rmsNormEps)
      : rmsNormCPU(tokenHidden, finalNormWeights, config.rmsNormEps, config.rmsNormWeightOffset);
    tokenEmbeddings.set(normalized, offset);
  }
  return tokenEmbeddings;
}

export function extractEmbeddingFromHidden(
  hiddenStates,
  numTokens,
  hiddenSize,
  embeddingMode,
  finalNormWeights,
  config,
  embeddingPostprocessor = null,
  normalizedTokenEmbeddings = null,
  finalNormBias = null
) {
  const tokenEmbeddings = normalizedTokenEmbeddings ?? extractTokenEmbeddingsFromHidden(
    hiddenStates,
    numTokens,
    hiddenSize,
    finalNormWeights,
    config,
    finalNormBias
  );
  const postprocessorConfig = config?.embeddingPostprocessor ?? null;
  const resolvedEmbeddingMode = postprocessorConfig?.poolingMode ?? embeddingMode;
  if (postprocessorConfig && embeddingMode !== resolvedEmbeddingMode) {
    throw new Error(
      `[Pipeline] embeddingMode "${embeddingMode}" conflicts with manifest poolingMode="${resolvedEmbeddingMode}".`
    );
  }
  let pooled;
  if (resolvedEmbeddingMode === 'last') {
    const offset = (numTokens - 1) * hiddenSize;
    pooled = tokenEmbeddings.slice(offset, offset + hiddenSize);
  } else if (resolvedEmbeddingMode === 'mean') {
    pooled = new Float32Array(hiddenSize);
    for (let token = 0; token < numTokens; token += 1) {
      const offset = token * hiddenSize;
      for (let column = 0; column < hiddenSize; column += 1) {
        pooled[column] += tokenEmbeddings[offset + column];
      }
    }
    for (let column = 0; column < hiddenSize; column += 1) {
      pooled[column] /= numTokens;
    }
  } else {
    throw new Error(`[Pipeline] unsupported embeddingMode "${resolvedEmbeddingMode}".`);
  }
  if (!postprocessorConfig) return pooled;
  if (!embeddingPostprocessor) {
    throw new Error('[Pipeline] Embedding postprocessor weights are missing for this manifest.');
  }
  let current = pooled;
  for (let index = 0; index < embeddingPostprocessor.projections.length; index += 1) {
    const projection = embeddingPostprocessor.projections[index];
    const projected = matmulCPU(
      current,
      projection.weight,
      1,
      projection.outputSize,
      projection.inputSize,
      'row'
    );
    if (projection.bias) {
      for (let column = 0; column < projected.length; column += 1) {
        projected[column] += projection.bias[column];
      }
    }
    current = projected;
  }
  if (embeddingPostprocessor.normalize === 'l2') {
    let sumSq = 0;
    for (let index = 0; index < current.length; index += 1) sumSq += current[index] * current[index];
    const norm = Math.sqrt(sumSq);
    if (norm > 0) {
      for (let index = 0; index < current.length; index += 1) current[index] /= norm;
    }
  }
  return current;
}
