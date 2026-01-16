// =============================================================================
// Merge Implementation
// =============================================================================

function overlay(
  path,
  manifestValue,
  runtimeValue,
  sources
) {
  if (runtimeValue !== undefined && runtimeValue !== null) {
    sources.set(path, 'runtime');
    return runtimeValue;
  }
  sources.set(path, 'manifest');
  return manifestValue;
}

function mergeAttention(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.attention';
  return {
    queryPreAttnScalar: overlay(
      `${prefix}.queryPreAttnScalar`,
      manifest.queryPreAttnScalar,
      runtime?.queryPreAttnScalar,
      sources
    ),
    attentionBias: overlay(
      `${prefix}.attentionBias`,
      manifest.attentionBias,
      runtime?.attentionBias,
      sources
    ),
    attnLogitSoftcapping: overlay(
      `${prefix}.attnLogitSoftcapping`,
      manifest.attnLogitSoftcapping,
      runtime?.attnLogitSoftcapping,
      sources
    ),
    slidingWindow: overlay(
      `${prefix}.slidingWindow`,
      manifest.slidingWindow,
      runtime?.slidingWindow,
      sources
    ),
    queryKeyNorm: overlay(
      `${prefix}.queryKeyNorm`,
      manifest.queryKeyNorm,
      runtime?.queryKeyNorm,
      sources
    ),
  };
}

function mergeNormalization(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.normalization';
  return {
    rmsNormEps: overlay(
      `${prefix}.rmsNormEps`,
      manifest.rmsNormEps,
      runtime?.rmsNormEps,
      sources
    ),
    rmsNormWeightOffset: overlay(
      `${prefix}.rmsNormWeightOffset`,
      manifest.rmsNormWeightOffset,
      runtime?.rmsNormWeightOffset,
      sources
    ),
    postAttentionNorm: overlay(
      `${prefix}.postAttentionNorm`,
      manifest.postAttentionNorm,
      runtime?.postAttentionNorm,
      sources
    ),
    preFeedforwardNorm: overlay(
      `${prefix}.preFeedforwardNorm`,
      manifest.preFeedforwardNorm,
      runtime?.preFeedforwardNorm,
      sources
    ),
    postFeedforwardNorm: overlay(
      `${prefix}.postFeedforwardNorm`,
      manifest.postFeedforwardNorm,
      runtime?.postFeedforwardNorm,
      sources
    ),
  };
}

function mergeFFN(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.ffn';
  return {
    activation: overlay(
      `${prefix}.activation`,
      manifest.activation,
      runtime?.activation,
      sources
    ),
    gatedActivation: overlay(
      `${prefix}.gatedActivation`,
      manifest.gatedActivation,
      runtime?.gatedActivation,
      sources
    ),
  };
}

function mergeRoPE(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.rope';
  return {
    ropeTheta: overlay(
      `${prefix}.ropeTheta`,
      manifest.ropeTheta,
      runtime?.ropeTheta,
      sources
    ),
    ropeLocalTheta: overlay(
      `${prefix}.ropeLocalTheta`,
      manifest.ropeLocalTheta,
      runtime?.ropeLocalTheta,
      sources
    ),
    ropeScalingType: overlay(
      `${prefix}.ropeScalingType`,
      manifest.ropeScalingType,
      runtime?.ropeScalingType,
      sources
    ),
    ropeScalingFactor: overlay(
      `${prefix}.ropeScalingFactor`,
      manifest.ropeScalingFactor,
      runtime?.ropeScalingFactor,
      sources
    ),
    yarnBetaFast: overlay(
      `${prefix}.yarnBetaFast`,
      manifest.yarnBetaFast,
      runtime?.yarnBetaFast,
      sources
    ),
    yarnBetaSlow: overlay(
      `${prefix}.yarnBetaSlow`,
      manifest.yarnBetaSlow,
      runtime?.yarnBetaSlow,
      sources
    ),
    yarnOriginalMaxPos: overlay(
      `${prefix}.yarnOriginalMaxPos`,
      manifest.yarnOriginalMaxPos,
      runtime?.yarnOriginalMaxPos,
      sources
    ),
  };
}

function mergeOutput(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.output';
  return {
    finalLogitSoftcapping: overlay(
      `${prefix}.finalLogitSoftcapping`,
      manifest.finalLogitSoftcapping,
      runtime?.finalLogitSoftcapping,
      sources
    ),
    tieWordEmbeddings: overlay(
      `${prefix}.tieWordEmbeddings`,
      manifest.tieWordEmbeddings,
      runtime?.tieWordEmbeddings,
      sources
    ),
    scaleEmbeddings: overlay(
      `${prefix}.scaleEmbeddings`,
      manifest.scaleEmbeddings,
      runtime?.scaleEmbeddings,
      sources
    ),
    embeddingTranspose: overlay(
      `${prefix}.embeddingTranspose`,
      manifest.embeddingTranspose,
      runtime?.embeddingTranspose,
      sources
    ),
    embeddingVocabSize: overlay(
      `${prefix}.embeddingVocabSize`,
      manifest.embeddingVocabSize,
      runtime?.embeddingVocabSize,
      sources
    ),
  };
}

function mergeChatTemplate(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.chatTemplate';
  return {
    type: overlay(
      `${prefix}.type`,
      manifest?.type ?? null,
      runtime?.type,
      sources
    ),
    enabled: overlay(
      `${prefix}.enabled`,
      manifest?.enabled ?? false,
      runtime?.enabled,
      sources
    ),
  };
}

// =============================================================================
// Main Merge Function
// =============================================================================

export function mergeConfig(
  manifest,
  runtimeOverrides
) {
  const sources = new Map();
  const manifestInf = manifest.inference;

  // Merge layerPattern with source tracking.
  let layerPattern = manifestInf.layerPattern ?? null;
  const runtimeLayerPattern = runtimeOverrides?.layerPattern;
  if (runtimeLayerPattern !== undefined && runtimeLayerPattern !== null) {
    layerPattern = runtimeLayerPattern;
    sources.set('inference.layerPattern', 'runtime');
  } else {
    sources.set('inference.layerPattern', 'manifest');
  }

  // Merge defaultKernelPath with source tracking.
  let defaultKernelPath = manifestInf.defaultKernelPath ?? null;
  const runtimeKernelPath = runtimeOverrides?.defaultKernelPath;
  if (runtimeKernelPath !== undefined && runtimeKernelPath !== null) {
    defaultKernelPath = runtimeKernelPath;
    sources.set('inference.defaultKernelPath', 'runtime');
  } else {
    sources.set('inference.defaultKernelPath', 'manifest');
  }

  // Merge chatTemplate with source tracking.
  const chatTemplate = mergeChatTemplate(
    manifestInf.chatTemplate,
    runtimeOverrides?.chatTemplate,
    sources
  );

  const inference = {
    attention: mergeAttention(manifestInf.attention, runtimeOverrides?.attention, sources),
    normalization: mergeNormalization(manifestInf.normalization, runtimeOverrides?.normalization, sources),
    ffn: mergeFFN(manifestInf.ffn, runtimeOverrides?.ffn, sources),
    rope: mergeRoPE(manifestInf.rope, runtimeOverrides?.rope, sources),
    output: mergeOutput(manifestInf.output, runtimeOverrides?.output, sources),
    layerPattern,
    chatTemplate,
    defaultKernelPath,
  };

  return {
    modelId: manifest.modelId,
    inference,
    architecture: manifest.architecture,
    _sources: sources,
  };
}

// =============================================================================
// Debug Utilities
// =============================================================================

export function formatConfigSources(merged) {
  const lines = [];

  for (const [path, source] of merged._sources) {
    const pathParts = path.split('.');
    let value = merged;
    for (const part of pathParts) {
      value = value?.[part];
    }

    const valueStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
    lines.push(`${path}: ${valueStr} (${source})`);
  }

  return lines.sort().join('\n');
}

export function getValuesBySource(
  merged,
  source
) {
  const result = [];

  for (const [path, src] of merged._sources) {
    if (src === source) {
      const pathParts = path.split('.');
      let value = merged;
      for (const part of pathParts) {
        value = value?.[part];
      }
      result.push([path, value]);
    }
  }

  return result;
}

export function summarizeSources(merged) {
  let manifest = 0;
  let runtime = 0;
  for (const source of merged._sources.values()) {
    if (source === 'manifest') manifest++;
    else if (source === 'runtime') runtime++;
  }
  return { manifest, runtime };
}
