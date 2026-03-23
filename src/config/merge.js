import { chooseDefined, chooseDefinedWithSource } from './merge-helpers.js';
import { log } from '../debug/index.js';

// =============================================================================
// Merge Implementation
// =============================================================================

function overlay(
  path,
  manifestValue,
  runtimeValue,
  sources
) {
  return chooseDefinedWithSource(path, runtimeValue, manifestValue, sources);
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
    attentionOutputGate: overlay(
      `${prefix}.attentionOutputGate`,
      manifest.attentionOutputGate,
      runtime?.attentionOutputGate,
      sources
    ),
    causal: overlay(
      `${prefix}.causal`,
      manifest.causal,
      runtime?.causal,
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
    swigluLimit: overlay(
      `${prefix}.swigluLimit`,
      manifest.swigluLimit,
      runtime?.swigluLimit,
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
    mropeInterleaved: overlay(
      `${prefix}.mropeInterleaved`,
      manifest.mropeInterleaved,
      runtime?.mropeInterleaved,
      sources
    ),
    mropeSection: overlay(
      `${prefix}.mropeSection`,
      manifest.mropeSection,
      runtime?.mropeSection,
      sources
    ),
    partialRotaryFactor: overlay(
      `${prefix}.partialRotaryFactor`,
      manifest.partialRotaryFactor,
      runtime?.partialRotaryFactor,
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
    ropeLocalScalingType: overlay(
      `${prefix}.ropeLocalScalingType`,
      manifest.ropeLocalScalingType,
      runtime?.ropeLocalScalingType,
      sources
    ),
    ropeLocalScalingFactor: overlay(
      `${prefix}.ropeLocalScalingFactor`,
      manifest.ropeLocalScalingFactor,
      runtime?.ropeLocalScalingFactor,
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
    ropeLocalYarnBetaFast: overlay(
      `${prefix}.ropeLocalYarnBetaFast`,
      manifest.ropeLocalYarnBetaFast,
      runtime?.ropeLocalYarnBetaFast,
      sources
    ),
    ropeLocalYarnBetaSlow: overlay(
      `${prefix}.ropeLocalYarnBetaSlow`,
      manifest.ropeLocalYarnBetaSlow,
      runtime?.ropeLocalYarnBetaSlow,
      sources
    ),
    ropeLocalYarnOriginalMaxPos: overlay(
      `${prefix}.ropeLocalYarnOriginalMaxPos`,
      manifest.ropeLocalYarnOriginalMaxPos,
      runtime?.ropeLocalYarnOriginalMaxPos,
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
    embeddingPostprocessor: overlay(
      `${prefix}.embeddingPostprocessor`,
      manifest.embeddingPostprocessor,
      runtime?.embeddingPostprocessor,
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
      manifest?.type,
      runtime?.type,
      sources
    ),
    enabled: overlay(
      `${prefix}.enabled`,
      manifest?.enabled,
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
  let layerPattern = manifestInf.layerPattern;
  const runtimeLayerPattern = runtimeOverrides?.layerPattern;
  if (runtimeLayerPattern !== undefined) {
    layerPattern = runtimeLayerPattern;
    sources.set('inference.layerPattern', 'runtime');
  } else {
    sources.set('inference.layerPattern', 'manifest');
  }

  // Merge chatTemplate with source tracking.
  const chatTemplate = mergeChatTemplate(
    manifestInf.chatTemplate,
    runtimeOverrides?.chatTemplate,
    sources
  );

  let pipeline = manifestInf.pipeline;
  const runtimePipeline = runtimeOverrides?.pipeline;
  if (runtimePipeline !== undefined) {
    pipeline = runtimePipeline;
    sources.set('inference.pipeline', 'runtime');
  } else {
    sources.set('inference.pipeline', 'manifest');
  }

  const inference = {
    attention: mergeAttention(manifestInf.attention, runtimeOverrides?.attention, sources),
    normalization: mergeNormalization(manifestInf.normalization, runtimeOverrides?.normalization, sources),
    ffn: mergeFFN(manifestInf.ffn, runtimeOverrides?.ffn, sources),
    rope: mergeRoPE(manifestInf.rope, runtimeOverrides?.rope, sources),
    output: mergeOutput(manifestInf.output, runtimeOverrides?.output, sources),
    pipeline,
    layerPattern,
    chatTemplate,
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
  if (!merged || !merged._sources || !(merged._sources instanceof Map)) {
    log.debug('Merge', 'formatConfigSources: input missing or has no valid _sources Map');
    return '';
  }

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
  if (!merged?._sources || !(merged._sources instanceof Map)) {
    log.debug('Merge', 'getValuesBySource: input missing or has no valid _sources Map');
    return [];
  }
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
  if (!merged?._sources || !(merged._sources instanceof Map)) {
    log.debug('Merge', 'summarizeSources: input missing or has no valid _sources Map');
    return { manifest: 0, runtime: 0 };
  }
  let manifest = 0;
  let runtime = 0;
  for (const source of merged._sources.values()) {
    if (source === 'manifest') manifest++;
    else if (source === 'runtime') runtime++;
  }
  return { manifest, runtime };
}

/**
 * Dump every tracked config field and its source.
 *
 * Returns a plain object mapping each dot-path field tracked in the merged
 * config's _sources map to the source that won ('manifest' or 'runtime').
 * Useful for debug/diagnostic output.
 *
 * @param {object} mergedConfig - A merged config with _sources Map
 * @returns {Object<string, string>} field-to-source mapping
 */
export function dumpConfigSources(mergedConfig) {
  const result = {};
  const sources = mergedConfig?._sources;
  if (!sources || typeof sources.forEach !== 'function') {
    return result;
  }
  sources.forEach((source, path) => {
    result[path] = source;
  });
  return result;
}
