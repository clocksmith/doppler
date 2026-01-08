/**
 * Config Merge Module
 *
 * Merges manifest inference config with runtime overrides and tracks
 * the source of each value. This enables tracing where any config
 * value came from during debugging.
 *
 * Architecture:
 *   - Manifest provides ALL values (required, no optionals)
 *   - Runtime can override any manifest value
 *   - _sources tracks 'manifest' or 'runtime' for each field
 *   - NO default fallback - if manifest is incomplete, loader validation fails
 *
 * @module config/merge
 */

// =============================================================================
// Merge Implementation
// =============================================================================

/**
 * Overlay runtime value on manifest value, tracking source.
 * Manifest value is required - runtime is optional override.
 * @param {string} path
 * @param {*} manifestValue
 * @param {*} runtimeValue
 * @param {Map<string, import('./merge.js').ConfigSource>} sources
 * @returns {*}
 */
function overlay(
  path,
  manifestValue,
  runtimeValue,
  sources
) {
  if (runtimeValue !== undefined) {
    sources.set(path, 'runtime');
    return runtimeValue;
  }
  sources.set(path, 'manifest');
  return manifestValue;
}

/**
 * Merge attention config.
 * @param {import('./schema/index.js').ManifestAttentionSchema} manifest
 * @param {Partial<import('./schema/index.js').ManifestAttentionSchema> | undefined} runtime
 * @param {Map<string, import('./merge.js').ConfigSource>} sources
 * @returns {import('./schema/index.js').ManifestAttentionSchema}
 */
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

/**
 * Merge normalization config.
 * @param {import('./schema/index.js').ManifestNormalizationSchema} manifest
 * @param {Partial<import('./schema/index.js').ManifestNormalizationSchema> | undefined} runtime
 * @param {Map<string, import('./merge.js').ConfigSource>} sources
 * @returns {import('./schema/index.js').ManifestNormalizationSchema}
 */
function mergeNormalization(
  manifest,
  runtime,
  sources
) {
  const prefix = 'inference.normalization';
  return {
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

/**
 * Merge FFN config.
 * @param {import('./schema/index.js').ManifestFFNSchema} manifest
 * @param {Partial<import('./schema/index.js').ManifestFFNSchema> | undefined} runtime
 * @param {Map<string, import('./merge.js').ConfigSource>} sources
 * @returns {import('./schema/index.js').ManifestFFNSchema}
 */
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

/**
 * Merge RoPE config.
 * @param {import('./schema/index.js').ManifestRoPESchema} manifest
 * @param {Partial<import('./schema/index.js').ManifestRoPESchema> | undefined} runtime
 * @param {Map<string, import('./merge.js').ConfigSource>} sources
 * @returns {import('./schema/index.js').ManifestRoPESchema}
 */
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

/**
 * Merge output config.
 * @param {import('./schema/index.js').ManifestOutputSchema} manifest
 * @param {Partial<import('./schema/index.js').ManifestOutputSchema> | undefined} runtime
 * @param {Map<string, import('./merge.js').ConfigSource>} sources
 * @returns {import('./schema/index.js').ManifestOutputSchema}
 */
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
  };
}

// =============================================================================
// Main Merge Function
// =============================================================================

/**
 * Merge manifest inference config with runtime overrides.
 *
 * Returns a fully resolved config with source tracking for every value.
 * The `_sources` map shows where each value came from:
 *   - 'manifest': from the model manifest (converter output)
 *   - 'runtime': from user's runtime override
 *
 * NOTE: Manifest must provide all values. If any field is missing,
 * the loader should have rejected the manifest before calling this.
 *
 * @param {import('./merge.js').ManifestInput} manifest - Model manifest with complete inference config
 * @param {import('./merge.js').RuntimeInferenceOverrides} [runtimeOverrides] - Optional runtime overrides
 * @returns {import('./merge.js').MergedConfig} Merged config with source tracking
 *
 * @example
 * ```typescript
 * const merged = mergeConfig(manifest);
 * console.log(merged.inference.normalization.rmsNormWeightOffset); // true
 * console.log(merged._sources.get('inference.normalization.rmsNormWeightOffset')); // 'manifest'
 * ```
 */
export function mergeConfig(
  manifest,
  runtimeOverrides
) {
  /** @type {Map<string, import('./merge.js').ConfigSource>} */
  const sources = new Map();
  const manifestInf = manifest.inference;

  // Merge layerPattern with source tracking
  // Use 'layerPattern' in runtimeOverrides to check explicit override (even if null)
  /** @type {import('./schema/index.js').ManifestLayerPatternSchema | null} */
  let layerPattern = manifestInf.layerPattern ?? null;
  if ('layerPattern' in (runtimeOverrides ?? {})) {
    layerPattern = /** @type {import('./schema/index.js').ManifestLayerPatternSchema} */ (runtimeOverrides?.layerPattern) ?? null;
    sources.set('inference.layerPattern', 'runtime');
  } else {
    sources.set('inference.layerPattern', 'manifest');
  }

  // Merge defaultKernelPath with source tracking
  // Use 'defaultKernelPath' in runtimeOverrides to check explicit override
  /** @type {string | null} */
  let defaultKernelPath = manifestInf.defaultKernelPath ?? null;
  if ('defaultKernelPath' in (runtimeOverrides ?? {})) {
    defaultKernelPath = /** @type {string} */ (runtimeOverrides?.defaultKernelPath) ?? null;
    sources.set('inference.defaultKernelPath', 'runtime');
  } else {
    sources.set('inference.defaultKernelPath', 'manifest');
  }

  /** @type {import('./merge.js').MergedInferenceConfig} */
  const inference = {
    attention: mergeAttention(manifestInf.attention, runtimeOverrides?.attention, sources),
    normalization: mergeNormalization(manifestInf.normalization, runtimeOverrides?.normalization, sources),
    ffn: mergeFFN(manifestInf.ffn, runtimeOverrides?.ffn, sources),
    rope: mergeRoPE(manifestInf.rope, runtimeOverrides?.rope, sources),
    output: mergeOutput(manifestInf.output, runtimeOverrides?.output, sources),
    layerPattern,
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

/**
 * Format merged config sources for logging.
 *
 * @param {import('./merge.js').MergedConfig} merged - Merged config with sources
 * @returns {string} Formatted string showing value sources
 *
 * @example
 * ```
 * inference.attention.slidingWindow: 4096 (manifest)
 * inference.normalization.rmsNormWeightOffset: true (manifest)
 * inference.rope.ropeTheta: 10000 (manifest)
 * ```
 */
export function formatConfigSources(merged) {
  /** @type {string[]} */
  const lines = [];

  for (const [path, source] of merged._sources) {
    const pathParts = path.split('.');
    /** @type {unknown} */
    let value = merged;
    for (const part of pathParts) {
      value = /** @type {Record<string, unknown>} */ (value)?.[part];
    }

    const valueStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
    lines.push(`${path}: ${valueStr} (${source})`);
  }

  return lines.sort().join('\n');
}

/**
 * Get config values by source.
 *
 * @param {import('./merge.js').MergedConfig} merged - Merged config with sources
 * @param {import('./merge.js').ConfigSource} source - Source to filter by
 * @returns {Array<[string, unknown]>} Array of [path, value] pairs from the specified source
 */
export function getValuesBySource(
  merged,
  source
) {
  /** @type {Array<[string, unknown]>} */
  const result = [];

  for (const [path, src] of merged._sources) {
    if (src === source) {
      const pathParts = path.split('.');
      /** @type {unknown} */
      let value = merged;
      for (const part of pathParts) {
        value = /** @type {Record<string, unknown>} */ (value)?.[part];
      }
      result.push([path, value]);
    }
  }

  return result;
}

/**
 * Summarize config sources for logging.
 *
 * @param {import('./merge.js').MergedConfig} merged
 * @returns {{ manifest: number; runtime: number }} Object with counts: { manifest: N, runtime: N }
 */
export function summarizeSources(merged) {
  let manifest = 0;
  let runtime = 0;
  for (const source of merged._sources.values()) {
    if (source === 'manifest') manifest++;
    else if (source === 'runtime') runtime++;
  }
  return { manifest, runtime };
}
