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

import {
  type ManifestInferenceSchema,
  type ManifestAttentionSchema,
  type ManifestNormalizationSchema,
  type ManifestFFNSchema,
  type ManifestRoPESchema,
  type ManifestOutputSchema,
  type ManifestLayerPatternSchema,
  type ArchitectureSchema,
} from './schema/index.js';

// =============================================================================
// Types
// =============================================================================

/** Source of a config value - only 'manifest' or 'runtime', never 'default' */
export type ConfigSource = 'manifest' | 'runtime';

/** Deep partial type for runtime overrides */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/** Manifest input for merge (subset of full manifest) */
export interface ManifestInput {
  modelId: string;
  inference: ManifestInferenceSchema;
  architecture?: ArchitectureSchema | string;
}

/** Runtime inference overrides */
export type RuntimeInferenceOverrides = DeepPartial<ManifestInferenceSchema>;

/**
 * Merged inference config with all values resolved.
 * Identical to ManifestInferenceSchema since manifest provides all values.
 */
export interface MergedInferenceConfig {
  attention: ManifestAttentionSchema;
  normalization: ManifestNormalizationSchema;
  ffn: ManifestFFNSchema;
  rope: ManifestRoPESchema;
  output: ManifestOutputSchema;
  layerPattern: ManifestLayerPatternSchema | null;
  defaultKernelPath: string | null;
}

/**
 * Full merged config with source tracking.
 */
export interface MergedConfig {
  /** Model identifier */
  modelId: string;

  /** Resolved inference configuration */
  inference: MergedInferenceConfig;

  /** Architecture info (if available) */
  architecture?: ArchitectureSchema | string;

  /**
   * Source tracking - dot-path to source.
   * Only 'manifest' or 'runtime' - no defaults.
   */
  _sources: Map<string, ConfigSource>;
}

// =============================================================================
// Merge Implementation
// =============================================================================

/**
 * Overlay runtime value on manifest value, tracking source.
 * Manifest value is required - runtime is optional override.
 */
function overlay<T>(
  path: string,
  manifestValue: T,
  runtimeValue: T | undefined,
  sources: Map<string, ConfigSource>
): T {
  if (runtimeValue !== undefined) {
    sources.set(path, 'runtime');
    return runtimeValue;
  }
  sources.set(path, 'manifest');
  return manifestValue;
}

/**
 * Merge attention config.
 */
function mergeAttention(
  manifest: ManifestAttentionSchema,
  runtime: DeepPartial<ManifestAttentionSchema> | undefined,
  sources: Map<string, ConfigSource>
): ManifestAttentionSchema {
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
 */
function mergeNormalization(
  manifest: ManifestNormalizationSchema,
  runtime: DeepPartial<ManifestNormalizationSchema> | undefined,
  sources: Map<string, ConfigSource>
): ManifestNormalizationSchema {
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
 */
function mergeFFN(
  manifest: ManifestFFNSchema,
  runtime: DeepPartial<ManifestFFNSchema> | undefined,
  sources: Map<string, ConfigSource>
): ManifestFFNSchema {
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
 */
function mergeRoPE(
  manifest: ManifestRoPESchema,
  runtime: DeepPartial<ManifestRoPESchema> | undefined,
  sources: Map<string, ConfigSource>
): ManifestRoPESchema {
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
 */
function mergeOutput(
  manifest: ManifestOutputSchema,
  runtime: DeepPartial<ManifestOutputSchema> | undefined,
  sources: Map<string, ConfigSource>
): ManifestOutputSchema {
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
 * @param manifest - Model manifest with complete inference config
 * @param runtimeOverrides - Optional runtime overrides
 * @returns Merged config with source tracking
 *
 * @example
 * ```typescript
 * const merged = mergeConfig(manifest);
 * console.log(merged.inference.normalization.rmsNormWeightOffset); // true
 * console.log(merged._sources.get('inference.normalization.rmsNormWeightOffset')); // 'manifest'
 * ```
 */
export function mergeConfig(
  manifest: ManifestInput,
  runtimeOverrides?: RuntimeInferenceOverrides
): MergedConfig {
  const sources = new Map<string, ConfigSource>();
  const manifestInf = manifest.inference;

  // Merge layerPattern with source tracking
  // Use 'layerPattern' in runtimeOverrides to check explicit override (even if null)
  let layerPattern: ManifestLayerPatternSchema | null = manifestInf.layerPattern ?? null;
  if ('layerPattern' in (runtimeOverrides ?? {})) {
    layerPattern = (runtimeOverrides?.layerPattern as ManifestLayerPatternSchema) ?? null;
    sources.set('inference.layerPattern', 'runtime');
  } else {
    sources.set('inference.layerPattern', 'manifest');
  }

  // Merge defaultKernelPath with source tracking
  // Use 'defaultKernelPath' in runtimeOverrides to check explicit override
  let defaultKernelPath: string | null = manifestInf.defaultKernelPath ?? null;
  if ('defaultKernelPath' in (runtimeOverrides ?? {})) {
    defaultKernelPath = (runtimeOverrides?.defaultKernelPath as string) ?? null;
    sources.set('inference.defaultKernelPath', 'runtime');
  } else {
    sources.set('inference.defaultKernelPath', 'manifest');
  }

  const inference: MergedInferenceConfig = {
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
 * @param merged - Merged config with sources
 * @returns Formatted string showing value sources
 *
 * @example
 * ```
 * inference.attention.slidingWindow: 4096 (manifest)
 * inference.normalization.rmsNormWeightOffset: true (manifest)
 * inference.rope.ropeTheta: 10000 (manifest)
 * ```
 */
export function formatConfigSources(merged: MergedConfig): string {
  const lines: string[] = [];

  for (const [path, source] of merged._sources) {
    const pathParts = path.split('.');
    let value: unknown = merged;
    for (const part of pathParts) {
      value = (value as Record<string, unknown>)?.[part];
    }

    const valueStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
    lines.push(`${path}: ${valueStr} (${source})`);
  }

  return lines.sort().join('\n');
}

/**
 * Get config values by source.
 *
 * @param merged - Merged config with sources
 * @param source - Source to filter by
 * @returns Array of [path, value] pairs from the specified source
 */
export function getValuesBySource(
  merged: MergedConfig,
  source: ConfigSource
): Array<[string, unknown]> {
  const result: Array<[string, unknown]> = [];

  for (const [path, src] of merged._sources) {
    if (src === source) {
      const pathParts = path.split('.');
      let value: unknown = merged;
      for (const part of pathParts) {
        value = (value as Record<string, unknown>)?.[part];
      }
      result.push([path, value]);
    }
  }

  return result;
}

/**
 * Summarize config sources for logging.
 *
 * @returns Object with counts: { manifest: N, runtime: N }
 */
export function summarizeSources(merged: MergedConfig): { manifest: number; runtime: number } {
  let manifest = 0;
  let runtime = 0;
  for (const source of merged._sources.values()) {
    if (source === 'manifest') manifest++;
    else if (source === 'runtime') runtime++;
  }
  return { manifest, runtime };
}
