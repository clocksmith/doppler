/**
 * Config Merge Module
 *
 * Merges manifest inference config with runtime overrides and tracks
 * the source of each value. This enables tracing where any config
 * value came from during debugging.
 *
 * Architecture:
 *   - Manifest provides ALL values (required, no optionals)
 *   - Runtime can override any manifest value when a field is non-null/undefined
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
  type ManifestChatTemplateSchema,
  type ArchitectureSchema,
} from './schema/index.js';

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
  chatTemplate: ManifestChatTemplateSchema;
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
 * Runtime overrides treat null/undefined as "not specified".
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
): MergedConfig;

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
export function formatConfigSources(merged: MergedConfig): string;

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
): Array<[string, unknown]>;

/**
 * Summarize config sources for logging.
 *
 * @returns Object with counts: { manifest: N, runtime: N }
 */
export function summarizeSources(merged: MergedConfig): { manifest: number; runtime: number };
