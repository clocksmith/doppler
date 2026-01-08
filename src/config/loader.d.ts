/**
 * Preset Loader
 *
 * Loads and merges model family presets with manifest config.
 * Implements config-as-code pattern: JSON presets, not if-statements.
 *
 * @module config/loader
 */

import type {
  PresetSchema,
  ResolvedConfigSchema,
  RawModelConfigSchema,
  ManifestSchema,
} from './schema/index.js';

/** Registry of all available presets */
export declare const PRESET_REGISTRY: Record<string, PresetSchema>;

/**
 * Get a preset by ID, with inheritance resolution.
 */
export function getPreset(id: string): PresetSchema | null;

/**
 * List all available preset IDs.
 */
export function listPresets(): string[];

/**
 * Resolve a preset with its parent chain merged.
 * Child values override parent values.
 */
export function resolvePreset(id: string): PresetSchema;

/**
 * Detect the best preset for a model based on its config.
 * Checks presets in order of specificity (most specific first).
 */
export function detectPreset(
  config: RawModelConfigSchema,
  architecture?: string
): string;

/**
 * Build a fully resolved config by merging:
 * 1. Base preset (resolved with inheritance)
 * 2. Manifest config overrides
 * 3. Architecture from manifest
 */
export function resolveConfig(
  manifest: ManifestSchema,
  presetId?: string
): ResolvedConfigSchema;
