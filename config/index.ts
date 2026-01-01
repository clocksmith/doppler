/**
 * Config Module Index
 *
 * Central export for config-as-code architecture.
 *
 * @module config
 */

// Schema types
export * from './schema/index.js';

// Preset loader
export {
  getPreset,
  listPresets,
  resolvePreset,
  detectPreset,
  resolveConfig,
  PRESET_REGISTRY,
} from './loader.js';
